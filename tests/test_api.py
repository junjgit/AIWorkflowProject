import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from api.app import create_app
from src.features import FEATURE_COLUMNS, build_feature_matrix


def _write_daily_json(path: Path, country="United Kingdom", days=240):
    start = pd.Timestamp("2019-01-01")
    records = []
    for i in range(days):
        d = start + pd.Timedelta(days=i)
        records.append({
            "country": country,
            "customer_id": i + 1,
            "day": d.day,
            "invoice": str(100000 + i),
            "month": d.month,
            "price": float(100 + (i % 17)),
            "stream_id": f"S{i % 13}",
            "times_viewed": int(5 + i % 7),
            "year": d.year,
        })
    path.write_text(json.dumps(records), encoding="utf-8")


def _prepare_model(data_dir: Path, model_dir: Path):
    from src.data_ingestion import convert_to_ts, fetch_data
    df = fetch_data(data_dir)
    ts = convert_to_ts(df)
    X, y, dates = build_feature_matrix(ts)
    model = LinearRegression().fit(X, y)
    artifact = {
        "estimator": model,
        "feature_columns": FEATURE_COLUMNS,
        "country_key": "all",
        "country_display_name": "All Countries",
        "model_name": "linear_regression",
        "model_version": "test",
        "forecast_horizon_days": 30,
        "train_start": str(pd.Timestamp(dates[0]).date()),
        "train_end": str(pd.Timestamp(dates[-1]).date()),
        "validation_rmse": 0.0,
        "validation_mae": 0.0,
        "baseline_rmse": 0.0,
        "training_rows": len(X),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "all.joblib")
    uk_artifact = dict(artifact)
    uk_artifact["country_key"] = "united_kingdom"
    uk_artifact["country_display_name"] = "United Kingdom"
    joblib.dump(uk_artifact, model_dir / "united_kingdom.joblib")
    manifest = {
        "model_version": "test",
        "forecast_horizon_days": 30,
        "countries": {
            "all": {"display_name": "All Countries", "artifact": "all.joblib", "model_name": "linear_regression"},
            "united_kingdom": {"display_name": "United Kingdom", "artifact": "united_kingdom.joblib", "model_name": "linear_regression"},
        },
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_health_and_predict_all(tmp_path):
    train = tmp_path / "train"
    prod = tmp_path / "prod"
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    metrics = tmp_path / "metrics"
    train.mkdir(); prod.mkdir(); metrics.mkdir()
    _write_daily_json(train / "train.json", days=240)
    _write_daily_json(prod / "prod.json", days=1)  # duplicate-safe combined history
    _prepare_model(train, models)
    app = create_app({
        "TESTING": True,
        "TRAIN_DATA_DIR": str(train),
        "PRODUCTION_DATA_DIR": str(prod),
        "MODEL_DIR": str(models),
        "LOG_DIR": str(logs),
        "METRICS_DIR": str(metrics),
    })
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["models_available"] == 2

    response = client.post("/predict", json={"country": "all", "date": "2019-08-28"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["country"] == "all"
    assert payload["forecast_horizon_days"] == 30
    assert np.isfinite(payload["forecast_30d_revenue"])
    assert (logs / "predict_log.csv").exists()

    country_response = client.post(
        "/predict", json={"country": "United Kingdom", "date": "2019-08-28"}
    )
    assert country_response.status_code == 200
    country_payload = country_response.get_json()
    assert country_payload["country"] == "united_kingdom"
    assert np.isfinite(country_payload["forecast_30d_revenue"])


def test_predict_rejects_missing_fields(tmp_path):
    app = create_app({"TESTING": True})
    response = app.test_client().post("/predict", json={"country": "all"})
    assert response.status_code == 400


def test_train_endpoint_invokes_training(monkeypatch, tmp_path):
    import api.app as app_module

    def fake_train(train_data_dir, model_dir, log_dir, comparison_output):
        return {"models_trained": 2, "countries": ["all", "united_kingdom"], "runtime_seconds": 0.01}

    monkeypatch.setattr(app_module, "train_all_models", fake_train)
    app = create_app({
        "TESTING": True,
        "TRAIN_DATA_DIR": str(tmp_path / "train"),
        "MODEL_DIR": str(tmp_path / "models"),
        "LOG_DIR": str(tmp_path / "logs"),
        "MODEL_COMPARISON": str(tmp_path / "comparison.csv"),
    })
    response = app.test_client().post("/train")
    assert response.status_code == 200
    assert response.get_json()["models_trained"] == 2


def test_logfile_endpoint_reads_requested_log(tmp_path):
    from src.logging_utils import log_prediction

    logs = tmp_path / "logs"
    log_prediction(
        logs,
        country="all",
        forecast_date="2019-08-01",
        forecast_30d_revenue=100.0,
        baseline_30d_revenue=95.0,
        model_name="linear_regression",
        model_version="test",
        runtime_ms=1.0,
    )
    app = create_app({"TESTING": True, "LOG_DIR": str(logs)})
    response = app.test_client().get("/logfile?type=predict")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["records"][0]["country"] == "all"
