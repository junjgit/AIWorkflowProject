"""Model comparison, training, serialization, and inference."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_ingestion import build_time_series, combine_data_dirs, fetch_data, normalize_country_key
from .features import FEATURE_COLUMNS, FORECAST_HORIZON_DAYS, build_feature_matrix, feature_row_for_date
from .logging_utils import log_prediction, log_training

MODEL_VERSION = "1.0"


def candidate_models() -> dict[str, object]:
    """Return deterministic candidate regression models."""
    return {
        "linear_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
            loss="squared_error",
        ),
    }


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compare_models(X: pd.DataFrame, y: np.ndarray, n_splits: int = 4) -> pd.DataFrame:
    """Compare candidate models with forward-chaining time-series validation."""
    if len(X) < 120:
        n_splits = min(3, max(2, len(X) // 30))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []

    for model_name, template in candidate_models().items():
        fold_rmse, fold_mae = [], []
        for train_idx, test_idx in splitter.split(X):
            model = clone(template)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            fold_rmse.append(_rmse(y[test_idx], pred))
            fold_mae.append(float(mean_absolute_error(y[test_idx], pred)))
        rows.append({
            "model_name": model_name,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_mae": float(np.mean(fold_mae)),
            "std_rmse": float(np.std(fold_rmse)),
            "folds": int(len(fold_rmse)),
        })

    # A transparent non-ML reference model: the previous 30 days of revenue.
    baseline_pred = X["revenue_30d"].to_numpy(dtype=float)
    rows.append({
        "model_name": "trailing_30d_baseline",
        "mean_rmse": _rmse(y, baseline_pred),
        "mean_mae": float(mean_absolute_error(y, baseline_pred)),
        "std_rmse": np.nan,
        "folds": 1,
    })
    return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)


def train_country_model(ts: pd.DataFrame, country_key: str, display_name: str) -> tuple[dict, pd.DataFrame]:
    """Compare candidates, select the best ML model, and fit it to all training data."""
    X, y, dates = build_feature_matrix(ts, include_target=True)
    comparison = compare_models(X, y)
    ml_comparison = comparison.loc[comparison["model_name"] != "trailing_30d_baseline"]
    selected_name = str(ml_comparison.iloc[0]["model_name"])
    estimator = clone(candidate_models()[selected_name])
    estimator.fit(X, y)
    artifact = {
        "estimator": estimator,
        "feature_columns": FEATURE_COLUMNS,
        "country_key": country_key,
        "country_display_name": display_name,
        "model_name": selected_name,
        "model_version": MODEL_VERSION,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "train_start": str(pd.Timestamp(dates[0]).date()),
        "train_end": str(pd.Timestamp(dates[-1]).date()),
        "validation_rmse": float(ml_comparison.iloc[0]["mean_rmse"]),
        "validation_mae": float(ml_comparison.iloc[0]["mean_mae"]),
        "baseline_rmse": float(comparison.loc[comparison["model_name"] == "trailing_30d_baseline", "mean_rmse"].iloc[0]),
        "training_rows": int(len(X)),
    }
    return artifact, comparison


def train_all_models(
    train_data_dir: str | Path,
    model_dir: str | Path,
    log_dir: str | Path,
    comparison_output: str | Path | None = None,
) -> dict:
    """Train aggregate plus top-10 country models and persist a manifest."""
    start = time.perf_counter()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_data(train_data_dir)
    series, country_map = build_time_series(df, top_n=10)
    comparison_rows = []
    manifest = {
        "model_version": MODEL_VERSION,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "training_data_start": str(df["invoice_date"].min().date()),
        "training_data_end": str(df["invoice_date"].max().date()),
        "countries": {},
    }

    for country_key, ts in series.items():
        country_start = time.perf_counter()
        artifact, comparison = train_country_model(ts, country_key, country_map[country_key])
        artifact_path = model_dir / f"{country_key}.joblib"
        joblib.dump(artifact, artifact_path)
        manifest["countries"][country_key] = {
            "display_name": country_map[country_key],
            "artifact": artifact_path.name,
            "model_name": artifact["model_name"],
            "validation_rmse": artifact["validation_rmse"],
            "validation_mae": artifact["validation_mae"],
            "baseline_rmse": artifact["baseline_rmse"],
        }
        tagged = comparison.copy()
        tagged.insert(0, "country", country_key)
        comparison_rows.append(tagged)
        log_training(
            log_dir,
            country=country_key,
            model_name=artifact["model_name"],
            model_version=MODEL_VERSION,
            train_start=artifact["train_start"],
            train_end=artifact["train_end"],
            validation_rmse=round(artifact["validation_rmse"], 4),
            validation_mae=round(artifact["validation_mae"], 4),
            runtime_seconds=round(time.perf_counter() - country_start, 3),
        )

    with (model_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    comparisons = pd.concat(comparison_rows, ignore_index=True)
    if comparison_output:
        path = Path(comparison_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparisons.to_csv(path, index=False)
    return {
        "models_trained": len(series),
        "countries": list(series.keys()),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "manifest": str(model_dir / "model_manifest.json"),
    }


def load_manifest(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "model_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Model manifest not found. Run model training first.")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_country_key(country: str, manifest: dict) -> str:
    """Resolve either a model key or human-readable country name."""
    raw = str(country).strip()
    if raw.lower() in {"all", "all countries", "all_countries"}:
        return "all"
    normalized = normalize_country_key(raw)
    if normalized in manifest["countries"]:
        return normalized
    for key, meta in manifest["countries"].items():
        if str(meta["display_name"]).lower() == raw.lower():
            return key
    raise ValueError(f"Unsupported country: {country}")


def load_artifact(model_dir: str | Path, country_key: str) -> dict:
    path = Path(model_dir) / f"{country_key}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found for {country_key}")
    return joblib.load(path)


def predict_revenue(
    country: str,
    forecast_date: str,
    data_dirs: list[str | Path],
    model_dir: str | Path,
    log_dir: str | Path | None = None,
) -> dict:
    """Predict 30-day revenue for a country/date and optionally write a log entry."""
    started = time.perf_counter()
    manifest = load_manifest(model_dir)
    country_key = resolve_country_key(country, manifest)
    artifact = load_artifact(model_dir, country_key)

    df = combine_data_dirs(data_dirs)
    if country_key == "all":
        from .data_ingestion import convert_to_ts
        ts = convert_to_ts(df)
    else:
        display_name = manifest["countries"][country_key]["display_name"]
        from .data_ingestion import convert_to_ts
        ts = convert_to_ts(df, country=display_name)

    X = feature_row_for_date(ts, forecast_date)
    prediction = float(artifact["estimator"].predict(X[artifact["feature_columns"]])[0])
    baseline = float(X["revenue_30d"].iloc[0])
    runtime_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "country": country_key,
        "country_display_name": artifact["country_display_name"],
        "forecast_date": str(pd.Timestamp(forecast_date).date()),
        "forecast_horizon_days": int(artifact["forecast_horizon_days"]),
        "forecast_30d_revenue": round(prediction, 2),
        "baseline_30d_revenue": round(baseline, 2),
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
        "runtime_ms": round(runtime_ms, 2),
    }
    if log_dir is not None:
        log_prediction(log_dir, **result)
    return result
