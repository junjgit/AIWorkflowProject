import numpy as np
import pandas as pd

from src.features import build_feature_matrix, feature_row_for_date, target_for_date
from src.modeling import compare_models, train_country_model


def synthetic_ts(days=520):
    dates = pd.date_range("2018-01-01", periods=days, freq="D")
    t = np.arange(days)
    revenue = 1000 + 2.0 * t + 140 * np.sin(2 * np.pi * t / 30) + 60 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame({
        "date": dates,
        "purchases": (50 + 5 * np.sin(2 * np.pi * t / 7)).astype(int),
        "unique_invoices": (30 + 3 * np.sin(2 * np.pi * t / 7)).astype(int),
        "unique_streams": (20 + 2 * np.sin(2 * np.pi * t / 7)).astype(int),
        "total_views": 500 + 20 * np.sin(2 * np.pi * t / 7),
        "year_month": dates.strftime("%Y-%m"),
        "revenue": revenue,
    })


def test_target_is_30_day_revenue():
    ts = synthetic_ts()
    d = pd.Timestamp("2018-06-01")
    expected = ts.loc[(ts["date"] >= d) & (ts["date"] <= d + pd.Timedelta(days=29)), "revenue"].sum()
    assert np.isclose(target_for_date(ts, d), expected)


def test_features_use_history_and_have_expected_columns():
    ts = synthetic_ts()
    row = feature_row_for_date(ts, "2018-06-01")
    assert "revenue_30d" in row.columns
    expected = ts.loc[(ts["date"] >= "2018-05-02") & (ts["date"] <= "2018-05-31"), "revenue"].sum()
    assert np.isclose(row.loc[0, "revenue_30d"], expected)


def test_multiple_models_are_compared():
    X, y, _ = build_feature_matrix(synthetic_ts())
    comparison = compare_models(X, y, n_splits=3)
    assert {"linear_regression", "random_forest", "gradient_boosting", "trailing_30d_baseline"}.issubset(
        set(comparison["model_name"])
    )


def test_model_training_returns_fitted_artifact():
    artifact, comparison = train_country_model(synthetic_ts(), "all", "All Countries")
    X, _, _ = build_feature_matrix(synthetic_ts())
    pred = artifact["estimator"].predict(X.tail(1)[artifact["feature_columns"]])
    assert len(pred) == 1
    assert np.isfinite(pred[0])
    assert artifact["forecast_horizon_days"] == 30
