"""Post-production model monitoring and gold-standard comparison."""
from __future__ import annotations

import json
import gc
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data_ingestion import build_time_series, combine_data_dirs, fetch_data
from .features import build_feature_matrix
from .modeling import load_artifact, load_manifest


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_post_production(
    train_data_dir: str | Path,
    production_data_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Evaluate frozen training models against known production-period revenue.

    Feature windows for each forecast date use only dates preceding the forecast
    date. The production files supply the later known outcomes used as the gold
    standard for performance monitoring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(model_dir)
    train_df = fetch_data(train_data_dir)
    prod_df = fetch_data(production_data_dir)
    combined = combine_data_dirs([train_data_dir, production_data_dir])
    prod_start = pd.Timestamp(prod_df["invoice_date"].min()).normalize()
    prod_end = pd.Timestamp(prod_df["invoice_date"].max()).normalize()
    final_forecast_start = prod_end - timedelta(days=29)

    # Re-create series from the combined history, but use the model manifest's
    # country set so evaluation matches the deployed model population exactly.
    from .data_ingestion import convert_to_ts
    prediction_rows = []
    metric_rows = []

    for country_key, meta in manifest["countries"].items():
        display_name = meta["display_name"]
        if country_key == "all":
            combined_ts = convert_to_ts(combined)
            train_ts = convert_to_ts(train_df)
        else:
            combined_ts = convert_to_ts(combined, country=display_name)
            train_ts = convert_to_ts(train_df, country=display_name)

        X, y, dates = build_feature_matrix(combined_ts, include_target=True)
        dates_pd = pd.to_datetime(dates)
        mask = (dates_pd >= prod_start) & (dates_pd <= final_forecast_start)
        X_prod = X.loc[mask].reset_index(drop=True)
        y_prod = y[mask]
        d_prod = dates_pd[mask]
        if len(X_prod) == 0:
            continue

        artifact = load_artifact(model_dir, country_key)
        pred = artifact["estimator"].predict(X_prod[artifact["feature_columns"]])
        baseline = X_prod["revenue_30d"].to_numpy(dtype=float)

        train_X, train_y, _ = build_feature_matrix(train_ts, include_target=True)
        shift_pct = 100.0 * (float(np.mean(y_prod)) - float(np.mean(train_y))) / (abs(float(np.mean(train_y))) + 1e-9)

        model_rmse = _rmse(y_prod, pred)
        model_mae = float(mean_absolute_error(y_prod, pred))
        baseline_rmse = _rmse(y_prod, baseline)
        baseline_mae = float(mean_absolute_error(y_prod, baseline))
        metric_rows.append({
            "country": country_key,
            "display_name": display_name,
            "n_forecasts": int(len(y_prod)),
            "model_rmse": model_rmse,
            "model_mae": model_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
            "rmse_improvement_pct": 100.0 * (baseline_rmse - model_rmse) / (baseline_rmse + 1e-9),
            "target_mean_shift_pct": shift_pct,
        })
        for d, actual, model_value, baseline_value in zip(d_prod, y_prod, pred, baseline):
            prediction_rows.append({
                "country": country_key,
                "forecast_date": str(pd.Timestamp(d).date()),
                "actual_30d_revenue": float(actual),
                "model_30d_revenue": float(model_value),
                "baseline_30d_revenue": float(baseline_value),
            })

        # Release per-country intermediates before the next country is processed.
        del combined_ts, train_ts, X, y, X_prod, y_prod, pred, baseline, train_X, train_y
        gc.collect()

    metrics = pd.DataFrame(metric_rows).sort_values("country")
    predictions = pd.DataFrame(prediction_rows).sort_values(["country", "forecast_date"])
    metrics.to_csv(output_dir / "post_production_metrics.csv", index=False)
    predictions.to_csv(output_dir / "post_production_predictions.csv", index=False)

    all_pred = predictions.loc[predictions["country"] == "all"].copy()
    if not all_pred.empty:
        all_pred["forecast_date"] = pd.to_datetime(all_pred["forecast_date"])
        plt.figure(figsize=(10, 5.5))
        plt.plot(all_pred["forecast_date"], all_pred["actual_30d_revenue"], label="Actual 30-day revenue")
        plt.plot(all_pred["forecast_date"], all_pred["model_30d_revenue"], label="Model forecast")
        plt.plot(all_pred["forecast_date"], all_pred["baseline_30d_revenue"], label="Trailing 30-day baseline")
        plt.title("Post-production 30-day revenue: model vs baseline")
        plt.xlabel("Forecast start date")
        plt.ylabel("Revenue")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "production_model_vs_baseline_all.png", dpi=160)
        plt.close()

    if not metrics.empty:
        ordered = metrics.sort_values("model_rmse")
        x = np.arange(len(ordered))
        width = 0.38
        plt.figure(figsize=(11, 5.5))
        plt.bar(x - width / 2, ordered["model_rmse"], width, label="Model RMSE")
        plt.bar(x + width / 2, ordered["baseline_rmse"], width, label="Baseline RMSE")
        plt.xticks(x, ordered["country"], rotation=45, ha="right")
        plt.yscale("log")
        plt.ylabel("RMSE (log scale)")
        plt.title("Post-production RMSE by country")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "production_rmse_by_country.png", dpi=160)
        plt.close()

    summary = {
        "production_start": str(prod_start.date()),
        "production_end": str(prod_end.date()),
        "last_fully_observed_forecast_start": str(final_forecast_start.date()),
        "countries_evaluated": int(len(metrics)),
    }
    all_row = metrics.loc[metrics["country"] == "all"]
    if not all_row.empty:
        row = all_row.iloc[0]
        summary["all_countries"] = {
            "model_rmse": float(row["model_rmse"]),
            "model_mae": float(row["model_mae"]),
            "baseline_rmse": float(row["baseline_rmse"]),
            "baseline_mae": float(row["baseline_mae"]),
            "rmse_improvement_pct": float(row["rmse_improvement_pct"]),
            "target_mean_shift_pct": float(row["target_mean_shift_pct"]),
        }
    (output_dir / "post_production_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
