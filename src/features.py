"""Feature engineering for 30-day revenue forecasting."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

FORECAST_HORIZON_DAYS = 30
MIN_HISTORY_DAYS = 70
FEATURE_COLUMNS = [
    "revenue_7d",
    "revenue_14d",
    "revenue_30d",
    "revenue_70d",
    "revenue_prev_year_30d",
    "avg_invoices_30d",
    "avg_views_30d",
    "avg_purchases_30d",
    "revenue_7_to_30_ratio",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]


def _indexed(ts: pd.DataFrame) -> pd.DataFrame:
    frame = ts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def _sum_window(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end < start:
        return 0.0
    return float(frame.loc[start:end, column].sum())


def _mean_window(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end < start:
        return 0.0
    values = frame.loc[start:end, column]
    return float(values.mean()) if len(values) else 0.0


def _feature_values(frame: pd.DataFrame, d: pd.Timestamp) -> dict[str, float]:
    prev_end = d - timedelta(days=1)
    rev7 = _sum_window(frame, "revenue", d - timedelta(days=7), prev_end)
    rev14 = _sum_window(frame, "revenue", d - timedelta(days=14), prev_end)
    rev30 = _sum_window(frame, "revenue", d - timedelta(days=30), prev_end)
    rev70 = _sum_window(frame, "revenue", d - timedelta(days=70), prev_end)
    prior_year_start = d - timedelta(days=365)
    prior_year_end = prior_year_start + timedelta(days=29)
    prior_year = _sum_window(frame, "revenue", prior_year_start, prior_year_end)

    avg_invoices = _mean_window(frame, "unique_invoices", d - timedelta(days=30), prev_end)
    avg_views = _mean_window(frame, "total_views", d - timedelta(days=30), prev_end)
    avg_purchases = _mean_window(frame, "purchases", d - timedelta(days=30), prev_end)
    ratio = rev7 / (rev30 + 1e-9)

    month_angle = 2 * np.pi * (d.month - 1) / 12.0
    dow_angle = 2 * np.pi * d.dayofweek / 7.0
    return {
        "revenue_7d": rev7,
        "revenue_14d": rev14,
        "revenue_30d": rev30,
        "revenue_70d": rev70,
        "revenue_prev_year_30d": prior_year,
        "avg_invoices_30d": avg_invoices,
        "avg_views_30d": avg_views,
        "avg_purchases_30d": avg_purchases,
        "revenue_7_to_30_ratio": ratio,
        "month_sin": float(np.sin(month_angle)),
        "month_cos": float(np.cos(month_angle)),
        "dow_sin": float(np.sin(dow_angle)),
        "dow_cos": float(np.cos(dow_angle)),
    }


def _validate_feature_date(frame: pd.DataFrame, d: pd.Timestamp) -> None:
    min_date, max_date = frame.index.min(), frame.index.max()
    if d - timedelta(days=MIN_HISTORY_DAYS) < min_date:
        raise ValueError(
            f"Forecast date {d.date()} does not have {MIN_HISTORY_DAYS} days of history; "
            f"earliest supported date is {(min_date + timedelta(days=MIN_HISTORY_DAYS)).date()}"
        )
    if d > max_date + timedelta(days=1):
        raise ValueError(
            f"Forecast date {d.date()} is beyond available history; latest supported date is "
            f"{(max_date + timedelta(days=1)).date()}"
        )


def feature_row_for_date(ts: pd.DataFrame, forecast_date: str | pd.Timestamp) -> pd.DataFrame:
    """Build a leakage-safe feature row for a requested forecast start date.

    Features use observations strictly before ``forecast_date``. The function
    allows the forecast date to be at most one day after the latest observed
    date, which matches the intended operational use after the newest daily
    transactions have been ingested.
    """
    frame = _indexed(ts)
    d = pd.Timestamp(forecast_date).normalize()
    _validate_feature_date(frame, d)
    return pd.DataFrame([_feature_values(frame, d)], columns=FEATURE_COLUMNS)


def _target_from_frame(frame: pd.DataFrame, d: pd.Timestamp, horizon_days: int) -> float:
    end = d + timedelta(days=horizon_days - 1)
    if end > frame.index.max():
        raise ValueError(f"Target window ending {end.date()} is not fully observed")
    return _sum_window(frame, "revenue", d, end)


def target_for_date(ts: pd.DataFrame, forecast_date: str | pd.Timestamp, horizon_days: int = FORECAST_HORIZON_DAYS) -> float:
    """Return known revenue from forecast_date through the next horizon-1 days."""
    frame = _indexed(ts)
    d = pd.Timestamp(forecast_date).normalize()
    return _target_from_frame(frame, d, horizon_days)


def build_feature_matrix(
    ts: pd.DataFrame,
    include_target: bool = True,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Create supervised features for every supported daily forecast date."""
    frame = _indexed(ts)
    start = frame.index.min() + timedelta(days=MIN_HISTORY_DAYS)
    end = frame.index.max()
    if include_target:
        end = end - timedelta(days=horizon_days - 1)
    if start > end:
        raise ValueError("Insufficient time-series history for feature construction")

    dates = pd.date_range(start, end, freq="D")
    rows = [_feature_values(frame, d) for d in dates]
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = None
    if include_target:
        y = np.array([_target_from_frame(frame, d, horizon_days) for d in dates], dtype=float)
    return X, y, dates.to_numpy(dtype="datetime64[D]")
