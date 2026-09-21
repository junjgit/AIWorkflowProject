"""CSV logging utilities for training and prediction events."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TRAIN_FIELDS = [
    "timestamp_utc", "country", "model_name", "model_version",
    "train_start", "train_end", "validation_rmse", "validation_mae", "runtime_seconds",
]
PREDICT_FIELDS = [
    "timestamp_utc", "country", "forecast_date", "forecast_30d_revenue",
    "baseline_30d_revenue", "model_name", "model_version", "runtime_ms",
]


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def log_training(log_dir: str | Path, **values) -> Path:
    path = Path(log_dir) / "train_log.csv"
    values = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **values}
    _append_csv(path, TRAIN_FIELDS, values)
    return path


def log_prediction(log_dir: str | Path, **values) -> Path:
    path = Path(log_dir) / "predict_log.csv"
    values = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **values}
    _append_csv(path, PREDICT_FIELDS, values)
    return path


def read_log(log_dir: str | Path, log_type: str) -> list[dict]:
    if log_type not in {"train", "predict"}:
        raise ValueError("log_type must be 'train' or 'predict'")
    path = Path(log_dir) / f"{log_type}_log.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
