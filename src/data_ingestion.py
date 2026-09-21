"""Data ingestion and time-series preparation for the AAVAIL capstone."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "country", "customer_id", "day", "invoice", "month",
    "price", "stream_id", "times_viewed", "year",
]
COLUMN_ALIASES = {
    "StreamID": "stream_id",
    "TimesViewed": "times_viewed",
    "total_price": "price",
}


def normalize_country_key(country: str) -> str:
    """Return a stable URL/model-safe country key."""
    key = re.sub(r"[^a-z0-9]+", "_", str(country).strip().lower()).strip("_")
    return key or "unknown"


def _json_files(data_dir: str | Path) -> list[Path]:
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in data directory: {path}")
    return files


def fetch_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and normalize all JSON transaction files from a directory.

    The source files contain several historical column-name variants. This
    function normalizes those variants, validates required fields, cleans
    invoice identifiers, constructs invoice_date, sorts the records, and
    returns one DataFrame.
    """
    frames: list[pd.DataFrame] = []
    for file_path in _json_files(data_dir):
        frame = pd.read_json(file_path)
        frame = frame.rename(columns=COLUMN_ALIASES)
        missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(CANONICAL_COLUMNS))
        if missing or unexpected:
            raise ValueError(
                f"Schema mismatch in {file_path.name}; missing={missing}, unexpected={unexpected}"
            )
        frames.append(frame[CANONICAL_COLUMNS].copy())

    df = pd.concat(frames, ignore_index=True)

    for column in ["year", "month", "day", "price", "times_viewed"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid_critical = df[["year", "month", "day", "price", "country"]].isna().any(axis=1)
    if invalid_critical.any():
        raise ValueError(f"Found {int(invalid_critical.sum())} rows with invalid critical fields")

    df["times_viewed"] = df["times_viewed"].fillna(0)
    df["invoice"] = (
        df["invoice"].astype(str).str.replace(r"\D+", "", regex=True).replace("", np.nan)
    )
    df["invoice_date"] = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int), day=df["day"].astype(int)),
        errors="raise",
    )
    df["country"] = df["country"].astype(str).str.strip()
    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df


def combine_data_dirs(data_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """Load multiple transaction directories into a single deduplicated frame."""
    frames = [fetch_data(directory) for directory in data_dirs]
    df = pd.concat(frames, ignore_index=True)
    # Files are month-partitioned, but deduplication makes the helper safe if
    # overlapping directories are supplied.
    subset = ["country", "customer_id", "invoice", "stream_id", "invoice_date", "price", "times_viewed"]
    df = df.drop_duplicates(subset=subset).sort_values("invoice_date").reset_index(drop=True)
    return df


def top_revenue_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top n countries ranked by total historical revenue."""
    table = (
        df.groupby("country", as_index=False)["price"]
        .sum()
        .rename(columns={"price": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    table["country_key"] = table["country"].map(normalize_country_key)
    return table[["country", "country_key", "total_revenue"]]


def convert_to_ts(df_orig: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Aggregate transactions to a complete daily time series."""
    if country is not None:
        available = set(df_orig["country"].unique())
        if country not in available:
            raise ValueError(f"Country not found: {country}")
        df = df_orig.loc[df_orig["country"] == country]
    else:
        df = df_orig

    if df.empty:
        raise ValueError("No records available for requested time series")

    grouped = (
        df.groupby("invoice_date")
        .agg(
            purchases=("price", "size"),
            unique_invoices=("invoice", "nunique"),
            unique_streams=("stream_id", "nunique"),
            total_views=("times_viewed", "sum"),
            revenue=("price", "sum"),
        )
        .sort_index()
    )
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_index, fill_value=0)
    grouped.index.name = "date"
    result = grouped.reset_index()
    result["year_month"] = result["date"].dt.strftime("%Y-%m")
    return result[[
        "date", "purchases", "unique_invoices", "unique_streams",
        "total_views", "year_month", "revenue",
    ]]


def build_time_series(df: pd.DataFrame, top_n: int = 10) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Create aggregate and top-country daily series.

    Returns
    -------
    series : dict
        Keys are ``all`` plus normalized country keys.
    country_map : dict
        Maps each key to the original display name.
    """
    ranking = top_revenue_countries(df, n=top_n)
    series = {"all": convert_to_ts(df)}
    country_map = {"all": "All Countries"}
    for row in ranking.itertuples(index=False):
        series[row.country_key] = convert_to_ts(df, country=row.country)
        country_map[row.country_key] = row.country
    return series, country_map


def data_summary(df: pd.DataFrame) -> dict:
    """Return a concise set of data-quality and coverage metrics."""
    return {
        "records": int(len(df)),
        "start_date": str(df["invoice_date"].min().date()),
        "end_date": str(df["invoice_date"].max().date()),
        "countries": int(df["country"].nunique()),
        "invoices": int(df["invoice"].nunique(dropna=True)),
        "total_revenue": float(df["price"].sum()),
        "missing_customer_id": int(df["customer_id"].isna().sum()),
        "missing_invoice": int(df["invoice"].isna().sum()),
    }
