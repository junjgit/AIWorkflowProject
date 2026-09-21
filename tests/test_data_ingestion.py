import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import convert_to_ts, fetch_data, normalize_country_key


def _write(path: Path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def test_fetch_data_normalizes_schema_and_invoice(tmp_path):
    records = [
        {"country": "United Kingdom", "customer_id": 1, "day": 1, "invoice": "A123B", "month": 1,
         "total_price": 10.5, "StreamID": "S1", "TimesViewed": 3, "year": 2019},
        {"country": "United Kingdom", "customer_id": 2, "day": 3, "invoice": "124", "month": 1,
         "total_price": 12.0, "StreamID": "S2", "TimesViewed": 4, "year": 2019},
    ]
    _write(tmp_path / "sample.json", records)
    df = fetch_data(tmp_path)
    assert "price" in df.columns
    assert "stream_id" in df.columns
    assert "times_viewed" in df.columns
    assert df.loc[0, "invoice"] == "123"
    assert pd.api.types.is_datetime64_any_dtype(df["invoice_date"])


def test_convert_to_ts_fills_missing_calendar_days(tmp_path):
    records = [
        {"country": "US", "customer_id": 1, "day": 1, "invoice": "1", "month": 1, "price": 10,
         "stream_id": "S1", "times_viewed": 1, "year": 2019},
        {"country": "US", "customer_id": 2, "day": 3, "invoice": "2", "month": 1, "price": 20,
         "stream_id": "S2", "times_viewed": 2, "year": 2019},
    ]
    _write(tmp_path / "sample.json", records)
    ts = convert_to_ts(fetch_data(tmp_path), country="US")
    assert len(ts) == 3
    assert ts.loc[1, "revenue"] == 0
    assert ts.loc[2, "revenue"] == 20


def test_fetch_data_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        fetch_data(tmp_path)


def test_country_key_is_stable():
    assert normalize_country_key("United Kingdom") == "united_kingdom"
