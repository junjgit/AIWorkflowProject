#!/usr/bin/env python
"""Generate reproducible EDA metrics and visualizations for Part 1."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_ingestion import data_summary, fetch_data, top_revenue_countries


def main() -> None:
    df = fetch_data(ROOT / "cs-train")
    metrics_dir = ROOT / "reports" / "metrics"
    figures_dir = ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = data_summary(df)
    ranking = top_revenue_countries(df, 10)
    monthly = (
        df.set_index("invoice_date")["price"]
        .resample("MS")
        .sum()
        .rename("revenue")
        .reset_index()
    )
    daily = (
        df.set_index("invoice_date")
        .resample("D")
        .agg(revenue=("price", "sum"), views=("times_viewed", "sum"), invoices=("invoice", "nunique"))
        .fillna(0)
        .reset_index()
    )

    (metrics_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranking.to_csv(metrics_dir / "top_10_countries.csv", index=False)
    monthly.to_csv(metrics_dir / "monthly_revenue.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    plt.bar(ranking["country"], ranking["total_revenue"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Historical revenue")
    plt.title("Top 10 countries by training-period revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_10_country_revenue.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    plt.plot(monthly["invoice_date"], monthly["revenue"], marker="o")
    plt.ylabel("Monthly revenue")
    plt.xlabel("Month")
    plt.title("Training-period monthly revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_revenue.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    plt.scatter(daily["views"], daily["revenue"], alpha=0.45, s=14)
    plt.xlabel("Daily views")
    plt.ylabel("Daily revenue")
    plt.title("Daily views and revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "daily_views_vs_revenue.png", dpi=160)
    plt.close()

    corr = float(daily[["views", "revenue"]].corr().iloc[0, 1])
    summary["daily_views_revenue_correlation"] = corr
    (metrics_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
