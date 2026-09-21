# Part 1 - Data Investigation

## Business opportunity

AAVAIL management needs a repeatable forecasting service that estimates revenue for the following 30 days. The service must support the company-wide total and country-specific projections so that managers can spend less time building their own forecasts and can use a common estimate for staffing and budget decisions.

The training data contain transaction-level purchases across multiple countries. The modeling population is limited to the ten countries with the greatest historical revenue, plus an all-country aggregate.

## Testable hypotheses

1. **Recent operating activity contains predictive signal.** Trailing revenue, invoices, purchases, and views should help explain the next 30 days of revenue.
2. **Country-level behavior differs enough to justify country-specific models.** Separate daily time series should improve the fit to local revenue patterns rather than forcing every country into one aggregate response.
3. **Calendar and prior-period patterns add information beyond a simple trailing-revenue baseline.** Month, day-of-week, and the comparable prior-year window are included as candidate features and evaluated empirically.

## Ideal data

The minimum useful data for this opportunity are transaction date, country, invoice identifier, revenue/price, and activity measures that can be aggregated over time. Additional business data would improve causal interpretation and forecast resilience, including promotions, pricing changes, product/service mix, customer acquisition and churn, planned launches, holidays, and country-level commercial events. Those variables are not available in the supplied case-study data and are therefore not used in this submission.

## Ingestion and data quality

`src/data_ingestion.py` automates ingestion of all JSON files in a supplied directory. It normalizes the known source-schema variants (`StreamID`, `TimesViewed`, `total_price`), cleans invoice identifiers, validates required fields, creates a canonical invoice date, and aggregates records to complete daily time series.

Training-period summary:

- Records: **815,011**
- Date range: **2017-11-28 to 2019-07-31**
- Countries: **43**
- Unique invoices: **42,646**
- Total revenue: **$3,914,197**
- Missing customer IDs: **189,762**; customer ID is not required for the forecast features
- Cleaned invoices with missing ID: **0**

## Revenue concentration

The ten highest-revenue countries account for **97.8%** of training-period revenue. The United Kingdom alone contributes **90.0%**, which is material when interpreting the all-country forecast.

| Rank | Country | Model key | Training revenue |
|---:|---|---|---:|
| 1 | United Kingdom | united_kingdom | $3,521,514 |
| 2 | EIRE | eire | $107,069 |
| 3 | Germany | germany | $49,272 |
| 4 | France | france | $40,565 |
| 5 | Norway | norway | $38,495 |
| 6 | Spain | spain | $16,041 |
| 7 | Hong Kong | hong_kong | $14,453 |
| 8 | Portugal | portugal | $13,529 |
| 9 | Singapore | singapore | $13,176 |
| 10 | Netherlands | netherlands | $12,323 |

## Exploratory findings

The training-period daily correlation between views and revenue is **0.482**, indicating a meaningful but incomplete relationship between engagement and realized revenue. In the aggregate supervised feature set, recent operating activity is more strongly associated with the 30-day target than raw recent revenue alone; this supports retaining invoice, view, and purchase activity alongside revenue windows.

The generated visual evidence is:

- `reports/figures/top_10_country_revenue.png`
- `reports/figures/monthly_revenue.png`
- `reports/figures/daily_views_vs_revenue.png`

The corresponding reusable generation code is `scripts/generate_eda.py`.

## Part 1 conclusion

The supplied data are sufficient to build and test the requested 30-day forecasting service. They also expose two operating realities that matter downstream: revenue is highly concentrated in the United Kingdom, and the available features are transactional rather than explanatory business drivers. Model performance therefore has to be evaluated against a simple recent-revenue baseline and monitored after deployment rather than assumed to remain stable.
