# Part 3 - Model Production and Final Report

## Production design

The deployable service contains four operating layers:

1. **Ingestion** - `src/data_ingestion.py` reads and validates transaction JSON files and constructs daily country time series.
2. **Model service** - `src/modeling.py` trains, serializes, loads, and predicts 30-day revenue for `all` and the ten highest-revenue countries.
3. **API and logging** - `api/app.py` exposes Flask endpoints for training, prediction, logs, health, and monitoring metrics. `src/logging_utils.py` records training and prediction activity.
4. **Monitoring** - `src/monitoring.py` compares frozen-model forecasts with known later revenue and the trailing-30-day baseline.

## Flask API

The API accepts a country and forecast-start date rather than requiring an end user to construct a model feature vector.

Endpoints:

- `GET /health`
- `POST /train`
- `POST /predict`
- `GET /logfile?type=train`
- `GET /logfile?type=predict`
- `GET /metrics`

Example request:

```json
{
  "country": "united_kingdom",
  "date": "2019-09-01"
}
```

The same endpoint accepts `all` to forecast all countries combined.

## Test-driven validation

The repository contains distinct tests for:

- data ingestion: `tests/test_data_ingestion.py`
- model and 30-day target behavior: `tests/test_model.py`
- logging and read/write isolation: `tests/test_logging.py`
- Flask API behavior: `tests/test_api.py`

All tests are invoked through one command:

```bash
python run_tests.py
```

Tests that write models or logs use temporary directories rather than the repository's production `models/` or `logs/` paths.

## Docker containerization

`Dockerfile` packages the application, source code, data, models, tests, and reports into a Python 3.13 image. The container starts the Flask API on port 8080 and includes a `/health` health check. `docker-compose.yml` provides the equivalent compose configuration with persistent model, log, and report mounts.

Build and run:

```bash
docker build -t aavail-revenue-forecast .
docker run --rm -p 8080:8080 aavail-revenue-forecast
```

## Post-production monitoring

The held-out production files span **2019-08-01 to 2019-12-06**. Because each label requires a fully observed 30-day future window, performance is evaluated through forecast start date **2019-11-07**.

For the all-country model, the held-out results are:

- Model RMSE: **167,430.4**
- Model MAE: **139,701.2**
- Trailing-30-day baseline RMSE: **64,956.0**
- Trailing-30-day baseline MAE: **59,852.8**
- Target mean shift versus training: **+20.9%**

The frozen selected model does **not** outperform the baseline on the aggregate held-out production period. This is the key post-production finding, not something to hide: the monitoring mechanism detects that validation performance did not transfer cleanly into the later operating period. The correct production response would be to investigate the shift, re-train with newly observed data, and reconsider the model-selection strategy before relying on the machine-learning forecast for business decisions.

Country-level results show mixed behavior, which reinforces the need to monitor each deployed population rather than infer local performance from the aggregate.

| Population | Forecasts | Model RMSE | Baseline RMSE | RMSE improvement | Target mean shift |
|---|---:|---:|---:|---:|---:|
| all | 99 | 167,430.4 | 64,956.0 | -157.8% | +20.9% |
| eire | 98 | 9,207.5 | 1,184.4 | -677.4% | -19.6% |
| france | 99 | 7,561.4 | 7,938.6 | +4.8% | +238.8% |
| germany | 99 | 1,754.4 | 1,605.5 | -9.3% | +80.4% |
| hong_kong | 74 | 2,745.4 | 4,480.9 | +38.7% | +169.3% |
| netherlands | 98 | 212.3 | 318.9 | +33.4% | +19.3% |
| norway | 99 | 602.8 | 234.6 | -157.0% | -75.6% |
| portugal | 97 | 3,720.0 | 1,428.8 | -160.4% | +24.5% |
| singapore | 48 | 1,864.9 | 1,781.3 | -4.7% | -75.1% |
| spain | 97 | 543.0 | 480.7 | -13.0% | +69.5% |
| united_kingdom | 99 | 174,998.0 | 63,600.3 | -175.2% | +17.7% |

Monitoring outputs:

- `reports/metrics/post_production_metrics.csv`
- `reports/metrics/post_production_predictions.csv`
- `reports/metrics/post_production_summary.json`
- `reports/figures/production_model_vs_baseline_all.png`
- `reports/figures/production_rmse_by_country.png`

The first monitoring figure directly compares actual 30-day revenue, the deployed model, and the baseline over time, satisfying the requirement to visualize model performance against the baseline.

## Final assessment

The submission implements the complete enterprise workflow required for peer review: automated ingestion, EDA, model comparison, training, country/date prediction, API serving, logging, unit tests, Docker packaging, and post-production monitoring. The held-out period also demonstrates why monitoring is part of the production system: several model populations experience material distribution or performance changes after the training period.
