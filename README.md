# AAVAIL 30-Day Revenue Forecasting Service

This repository contains the final IBM AI Enterprise Workflow Capstone submission for the AAVAIL revenue forecasting case study.

The business requirement is to forecast revenue for the next 30 days for either all countries combined or a specific high-revenue country. The project includes automated data ingestion, exploratory analysis, multiple-model comparison, model training and serialization, a Flask API, unit tests, Docker packaging, prediction/training logs, and post-production performance monitoring against held-out production data.

## Repository structure

```text
AIWorkflowProject/
├── api/                    Flask application
├── cs-train/               Training transaction JSON files
├── cs-production/          Held-out production transaction JSON files
├── models/                 Trained model artifacts and manifest
├── reports/
│   ├── figures/            EDA and monitoring visualizations
│   └── metrics/            Reproducible evaluation outputs
├── scripts/                EDA, training, and post-production scripts
├── src/                    Reusable ingestion, feature, model, logging, monitoring code
├── tests/                  Unit tests for data, model, logging, and API
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_tests.py
```

## Business objective

AAVAIL managers need a repeatable service that forecasts the following 30 days of revenue. Forecasts must be available for the full business and for the ten countries with the most historical revenue. The service is designed around a country and a forecast-start date rather than a raw feature vector.

## Forecast design

For a requested forecast date, features use only information available before that date. The target is total revenue from the forecast date through the following 29 days.

Historical features include trailing 7-, 14-, 30-, and 70-day revenue, the comparable 30-day window from the previous year, recent invoice/view/purchase activity, and calendar seasonality. The trailing 30-day revenue is retained as a transparent baseline.

Model selection compares:

- Linear regression
- Random forest regression
- Gradient boosting regression
- Trailing 30-day baseline

Machine-learning candidates are compared with forward-chaining time-series validation. The best machine-learning candidate is then re-trained on all available training observations for each country.

## Reproduce the submission

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate the exploratory analysis:

```bash
python scripts/generate_eda.py
```

Train all models:

```bash
python scripts/train_models.py
```

Run post-production monitoring against the held-out production files:

```bash
python scripts/post_production.py
```

Run all unit tests with one command:

```bash
python run_tests.py
```

## Run the Flask API

```bash
python -m api.app
```

The default service URL is `http://localhost:8080`.

### Endpoints

- `GET /health` - service and model availability
- `POST /train` - re-train all aggregate and country models
- `POST /predict` - 30-day revenue forecast for a country/date
- `GET /logfile?type=predict` - prediction log
- `GET /logfile?type=train` - training log
- `GET /metrics` - latest post-production monitoring summary

Example prediction request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"united_kingdom","date":"2019-09-01"}'
```

`country` may be `all`, a normalized country key such as `united_kingdom`, or the matching human-readable country name.

## Docker

Build and run the same application in a container:

```bash
docker build -t aavail-revenue-forecast .
docker run --rm -p 8080:8080 aavail-revenue-forecast
```

Run the test suite inside the image with:

```bash
docker run --rm aavail-revenue-forecast python run_tests.py
```

Or:

```bash
docker compose up --build
```

The Docker build context contains the API, source modules, model artifacts, unit tests, data required by the submission, and monitoring/report outputs.

## Continuous integration

`.github/workflows/ci.yml` runs on GitHub pushes and pull requests. It installs the pinned Python dependencies, runs the complete unit-test suite, and builds the Docker image. A green workflow therefore validates both the Python submission and the container definition in the repository environment used for peer review.

## Peer-review checklist

| Peer-review requirement | Evidence in this repository |
|---|---|
| Unit tests for API | `tests/test_api.py` |
| Unit tests for model | `tests/test_model.py` |
| Unit tests for logging | `tests/test_logging.py` |
| All tests run from one script | `python run_tests.py` |
| Mechanism to monitor performance | `src/monitoring.py`, `scripts/post_production.py`, `GET /metrics` |
| Test read/write isolation | Tests use temporary directories supplied to logging/model functions |
| API predicts country and all-country revenue | `POST /predict`; model manifest includes `all` plus top countries |
| Automated data ingestion | `src/data_ingestion.py` |
| Multiple models compared | `src/modeling.py`, `reports/metrics/model_comparison.csv` |
| EDA uses visualizations | `scripts/generate_eda.py`, `reports/figures/` |
| Containerized application definition | `Dockerfile`, `docker-compose.yml` |
| Model vs baseline visualization | `reports/figures/production_model_vs_baseline_all.png` |

## Reports

- `reports/Part1_Data_Investigation.md`
- `reports/Part2_Model_Iteration.md`
- `reports/Part3_Final_Report.md`

Each report is backed by the generated CSV/JSON outputs and figures in `reports/metrics` and `reports/figures`.
