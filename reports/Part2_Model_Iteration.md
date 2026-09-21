# Part 2 - Model Iteration

## Forecast target

For forecast start date `d`, the target is total revenue from `d` through `d + 29 days`. All engineered features use information strictly before `d`, preventing target leakage.

The reusable implementation is in `src/features.py`.

## Features

The model compares recent behavior at several horizons and includes:

- revenue over the previous 7, 14, 30, and 70 days
- revenue for the comparable 30-day window one year earlier
- average invoices, views, and purchases over the previous 30 days
- ratio of 7-day to 30-day revenue
- cyclical month and day-of-week features

The previous 30 days of revenue also serve as a transparent baseline forecast.

## Modeling approaches

Three machine-learning regressors are compared for every model population:

1. Linear regression with standardized features
2. Random forest regression
3. Gradient boosting regression

A trailing-30-day baseline is evaluated alongside them. Candidate models use forward-chaining `TimeSeriesSplit` validation so later observations are never used to predict earlier validation periods.

## Validation results

The selected machine-learning model is the candidate with the lowest mean time-series validation RMSE for each population. It is then re-trained on all training observations and serialized to `models/`.

| Population | Selected model | Validation RMSE | Validation MAE | Baseline RMSE |
|---|---|---:|---:|---:|
| all | random forest | 101,007.3 | 88,760.0 | 118,433.8 |
| eire | random forest | 4,680.5 | 3,668.0 | 5,705.0 |
| france | gradient boosting | 1,536.8 | 1,261.3 | 1,554.5 |
| germany | random forest | 1,074.8 | 844.4 | 1,368.6 |
| hong_kong | gradient boosting | 2,472.7 | 2,013.0 | 2,702.7 |
| netherlands | gradient boosting | 333.7 | 289.6 | 418.1 |
| norway | gradient boosting | 8,624.7 | 7,739.4 | 11,742.5 |
| portugal | linear regression | 1,245.6 | 843.9 | 2,294.7 |
| singapore | gradient boosting | 3,772.9 | 2,817.6 | 5,103.3 |
| spain | gradient boosting | 766.7 | 591.0 | 1,002.1 |
| united_kingdom | random forest | 116,346.6 | 104,566.7 | 115,959.0 |

The complete candidate-level results are in `reports/metrics/model_comparison.csv`.

## Model artifacts

Training produces **11 deployed artifacts**: `all` plus the ten highest-revenue countries. `models/model_manifest.json` records the model version, selected model family, validation metrics, and artifact filename for each population.

The full training workflow can be reproduced with:

```bash
python scripts/train_models.py
```

## Part 2 conclusion

Multiple algorithms were compared using a time-aware validation design instead of a random train/test split. The selected model family differs by country, which is consistent with the heterogeneous and sometimes sparse country-level series. Validation alone is not treated as sufficient evidence of deployment quality; Part 3 evaluates the frozen models against later held-out production data and compares them with the baseline.
