from src.logging_utils import log_prediction, log_training, read_log


def test_train_and_predict_logs_are_isolated_to_supplied_directory(tmp_path):
    log_training(
        tmp_path,
        country="all",
        model_name="linear_regression",
        model_version="test",
        train_start="2018-01-01",
        train_end="2019-01-01",
        validation_rmse=10.0,
        validation_mae=8.0,
        runtime_seconds=0.1,
    )
    log_prediction(
        tmp_path,
        country="all",
        forecast_date="2019-01-02",
        forecast_30d_revenue=100.0,
        baseline_30d_revenue=95.0,
        model_name="linear_regression",
        model_version="test",
        runtime_ms=2.0,
    )
    assert len(read_log(tmp_path, "train")) == 1
    assert len(read_log(tmp_path, "predict")) == 1
    assert (tmp_path / "train_log.csv").exists()
    assert (tmp_path / "predict_log.csv").exists()
