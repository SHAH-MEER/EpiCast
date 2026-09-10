import numpy as np

from epicast.metrics import forecast_metrics


def test_forecast_metrics_zero_error():
    actual = np.array([1.0, 2.0, 3.0])
    metrics = forecast_metrics(actual, actual)

    assert metrics == {"mae": 0.0, "rmse": 0.0, "mape": 0.0}


def test_forecast_metrics_known_values():
    actual = np.array([2.0, 4.0])
    predicted = np.array([1.0, 5.0])

    metrics = forecast_metrics(actual, predicted)

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["mape"] == 37.5  # mean(|1/2|, |1/4|) * 100
