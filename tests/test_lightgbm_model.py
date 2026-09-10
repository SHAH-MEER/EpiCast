import numpy as np
import pandas as pd

from epicast.train.lightgbm_model import evaluate_holdout, fit_one_step_model, forecast_recursive
from epicast.features import build_features


def _synthetic_series(n=160):
    weeks = pd.date_range("2018-01-07", periods=n, freq="W-SUN")
    y = 10 + 2 * np.sin(2 * np.pi * np.arange(n) / 52)
    return pd.DataFrame({"ds": weeks, "y": y})


def test_forecast_recursive_returns_one_prediction_per_horizon_step():
    series = _synthetic_series()
    train = series.iloc[:-4]

    model = fit_one_step_model(build_features(train))
    predictions = forecast_recursive(model, train, horizon=4)

    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


def test_evaluate_holdout_recovers_a_simple_seasonal_series():
    series = _synthetic_series()

    metrics = evaluate_holdout(series, horizon=4)

    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] < 1.0
