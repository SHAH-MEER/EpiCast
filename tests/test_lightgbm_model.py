import numpy as np
import pandas as pd

from epicast.train.lightgbm_model import (
    RecursiveLightGBMForecaster,
    evaluate_holdout,
    evaluate_holdout_with_interval,
    fit_one_step_model,
    fit_quantile_model,
    forecast_recursive,
    forecast_recursive_with_interval,
)
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


def test_forecast_recursive_with_interval_brackets_the_point_forecast():
    series = _synthetic_series()
    train = series.iloc[:-4]
    train_features = build_features(train)

    point_model = fit_one_step_model(train_features)
    lower_model = fit_quantile_model(train_features, 0.1)
    upper_model = fit_quantile_model(train_features, 0.9)

    predictions, lower, upper = forecast_recursive_with_interval(point_model, lower_model, upper_model, train, horizon=4)

    assert predictions.shape == lower.shape == upper.shape == (4,)
    assert np.all(lower <= upper)


def test_evaluate_holdout_with_interval_reports_coverage_and_width():
    series = _synthetic_series()

    metrics = evaluate_holdout_with_interval(series, horizon=4)

    assert set(metrics) == {"mae", "rmse", "mape", "interval_coverage", "mean_interval_width"}
    assert 0.0 <= metrics["interval_coverage"] <= 1.0
    assert metrics["mean_interval_width"] >= 0.0


def test_recursive_forecaster_pyfunc_returns_expected_columns():
    series = _synthetic_series()
    features = build_features(series)

    point_model = fit_one_step_model(features)
    lower_model = fit_quantile_model(features, 0.1)
    upper_model = fit_quantile_model(features, 0.9)
    forecaster = RecursiveLightGBMForecaster(point_model, lower_model, upper_model)

    output = forecaster.predict(context=None, model_input=series.tail(10), params={"horizon": 3})

    assert list(output.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(output) == 3
    assert output["ds"].iloc[0] > series["ds"].iloc[-1]
