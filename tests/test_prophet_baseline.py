import numpy as np
import pandas as pd

from epicast.train.prophet_baseline import evaluate_holdout


def test_evaluate_holdout_recovers_a_simple_seasonal_series():
    # Two years of weekly data with a clean annual cycle: Prophet should
    # nail the last few points, which mainly checks the train/test ds
    # alignment (pd.concat + tail(horizon)) rather than forecast skill.
    weeks = pd.date_range("2020-01-05", periods=104, freq="W-SUN")
    y = 10 + 2 * np.sin(2 * np.pi * np.arange(104) / 52)
    series = pd.DataFrame({"ds": weeks, "y": y})

    metrics = evaluate_holdout(series, horizon=4)

    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] < 1.0
