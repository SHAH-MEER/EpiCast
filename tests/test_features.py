import numpy as np
import pandas as pd
import pytest

from epicast.features import FEATURE_COLUMNS, build_features


def test_build_features_lags_align_to_prior_rows():
    series = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-05", periods=10, freq="W-SUN"),
            "y": np.arange(10, dtype=float),
        }
    )

    features = build_features(series)

    # lag_1 at row i should equal y at row i-1.
    assert features["lag_1"].iloc[5] == series["y"].iloc[4]
    assert pd.isna(features["lag_1"].iloc[0])  # no prior row to lag from
    assert all(col in features.columns for col in FEATURE_COLUMNS)


def test_build_features_rolling_mean_excludes_current_row():
    series = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-05", periods=6, freq="W-SUN"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        }
    )

    features = build_features(series)

    # rolling_mean_4 at the last row should average y[1:5] (shifted by 1), not include y[5]=100.
    assert features["rolling_mean_4"].iloc[5] == pytest.approx(np.mean([2.0, 3.0, 4.0, 5.0]))
