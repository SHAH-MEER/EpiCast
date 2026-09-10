"""Lag/rolling/seasonal feature engineering shared by the LightGBM training and serving paths.

Every feature here only looks backward from its own row (lags shift by >=1, rolling
windows shift by 1 before aggregating), so it's safe to compute over a full series and
slice train/test afterwards without leaking future values into past rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4, 52)
ROLLING_WINDOWS = (4, 8)

FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["week_sin", "week_cos"]
)


def build_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling/seasonal columns to a ds/y frame. Early rows get NaN features."""
    df = series.sort_values("ds").reset_index(drop=True).copy()

    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = df["y"].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    week_of_year = df["ds"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    return df
