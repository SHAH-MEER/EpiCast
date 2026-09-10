import numpy as np
import pandas as pd

from epicast.monitor.drift_report import (
    MONITORED_COLUMNS,
    dataset_drifted,
    run_drift_report,
    split_reference_and_current,
)


def _seasonal_series(n_years=5, seed=0, current_shift=0.0, current_weeks=12):
    """n_years of weekly data with a real annual cycle, so every calendar week repeats
    across years -- required for split_reference_and_current's same-week-of-year
    matching to find any reference rows at all. Only the most recent `current_weeks`
    are offset by `current_shift`, simulating a genuine seasonal anomaly.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 52
    weeks = pd.date_range("2015-01-04", periods=n, freq="W-SUN")
    seasonal = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 52)

    df = pd.DataFrame(
        {
            "week_start": weeks,
            "wili": seasonal + rng.normal(0, 0.1, n),
            "ili": seasonal + rng.normal(0, 0.1, n),
            "num_ili": (seasonal * 5000) + rng.normal(0, 200, n),
            "num_patients": rng.normal(800000, 5000, n),
        }
    )
    if current_shift:
        for column in MONITORED_COLUMNS:
            df.loc[df.index[-current_weeks:], column] += current_shift
    return df


def test_split_reference_and_current_matches_calendar_week_not_raw_history():
    df = _seasonal_series(n_years=5)

    reference, current = split_reference_and_current(df, current_weeks=12)

    assert len(current) == 12
    # Reference should be drawn only from years before the current window, and only
    # from the same ISO weeks the current window covers.
    assert reference["week_start"].max() < current["week_start"].min()
    current_iso_weeks = set(current["week_start"].dt.isocalendar().week)
    reference_iso_weeks = set(reference["week_start"].dt.isocalendar().week)
    assert reference_iso_weeks <= current_iso_weeks
    # 5 years of history, minus the year the current window is drawn from, should
    # leave multiple prior occurrences of each of those calendar weeks.
    assert len(reference) >= 12 * 3


def test_dataset_drifted_is_false_when_current_season_matches_history():
    df = _seasonal_series(n_years=6, seed=1, current_shift=0.0)
    reference, current = split_reference_and_current(df, current_weeks=12)

    snapshot = run_drift_report(reference, current)

    assert dataset_drifted(snapshot) is False


def test_dataset_drifted_is_true_for_a_genuine_seasonal_anomaly():
    df = _seasonal_series(n_years=6, seed=2, current_shift=5.0)
    reference, current = split_reference_and_current(df, current_weeks=12)

    snapshot = run_drift_report(reference, current)

    assert dataset_drifted(snapshot) is True
