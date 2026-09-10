"""Evidently drift report: flags when the most recently ingested weeks of ILI data
diverge from the historical distribution -- adjusted for season, not raw history.

ILI is strongly seasonal (winter peaks look nothing like summer troughs), so the
reference set is *not* "all prior history": it's every prior year's rows that fall on
the same calendar weeks as the current window (e.g. if current covers ISO weeks 23-34,
reference is every historical week 23-34 from earlier years). Comparing a summer window
against a reference mixing every season would flag "drift" essentially every time,
purely from seasonality -- confirmed empirically before writing this (current: wili
mean ~1.0 in a 0.87-1.32 range; naive whole-history reference: mean ~2.15 ranging up to
8.3 during flu peaks). A same-season comparison asks the actually useful question: is
this year's summer behaving like previous summers, not "does summer look like winter."

Simplification worth knowing: Phase 2's models refit on the *entire* series each
training run, so "reference" here isn't a held-out set the champion model has never
seen -- it's a stand-in for "what this time of year normally looks like." That's still
the real thing this report exists to catch: a live batch that looks nothing like the
historical pattern for this point in the season, which is exactly what would justify
retraining.

Monitored columns are the rate signal (wili, ili) the model actually forecasts, not the
raw counts (num_ili, num_patients) or the engineered lag/rolling/seasonal features.
num_ili/num_patients are excluded because CDC's ILINet reporting network has grown
substantially over the dataset's history -- confirmed empirically before writing this:
mean num_patients in a same-season reference window was ~1.15M vs. ~2.5M in the current
window, more than double, purely from more sentinel providers enrolling over the years,
unrelated to any actual disease dynamics. Including them made the dataset-level drift
test fail on effectively every run regardless of the real signal, which is worse than
no monitoring at all -- an alert that always fires trains people to ignore it. The
seasonal features are excluded because they're deterministic functions of the calendar
date and would be identical by construction across same-week-of-year comparisons.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset

DEFAULT_DATA = Path("data/processed/ilinet_national_weekly.csv")
DEFAULT_OUTPUT = Path("reports/drift_report.html")
DEFAULT_CURRENT_WEEKS = 12
MONITORED_COLUMNS = ["wili", "ili"]
DRIFTED_COLUMNS_COUNT_TYPE = "evidently:metric_v2:DriftedColumnsCount"

logger = logging.getLogger(__name__)


def split_reference_and_current(df: pd.DataFrame, current_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current = the most recent `current_weeks` weeks. Reference = every earlier
    year's rows on those same calendar (ISO) weeks, so the comparison is season-aware
    rather than "this window vs. every season mixed together."
    """
    ordered = df.sort_values("week_start").reset_index(drop=True)
    current = ordered.iloc[-current_weeks:]
    history = ordered.iloc[:-current_weeks]

    current_weeks_of_year = current["week_start"].dt.isocalendar().week.unique()
    reference = history[history["week_start"].dt.isocalendar().week.isin(current_weeks_of_year)]

    return reference, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
    report = Report([DataDriftPreset()], include_tests=True)
    return report.run(current_data=current[MONITORED_COLUMNS], reference_data=reference[MONITORED_COLUMNS])


def dataset_drifted(snapshot: Snapshot) -> bool:
    """True if the dataset-level "share of drifted columns" test failed, i.e. the
    share of monitored columns that individually drifted crossed DataDriftPreset's
    default threshold (50%).
    """
    for test in snapshot.dict()["tests"]:
        if test["metric_config"]["params"]["type"] == DRIFTED_COLUMNS_COUNT_TYPE:
            return test["status"] == "FAIL"
    raise RuntimeError("Expected a DriftedColumnsCount test in the drift report but found none")


def generate_report(
    data_path: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    current_weeks: int = DEFAULT_CURRENT_WEEKS,
) -> tuple[Snapshot, bool]:
    """Load data, run the season-aware drift report, save it as HTML, and return
    (snapshot, drifted). Shared by drift_report's CLI and retrain_trigger, so the two
    never drift apart on what "checking for drift" actually means.
    """
    df = pd.read_csv(data_path, parse_dates=["week_start"])
    reference, current = split_reference_and_current(df, current_weeks)
    logger.info(
        "Reference: %d weeks (%s -> %s). Current: %d weeks (%s -> %s).",
        len(reference),
        reference["week_start"].min().date(),
        reference["week_start"].max().date(),
        len(current),
        current["week_start"].min().date(),
        current["week_start"].max().date(),
    )

    snapshot = run_drift_report(reference, current)

    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output))
    logger.info("Drift report written to %s", output)

    return snapshot, dataset_drifted(snapshot)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-weeks", type=int, default=DEFAULT_CURRENT_WEEKS)
    args = parser.parse_args()

    _, drifted = generate_report(args.data, args.output, args.current_weeks)

    if drifted:
        logger.warning("DRIFT DETECTED: the current window diverges from the reference distribution")
        sys.exit(1)

    logger.info("No significant drift detected")


if __name__ == "__main__":
    main()
