"""Closed-loop retraining trigger: pull fresh data, check for drift, and if the
drift threshold is crossed, retrain and register a new LightGBM model version.

This is what turns Phase 6's drift report from "a report someone has to remember to
open" into an actual closed loop -- the difference CLAUDE.md draws between "an MLOps
demo" and "a closed-loop system." It's self-contained (it ingests its own fresh data
rather than assuming a CSV is already sitting there) so it can genuinely be run
standalone on a schedule -- cron, Task Scheduler, a scheduled CI job -- without any
other setup.

Retraining runs as a subprocess against the existing, already-tested
`epicast.train.lightgbm_model` CLI rather than importing and calling its internals
directly: that script's `main()` owns its own argument parsing and MLflow run
lifecycle, and reusing it as a subprocess is simpler and safer than trying to
re-enter it from another process's argv.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from epicast.data import DEFAULT_DATA
from epicast.ingest.fetch_ilinet import clean_weekly_series, fetch_ilinet, save_processed
from epicast.monitor.drift_report import DEFAULT_CURRENT_WEEKS, DEFAULT_OUTPUT, generate_report

logger = logging.getLogger(__name__)


def ingest(data_path: Path) -> None:
    raw = fetch_ilinet()
    clean = clean_weekly_series(raw)
    save_processed(clean, data_path)
    logger.info("Ingested %d weekly rows -> %s", len(clean), data_path)


def retrain() -> None:
    logger.info("Retraining and registering a new LightGBM model version")
    subprocess.run(
        [sys.executable, "-m", "epicast.train.lightgbm_model", "--register"],
        check=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-weeks", type=int, default=DEFAULT_CURRENT_WEEKS)
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Reuse the existing data file instead of re-fetching"
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain regardless of drift (for testing the trigger itself)"
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        ingest(args.data)

    _, drifted = generate_report(args.data, args.output, args.current_weeks)

    if drifted or args.force:
        logger.warning("Retraining triggered (%s)", "drift detected" if drifted else "--force")
        retrain()
    else:
        logger.info("No drift detected -- skipping retrain")


if __name__ == "__main__":
    main()
