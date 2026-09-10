"""Pull the CDC ILINet (FluView) weekly national ILI series from the Delphi Epidata API.

Delphi Epidata API docs: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
Anonymous access is rate-limited but sufficient for this project; no API key is required.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
from epiweeks import Week

EPIDATA_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
DEFAULT_REGION = "nat"
# Generously covers 2010 through the near future; Epidata simply omits epiweeks
# it has no data for, so an over-wide range is not an error.
DEFAULT_EPIWEEK_RANGE = "201001-202653"
DEFAULT_OUTPUT = Path("data/processed/ilinet_national_weekly.csv")

logger = logging.getLogger(__name__)


class EpidataError(RuntimeError):
    """Raised when the Epidata API responds with a non-success result."""


def fetch_ilinet(region: str = DEFAULT_REGION, epiweeks: str = DEFAULT_EPIWEEK_RANGE) -> pd.DataFrame:
    """Fetch raw FluView ILINet records for a region/epiweek range."""
    response = requests.get(
        EPIDATA_URL,
        params={"regions": region, "epiweeks": epiweeks},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("result") != 1:
        raise EpidataError(f"Epidata API returned result={payload.get('result')}: {payload.get('message')}")

    return pd.DataFrame(payload["epidata"])


def clean_weekly_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw ILINet records to one row per epiweek, with a real calendar date.

    Epiweeks (e.g. 201501) are CDC/MMWR weeks, not ISO weeks, so date conversion
    goes through the `epiweeks` package rather than a strptime directive.
    """
    df = raw.copy()
    df["week_start"] = pd.to_datetime(
        df["epiweek"].apply(lambda ew: Week.fromstring(str(ew)).startdate())
    )
    # The API can return multiple issues (revisions) per epiweek; keep the latest.
    df = df.sort_values(["epiweek", "issue"]).drop_duplicates(subset="epiweek", keep="last")
    columns = ["epiweek", "week_start", "wili", "ili", "num_ili", "num_patients"]
    return df[columns].sort_values("week_start").reset_index(drop=True)


def save_processed(df: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION, help="Epidata region code (default: nat)")
    parser.add_argument("--epiweeks", default=DEFAULT_EPIWEEK_RANGE, help="Epiweek range, e.g. 201001-202653")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the cleaned CSV")
    args = parser.parse_args()

    logger.info("Fetching ILINet region=%s epiweeks=%s", args.region, args.epiweeks)
    raw = fetch_ilinet(region=args.region, epiweeks=args.epiweeks)
    logger.info("Fetched %d raw records", len(raw))

    clean = clean_weekly_series(raw)
    output_path = save_processed(clean, args.output)
    logger.info(
        "Wrote %d weekly rows (%s to %s) -> %s",
        len(clean),
        clean["week_start"].min().date(),
        clean["week_start"].max().date(),
        output_path,
    )


if __name__ == "__main__":
    main()
