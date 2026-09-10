"""Shared access to the cleaned weekly ILI series for every model in this project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA = Path("data/processed/ilinet_national_weekly.csv")
DEFAULT_TARGET = "wili"


def load_series(data_path: Path = DEFAULT_DATA, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Load the cleaned weekly ILI series as a plain ds/y frame, sorted by date."""
    df = pd.read_csv(data_path, parse_dates=["week_start"])
    return df[["week_start", target]].rename(columns={"week_start": "ds", target: "y"}).sort_values("ds").reset_index(drop=True)
