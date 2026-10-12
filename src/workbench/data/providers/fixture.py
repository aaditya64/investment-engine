"""Frozen real historical prices loaded from a CSV, for network-free tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


class FixtureProvider:
    """CSV must have a 'Date' column plus one column per ticker."""

    def __init__(self, csv_path: Path) -> None:
        self._frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        missing = set(tickers) - set(self._frame.columns)
        if missing:
            raise KeyError(f"fixture does not contain tickers: {sorted(missing)}")

        window = self._frame.loc[str(start) : str(end), tickers]
        if window.empty:
            raise ValueError(f"fixture has no rows between {start} and {end}")
        return window
