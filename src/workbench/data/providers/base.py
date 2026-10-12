"""The one interface all price data sources implement."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class PriceProvider(Protocol):
    """Adjusted-close (total-return where possible) daily prices for a set of tickers."""

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Return prices indexed by date (ascending), one column per ticker."""
        ...
