"""Real market data via yfinance, behind the PriceProvider interface, cached on disk."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from workbench.data.cache import DEFAULT_CACHE_DIR, PriceCache


def _fetch_one(ticker: str, start: date, end: date) -> pd.Series:
    """Fetch one ticker's adjusted close, normalising LSE pence (GBp) quotes to GBP."""
    handle = yf.Ticker(ticker)
    history = handle.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if history.empty:
        raise ValueError(f"no price data returned for {ticker!r} between {start} and {end}")

    close = history["Close"]
    close.index = close.index.tz_localize(None)

    try:
        currency = handle.fast_info.get("currency")
    except Exception:
        currency = None
    if currency == "GBp":
        close = close / 100.0

    close.name = ticker
    return pd.Series(close)


class YFinanceProvider:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._cache = PriceCache(cache_dir)

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        series_by_ticker = {
            ticker: self._cache.get_or_fetch(ticker, start, end, _fetch_one) for ticker in tickers
        }
        frame = pd.concat(series_by_ticker, axis=1)
        return frame.dropna(how="all")
