"""Adjusted yfinance prices with one request-window cache; no provider framework."""

import hashlib
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path(".cache/prices")
CACHE_SECONDS = 24 * 60 * 60


class PriceDataError(RuntimeError):
    """Market data could not be fetched or cached."""


def get_prices(
    tickers: list[str], start: date, end: date, cache_dir: Path = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """Get daily adjusted closes, including end; cache each request for 24 hours.

    The cache key includes tickers and requested dates, so weekends/holidays do not
    prevent cache hits. LSE quotes in pence are normalised to GBP when identified.
    """
    key = hashlib.sha256(f"{','.join(tickers)}:{start}:{end}".encode()).hexdigest()
    path = cache_dir / f"{key}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError):
            pass  # Refetch an unreadable cache entry.

    try:
        columns = []
        for ticker in tickers:
            handle = yf.Ticker(ticker)
            history = handle.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
            if history.empty:
                raise PriceDataError(f"no price history returned for {ticker}")
            close = history["Close"].copy()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
            # A constant price-unit conversion leaves percentage returns unchanged.
            try:
                currency = handle.fast_info.get("currency")
            except Exception:
                currency = None
            if currency == "GBp":
                close /= 100
            columns.append(close.rename(ticker))
        frame: pd.DataFrame = pd.concat(columns, axis=1)
        frame = frame.sort_index().loc[str(start) : str(end)]
        if frame.empty:
            raise PriceDataError("no prices within the requested historical window")
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame
    except PriceDataError:
        raise
    except Exception as exc:
        raise PriceDataError(f"could not load market prices: {exc}") from exc
