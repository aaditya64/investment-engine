"""Global instrument lookup and adjusted prices converted to a GBP reporting basis."""

import hashlib
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from workbench.models import TICKER_PATTERN, Instrument

DEFAULT_CACHE_DIR = Path(".cache/prices")
CACHE_SECONDS = 24 * 60 * 60


class PriceDataError(RuntimeError):
    """Market data or currency conversion could not be fetched or cached."""


def search_assets(query: str) -> list[Instrument]:
    """Search Yahoo by ticker/name, retaining equities and ETFs across exchanges."""
    query = " ".join(query.split())
    if not query:
        return []
    try:
        quotes = yf.Search(
            query,
            max_results=15,
            news_count=0,
            lists_count=0,
            recommended=0,
            include_cb=False,
            timeout=10,
        ).quotes
        matches = {}
        for quote in quotes:
            ticker = str(quote.get("symbol", "")).upper()
            kind = quote.get("quoteType")
            if kind not in ("EQUITY", "ETF") or not re.fullmatch(TICKER_PATTERN, ticker):
                continue
            if len(ticker) > 32:
                continue
            matches[ticker] = Instrument(
                ticker=ticker,
                name=quote.get("longname") or quote.get("shortname") or ticker,
                asset_class="stock" if kind == "EQUITY" else "etf",
                exchange=quote.get("exchDisp") or quote.get("exchange") or "Unknown exchange",
            )
        return sorted(
            matches.values(),
            key=lambda asset: (
                asset.ticker.casefold() != query.casefold(),
                not asset.ticker.casefold().startswith(query.casefold()),
            ),
        )[:10]
    except Exception as exc:
        raise PriceDataError(f"asset search unavailable: {exc}") from exc


def _history(ticker: str, start: date, end: date) -> tuple[pd.Series, dict[str, object]]:
    """Adjusted daily closes in the listing's own currency, plus chart metadata."""
    handle = yf.Ticker(ticker)
    history = handle.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if history.empty:
        raise PriceDataError(f"no price history returned for {ticker}")
    close: pd.Series = history["Close"].copy().astype(float)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    metadata: dict[str, object] = dict(handle.get_history_metadata())
    return close.sort_index().loc[str(start) : str(end)], metadata


def get_prices(
    tickers: list[str], start: date, end: date, cache_dir: Path = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """Get GBP-adjusted prices, including end, cached per request for 24 hours.

    Foreign prices use historical CURGBP=X rates (GBP per currency unit). Missing
    FX dates may use a prior quote up to three calendar days old; future rates and
    backfilling are never used. Asset prices themselves are never filled.
    """
    key = hashlib.sha256(f"GBP-global-v1:{','.join(tickers)}:{start}:{end}".encode()).hexdigest()
    path = cache_dir / f"{key}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError):
            pass
    try:
        columns = []
        exchange_rates: dict[str, pd.Series] = {}
        for ticker in tickers:
            close, metadata = _history(ticker, start, end)
            if metadata.get("instrumentType") not in ("EQUITY", "ETF"):
                raise PriceDataError(f"{ticker} is not a supported stock or ETF")
            currency = metadata.get("currency")
            if currency in ("GBp", "GBX"):
                close /= 100
                currency = "GBP"
            if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
                raise PriceDataError(
                    f"unknown or unsupported quote currency for {ticker}: {currency!r}"
                )
            if currency != "GBP":
                if currency not in exchange_rates:
                    rates, _ = _history(f"{currency}GBP=X", start - timedelta(days=7), end)
                    rates = rates.dropna()
                    if rates.empty or not np.isfinite(rates.to_numpy()).all() or (rates <= 0).any():
                        raise PriceDataError(f"no valid {currency}/GBP exchange rates")
                    exchange_rates[currency] = rates
                rates = exchange_rates[currency].reindex(
                    close.index,
                    method="ffill",
                    tolerance=pd.Timedelta(days=3),
                )
                if rates[close.notna()].isna().all():
                    raise PriceDataError(
                        f"missing recent {currency}/GBP exchange rates for {ticker}"
                    )
                # Isolated FX gaps stay missing, so existing return/backtest handling
                # can omit them without inventing prices or rejecting a long window.
                close = close * rates
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
