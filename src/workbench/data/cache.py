"""The project's one disk-cache refresh path for price data, generic over the fetch
function so it can be unit-tested without a real network call."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(".cache")


class PriceCache:
    """Per-ticker on-disk parquet cache; a hit requires the cached range to fully cover
    [start, end], otherwise the whole range is re-fetched and the entry replaced."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _load(self, ticker: str) -> pd.Series | None:
        path = self._path(ticker)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        series = frame.iloc[:, 0]
        series.name = ticker
        return series

    def _save(self, ticker: str, series: pd.Series) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        series.to_frame(name=ticker).to_parquet(self._path(ticker))

    def get_or_fetch(
        self,
        ticker: str,
        start: date,
        end: date,
        fetch: Callable[[str, date, date], pd.Series],
    ) -> pd.Series:
        cached = self._load(ticker)
        if cached is not None and not cached.empty:
            cached_start = cached.index.min().date()
            cached_end = cached.index.max().date()
            if cached_start <= start and cached_end >= end:
                return cached.loc[str(start) : str(end)]

        fresh = fetch(ticker, start, end)
        self._save(ticker, fresh)
        return fresh
