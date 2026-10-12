from datetime import date

import pandas as pd

from workbench.data.cache import PriceCache


def _series(start: date, end: date, value: float = 1.0) -> pd.Series:
    index = pd.bdate_range(start, end)
    return pd.Series([value] * len(index), index=index)


def test_fetches_and_caches_on_first_call(tmp_path) -> None:
    cache = PriceCache(cache_dir=tmp_path)
    calls = []

    def fetch(ticker: str, start: date, end: date) -> pd.Series:
        calls.append((ticker, start, end))
        return _series(start, end)

    result = cache.get_or_fetch("X", date(2024, 1, 1), date(2024, 1, 31), fetch)
    assert len(calls) == 1
    assert not result.empty


def test_second_call_within_cached_range_does_not_refetch(tmp_path) -> None:
    cache = PriceCache(cache_dir=tmp_path)
    calls = []

    def fetch(ticker: str, start: date, end: date) -> pd.Series:
        calls.append((ticker, start, end))
        return _series(start, end)

    cache.get_or_fetch("X", date(2024, 1, 1), date(2024, 1, 31), fetch)
    cache.get_or_fetch("X", date(2024, 1, 5), date(2024, 1, 20), fetch)

    assert len(calls) == 1


def test_request_outside_cached_range_triggers_refetch(tmp_path) -> None:
    cache = PriceCache(cache_dir=tmp_path)
    calls = []

    def fetch(ticker: str, start: date, end: date) -> pd.Series:
        calls.append((ticker, start, end))
        return _series(start, end)

    cache.get_or_fetch("X", date(2024, 1, 1), date(2024, 1, 31), fetch)
    cache.get_or_fetch("X", date(2024, 2, 1), date(2024, 2, 28), fetch)

    assert len(calls) == 2


def test_different_tickers_cached_independently(tmp_path) -> None:
    cache = PriceCache(cache_dir=tmp_path)
    calls = []

    def fetch(ticker: str, start: date, end: date) -> pd.Series:
        calls.append(ticker)
        return _series(start, end)

    cache.get_or_fetch("X", date(2024, 1, 1), date(2024, 1, 31), fetch)
    cache.get_or_fetch("Y", date(2024, 1, 1), date(2024, 1, 31), fetch)

    assert calls == ["X", "Y"]
