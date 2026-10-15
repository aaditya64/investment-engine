import os
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from workbench.data import CACHE_SECONDS, PriceDataError, get_prices


@pytest.fixture
def market(monkeypatch):
    calls = []
    history = pd.DataFrame(
        {"Close": [10000.0, 10100.0, 10200.0]},
        index=pd.date_range("2024-01-03", periods=3, tz="Europe/London"),
    )

    def ticker(symbol):
        def fetch(**kwargs):
            calls.append((symbol, kwargs))
            return history.copy()

        return SimpleNamespace(history=fetch, fast_info={"currency": "GBp"})

    monkeypatch.setattr("workbench.data.yf.Ticker", ticker)
    return calls


def test_normalises_pence_inclusive_end_and_caches_weekend_window(tmp_path, market) -> None:
    start, end = date(2024, 1, 1), date(2024, 1, 7)
    first = get_prices(["ISF.L"], start, end, tmp_path)
    second = get_prices(["ISF.L"], start, end, tmp_path)
    assert first["ISF.L"].tolist() == [100, 101, 102]
    assert first.index.tz is None
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert market == [("ISF.L", {"start": start, "end": date(2024, 1, 8), "auto_adjust": True})]


def test_cache_separates_tickers_and_dates(tmp_path, market) -> None:
    get_prices(["ISF.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)
    get_prices(["VWRP.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)
    get_prices(["ISF.L"], date(2024, 1, 1), date(2024, 1, 8), tmp_path)
    assert len(market) == 3


def test_expired_or_corrupt_cache_refetches(tmp_path, market) -> None:
    args = (["ISF.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)
    get_prices(*args)
    path = next(tmp_path.glob("*.parquet"))
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime - CACHE_SECONDS - 1))
    get_prices(*args)
    path.write_text("invalid cache content")
    get_prices(*args)
    assert len(market) == 3


def test_empty_market_history_raises_clear_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "workbench.data.yf.Ticker",
        lambda _: SimpleNamespace(history=lambda **_: pd.DataFrame()),
    )
    with pytest.raises(PriceDataError, match="no price history.*VWRP.L"):
        get_prices(["VWRP.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)


def test_market_exception_is_wrapped(monkeypatch, tmp_path) -> None:
    def unavailable(_):
        raise OSError("connection unavailable")

    monkeypatch.setattr("workbench.data.yf.Ticker", unavailable)
    with pytest.raises(PriceDataError, match="connection unavailable"):
        get_prices(["VWRP.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)
