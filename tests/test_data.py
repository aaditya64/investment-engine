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

        return SimpleNamespace(
            history=fetch, get_history_metadata=lambda: {"currency": "GBp", "instrumentType": "ETF"}
        )

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


@pytest.mark.parametrize("currency", ["unknown", "GBx", None])
def test_unknown_currency_is_rejected(monkeypatch, tmp_path, currency) -> None:
    history = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-03", periods=3)
    )
    monkeypatch.setattr(
        "workbench.data.yf.Ticker",
        lambda _: SimpleNamespace(
            history=lambda **_: history,
            get_history_metadata=lambda: {"currency": currency, "instrumentType": "ETF"},
        ),
    )
    with pytest.raises(PriceDataError, match="unknown or unsupported quote currency"):
        get_prices(["VWRP.L"], date(2024, 1, 1), date(2024, 1, 7), tmp_path)
    assert not list(tmp_path.glob("*.parquet"))


def test_us_stocks_convert_to_gbp_and_share_one_fx_download(monkeypatch, tmp_path) -> None:
    index = pd.date_range("2024-01-03", periods=3)
    calls = []

    def ticker(symbol):
        def history(**kwargs):
            calls.append(symbol)
            values = [0.8, 0.75, 0.7] if symbol == "USDGBP=X" else [100, 110, 120]
            return pd.DataFrame({"Close": values}, index=index)

        return SimpleNamespace(
            history=history,
            get_history_metadata=lambda: {"currency": "USD", "instrumentType": "EQUITY"},
        )

    monkeypatch.setattr("workbench.data.yf.Ticker", ticker)
    prices = get_prices(["NVDA", "META"], index[0].date(), index[-1].date(), tmp_path)
    assert prices["NVDA"].tolist() == pytest.approx([80, 82.5, 84])
    assert prices["META"].tolist() == pytest.approx([80, 82.5, 84])
    assert calls.count("USDGBP=X") == 1


@pytest.mark.parametrize(
    "rate_date, succeeds", [("2024-01-05", True), ("2024-01-09", False), ("2024-01-01", False)]
)
def test_fx_alignment_uses_only_recent_prior_rates(
    monkeypatch, tmp_path, rate_date, succeeds
) -> None:
    def ticker(symbol):
        is_fx = symbol == "USDGBP=X"
        frame = pd.DataFrame(
            {"Close": [0.8] if is_fx else [100, 110]},
            index=pd.to_datetime([rate_date] if is_fx else ["2024-01-08", "2024-01-09"]),
        )
        return SimpleNamespace(
            history=lambda **_: frame,
            get_history_metadata=lambda: {"currency": "USD", "instrumentType": "EQUITY"},
        )

    monkeypatch.setattr("workbench.data.yf.Ticker", ticker)
    if succeeds:
        # Friday's rate can value Monday, but is too stale by Tuesday.
        prices = get_prices(["NVDA"], date(2024, 1, 8), date(2024, 1, 8), tmp_path)
        assert prices["NVDA"].tolist() == [80]
    elif rate_date == "2024-01-09":
        prices = get_prices(["NVDA"], date(2024, 1, 8), date(2024, 1, 9), tmp_path)
        assert pd.isna(prices["NVDA"].iloc[0])
        assert prices["NVDA"].iloc[1] == 88
    else:
        with pytest.raises(PriceDataError, match="missing recent USD/GBP"):
            get_prices(["NVDA"], date(2024, 1, 8), date(2024, 1, 9), tmp_path)


def test_direct_requests_for_non_stock_instruments_are_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "workbench.data.yf.Ticker",
        lambda _: SimpleNamespace(
            history=lambda **_: pd.DataFrame(
                {"Close": [100]}, index=pd.to_datetime(["2024-01-08"])
            ),
            get_history_metadata=lambda: {"currency": "USD", "instrumentType": "CRYPTOCURRENCY"},
        ),
    )
    with pytest.raises(PriceDataError, match="not a supported stock or ETF"):
        get_prices(["BTC-USD"], date(2024, 1, 8), date(2024, 1, 8), tmp_path)
