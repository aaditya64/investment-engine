from types import SimpleNamespace

import pytest

from workbench.data import PriceDataError, search_assets


def test_global_search_keeps_stocks_and_etfs_and_ranks_exact_ticker_first(monkeypatch) -> None:
    quotes = [
        {
            "symbol": "META.MX",
            "longname": "Meta Platforms",
            "quoteType": "EQUITY",
            "exchDisp": "Mexico",
        },
        {
            "symbol": "META",
            "longname": "Meta Platforms, Inc.",
            "quoteType": "EQUITY",
            "exchDisp": "NASDAQ",
        },
        {
            "symbol": "NVDA",
            "longname": "NVIDIA Corporation",
            "quoteType": "EQUITY",
            "exchDisp": "NASDAQ",
        },
        {
            "symbol": "EQGB.L",
            "longname": "Invesco EQQQ NASDAQ-100 GBP Hedged",
            "quoteType": "ETF",
            "exchange": "LSE",
        },
        {
            "symbol": "META",
            "longname": "Meta Platforms, Inc.",
            "quoteType": "EQUITY",
            "exchDisp": "NASDAQ",
        },
        {"symbol": "BTC-USD", "quoteType": "CRYPTOCURRENCY"},
        {"symbol": "NVDA=F", "quoteType": "FUTURE"},
        {"symbol": "0P0001", "quoteType": "MUTUALFUND"},
        {"symbol": "bad/ticker", "quoteType": "EQUITY"},
    ]

    def lookup(query, **kwargs):
        assert query == "meta"
        assert kwargs["news_count"] == 0
        return SimpleNamespace(quotes=quotes)

    monkeypatch.setattr("workbench.data.yf.Search", lookup)
    result = search_assets("  meta  ")
    assert result[0].ticker == "META"
    assert {asset.ticker for asset in result} == {"META", "META.MX", "NVDA", "EQGB.L"}
    assert result[0].exchange == "NASDAQ"
    assert result[-1].asset_class == "etf"


def test_search_handles_empty_queries_and_no_matches(monkeypatch) -> None:
    monkeypatch.setattr("workbench.data.yf.Search", lambda *_, **__: SimpleNamespace(quotes=[]))
    assert search_assets(" ") == []
    assert search_assets("nonexistent") == []


def test_search_provider_errors_are_not_disguised_as_no_matches(monkeypatch) -> None:
    def unavailable(*_, **__):
        raise OSError("connection unavailable")

    monkeypatch.setattr("workbench.data.yf.Search", unavailable)
    with pytest.raises(PriceDataError, match="asset search unavailable"):
        search_assets("Nvidia")
