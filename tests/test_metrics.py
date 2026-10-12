from datetime import date

import pytest

from workbench.analytics.metrics import (
    compute_portfolio_metrics,
    effective_n,
    herfindahl_index,
    weighted_expense_ratio,
)
from workbench.data.providers.synthetic import SyntheticProvider
from workbench.models.instrument import AssetClass, Instrument


def _instrument(ticker: str, ter: float | None) -> Instrument:
    return Instrument(
        ticker=ticker,
        name=ticker,
        asset_class=AssetClass.ETF,
        currency="GBP",
        region="Global",
        ter=ter,
    )


def test_herfindahl_and_effective_n_for_equal_weights() -> None:
    weights = {"A": 0.5, "B": 0.5}
    assert herfindahl_index(weights) == pytest.approx(0.5)
    assert effective_n(weights) == pytest.approx(2.0)


def test_herfindahl_and_effective_n_for_single_holding() -> None:
    weights = {"A": 1.0}
    assert herfindahl_index(weights) == pytest.approx(1.0)
    assert effective_n(weights) == pytest.approx(1.0)


def test_weighted_ter_averages_known_values() -> None:
    instruments = [_instrument("A", 0.001), _instrument("B", 0.003)]
    result = weighted_expense_ratio({"A": 0.5, "B": 0.5}, instruments)
    assert result == pytest.approx(0.002)


def test_weighted_ter_is_none_when_any_ter_unknown() -> None:
    instruments = [_instrument("A", 0.001), _instrument("B", None)]
    result = weighted_expense_ratio({"A": 0.5, "B": 0.5}, instruments)
    assert result is None


def test_compute_portfolio_metrics_returns_sane_values() -> None:
    provider = SyntheticProvider(
        seed=5, asset_classes={"A": AssetClass.STOCK, "B": AssetClass.BOND}
    )
    prices = provider.get_prices(["A", "B"], date(2015, 1, 1), date(2024, 12, 31))
    instruments = [_instrument("A", 0.0022), _instrument("B", 0.0015)]

    metrics = compute_portfolio_metrics(
        weights={"A": 0.6, "B": 0.4},
        instruments=instruments,
        prices=prices,
        risk_free_rate=0.04,
    )

    assert metrics.volatility > 0
    assert metrics.max_drawdown <= 0
    assert metrics.hhi == pytest.approx(0.6**2 + 0.4**2)
    assert metrics.effective_n == pytest.approx(1 / (0.6**2 + 0.4**2))
    assert metrics.weighted_ter == pytest.approx(0.6 * 0.0022 + 0.4 * 0.0015)


def test_concentrated_portfolio_has_lower_effective_n_than_diversified() -> None:
    provider = SyntheticProvider(
        seed=6, asset_classes={"A": AssetClass.STOCK, "B": AssetClass.BOND}
    )
    prices = provider.get_prices(["A", "B"], date(2015, 1, 1), date(2024, 12, 31))
    instruments = [_instrument("A", 0.001), _instrument("B", 0.001)]

    concentrated = compute_portfolio_metrics(
        {"A": 0.95, "B": 0.05}, instruments, prices, risk_free_rate=0.04
    )
    diversified = compute_portfolio_metrics(
        {"A": 0.5, "B": 0.5}, instruments, prices, risk_free_rate=0.04
    )

    assert concentrated.effective_n < diversified.effective_n
