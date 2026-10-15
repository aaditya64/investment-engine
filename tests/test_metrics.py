import numpy as np
import pandas as pd
import pytest

from workbench.metrics import estimate_returns, portfolio_metrics


def test_known_portfolio_metrics() -> None:
    metrics = portfolio_metrics(
        np.array([0.6, 0.4]), np.array([0.10, 0.05]), np.diag([0.04, 0.01]), 0.02
    )
    assert metrics.historical_return == pytest.approx(0.08)
    assert metrics.volatility == pytest.approx(np.sqrt(0.016))
    assert metrics.sharpe == pytest.approx(0.06 / np.sqrt(0.016))


def test_single_asset_estimates_use_simple_returns() -> None:
    prices = pd.DataFrame(
        {"A": [100, 110, 99, 118.8]}, index=pd.date_range("2024-01-01", periods=4)
    )
    mean, covariance = estimate_returns(prices, ["A"])
    # Daily returns are +10%, -10%, +20%. Covariance uses population normalisation.
    assert mean[0] == pytest.approx((0.1 - 0.1 + 0.2) / 3 * 252)
    assert covariance[0, 0] == pytest.approx(np.var([0.1, -0.1, 0.2]) * 252)


def test_missing_prices_are_not_filled_or_bridged() -> None:
    prices = pd.DataFrame(
        {"A": [100, 110, np.nan, 100, 120], "B": [100, 100, 200, 200, 200]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    mean, _ = estimate_returns(prices, ["A", "B"])
    # Only days 2 and 5 have valid one-step returns for both assets.
    # A's mean is 15%, B's is 0%; each shrinks halfway toward 7.5%.
    assert mean == pytest.approx(np.array([0.1125, 0.0375]) * 252)


def test_estimates_keep_ticker_order_and_covariance_is_positive_semidefinite(prices) -> None:
    tickers = ["VAGP.L", "VWRP.L", "ISF.L"]
    mean, covariance = estimate_returns(prices, tickers)
    reversed_mean, reversed_covariance = estimate_returns(prices, tickers[::-1])
    assert mean == pytest.approx(reversed_mean[::-1])
    assert covariance == pytest.approx(reversed_covariance[::-1, ::-1])
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


@pytest.mark.parametrize("bad_price", [0, -1, np.inf, -np.inf])
def test_rejects_invalid_prices(prices, bad_price) -> None:
    prices.iloc[0, 0] = bad_price
    with pytest.raises(ValueError, match="positive and finite"):
        estimate_returns(prices, ["VWRP.L"])


def test_rejects_missing_columns_and_insufficient_overlap(prices) -> None:
    with pytest.raises(ValueError, match="missing price data.*UNKNOWN"):
        estimate_returns(prices, ["UNKNOWN"])
    with pytest.raises(ValueError, match="two overlapping"):
        estimate_returns(prices.iloc[:2], ["VWRP.L"])
    prices["VAGP.L"] = np.nan
    with pytest.raises(ValueError, match="two overlapping"):
        estimate_returns(prices, ["VWRP.L", "VAGP.L"])


def test_rejects_duplicate_dates(prices) -> None:
    with pytest.raises(ValueError, match="unique dates"):
        estimate_returns(pd.concat([prices, prices.iloc[:1]]), ["VWRP.L"])


def test_zero_volatility_has_no_sharpe() -> None:
    with pytest.raises(ValueError, match="volatility is zero"):
        portfolio_metrics(np.array([1.0]), np.array([0.04]), np.zeros((1, 1)), 0.04)
