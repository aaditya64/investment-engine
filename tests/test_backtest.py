from datetime import date

import numpy as np
import pandas as pd
import pytest

from workbench.analytics.backtest import fixed_weight_backtest


def _two_asset_prices() -> pd.DataFrame:
    dates = pd.bdate_range(date(2024, 1, 1), date(2024, 6, 28))
    rng = np.random.default_rng(0)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(dates))))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, len(dates))))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_rejects_weights_not_summing_to_one() -> None:
    prices = _two_asset_prices()
    with pytest.raises(ValueError, match="must sum to 1"):
        fixed_weight_backtest(prices, {"A": 0.5, "B": 0.4})


def test_rejects_unknown_ticker() -> None:
    prices = _two_asset_prices()
    with pytest.raises(KeyError):
        fixed_weight_backtest(prices, {"A": 0.5, "C": 0.5})


def test_single_asset_backtest_equals_its_own_returns() -> None:
    prices = _two_asset_prices()
    path = fixed_weight_backtest(prices[["A"]], {"A": 1.0})
    expected = prices["A"].pct_change().dropna()
    assert np.allclose(path.to_numpy(), expected.to_numpy())


def test_rebalancing_resets_weights_at_period_start() -> None:
    dates = pd.bdate_range(date(2024, 1, 1), date(2024, 1, 31))
    a = pd.Series([100.0, 110.0] + [110.0] * (len(dates) - 2), index=dates)
    b = pd.Series([100.0] * len(dates), index=dates)
    prices = pd.DataFrame({"A": a, "B": b})

    path = fixed_weight_backtest(prices, {"A": 0.5, "B": 0.5}, rebalance="M")

    first_day_return = path.iloc[0]
    assert first_day_return == pytest.approx(0.5 * 0.10)


def test_output_length_matches_available_return_observations() -> None:
    prices = _two_asset_prices()
    path = fixed_weight_backtest(prices, {"A": 0.6, "B": 0.4})
    assert len(path) == len(prices) - 1
