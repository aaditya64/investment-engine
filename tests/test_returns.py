from datetime import date

import numpy as np
import pandas as pd
import pytest

from workbench.analytics.returns import (
    annualised_simple_return_from_log_mean,
    annualised_vol_from_log_std,
    log_returns,
    simple_returns,
    tickers_with_short_history,
)
from workbench.data.providers.synthetic import SyntheticProvider


def _flat_growth_prices(growth_per_day: float, n: int = 100) -> pd.DataFrame:
    values = 100.0 * (1 + growth_per_day) ** np.arange(n)
    return pd.DataFrame({"X": values})


def test_simple_returns_match_known_growth_rate() -> None:
    prices = _flat_growth_prices(0.01)
    returns = simple_returns(prices)
    assert np.allclose(returns["X"], 0.01)


def test_log_returns_drop_first_row() -> None:
    prices = _flat_growth_prices(0.01, n=10)
    returns = log_returns(prices)
    assert len(returns) == len(prices) - 1


def test_annualised_return_round_trip() -> None:
    daily_log_return = np.log(1.01)
    annual = annualised_simple_return_from_log_mean(daily_log_return, trading_days=252)
    assert annual == pytest.approx(1.01**252 - 1)


def test_annualised_vol_scales_by_sqrt_trading_days() -> None:
    assert annualised_vol_from_log_std(0.01, trading_days=252) == pytest.approx(0.01 * np.sqrt(252))


def test_tickers_with_short_history_flags_young_instrument() -> None:
    provider = SyntheticProvider(seed=0)
    long_history = provider.get_prices(["OLD"], date(2015, 1, 1), date(2024, 12, 31))
    short_history = provider.get_prices(["NEW"], date(2024, 1, 1), date(2024, 12, 31))
    combined = long_history.join(short_history, how="outer")

    flagged = tickers_with_short_history(combined, min_years=3.0)
    assert flagged == ["NEW"]
