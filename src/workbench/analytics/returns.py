"""Converting prices to returns, and returns to annualised figures. See docs/model.md
for why log vs. simple returns are each used where they are."""

from __future__ import annotations

import numpy as np
import pandas as pd

from workbench.constants import TRADING_DAYS_PER_YEAR

MIN_HISTORY_YEARS = 3.0
"""Below this many years of price history, CLAUDE.md requires an explicit warning."""


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(p_t / p_{t-1}). Drops the first (undefined) row."""
    ratios: pd.DataFrame = prices / prices.shift(1)
    return pd.DataFrame(np.log(ratios), index=ratios.index, columns=ratios.columns).dropna(
        how="all"
    )


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: p_t / p_{t-1} - 1. Drops the first (undefined) row."""
    return prices.pct_change().dropna(how="all")


def annualised_simple_return_from_log_mean(
    mean_daily_log_return: float, trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Converts a mean daily log return into an annualised simple (CAGR-style) return."""
    return float(np.exp(mean_daily_log_return * trading_days) - 1)


def annualised_vol_from_log_std(
    daily_log_std: float, trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    return float(daily_log_std * np.sqrt(trading_days))


def years_of_history(prices: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Non-missing observation count per column, expressed in years."""
    return prices.notna().sum() / trading_days


def tickers_with_short_history(
    prices: pd.DataFrame, min_years: float = MIN_HISTORY_YEARS
) -> list[str]:
    """Tickers below `min_years` of price history; detection only, caller decides what to do."""
    years = years_of_history(prices)
    return [str(ticker) for ticker, yrs in years.items() if yrs < min_years]
