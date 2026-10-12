"""Computes PortfolioMetrics for a fixed weight mix. See docs/model.md for which
figures come from the shrunk estimate vs. the realised historical path, and why."""

from __future__ import annotations

import numpy as np
import pandas as pd

from workbench.analytics.backtest import fixed_weight_backtest
from workbench.analytics.covariance import shrunk_annual_covariance
from workbench.analytics.expected_returns import DEFAULT_SHRINKAGE, shrink_to_grand_mean
from workbench.analytics.returns import log_returns
from workbench.constants import TRADING_DAYS_PER_YEAR
from workbench.models.instrument import Instrument
from workbench.models.portfolio import PortfolioMetrics


def herfindahl_index(weights: dict[str, float]) -> float:
    return float(sum(w**2 for w in weights.values()))


def effective_n(weights: dict[str, float]) -> float:
    return 1.0 / herfindahl_index(weights)


def weighted_expense_ratio(
    weights: dict[str, float], instruments: list[Instrument]
) -> float | None:
    """None (not 0) if any held instrument's TER is unknown."""
    ter_by_ticker = {i.ticker: i.ter for i in instruments}
    total = 0.0
    for ticker, weight in weights.items():
        ter = ter_by_ticker.get(ticker)
        if ter is None:
            return None
        total += weight * ter
    return total


def historical_max_drawdown(simple_return_path: pd.Series) -> float:
    """Largest peak-to-trough decline in wealth, as a negative decimal."""
    wealth = (1 + simple_return_path).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return float(drawdown.min())


def annualised_downside_deviation(
    simple_return_path: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    downside = simple_return_path[simple_return_path < 0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=0) * np.sqrt(trading_days))


def compute_portfolio_metrics(
    weights: dict[str, float],
    instruments: list[Instrument],
    prices: pd.DataFrame,
    risk_free_rate: float,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> PortfolioMetrics:
    tickers = list(weights)
    weight_vector = np.array([weights[t] for t in tickers])
    aligned_prices = prices[tickers].dropna()

    daily_log = log_returns(aligned_prices)[tickers]
    shrunk_daily_log_mean = shrink_to_grand_mean(daily_log.mean(), shrinkage)[tickers]
    annualised_simple_return = np.exp(shrunk_daily_log_mean.to_numpy() * TRADING_DAYS_PER_YEAR) - 1
    expected_return = float(weight_vector @ annualised_simple_return)

    annual_cov = shrunk_annual_covariance(daily_log).loc[tickers, tickers]
    variance = float(weight_vector @ annual_cov.to_numpy() @ weight_vector)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

    realised_path = fixed_weight_backtest(aligned_prices, weights)
    downside_dev = annualised_downside_deviation(realised_path)
    n_days = len(realised_path)
    growth_factor = float(np.prod(1 + realised_path.to_numpy()))
    realised_annual_return = growth_factor ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    sortino = (realised_annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    return PortfolioMetrics(
        expected_return=expected_return,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=historical_max_drawdown(realised_path),
        hhi=herfindahl_index(weights),
        effective_n=effective_n(weights),
        weighted_ter=weighted_expense_ratio(weights, instruments),
    )
