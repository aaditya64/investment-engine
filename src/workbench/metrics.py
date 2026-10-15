"""Historical estimates and portfolio metrics, with consistent annual units."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

from workbench.constants import MEAN_SHRINKAGE, TRADING_DAYS_PER_YEAR
from workbench.models import PortfolioMetrics

type FloatArray = NDArray[np.float64]


def estimate_returns(prices: pd.DataFrame, tickers: list[str]) -> tuple[FloatArray, FloatArray]:
    """Estimate annual returns and covariance from overlapping daily simple returns.

    Missing prices are never filled. A return requires both today's and the preceding
    row's prices for every selected asset. Means shrink 50% toward the cross-asset mean;
    Ledoit-Wolf regularises covariance. Both estimates use the same observations.
    """
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"missing price data for: {', '.join(sorted(missing))}")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("price history must have unique dates")
    aligned = prices.loc[:, tickers].sort_index().astype(float)
    values = aligned.to_numpy()
    if np.isinf(values).any() or (values <= 0).any():
        raise ValueError("prices must be positive and finite when present")
    returns = aligned.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 2:
        raise ValueError("need at least two overlapping daily returns for all selected assets")
    observations = returns.to_numpy(dtype=float)
    if not np.isfinite(observations).all():
        raise ValueError("daily returns must be finite")

    mean = observations.mean(axis=0)
    annual_mean = (1 - MEAN_SHRINKAGE) * mean + MEAN_SHRINKAGE * mean.mean()
    annual_mean *= TRADING_DAYS_PER_YEAR
    covariance = LedoitWolf(store_precision=False).fit(observations).covariance_
    return annual_mean, np.asarray(covariance * TRADING_DAYS_PER_YEAR, dtype=float)


def portfolio_metrics(
    weights: FloatArray, mean: FloatArray, covariance: FloatArray, risk_free_rate: float
) -> PortfolioMetrics:
    """Use the optimiser's annualised estimates, including the same risk-free rate."""
    variance = float(weights @ covariance @ weights)
    if not np.isfinite(variance) or variance <= 1e-20:
        raise ValueError("portfolio volatility is zero or undefined; Sharpe cannot be calculated")
    volatility = float(np.sqrt(variance))
    historical_return = float(weights @ mean)
    return PortfolioMetrics(
        historical_return=historical_return,
        volatility=volatility,
        sharpe=(historical_return - risk_free_rate) / volatility,
    )
