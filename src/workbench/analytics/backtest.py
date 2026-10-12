"""Replays a fixed target weight mix over real price history, rebalancing to target on
a fixed schedule. No transaction costs modelled. See docs/model.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_REBALANCE_FREQ = "M"
"""Pandas Period alias; "M" = calendar month. Not yet exposed as a CLI/API option."""


def fixed_weight_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = DEFAULT_REBALANCE_FREQ,
) -> pd.Series:
    """Daily simple-return path, weights reset to target at the start of each period."""
    tickers = list(weights)
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise KeyError(f"no price data for: {sorted(missing)}")

    weight_vector = np.array([weights[t] for t in tickers])
    if not np.isclose(weight_vector.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {weight_vector.sum()}")

    aligned = prices[tickers].dropna()
    if len(aligned) < 2:
        raise ValueError("need at least two overlapping price observations to backtest")

    daily_returns = aligned.pct_change().dropna()
    period_keys = daily_returns.index.to_series().dt.to_period(rebalance)

    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    for _period, day_index in daily_returns.groupby(period_keys).groups.items():
        current_weights = weight_vector.copy()
        for day in day_index:
            day_returns = daily_returns.loc[day].to_numpy()
            portfolio_returns.loc[day] = float(current_weights @ day_returns)
            grown = current_weights * (1 + day_returns)
            current_weights = grown / grown.sum()

    return portfolio_returns
