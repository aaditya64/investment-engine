"""Long-only Sharpe maximisation. No networking, caching, or preference mappings."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from workbench.metrics import FloatArray, estimate_returns, portfolio_metrics
from workbench.models import PortfolioRequest, PortfolioResult

WEIGHT_TOLERANCE = 1e-7


class OptimisationError(ValueError):
    """The numerical optimiser could not produce a usable allocation."""


def optimise_portfolio(request: PortfolioRequest, prices: pd.DataFrame) -> PortfolioResult:
    """Maximise Sharpe over the supplied historical window and requested weight bounds.

    SLSQP is a numerical solve, not a guarantee of a global maximum. Deterministic
    starts include a spread allocation and allocations favouring each selected asset.
    Only successful, constraint-respecting solves can become the result.
    """
    mean, covariance = estimate_returns(prices, request.tickers)
    bounds = [request.weight_bounds.get(ticker, (0.0, 1.0)) for ticker in request.tickers]
    lower, upper = np.asarray(bounds, dtype=float).T
    capacity = upper - lower
    excess = mean - request.risk_free_rate

    def objective(weights: FloatArray) -> float:
        variance = max(float(weights @ covariance @ weights), 1e-20)
        return -float(weights @ excess) / float(np.sqrt(variance))

    def gradient(weights: FloatArray) -> FloatArray:
        variance = max(float(weights @ covariance @ weights), 1e-20)
        return np.asarray(
            -excess / np.sqrt(variance)
            + (weights @ excess) * (covariance @ weights) / variance**1.5,
            dtype=float,
        )

    def valid(weights: FloatArray) -> bool:
        return bool(
            np.isfinite(weights).all()
            and abs(weights.sum() - 1) <= WEIGHT_TOLERANCE
            and (weights >= lower - WEIGHT_TOLERANCE).all()
            and (weights <= upper + WEIGHT_TOLERANCE).all()
        )

    # When the bounds force a single allocation, there is nothing to optimise.
    if abs(lower.sum() - 1) <= 1e-10:
        weights = lower.copy()
    elif abs(upper.sum() - 1) <= 1e-10:
        weights = upper.copy()
    else:
        spread = lower + (1 - lower.sum()) * capacity / capacity.sum()
        starts = [spread]
        for preferred in range(len(bounds)):
            start = lower.copy()
            remaining = 1 - start.sum()
            order = [preferred, *(i for i in range(len(bounds)) if i != preferred)]
            for i in order:
                addition = min(remaining, capacity[i])
                start[i] += addition
                remaining -= addition
            starts.append(start)

        candidates = []
        message = "no successful solve"
        for start in starts:
            solved = minimize(
                objective,
                start,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "eq",
                    "fun": lambda w: float(w.sum() - 1),
                    "jac": lambda w: np.ones_like(w),
                },
                options={"maxiter": 500, "ftol": 1e-10},
            )
            message = str(solved.message)
            if solved.success and valid(solved.x):
                candidates.append(solved.x)
        if not candidates:
            raise OptimisationError(f"could not optimise portfolio: {message}")
        weights = min(candidates, key=objective)

    if not valid(weights):
        raise OptimisationError("optimiser returned weights outside the requested constraints")
    # Clip only numerical bound violations; do not renormalise and move other bounds.
    weights = np.clip(weights, lower, upper)
    if not valid(weights):
        raise OptimisationError("optimiser returned weights that do not sum to 100%")
    return PortfolioResult(
        weights=dict(zip(request.tickers, map(float, weights), strict=True)),
        metrics=portfolio_metrics(weights, mean, covariance, request.risk_free_rate),
        risk_free_rate=request.risk_free_rate,
    )
