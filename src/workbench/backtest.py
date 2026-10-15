"""One chronological holdout: train once, then buy and hold alongside equal weights."""

import numpy as np
import pandas as pd

from workbench.constants import TRADING_DAYS_PER_YEAR
from workbench.metrics import FloatArray, price_history
from workbench.models import (
    BacktestMetrics,
    BacktestPortfolio,
    BacktestRequest,
    BacktestResult,
    GrowthPoint,
    PortfolioRequest,
)
from workbench.optimiser import WEIGHT_TOLERANCE, optimise_portfolio

STARTING_VALUE = 1000.0


def _buy_and_hold(
    prices: pd.DataFrame, weights: dict[str, float], risk_free_rate: float
) -> tuple[BacktestPortfolio, FloatArray]:
    """Prices include the purchase close and every subsequent test close.

    Units stay fixed, so weights drift rather than being rebalanced every day. Test
    metrics use actual portfolio returns with no shrinkage or covariance estimation.
    Missing quotes leave a gap; neither chart values nor daily returns are filled.
    """
    vector = np.array([weights[ticker] for ticker in prices.columns])
    growth = (prices.to_numpy() / prices.iloc[0].to_numpy()) @ vector
    returns = pd.Series(growth, index=prices.index).pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError(
            "backtest needs at least two daily returns between consecutive complete price rows; "
            "try a longer historical window or fewer assets"
        )
    annualised_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    metrics = BacktestMetrics(
        annualised_return=annualised_return,
        volatility=volatility,
        sharpe=(annualised_return - risk_free_rate) / volatility if volatility > 1e-10 else None,
    )
    return (
        BacktestPortfolio(weights=weights, metrics=metrics),
        growth[np.isfinite(growth)] * STARTING_VALUE,
    )


def backtest_portfolio(request: BacktestRequest, prices: pd.DataFrame) -> BacktestResult:
    """Split ordered price rows before fitting any estimates; never optimise on test data.

    The last training close is the purchase price and the £1,000 chart anchor. Every
    measured return ends strictly after that close. Chart only complete shared closes;
    daily metrics omit intervals spanning missing quotes rather than treating them as
    one-day returns. The split is fixed before inspecting missing test observations.
    """
    history = price_history(prices, request.tickers)
    split = int(len(history) * request.train_fraction)
    if split < 3 or len(history) - split < 2:
        raise ValueError("backtest needs at least three training prices and two test prices")
    training = history.iloc[:split]
    test = history.iloc[split - 1 :]
    purchase_missing = test.columns[test.iloc[0].isna()].tolist()
    if purchase_missing:
        raise ValueError(
            f"missing purchase prices on {test.index[0].date()} for {', '.join(purchase_missing)}; "
            "choose a different training percentage or historical window"
        )
    observed = test.dropna(how="any")
    missing = test.loc[test.isna().any(axis=1)]
    warnings = []
    if not missing.empty:
        dates = ", ".join(str(day.date()) for day in missing.index[:3])
        if len(missing) > 3:
            dates += ", …"
        tickers = ", ".join(missing.columns[missing.isna().any()].tolist())
        warnings.append(
            f"Missing quotes for {tickers} on {len(missing)} date(s) ({dates}). "
            "The growth chart skips these dates; daily metrics omit returns spanning those gaps."
        )

    # Pass an ordinary optimisation request; only training prices reach the estimator.
    fit_request = request.model_dump(exclude={"train_fraction"})
    allocation = optimise_portfolio(PortfolioRequest.model_validate(fit_request), training)
    optimised, optimised_values = _buy_and_hold(test, allocation.weights, request.risk_free_rate)

    equal_weights = dict.fromkeys(request.tickers, 1 / len(request.tickers))
    incompatible = [
        ticker
        for ticker, weight in equal_weights.items()
        if not (
            request.weight_bounds.get(ticker, (0, 1))[0] - WEIGHT_TOLERANCE
            <= weight
            <= request.weight_bounds.get(ticker, (0, 1))[1] + WEIGHT_TOLERANCE
        )
    ]
    baseline = None
    baseline_values = None
    reason = None
    if incompatible:
        reason = "Equal weights violate the starting weight bounds for: " + ", ".join(incompatible)
    else:
        baseline, baseline_values = _buy_and_hold(test, equal_weights, request.risk_free_rate)

    return BacktestResult(
        training_start=training.index[0].date(),
        training_end=training.index[-1].date(),
        test_start=observed.index[1].date(),
        test_end=observed.index[-1].date(),
        risk_free_rate=request.risk_free_rate,
        starting_value=STARTING_VALUE,
        optimised=optimised,
        equal_weight=baseline,
        baseline_unavailable_reason=reason,
        data_warnings=warnings,
        growth=[
            GrowthPoint(
                date=day.date(),
                optimised=float(optimised_values[i]),
                equal_weight=float(baseline_values[i]) if baseline_values is not None else None,
            )
            for i, day in enumerate(observed.index)
        ],
    )
