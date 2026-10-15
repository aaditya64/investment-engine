# Portfolio model

## Inputs and scope

Select from the eight existing instruments in `workbench.models.UNIVERSE`. The
historical window is a lookback in years ending today, default five years. Per-asset
bounds default to 0–100%. Portfolios are long-only, fully invested, and contain only
selected assets; zero allocations are allowed. The annual risk-free rate defaults to
4%, an assumption that users can override. It is a Sharpe benchmark, not a cash holding.

## Historical estimates

Fetch daily adjusted closes through yfinance with `auto_adjust=True`, including the
requested end date. Normalise identified GBp quotes to GBP. Keep the current instrument
universe; this model does not convert arbitrary foreign-currency assets.

Sort prices by date and require unique dates. For each asset calculate simple returns:

```text
r[t, i] = price[t, i] / price[t-1, i] - 1
```

Prices must be positive and finite when present. Never fill missing prices. Retain only
rows with a valid one-step return for every selected asset, requiring at least two such
observations. This avoids fabricating returns or calculating across an interior missing
price. A short or unrepresentative sample can still produce unreliable estimates.

Use 252 trading days per year. Retain the existing 50% mean shrinkage, applied to simple
returns consistently instead of the old mixture of log-return estimates and portfolio
simple-return metrics:

```text
raw_mean[i] = mean(r[:, i])
mu[i] = 252 * (0.5 * raw_mean[i] + 0.5 * mean(raw_mean))
Sigma = 252 * LedoitWolf(r).covariance_
```

Mean shrinkage pulls estimates toward the selected assets' common mean. Ledoit-Wolf
regularises covariance toward a scaled identity matrix. Both are fixed modelling
choices, not user preference sliders. See the [scikit-learn estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

`mu` is an annualised arithmetic historical estimate, not CAGR or a forecast.
Annualisation assumes the usual independent-daily-return scaling; it does not model
serial correlation. Shrinkage reduces some estimation noise but does not eliminate it.

## Objective and reported metrics

For weights `w` and the supplied annual risk-free rate `rf`:

```text
historical_return = w · mu
volatility = sqrt(wᵀ Sigma w)
sharpe = (historical_return - rf) / volatility

maximise sharpe
subject to sum(w) = 1 and lower[i] <= w[i] <= upper[i]
```

The objective and all reported metrics use the exact same estimates over all selected
assets, including those that end up at zero weight. Never re-estimate over only the
held assets. The result returns every selected ticker's weight and the rate actually
used. Zero or undefined volatility is an error, not an invented Sharpe ratio.

## Numerical solve

SciPy's existing SLSQP solver minimises negative Sharpe with an analytic gradient,
500 iterations maximum, and objective tolerance `1e-10`. Direct Sharpe is not the old
convex cvxpy penalty objective. See the [SciPy solver documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

Start deterministically from a feasible spread allocation and allocations favouring
each selected asset. Choose the best successful solve that satisfies all constraints.
Multiple starts help with negative excess returns and starting-point sensitivity;
they do not prove global optimality. If bounds uniquely determine the allocation,
calculate its metrics directly. Validate total weight and bounds within `1e-7`, clip
only numerical bound violations, and recheck. Do not renormalise weights in a way that
can break the requested bounds. Never substitute an equal-weight/heuristic fallback.

## Data cache and errors

One Parquet cache entry per ticker selection and requested date range, expiring after
24 hours. Cache coverage uses the requested dates, so weekends and market holidays do
not force repeated downloads. Unreadable entries are fetched again. No exposure,
synthetic-provider, or fixture-provider framework is needed; tests use DataFrames and
mock market fetching.

Validate selection, duplicate tickers, finite inputs, and feasible bounds before
fetching. Require each minimum <= maximum, total minima <= 100%, and total maxima >=
100%. HTTP validation/calculation errors return 422; market-data availability errors
return 502. Solver failure produces a clear error, never a plausible-looking allocation.
