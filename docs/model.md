# Portfolio model

## Inputs and scope

Search global stocks and ETFs by ticker/name, selecting up to 20 assets. The
historical window is a lookback in years ending today, default five years. Per-asset
bounds default to 0–100%. Portfolios are long-only, fully invested, and contain only
selected assets; zero allocations are allowed. The annual risk-free rate defaults to
4%, an assumption that users can override. It is a Sharpe benchmark, not a cash holding.

## Historical estimates

Fetch daily adjusted closes through yfinance with `auto_adjust=True`, including the
requested end date. Verify chart metadata identifies equities or ETFs and provides a
quote currency. Normalise GBp/GBX prices to GBP. For another three-letter currency
`CUR`, multiply prices by historical `CURGBP=X` closes (GBP per unit of CUR) **before**
estimating returns, covariance or backtest performance. For example, a $100 share at
0.80 GBP/USD is valued at £80; its GBP return includes subsequent FX movements.

FX rates align to each asset's local daily date. A missing rate can use only a prior
quote within three calendar days, never a later quote. Asset prices are not filled.
The daily alignment approximates asynchronous stock/FX closing times; it is not an
intraday execution simulation. Reuse each currency's FX history within a request and
cache only the GBP-converted frame under a versioned key. Unknown/unsupported quote
currencies or wholly unavailable rates fail explicitly. Isolated absent/stale rates
leave missing GBP prices and use the existing missing-data handling. Both endpoints return
`base_currency: "GBP"`; the risk-free benchmark also uses this reporting basis.

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

## Out-of-sample backtest and equal-weight baseline

`POST /backtest` adds `train_fraction`, default 0.8. Sort selected price rows
chronologically and split at `floor(row_count * train_fraction)` **before estimating
returns or covariance**. At least three training prices and two later test prices are
required. Missing training observations follow the estimator's existing rules; the
purchase close must have complete prices for all selected assets. A purchase-price
error identifies the affected date and tickers.

Fit the existing optimiser only on the training rows. The last training close is the
assumed frictionless purchase price and the £1,000 chart anchor. Each measured return
ends on a strictly later date, including the move into the first test close. The
response reports actual training/test dates. This is one holdout evaluation; no test
prices are used to choose or update weights.

Keep purchased units fixed. For starting weights `w`, purchase prices `P[0]`, and test
prices `P[t]`, the portfolio value and daily realised return are:

```text
V[t] = 1000 * sum(w[i] * P[t, i] / P[0, i])
r[t] = V[t] / V[t-1] - 1  # only when both adjacent original rows are complete

annualised_return = mean(r) * 252
volatility = sample_std(r, ddof=1) * sqrt(252)
sharpe = (annualised_return - rf) / volatility
```

Weights drift as prices move; there is no daily rebalancing, and bounds apply to the
starting allocation. Both portfolios use the same dates, adjusted prices, starting
value, and user-supplied constant annual risk-free rate. Transaction costs and taxes
are not modelled. The annualised return is arithmetic, not CAGR; these are actual
held-out return-path metrics with no shrinkage. Effectively zero realised volatility
(`<= 1e-10`) produces an undefined (`null`) Sharpe, while its growth curve remains valid.

Missing test quotes do not invalidate the entire backtest. The chart uses only closes
where all selected assets have prices, without filling missing values. Daily metrics
exclude each incomplete row and the return into its following row, so a multi-day
move across a missing quote is never counted as a one-day return. Require at least two
remaining daily observations. `data_warnings` names affected assets and example dates;
test start/end reflect the available shared valuation dates. Trailing incomplete quotes
are omitted. The original chronological split remains fixed before this handling,
so future missingness cannot change the training allocation.

The baseline starts with `1 / asset_count` in each selected asset and also buys and
holds. Include it only if those starting weights respect every supplied bound within
numerical tolerance; otherwise return an explanation and no baseline metrics/curve.
Do not silently alter equal weights to fit the constraints.

Tests assert the exact split and purchase/test boundary, known fixed-unit growth and
metrics, purchase-price errors, safe gap handling, baseline eligibility, zero-volatility
handling, and the
invariant that changing held-out prices cannot change the fitted allocation.
