# Model documentation

The maths behind `src/workbench`, kept in sync with the code (CLAUDE.md). Milestone 1
(foundations) only — preference→parameter mappings arrive with the Milestone 2 engine
and will be documented here as they're built.

## Constants

- `TRADING_DAYS_PER_YEAR = 252` (`workbench/constants.py`) — the one annualisation
  factor used everywhere; nothing re-derives it locally.
- `DEFAULT_RISK_FREE_RATE = 0.04` — an assumption, not a fact, always surfaced via
  `AssumptionsPanel`. Override with `--risk-free-rate`.

## Returns: log vs. simple

Both are computed (`analytics/returns.py`), each for a reason:

- **Log returns** (`ln(p_t / p_{t-1})`) are used internally for covariance/shrinkage,
  because they are additive over *time* for a single asset.
- **Simple returns** (`p_t / p_{t-1} - 1`) are used wherever returns are combined
  *across* assets into a portfolio return (backtest, Sortino, drawdown), because a
  portfolio's simple return is exactly the weight-weighted sum of its holdings' simple
  returns — which is not true for log returns.

Converting an estimate from one world to the other: an annualised simple return from a
mean daily log return is `exp(mean_daily_log_return * 252) - 1` (see
`annualised_simple_return_from_log_mean`).

## Expected returns: shrinkage

Raw historical mean returns are very noisy over the sample sizes available (a few years
of daily data). `analytics/expected_returns.shrink_to_grand_mean` pulls each
instrument's raw mean toward the cross-sectional grand mean across the selected
instruments:

```
shrunk_i = (1 - shrinkage) * raw_i + shrinkage * mean(raw)
```

**Judgement call:** V1 shrinks toward the grand mean (`DEFAULT_SHRINKAGE = 0.5`), not a
market-equilibrium (CAPM) prior. Equilibrium-based shrinkage (Black-Litterman) is
deferred to roadmap item 6 by deliberate choice, not oversight — conviction-as-a-view
is a natural fit for that later model.

## Covariance: Ledoit-Wolf shrinkage

`analytics/covariance.shrunk_annual_covariance` fits `sklearn.covariance.LedoitWolf` on
daily log returns and scales the result by `TRADING_DAYS_PER_YEAR` to annualise (i.e.
assumes i.i.d. daily returns — no attempt to model autocorrelation). Raw sample
covariance is not used anywhere.

## Portfolio metrics (`analytics/metrics.compute_portfolio_metrics`)

**Judgement call, documented:** this function mixes a forward-looking estimate with a
historical realisation, split by what each quantity can honestly represent:

| Metric | Basis | Why |
|---|---|---|
| `expected_return` | shrunk per-asset means, weight-dot | the whole point of shrinking is to use it, not the noisy raw history |
| `volatility` | `sqrt(wᵗ Σ w)`, Σ = shrunk annual covariance | consistent pairing with `expected_return`; same estimate family the Milestone 2 optimiser will use |
| `sharpe` | `(expected_return - rf) / volatility` | both inputs are the shrunk estimate |
| `sortino` | realised historical weighted-return path (periodic-rebalance backtest), downside deviation only | downside deviation is path-dependent; a covariance matrix has no notion of it |
| `max_drawdown` | same realised historical path | explicitly required to be a historical output, not an optimised-for quantity — max drawdown isn't convex (see CLAUDE.md engine-approach notes) |
| `hhi`, `effective_n` | weights only (`Σw_i²`, `1/hhi`) | no data dependency |
| `weighted_ter` | weight-dot of `Instrument.ter` | **`None`** (not 0) if any held instrument's TER is unknown — an unknown input must not silently become a known-looking number |

## Backtest (`analytics/backtest.fixed_weight_backtest`)

Replays a fixed target mix over real price history. **Judgement call:** rebalances back
to the target weights at the start of each calendar month (`DEFAULT_REBALANCE_FREQ =
"M"`), letting weights drift with daily compounding within the month. This matches the
conventional meaning of "how did a 60/40 portfolio perform" (the stated mix, held
constant), rather than a pure buy-and-hold drift. No transaction costs are modelled.
Not yet exposed as a CLI/API option.

## History-length warning

`analytics/returns.tickers_with_short_history` flags any ticker with less than
`MIN_HISTORY_YEARS = 3.0` years of non-missing price observations, per CLAUDE.md's
honesty rule. Detection only — the caller decides whether to exclude the ticker or use
a documented proxy, and must surface the warning either way (the `metrics` CLI always
does).

## SyntheticProvider

`data/providers/synthetic.py` generates fully synthetic price paths via a seeded
geometric Brownian motion — **no real market data is used anywhere in its
construction**. Per-asset-class annualised drift/volatility defaults:

| Asset class | Annual drift | Annual vol |
|---|---|---|
| Stock | 8% | 20% |
| ETF | 7% | 15% |
| Bond | 3% | 6% |
| Commodity | 5% | 18% |
| *(unknown ticker)* | 7% | 15% |

These are hand-set round numbers for testing and demos, not calibrated to anything. An
optional correlation matrix can be supplied (identity/independent by default) so tests
can assert behaviour against a *known* correlation structure. Any output from this
provider must be labelled "synthetic" wherever shown to a user — the `metrics` CLI does
this automatically via a warning.

## FixtureProvider

`data/providers/fixture.py` loads a frozen CSV of **real** historical prices for
deterministic, network-free tests. `tests/fixtures/prices_fixture.csv` is a 2022–2023
slice (VWRP.L, VAGP.L, ISF.L) sourced from Yahoo Finance via `yfinance`, frozen on disk.

## Currency handling

`YFinanceProvider` normalises LSE pence-quoted instruments (`fast_info.currency ==
"GBp"`) to pounds by dividing by 100. This is detected per-ticker via a live lookup, so
it cannot be exercised in tests (which never touch the network) — covered by manual
verification only (see Milestone 1 report).
