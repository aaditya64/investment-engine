# Portfolio Workbench

A small historical Sharpe optimiser. Select from the existing UK/GBP asset list,
choose a historical window, optionally set per-asset weight bounds and an annual
risk-free rate, then optimise. Results show allocation, historical annualised return,
annualised volatility, Sharpe ratio, and the risk-free rate used.

## Setup and run

```bash
uv sync
uv run uvicorn api.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Select assets, adjust the settings, and click **Optimise
portfolio**. The default lookback is five years; the risk-free rate defaults to an
assumed 4% annually and can be overridden. Weight bounds default to 0–100%; selected
assets may receive zero weight. There is no cash allocation or short-selling.

Use **Run historical test** directly after selecting assets and settings; there is no
need to press **Optimise portfolio** first. It fits its own allocation and evaluates
it on later data it was not fitted to. By default, the earliest 80% of price rows train the optimiser and the remaining
20% test it. Choose a different training percentage if needed. The result shows growth
of £1,000, realised annualised return/volatility/Sharpe, and the starting allocations
against an equal-weight baseline. Equal weights are omitted with an explanation when
they violate your starting bounds. Both portfolios buy and hold without rebalancing
or trading costs; weights may drift after purchase. Training/test dates are shown.

The existing Vue layout and allocation chart are retained. The frontend talks only to
FastAPI at http://127.0.0.1:8000. Price fetching uses yfinance with a 24-hour local
Parquet cache under `.cache/prices/`; the first request needs a network connection.

## API

`GET /instruments/universe` returns the fixed asset list. `POST /optimize` accepts:

```json
{
  "tickers": ["VWRP.L", "VAGP.L"],
  "lookback_years": 5,
  "weight_bounds": {"VWRP.L": [0.1, 0.6]},
  "risk_free_rate": 0.04
}
```

Only `tickers` is required. Rates and weights use decimals (`0.04` means 4%). The
response contains `weights`, `metrics` (`historical_return`, `volatility`, `sharpe`),
and `risk_free_rate`. The frontend uses percentage inputs and converts them at this
boundary. Invalid requests, unusable data, and failed solves return clear errors;
no fallback allocation is substituted. `/build` and the old preference contract have
been retired. Interactive API documentation is at http://127.0.0.1:8000/docs.

`POST /backtest` accepts the same fields plus optional `train_fraction` (default `0.8`,
strictly between 0 and 1):

```json
{
  "tickers": ["VWRP.L", "VAGP.L"],
  "lookback_years": 5,
  "train_fraction": 0.8,
  "risk_free_rate": 0.04
}
```

It returns training/test dates, the risk-free rate, `starting_value`, `optimised` and
optional `equal_weight` allocations/metrics, `baseline_unavailable_reason`, `data_warnings`, and dated
`growth` values. The first growth point is the purchase close at the end of training;
all measured returns end in the later test period. Realised test metrics use actual
portfolio returns, separate from the shrunk full-window optimisation estimates.
Zero realised volatility produces `null` Sharpe, displayed as undefined. A missing
purchase price produces an error naming the asset/date. Later gaps are shown in
`data_warnings`: the growth chart uses complete shared closes, and daily metrics omit
returns spanning missing quotes. No prices are filled. At least two valid adjacent-row
returns are required to calculate the test metrics.

## Python structure

```text
src/workbench/
  data.py         adjusted prices and a request-window cache
  metrics.py      historical estimates and portfolio metrics
  optimiser.py    constrained Sharpe maximisation
  backtest.py     one chronological holdout and equal-weight comparison
  models.py       request/result schemas and fixed asset list
  constants.py    annualisation and defaults
  cli.py          thin command-line wrapper
api/main.py       HTTP routes
frontend/src/     Vue UI and allocation chart
```

Historical simple-return means retain 50% shrinkage toward the selected assets'
common mean. Covariance uses Ledoit-Wolf shrinkage. SciPy SLSQP maximises the resulting
annualised Sharpe with deterministic starting allocations; it is a numerical solution,
not proof of a global maximum. All reported metrics use the same estimates as the
objective. See [docs/model.md](docs/model.md) for the definitions and limitations.

## CLI

```bash
uv run workbench optimize --tickers VWRP.L VAGP.L --lookback-years 5 \
  --bound VWRP.L=0.1,0.6 --risk-free-rate 0.04
```

For an offline run using the recorded test fixture:

```bash
uv run workbench optimize --tickers VWRP.L VAGP.L --lookback-years 30 \
  --prices-csv tests/fixtures/prices_fixture.csv
```

CSV data must have a `Date` column and a price column per selected ticker. The requested
historical window also applies to CSV input. The old `build`/`metrics` commands and
standalone experimental `main.py` have been retired.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/workbench api
cd frontend && npm run build
```

Tests use recorded prices, controlled calculations, and mocked data fetching. They
never contact Yahoo Finance. Backtest tests verify known buy-and-hold results and that
changing future prices cannot change the training allocation.
