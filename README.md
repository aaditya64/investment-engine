# Portfolio Workbench

An interactive portfolio construction and experimentation tool: choose instruments, say
how much you want each, set risk/horizon/constraints, see what the model produces and
why, tweak anything, see the consequences immediately.


## Setup

```bash
uv sync
```

## Commands

```bash
uv run pytest                                    # tests (never touch the network)
uv run ruff check . && uv run ruff format .
uv run mypy src/workbench/engine src/workbench/models
uv run workbench --help
```

### `metrics`: risk/return metrics for a fixed weight mix

```bash
uv run workbench metrics --tickers VWRP.L VAGP.L --weights 0.7 0.3
```

