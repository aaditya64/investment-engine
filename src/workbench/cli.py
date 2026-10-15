"""A small command-line wrapper over the same optimiser as the web app."""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from workbench import __version__
from workbench.constants import DEFAULT_RISK_FREE_RATE
from workbench.data import PriceDataError, get_prices
from workbench.models import PortfolioRequest
from workbench.optimiser import optimise_portfolio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="workbench", description="Historical Sharpe optimiser")
    parser.add_argument("--version", action="version", version=f"workbench {__version__}")
    commands = parser.add_subparsers(dest="command")
    optimize = commands.add_parser("optimize", help="optimise weights for the selected assets")
    optimize.add_argument("--tickers", nargs="+", required=True)
    optimize.add_argument("--lookback-years", type=float, default=5)
    optimize.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    optimize.add_argument(
        "--bound",
        action="append",
        default=[],
        metavar="TICKER=MIN,MAX",
        help="per-asset decimal bounds, e.g. VWRP.L=0.1,0.6; repeat for other assets",
    )
    optimize.add_argument(
        "--prices-csv",
        type=Path,
        help="use a local Date-indexed CSV instead of fetching prices",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        bounds = {}
        for spec in args.bound:
            ticker, raw = spec.split("=")
            lower, upper = raw.split(",")
            if ticker in bounds:
                raise ValueError(f"duplicate weight bound for {ticker}")
            bounds[ticker] = (float(lower), float(upper))
        request = PortfolioRequest(
            tickers=args.tickers,
            lookback_years=args.lookback_years,
            risk_free_rate=args.risk_free_rate,
            weight_bounds=bounds,
        )
        end = date.today()
        start = end - timedelta(days=round(request.lookback_years * 365.25))
        prices = (
            pd.read_csv(args.prices_csv, index_col="Date", parse_dates=True).loc[
                str(start) : str(end)
            ]
            if args.prices_csv
            else get_prices(request.tickers, start, end)
        )
        result = optimise_portfolio(request, prices)
    except (ValueError, OSError, PriceDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for ticker, weight in result.weights.items():
        print(f"{ticker:<10} {weight:>7.2%}")
    print(f"Historical annualised return: {result.metrics.historical_return:.2%}")
    print(f"Annualised volatility:        {result.metrics.volatility:.2%}")
    print(f"Sharpe ratio:                 {result.metrics.sharpe:.3f}")
    print(f"Annual risk-free rate:        {result.risk_free_rate:.2%}")
    print("Analysis and decision support, not financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
