"""Thin CLI over workbench's public API. No financial logic here; see architecture rule 1."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from workbench import __version__
from workbench.analytics import (
    DEFAULT_SHRINKAGE,
    compute_portfolio_metrics,
    tickers_with_short_history,
)
from workbench.constants import DEFAULT_RISK_FREE_RATE
from workbench.data.providers import FixtureProvider, SyntheticProvider, YFinanceProvider
from workbench.data.providers.base import PriceProvider
from workbench.models import AssetClass, AssumptionsPanel, Instrument, PortfolioResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workbench",
        description="Portfolio Workbench — decision support, not financial advice.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="print the installed version")

    metrics_parser = subparsers.add_parser(
        "metrics", help="compute risk/return metrics for a fixed weight mix"
    )
    metrics_parser.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    metrics_parser.add_argument("--weights", nargs="+", required=True, type=float, metavar="W")
    metrics_parser.add_argument(
        "--provider",
        choices=["live", "fixture", "synthetic"],
        default="live",
        help="live = yfinance (cached on disk); fixture/synthetic are network-free (default: live)",
    )
    metrics_parser.add_argument(
        "--fixture-path", type=Path, default=None, help="required when --provider fixture"
    )
    metrics_parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed when --provider synthetic (default: 0)"
    )
    metrics_parser.add_argument(
        "--lookback-years", type=float, default=5.0, help="history window to fetch (default: 5)"
    )
    metrics_parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    metrics_parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    metrics_parser.add_argument(
        "--ter",
        nargs="*",
        default=None,
        metavar="TICKER=VALUE",
        help="known expense ratios as decimals, e.g. VWRP.L=0.0022; unlisted tickers are unknown",
    )
    return parser


def _parse_weights(tickers: list[str], raw_weights: list[float]) -> dict[str, float] | str:
    if len(tickers) != len(raw_weights):
        return f"{len(tickers)} tickers but {len(raw_weights)} weights were given"
    total = sum(raw_weights)
    if abs(total - 1.0) > 1e-6:
        return f"weights must sum to 1, got {total}"
    return dict(zip(tickers, raw_weights, strict=True))


def _parse_ter_overrides(entries: list[str] | None) -> dict[str, float] | str:
    overrides: dict[str, float] = {}
    for entry in entries or []:
        ticker, separator, value = entry.partition("=")
        if not separator:
            return f"--ter entries must be TICKER=VALUE, got {entry!r}"
        try:
            overrides[ticker] = float(value)
        except ValueError:
            return f"--ter value for {ticker!r} is not a number: {value!r}"
    return overrides


def _build_provider(args: argparse.Namespace) -> PriceProvider | str:
    if args.provider == "live":
        return YFinanceProvider()
    if args.provider == "fixture":
        if args.fixture_path is None:
            return "--fixture-path is required when --provider fixture"
        return FixtureProvider(args.fixture_path)
    return SyntheticProvider(seed=args.seed)


def _print_metrics_result(
    result: PortfolioResult, provider_label: str, start: date, end: date
) -> None:
    print(f"Weights ({provider_label} prices, {start} to {end}):")
    for ticker, weight in result.weights.items():
        print(f"  {ticker:<12} {weight:>7.2%}")

    m = result.metrics
    print("\nMetrics:")
    print(f"  Expected return (annualised, shrunk estimate): {m.expected_return:>8.2%}")
    print(f"  Volatility (annualised):                       {m.volatility:>8.2%}")
    print(f"  Sharpe (vs. shrunk estimate):                  {m.sharpe:>8.2f}")
    print(f"  Sortino (historical):                          {m.sortino:>8.2f}")
    print(f"  Max drawdown (historical):                     {m.max_drawdown:>8.2%}")
    print(f"  HHI / effective N:                             {m.hhi:.3f} / {m.effective_n:.1f}")
    ter_label = f"{m.weighted_ter:.2%}" if m.weighted_ter is not None else "unknown"
    print(f"  Weighted TER:                                   {ter_label}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    a = result.assumptions
    print("\nAssumptions:")
    print(f"  Return estimate method: {a.return_estimate_method}")
    print(f"  Covariance method:      {a.covariance_method}")
    print(f"  Lookback window:        {a.lookback_window}")
    print(f"  Risk-free rate:         {a.risk_free_rate:.2%}")

    print("\nThis is analysis and decision support, not financial advice.")


def _run_metrics(args: argparse.Namespace) -> int:
    weights = _parse_weights(args.tickers, args.weights)
    if isinstance(weights, str):
        print(f"error: {weights}", file=sys.stderr)
        return 1

    ter_overrides = _parse_ter_overrides(args.ter)
    if isinstance(ter_overrides, str):
        print(f"error: {ter_overrides}", file=sys.stderr)
        return 1

    provider = _build_provider(args)
    if isinstance(provider, str):
        print(f"error: {provider}", file=sys.stderr)
        return 1

    tickers = args.tickers
    instruments = [
        Instrument(
            ticker=ticker,
            name=ticker,
            asset_class=AssetClass.ETF,
            currency="GBP",
            region="Unknown",
            ter=ter_overrides.get(ticker),
        )
        for ticker in tickers
    ]

    end = date.today()
    start = end - timedelta(days=round(args.lookback_years * 365.25))

    try:
        prices = provider.get_prices(tickers, start, end)
    except Exception as exc:
        print(f"error: could not fetch prices: {exc}", file=sys.stderr)
        return 1

    warnings: list[str] = []
    short_history = tickers_with_short_history(prices[tickers])
    if short_history:
        warnings.append(
            f"less than 3 years of price history (estimates may be unreliable): "
            f"{', '.join(short_history)}"
        )
    if args.provider == "synthetic":
        warnings.append("prices are SYNTHETIC (seeded random walk), not real market data")

    try:
        metrics = compute_portfolio_metrics(
            weights=weights,
            instruments=instruments,
            prices=prices,
            risk_free_rate=args.risk_free_rate,
            shrinkage=args.shrinkage,
        )
    except Exception as exc:
        print(f"error: could not compute metrics: {exc}", file=sys.stderr)
        return 1

    result = PortfolioResult(
        weights=weights,
        metrics=metrics,
        warnings=warnings,
        assumptions=AssumptionsPanel(
            return_estimate_method=f"historical daily log-return mean, "
            f"shrunk {args.shrinkage:.0%} toward the cross-sectional grand mean",
            lookback_window=f"{start} to {end} ({args.lookback_years:g}y requested)",
            covariance_method="Ledoit-Wolf shrinkage, annualised",
            risk_free_rate=args.risk_free_rate,
        ),
    )

    _print_metrics_result(result, provider_label=args.provider, start=start, end=end)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"workbench {__version__}")
        return 0
    if args.command == "metrics":
        return _run_metrics(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
