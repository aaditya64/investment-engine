#!/usr/bin/env python3
"""
Pre-investment portfolio explorer
=================================

Helps you decide WHAT to buy and in WHAT proportions before you have a portfolio.

Setup:
    pip install pandas numpy scipy matplotlib yfinance

Commands (all accept the common options listed under `python main.py -h`):
    python main.py assets                      # per-asset stats + correlations
    python main.py compare                     # equal weight / inverse-vol / min-variance /
                                               # max-Sharpe / risk-parity (+ your own mixes),
                                               # with an out-of-sample check
    python main.py frontier                    # efficient frontier plot
    python main.py deposit --amount 5000       # turn a strategy into pounds per fund
    python main.py project --lump 5000 --monthly 300 --horizon 15
                                               # Monte Carlo of possible outcomes

Examples:
    python main.py compare --tickers VWRP.L VAGP.L SGLN.L --max-weight 0.7
    python main.py compare --mix "70/30:VWRP.L=70,VAGP.L=30"
    python main.py project --strategy 70/30 --mix "70/30:VWRP.L=70,VAGP.L=30" --lump 5000 --monthly 300
    python main.py assets --demo               # synthetic data, no internet needed


"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

TD = 252  # trading days per year
CACHE_DIR = Path(".cache")


DEFAULT_UNIVERSE = {
    "VWRP.L": "Vanguard FTSE All-World (global equity, acc)",
    "VUAG.L": "Vanguard S&P 500 (US equity, acc)",
    "EQQQ.L": "Invesco Nasdaq-100 (US tech-heavy)",
    "ISF.L": "iShares Core FTSE 100 (UK equity)",
    "VFEA.L": "Vanguard FTSE Emerging Markets (acc)",
    "VAGP.L": "Vanguard Global Aggregate Bond (GBP hedged, acc)",
    "IGLT.L": "iShares Core UK Gilts",
    "SGLN.L": "iShares Physical Gold",
}


DEMO_PARAMS = {
    "VWRP.L": (0.075, 0.15, 0.95),
    "VUAG.L": (0.10, 0.17, 1.00),
    "EQQQ.L": (0.13, 0.21, 1.20),
    "ISF.L": (0.05, 0.14, 0.70),
    "VFEA.L": (0.04, 0.16, 0.80),
    "VAGP.L": (0.01, 0.07, 0.10),
    "IGLT.L": (0.00, 0.09, 0.05),
    "SGLN.L": (0.09, 0.14, 0.00),
}

STRATEGY_LABELS = {
    "equal": "Equal weight",
    "inv_vol": "Inverse volatility",
    "min_var": "Min variance",
    "max_sharpe": "Max Sharpe",
    "risk_parity": "Risk parity",
}


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def synthetic_prices(tickers: list[str], years: int, seed: int = 7) -> pd.DataFrame:
    """Fake correlated prices so the tool can be tried offline. NOT real data."""
    rng = np.random.default_rng(seed)
    n = years * TD
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mkt = rng.normal(0, 0.16 / math.sqrt(TD), n)
    cols = {}
    for t in tickers:
        mu, vol, beta = DEMO_PARAMS.get(t, (0.07, 0.16, 0.8))
        idio_var = max(vol**2 - (beta * 0.16) ** 2, (0.2 * vol) ** 2)
        e = rng.normal(0, math.sqrt(idio_var / TD), n)
        r = (mu + 0.5 * vol**2) / TD + beta * mkt + e 
        cols[t] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(cols, index=idx)


def download_prices(tickers: list[str], years: int, refresh: bool) -> pd.DataFrame:
    key = hashlib.md5(f"{sorted(tickers)}-{years}-{date.today()}".encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"prices_{key}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("error: yfinance not installed. Run: pip install yfinance (or use --demo)")
    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit("error: no data returned (network problem or bad tickers). Try --demo to test offline.")
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all")
    CACHE_DIR.mkdir(exist_ok=True)
    close.to_csv(cache)
    return close


def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = args.tickers or list(DEFAULT_UNIVERSE)
    if len(tickers) < 2:
        sys.exit("error: need at least 2 tickers")
    if args.demo:
        print("*** DEMO MODE: SYNTHETIC DATA. Results mean nothing about real markets. ***\n")
        prices = synthetic_prices(tickers, args.years)
    else:
        prices = download_prices(tickers, args.years, args.refresh)

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        print(f"warning: no data for {missing}; dropping them.")
    prices = prices[[t for t in tickers if t not in missing]]
    if prices.shape[1] < 2:
        sys.exit("error: fewer than 2 tickers have data")

    prices = prices.ffill(limit=5)
    firsts = prices.apply(lambda s: s.first_valid_index())
    limiter = firsts.idxmax()
    prices = prices.dropna()
    if len(prices) < TD:
        sys.exit("error: less than 1 year of common history across tickers; drop the newest fund.")
    yrs = len(prices) / TD
    print(
        f"Data: {prices.index[0].date()} -> {prices.index[-1].date()} "
        f"({yrs:.1f}y common history, {prices.shape[1]} assets). "
        f"Shortest history: {limiter}.\n"
    )
    if yrs < 5:
        print("warning: <5 years of data. Treat every number below as very noisy.\n")
    return prices, prices.pct_change().dropna()


# ----------------------------------------------------------------------------------------
# Metrics and backtesting
# ----------------------------------------------------------------------------------------
def stats(r: pd.Series, rf: float) -> dict:
    n = len(r)
    growth = (1 + r).cumprod()
    cagr = growth.iloc[-1] ** (TD / n) - 1
    vol = r.std() * math.sqrt(TD)
    sharpe = (r.mean() * TD - rf) / vol if vol > 0 else float("nan")
    maxdd = (growth / growth.cummax() - 1).min()
    worst1y = growth.pct_change(TD).min() if n > TD else float("nan")
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "Worst1Y": worst1y}


def backtest(rets: pd.DataFrame, w: np.ndarray, rebalance: str = "year") -> pd.Series:
    """Daily returns of a portfolio rebalanced back to `w` at period boundaries."""
    idx = rets.index
    keys = {
        "year": [idx.year],
        "quarter": [idx.year, idx.quarter],
        "month": [idx.year, idx.month],
        "never": [np.zeros(len(idx))],
    }[rebalance]
    value = 1.0
    parts = []
    for _, g in rets.groupby(keys):
        path = value * ((1 + g).cumprod() * w).sum(axis=1)
        parts.append(path)
        value = path.iloc[-1]
    values = pd.concat(parts)
    return values / values.shift(1, fill_value=1.0) - 1


# ----------------------------------------------------------------------------------------
# Allocation strategies
# ----------------------------------------------------------------------------------------
def annual_mu(rets: pd.DataFrame, shrink: float) -> np.ndarray:
    """Historical mean returns, pulled toward the cross-asset average (they're very noisy)."""
    mu = rets.mean().values * TD
    return (1 - shrink) * mu + shrink * mu.mean()


def annual_cov(rets: pd.DataFrame, delta: float = 0.1) -> np.ndarray:
    S = rets.cov().values * TD
    return (1 - delta) * S + delta * np.diag(np.diag(S))


def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    w = w.copy()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = (~over) & (w < cap - 1e-12)
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def solve(objective, n: int, cap: float, extra=()) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    res = minimize(
        objective,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, cap)] * n,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        warnings.warn(f"optimiser did not fully converge: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def parse_mix(spec: str, columns: list[str]) -> tuple[str, np.ndarray]:
    """'name:T1=70,T2=30' -> (name, weights aligned to columns, summing to 1)."""
    try:
        name, body = spec.split(":", 1)
        pairs = [p.split("=") for p in body.split(",")]
        raw = {t.strip(): float(v) for t, v in pairs}
    except ValueError:
        sys.exit(f"error: bad --mix '{spec}'. Format: name:TICKER=70,TICKER=30")
    unknown = [t for t in raw if t not in columns]
    if unknown:
        sys.exit(f"error: --mix '{name}' uses tickers not in the loaded data: {unknown}")
    w = np.array([raw.get(c, 0.0) for c in columns])
    if w.sum() <= 0 or (w < 0).any():
        sys.exit(f"error: --mix '{name}' needs positive weights")
    return name.strip(), w / w.sum()


def build_strategies(fit: pd.DataFrame, args) -> dict[str, np.ndarray]:
    n, cap = fit.shape[1], args.max_weight
    if cap * n < 1 - 1e-9:
        sys.exit(f"error: --max-weight {cap} is infeasible with {n} assets (need >= {1/n:.2f})")
    mu, S = annual_mu(fit, args.shrink), annual_cov(fit)
    vol = np.sqrt(np.diag(S))

    def port_var(w):
        return w @ S @ w

    def neg_sharpe(w):
        return -(mu @ w - args.rf) / math.sqrt(port_var(w))

    def risk_parity_obj(w):
        rc = w * (S @ w) / port_var(w)
        return 1e4 * np.sum((rc - 1 / n) ** 2)

    out = {
        "equal": apply_cap(np.full(n, 1 / n), cap),
        "inv_vol": apply_cap((1 / vol) / (1 / vol).sum(), cap),
        "min_var": solve(port_var, n, cap),
        "max_sharpe": solve(neg_sharpe, n, cap),
        "risk_parity": solve(risk_parity_obj, n, cap),
    }
    for spec in args.mix or []:
        name, w = parse_mix(spec, list(fit.columns))
        out[name] = w
    return out


def pick_strategy(strats: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in strats:
        sys.exit(f"error: unknown strategy '{name}'. Available: {', '.join(strats)}")
    return strats[name]


# ----------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------
def show(df: pd.DataFrame, pct=()) -> None:
    out = df.copy()
    for c in out.columns:
        if c in pct:
            out[c] = out[c].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "-")
        else:
            out[c] = out[c].map(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) else x)
    print(out.to_string())
    print()


def save_fig(fig, args, name: str) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    path = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved chart: {path}")


def label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


# ----------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------
def cmd_assets(args, prices, rets):
    rows = {t: stats(rets[t], args.rf) for t in rets.columns}
    df = pd.DataFrame(rows).T
    df.insert(0, "Name", [DEFAULT_UNIVERSE.get(t, "") for t in df.index])
    print("PER-ASSET STATS (full history)")
    show(df, pct=("CAGR", "Vol", "MaxDD", "Worst1Y"))

    print("CORRELATION OF DAILY RETURNS (low/negative = better diversifier)")
    corr = rets.corr()
    print(corr.round(2).to_string())
    print()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    (prices / prices.iloc[0] * 100).plot(ax=ax[0], lw=1.2)
    ax[0].set_title("Growth of 100 (price + dividends)")
    ax[0].set_ylabel("Value")
    im = ax[1].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1].set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax[1].set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[1].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set_title("Correlation")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    save_fig(fig, args, "assets")


def cmd_compare(args, prices, rets):
    if not 0.3 <= args.train_frac <= 0.9:
        sys.exit("error: --train-frac must be between 0.3 and 0.9")
    split = int(len(rets) * args.train_frac)
    train, test = rets.iloc[:split], rets.iloc[split:]
    print(
        f"Optimisers FIT on {train.index[0].date()} -> {train.index[-1].date()} "
        f"and are then TESTED on {test.index[0].date()} -> {test.index[-1].date()}."
    )
    if len(test) < TD:
        print("warning: out-of-sample window is under 1 year; lower --train-frac.")
    print()

    strats = build_strategies(train, args)
    rows, curves = {}, {}
    for name, w in strats.items():
        full = backtest(rets, w, args.rebalance)
        oos = backtest(test, w, args.rebalance)
        f, o = stats(full, args.rf), stats(oos, args.rf)
        rows[label(name)] = {
            "CAGR": f["CAGR"], "Vol": f["Vol"], "Sharpe": f["Sharpe"], "MaxDD": f["MaxDD"],
            "OOS CAGR": o["CAGR"], "OOS Vol": o["Vol"], "OOS Sharpe": o["Sharpe"], "OOS MaxDD": o["MaxDD"],
        }
        curves[label(name)] = (1 + full).cumprod()

    print(f"PERFORMANCE (rebalanced {args.rebalance}ly; left = full period incl. training data, right = out-of-sample)")
    show(pd.DataFrame(rows).T, pct=("CAGR", "Vol", "MaxDD", "OOS CAGR", "OOS Vol", "OOS MaxDD"))

    wdf = pd.DataFrame({label(k): v for k, v in strats.items()}, index=rets.columns)
    print("WEIGHTS (fitted on training window)")
    show(wdf, pct=tuple(wdf.columns))
    print("Read the OOS columns first: a strategy that shines in-sample but not out-of-sample is overfit.\n")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, v in curves.items():
        ax.plot(v.index, v.values, lw=1.4, label=k)
    ax.axvline(test.index[0], color="k", ls="--", lw=1)
    ax.text(test.index[0], ax.get_ylim()[1], " out-of-sample ->", va="top", fontsize=9)
    ax.set_title("Growth of 1 (strategy weights fitted before the dashed line)")
    ax.legend(fontsize=8)
    save_fig(fig, args, "compare")


def cmd_frontier(args, prices, rets):
    strats = build_strategies(rets, args)
    mu, S = annual_mu(rets, args.shrink), annual_cov(rets)
    n, cap = len(mu), args.max_weight

    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(n), size=20000)
    W = W[W.max(axis=1) <= cap][:5000]

    targets = np.linspace(mu.min(), mu.max(), 40)
    xs, ys = [], []
    for t in targets:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = solve(lambda w: w @ S @ w, n, cap, extra=[{"type": "eq", "fun": lambda w, t=t: mu @ w - t}])
            if abs(mu @ w - t) < 1e-4:
                xs.append(math.sqrt(w @ S @ w))
                ys.append(mu @ w)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(W):
        ax.scatter(np.sqrt(np.einsum("ij,jk,ik->i", W, S, W)), W @ mu, s=4, alpha=0.25, color="grey", label="random portfolios")
    ax.plot(xs, ys, color="C0", lw=2, label="efficient frontier")
    for i, t in enumerate(rets.columns):
        ax.scatter(math.sqrt(S[i, i]), mu[i], marker="o", color="C1")
        ax.annotate(t, (math.sqrt(S[i, i]), mu[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for name, w in strats.items():
        ax.scatter(math.sqrt(w @ S @ w), mu @ w, marker="*", s=180, label=label(name))
    ax.set_xlabel("Volatility (annual)")
    ax.set_ylabel(f"Expected return (history, {args.shrink:.0%} shrunk to average)")
    ax.set_title("Efficient frontier (long-only)")
    ax.legend(fontsize=8)
    save_fig(fig, args, "frontier")
    print("Expected returns are historical averages shrunk toward the mean; the frontier's top end is mostly noise.\n")


def cmd_deposit(args, prices, rets):
    strats = build_strategies(rets, args)
    w = pick_strategy(strats, args.strategy).copy()
    w[w < args.min_weight] = 0
    if w.sum() == 0:
        sys.exit("error: --min-weight removed everything")
    w = apply_cap(w / w.sum(), args.max_weight)
    amounts = np.round(w * args.amount, 2)
    amounts[np.argmax(amounts)] += round(args.amount - amounts.sum(), 2)  # absorb rounding
    df = pd.DataFrame(
        {"Name": [DEFAULT_UNIVERSE.get(t, "") for t in rets.columns], "Weight": w, "Amount": amounts},
        index=rets.columns,
    )
    df = df[df["Weight"] > 0].sort_values("Weight", ascending=False)
    print(f"DEPOSIT PLAN: {label(args.strategy)}, £{args.amount:,.2f} (weights < {args.min_weight:.0%} dropped)")
    out = df.copy()
    out["Weight"] = out["Weight"].map(lambda x: f"{x * 100:.1f}%")
    out["Amount"] = out["Amount"].map(lambda x: f"£{x:,.2f}")
    print(out.to_string())
    print(f"\nTotal: £{df['Amount'].sum():,.2f}")
    print("Check broker minimums, fees/spreads and whether you'd rather phase in over several months.\n")


def cmd_project(args, prices, rets):
    strats = build_strategies(rets, args)
    w = pick_strategy(strats, args.strategy)
    daily = backtest(rets, w, args.rebalance)
    monthly = ((1 + daily).groupby([daily.index.year, daily.index.month]).prod() - 1).values
    if len(monthly) < 36:
        sys.exit("error: need >=36 months of history to bootstrap")
    months = args.horizon * 12
    block = 6
    rng = np.random.default_rng(args.seed)
    nb = math.ceil(months / block)
    starts = rng.integers(0, len(monthly) - block + 1, size=(args.sims, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(args.sims, -1)[:, :months]
    r = monthly[idx] - args.haircut / 12

    paths = np.empty((args.sims, months + 1))
    paths[:, 0] = args.lump
    v = np.full(args.sims, float(args.lump))
    for t in range(months):
        v = (v + args.monthly) * (1 + r[:, t])
        paths[:, t + 1] = v

    contributed = args.lump + args.monthly * months
    deflator = (1 + args.inflation) ** args.horizon
    final = paths[:, -1]
    real = final / deflator
    pct = [10, 25, 50, 75, 90]
    print(
        f"PROJECTION: {label(args.strategy)} | £{args.lump:,.0f} now + £{args.monthly:,.0f}/month "
        f"for {args.horizon}y | {args.sims:,} sims"
    )
    hist = stats(daily, args.rf)
    print(
        f"Based on {len(monthly)} months of history (CAGR {hist['CAGR']:.1%}, vol {hist['Vol']:.1%}); "
        f"return haircut {args.haircut:.1%}/yr, inflation {args.inflation:.1%}/yr.\n"
    )
    rows = {
        f"{p}th pct": {"Nominal": np.percentile(final, p), "Today's money": np.percentile(real, p)}
        for p in pct
    }
    df = pd.DataFrame(rows).T.apply(lambda col: col.map(lambda x: f"£{x:,.0f}"))
    print(f"Total you put in: £{contributed:,.0f}")
    print(df.to_string())
    print(f"\nChance of ending below what you put in (nominal): {np.mean(final < contributed):.1%}")
    print(f"Chance of ending below what you put in (after inflation): {np.mean(real < contributed):.1%}")
    if args.target:
        print(f"Chance of reaching £{args.target:,.0f} (today's money): {np.mean(real >= args.target):.1%}")
    print("\nHistory-based bootstrap: if your data window was a bull market, so is this. Try --haircut 0.02.\n")

    x = np.arange(months + 1) / 12
    p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(x, np.percentile(paths, 25, axis=0), np.percentile(paths, 75, axis=0), alpha=0.35, label="25-75th pct")
    ax.fill_between(x, p10, p90, alpha=0.2, label="10-90th pct")
    ax.plot(x, p50, lw=2, label="median")
    ax.plot(x, args.lump + args.monthly * np.arange(months + 1), color="k", ls="--", lw=1, label="contributions only")
    ax.set_xlabel("Years")
    ax.set_ylabel("Nominal value (£)")
    ax.set_title(f"{label(args.strategy)}: simulated outcomes")
    ax.legend(fontsize=8)
    save_fig(fig, args, "project")


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tickers", nargs="+", help="Yahoo Finance symbols (default: built-in UK ETF list)")
    common.add_argument("--years", type=int, default=10, help="years of history to download (default 10)")
    common.add_argument("--rf", type=float, default=0.04, help="risk-free rate for Sharpe (default 0.04)")
    common.add_argument("--max-weight", type=float, default=0.6, help="cap per asset in optimisers (default 0.6)")
    common.add_argument("--shrink", type=float, default=0.5, help="shrink expected returns toward average, 0-1 (default 0.5)")
    common.add_argument("--rebalance", choices=["year", "quarter", "month", "never"], default="year")
    common.add_argument("--mix", action="append", help='your own mix, e.g. "70/30:VWRP.L=70,VAGP.L=30" (repeatable)')
    common.add_argument("--strategy", default="equal", help="strategy name for deposit/project (equal, inv_vol, min_var, max_sharpe, risk_parity, or a --mix name)")
    common.add_argument("--demo", action="store_true", help="use synthetic data (offline testing)")
    common.add_argument("--refresh", action="store_true", help="ignore the daily price cache")
    common.add_argument("--outdir", default="out", help="where charts are saved (default ./out)")

    p = argparse.ArgumentParser(description="Pre-investment portfolio explorer", formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("assets", parents=[common], help="per-asset stats and correlations")
    c = sub.add_parser("compare", parents=[common], help="compare allocation strategies with out-of-sample test")
    c.add_argument("--train-frac", type=float, default=0.6, help="fraction of history used to fit optimisers (default 0.6)")
    sub.add_parser("frontier", parents=[common], help="efficient frontier chart")
    d = sub.add_parser("deposit", parents=[common], help="split a deposit across funds")
    d.add_argument("--amount", type=float, required=True, help="pounds to deposit")
    d.add_argument("--min-weight", type=float, default=0.02, help="drop positions below this weight (default 0.02)")
    j = sub.add_parser("project", parents=[common], help="Monte Carlo projection")
    j.add_argument("--lump", type=float, default=0, help="initial lump sum")
    j.add_argument("--monthly", type=float, default=0, help="monthly contribution")
    j.add_argument("--horizon", type=int, default=10, help="years")
    j.add_argument("--inflation", type=float, default=0.025)
    j.add_argument("--haircut", type=float, default=0.0, help="subtract this from annual returns (e.g. 0.02)")
    j.add_argument("--target", type=float, help="target value in today's money")
    j.add_argument("--sims", type=int, default=10000)
    j.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    if not 0 < args.max_weight <= 1:
        sys.exit("error: --max-weight must be in (0, 1]")
    if not 0 <= args.shrink <= 1:
        sys.exit("error: --shrink must be in [0, 1]")
    prices, rets = load_data(args)
    {
        "assets": cmd_assets,
        "compare": cmd_compare,
        "frontier": cmd_frontier,
        "deposit": cmd_deposit,
        "project": cmd_project,
    }[args.cmd](args, prices, rets)


if __name__ == "__main__":
    main()
