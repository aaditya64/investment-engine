"""FastAPI routes for asset search, optimisation, and backtesting."""

from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from workbench.backtest import backtest_portfolio
from workbench.data import PriceDataError, get_prices, search_assets
from workbench.models import (
    BacktestRequest,
    BacktestResult,
    Instrument,
    PortfolioRequest,
    PortfolioResult,
)
from workbench.optimiser import optimise_portfolio

app = FastAPI(title="Portfolio Workbench API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/instruments/search")
def instrument_search(q: str = Query(min_length=1, max_length=80)) -> list[Instrument]:
    try:
        return search_assets(q)
    except PriceDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/optimize", response_model=PortfolioResult)
def optimize(body: PortfolioRequest) -> PortfolioResult:
    prices = _load_prices(body)
    try:
        return optimise_portfolio(body, prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/backtest", response_model=BacktestResult)
def backtest(body: BacktestRequest) -> BacktestResult:
    prices = _load_prices(body)
    try:
        return backtest_portfolio(body, prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _load_prices(body: PortfolioRequest) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=round(body.lookback_years * 365.25))
    try:
        return get_prices(body.tickers, start, end)
    except PriceDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
