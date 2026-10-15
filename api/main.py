"""FastAPI routes for the fixed-universe Sharpe optimiser."""

from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from workbench.data import PriceDataError, get_prices
from workbench.models import UNIVERSE, Instrument, PortfolioRequest, PortfolioResult
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


@app.get("/instruments/universe")
def instrument_universe() -> list[Instrument]:
    return UNIVERSE


@app.post("/optimize", response_model=PortfolioResult)
def optimize(body: PortfolioRequest) -> PortfolioResult:
    end = date.today()
    start = end - timedelta(days=round(body.lookback_years * 365.25))
    try:
        prices = get_prices(body.tickers, start, end)
    except PriceDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        return optimise_portfolio(body, prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
