"""The fixed asset selection and the small optimisation request/result contract."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workbench.constants import DEFAULT_RISK_FREE_RATE


class Instrument(BaseModel):
    ticker: str
    name: str
    asset_class: Literal["etf", "bond", "commodity"]


UNIVERSE = [
    Instrument(ticker="VWRP.L", name="Vanguard FTSE All-World (acc)", asset_class="etf"),
    Instrument(ticker="VUAG.L", name="Vanguard S&P 500 (acc)", asset_class="etf"),
    Instrument(ticker="EQQQ.L", name="Invesco Nasdaq-100", asset_class="etf"),
    Instrument(ticker="ISF.L", name="iShares Core FTSE 100", asset_class="etf"),
    Instrument(ticker="VFEA.L", name="Vanguard FTSE Emerging Markets (acc)", asset_class="etf"),
    Instrument(
        ticker="VAGP.L", name="Vanguard Global Aggregate Bond (GBP hedged, acc)", asset_class="bond"
    ),
    Instrument(ticker="IGLT.L", name="iShares Core UK Gilts", asset_class="bond"),
    Instrument(ticker="SGLN.L", name="iShares Physical Gold", asset_class="commodity"),
]


class PortfolioRequest(BaseModel):
    """Rates and weights are decimals; lookback is years of historical data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    tickers: list[str] = Field(min_length=1, max_length=len(UNIVERSE))
    lookback_years: float = Field(default=5.0, gt=0, le=30)
    weight_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    risk_free_rate: float = Field(default=DEFAULT_RISK_FREE_RATE, gt=-1)

    @model_validator(mode="after")
    def validate_selection_and_bounds(self) -> "PortfolioRequest":
        if len(self.tickers) != len(set(self.tickers)):
            raise ValueError("select each ticker only once")
        unknown = set(self.tickers) - {item.ticker for item in UNIVERSE}
        if unknown:
            raise ValueError(f"tickers outside the current selection: {', '.join(sorted(unknown))}")
        if set(self.weight_bounds) - set(self.tickers):
            raise ValueError("weight bounds must refer to selected tickers")
        bounds = [self.weight_bounds.get(ticker, (0.0, 1.0)) for ticker in self.tickers]
        if any(not 0 <= lower <= upper <= 1 for lower, upper in bounds):
            raise ValueError("each weight bound must satisfy 0 <= minimum <= maximum <= 1")
        if sum(lower for lower, _ in bounds) > 1 + 1e-10:
            raise ValueError("minimum weights exceed 100% in total")
        if sum(upper for _, upper in bounds) < 1 - 1e-10:
            raise ValueError("maximum weights cannot reach 100% in total")
        return self


class PortfolioMetrics(BaseModel):
    """Annualised historical estimates; return/volatility are decimals, Sharpe is a ratio."""

    model_config = ConfigDict(allow_inf_nan=False)

    historical_return: float
    volatility: float = Field(gt=0)
    sharpe: float


class PortfolioResult(BaseModel):
    weights: dict[str, float]
    metrics: PortfolioMetrics
    risk_free_rate: float


class BacktestRequest(PortfolioRequest):
    """Fit on the first train_fraction of price rows; evaluate on the rest."""

    train_fraction: float = Field(default=0.8, gt=0, lt=1)


class BacktestMetrics(BaseModel):
    """Realised arithmetic annual return, sample volatility, and annual Sharpe."""

    model_config = ConfigDict(allow_inf_nan=False)

    annualised_return: float
    volatility: float = Field(ge=0)
    sharpe: float | None  # Undefined when the realised return path has zero volatility.


class BacktestPortfolio(BaseModel):
    weights: dict[str, float]
    metrics: BacktestMetrics


class GrowthPoint(BaseModel):
    date: date
    optimised: float
    equal_weight: float | None


class BacktestResult(BaseModel):
    training_start: date
    training_end: date
    test_start: date
    test_end: date
    risk_free_rate: float
    starting_value: float
    optimised: BacktestPortfolio
    equal_weight: BacktestPortfolio | None
    baseline_unavailable_reason: str | None
    data_warnings: list[str] = Field(default_factory=list)
    growth: list[GrowthPoint]
