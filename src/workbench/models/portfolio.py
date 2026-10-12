"""The PortfolioRequest -> PortfolioResult contract; see CLAUDE.md architecture rule 1."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from workbench.models.constraints import Constraints
from workbench.models.instrument import Instrument
from workbench.models.preferences import Preferences


class PortfolioRequest(BaseModel):
    instruments: list[Instrument] = Field(min_length=1)
    preferences: Preferences
    constraints: Constraints = Field(default_factory=Constraints)

    @model_validator(mode="after")
    def _tickers_unique_and_referenced_keys_known(self) -> PortfolioRequest:
        tickers = [i.ticker for i in self.instruments]
        if len(tickers) != len(set(tickers)):
            raise ValueError(f"duplicate tickers in instruments: {tickers}")
        known = set(tickers)

        unknown_conviction = set(self.preferences.conviction) - known
        if unknown_conviction:
            raise ValueError(
                f"conviction references tickers not in instruments: {sorted(unknown_conviction)}"
            )

        unknown_bounds = set(self.constraints.per_instrument_bounds) - known
        if unknown_bounds:
            raise ValueError(
                f"per_instrument_bounds references tickers not in instruments: "
                f"{sorted(unknown_bounds)}"
            )
        return self


class PortfolioMetrics(BaseModel):
    """Annualised figures as decimals (0.07 = 7%/yr). See docs/model.md for the formulas."""

    expected_return: float = Field(
        description="Annualised expected return, historical estimate, shrunk."
    )
    volatility: float = Field(ge=0, description="Annualised volatility (std of returns).")
    sharpe: float = Field(
        description="Annualised Sharpe ratio using the risk-free rate in AssumptionsPanel."
    )
    sortino: float = Field(description="Annualised Sortino ratio (downside deviation only).")
    max_drawdown: float = Field(
        le=0,
        description="Largest historical peak-to-trough decline, as a negative decimal "
        "(-0.23 = -23%).",
    )
    hhi: float = Field(gt=0, le=1, description="Herfindahl-Hirschman index of weights: sum(w_i^2).")
    effective_n: float = Field(ge=1, description="Effective number of positions: 1 / hhi.")
    weighted_ter: float | None = Field(
        default=None,
        ge=0,
        description="Weight-averaged annual expense ratio. None if any held instrument's "
        "TER is unknown.",
    )


class AssumptionsPanel(BaseModel):
    """What the model assumed; always shown alongside a result (CLAUDE.md honesty rules)."""

    return_estimate_method: str
    lookback_window: str
    covariance_method: str
    risk_free_rate: float = Field(
        description="Annualised risk-free rate used for Sharpe/Sortino and the cash leg."
    )
    optimiser_objective: str | None = Field(
        default=None, description="None when no optimisation ran (e.g. a plain metrics query)."
    )
    constraints_active: list[str] = Field(
        default_factory=list, description="Constraints that bound the solution."
    )


class PortfolioResult(BaseModel):
    weights: dict[str, float] = Field(description="Ticker (or 'CASH') -> weight, summing to 1.")
    metrics: PortfolioMetrics
    exposures: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="E.g. {'geography': {...}, 'sector': {...}}. Empty until Milestone 4.",
    )
    warnings: list[str] = Field(default_factory=list)
    explanations: dict[str, str] = Field(
        default_factory=dict,
        description="Per-weight reasons, keyed by ticker. Empty until Milestone 3.",
    )
    assumptions: AssumptionsPanel

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> PortfolioResult:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total}")
        return self
