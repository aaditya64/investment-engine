"""Hard limits on the portfolio, distinct from Preferences. Feasibility against a given
instrument universe is the optimiser's job to detect, not this model's."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Constraints(BaseModel):
    max_single_stock: float = Field(
        default=1.0, ge=0, le=1, description="Cap on any one stock's weight."
    )
    max_single_etf: float = Field(
        default=1.0, ge=0, le=1, description="Cap on any one ETF's weight."
    )
    max_sector_exposure: float = Field(
        default=1.0, ge=0, le=1, description="Cap on total weight in any single sector."
    )
    min_cash: float = Field(
        default=0.0, ge=0, le=1, description="Minimum weight held as cash (risk-free)."
    )
    per_instrument_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Optional (min, max) weight bounds per ticker, each in [0, 1]. "
        "Overrides the class caps above for that ticker.",
    )

    @model_validator(mode="after")
    def _bounds_are_well_formed(self) -> Constraints:
        for ticker, (lo, hi) in self.per_instrument_bounds.items():
            if not 0 <= lo <= hi <= 1:
                raise ValueError(
                    f"per_instrument_bounds for {ticker!r} must satisfy "
                    f"0 <= min <= max <= 1, got ({lo}, {hi})"
                )
        return self
