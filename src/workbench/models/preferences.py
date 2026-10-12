"""User preferences, normalised to 0-1. Mapping to engine parameters lives in
engine/mapping.py, not here."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Preferences(BaseModel):
    conviction: dict[str, float] = Field(
        default_factory=dict,
        description="Per-instrument conviction in [0, 1], keyed by ticker. "
        "0 = no particular desire to hold it, 1 = strong conviction.",
    )
    risk_appetite: float = Field(
        ge=0, le=1, description="0 = most risk-averse, 1 = most risk-tolerant."
    )
    horizon_years: float = Field(gt=0, description="Investment horizon in years.")
    capital_preservation_vs_growth: float = Field(
        ge=0, le=1, description="0 = prioritise capital preservation, 1 = prioritise growth."
    )
    diversification_preference: float = Field(
        ge=0, le=1, description="0 = comfortable concentrating, 1 = strongly prefer spreading risk."
    )
    geography_tilts: dict[str, float] = Field(
        default_factory=dict,
        description="Optional small set of geography tilts in [-1, 1], keyed by region name.",
    )
    sector_tilts: dict[str, float] = Field(
        default_factory=dict,
        description="Optional small set of sector tilts in [-1, 1], keyed by sector name.",
    )

    @field_validator("conviction")
    @classmethod
    def _conviction_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for ticker, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"conviction for {ticker!r} must be in [0, 1], got {value}")
        return v

    @field_validator("geography_tilts", "sector_tilts")
    @classmethod
    def _tilts_in_signed_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if not -1 <= value <= 1:
                raise ValueError(f"tilt for {key!r} must be in [-1, 1], got {value}")
        return v
