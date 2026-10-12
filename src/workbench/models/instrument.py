"""The investable things a user can select. Cash is not an Instrument (see Constraints.min_cash)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AssetClass(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"


class Instrument(BaseModel):
    """A single tradeable instrument the user has selected. `ter` is a decimal fraction
    (0.0022 = 0.22%); unknown/unverified TER must be None, never guessed."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(min_length=1, description="Exchange ticker, e.g. 'VWRP.L'.")
    name: str = Field(min_length=1, description="Human-readable instrument name.")
    asset_class: AssetClass
    currency: str = Field(min_length=3, max_length=3, description="ISO 4217 code, e.g. 'GBP'.")
    region: str = Field(min_length=1, description="Primary market/region, e.g. 'Global', 'UK'.")
    ter: float | None = Field(
        default=None,
        ge=0,
        lt=1,
        description="Annual expense ratio as a decimal fraction, if known.",
    )
