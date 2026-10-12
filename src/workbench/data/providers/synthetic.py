"""Seeded, fully synthetic (non-market) price paths for ground-truth tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from workbench.constants import TRADING_DAYS_PER_YEAR
from workbench.models.instrument import AssetClass


@dataclass(frozen=True)
class GBMParams:
    """Annualised drift and volatility for a geometric Brownian motion."""

    annual_drift: float
    annual_vol: float


DEFAULT_PARAMS: dict[AssetClass, GBMParams] = {
    AssetClass.STOCK: GBMParams(annual_drift=0.08, annual_vol=0.20),
    AssetClass.ETF: GBMParams(annual_drift=0.07, annual_vol=0.15),
    AssetClass.BOND: GBMParams(annual_drift=0.03, annual_vol=0.06),
    AssetClass.COMMODITY: GBMParams(annual_drift=0.05, annual_vol=0.18),
}
FALLBACK_PARAMS = GBMParams(annual_drift=0.07, annual_vol=0.15)


class SyntheticProvider:
    def __init__(
        self,
        seed: int,
        asset_classes: dict[str, AssetClass] | None = None,
        params: dict[AssetClass, GBMParams] | None = None,
        correlation: np.ndarray | None = None,
    ) -> None:
        """correlation is a k x k matrix across `tickers`; defaults to independent walks."""
        self.seed = seed
        self.asset_classes = asset_classes or {}
        self.params = params or DEFAULT_PARAMS
        self.correlation = correlation

    def _params_for(self, ticker: str) -> GBMParams:
        asset_class = self.asset_classes.get(ticker)
        if asset_class is None:
            return FALLBACK_PARAMS
        return self.params.get(asset_class, FALLBACK_PARAMS)

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        n_steps = len(dates) - 1
        k = len(tickers)
        if n_steps < 1:
            raise ValueError("start/end must span at least two business days")

        correlation = self.correlation if self.correlation is not None else np.eye(k)
        if correlation.shape != (k, k):
            raise ValueError(
                f"correlation must be {k}x{k} for {k} tickers, got {correlation.shape}"
            )

        rng = np.random.default_rng(self.seed)
        cholesky = np.linalg.cholesky(correlation)
        independent_shocks = rng.standard_normal((n_steps, k))
        correlated_shocks = independent_shocks @ cholesky.T

        annual_drifts = np.array([self._params_for(t).annual_drift for t in tickers])
        annual_vols = np.array([self._params_for(t).annual_vol for t in tickers])
        daily_drift = annual_drifts / TRADING_DAYS_PER_YEAR
        daily_vol = annual_vols / np.sqrt(TRADING_DAYS_PER_YEAR)

        daily_log_returns = daily_drift + correlated_shocks * daily_vol
        log_prices = np.concatenate(
            [np.zeros((1, k)), np.cumsum(daily_log_returns, axis=0)], axis=0
        )
        prices = 100.0 * np.exp(log_prices)

        return pd.DataFrame(prices, index=dates, columns=tickers)
