"""Ledoit-Wolf shrunk covariance estimation. See docs/model.md."""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import LedoitWolf

from workbench.constants import TRADING_DAYS_PER_YEAR


def shrunk_annual_covariance(
    daily_log_returns: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Annualises by scaling by `trading_days` (assumes i.i.d. daily returns)."""
    estimator = LedoitWolf().fit(daily_log_returns.to_numpy())
    annual_cov = estimator.covariance_ * trading_days
    return pd.DataFrame(
        annual_cov, index=daily_log_returns.columns, columns=daily_log_returns.columns
    )
