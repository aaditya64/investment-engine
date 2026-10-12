"""Shrinks noisy historical mean returns toward the cross-sectional grand mean. See
docs/model.md for why (deferred: equilibrium/Black-Litterman shrinkage, roadmap item 6)."""

from __future__ import annotations

import pandas as pd

DEFAULT_SHRINKAGE = 0.5
"""0 = raw historical means, 1 = collapse every instrument to the grand mean."""


def shrink_to_grand_mean(
    mean_returns: pd.Series, shrinkage: float = DEFAULT_SHRINKAGE
) -> pd.Series:
    if not 0 <= shrinkage <= 1:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    grand_mean = mean_returns.mean()
    return (1 - shrinkage) * mean_returns + shrinkage * grand_mean
