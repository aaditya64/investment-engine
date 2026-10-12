from datetime import date

import numpy as np
import pytest

from workbench.data.providers.synthetic import GBMParams, SyntheticProvider
from workbench.models.instrument import AssetClass


def test_deterministic_given_same_seed() -> None:
    provider_a = SyntheticProvider(seed=42)
    provider_b = SyntheticProvider(seed=42)
    tickers = ["X", "Y"]
    start, end = date(2024, 1, 1), date(2024, 6, 30)

    frame_a = provider_a.get_prices(tickers, start, end)
    frame_b = provider_b.get_prices(tickers, start, end)

    pd_equal = (frame_a == frame_b).all().all()
    assert pd_equal


def test_different_seed_gives_different_paths() -> None:
    start, end = date(2024, 1, 1), date(2024, 6, 30)
    frame_a = SyntheticProvider(seed=1).get_prices(["X"], start, end)
    frame_b = SyntheticProvider(seed=2).get_prices(["X"], start, end)
    assert not (frame_a == frame_b).all().all()


def test_shape_matches_requested_business_days_and_tickers() -> None:
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    tickers = ["A", "B", "C"]
    frame = SyntheticProvider(seed=0).get_prices(tickers, start, end)
    expected_days = len([d for d in _business_days(start, end)])
    assert list(frame.columns) == tickers
    assert len(frame) == expected_days
    assert frame.iloc[0].tolist() == [100.0, 100.0, 100.0]


def _business_days(start: date, end: date) -> list[date]:
    import pandas as pd

    return list(pd.bdate_range(start, end))


def test_unknown_ticker_falls_back_to_default_params() -> None:
    frame = SyntheticProvider(seed=0).get_prices(["UNKNOWN"], date(2024, 1, 1), date(2024, 1, 31))
    assert not frame["UNKNOWN"].isna().any()


def test_asset_class_changes_the_simulated_volatility_regime() -> None:
    start, end = date(2015, 1, 1), date(2024, 12, 31)
    bond_provider = SyntheticProvider(seed=7, asset_classes={"B": AssetClass.BOND})
    stock_provider = SyntheticProvider(seed=7, asset_classes={"S": AssetClass.STOCK})

    bond_returns = bond_provider.get_prices(["B"], start, end)["B"].pct_change().dropna()
    stock_returns = stock_provider.get_prices(["S"], start, end)["S"].pct_change().dropna()

    assert bond_returns.std() < stock_returns.std()


def test_correlation_matrix_is_respected_approximately() -> None:
    rng_params = {
        AssetClass.STOCK: GBMParams(annual_drift=0.0, annual_vol=0.20),
    }
    correlated = SyntheticProvider(
        seed=123,
        asset_classes={"A": AssetClass.STOCK, "B": AssetClass.STOCK},
        params=rng_params,
        correlation=np.array([[1.0, 0.9], [0.9, 1.0]]),
    )
    frame = correlated.get_prices(["A", "B"], date(2010, 1, 1), date(2024, 12, 31))
    returns = frame.pct_change().dropna()
    empirical_corr = returns["A"].corr(returns["B"])
    assert empirical_corr > 0.8


def test_correlation_matrix_wrong_shape_raises() -> None:
    provider = SyntheticProvider(seed=0, correlation=np.eye(3))
    with pytest.raises(ValueError, match="correlation must be"):
        provider.get_prices(["A", "B"], date(2024, 1, 1), date(2024, 1, 31))
