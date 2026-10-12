from datetime import date

import numpy as np

from workbench.analytics.covariance import shrunk_annual_covariance
from workbench.analytics.returns import log_returns
from workbench.data.providers.synthetic import GBMParams, SyntheticProvider
from workbench.models.instrument import AssetClass


def test_shrunk_covariance_is_symmetric_positive_semidefinite() -> None:
    provider = SyntheticProvider(
        seed=1,
        asset_classes={"A": AssetClass.STOCK, "B": AssetClass.BOND},
    )
    prices = provider.get_prices(["A", "B"], date(2018, 1, 1), date(2024, 12, 31))
    daily_log = log_returns(prices)

    cov = shrunk_annual_covariance(daily_log)

    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigenvalues >= -1e-8).all()


def test_lower_vol_asset_class_has_lower_diagonal_variance() -> None:
    provider = SyntheticProvider(
        seed=2,
        asset_classes={"STOCK": AssetClass.STOCK, "BOND": AssetClass.BOND},
        params={
            AssetClass.STOCK: GBMParams(annual_drift=0.0, annual_vol=0.25),
            AssetClass.BOND: GBMParams(annual_drift=0.0, annual_vol=0.05),
        },
    )
    prices = provider.get_prices(["STOCK", "BOND"], date(2015, 1, 1), date(2024, 12, 31))
    daily_log = log_returns(prices)

    cov = shrunk_annual_covariance(daily_log)

    assert cov.loc["BOND", "BOND"] < cov.loc["STOCK", "STOCK"]
