import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from workbench.metrics import estimate_returns, portfolio_metrics
from workbench.models import PortfolioRequest
from workbench.optimiser import OptimisationError, optimise_portfolio

TICKERS = ["VWRP.L", "VAGP.L"]


def test_matches_analytic_tangency_portfolio(monkeypatch) -> None:
    mean = np.array([0.12, 0.08])
    covariance = np.array([[0.04, 0.006], [0.006, 0.01]])
    monkeypatch.setattr("workbench.optimiser.estimate_returns", lambda *_: (mean, covariance))
    request = PortfolioRequest(tickers=TICKERS, risk_free_rate=0.02)
    result = optimise_portfolio(request, pd.DataFrame())
    analytic = np.linalg.solve(covariance, mean - 0.02)
    analytic /= analytic.sum()
    assert list(result.weights.values()) == pytest.approx(analytic, abs=1e-5)
    assert result.metrics.sharpe == pytest.approx(
        portfolio_metrics(analytic, mean, covariance, 0.02).sharpe, abs=1e-8
    )


def test_obeys_binding_bounds(monkeypatch) -> None:
    monkeypatch.setattr(
        "workbench.optimiser.estimate_returns",
        lambda *_: (np.array([0.12, 0.08]), np.diag([0.04, 0.01])),
    )
    result = optimise_portfolio(
        PortfolioRequest(tickers=TICKERS, weight_bounds={"VWRP.L": (0.5, 0.7)}),
        pd.DataFrame(),
    )
    assert result.weights == pytest.approx({"VWRP.L": 0.5, "VAGP.L": 0.5}, abs=1e-6)


def test_negative_excess_returns_still_select_best_sharpe(monkeypatch) -> None:
    monkeypatch.setattr(
        "workbench.optimiser.estimate_returns",
        lambda *_: (np.array([0.01, -0.01]), np.diag([0.04, 0.01])),
    )
    result = optimise_portfolio(PortfolioRequest(tickers=TICKERS), pd.DataFrame())
    assert result.weights == pytest.approx({"VWRP.L": 1, "VAGP.L": 0}, abs=1e-6)
    assert result.metrics.sharpe == pytest.approx(-0.15)


def test_real_fixture_constraints_metrics_and_determinism(prices) -> None:
    request = PortfolioRequest(
        tickers=[*TICKERS, "ISF.L"],
        weight_bounds={ticker: (0.1, 0.6) for ticker in [*TICKERS, "ISF.L"]},
        risk_free_rate=0.01,
    )
    result = optimise_portfolio(request, prices)
    assert sum(result.weights.values()) == pytest.approx(1, abs=1e-7)
    assert all(0.1 - 1e-7 <= w <= 0.6 + 1e-7 for w in result.weights.values())
    assert result == optimise_portfolio(request, prices)
    mean, covariance = estimate_returns(prices, request.tickers)
    independently_reported = portfolio_metrics(
        np.array(list(result.weights.values())), mean, covariance, request.risk_free_rate
    )
    assert result.metrics == independently_reported
    assert result.risk_free_rate == 0.01


@pytest.mark.parametrize(
    "bounds",
    [
        {"VWRP.L": (0.3, 0.3), "VAGP.L": (0.7, 0.7)},
        {"VWRP.L": (0.3, 1), "VAGP.L": (0.7, 1)},
        {"VWRP.L": (0, 0.3), "VAGP.L": (0, 0.7)},
    ],
)
def test_bounds_that_force_one_allocation(prices, bounds) -> None:
    result = optimise_portfolio(PortfolioRequest(tickers=TICKERS, weight_bounds=bounds), prices)
    assert result.weights == pytest.approx({"VWRP.L": 0.3, "VAGP.L": 0.7})


def test_single_selected_asset_gets_all_weight(prices) -> None:
    result = optimise_portfolio(PortfolioRequest(tickers=["VWRP.L"]), prices)
    assert result.weights == {"VWRP.L": 1.0}


@pytest.mark.parametrize(
    "solved",
    [
        OptimizeResult(success=False, message="iteration limit", x=np.array([0.5, 0.5])),
        OptimizeResult(success=True, message="invalid weights", x=np.array([1.1, -0.1])),
        OptimizeResult(success=True, message="invalid total", x=np.array([0.3, 0.3])),
        OptimizeResult(success=True, message="nonfinite", x=np.array([np.nan, np.nan])),
    ],
)
def test_failed_or_invalid_solve_never_returns_fallback(monkeypatch, prices, solved) -> None:
    monkeypatch.setattr("workbench.optimiser.minimize", lambda *_, **__: solved)
    with pytest.raises(OptimisationError, match="could not optimise"):
        optimise_portfolio(PortfolioRequest(tickers=TICKERS), prices)


def test_zero_volatility_is_an_error() -> None:
    prices = pd.DataFrame({"VWRP.L": [100, 100, 100]}, index=pd.date_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="volatility is zero"):
        optimise_portfolio(PortfolioRequest(tickers=["VWRP.L"]), prices)
