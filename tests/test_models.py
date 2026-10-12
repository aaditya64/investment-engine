import pytest
from pydantic import ValidationError

from workbench.models import (
    AssetClass,
    AssumptionsPanel,
    Constraints,
    Instrument,
    PortfolioMetrics,
    PortfolioRequest,
    PortfolioResult,
    Preferences,
)


def make_instrument(ticker: str = "VWRP.L", **overrides: object) -> Instrument:
    defaults: dict[str, object] = {
        "ticker": ticker,
        "name": "Vanguard FTSE All-World",
        "asset_class": AssetClass.ETF,
        "currency": "GBP",
        "region": "Global",
        "ter": 0.0022,
    }
    defaults.update(overrides)
    return Instrument(**defaults)  # type: ignore[arg-type]


def make_preferences(**overrides: object) -> Preferences:
    defaults: dict[str, object] = {
        "conviction": {"VWRP.L": 0.8},
        "risk_appetite": 0.5,
        "horizon_years": 10,
        "capital_preservation_vs_growth": 0.5,
        "diversification_preference": 0.5,
    }
    defaults.update(overrides)
    return Preferences(**defaults)  # type: ignore[arg-type]


def test_instrument_rejects_ter_out_of_range() -> None:
    with pytest.raises(ValidationError):
        make_instrument(ter=1.5)


def test_instrument_allows_unknown_ter() -> None:
    instrument = make_instrument(ter=None)
    assert instrument.ter is None


def test_preferences_rejects_conviction_out_of_range() -> None:
    with pytest.raises(ValidationError):
        make_preferences(conviction={"VWRP.L": 1.5})


def test_preferences_rejects_tilt_out_of_range() -> None:
    with pytest.raises(ValidationError):
        make_preferences(geography_tilts={"US": 2.0})


def test_constraints_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        Constraints(per_instrument_bounds={"VWRP.L": (0.5, 0.2)})


def test_constraints_defaults_are_unconstrained() -> None:
    constraints = Constraints()
    assert constraints.max_single_stock == 1.0
    assert constraints.min_cash == 0.0
    assert constraints.per_instrument_bounds == {}


def test_portfolio_request_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValidationError):
        PortfolioRequest(
            instruments=[make_instrument(), make_instrument()],
            preferences=make_preferences(),
        )


def test_portfolio_request_rejects_unknown_conviction_ticker() -> None:
    with pytest.raises(ValidationError):
        PortfolioRequest(
            instruments=[make_instrument()],
            preferences=make_preferences(conviction={"NOT_SELECTED.L": 0.5}),
        )


def test_portfolio_request_rejects_unknown_bound_ticker() -> None:
    with pytest.raises(ValidationError):
        PortfolioRequest(
            instruments=[make_instrument()],
            preferences=make_preferences(),
            constraints=Constraints(per_instrument_bounds={"NOT_SELECTED.L": (0.0, 1.0)}),
        )


def test_portfolio_request_accepts_consistent_input() -> None:
    request = PortfolioRequest(
        instruments=[make_instrument()],
        preferences=make_preferences(),
        constraints=Constraints(per_instrument_bounds={"VWRP.L": (0.0, 1.0)}),
    )
    assert request.instruments[0].ticker == "VWRP.L"


def make_metrics(**overrides: object) -> PortfolioMetrics:
    defaults: dict[str, object] = {
        "expected_return": 0.07,
        "volatility": 0.15,
        "sharpe": 0.4,
        "sortino": 0.5,
        "max_drawdown": -0.23,
        "hhi": 1.0,
        "effective_n": 1.0,
        "weighted_ter": 0.0022,
    }
    defaults.update(overrides)
    return PortfolioMetrics(**defaults)  # type: ignore[arg-type]


def make_assumptions(**overrides: object) -> AssumptionsPanel:
    defaults: dict[str, object] = {
        "return_estimate_method": "shrunk historical mean",
        "lookback_window": "5y daily",
        "covariance_method": "Ledoit-Wolf",
        "risk_free_rate": 0.04,
    }
    defaults.update(overrides)
    return AssumptionsPanel(**defaults)  # type: ignore[arg-type]


def test_portfolio_result_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValidationError):
        PortfolioResult(
            weights={"VWRP.L": 0.5},
            metrics=make_metrics(),
            assumptions=make_assumptions(),
        )


def test_portfolio_result_accepts_weights_summing_to_one() -> None:
    result = PortfolioResult(
        weights={"VWRP.L": 1.0},
        metrics=make_metrics(),
        assumptions=make_assumptions(),
    )
    assert result.weights["VWRP.L"] == 1.0
    assert result.warnings == []
    assert result.explanations == {}
    assert result.exposures == {}
