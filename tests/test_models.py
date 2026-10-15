import pytest
from pydantic import ValidationError

from workbench.models import PortfolioRequest


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tickers": []}, "at least 1"),
        ({"tickers": ["VWRP.L", "VWRP.L"]}, "only once"),
        ({"tickers": ["AAPL"]}, "outside the current selection"),
        ({"weight_bounds": {"ISF.L": (0, 1)}}, "selected tickers"),
        ({"weight_bounds": {"VWRP.L": (-0.1, 1)}}, "minimum <= maximum"),
        ({"weight_bounds": {"VWRP.L": (0.7, 0.6)}}, "minimum <= maximum"),
        ({"weight_bounds": {"VWRP.L": (0, 1.1)}}, "minimum <= maximum"),
        ({"weight_bounds": {"VWRP.L": (0.6, 1), "VAGP.L": (0.6, 1)}}, "exceed 100%"),
        ({"weight_bounds": {"VWRP.L": (0, 0.4), "VAGP.L": (0, 0.4)}}, "reach 100%"),
        ({"weight_bounds": {"VWRP.L": (0, float("nan"))}}, "finite number"),
        ({"lookback_years": 0}, "greater than 0"),
        ({"lookback_years": float("inf")}, "finite number"),
        ({"risk_free_rate": float("nan")}, "finite number"),
        ({"risk_free_rate": -1}, "greater than -1"),
        ({"preferences": {}}, "Extra inputs"),
    ],
)
def test_invalid_requests(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        PortfolioRequest.model_validate({"tickers": ["VWRP.L", "VAGP.L"], **overrides})


def test_optional_defaults_and_feasible_fixed_weights() -> None:
    request = PortfolioRequest(tickers=["VWRP.L", "VAGP.L"])
    assert request.risk_free_rate == 0.04
    assert request.weight_bounds == {}
    assert request.lookback_years == 5
    PortfolioRequest(
        tickers=["VWRP.L", "VAGP.L"],
        weight_bounds={"VWRP.L": (0.3, 0.3), "VAGP.L": (0.7, 0.7)},
    )
