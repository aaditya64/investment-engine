from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import app
from workbench.data import PriceDataError
from workbench.optimiser import OptimisationError

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_market_data(monkeypatch, prices):
    monkeypatch.setattr("api.main.get_prices", lambda *_: prices)


def test_health_and_current_universe() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/instruments/universe")
    assert [item["ticker"] for item in response.json()] == [
        "VWRP.L",
        "VUAG.L",
        "EQQQ.L",
        "ISF.L",
        "VFEA.L",
        "VAGP.L",
        "IGLT.L",
        "SGLN.L",
    ]


def test_optimization_contract_and_rate_used(monkeypatch, prices) -> None:
    calls = []

    def fetch(tickers, start, end):
        calls.append((tickers, start, end))
        return prices

    monkeypatch.setattr("api.main.get_prices", fetch)
    response = client.post(
        "/optimize",
        json={
            "tickers": ["VWRP.L", "VAGP.L"],
            "lookback_years": 3,
            "risk_free_rate": 0.02,
            "weight_bounds": {"VWRP.L": [0.2, 0.6]},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"weights", "metrics", "risk_free_rate"}
    assert set(body["metrics"]) == {"historical_return", "volatility", "sharpe"}
    assert body["risk_free_rate"] == 0.02
    assert sum(body["weights"].values()) == pytest.approx(1, abs=1e-7)
    assert 0.2 - 1e-7 <= body["weights"]["VWRP.L"] <= 0.6 + 1e-7
    assert body["metrics"]["sharpe"] == pytest.approx(
        (body["metrics"]["historical_return"] - 0.02) / body["metrics"]["volatility"]
    )
    end = date.today()
    assert calls == [(["VWRP.L", "VAGP.L"], end - timedelta(days=round(3 * 365.25)), end)]


@pytest.mark.parametrize(
    "extra",
    [
        {"tickers": ["AAPL"]},
        {"preferences": {}},
        {"provider": "synthetic"},
        {"weight_bounds": {"VWRP.L": [0, 0.5]}},
        {"lookback_years": 0},
    ],
)
def test_invalid_requests_are_rejected_before_fetch(monkeypatch, extra) -> None:
    def unexpected(*_):
        pytest.fail("invalid requests must not fetch data")

    monkeypatch.setattr("api.main.get_prices", unexpected)
    response = client.post("/optimize", json={"tickers": ["VWRP.L"], **extra})
    assert response.status_code == 422


def test_market_data_failure_is_visible(monkeypatch) -> None:
    def unavailable(*_):
        raise PriceDataError("price source unavailable")

    monkeypatch.setattr("api.main.get_prices", unavailable)
    response = client.post("/optimize", json={"tickers": ["VWRP.L"]})
    assert response.status_code == 502
    assert response.json()["detail"] == "price source unavailable"


def test_solver_failure_is_visible(monkeypatch) -> None:
    def failed(*_):
        raise OptimisationError("solver failed")

    monkeypatch.setattr("api.main.optimise_portfolio", failed)
    response = client.post("/optimize", json={"tickers": ["VWRP.L"]})
    assert response.status_code == 422
    assert response.json()["detail"] == "solver failed"
