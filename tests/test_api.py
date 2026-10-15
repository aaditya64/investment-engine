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


def test_backtest_api_returns_comparison_on_held_out_dates() -> None:
    response = client.post(
        "/backtest",
        json={"tickers": ["VWRP.L", "VAGP.L"], "train_fraction": 0.7, "risk_free_rate": 0.02},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["training_end"] < body["test_start"] <= body["test_end"]
    assert body["growth"][0]["date"] == body["training_end"]
    assert body["growth"][1]["date"] == body["test_start"]
    assert body["growth"][0]["optimised"] == pytest.approx(1000)
    assert body["growth"][0]["equal_weight"] == pytest.approx(1000)
    assert body["equal_weight"]["weights"] == {"VWRP.L": 0.5, "VAGP.L": 0.5}
    assert body["risk_free_rate"] == 0.02
    assert set(body["optimised"]["metrics"]) == {"annualised_return", "volatility", "sharpe"}


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_backtest_split_does_not_fetch_prices(monkeypatch, fraction) -> None:
    def unexpected(*_):
        pytest.fail("invalid split must not fetch data")

    monkeypatch.setattr("api.main.get_prices", unexpected)
    assert (
        client.post(
            "/backtest", json={"tickers": ["VWRP.L"], "train_fraction": fraction}
        ).status_code
        == 422
    )


def test_backtest_data_error_is_visible(monkeypatch, prices) -> None:
    split = int(len(prices) * 0.8)
    prices.iloc[split - 1, 0] = float("nan")
    monkeypatch.setattr("api.main.get_prices", lambda *_: prices)
    response = client.post("/backtest", json={"tickers": ["VWRP.L"]})
    assert response.status_code == 422
    assert "missing purchase prices" in response.json()["detail"]
    assert "VWRP.L" in response.json()["detail"]


def test_backtest_test_quote_gap_returns_results_with_warning(monkeypatch, prices) -> None:
    prices.iloc[-5, 0] = float("nan")
    monkeypatch.setattr("api.main.get_prices", lambda *_: prices)
    response = client.post("/backtest", json={"tickers": ["VWRP.L", "VAGP.L"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "VWRP.L" in body["data_warnings"][0]
    assert str(prices.index[-5].date()) not in [point["date"] for point in body["growth"]]


def test_backtest_market_failure_is_visible(monkeypatch) -> None:
    def unavailable(*_):
        raise PriceDataError("price source unavailable")

    monkeypatch.setattr("api.main.get_prices", unavailable)
    response = client.post("/backtest", json={"tickers": ["VWRP.L"]})
    assert response.status_code == 502


def test_backtest_solver_failure_is_visible(monkeypatch) -> None:
    def failed(*_):
        raise OptimisationError("training solve failed")

    monkeypatch.setattr("workbench.backtest.optimise_portfolio", failed)
    response = client.post("/backtest", json={"tickers": ["VWRP.L"]})
    assert response.status_code == 422
    assert response.json()["detail"] == "training solve failed"
