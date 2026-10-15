import numpy as np
import pandas as pd
import pytest

from workbench.backtest import backtest_portfolio
from workbench.models import BacktestRequest
from workbench.optimiser import optimise_portfolio

TICKERS = ["VWRP.L", "VAGP.L"]


@pytest.fixture
def small_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VWRP.L": [100, 101, 99, 100, 110, 121, 108.9],
            "VAGP.L": [200, 198, 202, 200, 180, 198, 207.9],
        },
        index=pd.bdate_range("2024-01-02", periods=7),
    )


def test_known_buy_and_hold_growth_metrics_and_boundaries(small_prices) -> None:
    result = backtest_portfolio(
        BacktestRequest(
            tickers=TICKERS,
            train_fraction=0.6,
            weight_bounds={ticker: (0.5, 0.5) for ticker in TICKERS},
            risk_free_rate=0.02,
        ),
        small_prices,
    )
    # Fixed units: the second asset's starting price is £200, not £100.
    expected_growth = [1000, 1000, 1100, 1064.25]
    assert [point.optimised for point in result.growth] == pytest.approx(expected_growth)
    assert [point.equal_weight for point in result.growth] == pytest.approx(expected_growth)
    # A daily-rebalanced portfolio would finish at £1072.50, which is different.
    daily_returns = np.array([0, 0.1, -0.0325])
    expected_return = daily_returns.mean() * 252
    expected_volatility = daily_returns.std(ddof=1) * np.sqrt(252)
    assert result.optimised.metrics.annualised_return == pytest.approx(expected_return)
    assert result.optimised.metrics.volatility == pytest.approx(expected_volatility)
    assert result.optimised.metrics.sharpe == pytest.approx(
        (expected_return - 0.02) / expected_volatility
    )
    assert result.equal_weight.metrics == result.optimised.metrics
    assert result.equal_weight.weights == {ticker: 0.5 for ticker in TICKERS}
    dates = small_prices.index
    assert result.training_start == dates[0].date()
    assert result.training_end == dates[3].date()
    assert result.test_start == dates[4].date()
    assert result.test_end == dates[6].date()
    assert [point.date for point in result.growth] == [day.date() for day in dates[3:]]
    assert result.starting_value == 1000
    assert result.risk_free_rate == 0.02


def test_future_price_changes_cannot_change_training_allocation(monkeypatch, small_prices) -> None:
    seen = []

    def fit(request, training):
        seen.append(training.copy())
        return optimise_portfolio(request, training)

    monkeypatch.setattr("workbench.backtest.optimise_portfolio", fit)
    request = BacktestRequest(tickers=TICKERS, train_fraction=0.6)
    first = backtest_portfolio(request, small_prices)
    changed = small_prices.copy()
    changed.iloc[4:, 0] *= 10
    changed.iloc[4:, 1] *= 0.1
    second = backtest_portfolio(request, changed)
    assert first.optimised.weights == second.optimised.weights
    assert first.growth != second.growth
    for training in seen:
        pd.testing.assert_frame_equal(training, small_prices.iloc[:4], check_freq=False)
        assert training.index[-1].date() < first.test_start


def test_incompatible_equal_weights_are_omitted_with_reason(small_prices) -> None:
    result = backtest_portfolio(
        BacktestRequest(
            tickers=TICKERS,
            train_fraction=0.6,
            weight_bounds={"VWRP.L": (0.7, 0.7), "VAGP.L": (0.3, 0.3)},
        ),
        small_prices,
    )
    assert result.optimised.weights == {"VWRP.L": 0.7, "VAGP.L": 0.3}
    assert result.equal_weight is None
    assert "starting weight bounds" in result.baseline_unavailable_reason
    assert "VWRP.L" in result.baseline_unavailable_reason
    assert all(point.equal_weight is None for point in result.growth)


def test_zero_realised_volatility_returns_null_sharpe(small_prices) -> None:
    small_prices.iloc[4:] = small_prices.iloc[3].to_numpy()
    result = backtest_portfolio(BacktestRequest(tickers=TICKERS, train_fraction=0.6), small_prices)
    assert result.optimised.metrics.volatility == 0
    assert result.optimised.metrics.sharpe is None
    assert result.equal_weight.metrics.sharpe is None
    assert [point.optimised for point in result.growth] == pytest.approx([1000] * 4)
    # JSON must contain null, never NaN or Infinity.
    assert '"sharpe":null' in result.model_dump_json()


@pytest.mark.parametrize("fraction", [0.1, 0.95])
def test_insufficient_training_or_test_data_is_rejected(small_prices, fraction) -> None:
    with pytest.raises(ValueError, match="three training prices and two test prices"):
        backtest_portfolio(BacktestRequest(tickers=TICKERS, train_fraction=fraction), small_prices)


def test_missing_purchase_price_names_ticker_and_date(small_prices) -> None:
    small_prices.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing purchase prices on 2024-01-05 for VWRP.L"):
        backtest_portfolio(BacktestRequest(tickers=TICKERS, train_fraction=0.6), small_prices)


def test_gaps_are_visible_and_spanning_returns_are_not_treated_as_daily(small_prices) -> None:
    extra = pd.DataFrame(
        {"VWRP.L": [118, 120, 115], "VAGP.L": [207, 210, 214]},
        index=pd.bdate_range(small_prices.index[-1] + pd.offsets.BDay(), periods=3),
    )
    prices = pd.concat([small_prices, extra])
    prices.iloc[5, 0] = np.nan
    result = backtest_portfolio(
        BacktestRequest(
            tickers=TICKERS,
            train_fraction=0.4,
            weight_bounds={ticker: (0.5, 0.5) for ticker in TICKERS},
            risk_free_rate=0.02,
        ),
        prices,
    )
    assert [point.optimised for point in result.growth] == pytest.approx(
        [1000, 1000, 1064.25, 1107.5, 1125, 1110]
    )
    assert prices.index[5].date() not in [point.date for point in result.growth]
    daily = np.array([0, 1107.5 / 1064.25 - 1, 1125 / 1107.5 - 1, 1110 / 1125 - 1])
    assert result.optimised.metrics.annualised_return == pytest.approx(daily.mean() * 252)
    assert result.optimised.metrics.volatility == pytest.approx(daily.std(ddof=1) * np.sqrt(252))
    assert result.equal_weight.metrics == result.optimised.metrics
    assert "VWRP.L" in result.data_warnings[0]
    assert str(prices.index[5].date()) in result.data_warnings[0]


def test_incomplete_latest_quote_uses_last_observed_test_close(small_prices) -> None:
    small_prices.iloc[-1, 0] = np.nan
    result = backtest_portfolio(BacktestRequest(tickers=TICKERS, train_fraction=0.6), small_prices)
    assert result.test_end == small_prices.index[-2].date()
    assert result.growth[-1].date == result.test_end
    assert result.data_warnings


def test_gap_leaving_too_few_daily_returns_is_a_clear_error(small_prices) -> None:
    small_prices.iloc[4, 0] = np.nan
    with pytest.raises(ValueError, match="two daily returns.*longer historical window"):
        backtest_portfolio(BacktestRequest(tickers=TICKERS, train_fraction=0.6), small_prices)


def test_missing_future_quotes_do_not_change_the_training_allocation(prices) -> None:
    request = BacktestRequest(tickers=TICKERS)
    before = backtest_portfolio(request, prices)
    split = int(len(prices) * request.train_fraction)
    prices.iloc[split + 5, 0] = np.nan
    after = backtest_portfolio(request, prices)
    assert after.training_end == before.training_end
    assert after.optimised.weights == before.optimised.weights


def test_single_asset_baseline_matches_and_first_test_return_is_included(small_prices) -> None:
    result = backtest_portfolio(
        BacktestRequest(tickers=["VWRP.L"], train_fraction=0.6), small_prices
    )
    assert result.optimised == result.equal_weight
    assert [point.optimised for point in result.growth] == pytest.approx([1000, 1100, 1210, 1089])
    assert result.optimised.metrics.annualised_return == pytest.approx((0.1 + 0.1 - 0.1) / 3 * 252)


def test_backtest_orders_prices_chronologically(small_prices) -> None:
    request = BacktestRequest(tickers=TICKERS, train_fraction=0.6)
    assert backtest_portfolio(request, small_prices.iloc[::-1]) == backtest_portfolio(
        request, small_prices
    )
