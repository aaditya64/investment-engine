from pathlib import Path

import pytest

from workbench.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "prices_fixture.csv"


def test_optimize_offline_reports_agreed_outputs(capsys) -> None:
    assert (
        main(
            [
                "optimize",
                "--tickers",
                "VWRP.L",
                "VAGP.L",
                "--prices-csv",
                str(FIXTURE),
                "--lookback-years",
                "30",
                "--risk-free-rate",
                "0.02",
                "--bound",
                "VWRP.L=0.1,0.6",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    for label in (
        "VWRP.L",
        "VAGP.L",
        "Historical annualised return",
        "Annualised volatility",
        "Sharpe ratio",
        "Annual risk-free rate",
        "2.00%",
    ):
        assert label in output
    for removed in ("Sortino", "cash", "exposure", "Conviction"):
        assert removed not in output


@pytest.mark.parametrize("bound", ["bad-format", "VWRP.L=0.8,0.1", "VWRP.L=0,0.2"])
def test_bad_bounds_return_error_before_network(capsys, monkeypatch, bound) -> None:
    def unexpected(*_):
        pytest.fail("invalid bounds must not fetch data")

    monkeypatch.setattr("workbench.cli.get_prices", unexpected)
    assert main(["optimize", "--tickers", "VWRP.L", "--bound", bound]) == 1
    assert "error:" in capsys.readouterr().err


def test_help(capsys) -> None:
    assert main([]) == 0
    assert "optimize" in capsys.readouterr().out
