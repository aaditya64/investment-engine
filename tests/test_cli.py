from pathlib import Path

from workbench import __version__
from workbench.cli import main

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prices_fixture.csv"


def test_help_exits_cleanly() -> None:
    exit_code = main([])
    assert exit_code == 0


def test_version_command_prints_version(capsys) -> None:
    exit_code = main(["version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert __version__ in captured.out


def _metrics_args(*extra: str) -> list[str]:
    return [
        "metrics",
        "--tickers",
        "VWRP.L",
        "VAGP.L",
        "--weights",
        "0.7",
        "0.3",
        "--provider",
        "fixture",
        "--fixture-path",
        str(FIXTURE_PATH),
        *extra,
    ]


def test_metrics_with_fixture_provider_succeeds(capsys) -> None:
    exit_code = main(_metrics_args())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Expected return" in captured.out
    assert "not financial advice" in captured.out


def test_metrics_with_synthetic_provider_labels_output_as_synthetic(capsys) -> None:
    exit_code = main(
        ["metrics", "--tickers", "A", "B", "--weights", "0.6", "0.4", "--provider", "synthetic"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SYNTHETIC" in captured.out


def test_metrics_rejects_mismatched_tickers_and_weights(capsys) -> None:
    exit_code = main(_metrics_args())
    assert exit_code == 0

    exit_code = main(
        [
            "metrics",
            "--tickers",
            "VWRP.L",
            "VAGP.L",
            "--weights",
            "0.7",
            "--provider",
            "fixture",
            "--fixture-path",
            str(FIXTURE_PATH),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err


def test_metrics_rejects_weights_not_summing_to_one(capsys) -> None:
    exit_code = main(
        [
            "metrics",
            "--tickers",
            "VWRP.L",
            "VAGP.L",
            "--weights",
            "0.7",
            "0.1",
            "--provider",
            "fixture",
            "--fixture-path",
            str(FIXTURE_PATH),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "sum to 1" in captured.err


def test_metrics_with_fixture_requires_fixture_path(capsys) -> None:
    exit_code = main(
        ["metrics", "--tickers", "VWRP.L", "--weights", "1.0", "--provider", "fixture"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--fixture-path" in captured.err


def test_metrics_reports_unknown_ter_when_not_supplied(capsys) -> None:
    exit_code = main(_metrics_args())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "unknown" in captured.out


def test_metrics_uses_supplied_ter_overrides(capsys) -> None:
    exit_code = main(_metrics_args("--ter", "VWRP.L=0.0022", "VAGP.L=0.0025"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "unknown" not in captured.out
