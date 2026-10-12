import pandas as pd
import pytest

from workbench.analytics.expected_returns import shrink_to_grand_mean


def test_zero_shrinkage_keeps_raw_means() -> None:
    raw = pd.Series({"A": 0.10, "B": 0.02})
    shrunk = shrink_to_grand_mean(raw, shrinkage=0.0)
    assert shrunk["A"] == pytest.approx(0.10)
    assert shrunk["B"] == pytest.approx(0.02)


def test_full_shrinkage_collapses_to_grand_mean() -> None:
    raw = pd.Series({"A": 0.10, "B": 0.02})
    shrunk = shrink_to_grand_mean(raw, shrinkage=1.0)
    assert shrunk["A"] == pytest.approx(0.06)
    assert shrunk["B"] == pytest.approx(0.06)


def test_partial_shrinkage_pulls_toward_grand_mean() -> None:
    raw = pd.Series({"A": 0.10, "B": 0.02})
    shrunk = shrink_to_grand_mean(raw, shrinkage=0.5)
    assert raw["A"] > shrunk["A"] > 0.06
    assert raw["B"] < shrunk["B"] < 0.06


def test_rejects_shrinkage_out_of_range() -> None:
    with pytest.raises(ValueError, match="shrinkage must be"):
        shrink_to_grand_mean(pd.Series({"A": 0.1}), shrinkage=1.5)
