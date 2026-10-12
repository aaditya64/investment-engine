from datetime import date
from pathlib import Path

import pytest

from workbench.data.providers.fixture import FixtureProvider

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prices_fixture.csv"


def test_loads_requested_tickers_and_date_window() -> None:
    provider = FixtureProvider(FIXTURE_PATH)
    frame = provider.get_prices(["VWRP.L", "VAGP.L"], date(2022, 1, 4), date(2022, 1, 31))
    assert list(frame.columns) == ["VWRP.L", "VAGP.L"]
    assert frame.index.min().date() >= date(2022, 1, 4)
    assert frame.index.max().date() <= date(2022, 1, 31)
    assert not frame.empty


def test_raises_on_unknown_ticker() -> None:
    provider = FixtureProvider(FIXTURE_PATH)
    with pytest.raises(KeyError, match="NOT_IN_FIXTURE"):
        provider.get_prices(["NOT_IN_FIXTURE"], date(2022, 1, 1), date(2022, 12, 31))


def test_raises_on_empty_window() -> None:
    provider = FixtureProvider(FIXTURE_PATH)
    with pytest.raises(ValueError, match="no rows"):
        provider.get_prices(["VWRP.L"], date(1990, 1, 1), date(1990, 1, 31))
