from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    """Recorded prices; no test needs a market-data connection."""
    return pd.read_csv(
        Path(__file__).parent / "fixtures" / "prices_fixture.csv",
        index_col="Date",
        parse_dates=True,
    )
