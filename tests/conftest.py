import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    dates = pd.bdate_range("2013-01-01", "2018-12-31")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(dates))))
    opening = close * np.exp(rng.normal(0, 0.003, len(dates)))
    return pd.DataFrame(
        {
            "DATE": dates,
            "Close_AAPL": close,
            "Open_AAPL": opening,
            "High_AAPL": np.maximum(opening, close) * 1.01,
            "Low_AAPL": np.minimum(opening, close) * 0.99,
            "Volume_AAPL": rng.integers(1000, 10000, len(dates)),
        }
    )
