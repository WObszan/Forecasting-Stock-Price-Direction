import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_direction.data import load_prices, merge_available_data, validate_dates
from stock_direction.features import FEATURE_COLUMNS, build_features


def test_features_are_unchanged_when_future_prices_are_appended(prices):
    prefix = build_features(prices.iloc[:400], "AAPL")
    full = build_features(prices, "AAPL").iloc[:400]
    assert_frame_equal(prefix, full)
    assert np.isfinite(full[list(FEATURE_COLUMNS)].iloc[60:].to_numpy()).all()
    assert len(prefix) == 400  # Keep the complete session calendar for labeling.


def test_future_price_changes_do_not_change_past_features(prices):
    original = build_features(prices, "AAPL")
    changed = prices.copy()
    changed.loc[400:, ["Close_AAPL", "High_AAPL", "Low_AAPL", "Open_AAPL"]] *= 2
    modified = build_features(changed, "AAPL")
    assert_frame_equal(original.iloc[:400], modified.iloc[:400])


def test_loader_sorts_and_rejects_invalid_prices(prices, tmp_path):
    path = tmp_path / "prices.csv"
    prices.iloc[::-1].to_csv(path, index=False)
    assert_frame_equal(load_prices(path, "AAPL"), prices, check_dtype=False, check_like=True)
    prices.loc[0, "Close_AAPL"] = -1
    prices.to_csv(path, index=False)
    with pytest.raises(ValueError, match="positive"):
        load_prices(path, "AAPL")


def test_duplicate_dates_are_rejected(prices):
    with pytest.raises(ValueError, match="unique"):
        validate_dates(pd.concat([prices, prices.iloc[:1]]))


def test_macro_uses_availability_not_observation_date():
    sessions = pd.DataFrame({"DATE": pd.to_datetime(["2020-02-03", "2020-02-04", "2020-02-05"])})
    releases = pd.DataFrame({"Available_At": ["2020-02-04"], "CPI": [200.0]})
    joined = merge_available_data(sessions, releases)
    assert joined["CPI"].iloc[:2].isna().all()
    assert joined["CPI"].iloc[2] == 200.0


def test_macro_cannot_silently_overwrite_features():
    left = pd.DataFrame({"DATE": ["2020-02-05"], "CPI": [100]})
    right = pd.DataFrame({"Available_At": ["2020-02-04"], "CPI": [200]})
    with pytest.raises(ValueError, match="overlap"):
        merge_available_data(left, right)
