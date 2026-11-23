import numpy as np
import pandas as pd
import pytest

from stock_direction.targets import get_target, get_tbm_target


def test_binary_excludes_unknown_last_label_and_aligns_returns(prices):
    labeled = get_target(prices, "AAPL")
    assert len(labeled) == len(prices) - 1
    np.testing.assert_array_equal(labeled["Label_End"], prices["DATE"].iloc[1:])
    expected = np.log(prices["Close_AAPL"].shift(-1) / prices["Close_AAPL"]).iloc[:-1]
    np.testing.assert_allclose(labeled["Forward_Log_Return"], expected)
    np.testing.assert_array_equal(labeled["Target"], (expected > 0).astype(int))


def test_flat_prices_stay_in_unfiltered_binary_calendar(prices):
    prices["Close_AAPL"] = 100.0
    labeled = get_target(prices, "AAPL")
    assert len(labeled) == len(prices) - 1
    assert (labeled["Target"] == 0).all()
    assert get_target(prices, "AAPL", barrier_std=0.5).empty


def test_tbm_warmup_and_incomplete_horizon_are_not_neutral_labels(prices):
    labeled = get_tbm_target(prices, "AAPL", horizon=5)
    assert labeled["DATE"].min() == prices["DATE"].iloc[20]
    assert labeled["DATE"].max() == prices["DATE"].iloc[-6]
    assert labeled["Label_End"].notna().all()
    assert (labeled["Label_End"] > labeled["DATE"]).all()


@pytest.mark.parametrize("last, label", [(120, 1), (80, -1), (101, 0)])
def test_tbm_records_first_barrier_hit_or_timeout(last, label):
    frame = pd.DataFrame(
        {
            "DATE": pd.bdate_range("2020-01-01", periods=6),
            "Close_AAPL": [100, 102, 100, last, 100, 100],
        }
    )
    labeled = get_tbm_target(frame, "AAPL", horizon=2, vol_window=2, pt_sl=(1, 1))
    row = labeled.loc[labeled["DATE"] == frame["DATE"].iloc[2]].iloc[0]
    assert row["Target"] == label
    expected_end = 4 if label == 0 else 3
    assert row["Label_End"] == frame["DATE"].iloc[expected_end]


def test_tbm_zero_volatility_does_not_create_artificial_profit_labels(prices):
    prices["Close_AAPL"] = 100.0
    assert get_tbm_target(prices, "AAPL").empty


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"pt_sl": (-1, 1)}, {"vol_window": 1}])
def test_invalid_tbm_parameters_fail(prices, kwargs):
    with pytest.raises(ValueError):
        get_tbm_target(prices, "AAPL", **kwargs)


def test_filter_keeps_original_event_dates(prices):
    labeled = get_target(prices, "AAPL", barrier_std=1)
    next_dates = pd.Series(prices["DATE"].shift(-1).to_numpy(), index=prices["DATE"])
    assert len(labeled) < len(prices) - 1
    np.testing.assert_array_equal(labeled["Label_End"], next_dates.loc[labeled["DATE"]])
