import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from stock_direction.features import FEATURE_COLUMNS, build_features
from stock_direction.models import build_model
from stock_direction.targets import get_target, get_tbm_target
from stock_direction.validation import (
    PurgedTimeSeriesSplit,
    WalkForwardConfig,
    aligned_probabilities,
    block_bootstrap_accuracy,
    classification_metrics,
    walk_forward_validation,
)


def test_inner_split_purges_label_overlap_even_after_session_filtering():
    dates = pd.bdate_range("2020-01-01", periods=100)
    ends = dates + pd.offsets.BDay(5)
    keep = np.arange(0, 95, 3)
    dates, ends = dates[keep], ends[keep]
    splitter = PurgedTimeSeriesSplit(dates, ends, gap=1)
    for train, test in splitter.split(np.ones((len(dates), 1))):
        assert (ends[train] < dates[test[0]]).all()
        assert train.max() < test.min() - 1


def test_outer_purging_uses_event_dates_and_does_not_discard_test_rows(prices):
    frame = get_tbm_target(build_features(prices, "AAPL"), "AAPL", horizon=5)
    result = walk_forward_validation(
        frame,
        list(FEATURE_COLUMNS),
        build_model("majority"),
        target_type="tbm",
        config=WalkForwardConfig(2013, 2015, 2017, gap=2),
    )
    for row in result.folds.itertuples():
        assert row.train_last_label_end < pd.Timestamp(row.test_year, 1, 1)
        expected = frame.loc[frame["DATE"].dt.year == row.test_year]
        assert row.n_test == len(expected)
        assert row.n_purged_or_gap >= 2
    assert result.predictions["Date"].is_unique
    np.testing.assert_allclose(
        result.predictions[["Proba_-1", "Proba_0", "Proba_1"]].sum(axis=1), 1
    )


def test_adding_future_data_does_not_change_existing_folds(prices):
    frame = get_target(build_features(prices, "AAPL"), "AAPL")
    config = WalkForwardConfig(2013, 2015, 2016)
    short = frame.loc[frame["DATE"] < "2017-01-01"]
    a = walk_forward_validation(short, list(FEATURE_COLUMNS), build_model("lr"), config=config)
    b = walk_forward_validation(frame, list(FEATURE_COLUMNS), build_model("lr"), config=config)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    pd.testing.assert_frame_equal(a.folds, b.folds)


def test_test_targets_do_not_affect_feature_selection(prices):
    frame = get_target(build_features(prices, "AAPL"), "AAPL")
    config = WalkForwardConfig(2013, 2015, 2016)
    altered = frame.copy()
    mask = altered["DATE"].dt.year == 2016
    altered.loc[mask, "Target"] = 1 - altered.loc[mask, "Target"]
    model = build_model("lr", select_k=5)
    a = walk_forward_validation(frame, list(FEATURE_COLUMNS), model, config=config)
    b = walk_forward_validation(altered, list(FEATURE_COLUMNS), model, config=config)
    assert a.selected_features == b.selected_features
    np.testing.assert_array_equal(a.predictions["Predicted"], b.predictions["Predicted"])


def test_probability_columns_follow_semantic_classes_when_class_is_absent():
    X = np.ones((10, 2))
    model = DummyClassifier(strategy="most_frequent").fit(X, [0, 1] * 5)
    proba = aligned_probabilities(model, X, [-1, 0, 1])
    assert (proba[:, 0] == 0).all()
    assert (proba[:, 1] == 1).all()


def test_auc_is_computed_from_probability_and_single_class_is_undefined():
    actual, predicted = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    assert classification_metrics(actual, predicted, proba, [0, 1])["roc_auc"] == 1
    metrics = classification_metrics(actual[:2], predicted[:2], proba[:2], [0, 1])
    assert np.isnan(metrics["roc_auc"])
    actual_tbm = np.array([-1, -1, 1, 1])
    assert np.isnan(
        classification_metrics(actual_tbm, actual_tbm, np.ones((4, 3)) / 3, [-1, 0, 1])["roc_auc"]
    )


def test_target_metadata_is_rejected_as_features(prices):
    frame = get_target(build_features(prices, "AAPL"), "AAPL")
    model = Pipeline([("classifier", DummyClassifier())])
    with pytest.raises(ValueError, match="metadata"):
        walk_forward_validation(frame, ["Forward_Log_Return"], model)


def test_empty_evaluation_has_explicit_error(prices):
    frame = get_target(build_features(prices, "AAPL"), "AAPL")
    with pytest.raises(ValueError, match="No valid"):
        walk_forward_validation(
            frame,
            list(FEATURE_COLUMNS),
            build_model("lr"),
            config=WalkForwardConfig(2020, 2021, 2023),
        )


def test_bootstrap_accepts_series_with_nonpositional_index_and_is_reproducible():
    actual = pd.Series([0, 1] * 20, index=np.arange(100, 140))
    predicted = actual.to_list()
    a, low, high = block_bootstrap_accuracy(actual, predicted, n_bootstrap=20)
    b, _, _ = block_bootstrap_accuracy(actual.to_numpy(), predicted, n_bootstrap=20)
    np.testing.assert_array_equal(a, b)
    assert low == high == 1
    with pytest.raises(ValueError):
        block_bootstrap_accuracy(actual, predicted[:-1])
    with pytest.raises(ValueError):
        block_bootstrap_accuracy(actual, predicted, block_size=0)
