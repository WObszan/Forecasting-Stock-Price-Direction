import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from stock_direction.cli import main
from stock_direction.models import build_model, selected_feature_names
from stock_direction.tuning import tune_model


@pytest.mark.parametrize(
    "name", ["lr", "rf", "svm", "always_up", "majority", "random", "xgb", "ensemble"]
)
@pytest.mark.parametrize("classes", [[0, 1], [-1, 0, 1]])
def test_models_support_probability_and_original_labels(name, classes):
    if name in ("xgb", "ensemble"):
        pytest.importorskip("xgboost")
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(90, 3)))
    y = np.resize(classes, len(X))
    model = build_model(name)
    model.fit(X, y)
    assert set(model.predict(X)).issubset(classes)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1, atol=1e-6)
    assert list(model.classes_) == classes
    if name == "always_up":
        assert (model.predict(X) == 1).all()


def test_tuning_runs_inside_purged_temporal_cv():
    pytest.importorskip("optuna")
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(120, 3)))
    y = np.resize([0, 1], len(X))
    dates = pd.bdate_range("2020-01-01", periods=len(X))
    ends = dates + pd.offsets.BDay(5)
    model = tune_model(build_model("lr", select_k=2), X, y, dates, ends, n_trials=1)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (120, 2)


def test_svm_calibration_preprocessing_uses_only_its_own_training_prefix():
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(120, 3)), columns=["a", "b", "c"])
    y = np.resize([0, 1], len(X))
    model = build_model("svm", select_k=2, calibration_gap=5).fit(X, y)
    calibrated = model.named_steps["classifier"]
    splits = list(calibrated.cv.split(X))
    for member, (train, test) in zip(calibrated.calibrated_classifiers_, splits):
        assert train.max() + 5 < test.min()
        np.testing.assert_allclose(
            member.estimator.named_steps["scale"].mean_, X.iloc[train].mean()
        )
    assert 2 <= len(selected_feature_names(model, list(X.columns))) <= 3


@pytest.mark.parametrize("extra_models", [[], ["xgb", "ensemble"]])
def test_cli_creates_comparable_reproducible_reports_and_preserves_input(
    prices, tmp_path, extra_models
):
    if extra_models:
        pytest.importorskip("xgboost")
    source = tmp_path / "historical_AAPL_data.csv"
    prices.to_csv(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "run"
    args = [
        "--data-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--start-year",
        "2013",
        "--first-train-end-year",
        "2015",
        "--last-test-year",
        "2017",
    ]
    if extra_models:
        args.extend(["--models", "lr", "majority", "always_up", "random", *extra_models])
    assert main(args) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["input_sha256"] == before
    assert metadata["macro_included"] is False
    assert "cli.py" in metadata["source_sha256"]
    parameters = json.loads((output / "lr" / "model_parameters.json").read_text())
    assert parameters["2016"]["classifier__random_state"] == 42
    comparison = pd.read_csv(output / "comparison.csv")
    assert comparison["n_predictions"].nunique() == 1
    majority = comparison.loc[comparison["model"] == "majority"].iloc[0]
    assert majority["paired_delta_ci_low"] == majority["paired_delta_ci_high"] == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(SystemExit):
        main(args)
