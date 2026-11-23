"""Walk-forward evaluation and purged inner CV on actual label event dates."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit

from .data import validate_dates
from .models import selected_feature_names
from .targets import LABEL_METADATA


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Expanding inner splits with no training label reaching validation start."""

    def __init__(self, dates, label_ends, n_splits=3, gap=0):
        self.dates = pd.DatetimeIndex(dates)
        self.label_ends = pd.DatetimeIndex(label_ends)
        self.n_splits = n_splits
        self.gap = gap
        if len(self.dates) != len(self.label_ends):
            raise ValueError("dates and label_ends must have the same length")
        if (
            self.dates.hasnans
            or self.label_ends.hasnans
            or not self.dates.is_monotonic_increasing
            or not self.dates.is_unique
            or (self.label_ends <= self.dates).any()
        ):
            raise ValueError("Dates must be ordered and unique, with a later end for each label")

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.dates):
            raise ValueError("CV dates must align with X")
        for train, test in TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap).split(X):
            train = train[self.label_ends[train] < self.dates[test[0]]]
            if not len(train):
                raise ValueError("No training observations remain after inner purging")
            yield train, test


@dataclass(frozen=True)
class WalkForwardConfig:
    start_year: int = 2010
    first_train_end_year: int = 2015
    last_test_year: int = 2023
    gap: int = 0
    min_train: int = 200
    min_test: int = 10

    def __post_init__(self):
        if not self.start_year <= self.first_train_end_year < self.last_test_year:
            raise ValueError("Require start_year <= first_train_end_year < last_test_year")
        if self.gap < 0 or self.min_train < 1 or self.min_test < 1:
            raise ValueError("gap must be nonnegative; sample minima must be positive")


@dataclass
class EvaluationResult:
    folds: pd.DataFrame
    predictions: pd.DataFrame
    selected_features: dict[str, list[str]]
    model_parameters: dict[str, dict]


def aligned_probabilities(model, X, classes):
    """Keep a stable column for each semantic class, even when training misses one."""
    raw = np.asarray(model.predict_proba(X))
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for source, label in enumerate(model.classes_):
        if label not in classes:
            raise ValueError(f"Unexpected predicted class {label}")
        aligned[:, classes.index(label)] = raw[:, source]
    if not np.isfinite(aligned).all() or not np.allclose(aligned.sum(axis=1), 1):
        raise ValueError("Predicted probabilities must be finite and sum to one")
    return aligned


def classification_metrics(y_true, y_pred, probabilities, classes):
    """AUC uses probabilities; undefined AUC is NaN, never a made-up zero."""
    binary = classes == [0, 1]
    kwargs = {"pos_label": 1} if binary else {"average": "weighted"}
    auc = np.nan
    if set(np.unique(y_true)) == set(classes):
        if binary:
            auc = roc_auc_score(y_true, probabilities[:, classes.index(1)])
        else:
            auc = roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                average="weighted",
                multi_class="ovr",
            )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, y_pred)
            if len(np.unique(y_true)) > 1
            else accuracy_score(y_true, y_pred)
        ),
        "precision": precision_score(y_true, y_pred, zero_division=0, **kwargs),
        "recall": recall_score(y_true, y_pred, zero_division=0, **kwargs),
        "f1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "roc_auc": auc,
    }


def walk_forward_validation(
    df: pd.DataFrame,
    features: list[str],
    estimator,
    *,
    target_type: str = "binary",
    config: WalkForwardConfig | None = None,
    n_trials: int = 0,
    random_state: int = 42,
) -> EvaluationResult:
    """Evaluate every eligible test session; purge training labels before each year.

    gap is an optional number of additional TRAINING observations removed after
    purging. Its size never depends on the eventual dataset length. There is no
    post-test embargo in expanding-window evaluation because training contains
    only past observations. Selection and optional tuning run inside each fold.
    """
    config = config or WalkForwardConfig()
    if target_type not in ("binary", "tbm"):
        raise ValueError("target_type must be 'binary' or 'tbm'")
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide a nonempty list of distinct features")
    forbidden = set(features) & LABEL_METADATA
    if forbidden or any("target" in name.lower() or "forward" in name.lower() for name in features):
        raise ValueError(f"Target/date metadata cannot be used as features: {features}")
    if n_trials < 0:
        raise ValueError("n_trials must be nonnegative")
    df = validate_dates(df)
    df["Label_End"] = pd.to_datetime(df["Label_End"], errors="raise")
    if df["Label_End"].isna().any() or (df["Label_End"] <= df["DATE"]).any():
        raise ValueError("Each observation must have a label end later than its date")
    classes = [0, 1] if target_type == "binary" else [-1, 0, 1]
    if df["Target"].isna().any() or not set(df["Target"].unique()).issubset(classes):
        raise ValueError(f"Expected target labels from {classes}")
    # Do not fill missing values using information from a later observation.
    numeric = df[features].to_numpy(dtype=float)
    usable = np.isfinite(numeric).all(axis=1)
    years = df["DATE"].dt.year
    fold_rows, prediction_frames, selections, parameters = [], [], {}, {}
    for test_year in range(config.first_train_end_year + 1, config.last_test_year + 1):
        boundary = pd.Timestamp(year=test_year, month=1, day=1)
        eligible_train = usable & (years >= config.start_year) & (years < test_year)
        train = df.loc[eligible_train & (df["Label_End"] < boundary)]
        if config.gap:
            train = train.iloc[: -config.gap]
        test = df.loc[usable & (years == test_year)]
        if len(train) < config.min_train or len(test) < config.min_test:
            continue
        X_train, y_train = train[features], train["Target"]
        if y_train.nunique() < 2:
            raise ValueError(f"Training fold for {test_year} contains fewer than two classes")
        model = clone(estimator)
        if n_trials:
            from .tuning import tune_model

            model = tune_model(
                model,
                X_train,
                y_train,
                train["DATE"],
                train["Label_End"],
                n_trials=n_trials,
                random_state=random_state,
                gap=config.gap,
            )
        model.fit(X_train, y_train)
        selections[str(test_year)] = selected_feature_names(model, features)
        parameters[str(test_year)] = {
            key: value
            for key, value in model.get_params(deep=True).items()
            if value is None or isinstance(value, (str, bool, int, float))
        }
        pred = model.predict(test[features])
        proba = aligned_probabilities(model, test[features], classes)
        metrics = classification_metrics(test["Target"], pred, proba, classes)
        fold_rows.append(
            {
                "train_end_year": test_year - 1,
                "test_year": test_year,
                "n_train": len(train),
                "n_test": len(test),
                "n_purged_or_gap": int(eligible_train.sum()) - len(train),
                "train_last_date": train["DATE"].max(),
                "train_last_label_end": train["Label_End"].max(),
                "test_first_date": test["DATE"].min(),
                **metrics,
            }
        )
        detail = pd.DataFrame(
            {
                "Date": test["DATE"].to_numpy(),
                "Label_End": test["Label_End"].to_numpy(),
                "Actual": test["Target"].to_numpy(),
                "Predicted": pred,
            }
        )
        if "Forward_Log_Return" in test:
            detail["Forward_Log_Return"] = test["Forward_Log_Return"].to_numpy()
        for col, label in enumerate(classes):
            detail[f"Proba_{label}"] = proba[:, col]
        prediction_frames.append(detail)
    if not fold_rows:
        raise ValueError(
            "No valid walk-forward folds; check dates, feature warm-up and sample minima"
        )
    return EvaluationResult(
        pd.DataFrame(fold_rows),
        pd.concat(prediction_frames, ignore_index=True),
        selections,
        parameters,
    )


def block_bootstrap_accuracy(y_true, y_pred, block_size=20, n_bootstrap=1000, random_state=42):
    """Moving-block percentile interval for chronological predictions.

    This interval is descriptive; it is not a test of superiority to a baseline.
    Filtering sessions changes the time interpretation of each block.
    """
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if not isinstance(block_size, int) or not 1 <= block_size <= len(actual):
        raise ValueError("block_size must be between one and the number of observations")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    rng = np.random.default_rng(random_state)
    correct = (actual == predicted).astype(float)
    n = len(correct)
    blocks = (n + block_size - 1) // block_size
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=blocks)
        indices = (starts[:, None] + np.arange(block_size)).ravel()[:n]
        samples[i] = correct[indices].mean()
    low, high = np.percentile(samples, [2.5, 97.5])
    return samples, float(low), float(high)
