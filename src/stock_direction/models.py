"""Model factories. Every learned transform is fitted inside its training fold."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

MODEL_NAMES = ("lr", "rf", "svm", "xgb", "ensemble", "always_up", "majority", "random")


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapt classifiers requiring consecutive labels to original TBM labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X), dtype=int))

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)


def build_model(
    name: str = "lr",
    *,
    select_k: int | None = None,
    random_state: int = 42,
    calibration_gap: int = 5,
):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if select_k is not None and (not isinstance(select_k, int) or select_k < 1):
        raise ValueError("select_k must be a positive integer")
    if not isinstance(calibration_gap, int) or calibration_gap < 1:
        raise ValueError("calibration_gap must be a positive integer covering the label horizon")
    if name == "lr":
        classifier = LogisticRegression(max_iter=2000, random_state=random_state)
    elif name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "svm":
        # Calibrate chronologically. Preprocessing and selection are part of
        # each calibration fold, so its validation labels cannot select features.
        base_steps = [("scale", StandardScaler())]
        if select_k is not None:
            base_steps.append(("select", SelectKBest(f_classif, k=select_k)))
        base_steps.append(("svc", SVC(C=1.0, class_weight="balanced", random_state=random_state)))
        classifier = CalibratedClassifierCV(
            Pipeline(base_steps),
            method="sigmoid",
            ensemble=True,
            cv=TimeSeriesSplit(n_splits=3, gap=calibration_gap),
        )
        return Pipeline([("classifier", classifier)])
    elif name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install stock-direction[boosting] to use XGBoost") from exc
        classifier = EncodedClassifier(
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=1,
                random_state=random_state,
            )
        )
    elif name == "ensemble":
        # VotingClassifier encodes and restores its original labels as well.
        classifier = VotingClassifier(
            estimators=[
                (
                    key,
                    build_model(
                        key,
                        select_k=select_k,
                        random_state=random_state,
                        calibration_gap=calibration_gap,
                    ),
                )
                for key in ("lr", "rf", "svm", "xgb")
            ],
            voting="soft",
            n_jobs=1,
        )
        return Pipeline([("classifier", classifier)])
    elif name == "always_up":
        # +1 is profit/up for BOTH raw TBM (-1, 0, +1) and binary (0, 1).
        classifier = DummyClassifier(strategy="constant", constant=1)
    elif name == "majority":
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = DummyClassifier(strategy="stratified", random_state=random_state)

    steps = [("scale", StandardScaler())]
    if select_k is not None:
        steps.append(("select", SelectKBest(f_classif, k=select_k)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def selected_feature_names(model, features):
    """Union of features used by fitted pipelines, voters and calibration folds."""
    if isinstance(model, Pipeline):
        remaining = list(features)
        for name, step in model.steps:
            if name == "select":
                remaining = list(np.asarray(remaining)[step.get_support()])
        return selected_feature_names(model.steps[-1][1], remaining)
    if isinstance(model, VotingClassifier):
        members = model.estimators_
    elif isinstance(model, CalibratedClassifierCV):
        members = [member.estimator for member in model.calibrated_classifiers_]
    else:
        return list(features)
    used = {name for member in members for name in selected_feature_names(member, features)}
    return [name for name in features if name in used]
