"""Optional reproducible Optuna search confined to each outer training fold."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from .validation import PurgedTimeSeriesSplit


def tune_model(estimator, X, y, dates, label_ends, *, n_trials=10, random_state=42, gap=0):
    import optuna

    classifier = estimator.named_steps["classifier"]
    params = classifier.get_params()
    if "estimator__svc__C" in params:

        def suggest(trial):
            return {
                "classifier__estimator__svc__C": trial.suggest_float(
                    "estimator__svc__C",
                    0.01,
                    10,
                    log=True,
                )
            }
    elif "C" in params:

        def suggest(trial):
            return {"classifier__C": trial.suggest_float("C", 0.01, 10, log=True)}
    elif "min_samples_leaf" in params:

        def suggest(trial):
            return {
                "classifier__max_depth": trial.suggest_int("max_depth", 2, 8),
                "classifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            }
    else:
        raise ValueError("Optional tuning currently supports lr, rf and svm")

    cv = PurgedTimeSeriesSplit(dates, label_ends, n_splits=3, gap=gap)

    def objective(trial):
        candidate = clone(estimator).set_params(**suggest(trial))
        return float(
            np.mean(
                cross_val_score(
                    candidate,
                    X,
                    y,
                    cv=cv,
                    scoring="balanced_accuracy",
                    error_score="raise",
                    n_jobs=1,
                )
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    # Reuse parameter names without invoking a second search or a fresh trial.
    best = {f"classifier__{name}": value for name, value in study.best_params.items()}
    return clone(estimator).set_params(**best)
