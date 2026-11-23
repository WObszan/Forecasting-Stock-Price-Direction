"""Run comparable, offline experiments from versioned price snapshots."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prices
from .features import FEATURE_COLUMNS, build_features
from .models import MODEL_NAMES, build_model
from .news import NEWS_FEATURE_COLUMNS, add_news_features, load_scored_news
from .targets import get_target, get_tbm_target
from .validation import (
    WalkForwardConfig,
    block_bootstrap_accuracy,
    classification_metrics,
    walk_forward_validation,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ticker", choices=("AAPL", "GOOGL", "MSFT"), default="AAPL")
    result.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["lr", "majority", "always_up", "random"],
    )
    result.add_argument("--target", choices=("binary", "tbm"), default="binary")
    result.add_argument("--data-dir", type=Path, default=Path("data/raw/prices"))
    result.add_argument("--output-dir", type=Path, help="New directory; existing paths are refused")
    result.add_argument("--start-year", type=int, default=2010)
    result.add_argument("--first-train-end-year", type=int, default=2015)
    result.add_argument("--last-test-year", type=int, default=2023)
    result.add_argument(
        "--gap", type=int, default=0, help="Extra training rows removed after purging"
    )
    result.add_argument("--horizon", type=int, default=5, help="TBM horizon in trading sessions")
    result.add_argument(
        "--barrier-std",
        type=float,
        default=0,
        help="Binary future-move filter; nonzero means conditional evaluation",
    )
    result.add_argument("--select-k", type=int, help="Select K features inside each training fold")
    result.add_argument(
        "--tuning-trials", type=int, default=0, help="Inner purged CV; lr/rf/svm only"
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--news-csv", type=Path, help="Prepared scored news; compares prices vs prices+news"
    )
    result.add_argument("--news-latency-minutes", type=float, default=5)
    return result


def _versions():
    versions = {"python": platform.python_version()}
    for package in (
        "stock-direction",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "optuna",
        "vaderSentiment",
        "exchange-calendars",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def _git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.target == "tbm" and args.barrier_std != 0:
        arg_parser.error("--barrier-std only applies to binary labels")
    if args.select_k is not None and not 1 <= args.select_k <= len(FEATURE_COLUMNS):
        arg_parser.error(f"--select-k must be between 1 and {len(FEATURE_COLUMNS)}")
    if args.tuning_trials < 0:
        arg_parser.error("--tuning-trials cannot be negative")
    if args.tuning_trials and set(args.models).difference({"lr", "rf", "svm"}):
        arg_parser.error(
            "Tuning supports lr/rf/svm; run baselines separately with zero tuning trials"
        )
    if len(set(args.models)) != len(args.models):
        arg_parser.error("--models cannot contain duplicates")
    config = WalkForwardConfig(
        args.start_year,
        args.first_train_end_year,
        args.last_test_year,
        gap=args.gap,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or Path("reports/runs") / f"{args.ticker}_{args.target}_{stamp}"
    if output.exists():
        arg_parser.error(f"Output directory already exists: {output}; choose a new directory")
    source = args.data_dir / f"historical_{args.ticker}_data.csv"
    frame = build_features(load_prices(source, args.ticker), args.ticker)
    feature_sets = {"prices": list(FEATURE_COLUMNS)}
    news_metadata, news_coverage = None, None
    if args.news_csv:
        articles = load_scored_news(args.news_csv)
        if not articles["Ticker"].eq(args.ticker).any():
            arg_parser.error(f"No prepared news for {args.ticker}")
        frame, news_coverage = add_news_features(
            frame,
            articles,
            args.ticker,
            latency_minutes=args.news_latency_minutes,
        )
        annual = news_coverage.groupby(news_coverage["DATE"].dt.year)["News_Count"].sum()
        for year in range(args.first_train_end_year + 1, args.last_test_year + 1):
            if annual.get(year, 0) == 0:
                arg_parser.error(
                    f"No news coverage in test year {year}; choose a covered evaluation period"
                )
        feature_sets["prices_news"] = [*FEATURE_COLUMNS, *NEWS_FEATURE_COLUMNS]
        news_metadata = {
            "input_sha256": hashlib.sha256(args.news_csv.read_bytes()).hexdigest(),
            "articles_assigned": news_coverage.attrs["articles_assigned"],
            "fraction_sessions_with_news": float(news_coverage["News_Count"].gt(0).mean()),
            "test_fraction_sessions_with_news": float(
                news_coverage.loc[
                    news_coverage["DATE"].dt.year.between(
                        args.first_train_end_year + 1, args.last_test_year
                    ),
                    "News_Count",
                ]
                .gt(0)
                .mean()
            ),
            "latency_minutes": args.news_latency_minutes,
            "availability_is_publication_proxy": True,
        }
        manifest = args.news_csv.with_suffix(".metadata.json")
        if manifest.exists():
            news_metadata["preparation_manifest"] = json.loads(manifest.read_text())
            if (
                news_metadata["preparation_manifest"].get("output_sha256")
                != news_metadata["input_sha256"]
            ):
                arg_parser.error("Prepared news no longer match their provenance manifest")
    if args.target == "binary":
        labeled = get_target(frame, args.ticker, barrier_std=args.barrier_std)
    else:
        labeled = get_tbm_target(frame, args.ticker, horizon=args.horizon)
    classes = [0, 1] if args.target == "binary" else [-1, 0, 1]
    metadata = {
        "created_at_utc": stamp,
        "arguments": vars(args),
        "versions": _versions(),
        "git": _git_state(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "features": list(FEATURE_COLUMNS),
        "class_order": classes,
        "decision_time": "After regular-session close; labels use subsequent closes",
        "conditional_evaluation": args.target == "binary" and args.barrier_std > 0,
        "macro_included": False,
        "news_included": args.news_csv is not None,
        "news": news_metadata,
        "feature_sets": feature_sets,
        "bootstrap": {"block_size": 20, "n_bootstrap": 1000, "seed": args.seed},
        "not_a_trading_backtest": True,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
    }
    results, summary_rows, samples_by_model, run_definitions = {}, [], {}, {}
    runs = [(name, feature_set) for name in args.models for feature_set in feature_sets]
    # Complete all evaluations before writing the report directory.
    for name, feature_set in runs:
        run_name = f"{name}_{feature_set}" if args.news_csv else name
        run_definitions[run_name] = (name, feature_set)
        print(f"Evaluating {args.ticker} / {args.target} / {name} / {feature_set} ...", flush=True)
        model = build_model(
            name,
            select_k=args.select_k,
            random_state=args.seed,
            calibration_gap=args.horizon if args.target == "tbm" else 1,
        )
        result = walk_forward_validation(
            labeled,
            feature_sets[feature_set],
            model,
            target_type=args.target,
            config=config,
            n_trials=args.tuning_trials,
            random_state=args.seed,
        )
        predictions = result.predictions
        probabilities = predictions[[f"Proba_{c}" for c in classes]].to_numpy()
        metrics = classification_metrics(
            predictions["Actual"],
            predictions["Predicted"],
            probabilities,
            classes,
        )
        samples, low, high = block_bootstrap_accuracy(
            predictions["Actual"],
            predictions["Predicted"],
            random_state=args.seed,
        )
        results[run_name], samples_by_model[run_name] = result, samples
        summary_rows.append(
            {
                "model": name,
                "feature_set": feature_set,
                "ticker": args.ticker,
                "target": args.target,
                "n_predictions": len(predictions),
                **metrics,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if "majority" in args.models:
        for name, result in results.items():
            model_name, feature_set = run_definitions[name]
            baseline_name = f"majority_{feature_set}" if args.news_csv else "majority"
            base = results[baseline_name].predictions
            if not result.predictions[["Date", "Actual"]].equals(base[["Date", "Actual"]]):
                raise ValueError("Model and baseline test calendars differ")
            # The same seed, lengths and block size select the same observations.
            difference = samples_by_model[name] - samples_by_model[baseline_name]
            low, high = np.percentile(difference, [2.5, 97.5])
            row = (summary["model"] == model_name) & (summary["feature_set"] == feature_set)
            baseline_row = (summary["model"] == "majority") & (
                summary["feature_set"] == feature_set
            )
            summary.loc[row, "accuracy_delta_vs_majority"] = (
                summary.loc[row, "accuracy"].iloc[0] - summary.loc[baseline_row, "accuracy"].iloc[0]
            )
            summary.loc[row, "paired_delta_ci_low"] = low
            summary.loc[row, "paired_delta_ci_high"] = high
    if args.news_csv:
        for name in args.models:
            before, after = results[f"{name}_prices"], results[f"{name}_prices_news"]
            if not before.predictions[["Date", "Actual"]].equals(
                after.predictions[["Date", "Actual"]]
            ):
                raise ValueError("Price and news ablation calendars differ")
            difference = (
                samples_by_model[f"{name}_prices_news"] - samples_by_model[f"{name}_prices"]
            )
            low, high = np.percentile(difference, [2.5, 97.5])
            row = (summary["model"] == name) & (summary["feature_set"] == "prices_news")
            base = (summary["model"] == name) & (summary["feature_set"] == "prices")
            summary.loc[row, "accuracy_delta_vs_prices"] = (
                summary.loc[row, "accuracy"].iloc[0] - summary.loc[base, "accuracy"].iloc[0]
            )
            summary.loc[row, "paired_news_delta_ci_low"] = low
            summary.loc[row, "paired_news_delta_ci_high"] = high
    output.mkdir(parents=True, exist_ok=False)
    for name, result in results.items():
        destination = output / name
        destination.mkdir()
        result.folds.to_csv(destination / "folds.csv", index=False)
        result.predictions.to_csv(destination / "predictions.csv", index=False)
        (destination / "selected_features.json").write_text(
            json.dumps(result.selected_features, indent=2),
            encoding="utf-8",
        )
        (destination / "model_parameters.json").write_text(
            json.dumps(_json_clean(result.model_parameters), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    summary.to_csv(output / "comparison.csv", index=False)
    if news_coverage is not None:
        news_coverage.to_csv(output / "news_coverage.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(_json_clean(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Reports saved to {output}")
    return 0
