"""Read the preserved price snapshots; never use the legacy merged feature tables."""

from pathlib import Path

import numpy as np
import pandas as pd


def validate_dates(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    """Return a copy in chronological order, rejecting ambiguous observation dates."""
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="raise")
    if result[date_col].isna().any() or result[date_col].duplicated().any():
        raise ValueError(f"{date_col} must contain unique, nonmissing dates")
    return result.sort_values(date_col).reset_index(drop=True)


def load_prices(path: str | Path, ticker: str) -> pd.DataFrame:
    df = validate_dates(pd.read_csv(path))
    columns = [f"{field}_{ticker}" for field in ("Open", "High", "Low", "Close", "Volume")]
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    df = df[["DATE", *columns]].copy()
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV contains missing or infinite values")
    if (df[columns[:4]] <= 0).any().any() or (df[columns[4]] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if (
        df[f"High_{ticker}"]
        < df[[f"Open_{ticker}", f"Close_{ticker}", f"Low_{ticker}"]].max(axis=1) - 1e-8
    ).any():
        raise ValueError("High is below another OHLC price")
    if (
        df[f"Low_{ticker}"]
        > df[[f"Open_{ticker}", f"Close_{ticker}", f"High_{ticker}"]].min(axis=1) + 1e-8
    ).any():
        raise ValueError("Low is above another OHLC price")
    return df


def merge_available_data(prices: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    """Backward join on explicit availability timestamps, not observation periods.

    Caller must provide a point-in-time release history (including vintage values).
    DATE must describe the actual decision timestamp and Available_At the instant
    a release was usable. Equal timestamps are excluded conservatively. This
    helper cannot reconstruct vintages from the archived FRED CSV files.
    """
    left = validate_dates(prices)
    right = validate_dates(releases, "Available_At")
    overlapping = (set(left) & set(right)) - {"Available_At"}
    if overlapping:
        raise ValueError(f"Release columns overlap existing columns: {sorted(overlapping)}")
    return pd.merge_asof(
        left,
        right,
        left_on="DATE",
        right_on="Available_At",
        direction="backward",
        allow_exact_matches=False,
    )
