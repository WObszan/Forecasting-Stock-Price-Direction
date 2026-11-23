"""Deterministic features available at the close of each observation's session."""

import numpy as np
import pandas as pd

from .data import validate_dates

FEATURE_COLUMNS = (
    "log_return",
    "return_lag1",
    "return_lag2",
    "return_5",
    "return_20",
    "volatility_20",
    "intraday_return",
    "overnight_return",
    "range_pct",
    "volume_change",
    "relative_volume_20",
    "rsi_14",
    "macd_pct",
    "macd_hist_pct",
    "bollinger_position_20",
    "distance_sma_20",
    "distance_sma_60",
    "distance_high_20",
    "distance_low_20",
    "cusum",
    "cusum_pos",
    "cusum_neg",
    "return_zscore_20",
)


def build_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Keep the full calendar, including warm-up NaNs, for subsequent labeling.

    No full-sample normalization, anomaly model or target-dependent transforms.
    Recursive statistics use only the prefix available at each observation.
    """
    df = validate_dates(prices)
    close = df[f"Close_{ticker}"]
    volume = df[f"Volume_{ticker}"]
    returns = np.log(close / close.shift(1))
    df["log_return"] = returns
    for lag in (1, 2):
        df[f"return_lag{lag}"] = returns.shift(lag)
    for period in (5, 20):
        df[f"return_{period}"] = np.log(close / close.shift(period))
    df["volatility_20"] = returns.rolling(20).std()
    df["intraday_return"] = np.log(close / df[f"Open_{ticker}"])
    df["overnight_return"] = np.log(df[f"Open_{ticker}"] / close.shift(1))
    df["range_pct"] = (df[f"High_{ticker}"] - df[f"Low_{ticker}"]) / close
    df["volume_change"] = np.log1p(volume).diff()
    df["relative_volume_20"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["rsi_14"] = 100 * gain / (gain + loss).replace(0, np.nan)
    df.loc[(gain == 0) & (loss == 0), "rsi_14"] = 50
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["macd_pct"] = macd / close
    df["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    std = close.rolling(20).std().replace(0, np.nan)
    df["bollinger_position_20"] = (close - close.rolling(20).mean()) / (2 * std)
    for period in (20, 60):
        df[f"distance_sma_{period}"] = close / close.rolling(period).mean() - 1
    df["distance_high_20"] = close / df[f"High_{ticker}"].rolling(20).max() - 1
    df["distance_low_20"] = close / df[f"Low_{ticker}"].rolling(20).min() - 1

    past_mean = returns.expanding(min_periods=1).mean().shift(1).fillna(0)
    centered = returns - past_mean
    df["cusum"] = centered.cumsum()
    pos, neg = 0.0, 0.0
    positives, negatives = [], []
    for value in centered:
        if pd.isna(value):
            positives.append(np.nan)
            negatives.append(np.nan)
            continue
        pos, neg = max(0.0, pos + value), min(0.0, neg + value)
        positives.append(pos)
        negatives.append(neg)
    df["cusum_pos"], df["cusum_neg"] = positives, negatives
    past_std = returns.rolling(20).std().shift(1).replace(0, np.nan)
    df["return_zscore_20"] = (returns - returns.rolling(20).mean().shift(1)) / past_std
    df[list(FEATURE_COLUMNS)] = df[list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return df
