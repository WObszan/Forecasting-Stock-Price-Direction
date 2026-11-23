"""Labels with explicit event end dates for purging across validation boundaries."""

import numpy as np
import pandas as pd

from .data import validate_dates

LABEL_METADATA = {"Target", "Label_End", "Forward_Log_Return", "Available_At", "DATE", "index"}


def get_target(
    input_df: pd.DataFrame,
    ticker: str,
    vol_window: int = 20,
    barrier_std: float = 0.0,
) -> pd.DataFrame:
    """Next-session close direction: 0 = down/unchanged, 1 = up.

    An optional volatility filter removes small FUTURE moves. Such a sample is
    conditional on a future outcome and cannot represent a tradable calendar.
    The default retains every labeled session, including flat-price sessions.
    """
    if vol_window < 2 or not np.isfinite(barrier_std) or barrier_std < 0:
        raise ValueError("vol_window must be >= 2 and barrier_std finite and nonnegative")
    df = validate_dates(input_df)
    close = df[f"Close_{ticker}"]
    future = np.log(close.shift(-1) / close)
    target = (future > 0).astype(float).where(future.notna())
    if barrier_std:
        volatility = np.log(close / close.shift(1)).rolling(vol_window).std()
        target = target.where(future.abs() > barrier_std * volatility)
    df["Target"] = target
    df["Label_End"] = df["DATE"].shift(-1)
    df["Forward_Log_Return"] = future
    return df.dropna(subset=["Target", "Label_End"]).copy()


def get_tbm_target(
    input_df: pd.DataFrame,
    ticker: str,
    horizon: int = 5,
    pt_sl: tuple[float, float] = (1.3, 1.0),
    vol_window: int = 20,
) -> pd.DataFrame:
    """Close-sampled triple barrier: -1 loss, 0 timeout, +1 profit.

    Barriers are simple-price multiples of trailing log-return volatility, as
    in the original experiment. Intraday High/Low hits are NOT modeled. Skip
    warm-up, zero-volatility and incomplete-horizon observations explicitly.
    """
    if not isinstance(horizon, int) or horizon < 1 or vol_window < 2:
        raise ValueError("horizon must be a positive integer and vol_window >= 2")
    if len(pt_sl) != 2 or not all(np.isfinite(v) and v > 0 for v in pt_sl):
        raise ValueError("pt_sl must contain two positive finite barrier multipliers")
    df = validate_dates(input_df)
    close = df[f"Close_{ticker}"]
    vol = np.log(close / close.shift(1)).rolling(vol_window).std()
    target = np.full(len(df), np.nan)
    ends = pd.Series(pd.NaT, index=df.index, dtype=df["DATE"].dtype)
    for i in range(max(0, len(df) - horizon)):
        if not np.isfinite(vol.iloc[i]) or vol.iloc[i] <= 0:
            continue
        upper = close.iloc[i] * (1 + pt_sl[0] * vol.iloc[i])
        lower = close.iloc[i] * (1 - pt_sl[1] * vol.iloc[i])
        if lower <= 0:
            continue
        target[i] = 0
        end = i + horizon
        for j in range(i + 1, i + horizon + 1):
            if close.iloc[j] >= upper:
                target[i], end = 1, j
                break
            if close.iloc[j] <= lower:
                target[i], end = -1, j
                break
        ends.iloc[i] = df["DATE"].iloc[end]
    df["Target"], df["Label_End"] = target, ends
    df["Forward_Log_Return"] = np.log(close.shift(-1) / close)
    return df.dropna(subset=["Target", "Label_End"]).copy()
