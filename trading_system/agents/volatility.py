"""
Volatility Specialist: Per-asset realized volatility normalized to [0, 1].
Rule-based; no training required.
"""
import numpy as np
import pandas as pd

from ..config import VOLATILITY_WINDOW_DAYS, VOLATILITY_REF_WINDOW_DAYS
from ..data.loader import load_ohlcv
from ..utils.logger import get_logger

logger = get_logger()


def get_volatility_score(ticker: str) -> float:
    """
    Get normalized volatility score for a ticker.
    Higher = higher recent volatility (risk).
    Returns float in [0, 1].
    """
    df = load_ohlcv(ticker)
    if df.empty or "close" not in df.columns or len(df) < VOLATILITY_REF_WINDOW_DAYS:
        return 0.5

    ret = df["close"].pct_change().dropna()
    if len(ret) < VOLATILITY_WINDOW_DAYS:
        return 0.5

    vol_short = float(ret.tail(VOLATILITY_WINDOW_DAYS).std())
    vol_long = float(ret.tail(VOLATILITY_REF_WINDOW_DAYS).std())
    if not np.isfinite(vol_long) or vol_long < 1e-10:
        return 0.5
    if not np.isfinite(vol_short):
        return 0.5

    ratio = vol_short / vol_long
    score = float(np.clip(ratio, 0.0, 1.0))
    if not np.isfinite(score):
        return 0.5
    return score
