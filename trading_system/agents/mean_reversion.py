"""
Mean-Reversion Specialist: RSI and Z-score based oversold/overbought signal.
Rule-based; no training required.
"""
import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv
from ..data.processor import add_momentum_features
from ..utils.logger import get_logger

logger = get_logger()

RSI_PERIOD = 14
SMA_WINDOW = 20


def get_mean_reversion_score(ticker: str) -> float:
    """
    Get mean-reversion score for a ticker.
    Oversold (expect bounce) -> positive; overbought (expect pullback) -> negative.
    Returns float in [-1, 1].
    """
    df = load_ohlcv(ticker)
    if df.empty or "close" not in df.columns or len(df) < SMA_WINDOW + RSI_PERIOD:
        return 0.0

    df = add_momentum_features(df, window=SMA_WINDOW, rsi_period=RSI_PERIOD)
    if "rsi" not in df.columns:
        return 0.0

    row = df.iloc[-1]
    rsi = float(row["rsi"])
    close = float(row["close"])

    # Guard against NaN values from insufficient price history
    if not np.isfinite(rsi) or not np.isfinite(close):
        return 0.0

    sma = float(df["close"].rolling(SMA_WINDOW).mean().iloc[-1])
    if not np.isfinite(sma):
        return 0.0

    # Price-unit z-score: both numerator and denominator in the same (price) units.
    # The previous version divided by std_ret * sqrt(N) where std_ret is in return
    # units (~0.01), making the denominator ~0.045 while the numerator is in dollars
    # ($5+), causing the z-term to always saturate at the clip boundary.
    std_price = float(df["close"].rolling(SMA_WINDOW).std().iloc[-1])
    if not np.isfinite(std_price) or std_price < 1e-10:
        std_price = 1e-10

    z_score = (close - sma) / std_price
    z_term = float(np.clip(-z_score * 0.3, -0.3, 0.3))

    rsi_term = float((50 - rsi) / 50)
    score = float(np.clip(rsi_term + z_term, -1.0, 1.0))

    if not np.isfinite(score):
        return 0.0
    return score
