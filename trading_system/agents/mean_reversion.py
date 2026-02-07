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

    sma = df["close"].rolling(SMA_WINDOW).mean().iloc[-1]
    std_ret = df["close"].pct_change().dropna().tail(SMA_WINDOW).std()
    if pd.isna(std_ret) or std_ret < 1e-10:
        std_ret = 1e-10
    z_score = (close - sma) / (std_ret * (SMA_WINDOW ** 0.5) + 1e-10)
    z_term = np.clip(-z_score * 0.15, -0.3, 0.3)

    rsi_term = (50 - rsi) / 50
    score = np.clip(rsi_term + z_term, -1.0, 1.0)
    return float(score)
