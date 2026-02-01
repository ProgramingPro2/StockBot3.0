"""
Normalization, scaling, and data preparation for specialists.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ..config import TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH
from ..utils.logger import get_logger

logger = get_logger()


def _robust_scale(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, RobustScaler]:
    """Apply RobustScaler to columns; handle NaNs."""
    scaler = RobustScaler()
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        if mask.sum() < 2:
            continue
        df.loc[mask, col] = scaler.fit_transform(df.loc[mask, [col]].values)
    return df, scaler


def prepare_tft_data(
    df: pd.DataFrame,
    target_col: str = "forward_return_30d",
    group_id: str = "ticker",
    max_encoder_length: int = TFT_MAX_ENCODER_LENGTH,
    max_prediction_length: int = TFT_MAX_PREDICTION_LENGTH,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Prepare DataFrame for TimeSeriesDataSet / TFT.
    Expects OHLCV with columns: open, high, low, close, volume.
    Adds time_idx, group_ids, target, time_varying features.
    ticker: if set, used as group id (for multi-ticker TFT); otherwise "asset".
    """
    if df.empty or len(df) < max_encoder_length + max_prediction_length:
        return pd.DataFrame()

    df = df.copy()
    df = df.sort_index().reset_index()
    df["time_idx"] = np.arange(len(df))

    # Returns
    df["return_1d"] = df["close"].pct_change()
    df["forward_return_30d"] = df["close"].shift(-max_prediction_length) / df["close"] - 1
    df[group_id] = ticker if ticker else "asset"

    # Rolling features
    df["vol_20d"] = df["return_1d"].rolling(20).std()
    df["vol_20d"] = df["vol_20d"].bfill().fillna(0)
    df["return_1d"] = df["return_1d"].fillna(0)

    # Drop rows without valid target
    df = df.dropna(subset=[target_col])
    return df


def prepare_regime_features(spy_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Prepare SPY data for regime VAE.
    Returns DataFrame with rolling returns, volatility, volume features.
    """
    if spy_df.empty or len(spy_df) < window + 1:
        return pd.DataFrame()

    df = spy_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return pd.DataFrame()

    df["return"] = df["close"].pct_change()
    df["vol"] = df["return"].rolling(window).std()
    df["vol"] = df["vol"].bfill().fillna(0)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(window)
    if "volume" in df.columns:
        vol_sma = df["volume"].rolling(window).mean()
        df["volume_ratio"] = (df["volume"] / vol_sma).fillna(1).clip(0, 5)
    else:
        df["volume_ratio"] = 1.0
    feature_cols = ["return", "vol", "return_5d", "return_20d", "volume_ratio"]
    df = df.dropna()
    return df[feature_cols]
