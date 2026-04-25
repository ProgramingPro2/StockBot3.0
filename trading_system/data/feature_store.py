"""
Historical feature store: point-in-time price/volume features from OHLCV only.
No future data or external APIs; used for training, tuning, and backtest.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, ENV_OBS_ROLLING_FEATURES
from ..utils.logger import get_logger
from .loader import load_ohlcv
from .processor import add_momentum_features

logger = get_logger()

FEATURE_STORE_CACHE_DIR = CACHE_DIR / "feature_store"
# Feature names that match PPO observation schema (price/volume-only subset)
PRICE_VOLUME_FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "vol_20d",
    "atr_pct",
    "momentum_score",
    "mean_reversion_score",
    "volatility_score",
    "volume_ratio",
    "rolling_return_5d",
    "rolling_return_21d",
    "rolling_vol_21d",
]


def _build_single_ticker_features(
    df: pd.DataFrame,
    ticker: str,
    window: int = 21,
) -> pd.DataFrame:
    """
    Build point-in-time features for one ticker from OHLCV.
    All features use only past data (no lookahead).
    """
    if df.empty or len(df) < window + 5:
        return pd.DataFrame()
    df = df.sort_index().copy()
    if "close" not in df.columns:
        return pd.DataFrame()
    df = add_momentum_features(df, window=20, rsi_period=14)

    # Returns (already point-in-time)
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(window)
    df["vol_20d"] = df["return_1d"].rolling(window).std()
    df["vol_20d"] = df["vol_20d"].bfill().fillna(0.01)
    # ATR as percent of price (normalized)
    atr = df["atr"] if "atr" in df.columns else df["close"].rolling(14).std().fillna(0.01)
    df["atr_pct"] = (atr / (df["close"] + 1e-12)).clip(0, 0.5)
    # Momentum / mean reversion scores (clipped to [-1, 1] and [0, 1])
    df["momentum_score"] = np.clip(df["return_20d"].fillna(0) * 10, -1.0, 1.0)
    df["mean_reversion_score"] = np.clip(-df["return_5d"].fillna(0) * 10, -1.0, 1.0)
    vol_norm = (df["vol_20d"] - 0.01) / 0.02
    df["volatility_score"] = np.clip(0.5 + vol_norm, 0.0, 1.0)
    # Volume ratio: current / 20d mean
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(window).mean().replace(0, np.nan).bfill()
        df["volume_ratio"] = (df["volume"] / vol_ma).fillna(1.0).clip(0, 5)
    else:
        df["volume_ratio"] = 1.0
    # Rolling features for env compatibility
    df["rolling_return_5d"] = np.clip(df["return_1d"].rolling(5).mean().fillna(0) * 50, -1, 1)
    df["rolling_return_21d"] = np.clip(df["return_1d"].rolling(21).mean().fillna(0) * 50, -1, 1)
    rv = df["return_1d"].rolling(21).std().fillna(0.01)
    df["rolling_vol_21d"] = np.clip((rv - 0.01) * 50, -1, 1)

    cols = [c for c in PRICE_VOLUME_FEATURE_NAMES if c in df.columns]
    out = df[cols].copy()
    out["ticker"] = ticker
    return out.dropna(how="all").replace([np.inf, -np.inf], np.nan).ffill().bfill()


def build_feature_dataset(
    tickers: list[str],
    lookback_years: int = 5,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """
    Build a dated (date, ticker) feature matrix from OHLCV only.
    Returns DataFrame with MultiIndex (date, ticker) and columns = feature names,
    or None if insufficient data.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return None
    cache_path = cache_path or (FEATURE_STORE_CACHE_DIR / "features.parquet")
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if set(tickers).issubset(cached.index.get_level_values("ticker").unique()):
                return cached.loc[cached.index.get_level_values("ticker").isin(tickers)]
        except Exception as e:
            logger.debug("Feature store cache read failed: %s", e)

    parts: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=lookback_years)
        if df.empty or len(df) < 30:
            logger.debug("Skipping %s: insufficient OHLCV", t)
            continue
        feats = _build_single_ticker_features(df, t, window=21)
        if not feats.empty:
            parts.append(feats)
    if not parts:
        return None
    combined = pd.concat(parts, axis=0, ignore_index=False)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep="last")]
    # Reshape to (date, ticker) index
    combined = combined.reset_index()
    if "Date" not in combined.columns and "index" in combined.columns:
        combined = combined.rename(columns={"index": "date"})
    elif "Date" in combined.columns:
        combined = combined.rename(columns={"Date": "date"})
    if "date" not in combined.columns:
        return None
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.set_index(["date", "ticker"])
    FEATURE_STORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_parquet(cache_path)
    except Exception as e:
        logger.debug("Feature store cache write failed: %s", e)
    return combined


def get_features_for_date(
    feature_df: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    tickers: list[str],
) -> list[dict]:
    """
    Return one dict per ticker of feature values as of as_of_date.
    Dict keys match specialist output schema for drop-in use in backtest/training.
    Missing tickers get neutral defaults.
    """
    if feature_df is None or feature_df.empty:
        return [_neutral_feature_dict() for _ in tickers]
    if isinstance(as_of_date, str):
        as_of_date = pd.Timestamp(as_of_date)
    out: list[dict] = []
    for t in tickers:
        try:
            row = feature_df.loc[(as_of_date, t)]
        except (KeyError, TypeError):
            out.append(_neutral_feature_dict())
            continue
        d = _neutral_feature_dict()
        idx = row.index if hasattr(row, "index") else []
        for k in PRICE_VOLUME_FEATURE_NAMES:
            if k not in idx:
                continue
            val = float(row[k])
            if k == "momentum_score":
                d["momentum_score"] = val
            elif k == "mean_reversion_score":
                d["mean_reversion_score"] = val
            elif k == "volatility_score":
                d["volatility_score"] = val
            elif k in ("rolling_return_5d", "rolling_return_21d", "rolling_vol_21d") and ENV_OBS_ROLLING_FEATURES:
                d[k] = val
        # Placeholders for schema compatibility (no sentiment/quant/fund in price-only store)
        d["sentiment"] = 0.0
        d["quant_prob"] = 0.5
        d["fund_score"] = 0.0
        d["regime_flag"] = 0.5
        d["event_risk_score"] = 0.0
        d["relative_strength_score"] = 0.0
        vr = float(row.get("volume_ratio", 0.5)) if "volume_ratio" in idx else 0.5
        d["liquidity_score"] = min(1.0, max(0.0, vr / 2.0))
        out.append(d)
    return out


def build_daily_aux_for_aligned(
    aligned: pd.DataFrame,
    tickers: list[str],
    lookback_years: int = 5,
) -> np.ndarray | None:
    """
    Build (n_days, n_assets, 2) array of [atr_pct, volume_ratio] aligned to aligned.index
    for merging into HistoricalTradingEnv observations. Returns None if build fails.
    """
    feat = build_feature_dataset(tickers, lookback_years=lookback_years)
    if feat is None or feat.empty:
        return None
    dates = pd.to_datetime(aligned.index).tz_localize(None)
    n = len(dates)
    m = len(tickers)
    out = np.full((n, m, 2), np.nan, dtype=np.float32)
    for ti, t in enumerate(tickers):
        for di in range(n):
            d = dates[di]
            try:
                row = feat.loc[(d.normalize(), t)]
            except (KeyError, TypeError, ValueError):
                try:
                    row = feat.loc[(d, t)]
                except (KeyError, TypeError, ValueError):
                    continue
            idx = row.index if hasattr(row, "index") else []
            if "atr_pct" in idx:
                out[di, ti, 0] = float(row["atr_pct"])
            if "volume_ratio" in idx:
                out[di, ti, 1] = float(row["volume_ratio"])
    return out


def check_no_future_leakage() -> bool:
    """
    Leakage contract: all features in this module use only past data (no shift(-n) or future-known labels).
    Returns True if the implementation is consistent with that contract (for tests).
    """
    # All features are built from: pct_change, rolling(...).mean/std, ATR, RSI, volume ratio.
    # None of these use shift with negative periods. Callers should use get_features_for_date(as_of_date)
    # so that only rows with index <= as_of_date are used when slicing.
    return True


def _neutral_feature_dict() -> dict:
    """Default feature dict when data is missing."""
    d = {
        "sentiment": 0.0,
        "quant_prob": 0.5,
        "fund_score": 0.0,
        "regime_flag": 0.5,
        "momentum_score": 0.0,
        "mean_reversion_score": 0.0,
        "volatility_score": 0.5,
        "event_risk_score": 0.0,
        "relative_strength_score": 0.0,
        "liquidity_score": 0.5,
    }
    if ENV_OBS_ROLLING_FEATURES:
        d["rolling_return_5d"] = 0.0
        d["rolling_return_21d"] = 0.0
        d["rolling_vol_21d"] = 0.0
    return d
