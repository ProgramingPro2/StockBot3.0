"""
Fundamental Specialist: XGBoost classifier for intrinsic value.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..config import CACHE_MAX_AGE_DAYS, FUNDAMENTAL_CACHE_DIR, FUNDAMENTAL_MODEL_DIR, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_ESTIMATORS
from ..utils.logger import get_logger

logger = get_logger()

FEATURE_KEYS = [
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "debtToEquity",
    "returnOnEquity",
    "returnOnAssets",
    "freeCashflow",
    "operatingCashflow",
    "revenueGrowth",
    "earningsGrowth",
    "profitMargins",
    "currentRatio",
    "quickRatio",
    "dividendYield",
]

VALUE_LABELS = ["Overvalued", "Fair", "Undervalued"]  # 0, 1, 2


def _cache_path(ticker: str) -> Path:
    """Cache file path for fundamental features."""
    FUNDAMENTAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FUNDAMENTAL_CACHE_DIR / f"{ticker.upper()}.json"


def _extract_features(ticker: str) -> dict[str, float]:
    """Extract fundamental metrics from yfinance. Uses cache on network failure."""
    ticker = ticker.upper()
    cache_path = _cache_path(ticker)

    def _load_cache() -> dict[str, float] | None:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(data: dict[str, float]) -> None:
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug("Failed to cache fundamentals for %s: %s", ticker, e)

    # Try yfinance with retries for transient SSL/network errors
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info is None:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                break
            out = {}
            for k in FEATURE_KEYS:
                v = info.get(k)
                if v is not None and not isinstance(v, str):
                    out[k] = float(v)
            if out:
                _save_cache(out)
                return out
            if attempt < 2:
                time.sleep(1 + attempt)
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                logger.warning("yfinance fundamental fetch failed for %s after 3 attempts: %s. Trying cache.", ticker, e)

    # Fallback to cache (even if stale)
    cached = _load_cache()
    if cached:
        return cached

    return {}


def _build_training_data(tickers: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build labeled dataset for training (heuristic: future return percentile)."""
    rows = []
    for ticker in tickers:
        feat = _extract_features(ticker)
        if len(feat) < 5:
            continue
        rows.append({"ticker": ticker, **feat})

    if not rows:
        return pd.DataFrame(), np.array([])

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.select_dtypes(include=[np.number])

    # Heuristic label: use P/E percentile vs future return
    # For training, we use trailing 1y return as proxy for "value"
    # Overvalued = high P/E, Fair = mid, Undervalued = low P/E
    if "trailingPE" in df.columns:
        pe = df["trailingPE"].dropna()
        q33 = pe.quantile(0.33)
        q67 = pe.quantile(0.67)
        labels = np.ones(len(df), dtype=int)
        for i, v in enumerate(df["trailingPE"]):
            if pd.isna(v):
                labels[i] = 1
            elif v <= q33:
                labels[i] = 2  # Undervalued
            elif v >= q67:
                labels[i] = 0  # Overvalued
            else:
                labels[i] = 1  # Fair
    else:
        labels = np.ones(len(df), dtype=int)

    return df, labels


_model: xgb.XGBClassifier | None = None
_imputer: SimpleImputer | None = None
_scaler: StandardScaler | None = None


def _ensure_model(tickers: list[str] | None = None) -> bool:
    """Load or train model."""
    global _model, _imputer, _scaler

    path = FUNDAMENTAL_MODEL_DIR / "xgb_fundamental.json"
    imp_path = FUNDAMENTAL_MODEL_DIR / "imputer.joblib"
    scal_path = FUNDAMENTAL_MODEL_DIR / "scaler.joblib"
    if path.exists() and imp_path.exists() and scal_path.exists():
        try:
            _model = xgb.XGBClassifier()
            _model.load_model(str(path))
            _imputer = joblib.load(imp_path)
            _scaler = joblib.load(scal_path)
            return True
        except Exception as e:
            logger.warning("Failed to load fundamental model: %s", e)

    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "V", "WMT", "PG"]

    df, labels = _build_training_data(tickers)
    if df.empty or len(df) < 10:
        logger.warning("Insufficient data for fundamental model training")
        return False

    X = df[FEATURE_KEYS] if all(c in df.columns for c in FEATURE_KEYS) else df
    X = X.reindex(columns=FEATURE_KEYS, fill_value=np.nan)
    _imputer = SimpleImputer(strategy="median")
    X_imp = _imputer.fit_transform(X)
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X_imp)
    _model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        eval_metric="mlogloss",
    )
    _model.fit(X_scaled, labels)
    FUNDAMENTAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _model.save_model(str(path))
    joblib.dump(_imputer, imp_path)
    joblib.dump(_scaler, scal_path)
    return True


def get_intrinsic_value_score(ticker: str, retrain_tickers: list[str] | None = None) -> tuple[float, float]:
    """
    Get intrinsic value score for a ticker.
    Returns (score, confidence) in [-1, 1] and [0, 1].
    Undervalued=1, Fair=0, Overvalued=-1.
    """
    if not _ensure_model(retrain_tickers):
        return 0.0, 0.5

    feat = _extract_features(ticker)
    if len(feat) < 3:
        return 0.0, 0.5

    row = pd.DataFrame([feat]).reindex(columns=FEATURE_KEYS, fill_value=np.nan)
    X_imp = _imputer.transform(row)
    X_scaled = _scaler.transform(X_imp)
    pred = _model.predict(X_scaled)[0]
    proba = _model.predict_proba(X_scaled)[0]
    conf = float(np.max(proba))

    # Map 0->-1, 1->0, 2->1
    score = float(pred) - 1
    return score, conf
