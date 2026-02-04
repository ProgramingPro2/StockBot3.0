"""
Momentum Specialist: LightGBM for short-horizon momentum score.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..config import (
    MOMENTUM_FORWARD_DAYS,
    MOMENTUM_LEARNING_RATE,
    MOMENTUM_MAX_DEPTH,
    MOMENTUM_MODEL_DIR,
    MOMENTUM_N_ESTIMATORS,
    MOMENTUM_WINDOW_DAYS,
    YFINANCE_LOOKBACK_YEARS,
)
from ..data.loader import load_ohlcv
from ..data.processor import add_momentum_features
from ..utils.logger import get_logger

logger = get_logger()

FEATURE_COLS = ["return_5d", "return_20d", "rsi", "atr"]


def _build_training_data_single_pass(tickers: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build feature matrix and target in one pass per ticker."""
    X_list: list[pd.DataFrame] = []
    y_list: list[np.ndarray] = []
    for ticker in tickers:
        df = load_ohlcv(ticker, lookback_years=min(5, YFINANCE_LOOKBACK_YEARS))
        if df.empty or len(df) < MOMENTUM_WINDOW_DAYS + MOMENTUM_FORWARD_DAYS + 10:
            continue
        df = add_momentum_features(df, window=20)
        if not all(c in df.columns for c in FEATURE_COLS):
            continue
        df["forward_return"] = df["close"].shift(-MOMENTUM_FORWARD_DAYS) / df["close"] - 1
        df = df.dropna(subset=["forward_return"] + FEATURE_COLS)
        if len(df) < 50:
            continue
        X_list.append(df[FEATURE_COLS].copy())
        y_list.append(np.clip(df["forward_return"].values, -1.0, 1.0))
    if not X_list:
        return pd.DataFrame(), np.array([])
    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)
    if len(y) != len(X):
        y = y[: len(X)]
    return X, y


_model: lgb.LGBMRegressor | None = None
_imputer: SimpleImputer | None = None
_scaler: StandardScaler | None = None


def _ensure_model(tickers: list[str] | None = None, retrain: bool = False) -> bool:
    """Load or train Momentum model."""
    global _model, _imputer, _scaler

    path = MOMENTUM_MODEL_DIR / "lgb_momentum.joblib"
    imp_path = MOMENTUM_MODEL_DIR / "momentum_imputer.joblib"
    scal_path = MOMENTUM_MODEL_DIR / "momentum_scaler.joblib"
    if path.exists() and imp_path.exists() and scal_path.exists() and not retrain:
        try:
            _model = joblib.load(path)
            _imputer = joblib.load(imp_path)
            _scaler = joblib.load(scal_path)
            return True
        except Exception as e:
            logger.warning("Failed to load momentum model: %s", e)

    if tickers is None:
        tickers = ["SPY", "IWM", "XLE", "XLF", "EEM", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]

    X, y = _build_training_data_single_pass(tickers)
    if X.empty or len(X) < 100:
        logger.warning("Insufficient data for momentum model training")
        return False

    X = X.reindex(columns=FEATURE_COLS, fill_value=np.nan)
    _imputer = SimpleImputer(strategy="median")
    X_imp = _imputer.fit_transform(X)
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X_imp)

    _model = lgb.LGBMRegressor(
        n_estimators=MOMENTUM_N_ESTIMATORS,
        max_depth=MOMENTUM_MAX_DEPTH,
        learning_rate=MOMENTUM_LEARNING_RATE,
        verbosity=-1,
        force_col_wise=True,
    )
    _model.fit(X_scaled, y)

    MOMENTUM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(_model, path)
    joblib.dump(_imputer, imp_path)
    joblib.dump(_scaler, scal_path)
    return True


def get_momentum_score(ticker: str, retrain_tickers: list[str] | None = None) -> float:
    """
    Get short-horizon momentum score for a ticker.
    Returns float in [-1, 1] (negative = down momentum, positive = up momentum).
    """
    if not _ensure_model(retrain_tickers):
        return 0.0

    df = load_ohlcv(ticker)
    if df.empty or len(df) < MOMENTUM_WINDOW_DAYS + 5:
        return 0.0
    df = add_momentum_features(df, window=20)
    if not all(c in df.columns for c in FEATURE_COLS):
        return 0.0
    row = df[FEATURE_COLS].iloc[-1:].reindex(columns=FEATURE_COLS, fill_value=0)
    X_imp = _imputer.transform(row)
    X_scaled = _scaler.transform(X_imp)
    pred = _model.predict(X_scaled)[0]
    return float(np.clip(pred, -1.0, 1.0))
