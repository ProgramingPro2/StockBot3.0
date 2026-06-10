"""
Momentum Specialist: LightGBM for short-horizon momentum score.
"""
import os

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


def _build_training_data_single_pass(
    tickers: list[str],
    end_date: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Build feature/target matrices with a chronological 80/20 train/val split.

    A purge/embargo gap of MOMENTUM_FORWARD_DAYS bars is removed around the split
    boundary to prevent target overlap between train and validation sets.  Forward
    5-day targets overlap ~80% between adjacent samples, so without the embargo the
    validation set would share labeled information with training.
    """
    X_train_list: list[pd.DataFrame] = []
    y_train_list: list[np.ndarray] = []
    X_val_list: list[pd.DataFrame] = []
    y_val_list: list[np.ndarray] = []

    purge = MOMENTUM_FORWARD_DAYS  # bars to drop around the split boundary

    for ticker in tickers:
        df = load_ohlcv(ticker, lookback_years=min(5, YFINANCE_LOOKBACK_YEARS), end=end_date)
        if df.empty or len(df) < MOMENTUM_WINDOW_DAYS + MOMENTUM_FORWARD_DAYS + 10:
            continue
        df = add_momentum_features(df, window=20)
        if not all(c in df.columns for c in FEATURE_COLS):
            continue
        df["forward_return"] = df["close"].shift(-MOMENTUM_FORWARD_DAYS) / df["close"] - 1
        df = df.dropna(subset=["forward_return"] + FEATURE_COLS)
        if len(df) < 50:
            continue

        n = len(df)
        split = int(n * 0.8)
        # Purge rows within [split-purge, split) to eliminate label overlap
        train_end = max(0, split - purge)
        val_start = split

        df_train = df.iloc[:train_end]
        df_val = df.iloc[val_start:]

        if len(df_train) < 30:
            continue
        X_train_list.append(df_train[FEATURE_COLS].copy())
        y_train_list.append(np.clip(df_train["forward_return"].values, -1.0, 1.0))

        if len(df_val) >= 20:
            X_val_list.append(df_val[FEATURE_COLS].copy())
            y_val_list.append(np.clip(df_val["forward_return"].values, -1.0, 1.0))

    if not X_train_list:
        return pd.DataFrame(), np.array([]), pd.DataFrame(), np.array([])

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    if X_val_list:
        X_val = pd.concat(X_val_list, ignore_index=True)
        y_val = np.concatenate(y_val_list)
    else:
        X_val = pd.DataFrame()
        y_val = np.array([])

    return X_train, y_train, X_val, y_val


_model: lgb.LGBMRegressor | None = None
_imputer: SimpleImputer | None = None
_scaler: StandardScaler | None = None


def _ensure_model(
    tickers: list[str] | None = None,
    retrain: bool = False,
    end_date: str | None = None,
) -> bool:
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

    X_train, y_train, X_val, y_val = _build_training_data_single_pass(tickers, end_date=end_date)
    if X_train.empty or len(X_train) < 100:
        logger.warning("Insufficient data for momentum model training")
        return False

    X_train = X_train.reindex(columns=FEATURE_COLS, fill_value=np.nan)
    _imputer = SimpleImputer(strategy="median")
    X_imp = _imputer.fit_transform(X_train)
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X_imp)

    _njobs_raw = os.getenv("LIGHTGBM_N_JOBS", "-1").strip()
    _njobs = int(_njobs_raw) if _njobs_raw else -1
    _model = lgb.LGBMRegressor(
        n_estimators=MOMENTUM_N_ESTIMATORS,
        max_depth=MOMENTUM_MAX_DEPTH,
        learning_rate=MOMENTUM_LEARNING_RATE,
        verbosity=-1,
        force_col_wise=True,
        n_jobs=_njobs,
    )

    fit_kwargs: dict = {"X": X_scaled, "y": y_train}
    if not X_val.empty and len(y_val) > 0:
        X_val_r = X_val.reindex(columns=FEATURE_COLS, fill_value=np.nan)
        X_val_imp = _imputer.transform(X_val_r)
        X_val_scaled = _scaler.transform(X_val_imp)
        fit_kwargs["eval_set"] = [(X_val_scaled, y_val)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

    _model.fit(**fit_kwargs)

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
    result = float(np.clip(pred, -1.0, 1.0))
    if not np.isfinite(result):
        return 0.0
    return result
