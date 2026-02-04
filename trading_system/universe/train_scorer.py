"""
Train the universe ScorerModel from historical aligned returns.
Builds feature matrix and labels (top quartile forward return = 1) and fits GradientBoostingClassifier.
"""
import numpy as np

from ..config import ENV_OBS_ROLLING_FEATURES
from ..utils.logger import get_logger
from .features import feature_dict_to_vector
from .scorer import ScorerModel

logger = get_logger()


def _proxy_from_returns_np(returns: np.ndarray, start: int, window: int = 21) -> dict:
    """
    Proxy specialist outputs from rolling returns (numpy).
    Returns dict with sentiment, quant_prob, fund_score, regime_flag, and optionally rolling features.
    """
    lookback = returns[max(0, start - window) : start]
    lookback = lookback[~np.isnan(lookback)]
    if len(lookback) == 0:
        return {
            "sentiment": 0.0,
            "quant_prob": 0.5,
            "fund_score": 0.0,
            "regime_flag": 0.5,
            "rolling_return_5d": 0.0,
            "rolling_return_21d": 0.0,
            "rolling_vol_21d": 0.0,
        }
    mean_ret = float(np.mean(lookback))
    std_ret = float(np.std(lookback))
    if std_ret < 1e-8:
        std_ret = 0.01
    sentiment = float(np.clip(mean_ret * 12, -0.5, 0.5))
    quant_prob = 0.5 + float(np.clip(mean_ret * 20, -0.5, 0.5))
    fund_score = float(np.clip(-mean_ret * 15, -0.5, 0.5))
    regime_flag = 0.5 + float(np.clip((std_ret - 0.01) * 50, -0.5, 0.5))
    out = {
        "sentiment": sentiment,
        "quant_prob": quant_prob,
        "fund_score": fund_score,
        "regime_flag": regime_flag,
    }
    if ENV_OBS_ROLLING_FEATURES:
        ret_5d = float(np.mean(lookback[-5:])) if len(lookback) >= 5 else mean_ret
        ret_21d = float(np.mean(lookback[-21:])) if len(lookback) >= 21 else mean_ret
        vol_21d = float(np.std(lookback[-21:])) if len(lookback) >= 21 else std_ret
        if vol_21d < 1e-8:
            vol_21d = 0.01
        out["rolling_return_5d"] = float(np.clip(ret_5d * 50, -1, 1))
        out["rolling_return_21d"] = float(np.clip(ret_21d * 50, -1, 1))
        out["rolling_vol_21d"] = float(np.clip((vol_21d - 0.01) * 50, -1, 1))
    return out


def train_scorer_from_returns(
    returns_array: np.ndarray,
    n_assets: int,
    forward_days: int = 21,
    window: int = 21,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> None:
    """
    Build X (feature matrix) and y (binary: 1 if forward return in top quartile) from aligned returns.
    returns_array: shape (n_days, n_assets). Train ScorerModel and save.
    """
    n_days = returns_array.shape[0]
    min_start = window
    max_start = n_days - forward_days - 1
    if max_start <= min_start:
        logger.error("Insufficient history for training universe scorer (need %d+ days)", window + forward_days)
        return
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for start in range(min_start, max_start + 1):
        day_forward_returns: list[float] = []
        day_features: list[dict] = []
        for i in range(n_assets):
            ret_series = returns_array[:, i]
            proxy = _proxy_from_returns_np(ret_series, start, window)
            day_features.append(proxy)
            fwd = float(np.nanmean(ret_series[start : start + forward_days]))
            day_forward_returns.append(fwd)
        quartile = float(np.percentile(day_forward_returns, 75))
        for i in range(n_assets):
            X_list.append(feature_dict_to_vector(day_features[i]))
            y_list.append(1 if day_forward_returns[i] >= quartile else 0)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info("Universe scorer training: X %s, y sum=%d (positive)", X.shape, int(y.sum()))
    model = ScorerModel(load=False)
    model.train(X, y, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
