"""
Shared proxy-from-returns specialist features for env, training, tuning, and backtest.
Single source of truth to avoid drift between numpy/pandas code paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ENV_OBS_ROLLING_FEATURES


def normalize_action_max_unit_exposure(action: np.ndarray) -> np.ndarray:
    """Clip to [0,1] per asset; if sum > 1, scale so total exposure is 1 (matches live/backtest)."""
    a = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    s = float(np.sum(a))
    if s > 1.0 + 1e-8:
        a = (a / s).astype(np.float32)
    return a


def proxy_specialists_vectorized(returns_2d: np.ndarray) -> np.ndarray:
    """
    returns_2d shape (window, n_assets).
    Returns shape (n_assets, 13): sentiment, quant_prob, fund_score, regime_flag,
    momentum, mean_reversion, volatility, event_risk, relative_strength, liquidity,
    ret_5d_norm, ret_21d_norm, vol_21d_norm per asset.
    """
    returns_2d = np.asarray(returns_2d, dtype=np.float32)
    mean_ret = np.nanmean(returns_2d, axis=0).astype(np.float32)
    std_ret = np.nanstd(returns_2d, axis=0).astype(np.float32)
    std_ret = np.where(std_ret < 1e-8, 0.01, std_ret)

    sentiment = np.clip(mean_ret * 24, -1.0, 1.0)
    trend = np.clip(mean_ret * 20, -0.5, 0.5)
    quant_prob = np.clip(0.5 + trend, 0.0, 1.0)
    fund_score = np.clip(-mean_ret * 30, -1.0, 1.0)
    vol_norm = np.clip((std_ret - 0.01) * 50, -0.5, 0.5)
    regime_flag = np.clip(0.5 + vol_norm, 0.0, 1.0)
    momentum = np.clip(mean_ret * 15, -1.0, 1.0)
    mean_reversion = np.clip(-mean_ret * 40, -1.0, 1.0)
    volatility = np.clip((vol_norm + 1.0) / 2.0, 0.0, 1.0)
    event_risk = np.zeros_like(mean_ret, dtype=np.float32)
    relative_strength = np.zeros_like(mean_ret, dtype=np.float32)
    liquidity = np.full_like(mean_ret, 0.5, dtype=np.float32)

    n = returns_2d.shape[0]
    ret_5d = np.nanmean(returns_2d[-5:] if n >= 5 else returns_2d, axis=0)
    ret_21d = np.nanmean(returns_2d[-21:] if n >= 21 else returns_2d, axis=0)
    vol_21d = np.nanstd(returns_2d[-21:] if n >= 21 else returns_2d, axis=0)
    vol_21d = np.where(vol_21d < 1e-8, 0.01, vol_21d)

    ret_5d_norm = np.clip(ret_5d * 50, -1, 1)
    ret_21d_norm = np.clip(ret_21d * 50, -1, 1)
    vol_21d_norm = np.clip((vol_21d - 0.01) * 50, -1, 1)

    return np.column_stack(
        (
            sentiment,
            quant_prob,
            fund_score,
            regime_flag,
            momentum,
            mean_reversion,
            volatility,
            event_risk,
            relative_strength,
            liquidity,
            ret_5d_norm,
            ret_21d_norm,
            vol_21d_norm,
        )
    )


def proxy_specialists_from_returns_np(
    returns_1d: np.ndarray, start: int, window: int = 21
) -> tuple[float, ...]:
    """Single-asset path; delegates to vectorized (window, 1)."""
    returns_1d = np.asarray(returns_1d, dtype=np.float64)
    lookback = returns_1d[max(0, start - window) : start]
    lookback = lookback[~np.isnan(lookback)]
    if lookback.size == 0:
        return (0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.01)
    col = lookback.astype(np.float32).reshape(-1, 1)
    row = proxy_specialists_vectorized(col)[0]
    return tuple(float(x) for x in row)


def tuple_to_specialist_dict(tup: tuple[float, ...]) -> dict:
    """Convert 13-tuple to dict for TradingEnv / PPO."""
    (
        sentiment,
        quant_prob,
        fund_score,
        regime_flag,
        momentum_score,
        mean_reversion_score,
        volatility_score,
        event_risk_score,
        relative_strength_score,
        liquidity_score,
        ret_5d_norm,
        ret_21d_norm,
        vol_21d_norm,
    ) = tup
    out = {
        "sentiment": float(sentiment),
        "quant_prob": float(quant_prob),
        "fund_score": float(fund_score),
        "regime_flag": float(regime_flag),
        "momentum_score": float(momentum_score),
        "mean_reversion_score": float(mean_reversion_score),
        "volatility_score": float(volatility_score),
        "event_risk_score": float(event_risk_score),
        "relative_strength_score": float(relative_strength_score),
        "liquidity_score": float(liquidity_score),
    }
    if ENV_OBS_ROLLING_FEATURES:
        out["rolling_return_5d"] = float(ret_5d_norm)
        out["rolling_return_21d"] = float(ret_21d_norm)
        out["rolling_vol_21d"] = float(vol_21d_norm)
    return out


def proxy_specialists_from_returns_series(
    returns: pd.Series, start: int, window: int = 21
) -> dict:
    """Pandas Series of returns (aligned index); same semantics as backtest loop."""
    arr = returns.to_numpy(dtype=np.float64)
    tup = proxy_specialists_from_returns_np(arr, start, window)
    return tuple_to_specialist_dict(tup)


def specialist_dicts_from_window(returns_window: np.ndarray) -> list[dict]:
    """
    returns_window shape (window, n_assets).
    Returns list of n_assets dicts for PPO.predict.
    """
    F = proxy_specialists_vectorized(returns_window)
    return [tuple_to_specialist_dict(tuple(float(x) for x in F[i])) for i in range(F.shape[0])]


def precompute_proxy_feature_matrix(returns_matrix: np.ndarray, window: int = 21) -> np.ndarray:
    """
    returns_matrix (n_days, n_assets).
    Returns (n_days, n_assets, 13); rows t < window are zeros (unused in backtest).
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float32)
    n, m = returns_matrix.shape
    out = np.zeros((n, m, 13), dtype=np.float32)
    for t in range(window, n):
        w = returns_matrix[t - window : t, :]
        out[t] = proxy_specialists_vectorized(w)
    return out


def merge_daily_aux_into_proxy_row(
    features_row: np.ndarray, aux: np.ndarray | None
) -> np.ndarray:
    """
    features_row shape (13,) one asset from proxy_specialists_vectorized.
    aux shape (2,) = (atr_pct in [0,0.5] or nan, volume_ratio in [0,5] or nan).
    Maps into relative_strength (8) and liquidity (9) slots when aux is finite.
    """
    if aux is None or aux.size < 2:
        return features_row
    row = features_row.copy()
    atr = aux[0]
    vr = aux[1]
    if np.isfinite(atr):
        row[8] = np.clip((atr / 0.25) * 2.0 - 1.0, -1.0, 1.0)
    if np.isfinite(vr):
        row[9] = np.clip(vr / 5.0, 0.0, 1.0)
    return row
