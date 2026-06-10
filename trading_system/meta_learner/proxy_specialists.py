"""
Shared proxy-from-returns specialist features for env, training, tuning, and backtest.
Single source of truth to avoid drift between numpy/pandas code paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ENV_OBS_ROLLING_FEATURES, PROXY_XSRANK_ENABLED, PROXY_EXPANDING_ZSCORE


def _cross_sectional_rank(x: np.ndarray) -> np.ndarray:
    """Rank values cross-sectionally across assets; returns percentile rank in [-1, 1].
    With only 1 asset there is nothing to rank against, so returns zeros."""
    n = len(x)
    if n <= 1:
        return np.zeros_like(x, dtype=np.float32)
    ranks = np.argsort(np.argsort(x)).astype(np.float32)
    return ((ranks / max(n - 1, 1)) * 2.0 - 1.0).astype(np.float32)


def _expanding_zscore_3d(matrix: np.ndarray, min_periods: int = 63) -> np.ndarray:
    """
    Apply expanding-window z-score normalization along the time axis (axis 0).
    Uses Welford-style cumulative stats for O(n) time complexity.
    Shape: (n_days, n_assets, n_features). Returns values clipped to [-1, 1].
    Before min_periods is reached, returns the original values (no normalization applied).
    """
    out = matrix.copy().astype(np.float64)
    mat = matrix.astype(np.float64)
    running_sum = np.zeros_like(mat[0])
    running_sq  = np.zeros_like(mat[0])
    for t in range(len(mat)):
        running_sum += mat[t]
        running_sq  += mat[t] ** 2
        count = t + 1
        if count >= min_periods:
            mu    = running_sum / count
            var   = running_sq / count - mu ** 2
            sigma = np.sqrt(np.maximum(var, 1e-8))
            out[t] = np.clip((mat[t] - mu) / sigma, -3.0, 3.0) / 3.0
    return out.astype(np.float32)


def normalize_action_max_unit_exposure(action: np.ndarray) -> np.ndarray:
    """Clip to [0,1] per asset; if sum > 1, scale so total exposure is 1 (matches live/backtest)."""
    a = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    s = float(np.sum(a))
    if s > 1.0 + 1e-8:
        a = (a / s).astype(np.float32)
    return a


def sentiment_proxy_from_close_prices(close: np.ndarray | pd.Series) -> float:
    """
    Momentum sentiment proxy from a close-price series (point-in-time: use only this slice).
    Matches live fallback in ``_sentiment_proxy_from_ohlcv`` (mean of last 21 daily returns, scaled).
    """
    s = pd.Series(close, dtype=float) if isinstance(close, np.ndarray) else close.astype(float)
    if s is None or len(s) < 21:
        return 0.0
    ret = s.pct_change().dropna().tail(21)
    if ret.empty:
        return 0.0
    mean_ret = float(ret.mean())
    return float(max(-0.5, min(0.5, mean_ret * 12)))


def sentiment_proxy_batch_from_returns_block(rets: np.ndarray) -> np.ndarray:
    """
    Vectorized sentiment proxy for all assets at once.
    rets shape (n_days, n_assets) daily returns used to build cumulative closes from 1.0 (same as per-column loop).
    Returns shape (n_assets,) float64.
    """
    r = np.asarray(rets, dtype=np.float64)
    if r.size == 0:
        return np.zeros(r.shape[1] if r.ndim == 2 else 0, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("rets must be 2d")
    n, m = r.shape
    if n == 0:
        return np.zeros(m, dtype=np.float64)
    r = np.nan_to_num(r, nan=0.0)
    ones = np.ones((1, m), dtype=np.float64)
    closes = np.cumprod(np.vstack([ones, 1.0 + r]), axis=0)
    if closes.shape[0] < 3:
        return np.zeros(m, dtype=np.float64)
    dr = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-12)
    last = dr[-21:] if dr.shape[0] >= 21 else dr
    mu = np.mean(last, axis=0)
    return np.clip(mu * 12.0, -0.5, 0.5).astype(np.float64)


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

    if PROXY_XSRANK_ENABLED and returns_2d.shape[1] > 1:
        sentiment = _cross_sectional_rank(sentiment)
        momentum = _cross_sectional_rank(momentum)
        quant_prob = np.clip((_cross_sectional_rank(np.clip(mean_ret * 20, -0.5, 0.5)) + 1.0) / 2.0, 0.0, 1.0)
        relative_strength = _cross_sectional_rank(mean_ret)

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


def proxy_specialists_vectorized_batch(windows: np.ndarray) -> np.ndarray:
    """
    Batch over trailing windows. windows shape (B, window, n_assets).
    Returns (B, n_assets, 13).
    """
    win = np.asarray(windows, dtype=np.float32)
    if win.ndim != 3:
        raise ValueError("windows must have shape (batch, window, n_assets)")
    B, W, M = win.shape
    mean_ret = np.nanmean(win, axis=1).astype(np.float32)
    std_ret = np.nanstd(win, axis=1).astype(np.float32)
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
    event_risk = np.zeros((B, M), dtype=np.float32)
    relative_strength = np.zeros((B, M), dtype=np.float32)
    liquidity = np.full((B, M), 0.5, dtype=np.float32)

    tail5 = win[:, -5:, :] if W >= 5 else win
    tail21 = win[:, -21:, :] if W >= 21 else win
    ret_5d = np.nanmean(tail5, axis=1)
    ret_21d = np.nanmean(tail21, axis=1)
    vol_21d = np.nanstd(tail21, axis=1)
    vol_21d = np.where(vol_21d < 1e-8, 0.01, vol_21d)

    ret_5d_norm = np.clip(ret_5d * 50, -1, 1)
    ret_21d_norm = np.clip(ret_21d * 50, -1, 1)
    vol_21d_norm = np.clip((vol_21d - 0.01) * 50, -1, 1)

    if PROXY_XSRANK_ENABLED and M > 1:
        sentiment = np.apply_along_axis(_cross_sectional_rank, axis=1, arr=sentiment)
        momentum = np.apply_along_axis(_cross_sectional_rank, axis=1, arr=momentum)
        _trend_ranked = np.apply_along_axis(_cross_sectional_rank, axis=1, arr=np.clip(mean_ret * 20, -0.5, 0.5))
        quant_prob = np.clip((_trend_ranked + 1.0) / 2.0, 0.0, 1.0)
        relative_strength = np.apply_along_axis(_cross_sectional_rank, axis=1, arr=mean_ret)

    return np.stack(
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
        ),
        axis=-1,
    ).astype(np.float32)


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
    from numpy.lib.stride_tricks import sliding_window_view

    returns_matrix = np.asarray(returns_matrix, dtype=np.float32)
    n, m = returns_matrix.shape
    out = np.zeros((n, m, 13), dtype=np.float32)
    if n <= window:
        return out
    # sliding_window_view(..., axis=0) on (n, m) yields (n-window+1, m, window); we need (n-window+1, window, m).
    win_mwt = sliding_window_view(returns_matrix, window_shape=window, axis=0)
    win = np.transpose(win_mwt, (0, 2, 1))
    n_feat = n - window
    batch = win[:n_feat]
    out[window:] = proxy_specialists_vectorized_batch(batch)
    if PROXY_EXPANDING_ZSCORE:
        out = _expanding_zscore_3d(out)
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


def merge_daily_aux_into_proxy_matrix(features: np.ndarray, aux_row: np.ndarray | None) -> np.ndarray:
    """
    Vectorized aux merge. features shape (n_assets, 13), aux_row shape (n_assets, 2).
    Returns a new array (does not mutate features).
    """
    if aux_row is None or aux_row.size == 0:
        return features
    out = np.asarray(features, dtype=np.float32).copy()
    atr = aux_row[:, 0]
    vr = aux_row[:, 1]
    mask_atr = np.isfinite(atr)
    out[mask_atr, 8] = np.clip((atr[mask_atr] / 0.25) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)
    mask_vr = np.isfinite(vr)
    out[mask_vr, 9] = np.clip(vr[mask_vr] / 5.0, 0.0, 1.0).astype(np.float32)
    return out
