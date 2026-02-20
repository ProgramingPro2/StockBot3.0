"""
Shared helpers for building training/tuning envs.
Provides a picklable proxy specialist fn from historical returns (for SubprocVecEnv).
"""
import random
from typing import Callable

import numpy as np

from ..config import ENV_OBS_ROLLING_FEATURES
from .env import _proxy_specialists_from_returns_np


def _tuple_to_specialist_dict(
    tup: tuple,
) -> dict:
    """Convert _proxy_specialists_from_returns_np tuple to dict for TradingEnv."""
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


class ProxySpecialistFn:
    """
    Picklable callable that yields proxy specialist outputs from historical returns.
    Used by training and tuning with SubprocVecEnv.
    """

    def __init__(self, returns_array: np.ndarray, n_assets: int, window: int = 21):
        self.returns_array = np.asarray(returns_array, dtype=np.float32)
        self.n_assets = n_assets
        self.window = window
        self.n_days = self.returns_array.shape[0]

    def __call__(self) -> list[dict]:
        start = (
            random.randint(self.window, self.n_days - 1)
            if self.n_days > self.window
            else self.window
        )
        return [
            _tuple_to_specialist_dict(
                _proxy_specialists_from_returns_np(
                    self.returns_array[:, i], start, self.window
                )
            )
            for i in range(self.n_assets)
        ]


def create_proxy_specialist_fn(
    returns_array: np.ndarray, n_assets: int, window: int = 21
) -> Callable[[], list[dict]]:
    """Return a picklable callable that yields proxy specialist outputs from historical returns."""
    return ProxySpecialistFn(returns_array, n_assets, window)
