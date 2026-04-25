"""
Shared helpers for building training/tuning envs.
Provides a picklable proxy specialist fn from historical returns (for SubprocVecEnv).
"""
import random
from typing import Callable

import numpy as np

from .proxy_specialists import proxy_specialists_from_returns_np, tuple_to_specialist_dict


def _tuple_to_specialist_dict(tup: tuple) -> dict:
    """Convert proxy tuple to dict for TradingEnv (re-export)."""
    return tuple_to_specialist_dict(tup)


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
            tuple_to_specialist_dict(
                proxy_specialists_from_returns_np(
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
