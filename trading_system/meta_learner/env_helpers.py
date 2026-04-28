"""
Shared helpers for building training/tuning envs.
Provides a picklable proxy specialist fn from historical returns (for SubprocVecEnv).
"""
import random
from typing import Callable

import numpy as np
import pandas as pd

from ..config import sentiment_refresh_every_n_bars
from .proxy_specialists import (
    proxy_specialists_from_returns_np,
    sentiment_proxy_from_close_prices,
    tuple_to_specialist_dict,
)


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
        # Approximate live proxy-sentiment staleness (same cadence as backtest; no Ollama).
        self._last_sentiment_vec: np.ndarray | None = None

    def __call__(self) -> list[dict]:
        start = (
            random.randint(self.window, self.n_days - 1)
            if self.n_days > self.window
            else self.window
        )
        n_every = sentiment_refresh_every_n_bars()
        refresh = self._last_sentiment_vec is None or random.random() < (1.0 / n_every)
        if refresh:
            sentiments: list[float] = []
            for i in range(self.n_assets):
                # Synthetic cumulative close from daily returns up to ``start`` (scale-free for pct_change).
                rets = self.returns_array[1 : start + 1, i]
                if len(rets) == 0:
                    sentiments.append(0.0)
                else:
                    r = np.nan_to_num(rets.astype(np.float64), nan=0.0)
                    closes = np.cumprod(np.concatenate([[1.0], 1.0 + r]))
                    sentiments.append(sentiment_proxy_from_close_prices(pd.Series(closes)))
            self._last_sentiment_vec = np.array(sentiments, dtype=np.float64)
        out: list[dict] = []
        for i in range(self.n_assets):
            d = tuple_to_specialist_dict(
                proxy_specialists_from_returns_np(
                    self.returns_array[:, i], start, self.window
                )
            )
            d["sentiment"] = float(self._last_sentiment_vec[i])
            out.append(d)
        return out


def create_proxy_specialist_fn(
    returns_array: np.ndarray, n_assets: int, window: int = 21
) -> Callable[[], list[dict]]:
    """Return a picklable callable that yields proxy specialist outputs from historical returns."""
    return ProxySpecialistFn(returns_array, n_assets, window)
