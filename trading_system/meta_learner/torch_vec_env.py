"""
Batched HistoricalTradingEnv-compatible VecEnv on a single process.

Proxy features and returns live on CUDA for fast batched observation assembly;
per-env stepping mirrors HistoricalTradingEnv (rebalance_multi_asset, PDT).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence, Type

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from ..config import (
    ENV_EPISODE_DAYS,
    ENV_OBS_ROLLING_FEATURES,
    ENV_REWARD_DRAWDOWN_PENALTY,
    ENV_ROLLING_SHARPE_DAYS,
    MIN_HOLD_MINUTES_AFTER_BUY,
    MIN_REENTRY_MINUTES_AFTER_SELL,
    MIN_TRADE_NOTIONAL,
    REBALANCE_BAND_PCT,
    SLIPPAGE_BPS,
    STARTING_CAPITAL,
    TRANSACTION_COST_BPS,
)
from ..utils.pdt import pdt_tracker_from_config, rebalance_multi_asset, session_date_from_index
from .env import _compute_step_reward, _sanitize_obs
from .proxy_specialists import normalize_action_max_unit_exposure

_log = logging.getLogger("boardroom")


class BatchedHistoricalTorchVecEnv(VecEnv):
    """
    Vectorized HistoricalTradingEnv for ``historical_ratio >= 1.0`` (no MixedDataEnv).

    ``returns_array`` and ``proxy_feature_cache`` must match HistoricalTradingEnv
    (proxy shape ``(n_days, n_assets, 13)``).
    """

    def __init__(
        self,
        returns_array: np.ndarray,
        proxy_feature_cache: np.ndarray,
        n_assets: int,
        num_envs: int,
        *,
        episode_days: int | None = None,
        rolling_sharpe_days: int | None = None,
        drawdown_penalty: float | None = None,
        trading_dates: np.ndarray | pd.DatetimeIndex | list | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.returns_np = np.asarray(returns_array, dtype=np.float32)
        self.n_days, na = self.returns_np.shape
        if na != n_assets:
            raise ValueError("returns_array n_assets mismatch")
        pc = np.asarray(proxy_feature_cache, dtype=np.float32)
        if pc.shape != (self.n_days, n_assets, 13):
            raise ValueError(f"proxy_feature_cache must be {(self.n_days, n_assets, 13)}, got {pc.shape}")

        self.n_assets = n_assets
        self.num_envs = int(num_envs)
        self.episode_days = int(episode_days or ENV_EPISODE_DAYS)
        self.rolling_sharpe_days = int(rolling_sharpe_days or ENV_ROLLING_SHARPE_DAYS)
        self.drawdown_penalty = (
            float(drawdown_penalty) if drawdown_penalty is not None else float(ENV_REWARD_DRAWDOWN_PENALTY)
        )

        self.device = torch.device(device)
        self.returns_t = torch.from_numpy(self.returns_np).to(self.device)
        self.proxy_t = torch.from_numpy(pc).to(self.device)

        self._obs_per_asset = 14 if ENV_OBS_ROLLING_FEATURES else 11
        obs_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_per_asset * n_assets,),
            dtype=np.float32,
        )
        act_space = spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32)
        self.render_mode = None
        super().__init__(self.num_envs, obs_space, act_space)
        self.metadata = {"render_modes": []}

        self._trading_dates: pd.DatetimeIndex | None = None
        if trading_dates is not None:
            td = pd.DatetimeIndex(trading_dates)
            if len(td) != self.n_days:
                raise ValueError("trading_dates length must match returns_array n_days")
            self._trading_dates = td

        self._cursor = np.zeros(self.num_envs, dtype=np.int64)
        self._step_in_episode = np.zeros(self.num_envs, dtype=np.int64)
        self._holdings = np.zeros((self.num_envs, self.n_assets), dtype=np.float32)
        self._equity = np.ones(self.num_envs, dtype=np.float64)
        self._equity_peak = np.ones(self.num_envs, dtype=np.float64)
        self._last_buy_bar = np.full((self.num_envs, self.n_assets), -999, dtype=np.int64)
        self._last_sell_bar = np.full((self.num_envs, self.n_assets), -999, dtype=np.int64)
        self._pdt_trackers = [pdt_tracker_from_config() for _ in range(self.num_envs)]
        self._rng = np.random.default_rng()
        self._actions: np.ndarray | None = None

    def _reset_one(self, env_idx: int, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        min_start = 21 + self.rolling_sharpe_days
        max_start = self.n_days - self.episode_days - 1
        if max_start <= min_start:
            c = min_start
        else:
            c = int(self._rng.integers(min_start, max_start + 1))
        self._cursor[env_idx] = c
        self._step_in_episode[env_idx] = 0
        self._holdings[env_idx].fill(0.0)
        self._equity[env_idx] = 1.0
        self._equity_peak[env_idx] = 1.0
        self._last_buy_bar[env_idx].fill(-999)
        self._last_sell_bar[env_idx].fill(-999)
        self._pdt_trackers[env_idx] = pdt_tracker_from_config()

    def reset(self) -> VecEnvObs:
        for i in range(self.num_envs):
            self._reset_one(i, seed=None)
        self._reset_seeds()
        self._reset_options()
        return self._build_obs_batch()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        actions = self._actions
        self._actions = None

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        _day_m = 24 * 60
        min_hold_bars = 1 if MIN_HOLD_MINUTES_AFTER_BUY >= _day_m else 0
        min_reentry_bars = 1 if MIN_REENTRY_MINUTES_AFTER_SELL >= _day_m else 0

        for e in range(self.num_envs):
            self._step_in_episode[e] += 1
            act = normalize_action_max_unit_exposure(np.asarray(actions[e], dtype=np.float32))

            equity = float(self._equity[e])
            portfolio_value = equity * STARTING_CAPITAL
            rebalance_band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
            bar = int(self._cursor[e])
            if self._trading_dates is not None:
                sess = session_date_from_index(self._trading_dates, bar)
            else:
                sess = int(bar)

            h = self._holdings[e].astype(np.float64)
            target = np.asarray(act, dtype=np.float64)
            eh, lb, ls = rebalance_multi_asset(
                h,
                target,
                portfolio_value,
                rebalance_band,
                bar,
                self._last_buy_bar[e],
                self._last_sell_bar[e],
                min_hold_bars,
                min_reentry_bars,
                self._pdt_trackers[e],
                sess,
                portfolio_value,
            )
            self._last_buy_bar[e] = lb
            self._last_sell_bar[e] = ls
            effective_holdings = eh.astype(np.float32)
            turnover = float(np.sum(np.abs(effective_holdings - self._holdings[e])))
            self._holdings[e] = effective_holdings

            info: dict[str, Any] = {}
            if self._cursor[e] >= self.n_days:
                obs_row = self._obs_row_numpy(e)
                rewards[e] = 0.0
                dones[e] = True
                info["return"] = 0.0
                info["TimeLimit.truncated"] = False
                info["terminal_observation"] = obs_row
                self._reset_one(e, seed=None)
                infos.append(info)
                continue

            day_returns = self.returns_np[self._cursor[e]]
            portfolio_return = float(np.dot(effective_holdings, day_returns))
            new_eq = self._equity[e] * (1.0 + portfolio_return)
            self._equity[e] = new_eq
            self._equity_peak[e] = max(self._equity_peak[e], new_eq)

            self._cursor[e] += 1

            current = self._equity[e]
            # Match HistoricalTradingEnv: drawdown only after at least one return realized
            drawdown = (
                (self._equity_peak[e] - current) / (self._equity_peak[e] + 1e-8)
                if self._step_in_episode[e] >= 1
                else 0.0
            )

            rewards[e] = np.float32(
                _compute_step_reward(portfolio_return, turnover, float(drawdown), self.drawdown_penalty)
            )
            terminated = self._step_in_episode[e] >= self.episode_days or self._cursor[e] >= self.n_days
            truncated = False
            dones[e] = terminated or truncated
            info["return"] = portfolio_return
            info["TimeLimit.truncated"] = truncated and not terminated
            if dones[e]:
                obs_row = self._obs_row_numpy(e)
                info["terminal_observation"] = obs_row
                self._reset_one(e, seed=None)
            infos.append(info)

        obs = self._build_obs_batch()
        return obs, rewards, dones, infos

    def _obs_row_numpy(self, env_idx: int) -> np.ndarray:
        """Single-env observation as numpy (for terminal_observation)."""
        c = int(self._cursor[env_idx])
        if c == 0:
            features = np.zeros((self.n_assets, 13), dtype=np.float32)
            features[:, 1] = 0.5
            features[:, 3] = 0.5
            features[:, 6] = 0.5
            features[:, 9] = 0.5
            features[:, 12] = 0.01
        else:
            features = np.asarray(self.proxy_t[c].detach().cpu().numpy(), dtype=np.float32)
        holdings = self._holdings[env_idx]
        if ENV_OBS_ROLLING_FEATURES:
            per_asset = np.column_stack((features[:, :10], holdings, features[:, 10:13]))
        else:
            per_asset = np.column_stack((features[:, :10], holdings))
        return _sanitize_obs(per_asset.flatten().astype(np.float32))

    def _build_obs_batch(self) -> np.ndarray:
        """Stacked observations (num_envs, obs_dim) from CUDA proxy tensor."""
        c = torch.as_tensor(self._cursor, device=self.device, dtype=torch.long)
        # cursor may point past end briefly only before reset; clamp for indexing
        c_safe = torch.clamp(c, 0, self.n_days - 1)
        feats = self.proxy_t[c_safe]
        h = torch.as_tensor(self._holdings, device=self.device, dtype=torch.float32)
        if ENV_OBS_ROLLING_FEATURES:
            p1 = feats[:, :, :10]
            p2 = h.unsqueeze(-1)
            p3 = feats[:, :, 10:13]
            per = torch.cat([p1, p2, p3], dim=-1)
        else:
            per = torch.cat([feats[:, :, :10], h.unsqueeze(-1)], dim=-1)
        obs = per.reshape(self.num_envs, -1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        return obs.detach().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        indices = list(self._get_indices(indices))
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            return [val for _ in indices]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        indices = list(self._get_indices(indices))
        if method_name == "render":
            return [None for _ in indices]
        meth = getattr(self, method_name, None)
        if callable(meth):
            return [meth(*method_args, **method_kwargs) for _ in indices]
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        indices = list(self._get_indices(indices))
        return [False for _ in indices]

    def get_images(self) -> Sequence[Any]:
        return [None for _ in range(self.num_envs)]
