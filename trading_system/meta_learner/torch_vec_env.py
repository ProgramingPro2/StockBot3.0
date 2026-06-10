"""
Batched HistoricalTradingEnv-compatible VecEnv on a single process.

Proxy features and returns live on CUDA; env stepping is fully vectorized across
``(num_envs, n_assets)`` tensors. Observations are copied to CPU numpy only for SB3.
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
    ENV_IMF_RESTRICTIONS,
    ENV_OBS_ROLLING_FEATURES,
    ENV_REWARD_DRAWDOWN_PENALTY,
    ENV_ROLLING_SHARPE_DAYS,
    IMF_INTRADAY_MULTIPLIER,
    IMF_MIN_INTRADAY_EQUITY,
    MIN_HOLD_MINUTES_AFTER_BUY,
    MIN_REENTRY_MINUTES_AFTER_SELL,
    MIN_TRADE_NOTIONAL,
    REBALANCE_BAND_PCT,
    SLIPPAGE_BPS,
    STARTING_CAPITAL,
    TRANSACTION_COST_BPS,
)
from ..utils.imf import rebalance_multi_asset_batched
from .env import _sanitize_obs

_log = logging.getLogger("boardroom")

_COST_FRAC = float(TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0


def _normalize_action_batched(actions: torch.Tensor) -> torch.Tensor:
    a = actions.clamp(0.0, 1.0)
    s = a.sum(dim=-1, keepdim=True)
    mask = s > 1.0 + 1e-8
    return torch.where(mask, a / s.clamp(min=1e-8), a)


def _compute_step_reward_batched(
    portfolio_return: torch.Tensor,
    turnover: torch.Tensor,
    drawdown: torch.Tensor,
    drawdown_penalty: float,
) -> torch.Tensor:
    utility = torch.log(torch.clamp(1.0 + portfolio_return, min=1e-8))
    reward = utility - drawdown_penalty * drawdown - turnover * _COST_FRAC
    reward = torch.nan_to_num(reward, nan=0.0, posinf=3.0, neginf=-3.0)
    return reward.clamp(-3.0, 3.0)


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
        obs_noise_std: float = 0.0,
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

        self._min_start = 21 + self.rolling_sharpe_days
        self._max_start = max(self._min_start, self.n_days - self.episode_days - 1)

        _day_m = 24 * 60
        self._min_hold_bars = 1 if MIN_HOLD_MINUTES_AFTER_BUY >= _day_m else 0
        self._min_reentry_bars = 1 if MIN_REENTRY_MINUTES_AFTER_SELL >= _day_m else 0
        self._obs_noise_std = float(obs_noise_std)

        self._cursor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._step_in_episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._holdings = torch.zeros((self.num_envs, self.n_assets), device=self.device, dtype=torch.float32)
        self._equity = torch.ones(self.num_envs, device=self.device, dtype=torch.float64)
        self._equity_peak = torch.ones(self.num_envs, device=self.device, dtype=torch.float64)
        self._last_buy_bar = torch.full(
            (self.num_envs, self.n_assets), -999, device=self.device, dtype=torch.long
        )
        self._last_sell_bar = torch.full(
            (self.num_envs, self.n_assets), -999, device=self.device, dtype=torch.long
        )
        self._rng = np.random.default_rng()
        self._actions: np.ndarray | None = None

    def _reset_indices(self, indices: torch.Tensor, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n = int(indices.numel())
        if self._max_start <= self._min_start:
            starts = np.full(n, self._min_start, dtype=np.int64)
        else:
            starts = self._rng.integers(self._min_start, self._max_start + 1, size=n, dtype=np.int64)
        starts_t = torch.as_tensor(starts, device=self.device, dtype=torch.long)
        self._cursor[indices] = starts_t
        self._step_in_episode[indices] = 0
        self._holdings[indices] = 0.0
        self._equity[indices] = 1.0
        self._equity_peak[indices] = 1.0
        self._last_buy_bar[indices] = -999
        self._last_sell_bar[indices] = -999

    def _reset_one(self, env_idx: int, seed: int | None = None) -> None:
        self._reset_indices(torch.tensor([env_idx], device=self.device, dtype=torch.long), seed=seed)

    def reset(self) -> VecEnvObs:
        all_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_indices(all_idx, seed=None)
        self._reset_seeds()
        self._reset_options()
        return self._build_obs_batch()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        actions = torch.as_tensor(self._actions, device=self.device, dtype=torch.float32)
        self._actions = None

        self._step_in_episode += 1
        act = _normalize_action_batched(actions)

        old_holdings = self._holdings.clone()
        equity = self._equity
        portfolio_value = equity * float(STARTING_CAPITAL)
        rebalance_band = torch.maximum(
            torch.tensor(float(MIN_TRADE_NOTIONAL), device=self.device, dtype=portfolio_value.dtype),
            portfolio_value * float(REBALANCE_BAND_PCT),
        )

        effective_holdings, last_buy, last_sell = rebalance_multi_asset_batched(
            self._holdings,
            act,
            portfolio_value,
            rebalance_band,
            self._cursor,
            self._last_buy_bar,
            self._last_sell_bar,
            self._min_hold_bars,
            self._min_reentry_bars,
            imf_restrictions_enabled=ENV_IMF_RESTRICTIONS,
            min_intraday_equity=float(IMF_MIN_INTRADAY_EQUITY),
            intraday_multiplier=float(IMF_INTRADAY_MULTIPLIER),
            equity_dollars=portfolio_value,
        )
        turnover = (effective_holdings - old_holdings).abs().sum(dim=-1)
        self._holdings = effective_holdings
        self._last_buy_bar = last_buy
        self._last_sell_bar = last_sell

        past_end = self._cursor >= self.n_days
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        dones = past_end.clone()

        valid = ~past_end
        if valid.any():
            c = self._cursor[valid].clamp(0, self.n_days - 1)
            day_returns = self.returns_t[c]
            portfolio_return = (effective_holdings[valid] * day_returns).sum(dim=-1).to(torch.float64)
            new_eq = self._equity[valid] * (1.0 + portfolio_return)
            self._equity[valid] = new_eq
            self._equity_peak[valid] = torch.maximum(self._equity_peak[valid], new_eq)
            self._cursor[valid] += 1

            drawdown = torch.where(
                self._step_in_episode[valid] >= 1,
                (self._equity_peak[valid] - self._equity[valid]) / (self._equity_peak[valid] + 1e-8),
                torch.zeros_like(new_eq),
            )
            rewards[valid] = _compute_step_reward_batched(
                portfolio_return.to(torch.float32),
                turnover[valid].to(torch.float32),
                drawdown.to(torch.float32),
                self.drawdown_penalty,
            )

            terminated = (self._step_in_episode[valid] >= self.episode_days) | (
                self._cursor[valid] >= self.n_days
            )
            dones[valid] = terminated

        portfolio_return_all = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if valid.any():
            c = self._cursor[valid] - 1
            c = c.clamp(0, self.n_days - 1)
            pr = (effective_holdings[valid] * self.returns_t[c]).sum(dim=-1)
            portfolio_return_all[valid] = pr.to(torch.float32)

        infos: list[dict[str, Any]] = []
        # Build obs before resetting done envs — these are the terminal observations.
        terminal_obs = self._build_obs_batch()
        portfolio_returns_np = portfolio_return_all.detach().cpu().numpy()

        for e in range(self.num_envs):
            info: dict[str, Any] = {"return": float(portfolio_returns_np[e])}
            info["TimeLimit.truncated"] = False
            if past_end[e]:
                info["return"] = 0.0
                info["terminal_observation"] = terminal_obs[e].copy()
                self._reset_one(e, seed=None)
            elif bool(dones[e].item()):
                info["terminal_observation"] = terminal_obs[e].copy()
                self._reset_one(e, seed=None)
            infos.append(info)

        # Rebuild obs after resets: done envs now show post-reset state (SB3 VecEnv convention).
        obs_batch = self._build_obs_batch()
        rewards_np = rewards.detach().cpu().numpy().astype(np.float32)
        dones_np = dones.detach().cpu().numpy().astype(bool)
        return obs_batch, rewards_np, dones_np, infos

    def _build_obs_batch(self) -> np.ndarray:
        """Stacked observations (num_envs, obs_dim) from CUDA proxy tensor."""
        c_safe = self._cursor.clamp(0, self.n_days - 1)
        feats = self.proxy_t[c_safe]
        h = self._holdings
        if ENV_OBS_ROLLING_FEATURES:
            per = torch.cat([feats[:, :, :10], h.unsqueeze(-1), feats[:, :, 10:13]], dim=-1)
        else:
            per = torch.cat([feats[:, :, :10], h.unsqueeze(-1)], dim=-1)
        obs = per.reshape(self.num_envs, -1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        if self._obs_noise_std > 0.0:
            noise = torch.randn_like(obs) * self._obs_noise_std
            obs = (obs + noise).clamp(-1.0, 1.0)
        return _sanitize_obs(obs.detach().cpu().numpy().astype(np.float32))

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
