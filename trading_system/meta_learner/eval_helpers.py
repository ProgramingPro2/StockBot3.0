"""Deterministic evaluation helpers shared by training and tuning."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from ..config import PPO_EARLY_STOPPING_OVERFIT_THRESHOLD
from .env import HistoricalTradingEnv


def _seed_everything(seed: int) -> None:
    """Seed numpy, random, torch, and SB3 for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from stable_baselines3.common.utils import set_random_seed as _sb3_seed
        _sb3_seed(seed)
    except Exception:
        pass


def _compute_annualized_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio (sample std, ddof=1, sqrt(252) annualization)."""
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma <= 1e-12:
        return 0.0
    sharpe = (mu / sigma) * math.sqrt(252.0)
    if not np.isfinite(sharpe):
        return 0.0
    return float(sharpe)


def compute_annualized_sharpe(returns: np.ndarray) -> float:
    """Public wrapper: annualized Sharpe (sample std, ddof=1, sqrt(252)) from daily returns."""
    return _compute_annualized_sharpe(np.asarray(returns, dtype=np.float64))


def _predict_actions_deterministic(model: Any, obs: np.ndarray) -> np.ndarray:
    """Fast deterministic policy inference.

    Uses direct SB3 policy forward pass when the model has ``device`` (avoids
    SB3 predict overhead).  Falls back to ``model.predict()`` for lightweight
    test stubs or non-SB3 models.
    """
    if not hasattr(model, "device"):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)
    device = model.device
    obs_t = obs_as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device)
    with torch.inference_mode():
        actions, _, _ = model.policy.forward(obs_t, deterministic=True)
    return actions.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _deterministic_validation_walk_core(
    model: Any,
    env: HistoricalTradingEnv,
    n_assets: int,
) -> tuple[float, np.ndarray]:
    """Run deterministic walk on an existing env (mutates cursor/state). Caller owns env."""
    obs, _ = env.reset(seed=0)
    min_start = int(21 + env.rolling_sharpe_days)
    env._cursor = min(min_start, max(1, env.n_days - 2))
    env._step_in_episode = 0
    env._returns_history = []
    env._equity_history = [1.0]
    env._equity_peak = 1.0
    env._holdings = np.zeros(n_assets, dtype=np.float32)
    obs = env._get_obs()

    daily_returns: list[float] = []
    while env._cursor < env.n_days - 1:
        action = _predict_actions_deterministic(model, obs)
        obs, _, terminated, truncated, info = env.step(action)
        daily_returns.append(float(info.get("return", 0.0)))
        if terminated or truncated:
            break

    ret_arr = np.asarray(daily_returns, dtype=np.float32)
    return _compute_annualized_sharpe(ret_arr), ret_arr


def deterministic_validation_walk(
    model: Any,
    val_returns: np.ndarray,
    n_assets: int,
    daily_aux_features: np.ndarray | None = None,
    trading_dates: np.ndarray | pd.DatetimeIndex | list | None = None,
    *,
    eval_env: HistoricalTradingEnv | None = None,
    proxy_feature_cache: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate model on a deterministic full validation walk.

    If ``eval_env`` is provided, reuses that env (no close); otherwise constructs a temporary env.
    ``proxy_feature_cache`` is passed to ``HistoricalTradingEnv`` when constructing a new env.
    """
    returns = np.asarray(val_returns, dtype=np.float32)
    if returns.ndim != 2 or returns.shape[1] != n_assets:
        raise ValueError("val_returns must have shape (n_days, n_assets)")
    if returns.shape[0] <= 2:
        return 0.0, np.asarray([], dtype=np.float32)

    own_env = eval_env is None
    env = eval_env
    if own_env:
        env = HistoricalTradingEnv(
            returns_array=returns,
            n_assets=n_assets,
            episode_days=max(1, returns.shape[0] - 1),
            daily_aux_features=daily_aux_features,
            trading_dates=trading_dates,
            proxy_feature_cache=proxy_feature_cache,
        )

    try:
        return _deterministic_validation_walk_core(model, env, n_assets)
    finally:
        if own_env and env is not None:
            env.close()


class SharpeEvalCallback(BaseCallback):
    """Evaluate policy by deterministic validation Sharpe at fixed cadence."""

    def __init__(
        self,
        val_returns: np.ndarray,
        n_assets: int,
        eval_freq: int,
        *,
        min_delta: float = 0.0,
        best_model_save_path: Path | str | None = None,
        daily_aux_features: np.ndarray | None = None,
        trading_dates: np.ndarray | pd.DatetimeIndex | list | None = None,
        proxy_feature_cache: np.ndarray | None = None,
        optuna_trial: Any | None = None,
        patience: int = 0,
        overfit_threshold: float | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.val_returns = np.asarray(val_returns, dtype=np.float32)
        self.n_assets = n_assets
        self.eval_freq = int(eval_freq)
        self.min_delta = float(min_delta)
        self.patience = max(0, int(patience))
        # Use config default when not explicitly overridden; 0 disables the check.
        if overfit_threshold is None:
            overfit_threshold = PPO_EARLY_STOPPING_OVERFIT_THRESHOLD
        self.overfit_threshold = max(0.0, float(overfit_threshold))
        self._stagnant_evals = 0
        self.daily_aux_features = daily_aux_features
        self.trading_dates = trading_dates
        self.optuna_trial = optuna_trial
        self.best_model_save_path = (
            Path(best_model_save_path) if best_model_save_path is not None else None
        )
        self.last_mean_reward: float | None = None
        self.best_mean_reward: float | None = None
        self.eval_idx = 0
        self.has_improved = False
        self._eval_env = HistoricalTradingEnv(
            returns_array=self.val_returns,
            n_assets=n_assets,
            episode_days=max(1, self.val_returns.shape[0] - 1),
            daily_aux_features=daily_aux_features,
            trading_dates=trading_dates,
            proxy_feature_cache=proxy_feature_cache,
        )

    def close_eval_env(self) -> None:
        """Close the persistent validation env (idempotent)."""
        env = getattr(self, "_eval_env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self._eval_env = None

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        self.eval_idx += 1
        sharpe, _ = _deterministic_validation_walk_core(
            self.model,
            self._eval_env,
            self.n_assets,
        )
        self.last_mean_reward = float(sharpe)

        if self.optuna_trial is not None:
            import optuna

            self.optuna_trial.report(float(sharpe), self.eval_idx)
            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned()

        improved = (
            self.best_mean_reward is None
            or self.last_mean_reward > (self.best_mean_reward + self.min_delta)
        )
        if improved:
            self.best_mean_reward = self.last_mean_reward
            self.has_improved = True
            self._stagnant_evals = 0
            if self.best_model_save_path is not None:
                self.best_model_save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_save_path))
        else:
            # Hard stop if val Sharpe has degraded sharply below its best (overfitting signal).
            if (
                self.overfit_threshold > 0.0
                and self.best_mean_reward is not None
                and self.last_mean_reward < (self.best_mean_reward - self.overfit_threshold)
            ):
                return False
            self._stagnant_evals += 1
            if self.patience > 0 and self._stagnant_evals >= self.patience:
                return False
        return True

    def _on_training_end(self) -> None:
        self.close_eval_env()
