"""
PPO meta-learner via stable-baselines3.
Weights specialist opinions to decide position sizing.
"""
import logging
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

# When training runs this many timesteps or more, an additional dated copy is saved.
LONG_RUN_TIMESTEPS = 1_000_000

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..config import (
    ENV_EPISODE_DAYS,
    ENV_OBS_ROLLING_FEATURES,
    META_LEARNER_MODEL_DIR,
    PPO_CHECKPOINT_FREQ,
    PPO_CHECKPOINT_KEEP_LAST,
    PPO_CPU_THREADS,
    PPO_EARLY_STOPPING_EVAL_FREQ,
    PPO_EARLY_STOPPING_MIN_DELTA,
    PPO_EARLY_STOPPING_PATIENCE,
    PPO_WEIGHT_DECAY,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_NET_ARCH,
    PPO_N_ENVS,
    PPO_N_STEPS,
)
from ..utils.graceful_shutdown import should_stop
from ..utils.output_cleanup import prune_checkpoints
from .env import HistoricalTradingEnv, MixedDataEnv, TradingEnv
from .eval_helpers import SharpeEvalCallback
from .metrics_callback import JsonlMetricsCallback, metrics_path
from .proxy_specialists import normalize_action_max_unit_exposure
from .runs_index import (
    make_run_id,
    record_end,
    record_merge_extra,
    record_start,
    tb_log_dir_for,
)


class GracefulShutdownCallback(BaseCallback):
    """Stop SB3 training cleanly when a graceful shutdown is requested.

    Returning ``False`` from ``_on_step`` causes ``model.learn()`` to exit
    after the current rollout, which lets the caller run their post-training
    save / notification path before the process terminates.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._announced = False

    def _on_step(self) -> bool:
        if should_stop():
            if not self._announced:
                self._announced = True
                if self.verbose > 0:
                    logging.getLogger("boardroom").info(
                        "GracefulShutdown: stop requested; ending training after current rollout"
                    )
            return False
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when eval reward has not improved for `patience` consecutive evaluations.
    Requires an EvalCallback to run first in the callback list so last_mean_reward is set.
    Only counts no-improvement on steps where a new evaluation was just run (same cadence as eval_freq).
    """

    def __init__(
        self,
        eval_callback: BaseCallback,
        patience: int = 5,
        min_delta: float = 0.01,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward: float | None = None
        self.no_improvement_count = 0
        self._last_eval_n_calls = -1

    def _on_step(self) -> bool:
        # Only act when a new evaluation just ran (EvalCallback runs when n_calls % eval_freq == 0)
        eval_freq = getattr(self.eval_callback, "eval_freq", 0)
        if eval_freq <= 0 or self.eval_callback.n_calls % eval_freq != 0:
            return True
        if self.eval_callback.last_mean_reward is None:
            return True
        # Avoid counting the same eval twice
        if self.eval_callback.n_calls == self._last_eval_n_calls:
            return True
        self._last_eval_n_calls = self.eval_callback.n_calls
        current = self.eval_callback.last_mean_reward
        if not np.isfinite(current):
            if self.verbose > 0:
                logging.getLogger("boardroom").warning(
                    "EarlyStopping: skipping non-finite eval reward %.4f", current
                )
            return True
        if self.best_reward is None:
            self.best_reward = current
            self.no_improvement_count = 0
            if self.verbose > 0:
                logging.getLogger("boardroom").info(
                    "EarlyStopping: initial eval reward %.4f", current
                )
            return True
        if current > self.best_reward + self.min_delta:
            self.best_reward = current
            self.no_improvement_count = 0
            if self.verbose > 0:
                logging.getLogger("boardroom").info(
                    "EarlyStopping: new best eval reward %.4f", current
                )
            return True
        self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                logging.getLogger("boardroom").info(
                    "EarlyStopping: no improvement for %d evals (best=%.4f), stopping",
                    self.patience,
                    self.best_reward,
                )
            return False
        return True


def _random_specialist_outputs(n_assets: int, rng: np.random.Generator | None = None) -> list[dict]:
    """Random specialist outputs; use rng for reproducibility."""
    rng = rng or np.random.default_rng()
    return [
        {
            "sentiment": float(rng.uniform(-1, 1)),
            "quant_prob": float(rng.uniform(0, 1)),
            "fund_score": float(rng.uniform(-1, 1)),
            "regime_flag": float(rng.uniform(0, 1)),
            "momentum_score": float(rng.uniform(-1, 1)),
            "mean_reversion_score": float(rng.uniform(-1, 1)),
            "volatility_score": float(rng.uniform(0, 1)),
            "event_risk_score": float(rng.uniform(0, 1)),
            "relative_strength_score": float(rng.uniform(-1, 1)),
            "liquidity_score": float(rng.uniform(0, 1)),
        }
        for _ in range(n_assets)
    ]


class BootstrapReturnSampler:
    """Picklable: sample contiguous return blocks from historical data for synthetic TradingEnv."""

    def __init__(self, returns_array: np.ndarray, n_assets: int, episode_days: int, seed: int = 0):
        self.returns_array = np.asarray(returns_array, dtype=np.float32)
        self.n_rows = int(self.returns_array.shape[0])
        self.n_assets = int(n_assets)
        self.episode_days = int(episode_days)
        self._rng = np.random.default_rng(seed)

    def sample_block(self) -> np.ndarray:
        need = self.episode_days + 1
        if self.n_rows < need:
            return np.zeros((need, self.n_assets), dtype=np.float32)
        max_start = self.n_rows - need
        s = int(self._rng.integers(0, max_start + 1))
        return self.returns_array[s : s + need].copy()


class SpecialistWrapper(TradingEnv):
    """
    TradingEnv that injects specialist outputs on reset.
    Module-level for picklability with SubprocVecEnv.
    """

    def __init__(
        self,
        n_assets: int,
        specialist_fn: Callable[[], list[dict]] | None = None,
        bootstrap_sampler: BootstrapReturnSampler | None = None,
    ):
        super().__init__(n_assets=n_assets)
        self._specialist_fn = specialist_fn
        self._bootstrap_sampler = bootstrap_sampler

    def reset(self, seed: int | None = None, options: dict | None = None):
        opts = dict(options or {})
        rng = np.random.default_rng(seed if seed is not None else 0)
        if self._specialist_fn is not None:
            outputs = self._specialist_fn()
        else:
            outputs = _random_specialist_outputs(self.n_assets, rng)
        if isinstance(outputs, dict):
            outputs = [outputs] * self.n_assets
        outputs = (list(outputs) + [{}] * self.n_assets)[: self.n_assets]
        opts["specialist_outputs"] = outputs
        if self._bootstrap_sampler is not None:
            opts["bootstrap_returns"] = self._bootstrap_sampler.sample_block()
        return super().reset(seed=seed, options=opts)


def _make_env_fn(
    mode: str,
    historical_returns: np.ndarray | None,
    n_assets: int,
    historical_ratio: float,
    specialist_fn: Callable[[], list[dict]] | None,
    seed: int,
    daily_aux_features: np.ndarray | None = None,
    historical_trading_dates: Any = None,
) -> gym.Env:
    """
    Picklable factory that creates one env for SubprocVecEnv.
    mode: "historical" | "mixed" | "synthetic"
    """
    bootstrap_sampler: BootstrapReturnSampler | None = None
    if historical_returns is not None and mode in ("synthetic", "mixed"):
        bootstrap_sampler = BootstrapReturnSampler(
            historical_returns, n_assets, ENV_EPISODE_DAYS, seed
        )

    if mode == "historical" and historical_returns is not None:
        env = HistoricalTradingEnv(
            returns_array=historical_returns,
            n_assets=n_assets,
            daily_aux_features=daily_aux_features,
            trading_dates=historical_trading_dates,
        )
    elif mode == "mixed" and historical_returns is not None and 0 < historical_ratio < 1:
        hist_env = HistoricalTradingEnv(
            returns_array=historical_returns,
            n_assets=n_assets,
            daily_aux_features=daily_aux_features,
            trading_dates=historical_trading_dates,
        )
        synth_env = SpecialistWrapper(
            n_assets=n_assets,
            specialist_fn=specialist_fn,
            bootstrap_sampler=bootstrap_sampler,
        )
        env = MixedDataEnv(
            historical_env=hist_env,
            synthetic_env=synth_env,
            historical_ratio=historical_ratio,
        )
    else:
        env = SpecialistWrapper(
            n_assets=n_assets,
            specialist_fn=specialist_fn,
            bootstrap_sampler=bootstrap_sampler,
        )

    env.reset(seed=seed)
    return env


def _get_ppo_device() -> str:
    """PPO uses CPU (SB3 recommends CPU for MLP policy)."""
    from ..config import PPO_DEVICE
    if PPO_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_model_path(path: Path | str | None) -> Path | None:
    """Resolve model path: if not absolute, look under META_LEARNER_MODEL_DIR."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = META_LEARNER_MODEL_DIR / p.name
    return p if p.exists() else None


class PPOTrader:
    """PPO-based meta-learner for position sizing."""

    def __init__(
        self,
        n_assets: int = 1,
        load: bool = True,
        model_path: Path | str | None = None,
    ):
        self.n_assets = n_assets
        self.env = TradingEnv(n_assets=n_assets)
        self.model: PPO | None = None

        resolved = _resolve_model_path(model_path) if model_path else None
        if resolved is not None:
            try:
                self.model = PPO.load(str(resolved), env=self.env, device=_get_ppo_device())
            except Exception:
                self.model = None
        elif load:
            path = META_LEARNER_MODEL_DIR / "ppo_trader.zip"
            if path.exists():
                try:
                    self.model = PPO.load(str(path), env=self.env, device=_get_ppo_device())
                except Exception:
                    self.model = None

    def train(
        self,
        total_timesteps: int = 500_000,
        specialist_fn=None,
        historical_returns: np.ndarray | None = None,
        historical_ratio: float = 0.0,
        validation_returns: np.ndarray | None = None,
        ppo_kwargs: dict[str, Any] | None = None,
        n_envs: int | None = None,
        daily_aux_features: np.ndarray | None = None,
        historical_trading_dates: Any = None,
        validation_trading_dates: Any = None,
    ) -> Path | None:
        """
        Train PPO on simulated or historical episodes.
        specialist_fn: callable() -> list[dict] of length n_assets; each dict has
            sentiment, quant_prob, fund_score, regime_flag.
        historical_returns: if provided, shape (n_days, n_assets), used for HistoricalTradingEnv.
        historical_ratio: fraction of episodes on historical data (0.0 = synthetic only,
            1.0 = historical only, 0.5 = 50/50 mixed).
        validation_returns: if provided, used for early-stopping eval (out-of-sample vs training).
        ppo_kwargs: optional overrides for PPO hyperparameters (e.g. from tuning).
        n_envs: parallel env count for SubprocVecEnv; None = use PPO_N_ENVS.
        """
        kwargs = dict(ppo_kwargs or {})
        best_value = kwargs.pop("best_value", None)
        if best_value is not None:
            logging.getLogger("boardroom").info(
                "Training with tuned params (baseline eval reward: %.4f)", best_value
            )

        specialist_fn = specialist_fn  # None => use _random_specialist_outputs in factory
        n_envs = n_envs if n_envs is not None else PPO_N_ENVS

        if historical_returns is not None and historical_ratio >= 1.0:
            mode = "historical"
            logging.getLogger("boardroom").info("PPO training mode: 100%% historical")
        elif historical_returns is not None and 0 < historical_ratio < 1:
            historical_ratio = max(0.1, min(0.9, historical_ratio))
            mode = "mixed"
            logging.getLogger("boardroom").info(
                "PPO training mode: mixed (%.0f%% historical, %.0f%% synthetic)",
                historical_ratio * 100,
                (1 - historical_ratio) * 100,
            )
        else:
            mode = "synthetic"
            logging.getLogger("boardroom").info("PPO training mode: 100%% synthetic")

        if n_envs > 1:
            os.environ.setdefault("OMP_NUM_THREADS", "1")

        if n_envs <= 1:
            env = _make_env_fn(
                mode=mode,
                historical_returns=historical_returns,
                n_assets=self.n_assets,
                historical_ratio=historical_ratio,
                specialist_fn=specialist_fn,
                seed=0,
                daily_aux_features=daily_aux_features,
                historical_trading_dates=historical_trading_dates,
            )
        else:
            env_fns = [
                partial(
                    _make_env_fn,
                    mode=mode,
                    historical_returns=historical_returns,
                    n_assets=self.n_assets,
                    historical_ratio=historical_ratio,
                    specialist_fn=specialist_fn,
                    seed=i,
                    daily_aux_features=daily_aux_features,
                    historical_trading_dates=historical_trading_dates,
                )
                for i in range(n_envs)
            ]
            try:
                if sys.platform != "win32":
                    env = SubprocVecEnv(env_fns, start_method="forkserver")
                else:
                    env = SubprocVecEnv(env_fns)
            except TypeError:
                env = SubprocVecEnv(env_fns)
            logging.getLogger("boardroom").info(
                "PPO training with %d parallel envs (SubprocVecEnv)", n_envs
            )

        device = _get_ppo_device()
        if device == "cpu":
            torch.set_num_threads(PPO_CPU_THREADS)
            logging.getLogger("boardroom").info(
                "PPO training on CPU (threads=%s). Set PPO_DEVICE=cuda to try GPU; run 'benchmark-device' to compare.",
                PPO_CPU_THREADS,
            )
        else:
            logging.getLogger("boardroom").info(
                "PPO training on GPU (%s). Run 'benchmark-device' to compare CPU vs GPU.",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda",
            )

        # Per-run id + TensorBoard directory (shared with specialist phase when cmd_train set BOARDROOM_RUN_ID).
        run_kind = os.environ.get("BOARDROOM_RUN_KIND", "train")
        run_id_from_env = os.environ.get("BOARDROOM_RUN_ID", "").strip()
        if run_id_from_env:
            run_id = run_id_from_env
        else:
            run_id = make_run_id(run_kind)
        tb_dir = tb_log_dir_for(run_id)
        tb_dir.mkdir(parents=True, exist_ok=True)
        try:
            if run_id_from_env:
                record_merge_extra(
                    run_id,
                    {
                        "mode": mode,
                        "n_envs": int(getattr(env, "num_envs", 1) or 1),
                    },
                )
            else:
                record_start(
                    run_id,
                    run_kind,
                    timesteps=int(total_timesteps),
                    extra={"mode": mode, "n_envs": int(getattr(env, "num_envs", 1) or 1)},
                )
        except Exception:  # noqa: BLE001 - bookkeeping must never break training
            pass

        env_num_envs = getattr(env, "num_envs", 1)
        if self.model is None:
            policy_kwargs = kwargs.pop("policy_kwargs", None) or PPO_NET_ARCH
            if policy_kwargs is None:
                policy_kwargs = {}
            if isinstance(policy_kwargs, dict) and PPO_WEIGHT_DECAY > 0:
                policy_kwargs = dict(policy_kwargs)
                policy_kwargs.setdefault("optimizer_kwargs", {})
                policy_kwargs["optimizer_kwargs"]["weight_decay"] = PPO_WEIGHT_DECAY
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=kwargs.pop("learning_rate", PPO_LEARNING_RATE),
                n_steps=kwargs.pop("n_steps", PPO_N_STEPS),
                batch_size=kwargs.pop("batch_size", PPO_BATCH_SIZE),
                n_epochs=kwargs.pop("n_epochs", PPO_N_EPOCHS),
                gamma=kwargs.pop("gamma", PPO_GAMMA),
                gae_lambda=kwargs.pop("gae_lambda", PPO_GAE_LAMBDA),
                clip_range=kwargs.pop("clip_range", PPO_CLIP_RANGE),
                ent_coef=kwargs.pop("ent_coef", PPO_ENT_COEF),
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
                tensorboard_log=str(tb_dir.parent),
                **kwargs,
            )
        elif env_num_envs != self.model.n_envs:
            path = META_LEARNER_MODEL_DIR / "ppo_trader.zip"
            if path.exists():
                self.model = PPO.load(str(path), env=env, device=device)
            else:
                self.model.set_env(env)
        else:
            self.model.set_env(env)
        try:
            self.model.tensorboard_log = str(tb_dir.parent)
        except Exception:  # noqa: BLE001
            pass

        checkpoint_dir = META_LEARNER_MODEL_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks: list[BaseCallback] = [GracefulShutdownCallback(verbose=1)]
        if metrics_path() is not None:
            callbacks.append(JsonlMetricsCallback())
        if PPO_CHECKPOINT_FREQ > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=PPO_CHECKPOINT_FREQ,
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo_trader",
                )
            )
        sharpe_eval_callback: SharpeEvalCallback | None = None
        best_snapshot_path = META_LEARNER_MODEL_DIR / "ppo_best.zip"
        if (
            (historical_returns is not None or validation_returns is not None)
            and PPO_EARLY_STOPPING_EVAL_FREQ > 0
            and PPO_EARLY_STOPPING_PATIENCE > 0
        ):
            eval_returns = validation_returns if validation_returns is not None and len(validation_returns) > 0 else historical_returns
            eval_aux = None
            if (
                daily_aux_features is not None
                and daily_aux_features.shape[0] == len(eval_returns)
            ):
                eval_aux = daily_aux_features
            if (
                validation_returns is not None
                and validation_trading_dates is not None
                and len(validation_trading_dates) == len(eval_returns)
            ):
                eval_trading_dates = validation_trading_dates
            elif historical_trading_dates is not None:
                eval_trading_dates = historical_trading_dates[: len(eval_returns)]
            else:
                eval_trading_dates = None
            # Keep eval env type aligned with vec env usage (and avoid Subproc vs Monitor mismatch warnings).
            _ = DummyVecEnv(
                [
                    lambda: HistoricalTradingEnv(
                        returns_array=eval_returns,
                        n_assets=self.n_assets,
                        daily_aux_features=eval_aux,
                        trading_dates=eval_trading_dates,
                    )
                ]
            )
            eval_freq_calls = max(
                1, PPO_EARLY_STOPPING_EVAL_FREQ // max(1, getattr(env, "num_envs", 1))
            )
            sharpe_eval_callback = SharpeEvalCallback(
                val_returns=eval_returns,
                n_assets=self.n_assets,
                eval_freq=eval_freq_calls,
                min_delta=PPO_EARLY_STOPPING_MIN_DELTA,
                best_model_save_path=best_snapshot_path,
                daily_aux_features=eval_aux,
                trading_dates=eval_trading_dates,
                verbose=0,
            )
            callbacks.append(sharpe_eval_callback)
            callbacks.append(
                EarlyStoppingCallback(
                    sharpe_eval_callback,
                    patience=PPO_EARLY_STOPPING_PATIENCE,
                    min_delta=PPO_EARLY_STOPPING_MIN_DELTA,
                    verbose=1,
                )
            )
            logging.getLogger("boardroom").info(
                "Early stopping enabled: eval every %d env-callback steps (~%d train steps with n_envs=%d), patience=%d, min_delta=%.4f",
                eval_freq_calls,
                PPO_EARLY_STOPPING_EVAL_FREQ,
                getattr(env, "num_envs", 1),
                PPO_EARLY_STOPPING_PATIENCE,
                PPO_EARLY_STOPPING_MIN_DELTA,
            )
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                callback=callbacks if callbacks else None,
                tb_log_name=run_id,
                reset_num_timesteps=True,
            )
            run_status = "ok"
        except BaseException:
            try:
                record_end(run_id, status="failed")
            except Exception:  # noqa: BLE001
                pass
            raise
        else:
            try:
                record_end(run_id, status=run_status)
            except Exception:  # noqa: BLE001
                pass

        if (
            sharpe_eval_callback is not None
            and sharpe_eval_callback.has_improved
            and best_snapshot_path.exists()
        ):
            try:
                self.model = PPO.load(str(best_snapshot_path), env=env, device=device)
                logging.getLogger("boardroom").info(
                    "Restored best snapshot at Sharpe=%.4f from %s",
                    sharpe_eval_callback.best_mean_reward,
                    best_snapshot_path,
                )
            except Exception as ex:
                logging.getLogger("boardroom").warning(
                    "Failed to restore best snapshot from %s: %s",
                    best_snapshot_path,
                    ex,
                )
        elif sharpe_eval_callback is not None:
            logging.getLogger("boardroom").info(
                "No Sharpe improvement over min_delta; saving final model state"
            )

        ck_dir = META_LEARNER_MODEL_DIR / "checkpoints"
        prune_checkpoints(ck_dir, keep=PPO_CHECKPOINT_KEEP_LAST)

        META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(META_LEARNER_MODEL_DIR / "ppo_trader.zip"))
        dated_path: Path | None = None
        if total_timesteps >= LONG_RUN_TIMESTEPS:
            dated_name = f"ppo_trader_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
            dated_path = META_LEARNER_MODEL_DIR / dated_name
            self.model.save(str(dated_path))
            logging.getLogger("boardroom").info(
                "Long run (%d timesteps): saved dated model to %s",
                total_timesteps,
                dated_path,
            )
        return dated_path

    def predict(
        self,
        specialist_outputs_by_asset: list[dict],
        holdings: np.ndarray | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Get position sizes from meta-learner.
        specialist_outputs_by_asset: list of n_assets dicts with sentiment, quant_prob,
            fund_score, regime_flag.
        holdings: current position % per asset, or None for zeros.
        debug: if True, log obs shape, feature ranges, and raw action.
        Returns array of size n_assets with position % in [0, 1].
        """
        from ..utils.logger import get_logger

        log = get_logger()

        if self.model is None:
            return np.ones(self.n_assets, dtype=np.float32) * 0.5

        outputs = (list(specialist_outputs_by_asset) + [{}] * self.n_assets)[: self.n_assets]
        if holdings is None:
            holdings = np.zeros(self.n_assets, dtype=np.float32)

        parts: list[float] = []
        for i in range(self.n_assets):
            s = outputs[i] if i < len(outputs) else {}
            parts.append(float(s.get("sentiment", 0.0)))
            parts.append(float(s.get("quant_prob", 0.5)))
            parts.append(float(s.get("fund_score", 0.0)))
            parts.append(float(s.get("regime_flag", 0.5)))
            parts.append(float(s.get("momentum_score", 0.0)))
            parts.append(float(s.get("mean_reversion_score", 0.0)))
            parts.append(float(s.get("volatility_score", 0.5)))
            parts.append(float(s.get("event_risk_score", 0.0)))
            parts.append(float(s.get("relative_strength_score", 0.0)))
            parts.append(float(s.get("liquidity_score", 0.5)))
            parts.append(float(holdings[i]))
            if ENV_OBS_ROLLING_FEATURES:
                parts.append(float(s.get("rolling_return_5d", 0.0)))
                parts.append(float(s.get("rolling_return_21d", 0.0)))
                parts.append(float(s.get("rolling_vol_21d", 0.0)))
        obs = np.array(parts, dtype=np.float32)
        obs = np.clip(obs, -1, 1)
        action, _ = self.model.predict(obs, deterministic=True)
        action_clipped = normalize_action_max_unit_exposure(np.asarray(action, dtype=np.float32))

        if debug:
            per_asset = 11 + (3 if ENV_OBS_ROLLING_FEATURES else 0)
            names = [
                "sentiment",
                "quant_prob",
                "fund_score",
                "regime_flag",
                "momentum_score",
                "mean_reversion_score",
                "volatility_score",
                "event_risk_score",
                "relative_strength_score",
                "liquidity_score",
                "holdings",
            ]
            if ENV_OBS_ROLLING_FEATURES:
                names.extend(["rolling_return_5d", "rolling_return_21d", "rolling_vol_21d"])
            parts = []
            for i in range(self.n_assets):
                start = i * per_asset
                chunk = obs[start : start + per_asset]
                chunk_str = ", ".join(f"{n}={float(v):.3f}" for n, v in zip(names, chunk))
                parts.append(f"  asset_{i}: {chunk_str}")
            log.info(
                "DEBUG | PPO obs shape=%s min=%.3f max=%.3f:\n%s | raw_action=%s | clipped=%s",
                obs.shape,
                float(obs.min()),
                float(obs.max()),
                "\n".join(parts),
                [round(float(a), 4) for a in action],
                [round(float(a), 4) for a in action_clipped],
            )

        return action_clipped


def benchmark_device(
    n_assets: int,
    timesteps: int = 20_000,
    historical_returns: np.ndarray | None = None,
    specialist_fn=None,
) -> dict[str, float | None]:
    """
    Run a short PPO training burst on CPU and (if available) GPU; return seconds per run.
    Use to compare CPU vs GPU speed. Does not save any model.
    """
    def _default_specialist_fn():
        return [
            {
                "sentiment": float(np.random.uniform(-1, 1)),
                "quant_prob": float(np.random.uniform(0, 1)),
                "fund_score": float(np.random.uniform(-1, 1)),
                "regime_flag": float(np.random.uniform(0, 1)),
                "momentum_score": float(np.random.uniform(-1, 1)),
                "mean_reversion_score": float(np.random.uniform(-1, 1)),
                "volatility_score": float(np.random.uniform(0, 1)),
                "event_risk_score": float(np.random.uniform(0, 1)),
                "relative_strength_score": float(np.random.uniform(-1, 1)),
                "liquidity_score": float(np.random.uniform(0, 1)),
            }
            for _ in range(n_assets)
        ]

    specialist_fn = specialist_fn or _default_specialist_fn
    results: dict[str, float | None] = {}

    def _make_env():
        if historical_returns is not None:
            return HistoricalTradingEnv(returns_array=historical_returns, n_assets=n_assets)
        def _reset_fn():
            out = specialist_fn()
            if isinstance(out, dict):
                out = [out] * n_assets
            out = (list(out) + [{}] * n_assets)[:n_assets]
            return {"specialist_outputs": out}

        class SpecialistWrapper(TradingEnv):
            def reset(self, seed=None, options=None):
                opts = options or {}
                opts.update(_reset_fn())
                return super().reset(seed=seed, options=opts)

        return SpecialistWrapper(n_assets=n_assets)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            results["cuda"] = None
            continue
        if device == "cpu":
            torch.set_num_threads(PPO_CPU_THREADS)
        env = _make_env()
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            policy_kwargs=PPO_NET_ARCH or {},
            device=device,
            verbose=0,
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        results[device] = time.perf_counter() - t0

    return results
