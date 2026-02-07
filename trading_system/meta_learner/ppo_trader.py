"""
PPO meta-learner via stable-baselines3.
Weights specialist opinions to decide position sizing.
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# When training runs this many timesteps or more, an additional dated copy is saved.
LONG_RUN_TIMESTEPS = 1_000_000

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ..config import (
    ENV_OBS_ROLLING_FEATURES,
    META_LEARNER_MODEL_DIR,
    PPO_CHECKPOINT_FREQ,
    PPO_CPU_THREADS,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_NET_ARCH,
    PPO_N_STEPS,
)
from .env import HistoricalTradingEnv, TradingEnv


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
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> Path | None:
        """
        Train PPO on simulated or historical episodes.
        specialist_fn: callable() -> list[dict] of length n_assets; each dict has
            sentiment, quant_prob, fund_score, regime_flag.
        historical_returns: if provided, shape (n_days, n_assets), use HistoricalTradingEnv.
        ppo_kwargs: optional overrides for PPO hyperparameters (e.g. from tuning).
        """
        import numpy as np

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
                }
                for _ in range(self.n_assets)
            ]

        specialist_fn = specialist_fn or _default_specialist_fn
        kwargs = dict(ppo_kwargs or {})
        best_value = kwargs.pop("best_value", None)
        if best_value is not None:
            logging.getLogger("boardroom").info(
                "Training with tuned params (baseline eval reward: %.4f)", best_value
            )

        if historical_returns is not None:
            env = HistoricalTradingEnv(
                returns_array=historical_returns,
                n_assets=self.n_assets,
            )
        else:
            def _reset_fn():
                if specialist_fn:
                    outputs = specialist_fn()
                    if isinstance(outputs, dict):
                        outputs = [outputs] * self.n_assets
                    outputs = (list(outputs) + [{}] * self.n_assets)[: self.n_assets]
                    return {"specialist_outputs": outputs}
                return {}

            class SpecialistWrapper(TradingEnv):
                def reset(self, seed=None, options=None):
                    opts = options or {}
                    opts.update(_reset_fn())
                    return super().reset(seed=seed, options=opts)

            env = SpecialistWrapper(n_assets=self.n_assets)

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

        if self.model is None:
            policy_kwargs = kwargs.pop("policy_kwargs", None) or PPO_NET_ARCH
            if policy_kwargs is None:
                policy_kwargs = {}
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
                **kwargs,
            )
        else:
            self.model.set_env(env)

        checkpoint_dir = META_LEARNER_MODEL_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=PPO_CHECKPOINT_FREQ,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_trader",
        ) if PPO_CHECKPOINT_FREQ > 0 else None

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=checkpoint_callback,
        )

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
            parts.append(float(holdings[i]))
            if ENV_OBS_ROLLING_FEATURES:
                parts.append(float(s.get("rolling_return_5d", 0.0)))
                parts.append(float(s.get("rolling_return_21d", 0.0)))
                parts.append(float(s.get("rolling_vol_21d", 0.0)))
        obs = np.array(parts, dtype=np.float32)
        obs = np.clip(obs, -1, 1)
        action, _ = self.model.predict(obs, deterministic=True)
        action_clipped = np.clip(action, 0, 1)

        if debug:
            per_asset = 9 + (3 if ENV_OBS_ROLLING_FEATURES else 0)
            names = [
                "sentiment",
                "quant_prob",
                "fund_score",
                "regime_flag",
                "momentum_score",
                "mean_reversion_score",
                "volatility_score",
                "event_risk_score",
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
