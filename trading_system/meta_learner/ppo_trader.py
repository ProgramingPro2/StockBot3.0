"""
PPO meta-learner via stable-baselines3.
Weights specialist opinions to decide position sizing.
"""
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ..config import (
    DEVICE,
    META_LEARNER_MODEL_DIR,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
)
from .env import TradingEnv


def _get_device() -> str:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PPOTrader:
    """PPO-based meta-learner for position sizing."""

    def __init__(self, n_assets: int = 1, load: bool = True):
        self.n_assets = n_assets
        self.env = TradingEnv(n_assets=n_assets)
        self.model: PPO | None = None

        if load:
            path = META_LEARNER_MODEL_DIR / "ppo_trader.zip"
            if path.exists():
                try:
                    self.model = PPO.load(str(path), env=self.env, device=_get_device())
                except Exception:
                    self.model = None

    def train(self, total_timesteps: int = 50_000, specialist_fn=None) -> None:
        """
        Train PPO on simulated episodes.
        specialist_fn: callable() -> dict with sentiment, quant_prob, fund_score, regime_flag.
        If None, uses random simulated outputs.
        """
        import numpy as np

        def _default_specialist_fn():
            return {
                "sentiment": float(np.random.uniform(-1, 1)),
                "quant_prob": float(np.random.uniform(0, 1)),
                "fund_score": float(np.random.uniform(-1, 1)),
                "regime_flag": float(np.random.uniform(0, 1)),
            }

        specialist_fn = specialist_fn or _default_specialist_fn

        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=PPO_LEARNING_RATE,
                n_steps=PPO_N_STEPS,
                batch_size=PPO_BATCH_SIZE,
                n_epochs=PPO_N_EPOCHS,
                gamma=PPO_GAMMA,
                gae_lambda=PPO_GAE_LAMBDA,
                clip_range=PPO_CLIP_RANGE,
                device=_get_device(),
                verbose=1,
            )

        def _reset_fn():
            if specialist_fn:
                outputs = specialist_fn()
                return {"specialist_outputs": outputs}
            return {}

        # Wrap env for specialist injection
        class SpecialistWrapper(TradingEnv):
            def reset(self, seed=None, options=None):
                opts = options or {}
                opts.update(_reset_fn())
                return super().reset(seed=seed, options=opts)

        wrapped = SpecialistWrapper(n_assets=self.n_assets)
        self.model.set_env(wrapped)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

        META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(META_LEARNER_MODEL_DIR / "ppo_trader.zip"))

    def predict(
        self,
        sentiment: float,
        quant_prob: float,
        fund_score: float,
        regime_flag: float,
    ) -> np.ndarray:
        """
        Get position sizes from meta-learner.
        Returns array of size n_assets with position % in [0, 1].
        """
        if self.model is None:
            return np.ones(self.n_assets, dtype=np.float32) * 0.5

        obs = np.array(
            [sentiment, quant_prob, fund_score, regime_flag] + [0.0] * self.n_assets,
            dtype=np.float32,
        )
        obs = np.clip(obs, -1, 1)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(action, 0, 1)
