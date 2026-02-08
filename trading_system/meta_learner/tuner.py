"""
Optuna-based hyperparameter tuning for PPO meta-learner.
Samples PPO and env hyperparameters, trains on HistoricalTradingEnv, maximizes eval reward.
"""
import json
import math
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from ..config import (
    META_LEARNER_MODEL_DIR,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
)
import numpy as np

from .env import HistoricalTradingEnv, MixedDataEnv, TradingEnv


PPO_TUNE_PARAMS_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_best.json"
PPO_TUNE_STUDY_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_study.db"


def _get_ppo_device() -> str:
    """PPO uses CPU (SB3 recommends CPU for MLP policy)."""
    from ..config import PPO_DEVICE
    if PPO_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample PPO and env hyperparameters for a trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "policy_kwargs": _sample_net_arch(trial),
        "historical_ratio": trial.suggest_float("historical_ratio", 0.1, 0.9),
    }


def _sample_net_arch(trial: optuna.Trial) -> dict:
    arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    if arch == "tiny":
        net_arch = {"pi": [64], "vf": [64]}
    elif arch == "small":
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
    else:
        net_arch = {"pi": [128, 128], "vf": [128, 128]}
    return {"net_arch": net_arch}


class TrialEvalCallback(EvalCallback):
    """EvalCallback that reports to Optuna for pruning."""

    def __init__(self, eval_env, trial: optuna.Trial, n_eval_episodes: int = 5, eval_freq: int = 10000, **kwargs):
        super().__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            if self.last_mean_reward is not None:
                self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def run_tuning(
    returns_array,
    n_assets: int,
    n_trials: int = 50,
    n_timesteps: int = 100_000,
    n_evaluations: int = 5,
    n_eval_episodes: int = 5,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter tuning.
    Returns best params dict.
    """
    eval_freq = max(n_timesteps // n_evaluations, 1)

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
            for _ in range(n_assets)
        ]

    def _make_synthetic_env():
        def _reset_fn():
            outputs = _default_specialist_fn()
            return {"specialist_outputs": outputs}

        class SpecialistWrapper(TradingEnv):
            def reset(self, seed=None, options=None):
                opts = options or {}
                opts.update(_reset_fn())
                return super().reset(seed=seed, options=opts)

        return SpecialistWrapper(n_assets=n_assets)

    def objective(trial: optuna.Trial) -> float:
        params = sample_ppo_params(trial)
        historical_ratio = params.pop("historical_ratio")

        if historical_ratio >= 1.0:
            env = HistoricalTradingEnv(returns_array=returns_array, n_assets=n_assets)
        elif historical_ratio > 0:
            historical_env = HistoricalTradingEnv(
                returns_array=returns_array, n_assets=n_assets
            )
            synthetic_env = _make_synthetic_env()
            env = MixedDataEnv(
                historical_env=historical_env,
                synthetic_env=synthetic_env,
                historical_ratio=historical_ratio,
            )
        else:
            env = _make_synthetic_env()

        env = Monitor(env)
        eval_env = HistoricalTradingEnv(returns_array=returns_array, n_assets=n_assets)
        eval_env = Monitor(eval_env)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            policy_kwargs=params["policy_kwargs"],
            device=_get_ppo_device(),
            verbose=0,
        )

        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0,
        )

        try:
            model.learn(total_timesteps=n_timesteps, callback=eval_callback, progress_bar=False)
        except AssertionError:
            env.close()
            eval_env.close()
            return float("nan")

        env.close()
        eval_env.close()

        if eval_callback.is_pruned:
            raise optuna.TrialPruned()

        return eval_callback.last_mean_reward if eval_callback.last_mean_reward is not None else float("nan")

    torch.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(n_evaluations // 3, 1))

    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{PPO_TUNE_STUDY_PATH}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name="ppo_tune",
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        prev = _previous_best_to_enqueue()
        if prev is not None:
            study.enqueue_trial(prev)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[_save_best_params])

    if study.best_trial is None:
        return load_tuned_params() or {}

    raw = study.best_trial.params
    best_value = float(study.best_value)
    best = {
        "learning_rate": raw["learning_rate"],
        "n_steps": raw["n_steps"],
        "batch_size": raw["batch_size"],
        "n_epochs": raw["n_epochs"],
        "gamma": raw["gamma"],
        "gae_lambda": raw["gae_lambda"],
        "clip_range": raw["clip_range"],
        "ent_coef": raw["ent_coef"],
        "policy_kwargs": _net_arch_from_name(raw.get("net_arch", "small")),
        "historical_ratio": raw.get("historical_ratio", 1.0),
    }
    if not math.isnan(best_value):
        best["best_value"] = best_value

    with open(PPO_TUNE_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)

    return best


def _net_arch_from_name(arch: str) -> dict:
    if arch == "tiny":
        return {"net_arch": {"pi": [64], "vf": [64]}}
    if arch == "medium":
        return {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
    return {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}


def _net_arch_to_name(policy_kwargs: dict) -> str:
    """Map policy_kwargs net_arch dict back to categorical name for enqueue_trial."""
    na = (policy_kwargs or {}).get("net_arch") or {}
    pi = na.get("pi") or []
    vf = na.get("vf") or []
    if pi == [64] and vf == [64]:
        return "tiny"
    if pi == [128, 128] and vf == [128, 128]:
        return "medium"
    return "small"


def _previous_best_to_enqueue() -> dict[str, Any] | None:
    """Build params dict for enqueue_trial from ppo_tune_best.json, or None."""
    if not PPO_TUNE_PARAMS_PATH.exists():
        return None
    try:
        with open(PPO_TUNE_PARAMS_PATH) as f:
            data = json.load(f)
    except Exception:
        return None
    pk = data.get("policy_kwargs")
    if not isinstance(pk, dict):
        return None
    keys = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef")
    if any(data.get(k) is None for k in keys):
        return None
    net_arch = _net_arch_to_name(pk)
    result = {
        "learning_rate": data["learning_rate"],
        "n_steps": data["n_steps"],
        "batch_size": data["batch_size"],
        "n_epochs": data["n_epochs"],
        "gamma": data["gamma"],
        "gae_lambda": data["gae_lambda"],
        "clip_range": data["clip_range"],
        "ent_coef": data["ent_coef"],
        "net_arch": net_arch,
    }
    if "historical_ratio" in data:
        result["historical_ratio"] = data["historical_ratio"]
    return result


def _save_best_params(study: optuna.Study, trial: optuna.Trial) -> None:
    """Callback: save ppo_tune_best.json whenever this trial is the new best."""
    if study.best_trial is not trial:
        return
    raw = trial.params
    best_value = float(study.best_value)
    best = {
        "learning_rate": raw["learning_rate"],
        "n_steps": raw["n_steps"],
        "batch_size": raw["batch_size"],
        "n_epochs": raw["n_epochs"],
        "gamma": raw["gamma"],
        "gae_lambda": raw["gae_lambda"],
        "clip_range": raw["clip_range"],
        "ent_coef": raw["ent_coef"],
        "policy_kwargs": _net_arch_from_name(raw.get("net_arch", "small")),
        "historical_ratio": raw.get("historical_ratio", 1.0),
    }
    if not math.isnan(best_value):
        best["best_value"] = best_value
    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PPO_TUNE_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)


def load_tuned_params() -> dict[str, Any] | None:
    """Load best params from tuning for use with PPOTrader.train(ppo_kwargs=...)."""
    if not PPO_TUNE_PARAMS_PATH.exists():
        return None
    try:
        with open(PPO_TUNE_PARAMS_PATH) as f:
            data = json.load(f)
        if "policy_kwargs" in data and isinstance(data["policy_kwargs"], dict):
            return data
        return None
    except Exception:
        return None
