"""
Optuna-based hyperparameter tuning for PPO meta-learner.
Samples PPO and env hyperparameters, trains on HistoricalTradingEnv, maximizes eval reward.
Uses same env design as training: SubprocVecEnv, proxy-from-returns specialists.
"""
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from ..config import (
    META_LEARNER_MODEL_DIR,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_CPU_THREADS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_N_ENVS,
    TRAIN_HISTORICAL_RATIO,
    TRAIN_HISTORICAL_RATIO_OVERRIDE,
)
import numpy as np
import pandas as pd

from .env import HistoricalTradingEnv, MixedDataEnv
from .env_helpers import create_proxy_specialist_fn
from .eval_helpers import SharpeEvalCallback, deterministic_validation_walk
from .ppo_trader import _make_env_fn


PPO_TUNE_PARAMS_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_best.json"
PPO_TUNE_STUDY_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_study.db"


def _get_ppo_device() -> str:
    """PPO uses CPU (SB3 recommends CPU for MLP policy)."""
    from ..config import PPO_DEVICE
    if PPO_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample_ppo_params(
    trial: optuna.Trial,
    fixed_historical_ratio: float | None = None,
) -> dict[str, Any]:
    """Sample PPO and env hyperparameters for a trial."""
    if fixed_historical_ratio is not None:
        historical_ratio = max(0.0, min(1.0, fixed_historical_ratio))
    elif TRAIN_HISTORICAL_RATIO_OVERRIDE:
        historical_ratio = max(0.0, min(1.0, TRAIN_HISTORICAL_RATIO))
    else:
        historical_ratio = trial.suggest_float("historical_ratio", 0.1, 0.9)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "policy_kwargs": _sample_net_arch(trial),
        "historical_ratio": historical_ratio,
    }


def _sample_net_arch(trial: optuna.Trial) -> dict:
    arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    if arch == "small":
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
    else:
        net_arch = {"pi": [128, 128], "vf": [128, 128]}
    return {"net_arch": net_arch}


def run_tuning(
    returns_array: np.ndarray,
    n_assets: int,
    n_trials: int = 50,
    n_timesteps: int = 100_000,
    n_evaluations: int = 5,
    n_eval_episodes: int = 10,
    timeout: int | None = None,
    n_envs: int | None = None,
    study_name: str = "ppo_tune",
    load_if_exists: bool = True,
    fixed_historical_ratio: float | None = None,
    single_thread: bool = False,
    params_path: Path | None = None,
    show_progress: bool = True,
    historical_trading_dates: Any = None,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter tuning.
    Uses same env design as training: proxy-from-returns specialists, optional SubprocVecEnv.
    Returns best params dict.
    """
    n_envs = n_envs if n_envs is not None else PPO_N_ENVS
    # EvalCallback n_calls scales with vec env rollouts; align eval cadence with n_envs
    eval_freq = max((n_timesteps // n_evaluations) // max(1, n_envs), 1)
    params_path = params_path or PPO_TUNE_PARAMS_PATH
    returns_array = np.asarray(returns_array, dtype=np.float32)

    # Use a validation slice for eval so tuning does not overfit to the same data as training
    n = len(returns_array)
    val_size = max(21, min(252, n // 5))
    if n > val_size + 252:
        train_returns = returns_array[: n - val_size]
        val_returns = returns_array[n - val_size :]
    else:
        train_returns = returns_array
        val_returns = returns_array

    train_trading_dates = None
    val_trading_dates = None
    if historical_trading_dates is not None:
        td = pd.DatetimeIndex(historical_trading_dates)
        if len(td) != n:
            raise ValueError("historical_trading_dates length must match returns_array")
        train_trading_dates = td[: len(train_returns)]
        if n > val_size + 252:
            val_trading_dates = td[len(train_returns) : n]
        else:
            val_trading_dates = td[: len(val_returns)]

    proxy_specialist_fn = create_proxy_specialist_fn(train_returns, n_assets)

    def _save_best_cb(study: optuna.Study, trial: optuna.Trial) -> None:
        _save_best_params(study, trial, params_path)

    def objective(trial: optuna.Trial) -> float:
        params = sample_ppo_params(trial, fixed_historical_ratio=fixed_historical_ratio)
        historical_ratio = params.pop("historical_ratio")

        if historical_ratio >= 1.0:
            mode = "historical"
        elif historical_ratio > 0:
            mode = "mixed"
        else:
            mode = "synthetic"

        if n_envs <= 1:
            env = _make_env_fn(
                mode=mode,
                historical_returns=train_returns,
                n_assets=n_assets,
                historical_ratio=historical_ratio,
                specialist_fn=proxy_specialist_fn,
                seed=trial.number,
                daily_aux_features=None,
                historical_trading_dates=train_trading_dates,
            )
        else:
            env_fns = [
                partial(
                    _make_env_fn,
                    mode=mode,
                    historical_returns=train_returns,
                    n_assets=n_assets,
                    historical_ratio=historical_ratio,
                    specialist_fn=proxy_specialist_fn,
                    seed=trial.number * 1000 + i,
                    daily_aux_features=None,
                    historical_trading_dates=train_trading_dates,
                )
                for i in range(n_envs)
            ]
            try:
                import sys

                if sys.platform != "win32":
                    env = SubprocVecEnv(env_fns, start_method="forkserver")
                else:
                    env = SubprocVecEnv(env_fns)
            except TypeError:
                env = SubprocVecEnv(env_fns)

        env = VecMonitor(env) if isinstance(env, VecEnv) else Monitor(env)
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

        sharpe_eval_callback = SharpeEvalCallback(
            val_returns=val_returns,
            n_assets=n_assets,
            eval_freq=max(1, n_timesteps // max(1, n_envs)),
            min_delta=0.0,
            trading_dates=val_trading_dates,
            verbose=0,
        )

        try:
            model.learn(
                total_timesteps=n_timesteps,
                callback=sharpe_eval_callback,
                progress_bar=show_progress,
            )
        except (AssertionError, ValueError):
            return float("nan")
        finally:
            env.close()

        sharpe, _ = deterministic_validation_walk(
            model,
            val_returns,
            n_assets,
            trading_dates=val_trading_dates,
        )
        trial.report(sharpe, 1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return sharpe

    device = _get_ppo_device()
    if device == "cpu":
        torch.set_num_threads(1 if single_thread else PPO_CPU_THREADS)
    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(n_evaluations // 3, 1))

    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{PPO_TUNE_STUDY_PATH}"
    if not load_if_exists:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
    if len(study.trials) == 0:
        prev = _previous_best_to_enqueue(params_path)
        if prev is not None:
            study.enqueue_trial(prev)

    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, callbacks=[_save_best_cb],
        show_progress_bar=show_progress,
    )

    if study.best_trial is None:
        return load_tuned_params(params_path) or {}

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

    with open(params_path, "w") as f:
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


def _previous_best_to_enqueue(path: Path | None = None) -> dict[str, Any] | None:
    """Build params dict for enqueue_trial from ppo_tune_best.json, or None."""
    p = path or PPO_TUNE_PARAMS_PATH
    if not p.exists():
        return None
    try:
        with open(p) as f:
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
    if net_arch == "tiny":
        # Tiny is no longer in the tuning search space; map legacy payloads to nearest supported arch.
        net_arch = "small"
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


def _save_best_params(
    study: optuna.Study,
    trial: optuna.Trial,
    params_path: Path | None = None,
) -> None:
    """Callback: save best params JSON whenever this trial is the new best."""
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
    p = params_path or PPO_TUNE_PARAMS_PATH
    with open(p, "w") as f:
        json.dump(best, f, indent=2)


def get_tune_status(
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Return status of tuning studies: list of study names and summary for each.
    If study_name is given, include best trial and best value for that study.
    """
    storage = f"sqlite:///{PPO_TUNE_STUDY_PATH}"
    if not PPO_TUNE_STUDY_PATH.exists():
        return {"study_names": [], "studies": {}}
    try:
        study_names = optuna.study.get_all_study_names(storage=storage)
    except Exception:
        return {"study_names": [], "studies": {}}
    result: dict[str, Any] = {"study_names": study_names, "studies": {}}
    for name in study_names:
        if study_name is not None and name != study_name:
            continue
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            n_trials = len(study.trials)
            best_value = float(study.best_value) if study.best_trial else None
            result["studies"][name] = {
                "n_trials": n_trials,
                "best_value": best_value,
                "best_trial": study.best_trial.number if study.best_trial else None,
            }
        except Exception:
            result["studies"][name] = {"error": "failed to load"}
    return result


def load_tuned_params(path: Path | None = None) -> dict[str, Any] | None:
    """Load best params from tuning for use with PPOTrader.train(ppo_kwargs=...)."""
    p = path or PPO_TUNE_PARAMS_PATH
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        if "policy_kwargs" in data and isinstance(data["policy_kwargs"], dict):
            return data
        return None
    except Exception:
        return None
