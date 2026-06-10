"""
Optuna-based hyperparameter tuning for PPO meta-learner.
Samples PPO and env hyperparameters, trains on HistoricalTradingEnv, maximizes eval reward.
Uses same env design as training: SubprocVecEnv, proxy-from-returns specialists.
"""
import json
import logging
import math
import multiprocessing
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.distributions import CategoricalDistribution
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import ExperimentalWarning
from optuna.storages import RDBStorage
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from ..config import (
    META_LEARNER_MODEL_DIR,
    PPO_CPU_THREADS,
    PPO_N_ENVS,
    PPO_USE_TORCH_VEC,
    TRAIN_HISTORICAL_RATIO,
    TRAIN_HISTORICAL_RATIO_OVERRIDE,
)
from ..utils.graceful_shutdown import should_stop
from .bench_trace import trace_block, trace_enabled
from .env import HistoricalTradingEnv, MixedDataEnv
from .env_helpers import create_proxy_specialist_fn
from .eval_helpers import SharpeEvalCallback, deterministic_validation_walk
from .metrics_callback import JsonlMetricsCallback, metrics_path
from .ppo_trader import GracefulShutdownCallback, _make_env_fn
from .proxy_specialists import precompute_proxy_feature_matrix
from .runs_index import make_run_id, record_end, record_start, tb_log_dir_for

_log = logging.getLogger("boardroom")

# Default categorical search spaces (new studies). Resumed studies must reuse the same choices
# Optuna recorded on the first trial — see ``_categorical_choices``.
_NET_ARCH_CHOICES = ("tiny", "small", "medium", "large")
_N_STEPS_CHOICES = (2048, 4096, 8192)
_BATCH_SIZE_CHOICES = (256, 512, 1024, 2048)
_N_EPOCHS_CHOICES = (3, 5, 10)
_VF_COEF_CHOICES = (0.25, 0.5, 1.0)

# SubprocVecEnv cannot be created inside multiprocessing.Pool workers (they are daemon=True).
_SUBPROC_WARNED = False


def _subproc_vec_env_allowed() -> bool:
    """False in daemon workers (e.g. multiprocessing.Pool); those cannot spawn child processes."""
    try:
        return not multiprocessing.current_process().daemon
    except Exception:
        return True


# SubprocVecEnv / multiprocessing teardown after SIGTERM: workers may already be gone.
_VEC_ENV_CLOSE_IPC_ERRORS: tuple[type[BaseException], ...] = (
    BrokenPipeError,
    EOFError,
    OSError,
    ConnectionError,
    AttributeError,
)


def _safe_vec_env_close(env: Any) -> None:
    """Close vec env; ignore IPC errors when workers already exited (e.g. graceful shutdown)."""
    try:
        env.close()
    except _VEC_ENV_CLOSE_IPC_ERRORS as exc:
        _log.debug(
            "Ignored vec env close IPC error%s: %s",
            " after graceful shutdown" if should_stop() else "",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("Ignored vec env close error: %s", exc)


PPO_TUNE_PARAMS_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_best.json"
PPO_TUNE_STUDY_PATH = META_LEARNER_MODEL_DIR / "ppo_tune_study.db"
PPO_TUNE_JOURNAL_DIR = META_LEARNER_MODEL_DIR / "ppo_tune_journal"
PPO_TUNE_JOURNAL_LOG = PPO_TUNE_JOURNAL_DIR / "journal.log"

_tune_rdb_storage: RDBStorage | None = None
_tune_journal_storage: Any | None = None


def _journal_storage_available() -> bool:
    try:
        from optuna.storages import JournalStorage  # noqa: F401
        from optuna.storages.journal import JournalFileBackend  # noqa: F401

        return True
    except ImportError:
        return False


def get_tune_journal_storage() -> Any:
    """Append-only Optuna journal backend for parallel tuning (lower contention than SQLite)."""
    global _tune_journal_storage
    if _tune_journal_storage is None:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        PPO_TUNE_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        journal_log = str(PPO_TUNE_JOURNAL_LOG)
        lock_obj = None
        try:
            from optuna.storages.journal import JournalFileSymlinkLock

            lock_obj = JournalFileSymlinkLock(f"{journal_log}.lock")
        except ImportError:
            pass
        backend = JournalFileBackend(journal_log, lock_obj=lock_obj)
        _tune_journal_storage = JournalStorage(backend)
    return _tune_journal_storage


def get_tune_storage(*, use_journal_storage: bool = False) -> Any:
    """Return Optuna storage backend (journal for parallel workers, SQLite otherwise)."""
    if use_journal_storage and _journal_storage_available():
        return get_tune_journal_storage()
    return get_tune_rdb_storage()


def split_tune_train_val(returns_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split returns into train/validation slices used by tuning."""
    returns_array = np.asarray(returns_array, dtype=np.float32)
    n = len(returns_array)
    val_size = max(21, min(252, n // 5))
    if n > val_size + 252:
        return returns_array[: n - val_size], returns_array[n - val_size :]
    return returns_array, returns_array


def get_tune_rdb_storage() -> RDBStorage:
    """
    Shared Optuna SQLite backend with a long busy-timeout and heartbeats.

    Parallel ``ProcessPoolExecutor`` workers contend on the same DB; defaults
    without ``timeout`` can block indefinitely under load. Heartbeats let Optuna
    mark stale RUNNING trials after ``grace_period`` when a worker dies.
    """
    global _tune_rdb_storage
    if _tune_rdb_storage is None:
        META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ExperimentalWarning)
            _tune_rdb_storage = RDBStorage(
                url=f"sqlite:///{PPO_TUNE_STUDY_PATH}",
                engine_kwargs={"connect_args": {"timeout": 120.0}},
                heartbeat_interval=60,
                grace_period=120,
            )
    return _tune_rdb_storage


def tune_best_params_json_path(study_name: str) -> Path:
    """Per-study JSON path so concurrent or sequential studies do not clobber each other."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (study_name or "").strip()) or "study"
    return META_LEARNER_MODEL_DIR / f"ppo_tune_best_{safe}.json"


def _pack_best_params_from_trial_raw(raw: dict[str, Any], best_value: float | None) -> dict[str, Any]:
    """Serialize Optuna trial.params (plus optional best_value) to the ppo_tune_best.json shape."""
    best: dict[str, Any] = {
        "learning_rate": raw["learning_rate"],
        "n_steps": raw["n_steps"],
        "batch_size": raw["batch_size"],
        "n_epochs": raw["n_epochs"],
        "gamma": raw["gamma"],
        "gae_lambda": raw["gae_lambda"],
        "clip_range": raw["clip_range"],
        "ent_coef": raw["ent_coef"],
        "vf_coef": raw.get("vf_coef", 0.5),
        "weight_decay": raw.get("weight_decay", 0.0001),
        "obs_noise_std": raw.get("obs_noise_std", 0.0),
        "policy_kwargs": _net_arch_from_name(raw.get("net_arch", "small")),
        "historical_ratio": raw.get("historical_ratio", 1.0),
    }
    if best_value is not None and not math.isnan(best_value):
        best["best_value"] = best_value
    return best


def load_tuned_params_from_study(study_name: str) -> dict[str, Any] | None:
    """Load best PPO hyperparameters from a named Optuna study (same storage as tune)."""
    name = (study_name or "").strip()
    if not name:
        return None
    storage = None
    if PPO_TUNE_JOURNAL_LOG.exists() and _journal_storage_available():
        try:
            storage = get_tune_journal_storage()
            study = optuna.load_study(study_name=name, storage=storage)
            if study.best_trial is not None:
                bv = float(study.best_value) if study.best_value is not None else float("nan")
                return _pack_best_params_from_trial_raw(
                    dict(study.best_trial.params),
                    bv if math.isfinite(bv) else None,
                )
        except Exception:
            pass
    if not PPO_TUNE_STUDY_PATH.exists():
        return None
    try:
        study = optuna.load_study(study_name=name, storage=get_tune_rdb_storage())
    except Exception:
        return None
    if study.best_trial is None:
        return None
    try:
        bv = float(study.best_value) if study.best_value is not None else float("nan")
    except Exception:
        bv = float("nan")
    try:
        return _pack_best_params_from_trial_raw(dict(study.best_trial.params), bv if math.isfinite(bv) else None)
    except KeyError:
        return None


def _get_ppo_device() -> str:
    """Use CUDA when available unless PPO_DEVICE forces CPU."""
    from ..config import PPO_DEVICE

    lowered = (PPO_DEVICE or "cuda").strip().lower()
    if lowered in ("cpu", "no", "false", "0"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _categorical_choices(study: optuna.Study, name: str, default: tuple[Any, ...]) -> list[Any]:
    """Reuse categorical choices from an existing study so resumed runs do not change the value space."""
    for t in reversed(study.trials):
        if name not in t.params:
            continue
        dist = t.distributions.get(name)
        if isinstance(dist, CategoricalDistribution):
            choices = list(dist.choices)
            if choices != list(default) and len(study.trials) > 0:
                _log.debug(
                    "Study %r: reusing legacy categorical %s=%s (default would be %s)",
                    study.study_name,
                    name,
                    choices,
                    list(default),
                )
            return choices
    if len(study.trials) > 0:
        _log.debug(
            "Study %r: no prior %s distribution; using default %s",
            study.study_name,
            name,
            list(default),
        )
    return list(default)


def sample_ppo_params(
    trial: optuna.Trial,
    fixed_historical_ratio: float | None = None,
) -> dict[str, Any]:
    """Sample PPO and env hyperparameters for a trial."""
    study = trial.study
    if fixed_historical_ratio is not None:
        historical_ratio = max(0.0, min(1.0, fixed_historical_ratio))
    elif TRAIN_HISTORICAL_RATIO_OVERRIDE:
        historical_ratio = max(0.0, min(1.0, TRAIN_HISTORICAL_RATIO))
    else:
        historical_ratio = trial.suggest_float("historical_ratio", 0.1, 0.9)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    vf_coef = trial.suggest_categorical(
        "vf_coef",
        _categorical_choices(study, "vf_coef", _VF_COEF_CHOICES),
    )
    obs_noise_std = trial.suggest_float("obs_noise_std", 0.0, 0.05)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "n_steps": trial.suggest_categorical(
            "n_steps",
            _categorical_choices(study, "n_steps", _N_STEPS_CHOICES),
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            _categorical_choices(study, "batch_size", _BATCH_SIZE_CHOICES),
        ),
        "n_epochs": trial.suggest_categorical(
            "n_epochs",
            _categorical_choices(study, "n_epochs", _N_EPOCHS_CHOICES),
        ),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "vf_coef": float(vf_coef),
        "weight_decay": float(weight_decay),
        "obs_noise_std": float(obs_noise_std),
        "policy_kwargs": _sample_net_arch(trial),
        "historical_ratio": historical_ratio,
    }


def _sample_net_arch(trial: optuna.Trial) -> dict:
    arch = trial.suggest_categorical(
        "net_arch",
        _categorical_choices(trial.study, "net_arch", _NET_ARCH_CHOICES),
    )
    if arch == "tiny":
        net_arch = {"pi": [64], "vf": [64]}
    elif arch == "small":
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
    elif arch == "medium":
        net_arch = {"pi": [128, 128], "vf": [128, 128]}
    elif arch == "large":
        net_arch = {"pi": [256, 256], "vf": [256, 256]}
    else:
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
    return {"net_arch": net_arch}


def run_tuning(
    returns_array: np.ndarray,
    n_assets: int,
    n_trials: int = 50,
    n_timesteps: int = 100_000,
    n_evaluations: int = 3,
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
    use_torch_vec: bool | None = None,
    eval_patience: int = 0,
    disable_tensorboard: bool = False,
    precomputed_train_proxy: np.ndarray | None = None,
    precomputed_val_proxy: np.ndarray | None = None,
    use_journal_storage: bool = False,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter tuning.
    Uses same env design as training: proxy-from-returns specialists, optional SubprocVecEnv.
    Returns best params dict.
    """
    n_envs = n_envs if n_envs is not None else PPO_N_ENVS
    use_torch_vec_eff = PPO_USE_TORCH_VEC if use_torch_vec is None else bool(use_torch_vec)
    # SB3 callback n_calls advances per vec-env rollout; spread evals across the trial
    eval_freq = max(1, n_timesteps // (max(1, n_evaluations) * max(1, n_envs)))
    params_path = params_path or PPO_TUNE_PARAMS_PATH
    returns_array = np.asarray(returns_array, dtype=np.float32)

    _dev0 = _get_ppo_device()
    if _dev0 == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Use a validation slice for eval so tuning does not overfit to the same data as training
    n = len(returns_array)
    train_returns, val_returns = split_tune_train_val(returns_array)
    val_size = len(val_returns) if len(train_returns) != len(val_returns) else max(21, min(252, n // 5))

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

    if precomputed_train_proxy is not None:
        train_proxy_cache = precomputed_train_proxy
    else:
        train_proxy_cache = precompute_proxy_feature_matrix(train_returns)
    if precomputed_val_proxy is not None:
        val_proxy_cache = precomputed_val_proxy
    elif len(val_returns) > 21:
        val_proxy_cache = precompute_proxy_feature_matrix(val_returns)
    else:
        val_proxy_cache = None

    proxy_specialist_fn = create_proxy_specialist_fn(train_returns, n_assets)

    def _save_best_cb(study: optuna.Study, trial: optuna.Trial) -> None:
        _save_best_params(study, trial, params_path)

    def _graceful_stop_cb(study: optuna.Study, trial: optuna.Trial) -> None:
        if should_stop():
            study.stop()

    use_dummy_vec = os.environ.get("PPO_USE_DUMMY_VEC", "").strip().lower() in ("1", "true", "yes")

    def objective(trial: optuna.Trial) -> float:
        if n_envs > 1:
            os.environ.setdefault("OMP_NUM_THREADS", "1")

        params = sample_ppo_params(trial, fixed_historical_ratio=fixed_historical_ratio)
        if params["n_steps"] * n_envs > n_timesteps:
            raise optuna.TrialPruned()
        historical_ratio = params.pop("historical_ratio")
        trial_weight_decay = float(params.pop("weight_decay", 0.0001))
        trial_vf_coef = float(params.pop("vf_coef", 0.5))
        trial_obs_noise_std = float(params.pop("obs_noise_std", 0.0))
        # Apply weight decay (L2 regularization) via Adam optimizer_kwargs
        pk = params.get("policy_kwargs") or {}
        pk = dict(pk)
        pk.setdefault("optimizer_kwargs", {})
        pk["optimizer_kwargs"]["weight_decay"] = trial_weight_decay
        params["policy_kwargs"] = pk

        if historical_ratio >= 1.0:
            mode = "historical"
        elif historical_ratio > 0:
            mode = "mixed"
        else:
            mode = "synthetic"

        if trace_enabled():
            import logging

            logging.getLogger("boardroom").info(
                "BENCH_TRACE tune trial=%s n_envs=%s eval_freq=%s dummy_vec=%s torch_vec=%s",
                trial.number,
                n_envs,
                eval_freq,
                use_dummy_vec,
                use_torch_vec_eff and mode == "historical",
            )

        use_torch_vec_path = use_torch_vec_eff and mode == "historical"

        if use_torch_vec_path and n_envs >= 1:
            from .torch_vec_env import BatchedHistoricalTorchVecEnv

            with trace_block("tune_make_torch_vec_env", trial=trial.number, n_envs=n_envs):
                env = BatchedHistoricalTorchVecEnv(
                    train_returns,
                    train_proxy_cache,
                    n_assets,
                    n_envs,
                    trading_dates=train_trading_dates,
                    device=_get_ppo_device(),
                )
        elif n_envs <= 1:
            with trace_block("tune_make_env_single", trial=trial.number):
                env = _make_env_fn(
                    mode=mode,
                    historical_returns=train_returns,
                    n_assets=n_assets,
                    historical_ratio=historical_ratio,
                    specialist_fn=proxy_specialist_fn,
                    seed=trial.number,
                    daily_aux_features=None,
                    historical_trading_dates=train_trading_dates,
                    proxy_feature_cache=train_proxy_cache,
                    obs_noise_std=trial_obs_noise_std,
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
                    proxy_feature_cache=train_proxy_cache,
                    obs_noise_std=trial_obs_noise_std,
                )
                for i in range(n_envs)
            ]
            with trace_block("tune_make_vec_env", trial=trial.number, n_envs=n_envs):
                global _SUBPROC_WARNED
                force_dummy = use_dummy_vec or not _subproc_vec_env_allowed()
                if force_dummy:
                    if not use_dummy_vec and not _SUBPROC_WARNED:
                        _log.info(
                            "Daemon worker: using DummyVecEnv instead of SubprocVecEnv "
                            "(nested multiprocessing is unsupported). Use --parallel-trials 1 if "
                            "this message appears during tuning."
                        )
                        _SUBPROC_WARNED = True
                    env = DummyVecEnv(env_fns)
                else:
                    try:
                        import sys

                        if sys.platform != "win32":
                            env = SubprocVecEnv(env_fns, start_method="forkserver")
                        else:
                            env = SubprocVecEnv(env_fns)
                    except TypeError:
                        env = SubprocVecEnv(env_fns)

        env = VecMonitor(env) if isinstance(env, VecEnv) else Monitor(env)
        run_id: str | None = None
        tb_log_name: str | None = None
        if not disable_tensorboard:
            run_id = make_run_id(f"tune_t{trial.number}")
            tb_dir = tb_log_dir_for(run_id)
            tb_log_name = run_id
            try:
                record_start(
                    run_id,
                    "tune",
                    timesteps=int(n_timesteps),
                    extra={
                        "trial": int(trial.number),
                        "study": study_name,
                        "params": {
                            **{k: params.get(k) for k in (
                                "learning_rate",
                                "n_steps",
                                "batch_size",
                                "n_epochs",
                                "gamma",
                                "gae_lambda",
                                "clip_range",
                                "ent_coef",
                            )},
                            "vf_coef": trial_vf_coef,
                            "weight_decay": trial_weight_decay,
                            "obs_noise_std": trial_obs_noise_std,
                        },
                    },
                )
            except Exception:  # noqa: BLE001
                pass
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
            vf_coef=trial_vf_coef,
            policy_kwargs=params["policy_kwargs"],
            device=_get_ppo_device(),
            verbose=0,
            tensorboard_log=None if disable_tensorboard else str(tb_dir.parent),
        )
        if _get_ppo_device() == "cuda" and os.environ.get("PPO_TORCH_COMPILE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                model.policy = torch.compile(model.policy, mode="reduce-overhead")
            except Exception as compile_exc:
                _log.warning("PPO_TORCH_COMPILE skipped in tune: %s", compile_exc)

        sharpe_eval_callback = SharpeEvalCallback(
            val_returns=val_returns,
            n_assets=n_assets,
            eval_freq=eval_freq,
            min_delta=0.0,
            trading_dates=val_trading_dates,
            proxy_feature_cache=val_proxy_cache,
            optuna_trial=trial,
            patience=eval_patience,
            verbose=0,
        )
        callbacks: list = [GracefulShutdownCallback(verbose=1), sharpe_eval_callback]
        if metrics_path() is not None:
            callbacks.insert(0, JsonlMetricsCallback())

        trial_status = "ok"
        try:
            with trace_block("tune_learn", trial=trial.number, timesteps=n_timesteps):
                learn_kwargs: dict[str, Any] = {
                    "total_timesteps": n_timesteps,
                    "callback": callbacks,
                    "progress_bar": show_progress,
                    "reset_num_timesteps": True,
                }
                if tb_log_name is not None:
                    learn_kwargs["tb_log_name"] = tb_log_name
                model.learn(**learn_kwargs)
        except optuna.TrialPruned:
            trial_status = "pruned"
            if run_id is not None:
                try:
                    record_end(run_id, status=trial_status)
                except Exception:  # noqa: BLE001
                    pass
            raise
        except (AssertionError, ValueError):
            trial_status = "failed"
            if run_id is not None:
                try:
                    record_end(run_id, status=trial_status)
                except Exception:  # noqa: BLE001
                    pass
        else:
            if run_id is not None:
                try:
                    record_end(run_id, status=trial_status)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            sharpe_eval_callback.close_eval_env()
            _safe_vec_env_close(env)

        if trial_status == "failed":
            return float("nan")

        if should_stop() and trial_status == "ok":
            if run_id is not None:
                try:
                    record_end(run_id, status="stopped")
                except Exception:  # noqa: BLE001
                    pass
            return float("nan")

        last = sharpe_eval_callback.last_mean_reward
        if last is None or not math.isfinite(float(last)):
            sharpe, _ = deterministic_validation_walk(
                model,
                val_returns,
                n_assets,
                trading_dates=val_trading_dates,
                proxy_feature_cache=val_proxy_cache,
            )
        else:
            sharpe = float(last)
        return sharpe

    device = _get_ppo_device()
    if device == "cpu":
        torch.set_num_threads(1 if single_thread else PPO_CPU_THREADS)
    sampler = TPESampler(n_startup_trials=10, seed=42)
    # HyperbandPruner creates bracket sub-studies lazily; parallel workers race to look them
    # up before they exist in JournalStorage, causing KeyError on the first trial.  Use
    # MedianPruner for any parallel-journal run; HyperbandPruner is fine for single-worker runs.
    pruner_env = os.environ.get("OPTUNA_PRUNER", "").strip().lower()
    use_median = pruner_env == "median" or (use_journal_storage and pruner_env != "hyperband")
    if use_median:
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(n_evaluations // 3, 1))
    else:
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=max(n_evaluations, 2),
            reduction_factor=3,
        )

    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    use_journal = use_journal_storage and _journal_storage_available()
    if use_journal_storage and not use_journal:
        _log.warning(
            "JournalStorage unavailable (Optuna >= 3.1 required); falling back to SQLite."
        )
    storage = get_tune_storage(use_journal_storage=use_journal)
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
    if len(study.trials) > 0:
        legacy_arch = _categorical_choices(study, "net_arch", _NET_ARCH_CHOICES)
        if tuple(legacy_arch) != _NET_ARCH_CHOICES:
            _log.info(
                "Resuming study %r with net_arch choices %s (expanded default %s is only used for new studies). "
                "Pass --new or a new --study-name to search the full space.",
                study_name,
                legacy_arch,
                list(_NET_ARCH_CHOICES),
            )
    if len(study.trials) == 0:
        prev = _previous_best_to_enqueue(params_path)
        if prev is not None:
            study.enqueue_trial(prev)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_save_best_cb, _graceful_stop_cb],
        show_progress_bar=show_progress,
    )

    return export_best_params_after_study(study_name, params_path, use_journal_storage=use_journal)


def export_best_params_after_study(
    study_name: str,
    params_path: Path | None = None,
    *,
    use_journal_storage: bool = False,
) -> dict[str, Any]:
    """Load study from default storage and write best params JSON (same as end of run_tuning)."""
    storage = get_tune_storage(use_journal_storage=use_journal_storage)
    p = params_path or PPO_TUNE_PARAMS_PATH
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return load_tuned_params(p) or {}
    if study.best_trial is None:
        return load_tuned_params(p) or {}
    raw = dict(study.best_trial.params)
    best_value = float(study.best_value)
    best = _pack_best_params_from_trial_raw(raw, best_value if not math.isnan(best_value) else None)
    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(best, f, indent=2)
    if p.resolve() != PPO_TUNE_PARAMS_PATH.resolve():
        try:
            with open(PPO_TUNE_PARAMS_PATH, "w") as f:
                json.dump(best, f, indent=2)
        except OSError:
            pass
    return best


def _net_arch_from_name(arch: str) -> dict:
    if arch == "tiny":
        return {"net_arch": {"pi": [64], "vf": [64]}}
    if arch == "medium":
        return {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
    if arch == "large":
        return {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
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
    if pi == [256, 256] and vf == [256, 256]:
        return "large"
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
    if "vf_coef" in data:
        result["vf_coef"] = data["vf_coef"]
    if "weight_decay" in data:
        result["weight_decay"] = data["weight_decay"]
    if "obs_noise_std" in data:
        result["obs_noise_std"] = data["obs_noise_std"]
    return result


def _save_best_params(
    study: optuna.Study,
    trial: optuna.Trial,
    params_path: Path | None = None,
) -> None:
    """Callback: save best params JSON whenever this trial is the new best."""
    if study.best_trial is not trial:
        return
    raw = dict(trial.params)
    best_value = float(study.best_value)
    best = _pack_best_params_from_trial_raw(raw, best_value if not math.isnan(best_value) else None)
    META_LEARNER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p = params_path or PPO_TUNE_PARAMS_PATH
    with open(p, "w") as f:
        json.dump(best, f, indent=2)
    if p.resolve() != PPO_TUNE_PARAMS_PATH.resolve():
        try:
            with open(PPO_TUNE_PARAMS_PATH, "w") as f:
                json.dump(best, f, indent=2)
        except OSError:
            pass


def _collect_study_summaries(
    storage: Any,
    study_name: str | None,
    into_names: list[str],
    into_studies: dict[str, Any],
) -> None:
    """Query one storage backend and merge results into the provided accumulators."""
    try:
        names = optuna.study.get_all_study_names(storage=storage)
    except Exception:
        return
    for name in names:
        if name not in into_names:
            into_names.append(name)
        if study_name is not None and name != study_name:
            continue
        if name in into_studies:
            continue
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            n_trials = len(study.trials)
            best_value = float(study.best_value) if study.best_trial else None
            into_studies[name] = {
                "n_trials": n_trials,
                "best_value": best_value,
                "best_trial": study.best_trial.number if study.best_trial else None,
            }
        except Exception:
            into_studies[name] = {"error": "failed to load"}


def get_tune_status(
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Return status of tuning studies: list of study names and summary for each.
    Queries both the SQLite and JournalStorage backends so studies from parallel
    tuning runs (which use JournalStorage) appear alongside SQLite studies.
    If study_name is given, include best trial and best value for that study.
    """
    study_names: list[str] = []
    studies: dict[str, Any] = {}

    if PPO_TUNE_STUDY_PATH.exists():
        try:
            _collect_study_summaries(get_tune_rdb_storage(), study_name, study_names, studies)
        except Exception:
            pass

    if _journal_storage_available() and PPO_TUNE_JOURNAL_LOG.exists():
        try:
            _collect_study_summaries(get_tune_journal_storage(), study_name, study_names, studies)
        except Exception:
            pass

    return {"study_names": sorted(study_names), "studies": studies}


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
