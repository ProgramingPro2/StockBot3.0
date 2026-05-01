"""
Parallel Optuna workers for tuning (same SQLite study) and optional tune bench.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from ..config import PPO_N_ENVS
from .proxy_specialists import precompute_proxy_feature_matrix
from .torch_vec_env import BatchedHistoricalTorchVecEnv
from .tuner import PPO_TUNE_STUDY_PATH, export_best_params_after_study, run_tuning

_log = logging.getLogger("boardroom")


def _start_parallel_tune_stall_watchdog(db_path: Path, stop_evt: threading.Event, stall_sec: float) -> threading.Thread:
    """
    Parent-only watchdog: if the study DB file sees no mtime change for ``stall_sec``,
    assume workers wedged and exit the process (so systemd/docker can restart).

    Opt-in via ``PPO_TUNE_STALL_TIMEOUT_SEC`` (seconds). Default 0 = disabled.
    Long single-trial phases may not touch the DB; set high enough for your
    ``--timesteps`` and ``--parallel-trials`` or leave disabled.
    """

    def run() -> None:
        try:
            last_mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
        except OSError:
            last_mtime = 0.0
        same_since = time.monotonic()
        poll = min(60.0, max(5.0, stall_sec / 10.0))
        while not stop_evt.wait(poll):
            try:
                mt = db_path.stat().st_mtime if db_path.exists() else 0.0
            except OSError:
                mt = last_mtime
            if mt != last_mtime:
                last_mtime = mt
                same_since = time.monotonic()
            elif time.monotonic() - same_since >= stall_sec:
                _log.error(
                    "Parallel tune stall watchdog: no writes to %s for %ss; exiting.",
                    db_path,
                    int(stall_sec),
                )
                os._exit(2)

    th = threading.Thread(target=run, name="tune-stall-watchdog", daemon=True)
    th.start()
    return th


def enable_sqlite_wal(db_path: Path) -> None:
    """Enable WAL on the Optuna SQLite file so concurrent workers do not deadlock."""
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=120.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
    finally:
        conn.close()


def _split_trials(n_trials: int, n_workers: int) -> list[int]:
    if n_workers <= 0:
        return [n_trials]
    base = n_trials // n_workers
    rem = n_trials % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _parallel_worker_entry(payload: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    """Child process: run a slice of Optuna trials on the shared study."""
    rank, n_trials_i, kwargs = payload
    if n_trials_i <= 0:
        return {}
    kw = dict(kwargs)
    kw["n_trials"] = n_trials_i
    kw["show_progress"] = bool(kw.get("show_progress")) and rank == 0
    return run_tuning(**kw)


def run_parallel_tuning(
    *,
    parallel_trials: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run ``run_tuning`` from ``parallel_trials`` processes sharing the same Optuna storage.

    Each worker receives ``ceil(n_trials / parallel_trials)`` trials (remainder spread
    across first workers).
    """
    from multiprocessing import get_context

    n_workers = max(1, int(parallel_trials))
    n_trials = int(kwargs.get("n_trials", 50))
    if n_workers <= 1:
        return run_tuning(**kwargs)

    enable_sqlite_wal(PPO_TUNE_STUDY_PATH)
    per_worker = _split_trials(n_trials, n_workers)
    ctx = get_context("spawn")
    payloads: list[tuple[int, int, dict[str, Any]]] = [
        (rank, per_worker[rank], kwargs) for rank in range(n_workers) if per_worker[rank] > 0
    ]
    if not payloads:
        return run_tuning(**kwargs)

    # multiprocessing.Pool workers are daemon=True and cannot spawn SubprocVecEnv children
    # inside Optuna trials. ProcessPoolExecutor workers are non-daemon (see PEP-like behavior
    # in CPython), so SB3 vector env subprocesses work.
    from concurrent.futures import ProcessPoolExecutor

    max_workers = len(payloads)
    stall_raw = os.environ.get("PPO_TUNE_STALL_TIMEOUT_SEC", "0").strip()
    try:
        stall_sec = float(stall_raw)
    except ValueError:
        stall_sec = 0.0
    stop_watchdog = threading.Event()
    watchdog_thread: threading.Thread | None = None
    if stall_sec > 0:
        watchdog_thread = _start_parallel_tune_stall_watchdog(PPO_TUNE_STUDY_PATH, stop_watchdog, stall_sec)
    try:
        if sys.version_info >= (3, 11):
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                list(ex.map(_parallel_worker_entry, payloads, chunksize=1))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_parallel_worker_entry, payloads, chunksize=1))
    finally:
        stop_watchdog.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)

    study_name = str(kwargs.get("study_name", "ppo_tune"))
    params_path = kwargs.get("params_path")
    return export_best_params_after_study(study_name, params_path)


def run_tune_bench(
    *,
    returns_array: np.ndarray,
    n_assets: int,
    n_envs: int | None = None,
    timesteps: int = 8192,
    historical_trading_dates: Any = None,
) -> dict[str, Any]:
    """
    Run a short PPO learn on legacy Subproc vs batched torch vec env; report steps/sec.

    Optional: sample GPU utilization via nvidia-smi once before and once after (best effort).
    """
    from functools import partial
    import sys

    from stable_baselines3.common.vec_env import SubprocVecEnv

    from .ppo_trader import _make_env_fn

    n_envs = n_envs if n_envs is not None else min(8, PPO_N_ENVS)
    returns_array = np.asarray(returns_array, dtype=np.float32)
    proxy = precompute_proxy_feature_matrix(returns_array)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _sample_gpu_util() -> str | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            return out.strip().split("\n")[0] if out.strip() else None
        except (OSError, subprocess.SubprocessError, FileNotFoundError):
            return None

    buf = int(n_envs) * 2048
    bs = min(512, buf)
    while bs > 64 and buf % bs != 0:
        bs -= 1
    fixed = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": bs,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    }

    def _bench_subproc() -> float:
        env_fns = [
            partial(
                _make_env_fn,
                mode="historical",
                historical_returns=returns_array,
                n_assets=n_assets,
                historical_ratio=1.0,
                specialist_fn=None,
                seed=i,
                daily_aux_features=None,
                historical_trading_dates=historical_trading_dates,
                proxy_feature_cache=proxy,
            )
            for i in range(n_envs)
        ]
        if sys.platform != "win32":
            env = SubprocVecEnv(env_fns, start_method="forkserver")
        else:
            env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)
        model = PPO("MlpPolicy", env, device=device, verbose=0, **fixed)
        t0 = time.perf_counter()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        elapsed = time.perf_counter() - t0
        env.close()
        return timesteps / max(elapsed, 1e-9)

    def _bench_torch_vec() -> float:
        env = BatchedHistoricalTorchVecEnv(
            returns_array,
            proxy,
            n_assets,
            n_envs,
            trading_dates=historical_trading_dates,
            device=device,
        )
        env = VecMonitor(env)
        model = PPO("MlpPolicy", env, device=device, verbose=0, **fixed)
        t0 = time.perf_counter()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        elapsed = time.perf_counter() - t0
        env.close()
        return timesteps / max(elapsed, 1e-9)

    gpu_before = _sample_gpu_util()
    sps_sub = _bench_subproc()
    mid = _sample_gpu_util()
    sps_torch = _bench_torch_vec()
    gpu_after = _sample_gpu_util()

    return {
        "device": device,
        "n_envs": n_envs,
        "timesteps": timesteps,
        "steps_per_sec_subproc": round(sps_sub, 2),
        "steps_per_sec_torch_vec": round(sps_torch, 2),
        "nvidia_smi_util_before": gpu_before,
        "nvidia_smi_util_mid": mid,
        "nvidia_smi_util_after": gpu_after,
    }
