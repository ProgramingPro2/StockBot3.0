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
from .tuner import (
    PPO_TUNE_JOURNAL_LOG,
    PPO_TUNE_STUDY_PATH,
    _journal_storage_available,
    export_best_params_after_study,
    run_tuning,
    split_tune_train_val,
)

_log = logging.getLogger("boardroom")

# Estimated CUDA context + small MlpPolicy footprint per parallel Optuna worker.
_GPU_BYTES_PER_WORKER = int(0.6 * 1024**3)
_GPU_RESERVE_BYTES = int(1.0 * 1024**3)


def _cap_parallel_workers_by_gpu(requested: int) -> int:
    """Cap parallel Optuna workers so multiple CUDA contexts do not OOM the GPU."""
    requested = max(1, int(requested))
    if not torch.cuda.is_available():
        _log.info("Parallel tuning: %d workers (CUDA unavailable; CPU-only).", requested)
        return requested
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        _log.warning("Parallel tuning: could not read GPU memory (%s); using %d workers.", exc, requested)
        return requested

    usable = max(0, int(free_bytes) - _GPU_RESERVE_BYTES)
    max_by_mem = max(1, usable // max(_GPU_BYTES_PER_WORKER, 1))
    capped = min(requested, max_by_mem)
    if capped < requested:
        _log.warning(
            "Parallel tuning: capping workers from %d to %d "
            "(free GPU mem %.2f GB, reserve %.2f GB, ~%.2f GB/worker).",
            requested,
            capped,
            free_bytes / 1e9,
            _GPU_RESERVE_BYTES / 1e9,
            _GPU_BYTES_PER_WORKER / 1e9,
        )
    else:
        _log.info(
            "Parallel tuning: %d workers (free GPU mem %.2f GB, reserve %.2f GB, ~%.2f GB/worker).",
            capped,
            free_bytes / 1e9,
            _GPU_RESERVE_BYTES / 1e9,
            _GPU_BYTES_PER_WORKER / 1e9,
        )
    return capped


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


_JOURNAL_NOT_FOUND_MSG = "Record does not exist."


def _parallel_worker_entry(payload: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    """Child process: run a slice of Optuna trials on the shared study."""
    rank, n_trials_i, kwargs = payload
    if n_trials_i <= 0:
        return {}
    kw = dict(kwargs)
    kw["n_trials"] = n_trials_i
    kw["show_progress"] = bool(kw.get("show_progress")) and rank == 0
    try:
        return run_tuning(**kw)
    except KeyError as exc:
        # Optuna JournalStorage raises KeyError('Record does not exist.') when the
        # journal log is in a torn state during graceful shutdown (SIGTERM). Treat
        # this as a clean exit so the parent can still recover best params.
        if _JOURNAL_NOT_FOUND_MSG in str(exc):
            _log.debug(
                "Worker %d: Optuna journal KeyError during shutdown (ignored): %s", rank, exc
            )
            return {}
        raise


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

    n_workers = _cap_parallel_workers_by_gpu(max(1, int(parallel_trials)))
    n_trials = int(kwargs.get("n_trials", 50))
    if n_workers <= 1:
        return run_tuning(**kwargs)

    kwargs = dict(kwargs)
    use_journal = _journal_storage_available()
    kwargs["use_journal_storage"] = use_journal
    if use_journal:
        _log.info("Parallel tuning: using Optuna JournalStorage (lower SQLite contention).")
    else:
        enable_sqlite_wal(PPO_TUNE_STUDY_PATH)
        _log.warning(
            "Parallel tuning: JournalStorage unavailable; using SQLite with WAL."
        )

    returns_array = np.asarray(kwargs["returns_array"], dtype=np.float32)
    train_returns, val_returns = split_tune_train_val(returns_array)
    kwargs["precomputed_train_proxy"] = precompute_proxy_feature_matrix(train_returns)
    if len(val_returns) > 21:
        kwargs["precomputed_val_proxy"] = precompute_proxy_feature_matrix(val_returns)
    else:
        kwargs["precomputed_val_proxy"] = None

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
    from concurrent.futures.process import BrokenProcessPool

    max_workers = len(payloads)
    stall_raw = os.environ.get("PPO_TUNE_STALL_TIMEOUT_SEC", "0").strip()
    try:
        stall_sec = float(stall_raw)
    except ValueError:
        stall_sec = 0.0
    stop_watchdog = threading.Event()
    watchdog_thread: threading.Thread | None = None
    watchdog_path = PPO_TUNE_JOURNAL_LOG if use_journal else PPO_TUNE_STUDY_PATH
    if stall_sec > 0:
        watchdog_thread = _start_parallel_tune_stall_watchdog(watchdog_path, stop_watchdog, stall_sec)
    try:
        if sys.version_info >= (3, 11):
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                list(ex.map(_parallel_worker_entry, payloads, chunksize=1))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_parallel_worker_entry, payloads, chunksize=1))
    except (BrokenProcessPool, KeyError) as exc:
        # BrokenProcessPool: worker process was killed (OOM, hard SIGKILL).
        # KeyError: Optuna JournalStorage 'Record does not exist.' can escape the
        # worker's own handler if the journal tears mid-replay (e.g. SIGTERM race).
        _log.error(
            "Parallel tune: a worker exited abruptly (%s). "
            "Saving best-known params from the study (SIGTERM or Subproc env crash).",
            exc,
        )
        study_name = str(kwargs.get("study_name", "ppo_tune"))
        params_path = kwargs.get("params_path")
        return export_best_params_after_study(
            study_name,
            params_path,
            use_journal_storage=use_journal,
        )
    finally:
        stop_watchdog.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.0)

    study_name = str(kwargs.get("study_name", "ppo_tune"))
    params_path = kwargs.get("params_path")
    return export_best_params_after_study(
        study_name,
        params_path,
        use_journal_storage=use_journal,
    )


def run_tune_bench(
    *,
    returns_array: np.ndarray,
    n_assets: int,
    n_envs: int | None = None,
    timesteps: int = 8192,
    historical_trading_dates: Any = None,
    torch_vec_only: bool = False,
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
    if torch_vec_only:
        sps_sub = 0.0
        mid = _sample_gpu_util()
        sps_torch = _bench_torch_vec()
        gpu_after = _sample_gpu_util()
    else:
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


def run_tune_bench_sweep(
    *,
    returns_array: np.ndarray,
    n_assets: int,
    timesteps: int = 8192,
    historical_trading_dates: Any = None,
    n_envs_grid: list[int] | None = None,
) -> dict[str, Any]:
    """
    Sweep torch-vec ``n_envs`` and estimate trials/hour for parallel worker counts.

    Returns the best configuration by estimated env-steps per second (proxy for HPO throughput).
    """
    if n_envs_grid is None:
        n_envs_grid = [64, 128, 256, 512]

    returns_array = np.asarray(returns_array, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for n_envs in n_envs_grid:
        try:
            bench = run_tune_bench(
                returns_array=returns_array,
                n_assets=n_assets,
                n_envs=n_envs,
                timesteps=timesteps,
                historical_trading_dates=historical_trading_dates,
                torch_vec_only=True,
            )
        except Exception as exc:
            _log.warning("Bench sweep skipped n_envs=%s: %s", n_envs, exc)
            continue
        sps = float(bench.get("steps_per_sec_torch_vec") or 0.0)
        row = {
            "n_envs": n_envs,
            "steps_per_sec_torch_vec": sps,
            "steps_per_sec_subproc": bench.get("steps_per_sec_subproc"),
        }
        for parallel_trials in (1, 2, 3, 4, 6):
            capped = _cap_parallel_workers_by_gpu(parallel_trials)
            row[f"est_sps_parallel_{parallel_trials}"] = round(sps * capped, 2)
        rows.append(row)
        score = sps * _cap_parallel_workers_by_gpu(3)
        if best is None or score > float(best.get("score", 0.0)):
            best = {
                "n_envs": n_envs,
                "parallel_trials": min(3, _cap_parallel_workers_by_gpu(3)),
                "steps_per_sec_torch_vec": sps,
                "score": score,
                "use_torch_vec": True,
            }

    return {"rows": rows, "best": best or {}}
