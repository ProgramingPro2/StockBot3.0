"""Subprocess-backed Boardroom job manager.

Spawns ``python -m trading_system <cmd> ...`` with a per-job working
directory, tees stdout/stderr to ``log.txt`` and broadcasts both log lines
and ``metrics.jsonl`` rows over asyncio queues so WebSocket clients can
stream them.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .paths import JOBS_DIR, JOBS_REGISTRY, PROJECT_ROOT, ensure_dirs
from .schemas import JobInfo, JobKind


SINGLE_INSTANCE_KINDS: set[JobKind] = {"paper", "live"}
ALL_KINDS: set[JobKind] = {
    "train",
    "tune",
    "backtest",
    "paper",
    "live",
    "autodiscover",
    "train-universe",
    "benchmark-device",
    "test",
}


@dataclass
class _Broadcaster:
    """Many-to-many fan-out for log lines or metric rows."""

    subscribers: list[asyncio.Queue[str | None]] = field(default_factory=list)
    closed: bool = False
    history: list[str] = field(default_factory=list)
    history_limit: int = 2000
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def subscribe(self, replay: bool = True) -> asyncio.Queue[str | None]:
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1024)
        with self._lock:
            self.subscribers.append(q)
            history = list(self.history) if replay else []
            already_closed = self.closed
        for line in history:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                break
        if already_closed:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue[str | None]) -> None:
        with self._lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def push(self, line: str) -> None:
        with self._lock:
            if self.closed:
                return
            self.history.append(line)
            if len(self.history) > self.history_limit:
                del self.history[: len(self.history) - self.history_limit]
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                # drop the oldest item to make room
                try:
                    q.get_nowait()
                    q.put_nowait(line)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


@dataclass
class _Job:
    info: JobInfo
    process: subprocess.Popen[bytes] | None = None
    log_broadcaster: _Broadcaster = field(default_factory=_Broadcaster)
    metrics_broadcaster: _Broadcaster = field(default_factory=_Broadcaster)
    log_path: Path | None = None
    metrics_path: Path | None = None
    result_path: Path | None = None
    job_dir: Path | None = None
    threads: list[threading.Thread] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _argv_for(kind: JobKind, params: dict[str, Any]) -> list[str]:
    """Translate params into a Typer-compatible CLI invocation.

    All commands run with ``python -m trading_system <kind> <flags>``.
    """
    argv: list[str] = [sys.executable, "-u", "-m", "trading_system", kind]
    bool_flags = {
        "use_historical": "--use-historical",
        "use_tuned": "--use-tuned",
        "retrain": "--retrain",
        "ignore_early_stopping": "--ignore-early-stopping",
        "autodiscover": "--autodiscover",
        "save": "--save",
        "single_thread": "--single-thread",
        "new": "--new",
        "debug": "--debug",
        "torch_vec": "--torch-vec",
    }
    value_flags = {
        "timesteps": "--timesteps",
        "tickers": "--tickers",
        "historical_ratio": "--historical-ratio",
        "n_envs": "--n-envs",
        "model": "--model",
        "interval": "--interval",
        "n": "--n",
        "n_trials": "--n-trials",
        "timeout": "--timeout",
        "study_name": "--study-name",
        "modes": "--modes",
        "forward_days": "--forward-days",
        "n_estimators": "--n-estimators",
        "max_depth": "--max-depth",
        "learning_rate": "--learning-rate",
        "tuned_study": "--tuned-study",
        "parallel_trials": "--parallel-trials",
    }
    for key, flag in value_flags.items():
        if key in params and params[key] is not None and params[key] != "":
            argv.extend([flag, str(params[key])])
    for key, flag in bool_flags.items():
        if params.get(key):
            argv.append(flag)
    return argv


def _open_log(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab", buffering=0)


def _signal_pid_process_group(pid: int, sig: int) -> None:
    """Send ``sig`` to the process group for ``pid``, then fall back to ``pid`` alone."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _load_registry() -> list[JobInfo]:
    if not JOBS_REGISTRY.exists():
        return []
    try:
        rows = json.loads(JOBS_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out: list[JobInfo] = []
    for row in rows:
        try:
            out.append(JobInfo(**row))
        except Exception:  # noqa: BLE001
            continue
    return out


class JobManager:
    """Owns the active jobs map and their subprocess lifecycle."""

    def __init__(self) -> None:
        ensure_dirs()
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: list[JobInfo] = _load_registry()
        self._reconcile_stale_registry_rows()

    # --- runtime wiring ---
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _write_registry_disk(self, active_jobs: Iterable[_Job]) -> None:
        """Persist merged job rows so restarts do not drop archived entries.

        Active (in-memory) jobs overwrite same-id rows from ``_history``.
        """
        ensure_dirs()
        with self._lock:
            by_id: dict[str, JobInfo] = {h.id: h.model_copy() for h in self._history}
            for job in active_jobs:
                by_id[job.info.id] = job.info.model_copy()
            rows = sorted(by_id.values(), key=lambda x: (x.started or 0.0, x.id))
            payload = [r.model_dump() for r in rows]
        tmp = JOBS_REGISTRY.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(JOBS_REGISTRY)

    def _reconcile_stale_registry_rows(self) -> None:
        """Turn non-terminal on-disk jobs into cancelled after a process restart.

        Without this, the UI keeps showing ``running`` while ``stop``/``kill``
        only touch live ``_Job`` handles, so controls appear broken.
        """
        terminal = frozenset({"completed", "failed", "cancelled"})
        now = time.time()
        note = (
            "Closed after server restart (job was no longer tracked). "
            "If a pid was recorded, SIGTERM was sent to its process group."
        )
        changed = False
        new_list: list[JobInfo] = []
        for h in self._history:
            if h.status in terminal:
                new_list.append(h)
                continue
            changed = True
            if h.pid:
                _signal_pid_process_group(h.pid, signal.SIGTERM)
            new_list.append(
                h.model_copy(
                    update={
                        "status": "cancelled",
                        "ended": now,
                        "exit_code": -1,
                        "note": note + (f" pid={h.pid}" if h.pid else ""),
                    }
                )
            )
        if changed:
            self._history = new_list
            self._write_registry_disk(())

    def _signal_history_job(self, job_id: str, *, use_kill: bool) -> JobInfo | None:
        """Stop/kill a job that only exists in persisted history (orphan row)."""
        with self._lock:
            idx: int | None = None
            snapshot: JobInfo | None = None
            for i, h in enumerate(self._history):
                if h.id == job_id:
                    idx, snapshot = i, h
                    break
        if idx is None or snapshot is None:
            return None
        if snapshot.status not in {"pending", "running", "stopping"}:
            return snapshot.model_copy()
        pid = snapshot.pid
        if pid:
            if use_kill:
                _signal_pid_process_group(pid, signal.SIGTERM)
                time.sleep(0.5)
                _signal_pid_process_group(pid, signal.SIGKILL)
            else:
                _signal_pid_process_group(pid, signal.SIGTERM)
        now = time.time()
        note = (
            "Force-killed orphaned subprocess (job was not tracked by this server process)."
            if use_kill
            else "SIGTERM sent to orphaned subprocess (job was not tracked by this server process)."
        )
        with self._lock:
            for i, h in enumerate(self._history):
                if h.id == job_id:
                    upd = h.model_copy(
                        update={
                            "status": "cancelled",
                            "ended": now,
                            "exit_code": -1 if use_kill else None,
                            "note": note + (f" pid={pid}" if pid else ""),
                        }
                    )
                    self._history[i] = upd
                    break
        self._write_registry_disk(self._jobs.values())
        return self.get_info(job_id)

    # --- queries ---
    def list_running(self) -> list[JobInfo]:
        with self._lock:
            return [job.info.model_copy() for job in self._jobs.values() if job.info.status in {"pending", "running", "stopping"}]

    def list_history(self, limit: int = 50) -> list[JobInfo]:
        with self._lock:
            done = [job.info.model_copy() for job in self._jobs.values() if job.info.status in {"completed", "failed", "cancelled"}]
            done.sort(key=lambda j: j.ended or j.started or 0, reverse=True)
            history = list(self._history)
            existing_ids = {j.id for j in done}
            for h in history:
                if h.id not in existing_ids:
                    done.append(h)
            return done[:limit]

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_info(self, job_id: str) -> JobInfo | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return job.info.model_copy()
            for h in self._history:
                if h.id == job_id:
                    return h.model_copy()
            return None

    # --- start / stop ---
    def start(self, kind: JobKind, params: dict[str, Any]) -> JobInfo:
        if kind not in ALL_KINDS:
            raise ValueError(f"Unknown job kind: {kind}")
        if kind in SINGLE_INSTANCE_KINDS:
            self._enforce_single_instance(kind)
        # Don't allow paper + live concurrently.
        if kind == "live":
            self._enforce_single_instance("paper", note="paper trading is currently running")
        if kind == "paper":
            self._enforce_single_instance("live", note="live trading is currently running")

        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "log.txt"
        metrics_path = job_dir / "metrics.jsonl"
        result_path = job_dir / "result.json"
        params_path = job_dir / "params.json"
        params_path.write_text(json.dumps(params, indent=2, default=str))

        argv = _argv_for(kind, params)
        env = os.environ.copy()
        env["BOARDROOM_METRICS_PATH"] = str(metrics_path)
        env["BOARDROOM_RESULT_PATH"] = str(result_path)
        env["BOARDROOM_JOB_ID"] = job_id
        env["BOARDROOM_RUN_KIND"] = kind
        env["PYTHONUNBUFFERED"] = "1"

        info = JobInfo(
            id=job_id,
            kind=kind,
            status="pending",
            started=time.time(),
            args=argv,
            params=params,
            log_path=str(log_path),
            metrics_path=str(metrics_path),
            result_path=str(result_path),
        )
        job = _Job(
            info=info,
            log_path=log_path,
            metrics_path=metrics_path,
            result_path=result_path,
            job_dir=job_dir,
            loop=self._loop,
        )

        log_file = _open_log(log_path)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            log_file.close()
            raise RuntimeError(f"Could not spawn process: {exc}") from exc

        job.process = proc
        job.info.pid = proc.pid
        job.info.status = "running"

        log_thread = threading.Thread(
            target=self._reader_loop,
            args=(job, proc, log_file),
            name=f"job-{job_id}-log",
            daemon=True,
        )
        metrics_thread = threading.Thread(
            target=self._metrics_tail_loop,
            args=(job,),
            name=f"job-{job_id}-metrics",
            daemon=True,
        )
        log_thread.start()
        metrics_thread.start()
        job.threads.extend([log_thread, metrics_thread])

        with self._lock:
            self._jobs[job_id] = job
        self._write_registry_disk(self._jobs.values())
        return info.model_copy()

    def stop(self, job_id: str) -> JobInfo | None:
        """Request a graceful shutdown: SIGTERM only, no escalation.

        The subprocess listens for SIGTERM and finishes its current unit of
        work (Optuna trial, SB3 rollout, paper/live cycle) before exiting.
        Use :meth:`kill` to force-quit if the process refuses to exit.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job and job.process and job.process.poll() is None:
            job.cancel_event.set()
            job.info.status = "stopping"
            self._send_signal(job, signal.SIGTERM)
            return job.info.model_copy()
        if job:
            return job.info.model_copy()
        return self._signal_history_job(job_id, use_kill=False)

    def kill(self, job_id: str, grace_seconds: float = 0.5) -> JobInfo | None:
        """Force-kill the job: brief SIGTERM, then SIGKILL the process group."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job and job.process and job.process.poll() is None:
            job.cancel_event.set()
            job.info.status = "stopping"
            self._send_signal(job, signal.SIGTERM)
            try:
                job.process.wait(timeout=max(0.0, grace_seconds))
            except subprocess.TimeoutExpired:
                self._send_signal(job, signal.SIGKILL)
                try:
                    job.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            return job.info.model_copy()
        if job:
            return job.info.model_copy()
        return self._signal_history_job(job_id, use_kill=True)

    def _send_signal(self, job: _Job, sig: int) -> None:
        if not job.process:
            return
        try:
            os.killpg(os.getpgid(job.process.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            if sig == signal.SIGKILL:
                job.process.kill()
            else:
                job.process.terminate()
        except Exception:  # noqa: BLE001
            pass

    def stop_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.info.status in {"running", "pending", "stopping"}:
                self.kill(job.info.id)

    # --- streaming ---
    def subscribe_logs(self, job_id: str, replay: bool = True) -> asyncio.Queue[str | None] | None:
        job = self.get(job_id)
        if not job:
            return None
        return job.log_broadcaster.subscribe(replay=replay)

    def subscribe_metrics(self, job_id: str, replay: bool = True) -> asyncio.Queue[str | None] | None:
        job = self.get(job_id)
        if not job:
            return None
        return job.metrics_broadcaster.subscribe(replay=replay)

    def unsubscribe_logs(self, job_id: str, q: asyncio.Queue[str | None]) -> None:
        job = self.get(job_id)
        if job:
            job.log_broadcaster.unsubscribe(q)

    def unsubscribe_metrics(self, job_id: str, q: asyncio.Queue[str | None]) -> None:
        job = self.get(job_id)
        if job:
            job.metrics_broadcaster.unsubscribe(q)

    def read_log_text(self, job_id: str, max_bytes: int = 256_000) -> str:
        job = self.get(job_id)
        if not job or not job.log_path or not job.log_path.exists():
            for h in self._history:
                if h.id == job_id and h.log_path:
                    p = Path(h.log_path)
                    if p.exists():
                        return p.read_text()[-max_bytes:]
            return ""
        return job.log_path.read_text()[-max_bytes:]

    def read_metrics(self, job_id: str) -> list[dict[str, Any]]:
        info = self.get_info(job_id)
        if not info or not info.metrics_path:
            return []
        path = Path(info.metrics_path)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return rows

    def read_result(self, job_id: str) -> dict[str, Any] | None:
        info = self.get_info(job_id)
        if not info or not info.result_path:
            return None
        p = Path(info.result_path)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    # --- internals ---
    def _enforce_single_instance(self, kind: str, *, note: str | None = None) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job.info.kind == kind and job.info.status in {"pending", "running", "stopping"}:
                    raise RuntimeError(
                        note or f"A '{kind}' job is already running (id={job.info.id})."
                    )

    def _reader_loop(self, job: _Job, proc: subprocess.Popen[bytes], log_file: Any) -> None:
        try:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                try:
                    log_file.write(raw)
                except Exception:  # noqa: BLE001
                    pass
                line = raw.decode(errors="replace").rstrip("\n")
                job.log_broadcaster.push(line)
        finally:
            try:
                log_file.close()
            except Exception:  # noqa: BLE001
                pass
            code = proc.wait()
            job.info.exit_code = code
            job.info.ended = time.time()
            job.info.status = "completed" if code == 0 else (
                "cancelled" if job.cancel_event.is_set() else "failed"
            )
            job.log_broadcaster.close()
            job.metrics_broadcaster.close()
            with self._lock:
                self._history.insert(0, job.info.model_copy())
                self._history = self._history[:200]
            self._write_registry_disk(self._jobs.values())

    def _metrics_tail_loop(self, job: _Job) -> None:
        path = job.metrics_path
        if path is None:
            return
        # Wait briefly for the file to appear
        for _ in range(50):
            if path.exists() or job.info.status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        try:
            handle = path.open("r")
        except FileNotFoundError:
            return
        try:
            while True:
                if job.info.status in {"completed", "failed", "cancelled"} and not job.process_alive():
                    # Drain the rest, then exit.
                    rest = handle.read()
                    if rest:
                        for line in rest.splitlines():
                            line = line.strip()
                            if line:
                                job.metrics_broadcaster.push(line)
                    break
                where = handle.tell()
                line = handle.readline()
                if not line:
                    time.sleep(0.5)
                    handle.seek(where)
                    continue
                trimmed = line.strip()
                if trimmed:
                    job.metrics_broadcaster.push(trimmed)
        finally:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass


def _process_alive(self) -> bool:  # type: ignore[no-redef]
    return bool(self.process and self.process.poll() is None)


_Job.process_alive = _process_alive  # type: ignore[attr-defined]


_manager: JobManager | None = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def shutdown() -> None:
    global _manager
    if _manager is not None:
        _manager.stop_all()
        _manager = None
