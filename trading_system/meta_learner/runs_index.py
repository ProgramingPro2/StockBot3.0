"""Track training/tuning runs in a JSON index next to the model directory.

Each row records the run id (matching the TensorBoard subdir name), kind,
start/end timestamps, the resolved tickers/timesteps, and paths to the TB
log directory plus per-job metrics/result files. The web UI reads this to
render the TensorBoard run picker.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import META_LEARNER_MODEL_DIR

RUNS_INDEX_PATH = META_LEARNER_MODEL_DIR / "runs.json"
TENSORBOARD_DIR = META_LEARNER_MODEL_DIR / "tensorboard"

_lock = threading.Lock()


def _isoformat(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _read() -> list[dict[str, Any]]:
    if not RUNS_INDEX_PATH.exists():
        return []
    try:
        return json.loads(RUNS_INDEX_PATH.read_text() or "[]")
    except (json.JSONDecodeError, OSError):
        return []


def _write(rows: list[dict[str, Any]]) -> None:
    RUNS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2, default=str))
    tmp.replace(RUNS_INDEX_PATH)


def make_run_id(kind: str) -> str:
    """Generate a unique run id like ``train_20260427_152130`` or use BOARDROOM_RUN_ID."""
    forced = os.environ.get("BOARDROOM_RUN_ID")
    if forced:
        return forced
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{stamp}"


def tb_log_dir_for(run_id: str) -> Path:
    """Resolve the per-run TensorBoard log directory."""
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    return TENSORBOARD_DIR / run_id


def record_start(
    run_id: str,
    kind: str,
    *,
    tickers: Iterable[str] | None = None,
    timesteps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "kind": kind,
        "started": _isoformat(),
        "ended": None,
        "status": "running",
        "tickers": list(tickers) if tickers else None,
        "timesteps": timesteps,
        "tb_path": str(tb_log_dir_for(run_id)),
        "metrics_path": os.environ.get("BOARDROOM_METRICS_PATH"),
        "result_path": os.environ.get("BOARDROOM_RESULT_PATH"),
        "job_id": os.environ.get("BOARDROOM_JOB_ID"),
    }
    if extra:
        row.update(extra)
    with _lock:
        rows = _read()
        rows.append(row)
        _write(rows)
    return row


def record_end(run_id: str, *, status: str = "ok", extra: dict[str, Any] | None = None) -> None:
    with _lock:
        rows = _read()
        for row in rows:
            if row.get("run_id") == run_id:
                row["ended"] = _isoformat()
                row["status"] = status
                if extra:
                    row.update(extra)
                break
        else:
            rows.append({
                "run_id": run_id,
                "ended": _isoformat(),
                "status": status,
                **(extra or {}),
            })
        _write(rows)


def current_run_id() -> str | None:
    """Return ``BOARDROOM_RUN_ID`` when set (``cmd_train`` / job manager set this before specialists)."""
    forced = os.environ.get("BOARDROOM_RUN_ID", "").strip()
    return forced or None


def record_subphase(run_id: str, phase: str, *, tb_path: str) -> None:
    """Append a specialist or other sub-run entry to ``subphases`` on the matching run row."""
    entry: dict[str, Any] = {"phase": phase, "tb_path": tb_path, "started": _isoformat()}
    with _lock:
        rows = _read()
        for row in rows:
            if row.get("run_id") == run_id:
                subs = list(row.get("subphases") or [])
                subs.append(entry)
                row["subphases"] = subs
                _write(rows)
                return


def record_merge_extra(run_id: str, extra: dict[str, Any]) -> None:
    """Merge ``extra`` keys into an existing run row (e.g. PPO ``mode`` after env is built)."""
    if not extra:
        return
    with _lock:
        rows = _read()
        for row in rows:
            if row.get("run_id") == run_id:
                row.update(extra)
                _write(rows)
                return


def list_runs() -> list[dict[str, Any]]:
    with _lock:
        return list(reversed(_read()))


__all__ = [
    "RUNS_INDEX_PATH",
    "TENSORBOARD_DIR",
    "current_run_id",
    "list_runs",
    "make_run_id",
    "record_end",
    "record_merge_extra",
    "record_start",
    "record_subphase",
    "tb_log_dir_for",
]
