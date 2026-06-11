"""Job lifecycle endpoints + WebSocket streams for logs and metrics."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..job_manager import get_manager
from ..schemas import (
    AutodiscoverParams,
    BacktestParams,
    BenchmarkParams,
    JobInfo,
    JobListResponse,
    PaperLiveParams,
    SpecialistCacheParams,
    StartJobResponse,
    StopJobResponse,
    TrainParams,
    TrainUniverseParams,
    TuneParams,
    TuneStudiesResponse,
    TuneStudySummary,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/tune/studies", response_model=TuneStudiesResponse)
def list_tune_studies() -> TuneStudiesResponse:
    """List Optuna studies in local tuning storage (for resume picker in the UI)."""
    from trading_system.meta_learner.tuner import get_tune_status

    status = get_tune_status(study_name=None)
    names = sorted(status.get("study_names", []))
    raw = status.get("studies", {})
    studies: list[TuneStudySummary] = []
    for name in names:
        info = raw.get(name, {})
        if "error" in info:
            studies.append(TuneStudySummary(name=name, error=info["error"]))
            continue
        studies.append(
            TuneStudySummary(
                name=name,
                n_trials=int(info.get("n_trials", 0)),
                best_value=info.get("best_value"),
                best_trial=info.get("best_trial"),
            )
        )
    return TuneStudiesResponse(studies=studies)


@router.get("", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    mgr = get_manager()
    return JobListResponse(running=mgr.list_running(), history=mgr.list_history())


@router.get("/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> JobInfo:
    mgr = get_manager()
    info = mgr.get_info(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    return info


@router.post("/train/start", response_model=StartJobResponse)
def start_train(params: TrainParams) -> StartJobResponse:
    info = get_manager().start("train", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/tune/start", response_model=StartJobResponse)
def start_tune(params: TuneParams) -> StartJobResponse:
    info = get_manager().start("tune", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/backtest/start", response_model=StartJobResponse)
def start_backtest(params: BacktestParams) -> StartJobResponse:
    info = get_manager().start("backtest", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/paper/start", response_model=StartJobResponse)
def start_paper(params: PaperLiveParams) -> StartJobResponse:
    info = get_manager().start("paper", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/live/start", response_model=StartJobResponse)
def start_live(params: PaperLiveParams) -> StartJobResponse:
    info = get_manager().start("live", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/autodiscover/start", response_model=StartJobResponse)
def start_autodiscover(params: AutodiscoverParams) -> StartJobResponse:
    info = get_manager().start("autodiscover", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/train-universe/start", response_model=StartJobResponse)
def start_train_universe(params: TrainUniverseParams) -> StartJobResponse:
    info = get_manager().start("train-universe", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/benchmark/start", response_model=StartJobResponse)
def start_benchmark(params: BenchmarkParams) -> StartJobResponse:
    info = get_manager().start("benchmark-device", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.post("/specialist-cache/start", response_model=StartJobResponse)
def start_specialist_cache(params: SpecialistCacheParams) -> StartJobResponse:
    info = get_manager().start("specialist-cache", params.model_dump(exclude_none=True))
    return StartJobResponse(job=info)


@router.get("/specialist-cache/status")
def get_specialist_cache_status() -> dict[str, Any]:
    """Return all completed caches and any TFT checkpoints in the cache directory."""
    import numpy as np

    from trading_system.meta_learner.specialist_features import (
        SPECIALIST_FEATURES_CACHE_DIR,
    )

    caches: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    cache_dir = SPECIALIST_FEATURES_CACHE_DIR
    if not cache_dir.exists():
        return {"caches": caches, "checkpoints": checkpoints}

    # Completed caches (look for *_meta.json sidecar files)
    for meta_path in sorted(cache_dir.glob("*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            key = meta.get("cache_key") or meta_path.name.replace("_meta.json", "")
            npz = cache_dir / f"{key}.npz"
            if not npz.exists():
                continue
            slot_src = meta.get("slot_sources", {})
            real_slots = sum(1 for v in slot_src.values() if v in ("specialist", "real"))
            proxy_slots = len(slot_src) - real_slots
            caches.append({
                "cache_key": key,
                "start_date": meta.get("start_date"),
                "end_date": meta.get("end_date"),
                "n_days": meta.get("n_days"),
                "n_tickers": meta.get("n_tickers"),
                "tickers": meta.get("tickers", []),
                "size_bytes": npz.stat().st_size,
                "mtime": npz.stat().st_mtime,
                "slot_sources": slot_src,
                "real_slots": real_slots,
                "proxy_slots": proxy_slots,
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "refit_every": meta.get("refit_every"),
            })
        except Exception:
            continue

    # TFT checkpoints (partial builds)
    for ckpt in sorted(cache_dir.glob("tft_wf_ckpt_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = np.load(str(ckpt), allow_pickle=False)
            last_bar = int(data["last_bar"])
            out_shape = data["out"].shape
            n_days = out_shape[0]
            caches_for_key = [c for c in caches if c.get("cache_key") == ckpt.stem.replace("tft_wf_ckpt_", "")]
            # Try to find matching meta from any cache that overlaps
            checkpoints.append({
                "checkpoint_id": ckpt.name,
                "last_bar": last_bar,
                "n_days": n_days,
                "progress_pct": round(last_bar / max(n_days, 1), 4),
                "tickers": [],
                "start_date": None,
                "end_date": None,
                "mtime": ckpt.stat().st_mtime,
            })
        except Exception:
            continue

    return {"caches": caches, "checkpoints": checkpoints}


@router.delete("/specialist-cache/{cache_key}")
def delete_specialist_cache(cache_key: str) -> dict[str, Any]:
    """Delete a completed specialist cache (.npz + _meta.json)."""
    from trading_system.meta_learner.specialist_features import SPECIALIST_FEATURES_CACHE_DIR

    deleted: list[str] = []
    for suffix in (".npz", "_meta.json"):
        p = SPECIALIST_FEATURES_CACHE_DIR / f"{cache_key}{suffix}"
        if p.exists():
            p.unlink()
            deleted.append(p.name)
    return {"deleted": deleted}


@router.delete("/specialist-cache/checkpoint/{checkpoint_id}")
def delete_specialist_checkpoint(checkpoint_id: str) -> dict[str, Any]:
    """Delete a TFT checkpoint file."""
    from trading_system.meta_learner.specialist_features import SPECIALIST_FEATURES_CACHE_DIR

    # Sanitize: only allow the expected filename pattern
    if not checkpoint_id.startswith("tft_wf_ckpt_") or ".." in checkpoint_id:
        raise HTTPException(status_code=400, detail="invalid checkpoint id")
    p = SPECIALIST_FEATURES_CACHE_DIR / checkpoint_id
    if not p.exists():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    p.unlink()
    return {"deleted": checkpoint_id}


@router.post("/{job_id}/stop", response_model=StopJobResponse)
def stop_job(job_id: str) -> StopJobResponse:
    """Graceful stop: SIGTERM the subprocess so it can finish its current
    Optuna trial, SB3 rollout, or paper/live cycle and exit cleanly."""
    mgr = get_manager()
    info = mgr.get_info(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    out = mgr.stop(job_id) or info
    return StopJobResponse(ok=True, job=out)


@router.post("/{job_id}/kill", response_model=StopJobResponse)
def kill_job(job_id: str) -> StopJobResponse:
    """Force kill: SIGTERM then SIGKILL the subprocess group immediately.
    Use when graceful stop is wedged or too slow; partial saves not guaranteed."""
    mgr = get_manager()
    info = mgr.get_info(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    out = mgr.kill(job_id) or info
    return StopJobResponse(ok=True, job=out)


@router.get("/{job_id}/log")
def get_job_log(job_id: str) -> dict[str, Any]:
    mgr = get_manager()
    info = mgr.get_info(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    return {"text": mgr.read_log_text(job_id)}


@router.get("/{job_id}/metrics")
def get_job_metrics(job_id: str) -> dict[str, Any]:
    mgr = get_manager()
    info = mgr.get_info(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    return {"rows": mgr.read_metrics(job_id)}


@router.get("/{job_id}/result")
def get_job_result(job_id: str) -> dict[str, Any] | None:
    mgr = get_manager()
    return mgr.read_result(job_id)


@router.websocket("/{job_id}/ws/logs")
async def ws_logs(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    mgr = get_manager()
    queue = mgr.subscribe_logs(job_id)
    if queue is None:
        await websocket.send_json({"error": "job not found"})
        await websocket.close()
        return
    try:
        while True:
            line = await queue.get()
            if line is None:
                await websocket.send_json({"event": "end"})
                break
            await websocket.send_json({"event": "log", "line": line})
    except WebSocketDisconnect:
        pass
    finally:
        mgr.unsubscribe_logs(job_id, queue)


@router.websocket("/{job_id}/ws/metrics")
async def ws_metrics(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    mgr = get_manager()
    queue = mgr.subscribe_metrics(job_id)
    if queue is None:
        await websocket.send_json({"error": "job not found"})
        await websocket.close()
        return
    try:
        while True:
            raw = await queue.get()
            if raw is None:
                await websocket.send_json({"event": "end"})
                break
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            await websocket.send_json({"event": "metrics", "row": payload})
    except WebSocketDisconnect:
        pass
    finally:
        mgr.unsubscribe_metrics(job_id, queue)
