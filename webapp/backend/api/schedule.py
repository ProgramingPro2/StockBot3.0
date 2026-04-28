"""Read/write the scheduler YAML and proxy run-now/stop calls."""
from __future__ import annotations

from typing import Any

import yaml
from fastapi import APIRouter, Body, HTTPException

from .. import schedule_manager
from ..schemas import ScheduleDocument, ScheduleService, ScheduleStatus

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.get("", response_model=ScheduleDocument)
def get_schedule() -> ScheduleDocument:
    return ScheduleDocument(raw=schedule_manager.read_text(), parsed=schedule_manager.parse())


@router.put("", response_model=ScheduleDocument)
def put_schedule(payload: dict[str, str] = Body(...)) -> ScheduleDocument:
    text = payload.get("raw", "")
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"invalid YAML: {exc}") from exc
    schedule_manager.write_text(text)
    return ScheduleDocument(raw=schedule_manager.read_text(), parsed=schedule_manager.parse())


@router.get("/status", response_model=ScheduleStatus)
def status() -> ScheduleStatus:
    info: dict[str, Any] = schedule_manager.status()
    services: list[ScheduleService] = []
    for sid, svc in (info.get("services") or {}).items():
        services.append(
            ScheduleService(
                id=sid,
                type=str(svc.get("type", "")),
                in_window=bool(svc.get("in_window", False)),
                running=bool(svc.get("running", False)),
                pid=svc.get("pid"),
                log_path=svc.get("log_path"),
                window=svc.get("window") or {},
            )
        )
    return ScheduleStatus(timezone=info.get("timezone"), services=services)


@router.post("/{service_id}/run-now")
def run_now(service_id: str) -> dict[str, bool]:
    try:
        schedule_manager.run_now(service_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/{service_id}/stop")
def stop_service(service_id: str) -> dict[str, bool]:
    schedule_manager.stop(service_id)
    return {"ok": True}
