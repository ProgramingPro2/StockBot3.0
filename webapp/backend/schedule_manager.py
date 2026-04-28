"""Read/write the scheduler YAML and proxy status calls to the trading_system scheduler."""
from __future__ import annotations

from typing import Any

import yaml

from trading_system.scheduler import (
    run_job_now as _run_job_now,
    status as _scheduler_status,
    stop_service as _stop_service,
)

from .paths import SCHEDULE_FILE


def read_text() -> str:
    if SCHEDULE_FILE.exists():
        return SCHEDULE_FILE.read_text()
    return ""


def parse() -> dict[str, Any]:
    text = read_text()
    if not text.strip():
        return {}
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return {"_error": str(exc)}


def write_text(text: str) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(text)


def status() -> dict[str, Any]:
    try:
        return _scheduler_status(SCHEDULE_FILE)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "services": {}}


def run_now(service_id: str) -> None:
    _run_job_now(SCHEDULE_FILE, service_id)


def stop(service_id: str) -> None:
    _stop_service(service_id)
