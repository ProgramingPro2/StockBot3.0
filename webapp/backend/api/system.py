"""Host telemetry endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from .. import system_info
from ..schemas import SystemInfo

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("", response_model=SystemInfo)
def get_system() -> SystemInfo:
    return system_info.snapshot()
