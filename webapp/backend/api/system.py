"""Host telemetry endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from .. import system_info
from ..schemas import PpoCalibrationResponse, SystemInfo

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("", response_model=SystemInfo)
def get_system() -> SystemInfo:
    return system_info.snapshot()


@router.get("/ppo-calibration", response_model=PpoCalibrationResponse)
def get_ppo_calibration() -> PpoCalibrationResponse:
    """Heuristic PPO parallelism recommendations for this host (no training run)."""
    return system_info.ppo_calibration_hints()
