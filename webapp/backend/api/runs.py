"""TensorBoard run index for the UI."""
from __future__ import annotations

from fastapi import APIRouter

from trading_system.meta_learner.runs_index import list_runs

from ..schemas import RunInfo, RunsResponse

router = APIRouter(prefix="/api/runs", tags=["runs"])


_KNOWN_FIELDS = set(RunInfo.model_fields.keys())


@router.get("", response_model=RunsResponse)
def get_runs() -> RunsResponse:
    raw = list_runs()
    runs: list[RunInfo] = []
    for row in raw:
        known = {k: row.get(k) for k in _KNOWN_FIELDS if k in row}
        extra = {k: v for k, v in row.items() if k not in _KNOWN_FIELDS}
        try:
            runs.append(RunInfo(**known, extra=extra))
        except Exception:  # noqa: BLE001
            continue
    return RunsResponse(runs=runs)
