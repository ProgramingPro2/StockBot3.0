"""Serve files from the outputs directory (PNGs, log files, etc.)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from ..paths import OUTPUTS_DIR
from ..schemas import OutputItem, OutputsResponse

router = APIRouter(prefix="/api/outputs", tags=["outputs"])


def _kind(path: Path) -> Literal["png", "log", "txt", "other"]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "png"
    if suffix == ".log":
        return "log"
    if suffix == ".txt":
        return "txt"
    return "other"


def _resolve(name: str) -> Path:
    base = OUTPUTS_DIR.resolve()
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes outputs/") from exc
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="not found")
    return candidate


@router.get("", response_model=OutputsResponse)
def list_outputs() -> OutputsResponse:
    if not OUTPUTS_DIR.exists():
        return OutputsResponse(items=[])
    items: list[OutputItem] = []
    for path in OUTPUTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        rel = path.relative_to(OUTPUTS_DIR)
        st = path.stat()
        kind = _kind(path)
        series_name: str | None = None
        if kind == "png":
            json_sibling = path.with_suffix(".json")
            if json_sibling.is_file():
                series_name = str(json_sibling.relative_to(OUTPUTS_DIR))
        items.append(
            OutputItem(
                name=str(rel),
                path=str(path),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
                kind=kind,
                series_name=series_name,
            )
        )
    items.sort(key=lambda i: i.mtime, reverse=True)
    return OutputsResponse(items=items)


@router.get("/file/{name:path}")
def get_file(name: str):
    path = _resolve(name)
    media_type = None
    if path.suffix.lower() == ".png":
        media_type = "image/png"
        return FileResponse(path, media_type=media_type)
    if path.suffix.lower() in {".log", ".txt", ".jsonl", ".json"}:
        return PlainTextResponse(path.read_text(errors="replace"))
    return FileResponse(path)
