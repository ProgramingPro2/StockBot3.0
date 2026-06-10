"""List PPO models and manage the live default."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import env_manager
from ..paths import META_DIR, PROJECT_ROOT, TRADING_SYSTEM_DIR
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/api/models", tags=["models"])


def _live_default_setting() -> str | None:
    return os.environ.get("LIVE_PPO_MODEL_PATH") or None


def _portable_model_env_path(path: Path) -> str:
    """Path relative to project root (portable between Docker and host)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return path.name


def _resolve_live_model_path(raw: str | None) -> Path | None:
    """Resolve LIVE_PPO_MODEL_PATH to an on-disk model file."""
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    p = Path(text)
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        candidate = META_DIR / p.name
        return candidate.resolve() if candidate.exists() else None
    for base in (PROJECT_ROOT, TRADING_SYSTEM_DIR, META_DIR):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    candidate = (META_DIR / p.name).resolve()
    return candidate if candidate.exists() else None


def _read_sidecar(zip_path: Path) -> dict[str, Any]:
    sidecar = Path(str(zip_path) + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _list_models(live_resolved: Path | None = None) -> list[ModelInfo]:
    if not META_DIR.exists():
        return []
    out: list[ModelInfo] = []
    for path in sorted(META_DIR.glob("ppo_trader*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = path.stat()
        meta = _read_sidecar(path)
        resolved = path.resolve()
        out.append(
            ModelInfo(
                name=path.name,
                path=str(path),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
                is_live_default=(
                    live_resolved is not None and resolved == live_resolved.resolve()
                ),
                feature_source=meta.get("feature_source") or None,
                saved_at=meta.get("saved_at") or None,
                n_assets=meta.get("n_assets") or None,
            )
        )
    return out


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    live_raw = _live_default_setting()
    live_resolved = _resolve_live_model_path(live_raw)
    return ModelsResponse(
        models=_list_models(live_resolved),
        live_default=live_raw,
    )


@router.post("/live-default")
def set_live_default(name: str | None = None) -> dict[str, str | None]:
    """Update LIVE_PPO_MODEL_PATH in .env (project-relative under trading_system/)."""
    if name:
        path = META_DIR / name if not Path(name).is_absolute() else Path(name)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"model not found: {name}")
        portable = _portable_model_env_path(path)
        env_manager.update({"LIVE_PPO_MODEL_PATH": portable})
        os.environ["LIVE_PPO_MODEL_PATH"] = portable
        return {"live_default": portable}
    env_manager.update({"LIVE_PPO_MODEL_PATH": ""})
    os.environ.pop("LIVE_PPO_MODEL_PATH", None)
    return {"live_default": None}


@router.delete("/{name}")
def delete_model(name: str) -> dict[str, bool]:
    path = META_DIR / name
    if not path.exists() or path.suffix != ".zip" or not path.name.startswith("ppo_trader"):
        raise HTTPException(status_code=404, detail="model not found")
    path.unlink()
    return {"ok": True}
