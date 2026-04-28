"""List PPO models and manage the live default."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import env_manager
from ..paths import META_DIR
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/api/models", tags=["models"])


def _live_default_setting() -> str | None:
    return os.environ.get("LIVE_PPO_MODEL_PATH") or None


def _list_models() -> list[ModelInfo]:
    if not META_DIR.exists():
        return []
    out: list[ModelInfo] = []
    for path in sorted(META_DIR.glob("ppo_trader*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = path.stat()
        out.append(
            ModelInfo(
                name=path.name,
                path=str(path),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            )
        )
    return out


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    live = _live_default_setting()
    return ModelsResponse(
        models=_list_models(),
        live_default=live,
    )


@router.post("/live-default")
def set_live_default(name: str | None = None) -> dict[str, str | None]:
    """Update LIVE_PPO_MODEL_PATH in .env (resolved relative to META_DIR)."""
    if name:
        path = META_DIR / name if not Path(name).is_absolute() else Path(name)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"model not found: {name}")
        env_manager.update({"LIVE_PPO_MODEL_PATH": str(path)})
        os.environ["LIVE_PPO_MODEL_PATH"] = str(path)
        return {"live_default": str(path)}
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
