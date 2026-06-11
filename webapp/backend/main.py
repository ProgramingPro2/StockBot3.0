"""Boardroom web UI - FastAPI entrypoint.

Run locally (default port 28473, override with BOARDROOM_PORT):

    uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 28473
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import tensorboard_proxy
from .api import config as config_routes
from .api import jobs as jobs_routes
from .api import models as models_routes
from .api import outputs as outputs_routes
from .api import runs as runs_routes
from .api import schedule as schedule_routes
from .api import system as system_routes
from .api import tickers as tickers_routes
from .job_manager import get_manager, shutdown
from .paths import BOARDROOM_HTTP_PORT_DEFAULT, FRONTEND_DIST, ensure_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    get_manager().attach_loop(asyncio.get_running_loop())
    if os.environ.get("BOARDROOM_DISABLE_TB", "").lower() not in ("1", "true", "yes"):
        tensorboard_proxy.start_in_background()
    try:
        yield
    finally:
        try:
            shutdown()
        finally:
            tensorboard_proxy.stop()


app = FastAPI(
    title="The Boardroom - Web UI",
    version="2.0.0",
    description="Control plane for The Boardroom hierarchical multi-agent trading system.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("BOARDROOM_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_routes.router)
app.include_router(config_routes.router)
app.include_router(models_routes.router)
app.include_router(outputs_routes.router)
app.include_router(schedule_routes.router)
app.include_router(tickers_routes.router)
app.include_router(runs_routes.router)
app.include_router(system_routes.router)
app.include_router(tensorboard_proxy.router)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/version", tags=["system"])
def version() -> dict[str, str]:
    return {"version": app.version, "name": app.title}


# --- Frontend static serving (built React app) ---


def _mount_frontend() -> None:
    if not FRONTEND_DIST.exists():
        return
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False),
        name="frontend-assets",
    )
    icon_dir = FRONTEND_DIST
    if (icon_dir / "favicon.ico").exists():
        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> FileResponse:  # noqa: D401
            return FileResponse(icon_dir / "favicon.ico")

    index = FRONTEND_DIST / "index.html"

    @app.get("/", include_in_schema=False)
    def root_index() -> Response:
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"detail": "frontend not built"}, status_code=404)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Response:
        if full_path.startswith(("api/", "tb", "assets/")):
            return JSONResponse({"detail": "not found"}, status_code=404)
        # Try direct file (e.g. logo.svg)
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"detail": "frontend not built"}, status_code=404)


_mount_frontend()


def main() -> None:
    """uvicorn entry point for ``python -m webapp.backend``."""
    import uvicorn

    uvicorn.run(
        "webapp.backend.main:app",
        host=os.environ.get("BOARDROOM_HOST", "0.0.0.0"),
        port=int(os.environ.get("BOARDROOM_PORT", str(BOARDROOM_HTTP_PORT_DEFAULT))),
        reload=os.environ.get("BOARDROOM_RELOAD", "0") in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
