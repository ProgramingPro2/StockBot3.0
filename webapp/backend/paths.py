"""Filesystem locations used by the web backend.

Centralizing these makes it easy to override via environment variables in
Docker (e.g. when mounting volumes at non-standard paths).
"""
from __future__ import annotations

import os
from pathlib import Path

from trading_system.config import (
    BASE_DIR as TRADING_BASE_DIR,
    META_LEARNER_MODEL_DIR,
    OUTPUT_DIR,
)

# Default HTTP port for the web UI (avoids common dev ports like 8000/8080/3000).
BOARDROOM_HTTP_PORT_DEFAULT: int = 28473

PROJECT_ROOT: Path = TRADING_BASE_DIR.parent
TRADING_SYSTEM_DIR: Path = TRADING_BASE_DIR
OUTPUTS_DIR: Path = OUTPUT_DIR
MODELS_DIR: Path = META_LEARNER_MODEL_DIR.parent
META_DIR: Path = META_LEARNER_MODEL_DIR
TENSORBOARD_DIR: Path = META_DIR / "tensorboard"

JOBS_DIR: Path = OUTPUTS_DIR / "jobs"
JOBS_REGISTRY: Path = JOBS_DIR / "registry.json"

ENV_FILE: Path = Path(os.environ.get("BOARDROOM_ENV_FILE", str(PROJECT_ROOT / ".env")))
ENV_EXAMPLE_FILE: Path = PROJECT_ROOT / ".env.example"
ENV_HINTS_FILE: Path = PROJECT_ROOT / "webapp" / "backend" / "data" / "env_var_hints.json"
SCHEDULE_FILE: Path = Path(
    os.environ.get("BOARDROOM_SCHEDULE_FILE", str(PROJECT_ROOT / "config" / "schedule.yaml"))
)

FRONTEND_DIST: Path = Path(
    os.environ.get("BOARDROOM_FRONTEND_DIST", str(PROJECT_ROOT / "webapp" / "frontend" / "dist"))
)


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
