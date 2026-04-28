"""Reverse-proxy TensorBoard so it shares the same origin as the web UI.

Spawns ``tensorboard --logdir <META>/tensorboard`` once on startup and
forwards `/tb/...` HTTP requests to it. WebSocket support isn't needed by
the TensorBoard frontend.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Response

from .paths import META_DIR

router = APIRouter(tags=["tensorboard"])

# Must match TensorBoard's --path_prefix so the SPA issues `/tb/...` API paths
# under the same origin as the iframe (new TB uses root-absolute `/experiment/...`
# unless the server is started with this prefix).
_TB_PATH_PREFIX = os.environ.get("BOARDROOM_TB_PATH_PREFIX", "/tb").rstrip("/") or "/tb"

_TB_HOST = os.environ.get("BOARDROOM_TB_HOST", "127.0.0.1")
_TB_PORT = int(os.environ.get("BOARDROOM_TB_PORT", "6006"))

_proc: Optional[subprocess.Popen[bytes]] = None
_proc_lock = threading.Lock()


def upstream_url() -> str:
    return os.environ.get("BOARDROOM_TB_UPSTREAM") or f"http://{_TB_HOST}:{_TB_PORT}"


def is_external() -> bool:
    """When BOARDROOM_TB_UPSTREAM is set, we don't manage the TB process ourselves."""
    return bool(os.environ.get("BOARDROOM_TB_UPSTREAM"))


def _logdir() -> Path:
    p = META_DIR / "tensorboard"
    p.mkdir(parents=True, exist_ok=True)
    return p


def start_in_background() -> None:
    """Start TensorBoard if not externally managed and not already running."""
    if is_external():
        return
    global _proc
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return
        binary = shutil.which("tensorboard") or shutil.which("tb")
        if not binary:
            # SB3 already requires `tensorboard` in requirements; if missing, just skip.
            return
        # Do not pass --bind_all together with --host; recent TensorBoard exits with an error.
        cmd = [
            binary,
            "--logdir",
            str(_logdir()),
            "--host",
            _TB_HOST,
            "--port",
            str(_TB_PORT),
            "--reload_interval",
            "5",
            "--path_prefix",
            _TB_PATH_PREFIX,
        ]
        try:
            _proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            _proc = None


def stop() -> None:
    global _proc
    with _proc_lock:
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None


_TB_METHODS = ["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"]


@router.api_route("/tb", methods=_TB_METHODS)
@router.api_route("/tb/", methods=_TB_METHODS)
async def tb_root(request: Request) -> Response:
    return await _proxy(request, "")


@router.api_route("/tb/{path:path}", methods=_TB_METHODS)
async def tb_proxy(request: Request, path: str) -> Response:
    return await _proxy(request, path)


async def _proxy(request: Request, path: str) -> Response:
    base = upstream_url().rstrip("/")
    prefix = _TB_PATH_PREFIX if _TB_PATH_PREFIX.startswith("/") else f"/{_TB_PATH_PREFIX}"
    suffix = (path or "").lstrip("/")
    if suffix:
        target = f"{base}{prefix}/{suffix}"
    else:
        target = f"{base}{prefix}/"
    query = request.url.query
    if query:
        target = f"{target}?{query}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            upstream = await client.request(
                request.method,
                target,
                content=await request.body(),
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            )
    except httpx.HTTPError as exc:
        return Response(
            content=f"TensorBoard upstream unavailable: {exc}",
            status_code=502,
            media_type="text/plain",
        )
    excluded = {"transfer-encoding", "content-encoding", "content-length", "connection"}
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


def _ensure_tb_module() -> None:
    """If `tensorboard` isn't on PATH but is importable, expose `python -m tensorboard.main`."""
    binary = shutil.which("tensorboard")
    if binary:
        return
    try:
        import importlib

        importlib.import_module("tensorboard.main")
    except Exception:  # noqa: BLE001
        return
    # Provide a thin shell shim by invoking `python -m tensorboard.main`.

    def _wrapped_start() -> None:
        global _proc
        with _proc_lock:
            if _proc is not None and _proc.poll() is None:
                return
            cmd = [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(_logdir()),
                "--host",
                _TB_HOST,
                "--port",
                str(_TB_PORT),
                "--reload_interval",
                "5",
                "--path_prefix",
                _TB_PATH_PREFIX,
            ]
            _proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    global start_in_background
    start_in_background = _wrapped_start  # type: ignore[assignment]


_ensure_tb_module()
