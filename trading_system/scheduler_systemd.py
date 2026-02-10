"""
Install or uninstall the schedule daemon as a systemd user service (Linux only).
Enables run-on-startup without root.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .utils.logger import get_logger

logger = get_logger()

SERVICE_NAME = "trading_system_schedule.service"


def _user_unit_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _unit_path() -> Path:
    return _user_unit_dir() / SERVICE_NAME


def _schedule_log_path(project_root: Path) -> Path:
    """Path for schedule daemon stdout/stderr log (created by systemd)."""
    return project_root / "trading_system" / "outputs" / "schedule.log"


def _unit_content(project_root: Path, config_path: Path, python_executable: str) -> str:
    exec_start = (
        f"{python_executable} -m trading_system schedule run --config {config_path!s}"
    )
    log_path = _schedule_log_path(project_root)
    return f"""[Unit]
Description=StockBot schedule daemon (time-window live/paper + cron train/backtest)
After=network-online.target

[Service]
Type=simple
WorkingDirectory={project_root!s}
ExecStart={exec_start}
Restart=on-failure
RestartSec=10
StandardOutput=append:{log_path!s}
StandardError=append:{log_path!s}

[Install]
WantedBy=default.target
"""


def install(
    config_path: Path,
    project_root: Path,
    python_executable: str | None = None,
) -> None:
    """Install systemd user service; enable and start it. Linux only."""
    if sys.platform != "linux":
        logger.error("schedule install is supported only on Linux (systemd).")
        raise SystemExit(1)
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    python_executable = (python_executable or sys.executable).strip()
    unit_dir = _user_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "trading_system" / "outputs").mkdir(parents=True, exist_ok=True)
    path = _unit_path()
    content = _unit_content(project_root, config_path, python_executable)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", path)
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", SERVICE_NAME],
        ["systemctl", "--user", "start", SERVICE_NAME],
    ):
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Command failed: %s (exit %d)", " ".join(cmd), result.returncode)
            raise SystemExit(result.returncode)
    log_path = _schedule_log_path(project_root)
    logger.info("Schedule service installed and started. Use 'schedule uninstall' to remove.")
    logger.info("For start at boot without being logged in, run: loginctl enable-linger $USER")
    logger.info("Logs: %s and %s", project_root / "trading_system" / "outputs" / "boardroom.log", log_path)
    logger.info("Verify: systemctl --user status trading_system_schedule.service  |  tail -f %s", log_path)


def uninstall() -> None:
    """Stop, disable, and remove the systemd user service. Linux only."""
    if sys.platform != "linux":
        logger.error("schedule uninstall is supported only on Linux (systemd).")
        raise SystemExit(1)
    path = _unit_path()
    for cmd in (
        ["systemctl", "--user", "stop", SERVICE_NAME],
        ["systemctl", "--user", "disable", SERVICE_NAME],
    ):
        subprocess.run(cmd, check=False)
    if path.exists():
        path.unlink()
        logger.info("Removed %s", path)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    logger.info("Schedule service uninstalled.")
