"""
Scheduler daemon: systemd-like supervisor for time-window services (live/paper) and
cron one-shot jobs (train/backtest). Keeps live process running only between 9:30-17:00
weekdays (configurable); restarts if it exits in-window; stops outside window.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import BASE_DIR, CYCLE_INTERVAL_MIN
from .utils.logger import get_logger

logger = get_logger()

PROJECT_ROOT = BASE_DIR.parent

# Service id -> (Popen, log_file handle)
_services: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
_services_lock = threading.Lock()

# Restart backoff: service id -> next allowed start time (monotonic)
_restart_after: dict[str, float] = {}
RESTART_DELAY_SEC = 10


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for schedule. Install with: pip install PyYAML")
    if not config_path.exists():
        raise FileNotFoundError(f"Schedule config not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _parse_services(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("services") or []
    if not isinstance(raw, list):
        raise ValueError("config 'services' must be a list")
    services = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(f"services[{i}] must be a dict")
        for key in ("id", "type", "window"):
            if key not in s:
                raise ValueError(f"services[{i}] missing required key: {key}")
        t = str(s["type"]).lower()
        if t not in ("live", "paper"):
            raise ValueError(f"services[{i}] type must be live|paper, got: {s['type']}")
        s = dict(s)
        s["type"] = t
        s.setdefault("options", {})
        if not isinstance(s["options"], dict):
            raise ValueError(f"services[{i}] options must be a dict")
        w = s["window"]
        if not isinstance(w, dict):
            raise ValueError(f"services[{i}] window must be a dict with start, end, weekdays")
        for k in ("start", "end", "weekdays"):
            if k not in w:
                raise ValueError(f"services[{i}] window must have {k!r}")
        services.append(s)
    return services


def _parse_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("jobs") or []
    if not isinstance(raw, list):
        raise ValueError("config 'jobs' must be a list")
    jobs = []
    for i, job in enumerate(raw):
        if not isinstance(job, dict):
            raise ValueError(f"job[{i}] must be a dict")
        for key in ("id", "type", "when"):
            if key not in job:
                raise ValueError(f"job[{i}] missing required key: {key}")
        t = str(job["type"]).lower()
        if t not in ("train", "backtest"):
            raise ValueError(f"job[{i}] type must be train|backtest, got: {job['type']}")
        job = dict(job)
        job["type"] = t
        job.setdefault("options", {})
        if not isinstance(job["options"], dict):
            raise ValueError(f"job[{i}] options must be a dict")
        jobs.append(job)
    return jobs


def _get_tz(config: dict[str, Any]) -> Any:
    tz_name = config.get("timezone") or "America/New_York"
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(str(tz_name))
    except Exception:
        pass
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        pass
    return None


def _in_window(service: dict[str, Any], tz: Any) -> bool:
    """True if current time in tz is inside service's window (weekday and time range)."""
    if tz is None:
        return False
    now = datetime.now(tz)
    w = service["window"]
    weekdays = w["weekdays"]
    if not isinstance(weekdays, list):
        weekdays = [weekdays]
    # Python weekday: Monday=0, Sunday=6. Cron often uses 1-5 = Mon-Fri (1=Mon, 5=Fri)
    # Support both 0-6 (Python) and 1-7 (cron: 1=Mon, 7=Sun). Map 1-7 -> 0-6.
    current_weekday = now.weekday()  # 0=Mon, 6=Sun
    allowed = set()
    for d in weekdays:
        d = int(d)
        if 1 <= d <= 7:
            allowed.add(d - 1)  # 1=Mon -> 0, 7=Sun -> 6
        elif 0 <= d <= 6:
            allowed.add(d)
    if current_weekday not in allowed:
        return False
    start_s = str(w["start"]).strip()
    end_s = str(w["end"]).strip()
    # Parse "09:30" / "17:00"
    try:
        start_parts = start_s.split(":")
        end_parts = end_s.split(":")
        start_minutes = int(start_parts[0]) * 60 + (int(start_parts[1]) if len(start_parts) > 1 else 0)
        end_minutes = int(end_parts[0]) * 60 + (int(end_parts[1]) if len(end_parts) > 1 else 0)
    except (ValueError, IndexError):
        return False
    now_minutes = now.hour * 60 + now.minute
    return start_minutes <= now_minutes < end_minutes


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_live_argv(options: dict[str, Any]) -> list[str]:
    opts = options or {}
    argv = [sys.executable, "-m", "trading_system", "live"]
    interval = opts.get("interval") if opts.get("interval") is not None else CYCLE_INTERVAL_MIN
    argv.extend(["--interval", str(int(interval))])
    if opts.get("autodiscover"):
        argv.append("--autodiscover")
    if opts.get("tickers"):
        argv.extend(["--tickers", str(opts["tickers"])])
    return argv


def _build_paper_argv(options: dict[str, Any]) -> list[str]:
    opts = options or {}
    argv = [sys.executable, "-m", "trading_system", "paper"]
    interval = opts.get("interval") if opts.get("interval") is not None else CYCLE_INTERVAL_MIN
    argv.extend(["--interval", str(int(interval))])
    if opts.get("autodiscover"):
        argv.append("--autodiscover")
    if opts.get("tickers"):
        argv.extend(["--tickers", str(opts["tickers"])])
    return argv


def _resolve_log_path(service: dict[str, Any]) -> Path:
    log_spec = service.get("log")
    if log_spec:
        p = Path(log_spec)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    return PROJECT_ROOT / "trading_system" / "outputs" / f"scheduler_{service['id']}.log"


def _start_service(service: dict[str, Any]) -> None:
    """Start the service subprocess; stdout/stderr to service log path."""
    job_id = service["id"]
    job_type = service["type"]
    options = service.get("options") or {}
    argv = _build_live_argv(options) if job_type == "live" else _build_paper_argv(options)
    log_path = _resolve_log_path(service)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = _now()
    logger.info("Scheduler starting service %s (%s) at %s (log: %s)", job_id, job_type, now_str, log_path)
    try:
        log_file = open(log_path, "a", encoding="utf-8")
        log_file.write(f"[{now_str}] Scheduler started {job_id} ({job_type}). Child output below.\n")
        log_file.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with _services_lock:
            old = _services.pop(job_id, None)
            if old is not None:
                try:
                    old[1].close()
                except Exception:
                    pass
            _services[job_id] = (proc, log_file)
    except Exception as e:
        logger.exception("Scheduler failed to start service %s: %s", job_id, e)


def _stop_service(service_id: str) -> None:
    """Stop the service subprocess and close its log file."""
    with _services_lock:
        pair = _services.pop(service_id, None)
    if pair is None:
        return
    proc, log_file = pair
    try:
        log_file.close()
    except Exception:
        pass
    if proc.poll() is not None:
        return
    logger.info("Scheduler stopping service %s at %s", service_id, _now())
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception as e:
        logger.warning("Scheduler error stopping service %s: %s", service_id, e)


def _tick_services(services: list[dict[str, Any]], tz: Any) -> None:
    """Check each service: in window -> ensure running; outside window -> stop."""
    now_mono = time.monotonic()
    for service in services:
        sid = service["id"]
        in_win = _in_window(service, tz)
        with _services_lock:
            pair = _services.get(sid)
            proc = pair[0] if pair else None
        if in_win:
            if proc is None:
                # Optional restart delay
                after = _restart_after.get(sid, 0)
                if now_mono >= after:
                    _start_service(service)
                    _restart_after.pop(sid, None)
                # else skip this tick
            else:
                if proc.poll() is not None:
                    code = proc.returncode
                    logger.info("Service %s exited with code %s; will restart in %ds", sid, code, RESTART_DELAY_SEC)
                    _restart_after[sid] = now_mono + RESTART_DELAY_SEC
                    with _services_lock:
                        _services.pop(sid, None)
                    try:
                        log_file = pair[1]
                        log_file.write(f"[{_now()}] Child exited with code {code}. Restart in {RESTART_DELAY_SEC}s.\n")
                        log_file.flush()
                        log_file.close()
                    except Exception:
                        pass
        else:
            if proc is not None:
                _stop_service(sid)


def _run_cron_jobs_worker(config_path: Path, config: dict[str, Any]) -> None:
    """Background thread: run APScheduler for one-shot jobs (train/backtest) only."""
    jobs = _parse_jobs(config)
    if not jobs:
        return
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler not installed; one-shot jobs (train/backtest) disabled.")
        return
    tz = _get_tz(config)
    scheduler = BlockingScheduler(timezone=tz)
    for job in jobs:
        job_id = job["id"]
        when = job["when"]
        try:
            trigger = CronTrigger.from_crontab(str(when), timezone=tz)
        except Exception as e:
            logger.error("Invalid cron for job %s when=%r: %s", job_id, when, e)
            continue
        scheduler.add_job(
            _run_one_shot,
            trigger=trigger,
            args=[job],
            id=job_id,
            name=job_id,
        )
    logger.info("Scheduler loaded %d one-shot job(s). Cron thread started.", len(jobs))
    scheduler.start()


def _build_train_argv(options: dict[str, Any]) -> list[str]:
    opts = options or {}
    argv = [sys.executable, "-m", "trading_system", "train"]
    if "timesteps" in opts:
        argv.extend(["--timesteps", str(int(opts["timesteps"]))])
    if opts.get("use_historical"):
        argv.append("--use-historical")
    if opts.get("use_tuned"):
        argv.append("--use-tuned")
    if opts.get("tuned_study"):
        argv.extend(["--tuned-study", str(opts["tuned_study"])])
    if opts.get("retrain"):
        argv.append("--retrain")
    if opts.get("historical_ratio") is not None:
        argv.extend(["--historical-ratio", str(float(opts["historical_ratio"]))])
    if opts.get("tickers"):
        argv.extend(["--tickers", str(opts["tickers"])])
    return argv


def _build_backtest_argv(options: dict[str, Any]) -> list[str]:
    opts = options or {}
    argv = [sys.executable, "-m", "trading_system", "backtest"]
    if opts.get("model"):
        argv.extend(["--model", str(opts["model"])])
    if opts.get("tickers"):
        argv.extend(["--tickers", str(opts["tickers"])])
    return argv


def _run_one_shot(job: dict[str, Any]) -> None:
    """Run train or backtest in a subprocess; log and return."""
    job_id = job["id"]
    job_type = job["type"]
    options = job.get("options") or {}
    if job_type == "train":
        argv = _build_train_argv(options)
    else:
        argv = _build_backtest_argv(options)
    logger.info("Scheduler running one-shot job %s (%s) at %s", job_id, job_type, _now())
    try:
        result = subprocess.run(argv, cwd=PROJECT_ROOT, timeout=None)
        if result.returncode != 0:
            logger.warning("Scheduler job %s exited with code %d", job_id, result.returncode)
        else:
            logger.info("Scheduler job %s completed successfully", job_id)
    except Exception as e:
        logger.exception("Scheduler job %s failed: %s", job_id, e)


def run_scheduler(config_path: Path, tick_interval_sec: int = 60) -> None:
    """Run the supervisor: time-window services in main loop, one-shot jobs in background thread."""
    config = _load_config(config_path)
    services = _parse_services(config)
    jobs = _parse_jobs(config)
    tz = _get_tz(config)
    tz_name = getattr(tz, "key", str(tz)) if tz else "None"
    logger.info("Scheduler using timezone: %s", tz_name)
    logger.info("Scheduler loaded %d service(s), %d one-shot job(s). Starting.", len(services), len(jobs))
    if services:
        for s in services:
            w = s["window"]
            logger.info("Service %s: window %s-%s weekdays %s", s["id"], w["start"], w["end"], w["weekdays"])
    if jobs:
        cron_thread = threading.Thread(target=_run_cron_jobs_worker, args=(config_path, config), daemon=True)
        cron_thread.start()
    while True:
        try:
            _tick_services(services, tz)
        except Exception as e:
            logger.exception("Scheduler tick error: %s", e)
        time.sleep(tick_interval_sec)


def status(config_path: Path) -> dict[str, Any]:
    """Return status of all services: in_window, running, pid, log_path per service."""
    config = _load_config(config_path)
    services = _parse_services(config)
    tz = _get_tz(config)
    result: dict[str, Any] = {"timezone": getattr(tz, "key", str(tz)) if tz else None, "services": {}}
    for service in services:
        sid = service["id"]
        in_win = _in_window(service, tz)
        with _services_lock:
            pair = _services.get(sid)
        proc = pair[0] if pair else None
        running = proc is not None and proc.poll() is None
        pid = proc.pid if (proc and running) else None
        log_path = _resolve_log_path(service)
        result["services"][sid] = {
            "in_window": in_win,
            "running": running,
            "pid": pid,
            "log_path": str(log_path),
            "window": service["window"],
        }
    return result


def run_job_now(config_path: Path, job_id: str) -> None:
    """Start a service by id immediately (ignore window). For testing."""
    config = _load_config(config_path)
    services = _parse_services(config)
    service = next((s for s in services if s["id"] == job_id), None)
    if service is None:
        job_ids = [s["id"] for s in services]
        raise ValueError(f"Service {job_id!r} not found. Known: {job_ids}")
    _start_service(service)
    log_path = _resolve_log_path(service)
    logger.info("Service %s started in background. Log: %s", job_id, log_path)


def stop_service(job_id: str) -> None:
    """Stop a running service by id (e.g. for schedule stop command)."""
    _stop_service(job_id)
