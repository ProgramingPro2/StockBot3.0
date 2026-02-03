"""
Python logging configuration for The Boardroom.
File-based logging, no console spam.
"""
import logging
import sys
from pathlib import Path

_loggers: dict[str, logging.Logger] = {}
_debug_enabled: bool = False


def setup_logger(
    name: str = "boardroom",
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a file-based logger."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "boardroom.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(fh)

    # Console handler (minimal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(ch)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "boardroom") -> logging.Logger:
    """Get existing logger or create default one."""
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


def set_debug(enabled: bool) -> None:
    """Enable or disable debug logging for feature verification."""
    global _debug_enabled
    _debug_enabled = enabled


def is_debug() -> bool:
    """Return whether debug mode is enabled."""
    return _debug_enabled
