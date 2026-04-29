"""Cooperative shutdown plumbing for long-running CLI commands.

Long-running subcommands (``train``, ``tune``, ``paper``, ``live``) wire into
:func:`should_stop` so the Boardroom UI's Stop button can request a clean
exit: SB3 callbacks return ``False`` after the current rollout, Optuna stops
queueing new trials, and the paper/live loops finish their current cycle.

A second SIGTERM/SIGINT after the first is treated as a hard exit so a
frustrated user can always escape with Ctrl-C twice or by hitting Kill in the
UI.
"""
from __future__ import annotations

import signal
import threading
from typing import Iterable

from .logger import get_logger

_stop_event = threading.Event()
_handlers_installed = False
_lock = threading.Lock()

_DEFAULT_SIGNALS: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)


def install_signal_handlers(signals: Iterable[int] | None = None) -> None:
    """Install handlers that flip the stop event on first signal.

    Idempotent. The second matching signal restores the default behaviour so
    the process exits immediately on the next press.
    """
    global _handlers_installed
    with _lock:
        if _handlers_installed:
            return
        targets = tuple(signals) if signals is not None else _DEFAULT_SIGNALS
        for sig in targets:
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                # ValueError: signal only works in main thread (e.g. tests)
                # OSError: signal not supported on this platform
                continue
        _handlers_installed = True


def _handle_signal(signum: int, _frame) -> None:
    log = get_logger()
    if _stop_event.is_set():
        log.warning("Received signal %d again; restoring default handler for hard exit.", signum)
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
        return
    log.info("Received signal %d; requesting graceful shutdown.", signum)
    _stop_event.set()


def request_stop() -> None:
    """Programmatically request a graceful stop (e.g. from tests)."""
    _stop_event.set()


def should_stop() -> bool:
    """True once a stop has been requested."""
    return _stop_event.is_set()


def stop_event() -> threading.Event:
    """Return the underlying event so callers can ``wait`` on it."""
    return _stop_event


def reset_for_tests() -> None:
    """Clear the stop event (test-only helper)."""
    _stop_event.clear()
