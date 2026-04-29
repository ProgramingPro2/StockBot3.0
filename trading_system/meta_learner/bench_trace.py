"""Optional wall-clock tracing for training / tuning (set BENCH_TRACE=1)."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

_log = logging.getLogger("boardroom.bench")


def trace_enabled() -> bool:
    return os.environ.get("BENCH_TRACE", "").strip().lower() in ("1", "true", "yes")


def trace_event(name: str, **kwargs: Any) -> None:
    if not trace_enabled():
        return
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    _log.info("BENCH_TRACE %s %s", name, parts)


@contextmanager
def trace_block(name: str, **kwargs: Any) -> Iterator[None]:
    if not trace_enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
        _log.info("BENCH_TRACE %s elapsed_s=%.4f %s", name, elapsed, parts)
