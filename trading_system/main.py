"""
CLI entry point for The Boardroom trading system.

The implementation lives in :mod:`trading_system.cli.app` (Typer + Rich).
We re-export internal helpers here so existing tests keep working.
"""
from __future__ import annotations

from .cli.app import main
from .cli.run import (
    _chronological_train_test_split,
    _compute_backtest_stats,
    _exposure_and_weights,
    _load_aligned_returns,
    _proxy_specialists_from_returns,
    _run_backtest,
)

__all__ = [
    "main",
    "_chronological_train_test_split",
    "_compute_backtest_stats",
    "_exposure_and_weights",
    "_load_aligned_returns",
    "_proxy_specialists_from_returns",
    "_run_backtest",
]


if __name__ == "__main__":
    main()
