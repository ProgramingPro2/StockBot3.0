"""Backwards-compat shim.

The CLI is now built on :mod:`trading_system.cli.app` (Typer + Rich). This
module is kept so existing imports of ``trading_system.cli.parser.main``
continue to work.
"""
from __future__ import annotations

from .app import app, main

__all__ = ["app", "main"]
