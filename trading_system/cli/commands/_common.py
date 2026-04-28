"""Helpers shared across Typer commands."""
from __future__ import annotations

import argparse
from typing import Any


def make_args(**kwargs: Any) -> argparse.Namespace:
    """Build an argparse.Namespace from typed Typer options.

    The existing `cmd_*` functions in `trading_system.cli.run` consume an
    `argparse.Namespace`. We adapt the Typer parameters to that shape so we
    can reuse the implementation without rewriting it.
    """
    return argparse.Namespace(**kwargs)
