"""Typer-based CLI for the Boardroom trading system.

This is the modern entry point that replaces the older argparse parser.
It's wired into both ``python -m trading_system`` and the ``boardroom``
console script.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="boardroom",
    help=(
        "[bold cyan]The Boardroom[/bold cyan]: hierarchical multi-agent trading system.\n\n"
        "Specialists (narrative, quantitative, fundamental, regime, momentum, ...) feed "
        "into a PPO meta-learner that decides position sizes for a configurable list of tickers."
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _register_all() -> None:
    from .commands import (
        autodiscover as autodiscover_cmd,
        backtest as backtest_cmd,
        live as live_cmd,
        misc as misc_cmd,
        paper as paper_cmd,
        schedule as schedule_cmd,
        train as train_cmd,
        tune as tune_cmd,
    )

    train_cmd.register(app)
    tune_cmd.register(app)
    backtest_cmd.register(app)
    paper_cmd.register(app)
    live_cmd.register(app)
    autodiscover_cmd.register(app)
    schedule_cmd.register(app)
    misc_cmd.register(app)


_register_all()


def main() -> None:
    """Entry point for ``python -m trading_system`` and ``boardroom`` console script."""
    app()


__all__ = ["app", "main"]
