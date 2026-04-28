"""`boardroom live` - live Alpaca trading loop."""
from __future__ import annotations

from typing import Optional

import typer

from ...config import CYCLE_INTERVAL_MIN
from ...utils.logger import set_debug
from ..formatting import banner, kv_panel, set_quiet
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command("live", help="Run live trading with Alpaca. Places real orders.")
    def live(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        interval: int = typer.Option(CYCLE_INTERVAL_MIN, "--interval"),
        autodiscover: bool = typer.Option(False, "--autodiscover"),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        from ..run import cmd_live

        set_debug(debug)
        set_quiet(json_output)
        if not json_output:
            banner(
                "[bold red]Boardroom: live[/bold red]",
                "Real orders via Alpaca will be placed.",
                style="red",
            )
            kv_panel(
                "Run config",
                {
                    "Tickers": tickers or "TRADE_TICKERS",
                    "Interval (min)": interval,
                    "Autodiscover": str(autodiscover),
                },
            )
        cmd_live(
            make_args(
                tickers=tickers,
                interval=interval,
                autodiscover=autodiscover,
                debug=debug,
            )
        )
