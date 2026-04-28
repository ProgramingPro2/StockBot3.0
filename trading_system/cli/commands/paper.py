"""`boardroom paper` - paper trading loop."""
from __future__ import annotations

from typing import Optional

import typer

from ...config import CYCLE_INTERVAL_MIN
from ...utils.logger import set_debug
from ..formatting import banner, kv_panel, set_quiet
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command("paper", help="Run the paper trading loop. No real orders are placed.")
    def paper(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        interval: int = typer.Option(CYCLE_INTERVAL_MIN, "--interval", help="Minutes between decisions."),
        autodiscover: bool = typer.Option(False, "--autodiscover", help="Resolve trade list via universe autodiscovery."),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        from ..run import cmd_paper

        set_debug(debug)
        set_quiet(json_output)
        if not json_output:
            banner("[bold]Boardroom: paper[/bold]", "Simulated trading loop", style="yellow")
            kv_panel(
                "Run config",
                {
                    "Tickers": tickers or "TRADE_TICKERS",
                    "Interval (min)": interval,
                    "Autodiscover": str(autodiscover),
                },
            )
        cmd_paper(
            make_args(
                tickers=tickers,
                interval=interval,
                autodiscover=autodiscover,
                debug=debug,
            )
        )
