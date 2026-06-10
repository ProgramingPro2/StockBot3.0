"""`boardroom backtest` - portfolio backtest with trained models."""
from __future__ import annotations

from typing import Optional

import typer

from ...utils.logger import set_debug
from ..formatting import banner, kv_panel, set_quiet
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command("backtest", help="Run a portfolio backtest with the chosen PPO model.")
    def backtest(
        tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma-separated tickers."),
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help="PPO model file (e.g. ppo_trader_2026-04-27.zip). Path or filename under models/meta_learner.",
        ),
        in_sample: bool = typer.Option(
            False,
            "--in-sample",
            help="Evaluate on all available data (no chronological train/test split). Default is out-of-sample.",
        ),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json", help="Suppress Rich output; only emit machine-readable result."),
    ) -> None:
        from ..run import cmd_backtest

        set_debug(debug)
        set_quiet(json_output)
        if not json_output:
            banner("[bold]Boardroom: backtest[/bold]", "Out-of-sample portfolio simulation", style="green")
            kv_panel(
                "Run config",
                {
                    "Tickers": tickers or "TRADE_TICKERS",
                    "Model": model or "(latest / default)",
                    "Window": "in-sample (all data)" if in_sample else "out-of-sample (test split)",
                },
            )
        cmd_backtest(make_args(tickers=tickers, model=model, in_sample=in_sample, debug=debug))
