"""`boardroom test`, `benchmark-device`, and other utility commands."""
from __future__ import annotations

from typing import Optional

import typer

from ...utils.logger import set_debug
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command("test", help="Run minimal verification across modes (sentiment, backtest, paper, train, tune, live).")
    def test_cmd(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        modes: Optional[str] = typer.Option(
            None,
            "--modes",
            help="Comma-separated modes: sentiment,backtest,paper,train,tune,live (default: all).",
        ),
        debug: bool = typer.Option(False, "--debug"),
    ) -> None:
        from ..run import cmd_test

        set_debug(debug)
        cmd_test(make_args(tickers=tickers, modes=modes, debug=debug))

    @app.command("benchmark-device", help="Compare PPO training speed: CPU vs GPU.")
    def benchmark(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        timesteps: int = typer.Option(20_000, "--timesteps"),
        use_historical: bool = typer.Option(False, "--use-historical"),
        use_tuned: bool = typer.Option(False, "--use-tuned", help="Use hyperparameters from ppo_tune_best.json"),
    ) -> None:
        from ..run import cmd_benchmark_device

        cmd_benchmark_device(
            make_args(
                tickers=tickers,
                timesteps=timesteps,
                use_historical=use_historical,
                use_tuned=use_tuned,
            )
        )
