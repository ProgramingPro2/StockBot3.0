"""`boardroom autodiscover` and `train-universe` - universe scoring."""
from __future__ import annotations

from typing import Optional

import typer

from ...utils.logger import set_debug
from ..formatting import banner, set_quiet
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command("autodiscover", help="Score the universe and select the top N tickers.")
    def autodiscover(
        n: Optional[int] = typer.Option(None, "--n", help="Number of tickers to select (default: TRADE_LIST_SIZE)."),
        save: bool = typer.Option(False, "--save", help="Save list to outputs/current_trade_list.txt."),
        quiet: bool = typer.Option(False, "--quiet", help="Don't print tickers to stdout."),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        from ..run import cmd_autodiscover

        set_debug(debug)
        set_quiet(json_output)
        if not (quiet or json_output):
            banner("[bold]Boardroom: autodiscover[/bold]", "Universe scoring", style="blue")
        cmd_autodiscover(make_args(n=n, save=save, quiet=quiet, debug=debug))

    @app.command("train-universe", help="Train the universe ScorerModel from historical aligned returns.")
    def train_universe(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        forward_days: int = typer.Option(21, "--forward-days"),
        n_estimators: int = typer.Option(100, "--n-estimators"),
        max_depth: int = typer.Option(4, "--max-depth"),
        learning_rate: float = typer.Option(0.05, "--learning-rate"),
        debug: bool = typer.Option(False, "--debug"),
    ) -> None:
        from ..run import cmd_train_universe

        set_debug(debug)
        cmd_train_universe(
            make_args(
                tickers=tickers,
                forward_days=forward_days,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                debug=debug,
            )
        )
