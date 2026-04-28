"""`boardroom tune` - Optuna hyperparameter search."""
from __future__ import annotations

from typing import Optional

import typer

from ...utils.logger import set_debug
from ..formatting import banner, kv_panel, set_quiet
from ._common import make_args


def register(app: typer.Typer) -> None:
    tune_app = typer.Typer(
        help="Optuna PPO hyperparameter tuning.",
        no_args_is_help=False,
    )
    app.add_typer(tune_app, name="tune")

    @tune_app.callback(invoke_without_command=True)
    def tune_default(
        ctx: typer.Context,
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        n_trials: Optional[int] = typer.Option(None, "--n-trials"),
        timesteps: Optional[int] = typer.Option(None, "--timesteps"),
        timeout: int = typer.Option(0, "--timeout", help="Max seconds (0 = no limit)."),
        new: bool = typer.Option(False, "--new", help="Start a new study."),
        study_name: Optional[str] = typer.Option(None, "--study-name"),
        n_envs: Optional[int] = typer.Option(None, "--n-envs"),
        use_historical: bool = typer.Option(False, "--use-historical"),
        historical_ratio: Optional[float] = typer.Option(None, "--historical-ratio"),
        interactive: bool = typer.Option(False, "--interactive", help="Prompt for options."),
        single_thread: bool = typer.Option(False, "--single-thread"),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        if ctx.invoked_subcommand is not None:
            return
        from ..run import cmd_tune

        set_debug(debug)
        set_quiet(json_output)
        if not json_output:
            banner("[bold]Boardroom: tune[/bold]", "Hyperparameter search (Optuna)", style="magenta")
            kv_panel(
                "Run config",
                {
                    "Tickers": tickers or "TRADE_TICKERS",
                    "Trials": n_trials or 50,
                    "Timesteps/trial": timesteps or 100_000,
                    "Timeout (s)": timeout or 0,
                    "Study name": study_name or "ppo_tune",
                    "New study": str(new),
                    "Use historical": str(use_historical),
                },
            )
        args = make_args(
            tickers=tickers,
            n_trials=n_trials,
            timesteps=timesteps,
            timeout=timeout,
            new=new,
            study_name=study_name,
            n_envs=n_envs,
            use_historical=use_historical,
            historical_ratio=historical_ratio,
            interactive=interactive,
            single_thread=single_thread,
            debug=debug,
            tune_action="run",
        )
        cmd_tune(args)

    @tune_app.command("run", help="Run tuning (same as default).")
    def tune_run(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        n_trials: Optional[int] = typer.Option(None, "--n-trials"),
        timesteps: Optional[int] = typer.Option(None, "--timesteps"),
        timeout: int = typer.Option(0, "--timeout"),
        new: bool = typer.Option(False, "--new"),
        study_name: Optional[str] = typer.Option(None, "--study-name"),
        n_envs: Optional[int] = typer.Option(None, "--n-envs"),
        use_historical: bool = typer.Option(False, "--use-historical"),
        historical_ratio: Optional[float] = typer.Option(None, "--historical-ratio"),
        interactive: bool = typer.Option(False, "--interactive"),
        single_thread: bool = typer.Option(False, "--single-thread"),
        debug: bool = typer.Option(False, "--debug"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        from ..run import cmd_tune

        set_debug(debug)
        set_quiet(json_output)
        args = make_args(
            tickers=tickers,
            n_trials=n_trials,
            timesteps=timesteps,
            timeout=timeout,
            new=new,
            study_name=study_name,
            n_envs=n_envs,
            use_historical=use_historical,
            historical_ratio=historical_ratio,
            interactive=interactive,
            single_thread=single_thread,
            debug=debug,
            tune_action="run",
        )
        cmd_tune(args)

    @tune_app.command("status", help="Show tuning study status and best trial.")
    def tune_status(
        study_name: Optional[str] = typer.Option(None, "--study-name"),
    ) -> None:
        from ..run import cmd_tune_status

        cmd_tune_status(make_args(study_name=study_name, tune_action="status"))
