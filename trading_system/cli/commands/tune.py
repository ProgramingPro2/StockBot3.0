"""`boardroom tune` - Optuna hyperparameter search."""
from __future__ import annotations

import os
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
        n_evaluations: Optional[int] = typer.Option(
            None,
            "--n-evaluations",
            help="Validation Sharpe evals per trial (default: 3).",
        ),
        eval_patience: Optional[int] = typer.Option(
            None,
            "--eval-patience",
            help="Stop trial early after N evals without Sharpe improvement (0 = disabled).",
        ),
        no_tb: bool = typer.Option(
            False,
            "--no-tb",
            help="Disable per-trial TensorBoard logging during tuning.",
        ),
        timeout: int = typer.Option(0, "--timeout", help="Max seconds (0 = no limit)."),
        new: bool = typer.Option(False, "--new", help="Start a new study."),
        study_name: Optional[str] = typer.Option(None, "--study-name"),
        n_envs: Optional[int] = typer.Option(None, "--n-envs"),
        parallel_trials: Optional[int] = typer.Option(
            None,
            "--parallel-trials",
            help="Optuna worker processes sharing the same study (default: PPO_PARALLEL_TRIALS or 1).",
        ),
        torch_vec: bool = typer.Option(
            False,
            "--torch-vec",
            help="Use batched single-process vec env for historical-only tuning.",
        ),
        bench: bool = typer.Option(
            False,
            "--bench",
            help="Run a short Subproc vs torch-vec benchmark and exit.",
        ),
        use_historical: bool = typer.Option(False, "--use-historical"),
        historical_ratio: Optional[float] = typer.Option(None, "--historical-ratio"),
        interactive: bool = typer.Option(False, "--interactive", help="Prompt for options."),
        pick_study: bool = typer.Option(
            False,
            "--pick-study",
            help="When continuing (not --new), list studies and prompt which to resume (needs a TTY unless --study-name is set).",
        ),
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
                    "n_evaluations": str(n_evaluations if n_evaluations is not None else 3),
                    "eval_patience": str(eval_patience if eval_patience is not None else 0),
                    "no_tb": str(no_tb),
                    "Timeout (s)": timeout or 0,
                    "Study name": study_name or "ppo_tune",
                    "New study": str(new),
                    "Pick study": str(pick_study),
                    "Use historical": str(use_historical),
                    "n_envs": str(n_envs if n_envs is not None else "PPO_N_ENVS env default"),
                    "parallel_trials": str(parallel_trials if parallel_trials is not None else "env default"),
                    "torch_vec": str(torch_vec),
                    "bench": str(bench),
                },
            )
        args = make_args(
            tickers=tickers,
            n_trials=n_trials,
            timesteps=timesteps,
            n_evaluations=n_evaluations,
            eval_patience=eval_patience,
            no_tb=no_tb,
            timeout=timeout,
            new=new,
            study_name=study_name,
            n_envs=n_envs,
            parallel_trials=parallel_trials,
            torch_vec=torch_vec,
            bench=bench,
            use_historical=use_historical,
            historical_ratio=historical_ratio,
            interactive=interactive,
            pick_study=pick_study,
            single_thread=single_thread,
            debug=debug,
            json_output=json_output,
            tune_action="run",
        )
        cmd_tune(args)

    @tune_app.command("run", help="Run tuning (same as default).")
    def tune_run(
        tickers: Optional[str] = typer.Option(None, "--tickers"),
        n_trials: Optional[int] = typer.Option(None, "--n-trials"),
        timesteps: Optional[int] = typer.Option(None, "--timesteps"),
        n_evaluations: Optional[int] = typer.Option(
            None,
            "--n-evaluations",
            help="Validation Sharpe evals per trial (default: 3).",
        ),
        eval_patience: Optional[int] = typer.Option(
            None,
            "--eval-patience",
            help="Stop trial early after N evals without Sharpe improvement (0 = disabled).",
        ),
        no_tb: bool = typer.Option(
            False,
            "--no-tb",
            help="Disable per-trial TensorBoard logging during tuning.",
        ),
        timeout: int = typer.Option(0, "--timeout"),
        new: bool = typer.Option(False, "--new"),
        study_name: Optional[str] = typer.Option(None, "--study-name"),
        n_envs: Optional[int] = typer.Option(None, "--n-envs"),
        parallel_trials: Optional[int] = typer.Option(
            None,
            "--parallel-trials",
            help="Optuna worker processes sharing the same study (default: PPO_PARALLEL_TRIALS or 1).",
        ),
        torch_vec: bool = typer.Option(False, "--torch-vec"),
        bench: bool = typer.Option(False, "--bench"),
        use_historical: bool = typer.Option(False, "--use-historical"),
        historical_ratio: Optional[float] = typer.Option(None, "--historical-ratio"),
        interactive: bool = typer.Option(False, "--interactive"),
        pick_study: bool = typer.Option(
            False,
            "--pick-study",
            help="When continuing (not --new), list studies and prompt which to resume (needs a TTY unless --study-name is set).",
        ),
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
            n_evaluations=n_evaluations,
            eval_patience=eval_patience,
            no_tb=no_tb,
            timeout=timeout,
            new=new,
            study_name=study_name,
            n_envs=n_envs,
            parallel_trials=parallel_trials,
            torch_vec=torch_vec,
            bench=bench,
            use_historical=use_historical,
            historical_ratio=historical_ratio,
            interactive=interactive,
            pick_study=pick_study,
            single_thread=single_thread,
            debug=debug,
            json_output=json_output,
            tune_action="run",
        )
        cmd_tune(args)

    @tune_app.command("status", help="Show tuning study status and best trial.")
    def tune_status(
        study_name: Optional[str] = typer.Option(None, "--study-name"),
    ) -> None:
        from ..run import cmd_tune_status

        cmd_tune_status(make_args(study_name=study_name, tune_action="status"))

    @tune_app.command(
        "export",
        help="Export best params from a study to JSON (useful after interrupted runs).",
    )
    def tune_export(
        study_name: Optional[str] = typer.Option(
            None,
            "--study-name",
            help="Study to export. Lists available studies if omitted.",
        ),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        from ..run import cmd_tune_export

        cmd_tune_export(make_args(study_name=study_name, json_output=json_output))
