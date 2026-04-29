"""`boardroom train` - train specialists and the PPO meta-learner."""
from __future__ import annotations

from typing import Optional

import typer

from ...config import PPO_N_ENVS, TRAIN_HISTORICAL_RATIO, TRADE_TICKERS
from ...utils.logger import set_debug
from ..formatting import banner, kv_panel, set_quiet, trade_list
from ._common import make_args


def register(app: typer.Typer) -> None:
    @app.command(
        "train",
        help="Train specialists (TFT, fundamental, regime, momentum) and the PPO meta-learner.",
    )
    def train(
        timesteps: int = typer.Option(500_000, "--timesteps", "-t", help="PPO training timesteps."),
        tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma-separated tickers (default: TRADE_TICKERS)."),
        retrain: bool = typer.Option(False, "--retrain", help="Force retrain specialists."),
        use_historical: bool = typer.Option(False, "--use-historical", help="Train on real returns only (historical_ratio=1.0)."),
        historical_ratio: Optional[float] = typer.Option(
            None,
            "--historical-ratio",
            help="Fraction of episodes on historical vs synthetic data (0.0-1.0).",
        ),
        use_tuned: bool = typer.Option(False, "--use-tuned", help="Use hyperparameters from a previous tune run."),
        tuned_study: Optional[str] = typer.Option(
            None,
            "--tuned-study",
            help="Optuna study name to load best params from (see: tune status). Omit to use models/meta_learner/ppo_tune_best.json.",
        ),
        n_envs: Optional[int] = typer.Option(None, "--n-envs", help="Parallel envs for SubprocVecEnv (default: PPO_N_ENVS)."),
        ignore_early_stopping: bool = typer.Option(
            False,
            "--ignore-early-stopping",
            help="Train for the full timestep budget; do not stop on validation plateau.",
        ),
        debug: bool = typer.Option(False, "--debug", help="Enable verbose specialist verification logging."),
        json_output: bool = typer.Option(False, "--json", help="Emit only machine-readable result and suppress Rich output."),
    ) -> None:
        from ..run import cmd_train

        set_debug(debug)
        set_quiet(json_output)

        if not json_output:
            banner(
                "[bold]Boardroom: train[/bold]",
                "Specialist training + PPO meta-learner",
                style="cyan",
            )
            ratio = (
                1.0 if use_historical else (
                    historical_ratio if historical_ratio is not None else TRAIN_HISTORICAL_RATIO
                )
            )
            kv_panel(
                "Run config",
                {
                    "Tickers": tickers or ",".join(TRADE_TICKERS),
                    "Timesteps": f"{timesteps:,}",
                    "Historical ratio": f"{ratio:.2f}",
                    "Use tuned": str(use_tuned),
                    "Tuned study": tuned_study or "(default JSON)",
                    "Retrain specialists": str(retrain),
                    "Parallel envs": str(n_envs or PPO_N_ENVS),
                    "Ignore early stopping": str(ignore_early_stopping),
                },
            )
            trade_list((tickers or ",".join(TRADE_TICKERS)).split(","), title="Trade list")

        args = make_args(
            tickers=tickers,
            retrain=retrain,
            timesteps=timesteps,
            use_historical=use_historical,
            historical_ratio=historical_ratio,
            use_tuned=use_tuned,
            tuned_study=tuned_study,
            n_envs=n_envs,
            ignore_early_stopping=ignore_early_stopping,
            debug=debug,
        )
        cmd_train(args)
