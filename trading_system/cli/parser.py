"""CLI argument parser and command dispatch."""
import argparse

from ..config import BASE_DIR
from ..utils.logger import set_debug
from .run import (
    cmd_autodiscover,
    cmd_backtest,
    cmd_benchmark_device,
    cmd_live,
    cmd_paper,
    cmd_schedule,
    cmd_test,
    cmd_train,
    cmd_train_universe,
    cmd_tune,
    cmd_tune_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="The Boardroom - Hierarchical Multi-Agent Trading System")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_debug(p):
        p.add_argument("--debug", action="store_true", help="Enable debug logging: specialist verification, obs/action dump")

    default_schedule = BASE_DIR.parent / "config" / "schedule.yaml"
    schedule_p = sub.add_parser("schedule", help="Run scheduler daemon or install/uninstall for startup (train/backtest/live per config)")
    schedule_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML (default: config/schedule.yaml)")
    schedule_sub = schedule_p.add_subparsers(dest="schedule_action")
    schedule_p.set_defaults(func=cmd_schedule, schedule_action="run")
    run_p = schedule_sub.add_parser("run", help="Run scheduler daemon (time-window services + cron one-shots)")
    run_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML")
    status_p = schedule_sub.add_parser("status", help="Show service status (in window, running, pid, log path)")
    status_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML")
    status_p.set_defaults(schedule_action="status")
    install_p = schedule_sub.add_parser(
        "install",
        help="Install systemd user service for run on startup (Linux only). Use loginctl enable-linger $USER for start at boot without being logged in.",
    )
    install_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML")
    install_p.set_defaults(schedule_action="install")
    uninstall_p = schedule_sub.add_parser("uninstall", help="Remove systemd user service (Linux only)")
    uninstall_p.set_defaults(schedule_action="uninstall")
    run_now_p = schedule_sub.add_parser(
        "run-now",
        help="Start a service by id immediately (ignore window). E.g. weekday_live.",
    )
    run_now_p.add_argument("job_id", type=str, help="Service id (e.g. weekday_live)")
    run_now_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML")
    run_now_p.set_defaults(schedule_action="run-now")
    stop_p = schedule_sub.add_parser("stop", help="Stop a running service by id")
    stop_p.add_argument("job_id", type=str, help="Service id (e.g. weekday_live)")
    stop_p.add_argument("--config", type=str, default=str(default_schedule), help="Path to schedule YAML")
    stop_p.set_defaults(schedule_action="stop")

    train_p = sub.add_parser("train", help="Train specialists and meta-learner on TRADE_TICKERS")
    _add_debug(train_p)
    train_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    train_p.add_argument("--retrain", action="store_true", help="Force retrain specialists")
    train_p.add_argument("--timesteps", type=int, default=500_000, help="PPO timesteps")
    train_p.add_argument("--use-historical", action="store_true", help="Train PPO on real historical returns only (equivalent to --historical-ratio 1.0)")
    train_p.add_argument("--historical-ratio", type=float, default=None, help="Fraction of episodes on historical vs synthetic data. For mixed mode use 0.1-0.9. 0 = synthetic only, 1 = historical only.")
    train_p.add_argument("--use-tuned", action="store_true", help="Use hyperparameters from previous 'tune' run")
    train_p.add_argument("--n-envs", type=int, default=None, help="Parallel env count for SubprocVecEnv (default: PPO_N_ENVS)")
    train_p.set_defaults(func=cmd_train)

    benchmark_p = sub.add_parser("benchmark-device", help="Compare PPO training speed: CPU vs GPU (short run, no model saved)")
    benchmark_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: TRADE_TICKERS)")
    benchmark_p.add_argument("--timesteps", type=int, default=20_000, help="Timesteps per device (default: 20000)")
    benchmark_p.add_argument("--use-historical", action="store_true", help="Use historical returns env (matches train with --use-historical)")
    benchmark_p.set_defaults(func=cmd_benchmark_device)

    tune_p = sub.add_parser("tune", help="Run Optuna hyperparameter tuning for PPO or show status")
    _add_debug(tune_p)
    tune_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    tune_p.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials (default: 50)")
    tune_p.add_argument("--timesteps", type=int, default=None, help="PPO timesteps per trial (default: 100000)")
    tune_p.add_argument("--timeout", type=int, default=0, help="Max seconds for tuning (0 = no limit, default)")
    tune_p.add_argument("--new", action="store_true", help="Start a new study (reuse name or use --study-name)")
    tune_p.add_argument("--study-name", type=str, default=None, help="Optuna study name (default: ppo_tune; with --new, optional)")
    tune_p.add_argument("--n-envs", type=int, default=None, help="Parallel env count for SubprocVecEnv (default: PPO_N_ENVS)")
    tune_p.add_argument("--use-historical", action="store_true", help="Fix historical_ratio=1.0 (no mixed trials)")
    tune_p.add_argument("--historical-ratio", type=float, default=None, help="Fixed historical ratio (overrides trial sampling)")
    tune_p.add_argument("--interactive", action="store_true", help="Prompt for options interactively")
    tune_p.add_argument("--single-thread", action="store_true", help="Use single PyTorch thread (reproducibility)")
    tune_sub = tune_p.add_subparsers(dest="tune_action")
    tune_run_p = tune_sub.add_parser("run", help="Run tuning (default when no subcommand)")
    tune_run_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    tune_run_p.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials (default: 50)")
    tune_run_p.add_argument("--timesteps", type=int, default=None, help="PPO timesteps per trial (default: 100000)")
    tune_run_p.add_argument("--timeout", type=int, default=0, help="Max seconds for tuning (0 = no limit, default)")
    tune_run_p.add_argument("--new", action="store_true", help="Start a new study (reuse name or use --study-name)")
    tune_run_p.add_argument("--study-name", type=str, default=None, help="Optuna study name (default: ppo_tune; with --new, optional)")
    tune_run_p.add_argument("--n-envs", type=int, default=None, help="Parallel env count for SubprocVecEnv (default: PPO_N_ENVS)")
    tune_run_p.add_argument("--use-historical", action="store_true", help="Fix historical_ratio=1.0 (no mixed trials)")
    tune_run_p.add_argument("--historical-ratio", type=float, default=None, help="Fixed historical ratio (overrides trial sampling)")
    tune_run_p.add_argument("--interactive", action="store_true", help="Prompt for options interactively")
    tune_run_p.add_argument("--single-thread", action="store_true", help="Use single PyTorch thread (reproducibility)")
    tune_run_p.set_defaults(func=cmd_tune, tune_action="run")
    tune_p.set_defaults(func=cmd_tune, tune_action="run")
    tune_status_p = tune_sub.add_parser("status", help="Show tuning study status and best trial")
    tune_status_p.add_argument("--study-name", type=str, default=None, help="Show only this study (default: all)")
    tune_status_p.set_defaults(func=cmd_tune_status, tune_action="status")

    back_p = sub.add_parser("backtest", help="Backtest portfolio with trained models")
    _add_debug(back_p)
    back_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    back_p.add_argument("--model", type=str, default=None, help="PPO model file (e.g. ppo_trader_2026-02-07.zip). If omitted and TTY, prompt to choose.")
    back_p.set_defaults(func=cmd_backtest)

    paper_p = sub.add_parser("paper", help="Paper trading loop")
    _add_debug(paper_p)
    paper_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    paper_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    paper_p.add_argument("--autodiscover", action="store_true", help="Run universe autodiscovery to get trade list (top N by scorer)")
    paper_p.set_defaults(func=cmd_paper)

    live_p = sub.add_parser("live", help="Live trading with Alpaca")
    _add_debug(live_p)
    live_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    live_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    live_p.add_argument("--autodiscover", action="store_true", help="Run universe autodiscovery to get trade list (top N by scorer)")
    live_p.set_defaults(func=cmd_live)

    autodiscover_p = sub.add_parser("autodiscover", help="Run autodiscovery once: score universe, select top N, print or save list")
    _add_debug(autodiscover_p)
    autodiscover_p.add_argument("--n", type=int, default=None, help="Number of tickers to select (default: TRADE_LIST_SIZE)")
    autodiscover_p.add_argument("--save", action="store_true", help="Save list to outputs/current_trade_list.txt")
    autodiscover_p.add_argument("--quiet", action="store_true", help="Do not print tickers to stdout")
    autodiscover_p.set_defaults(func=cmd_autodiscover)

    train_universe_p = sub.add_parser("train-universe", help="Train universe ScorerModel from historical returns (top quartile = 1)")
    _add_debug(train_universe_p)
    train_universe_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers for training data (default: TRADE_TICKERS)")
    train_universe_p.add_argument("--forward-days", type=int, default=21, help="Forward return horizon for label")
    train_universe_p.add_argument("--n-estimators", type=int, default=100, help="GradientBoosting n_estimators")
    train_universe_p.add_argument("--max-depth", type=int, default=4, help="GradientBoosting max_depth")
    train_universe_p.add_argument("--learning-rate", type=float, default=0.05, help="GradientBoosting learning_rate")
    train_universe_p.set_defaults(func=cmd_train_universe)

    test_p = sub.add_parser("test", help="Run minimal verification across all modes (sentiment, backtest, paper, train, tune, live)")
    _add_debug(test_p)
    test_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    test_p.add_argument("--modes", type=str, default=None, help="Comma-separated modes: sentiment,backtest,paper,train,tune,live (default: all)")
    test_p.set_defaults(func=cmd_test)

    args = parser.parse_args()
    set_debug(getattr(args, "debug", False))
    args.func(args)
