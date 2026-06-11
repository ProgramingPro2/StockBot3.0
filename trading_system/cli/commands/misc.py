"""`boardroom test`, `benchmark-device`, and other utility commands."""
from __future__ import annotations

from typing import Optional

import typer

from ...utils.logger import set_debug
from ..formatting import write_result_path
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

    @app.command(
        "specialist-cache",
        help="Pre-generate the walk-forward specialist feature cache (Mode B / FEATURE_SOURCE=specialist training).",
    )
    def specialist_cache(
        tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma-separated; default TRADE_TICKERS."),
        start: str = typer.Option("2015-01-01", "--start", help="Start date (YYYY-MM-DD)."),
        end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD); default today."),
        refit_every: Optional[int] = typer.Option(
            None, "--refit-every", help="Walk-forward refit cadence in bars; default SPECIALIST_REFIT_EVERY_BARS."
        ),
        window: int = typer.Option(21, "--window", help="Proxy lookback window."),
        force_rebuild: bool = typer.Option(False, "--force-rebuild", help="Regenerate even if a compatible cache exists."),
        debug: bool = typer.Option(False, "--debug"),
    ) -> None:
        import time

        import pandas as pd

        from ...config import TRADE_TICKERS
        from ...meta_learner.specialist_features import get_or_build_specialist_matrix

        set_debug(debug)
        tick_list = (
            [t.strip().upper() for t in tickers.split(",") if t.strip()]
            if tickers
            else list(TRADE_TICKERS)
        )
        end_str = end or str(pd.Timestamp.today().date())
        t0 = time.time()
        matrix, dates, meta = get_or_build_specialist_matrix(
            tickers=tick_list,
            start=start,
            end=end_str,
            window=window,
            refit_every=refit_every,
            force_rebuild=force_rebuild,
        )
        elapsed = time.time() - t0
        slot_names = [
            "sentiment", "quant_prob", "fund_score", "regime_flag", "momentum_score",
            "mean_reversion", "volatility", "event_risk", "relative_strength", "liquidity",
            "ret_5d_norm", "ret_21d_norm", "vol_21d_norm",
        ]
        slot_sources = {
            slot_names[int(k)]: v
            for k, v in sorted(meta.get("slot_sources", {}).items(), key=lambda x: int(x[0]))
        }
        real_slots = sum(1 for v in slot_sources.values() if v == "specialist")
        proxy_slots = len(slot_sources) - real_slots
        result = {
            "status": "ok",
            "shape": list(matrix.shape),
            "n_days": int(len(dates)),
            "n_tickers": int(matrix.shape[1]),
            "n_slots": int(matrix.shape[2]),
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
            "tickers": tick_list,
            "elapsed_seconds": round(elapsed, 1),
            "refit_every_bars": refit_every or meta.get("refit_every"),
            "real_slots": real_slots,
            "proxy_slots": proxy_slots,
            "slot_sources": slot_sources,
            "cache_key": meta.get("cache_key"),
        }
        write_result_path(result)
        typer.echo(
            f"Specialist cache ready in {elapsed:.1f}s | shape={matrix.shape} | "
            f"{dates[0].date()} -> {dates[-1].date()} ({len(dates)} trading days) | "
            f"{real_slots} real slots / {proxy_slots} proxy"
        )
        for slot_name, source in slot_sources.items():
            typer.echo(f"  {slot_name:<20} {source}")

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
