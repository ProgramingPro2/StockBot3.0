"""
CLI command implementations for The Boardroom (train, backtest, paper, live, schedule).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..meta_learner import PPOTrader

import numpy as np

# Suppress known harmless warnings
warnings.filterwarnings(
    "ignore",
    message=".*is an instance of `nn.Module` and is already saved during checkpointing.*",
    module="lightning.pytorch.utilities.parsing",
)
warnings.filterwarnings(
    "ignore",
    message=".*You are trying to run PPO on the GPU.*",
)
import pandas as pd

from ..config import (
    BASE_DIR,
    BOARDROOM_SUMMARY_DISCORD,
    BOARDROOM_SUMMARY_ENABLED,
    CYCLE_INTERVAL_MIN,
    ENV_ALLOW_CASH_EXPOSURE,
    sentiment_refresh_every_n_bars,
    ENV_OBS_ROLLING_FEATURES,
    ENV_PDT_RESTRICTIONS,
    FRACTIONAL,
    PDT_MIN_ACCOUNT_EQUITY,
    LIVE_PPO_MODEL_PATH,
    META_LEARNER_MODEL_DIR,
    MIN_HOLD_MINUTES_AFTER_BUY,
    MIN_REENTRY_MINUTES_AFTER_SELL,
    MIN_TRADE_NOTIONAL,
    OLLAMA_INTERVAL_MIN,
    OUTPUT_DIR,
    REBALANCE_BAND_PCT,
    SLIPPAGE_BPS,
    STARTING_CAPITAL,
    TRAIN_HISTORICAL_RATIO,
    TRAIN_HISTORICAL_RATIO_OVERRIDE,
    TRANSACTION_COST_BPS,
    TRADE_LIST_SIZE,
    TRADE_TICKERS,
)
from ..utils.graceful_shutdown import should_stop, stop_event
from ..utils.logger import get_logger, is_debug, set_debug, setup_logger
from ..utils.pdt import (
    pdt_tracker_from_config,
    rebalance_multi_asset,
    session_date_from_index,
)

setup_logger()
logger = get_logger()


def _interruptible_sleep(seconds: float) -> bool:
    """Sleep up to ``seconds`` while watching the graceful-shutdown event.

    Returns True if a stop was requested and the sleep was cut short.
    """
    if seconds <= 0:
        return should_stop()
    return stop_event().wait(timeout=seconds)


def _verify_specialist_outputs(outputs: list[dict], tickers: list[str] | None = None) -> None:
    """Log specialist outputs with range checks and NaN/inf detection when debug is enabled."""
    if not is_debug():
        return
    tickers = tickers or [f"asset_{i}" for i in range(len(outputs))]
    for i, (out, t) in enumerate(zip(outputs, tickers)):
        s = out.get("sentiment", 0.0)
        q = out.get("quant_prob", 0.5)
        f = out.get("fund_score", 0.0)
        r = out.get("regime_flag", 0.5)
        mom = out.get("momentum_score", 0.0)
        mr = out.get("mean_reversion_score", 0.0)
        vol = out.get("volatility_score", 0.5)
        evt = out.get("event_risk_score", 0.0)
        rs = out.get("relative_strength_score", 0.0)
        liq = out.get("liquidity_score", 0.5)
        bad = []
        for name, val in [
            ("sentiment", s),
            ("quant_prob", q),
            ("fund_score", f),
            ("regime_flag", r),
            ("momentum_score", mom),
            ("mean_reversion_score", mr),
            ("volatility_score", vol),
            ("event_risk_score", evt),
            ("relative_strength_score", rs),
            ("liquidity_score", liq),
        ]:
            try:
                v = float(val)
                if np.isnan(v) or np.isinf(v):
                    bad.append(f"{name}=NaN/inf")
            except (TypeError, ValueError):
                bad.append(f"{name}=non-numeric")
        if not (-1 <= s <= 1):
            bad.append(f"sentiment={s}")
        if not (0 <= q <= 1):
            bad.append(f"quant_prob={q}")
        if not (-1 <= f <= 1):
            bad.append(f"fund_score={f}")
        if not (0 <= r <= 1):
            bad.append(f"regime_flag={r}")
        if not (-1 <= mom <= 1):
            bad.append(f"momentum_score={mom}")
        if not (-1 <= mr <= 1):
            bad.append(f"mean_reversion_score={mr}")
        if not (0 <= vol <= 1):
            bad.append(f"volatility_score={vol}")
        if not (0 <= evt <= 1):
            bad.append(f"event_risk_score={evt}")
        if not (-1 <= rs <= 1):
            bad.append(f"relative_strength_score={rs}")
        if not (0 <= liq <= 1):
            bad.append(f"liquidity_score={liq}")
        msg = f"DEBUG | {t} | sentiment={s:.3f} quant_prob={q:.3f} fund_score={f:.3f} regime_flag={r:.3f} momentum={mom:.3f} mean_rev={mr:.3f} vol={vol:.3f} event={evt:.3f} rs={rs:.3f} liq={liq:.3f}"
        if ENV_OBS_ROLLING_FEATURES:
            r5 = out.get("rolling_return_5d", 0.0)
            r21 = out.get("rolling_return_21d", 0.0)
            v21 = out.get("rolling_vol_21d", 0.0)
            msg += f" | rolling_5d={r5:.3f} rolling_21d={r21:.3f} vol_21d={v21:.3f}"
        if bad:
            msg += f" | ISSUES: {bad}"
        logger.info(msg)


def _sentiment_proxy_from_ohlcv(ticker: str) -> float:
    """Compute momentum-based sentiment proxy from OHLCV when real sentiment unavailable."""
    from ..data.loader import load_ohlcv
    from ..meta_learner.proxy_specialists import sentiment_proxy_from_close_prices

    df = load_ohlcv(ticker)
    if df.empty or "close" not in df.columns:
        return 0.0
    return sentiment_proxy_from_close_prices(df["close"])


def _rolling_features_from_ohlcv(close_series) -> dict:
    """Compute rolling_return_5d, rolling_return_21d, rolling_vol_21d from close prices."""
    if close_series.empty or len(close_series) < 5:
        return {"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0}
    ret = close_series.pct_change().dropna()
    if ret.empty or len(ret) < 5:
        return {"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0}
    ret_5d = float(ret.tail(5).mean()) if len(ret) >= 5 else 0.0
    ret_21d = float(ret.tail(21).mean()) if len(ret) >= 21 else float(ret.mean())
    vol_21d = float(ret.tail(21).std()) if len(ret) >= 21 else float(ret.std())
    if vol_21d < 1e-8:
        vol_21d = 0.01
    return {
        "rolling_return_5d": max(-1, min(1, ret_5d * 50)),
        "rolling_return_21d": max(-1, min(1, ret_21d * 50)),
        "rolling_vol_21d": max(-1, min(1, (vol_21d - 0.01) * 50)),
    }


def _run_specialists(ticker: str = "SPY", force_refresh_sentiment: bool = False) -> dict:
    """Run all specialists and return dict of outputs. Uses cache on network failure.
    force_refresh_sentiment: when True, call Ollama for sentiment; when False, use cached sentiment if available."""
    from ..agents import (
        get_sentiment,
        get_trend_probability,
        get_intrinsic_value_score,
        detect_regime,
        get_momentum_score,
        get_mean_reversion_score,
        get_volatility_score,
        get_event_risk_score,
        get_relative_strength_score,
        get_liquidity_score,
    )
    from ..data.loader import load_ohlcv

    sentiment, sentiment_conf, from_llm = get_sentiment(ticker, force_refresh=force_refresh_sentiment)
    if not from_llm:
        sentiment = _sentiment_proxy_from_ohlcv(ticker)
    quant_prob = get_trend_probability(ticker)
    fund_score, fund_conf = get_intrinsic_value_score(ticker)
    spy = load_ohlcv("SPY")
    regime_result = detect_regime(spy)
    regime_flag = regime_result["flag"]
    momentum_score = get_momentum_score(ticker)
    mean_reversion_score = get_mean_reversion_score(ticker)
    volatility_score = get_volatility_score(ticker)
    event_risk_score = get_event_risk_score(ticker)
    relative_strength_score = get_relative_strength_score(ticker)
    liquidity_score = get_liquidity_score(ticker)

    out = {
        "sentiment": sentiment,
        "quant_prob": quant_prob,
        "fund_score": fund_score,
        "regime_flag": regime_flag,
        "momentum_score": momentum_score,
        "mean_reversion_score": mean_reversion_score,
        "volatility_score": volatility_score,
        "event_risk_score": event_risk_score,
        "relative_strength_score": relative_strength_score,
        "liquidity_score": liquidity_score,
        "regime_name": regime_result["regime"],
    }
    if ENV_OBS_ROLLING_FEATURES:
        ticker_ohlcv = load_ohlcv(ticker)
        if not ticker_ohlcv.empty and "close" in ticker_ohlcv.columns:
            out.update(_rolling_features_from_ohlcv(ticker_ohlcv["close"]))
        else:
            out.update({"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0})
    return out


def _position_used(position_pct: float) -> float:
    """Use 50% when policy outputs ~0 so backtest/paper/live have meaningful exposure."""
    return 0.5 if position_pct < 0.01 else position_pct


def _exposure_and_weights(action: np.ndarray, allow_cash_exposure: bool) -> tuple[float, np.ndarray]:
    """
    Compute total exposure and normalized weights from raw action.
    If allow_cash_exposure: exposure = min(1, sum(action)), weights = action / sum(action).
    Else: exposure = 1.0 (100% invested), weights = action / sum(action).
    """
    action_sum = float(np.sum(action)) + 1e-8
    if allow_cash_exposure:
        exposure = min(1.0, action_sum)
    else:
        exposure = 1.0
    weights = np.array(action, dtype=np.float64) / action_sum
    return exposure, weights


def _tickers_from_args(args: argparse.Namespace) -> list[str]:
    """Resolve ticker list: --tickers override or config TRADE_TICKERS."""
    if getattr(args, "tickers", None):
        return [s.strip().upper() for s in args.tickers.split(",") if s.strip()]
    return list(TRADE_TICKERS)


def _list_meta_models() -> list[tuple[Path, float]]:
    """Return (path, mtime) for each ppo_trader*.zip in META_LEARNER_MODEL_DIR, newest first."""
    paths = list(META_LEARNER_MODEL_DIR.glob("ppo_trader*.zip"))
    return sorted(((p, p.stat().st_mtime) for p in paths), key=lambda x: -x[1])


def _resolve_backtest_model_path(path_str: str) -> Path | None:
    """Resolve --model value to an absolute Path; if not absolute, look under META_LEARNER_MODEL_DIR."""
    p = Path(path_str)
    if not p.is_absolute():
        p = META_LEARNER_MODEL_DIR / p.name
    return p if p.exists() else None


def _load_aligned_returns(
    tickers: list[str],
    lookback_years: int | None = None,
    max_days: int | None = None,
):
    """
    Load and align OHLCV returns for tickers.
    Returns (aligned_df, returns_array).
    lookback_years: years of history to load (default from config TRAIN_WINDOW_YEARS).
    max_days: if set, keep only the last max_days rows (explicit override only; default uses full loaded window).
    """
    from ..config import TRAIN_WINDOW_YEARS
    from ..data.loader import load_ohlcv

    years = lookback_years if lookback_years is not None else TRAIN_WINDOW_YEARS
    dfs: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=years)
        if df.empty or len(df) < 252:
            return None, None
        df = df[["close"]].copy()
        df.columns = [t]
        df["return_" + t] = df[t].pct_change()
        dfs.append(df)

    aligned = dfs[0]
    for df in dfs[1:]:
        aligned = aligned.join(df, how="inner")
    aligned = aligned.dropna(how="all")
    if max_days is not None:
        aligned = aligned.tail(max_days)
    if len(aligned) < 252:
        return None, None

    return_cols = ["return_" + t for t in tickers]
    returns_array = aligned[return_cols].fillna(0).values.astype(np.float32)
    return aligned, returns_array


def _chronological_train_test_split(
    aligned: pd.DataFrame,
    returns_array: np.ndarray,
    test_days: int | None = None,
):
    """
    Split aligned returns into train and test by time (no shuffle).
    Returns (train_returns, test_returns, aligned_train, aligned_test) or (None, None, None, None) if too short.
    """
    from ..config import TEST_WINDOW_DAYS

    n = len(returns_array)
    test_len = test_days if test_days is not None else TEST_WINDOW_DAYS
    if n < test_len + 252:
        return None, None, None, None
    train_len = n - test_len
    train_returns = returns_array[:train_len]
    test_returns = returns_array[train_len:]
    aligned_train = aligned.iloc[:train_len]
    aligned_test = aligned.iloc[train_len:]
    return train_returns, test_returns, aligned_train, aligned_test


def _tuned_params_payload_for_train(args: argparse.Namespace) -> dict[str, Any] | None:
    """Best tuned hyperparameters dict for training, or None."""
    if not getattr(args, "use_tuned", False):
        return None
    from ..meta_learner.tuner import load_tuned_params, load_tuned_params_from_study

    study = getattr(args, "tuned_study", None)
    if study and str(study).strip():
        return load_tuned_params_from_study(str(study).strip())
    return load_tuned_params()


def cmd_train(args: argparse.Namespace) -> None:
    """Train specialists and meta-learner on TRADE_TICKERS (5 stocks)."""
    import numpy as np

    from ..agents.fundamental import _ensure_model as ensure_fundamental
    from ..agents.momentum import _ensure_model as ensure_momentum
    from ..agents.quantitative import _ensure_model as ensure_quantitative
    from ..agents.regime import _ensure_model as ensure_regime
    from ..meta_learner import PPOTrader
    from ..meta_learner.runs_index import make_run_id, record_end, record_start

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    run_kind = os.environ.get("BOARDROOM_RUN_KIND", "train")
    if not os.environ.get("BOARDROOM_RUN_ID", "").strip():
        os.environ["BOARDROOM_RUN_ID"] = make_run_id(run_kind)
    run_boardroom_id = os.environ["BOARDROOM_RUN_ID"].strip()
    try:
        record_start(
            run_boardroom_id,
            run_kind,
            tickers=tickers,
            timesteps=getattr(args, "timesteps", None),
            extra={"phase": "specialists"},
        )
    except Exception:  # noqa: BLE001
        pass

    logger.info("Training specialists for %d tickers: %s", n_assets, tickers)
    try:
        parallel = os.environ.get("TRAIN_SPECIALISTS_PARALLEL", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        def _run_specialists_sequential() -> None:
            ensure_quantitative(tickers, retrain=args.retrain)
            ensure_fundamental(tickers)
            ensure_regime(retrain=args.retrain)
            ensure_momentum(tickers, retrain=args.retrain)

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = {}
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures[ex.submit(ensure_quantitative, tickers, args.retrain)] = "quantitative"
                futures[ex.submit(ensure_fundamental, tickers)] = "fundamental"
                futures[ex.submit(ensure_regime, args.retrain)] = "regime"
                futures[ex.submit(ensure_momentum, tickers, args.retrain)] = "momentum"
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                    except BaseException:
                        logger.exception("Specialist %s failed", name)
                        raise
        else:
            _run_specialists_sequential()
    except BaseException:
        try:
            record_end(run_boardroom_id, status="failed")
        except Exception:  # noqa: BLE001
            pass
        raise

    historical_ratio = getattr(args, "historical_ratio", None)
    if historical_ratio is not None:
        pass
    elif getattr(args, "use_historical", False):
        historical_ratio = 1.0
    else:
        historical_ratio = TRAIN_HISTORICAL_RATIO

    user_set_ratio = (getattr(args, "historical_ratio", None) is not None) or getattr(args, "use_historical", False) or TRAIN_HISTORICAL_RATIO_OVERRIDE
    if not user_set_ratio and getattr(args, "use_tuned", False):
        tuned = _tuned_params_payload_for_train(args)
        if tuned and "historical_ratio" in tuned:
            historical_ratio = tuned["historical_ratio"]
            logger.info("Using historical_ratio from tuned params: %.2f", historical_ratio)

    historical_returns = None
    validation_returns = None
    aligned_full = None
    aligned_test = None
    if historical_ratio > 0:
        aligned_full, returns_array = _load_aligned_returns(tickers)
        if returns_array is not None and aligned_full is not None:
            train_returns, _test_returns, _aligned_train, aligned_test = _chronological_train_test_split(
                aligned_full, returns_array
            )
            if train_returns is not None:
                n_train = len(train_returns)
                val_days = max(21, min(252, n_train // 5))  # validation slice: 20% or up to 252 days, min 21
                if n_train > val_days + 252:  # enough for train + validation
                    historical_returns = train_returns[: n_train - val_days]
                    validation_returns = train_returns[n_train - val_days :]
                    logger.info(
                        "Using chronological split: training on %d days; validation %d days (early stopping); test window held out",
                        len(historical_returns),
                        len(validation_returns),
                    )
                else:
                    historical_returns = train_returns
                    logger.info(
                        "Using chronological split: training on %d days; test window (%d days) held out for evaluation",
                        len(train_returns),
                        len(returns_array) - len(train_returns),
                    )
                logger.info("Test window: %d days held out for final backtest", len(returns_array) - len(train_returns))
            else:
                historical_returns = returns_array
                logger.info("Using historical returns for PPO training (%d days)", len(returns_array))
        elif returns_array is not None:
            historical_returns = returns_array
            logger.info("Using historical returns for PPO training (%d days)", len(returns_array))
        else:
            logger.warning("Could not load historical data; falling back to synthetic env")
            historical_ratio = 0.0

    logger.info("Training meta-learner (PPO)...")
    daily_aux_features = None
    if aligned_full is not None and historical_returns is not None:
        try:
            from ..data.feature_store import build_daily_aux_for_aligned

            full_aux = build_daily_aux_for_aligned(aligned_full, tickers)
            if full_aux is not None and full_aux.shape[0] >= len(historical_returns):
                daily_aux_features = full_aux[: len(historical_returns)].copy()
        except Exception as ex:
            logger.debug("Daily aux for env skipped: %s", ex)

    # When using tuned params, force fresh model so hyperparameters are applied
    load_model = not args.retrain and not getattr(args, "use_tuned", False)
    trader = PPOTrader(n_assets=n_assets, load=load_model)

    from ..meta_learner.env_helpers import create_proxy_specialist_fn

    ppo_kwargs = None
    if getattr(args, "use_tuned", False):
        from ..config import PPO_N_STEPS

        ppo_kwargs = _tuned_params_payload_for_train(args)
        if ppo_kwargs:
            ppo_kwargs = dict(ppo_kwargs)
            ppo_kwargs.pop("historical_ratio", None)
            ppo_kwargs.pop("best_value", None)
            tuned_n = ppo_kwargs.get("n_steps")
            use_full_tuned_n = os.environ.get(
                "PPO_TRAIN_USE_FULL_TUNED_N_STEPS", ""
            ).strip().lower() in ("1", "true", "yes")
            if (
                not use_full_tuned_n
                and tuned_n is not None
                and tuned_n > PPO_N_STEPS
            ):
                ppo_kwargs["n_steps"] = PPO_N_STEPS
            tuned_lr = ppo_kwargs.get("learning_rate")
            if (
                tuned_lr is not None
                and args.timesteps >= 200_000
                and float(tuned_lr) < 1e-4
            ):
                new_lr = 1e-4
                logger.warning(
                    "Tuned LR %.2e is small for %d steps; clamping to %.2e",
                    float(tuned_lr),
                    args.timesteps,
                    new_lr,
                )
                ppo_kwargs["learning_rate"] = new_lr
            ts = getattr(args, "tuned_study", None)
            if ts and str(ts).strip():
                logger.info("Using tuned hyperparameters from Optuna study %r", str(ts).strip())
            else:
                logger.info("Using tuned hyperparameters from ppo_tune_best.json")
        else:
            ts = getattr(args, "tuned_study", None)
            if ts and str(ts).strip():
                logger.warning(
                    "No tuned params for study %r (no completed trials or study missing); run tune for this study first",
                    str(ts).strip(),
                )
            else:
                logger.warning("No tuned params found; run 'tune' first")

    # Training uses proxy-from-returns only (no LLM sentiment). Matches backtest.
    specialist_fn_for_train = (
        create_proxy_specialist_fn(historical_returns, n_assets)
        if historical_returns is not None
        else None
    )
    historical_trading_dates = None
    validation_trading_dates = None
    if historical_returns is not None and aligned_full is not None:
        nh = len(historical_returns)
        historical_trading_dates = aligned_full.index[:nh]
        if validation_returns is not None:
            validation_trading_dates = aligned_full.index[nh : nh + len(validation_returns)]
    train_proxy_cache = None
    if historical_returns is not None and len(historical_returns) > 21:
        from ..meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        train_proxy_cache = precompute_proxy_feature_matrix(
            np.asarray(historical_returns, dtype=np.float32)
        )

    dated_path = trader.train(
        total_timesteps=args.timesteps,
        specialist_fn=specialist_fn_for_train,
        historical_returns=historical_returns,
        historical_ratio=historical_ratio,
        validation_returns=validation_returns,
        ppo_kwargs=ppo_kwargs,
        n_envs=getattr(args, "n_envs", None),
        daily_aux_features=daily_aux_features,
        historical_trading_dates=historical_trading_dates,
        validation_trading_dates=validation_trading_dates,
        proxy_feature_cache=train_proxy_cache,
        ignore_early_stopping=getattr(args, "ignore_early_stopping", False),
    )
    logger.info("Training complete. Models saved.")

    if args.timesteps >= 1_000_000:
        from datetime import datetime

        from ..broker.discord_notifier import send_training_complete_embed
        from ..utils.plotting import save_performance_plot, save_performance_series

        # Report out-of-sample when test window was held out; otherwise full backtest
        backtest_aligned = aligned_test if aligned_test is not None else None
        if backtest_aligned is not None:
            logger.info("Running training-complete backtest on held-out test window (out-of-sample)")
        result = _run_backtest(tickers, trader, aligned_df=backtest_aligned)
        if result is not None:
            (
                equity,
                returns,
                bh_returns,
                equity_dollars,
                rebalance_dates,
                rebalance_direction,
                rebalance_trades,
                weights_history,
            ) = result
            stats = _compute_backtest_stats(equity, returns, bh_returns, STARTING_CAPITAL)
            plot_filename = f"training_complete_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            plot_path = save_performance_plot(
                equity_dollars,
                returns,
                filename=plot_filename,
                rebalance_dates=rebalance_dates,
                rebalance_direction=rebalance_direction,
                rebalance_trades=rebalance_trades,
            )
            series_path = save_performance_series(
                equity_dollars=equity_dollars,
                bh_returns=bh_returns,
                rebalance_dates=rebalance_dates,
                rebalance_direction=rebalance_direction,
                rebalance_trades=rebalance_trades,
                tickers=tickers,
                weights_history=weights_history,
                starting_capital=float(STARTING_CAPITAL),
                filename=Path(plot_filename).with_suffix(".json").name,
                kind="train",
            )
            dated_name = dated_path.name if dated_path else None
            try:
                from .formatting import is_quiet, stats_table, write_result_path

                if not is_quiet():
                    stats_table(stats, starting_capital=STARTING_CAPITAL)
                write_result_path({
                    "kind": "train",
                    "tickers": tickers,
                    "timesteps": args.timesteps,
                    "historical_ratio": historical_ratio,
                    "model_path": str(dated_path) if dated_path else None,
                    "plot_path": str(plot_path),
                    "series_path": str(series_path),
                    "stats": stats,
                })
            except Exception as exc:  # noqa: BLE001 - best-effort UI integration
                logger.debug("Rich/result emit skipped: %s", exc)
            if send_training_complete_embed(
                args.timesteps, stats, plot_path, dated_name,
                historical_ratio=historical_ratio,
            ):
                logger.info("Sent training-complete notification to Discord")
            else:
                logger.debug("Discord webhook not configured or send failed")
        else:
            logger.warning("Skipping training-complete Discord notification: insufficient backtest data")


def cmd_benchmark_device(args: argparse.Namespace) -> None:
    """Run a short PPO burst on CPU and GPU (if available) and report timings to compare speed."""
    import torch

    from ..meta_learner.ppo_trader import benchmark_device

    tickers = _tickers_from_args(args)
    n_assets = len(tickers) if tickers else len(TRADE_TICKERS)
    if n_assets == 0:
        n_assets = 5
        tickers = list(TRADE_TICKERS)[:5]

    timesteps = getattr(args, "timesteps", 20_000)
    use_historical = getattr(args, "use_historical", False)

    logger.info("Device benchmark: PyTorch %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    historical_returns = None
    if use_historical and tickers:
        _, returns_array = _load_aligned_returns(tickers)
        if returns_array is not None:
            historical_returns = returns_array
            logger.info("Using historical returns (%d days, %d assets)", len(returns_array), returns_array.shape[1])
    if historical_returns is None:
        logger.info("Using synthetic env for benchmark (use --use-historical to match train setup)")

    ppo_kwargs = None
    if getattr(args, "use_tuned", False):
        from ..meta_learner.tuner import load_tuned_params

        tuned = load_tuned_params()
        if tuned:
            ppo_kwargs = dict(tuned)
            ppo_kwargs.pop("historical_ratio", None)
            ppo_kwargs.pop("best_value", None)
            logger.info("Benchmark using tuned PPO hyperparameters from ppo_tune_best.json")

    logger.info("Running PPO for %d timesteps on each device...", timesteps)
    results = benchmark_device(
        n_assets=n_assets,
        timesteps=timesteps,
        historical_returns=historical_returns,
        specialist_fn=None,
        ppo_kwargs=ppo_kwargs,
    )

    cpu_sec = results.get("cpu")
    cuda_sec = results.get("cuda")
    logger.info("--- Device benchmark results ---")
    logger.info("CPU:  %s sec (%.1f steps/sec)", f"{cpu_sec:.2f}" if cpu_sec is not None else "N/A", timesteps / cpu_sec if cpu_sec else 0)
    if cuda_sec is not None:
        logger.info("GPU:  %s sec (%.1f steps/sec)", f"{cuda_sec:.2f}", timesteps / cuda_sec)
        if cpu_sec is not None and cpu_sec > 0:
            ratio = cuda_sec / cpu_sec
            if ratio < 1.0:
                logger.info("GPU is %.2fx faster than CPU. Set PPO_DEVICE=cuda in .env for training.", 1.0 / ratio)
            else:
                logger.info("CPU is %.2fx faster than GPU. Keep PPO_DEVICE=cpu (default).", ratio)
    else:
        logger.info("GPU: not available (CUDA not detected)")
    logger.info("--------------------------------")


def _compute_backtest_stats(
    equity: pd.Series,
    strategy_returns: pd.Series,
    bh_returns: pd.Series,
    starting_capital: float | None = None,
) -> dict:
    """Compute backtest statistics; return dict for logging or Discord."""
    total_return_pct = (float(equity.iloc[-1]) - 1.0) * 100
    n_days = len(equity)
    years = n_days / 252.0
    cagr = (float(equity.iloc[-1]) ** (1 / max(years, 0.01)) - 1.0) * 100 if years > 0 else 0.0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd_pct = float(drawdown.min()) * 100
    vol_ann = float(strategy_returns.std() * (252 ** 0.5)) * 100
    sharpe = float(strategy_returns.mean() / (strategy_returns.std() + 1e-8) * (252 ** 0.5))
    win_rate = float((strategy_returns > 0).sum() / max(len(strategy_returns), 1)) * 100
    bh_equity = (1 + bh_returns.fillna(0)).cumprod()
    bh_return_pct = (float(bh_equity.iloc[-1]) - 1.0) * 100
    stats = {
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "max_dd_pct": max_dd_pct,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_days": n_days,
        "bh_return_pct": bh_return_pct,
        "vs_bh": total_return_pct - bh_return_pct,
    }
    if starting_capital is not None and starting_capital > 0:
        stats["final_value"] = float(equity.iloc[-1]) * starting_capital
        stats["pl_dollars"] = stats["final_value"] - starting_capital
        stats["peak_value"] = float(cummax.max()) * starting_capital
        stats["max_dd_dollars"] = float(drawdown.min()) * stats["peak_value"]
    return stats


def _backtest_stats(
    equity: pd.Series,
    strategy_returns: pd.Series,
    bh_returns: pd.Series,
    starting_capital: float | None = None,
) -> dict:
    """Log final backtest statistics (P/L, drawdown, etc.). Optionally scale to starting_capital for dollar values.

    Returns the computed stats dict so callers can render them with Rich or
    persist them via BOARDROOM_RESULT_PATH.
    """
    stats = _compute_backtest_stats(equity, strategy_returns, bh_returns, starting_capital)
    logger.info("--- Backtest final stats ---")
    if starting_capital is not None and starting_capital > 0 and "final_value" in stats:
        logger.info("Starting capital: $%.2f  Final value: $%.2f  P/L: $%+.2f", starting_capital, stats["final_value"], stats["pl_dollars"])
        logger.info("Peak value: $%.2f  Max drawdown: $%.2f (%.2f%%)", stats["peak_value"], stats["max_dd_dollars"], stats["max_dd_pct"])
    logger.info("Total return (P/L): %.2f%%", stats["total_return_pct"])
    logger.info("CAGR: %.2f%%", stats["cagr"])
    logger.info("Max drawdown: %.2f%%", stats["max_dd_pct"])
    logger.info("Volatility (ann.): %.2f%%", stats["vol_ann"])
    logger.info("Sharpe ratio: %.3f", stats["sharpe"])
    logger.info("Win rate (days): %.1f%%", stats["win_rate"])
    logger.info("Trading days: %d", stats["n_days"])
    logger.info("Buy-and-hold return: %.2f%%", stats["bh_return_pct"])
    logger.info("Strategy vs B&H: %+.2f%%", stats["vs_bh"])
    logger.info("----------------------------")
    return stats


def _proxy_specialists_from_returns(
    returns: pd.Series, start: int, window: int = 21
) -> dict:
    """Derive proxy specialist outputs from rolling returns for backtest (shared module)."""
    from ..meta_learner.proxy_specialists import proxy_specialists_from_returns_series

    return proxy_specialists_from_returns_series(returns, start, window)


def _run_backtest(
    tickers: list[str],
    trader: PPOTrader,
    aligned_df: pd.DataFrame | None = None,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    list,
    list,
    list[list[tuple[str, float]]],
    np.ndarray,
] | None:
    """
    Run backtest day-by-day with live-style threshold.
    When aligned_df is provided (e.g. test window), use it for out-of-sample evaluation.
    Otherwise load and align OHLCV internally.
    Return (equity, strategy_returns, bh_returns, equity_dollars, rebalance_dates,
    rebalance_direction, rebalance_trades, weights_history) or None if insufficient data.
    weights_history: shape (n_bars, n_assets), per-bar effective weights (zeros before warmup).
    """
    if aligned_df is not None:
        aligned = aligned_df
        if len(aligned) < 252:
            return None
    else:
        from ..config import TRAIN_WINDOW_YEARS
        from ..data.loader import load_ohlcv

        dfs: list[pd.DataFrame] = []
        for t in tickers:
            df = load_ohlcv(t, lookback_years=TRAIN_WINDOW_YEARS)
            if df.empty or len(df) < 252:
                return None
            df = df[["close"]].copy()
            df.columns = [t]
            df["return_" + t] = df[t].pct_change()
            dfs.append(df)
        aligned = dfs[0]
        for df in dfs[1:]:
            aligned = aligned.join(df, how="inner")
        aligned = aligned.dropna(how="all")
        if len(aligned) < 252:
            return None

    n_assets = len(tickers)
    return_cols = ["return_" + t for t in tickers]
    n = len(aligned)
    window = 21
    _day_minutes = 24 * 60
    min_hold_bars = 1 if MIN_HOLD_MINUTES_AFTER_BUY >= _day_minutes else 0
    min_reentry_bars = 1 if MIN_REENTRY_MINUTES_AFTER_SELL >= _day_minutes else 0
    strategy_returns = pd.Series(0.0, index=aligned.index, dtype=float)
    equity_arr = np.ones(n, dtype=np.float64)
    weights_history = np.zeros((n, n_assets), dtype=np.float32)
    rebalance_dates: list = []
    rebalance_direction: list = []
    rebalance_trades: list[list[tuple[str, float]]] = []
    effective_weights: np.ndarray | None = None
    prev_position_sum: float | None = None
    last_buy_bar: np.ndarray = np.full(n_assets, -999, dtype=np.int64)
    last_sell_bar: np.ndarray = np.full(n_assets, -999, dtype=np.int64)
    cost_bps = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
    turnover_pct = 0.0
    pdt_tracker = pdt_tracker_from_config()

    returns_mat = aligned[return_cols].fillna(0).values.astype(np.float32)
    from ..meta_learner.proxy_specialists import (
        precompute_proxy_feature_matrix,
        sentiment_proxy_from_close_prices,
        tuple_to_specialist_dict,
    )

    precomp = precompute_proxy_feature_matrix(returns_mat, window)
    ollama_every_n_bars = sentiment_refresh_every_n_bars()
    cached_sentiment = np.zeros(n_assets, dtype=np.float64)

    for t in range(window, n):
        specialist_list = [
            tuple_to_specialist_dict(tuple(float(x) for x in precomp[t, i]))
            for i in range(n_assets)
        ]
        bar_off = t - window
        if bar_off % ollama_every_n_bars == 0:
            for i, sym in enumerate(tickers):
                close_slice = aligned[sym].iloc[: t + 1]
                cached_sentiment[i] = sentiment_proxy_from_close_prices(close_slice)
        for i in range(n_assets):
            specialist_list[i]["sentiment"] = float(cached_sentiment[i])
        # Pass current portfolio weights so PPO sees same observation structure as in training
        current_holdings = (
            effective_weights.copy()
            if effective_weights is not None
            else np.zeros(n_assets, dtype=np.float64)
        )
        action = trader.predict(
            specialist_outputs_by_asset=specialist_list,
            holdings=current_holdings,
            debug=is_debug(),
        )
        action = np.array(action, dtype=np.float64)
        if not ENV_ALLOW_CASH_EXPOSURE:
            action = np.array([_position_used(float(a)) for a in action])
        exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
        target_holdings = exposure * weights  # actual fractional holdings (sum = exposure), matches PPO env semantics
        portfolio_value = equity_arr[t - 1] * STARTING_CAPITAL
        rebalance_band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
        traded = False

        sessd = session_date_from_index(aligned.index, t)
        if effective_weights is None:
            if ENV_PDT_RESTRICTIONS and pdt_tracker.should_block_new_opens(portfolio_value, sessd):
                effective_weights = np.zeros(n_assets, dtype=np.float64)
                prev_position_sum = 0.0
                traded = False
            else:
                effective_weights = np.array(target_holdings, dtype=np.float64)
                for i in range(n_assets):
                    if target_holdings[i] > 1e-6:
                        last_buy_bar[i] = t
                prev_position_sum = float(np.sum(effective_weights))
                rebalance_dates.append(aligned.index[t])
                rebalance_trades.append([(tickers[i], float(target_holdings[i] * 100)) for i in range(n_assets) if target_holdings[i] > 1e-6])
                rebalance_direction.append(1 if exposure > 0 else 0)
                turnover_pct = float(np.sum(np.abs(effective_weights)))
                traded = True
        else:
            new_effective, last_buy_bar, last_sell_bar = rebalance_multi_asset(
                effective_weights,
                target_holdings,
                portfolio_value,
                rebalance_band,
                t,
                last_buy_bar,
                last_sell_bar,
                min_hold_bars,
                min_reentry_bars,
                pdt_tracker,
                sessd,
                portfolio_value,
            )
            traded = np.any(np.abs(new_effective - effective_weights) > 1e-6)
            turnover_pct = float(np.sum(np.abs(new_effective - effective_weights)))
            if traded:
                rebalance_dates.append(aligned.index[t])
                deltas = [(tickers[i], float((new_effective[i] - effective_weights[i]) * 100)) for i in range(n_assets) if abs(new_effective[i] - effective_weights[i]) > 1e-6]
                rebalance_trades.append(deltas)
                position_sum = float(np.sum(new_effective))
                rebalance_direction.append(1 if prev_position_sum is not None and position_sum > prev_position_sum else (-1 if prev_position_sum is not None and position_sum < prev_position_sum else 0))
                prev_position_sum = position_sum
            effective_weights = new_effective

        # Portfolio return = dot(holdings, returns); holdings sum to exposure (rest is cash at 0 return)
        basket_ret = sum(effective_weights[i] * aligned[return_cols[i]].iloc[t] for i in range(n_assets))
        strategy_returns.iloc[t] = basket_ret
        equity_arr[t] = equity_arr[t - 1] * (1.0 + basket_ret)
        if traded:
            equity_arr[t] *= max(0.0, 1.0 - cost_bps * turnover_pct)

        weights_history[t, :] = effective_weights.astype(np.float32)

    strategy_returns = strategy_returns.fillna(0)
    equity = pd.Series(equity_arr, index=aligned.index)
    equity_dollars = equity * STARTING_CAPITAL
    bh_returns = aligned[return_cols].mean(axis=1)
    return (
        equity,
        strategy_returns.dropna(),
        bh_returns,
        equity_dollars,
        rebalance_dates,
        rebalance_direction,
        rebalance_trades,
        weights_history,
    )


def _prompt_resume_study_name() -> str | None:
    """List studies in Optuna storage and prompt which to resume.

    Returns ``None`` if storage has no studies (caller keeps default ``ppo_tune``).
    Otherwise returns the chosen study name (including default on empty input).
    """
    from ..meta_learner.tuner import get_tune_status

    status = get_tune_status(study_name=None)
    names = sorted(status.get("study_names", []))
    studies = status.get("studies", {})
    if not names:
        print("No existing studies in storage; will use default study name ppo_tune.")
        return None

    print("Existing studies:")
    for i, name in enumerate(names, start=1):
        info = studies.get(name, {})
        if "error" in info:
            extra = " (load error)"
        else:
            n_tr = info.get("n_trials", 0)
            bv = info.get("best_value")
            bt = info.get("best_trial")
            if bv is not None:
                extra = f"  n_trials={n_tr}  best_value={bv:.4f}  trial={bt}"
            else:
                extra = f"  n_trials={n_tr}  (no completed trials)"
        print(f"  {i}. {name}{extra}")

    default = "ppo_tune" if "ppo_tune" in names else names[0]
    print(
        f"Resume which study? Index (1-{len(names)}) or exact name "
        f"[default: {default}]: ",
        end="",
    )
    for _ in range(3):
        raw = input().strip()
        if not raw:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(names):
                return names[idx - 1]
        if raw in names:
            return raw
        print(f"Invalid. Enter 1-{len(names)} or exact study name [default: {default}]: ", end="")
    print(f"Using default after invalid input: {default}")
    return default


def _tune_interactive_prompt() -> dict:
    """Prompt for tuning options; return dict of n_trials, timesteps, timeout, new, study_name, n_envs, use_historical, single_thread."""
    from ..config import PPO_N_ENVS

    result = {}
    print("Start new study or continue existing? [c]ontinue / [n]ew (default: continue): ", end="")
    result["new"] = input().strip().lower().startswith("n")
    if result["new"]:
        print("Study name (optional; press Enter for auto-generated): ", end="")
        result["study_name"] = input().strip() or None
    else:
        result["study_name"] = _prompt_resume_study_name()
    print("Number of trials (default: 50): ", end="")
    raw = input().strip()
    result["n_trials"] = int(raw) if raw.isdigit() else 50
    print("Timesteps per trial (default: 100000): ", end="")
    raw = input().strip()
    result["timesteps"] = int(raw) if raw.isdigit() else 100_000
    print("Timeout in seconds (0 = no limit, default: 0): ", end="")
    raw = input().strip()
    result["timeout"] = int(raw) if raw.isdigit() else 0
    print("Parallel envs (default: %d): " % PPO_N_ENVS, end="")
    raw = input().strip()
    result["n_envs"] = int(raw) if raw.isdigit() else None
    print("Use historical only (no mixed trials)? y/[n] (default: n): ", end="")
    result["use_historical"] = input().strip().lower().startswith("y")
    print("Single PyTorch thread (reproducibility)? y/[n]: ", end="")
    result["single_thread"] = input().strip().lower().startswith("y")
    return result


def cmd_tune(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter tuning for PPO on historical data."""
    import time

    from ..config import PPO_N_ENVS
    from ..meta_learner.tuner import PPO_TUNE_PARAMS_PATH, run_tuning, tune_best_params_json_path

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    n_trials = getattr(args, "n_trials", None) or 50
    timesteps = getattr(args, "timesteps", None) or 100_000
    timeout = getattr(args, "timeout", 0) or 0
    start_new = getattr(args, "new", False)
    study_name_arg = getattr(args, "study_name", None)
    n_envs = getattr(args, "n_envs", None)
    use_historical = getattr(args, "use_historical", False)
    historical_ratio_arg = getattr(args, "historical_ratio", None)
    interactive = getattr(args, "interactive", False)
    single_thread = getattr(args, "single_thread", False)
    pick_study = getattr(args, "pick_study", False)
    json_output = getattr(args, "json_output", False)

    if interactive:
        prompted = _tune_interactive_prompt()
        n_trials = prompted["n_trials"]
        timesteps = prompted["timesteps"]
        timeout = prompted["timeout"] or 0
        start_new = prompted["new"]
        study_name_arg = prompted.get("study_name") or study_name_arg
        n_envs = prompted.get("n_envs") if prompted.get("n_envs") is not None else n_envs
        use_historical = prompted.get("use_historical", False)
        single_thread = prompted.get("single_thread", False)

    if pick_study and not start_new and study_name_arg is None:
        if json_output or not sys.stdin.isatty():
            logger.error(
                "--pick-study needs an interactive terminal when --study-name is not set. "
                "Use --study-name NAME (list studies: python -m trading_system tune status)."
            )
            return
        picked = _prompt_resume_study_name()
        study_name_arg = picked if picked is not None else study_name_arg

    if start_new and not study_name_arg:
        study_name_arg = f"ppo_tune_{int(time.time())}"
    study_name = study_name_arg or "ppo_tune"
    load_if_exists = not start_new
    fixed_historical_ratio = None
    if use_historical:
        fixed_historical_ratio = 1.0
    elif historical_ratio_arg is not None:
        fixed_historical_ratio = max(0.0, min(1.0, historical_ratio_arg))

    aligned, returns_array = _load_aligned_returns(tickers)
    if returns_array is None or aligned is None:
        logger.error("Could not load historical data for tuning")
        return

    # Walk-forward: tune on train window only; test window held out for final evaluation
    train_returns, _test_returns, _aligned_train, _aligned_test = _chronological_train_test_split(
        aligned, returns_array
    )
    tuning_returns = train_returns if train_returns is not None else returns_array
    if train_returns is not None:
        logger.info(
            "Using chronological split: tuning on %d days, test window held out (%d days)",
            len(tuning_returns),
            len(returns_array) - len(tuning_returns),
        )

    logger.info(
        "Running hyperparameter tuning for %d tickers (%d trials, %d timesteps/trial, study=%s, n_envs=%s)...",
        n_assets,
        n_trials,
        timesteps,
        study_name,
        n_envs if n_envs is not None else PPO_N_ENVS,
    )
    tune_trading_dates = aligned.index[: len(tuning_returns)]
    per_study_params_path = tune_best_params_json_path(study_name)
    best = run_tuning(
        returns_array=tuning_returns,
        n_assets=n_assets,
        n_trials=n_trials,
        n_timesteps=timesteps,
        timeout=timeout if timeout > 0 else None,
        n_envs=n_envs,
        study_name=study_name,
        load_if_exists=load_if_exists,
        fixed_historical_ratio=fixed_historical_ratio,
        single_thread=single_thread,
        params_path=per_study_params_path,
        historical_trading_dates=tune_trading_dates,
    )
    logger.info(
        "Tuning complete. Best params for study %r: %s (also mirrored to %s for train --use-tuned without --tuned-study).",
        study_name,
        per_study_params_path,
        PPO_TUNE_PARAMS_PATH,
    )
    try:
        from ..broker.discord_notifier import send_hpo_complete_embed
        if send_hpo_complete_embed(
            n_trials=n_trials,
            n_tickers=n_assets,
            timesteps_per_trial=timesteps,
            best_params=best,
            best_value=best.get("best_value"),
        ):
            logger.info("Sent HPO-complete notification to Discord")
    except Exception as exc:
        logger.debug("Discord webhook not configured or HPO notification failed: %s", exc)
    for k, v in best.items():
        if k != "policy_kwargs":
            logger.info("  %s: %s", k, v)
    logger.info(
        "Run e.g. train --use-historical --use-tuned --tuned-study %s (or omit --tuned-study to use %s).",
        study_name,
        PPO_TUNE_PARAMS_PATH.name,
    )
    try:
        from .formatting import write_result_path

        write_result_path({
            "kind": "tune",
            "tickers": tickers,
            "n_trials": n_trials,
            "timesteps_per_trial": timesteps,
            "study_name": study_name,
            "best_params": {k: v for k, v in best.items() if k != "policy_kwargs"},
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("Rich/result emit skipped: %s", exc)


def cmd_tune_status(args: argparse.Namespace) -> None:
    """Show tuning study status: studies in storage, best trial, best value."""
    from ..meta_learner.tuner import PPO_TUNE_PARAMS_PATH, PPO_TUNE_STUDY_PATH, get_tune_status

    study_name = getattr(args, "study_name", None)
    status = get_tune_status(study_name=study_name)
    names = status.get("study_names", [])
    studies = status.get("studies", {})
    if not names:
        logger.info("No tuning studies found. Storage: %s", PPO_TUNE_STUDY_PATH)
        return
    logger.info("Tuning storage: %s", PPO_TUNE_STUDY_PATH)
    logger.info("Best params file: %s", PPO_TUNE_PARAMS_PATH)
    for name in names:
        info = studies.get(name, {})
        if "error" in info:
            logger.info("  %s: %s", name, info["error"])
            continue
        n_trials = info.get("n_trials", 0)
        best_value = info.get("best_value")
        best_trial = info.get("best_trial")
        best_str = f" best_value=%.4f (trial %s)" % (best_value, best_trial) if best_value is not None else " (no completed trials)"
        logger.info("  %s: n_trials=%d%s", name, n_trials, best_str)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest portfolio (5 tickers) with trained models; save performance plot."""
    from datetime import datetime

    from ..meta_learner import PPOTrader
    from ..utils.plotting import save_performance_plot, save_performance_series

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    model_path: Path | None = None
    if getattr(args, "model", None):
        model_path = _resolve_backtest_model_path(args.model)
        if model_path is None:
            logger.error("Model not found: %s", args.model)
            return
    elif sys.stdin.isatty():
        models = _list_meta_models()
        if not models:
            logger.error("No PPO models found in %s. Run 'train' first.", META_LEARNER_MODEL_DIR)
            return
        print("Available models (newest first):")
        for i, (p, mtime) in enumerate(models, 1):
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print("  %d. %s  (%s)" % (i, p.name, ts))
        try:
            raw = input("Select model (1-%d or path): " % len(models)).strip()
        except EOFError:
            raw = ""
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                model_path = models[idx - 1][0]
        elif raw:
            model_path = _resolve_backtest_model_path(raw)
        if model_path is None and raw:
            logger.warning("Invalid selection; using default model.")
    else:
        default_path = META_LEARNER_MODEL_DIR / "ppo_trader.zip"
        if default_path.exists():
            logger.info("Using default model: %s", default_path.name)
        else:
            logger.warning("Default model %s not found; backtest may use no/empty policy.", default_path.name)

    trader = PPOTrader(n_assets=n_assets, model_path=model_path) if model_path else PPOTrader(n_assets=n_assets)

    logger.info("Running portfolio backtest for %s...", tickers)
    result = _run_backtest(tickers, trader)
    if result is None:
        logger.error("Insufficient data for backtest")
        return
    (
        equity,
        returns,
        bh_returns,
        equity_dollars,
        rebalance_dates,
        rebalance_direction,
        rebalance_trades,
        weights_history,
    ) = result

    stats = _backtest_stats(equity, returns, bh_returns, starting_capital=STARTING_CAPITAL)
    plot_filename = f"backtest_portfolio_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    path = save_performance_plot(
        equity_dollars,
        returns,
        filename=plot_filename,
        rebalance_dates=rebalance_dates,
        rebalance_direction=rebalance_direction,
        rebalance_trades=rebalance_trades,
    )
    logger.info("Saved plot to %s", path)

    series_path = save_performance_series(
        equity_dollars=equity_dollars,
        bh_returns=bh_returns,
        rebalance_dates=rebalance_dates,
        rebalance_direction=rebalance_direction,
        rebalance_trades=rebalance_trades,
        tickers=tickers,
        weights_history=weights_history,
        starting_capital=float(STARTING_CAPITAL),
        filename=Path(plot_filename).with_suffix(".json").name,
        kind="backtest",
    )

    # Rich stats panel + machine-readable result for the web UI / JSON consumers.
    try:
        from .formatting import is_quiet, stats_table, write_result_path

        if not is_quiet():
            stats_table(stats, starting_capital=STARTING_CAPITAL)
        write_result_path({
            "kind": "backtest",
            "tickers": tickers,
            "model": str(model_path) if model_path else None,
            "plot_path": str(path),
            "series_path": str(series_path),
            "stats": stats,
        })
    except Exception as exc:  # noqa: BLE001 - best-effort UI integration
        logger.debug("Rich/result emit skipped: %s", exc)


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: poll data, run specialists for all tickers, log decisions."""
    import time

    from ..broker import send_discord_notification
    from ..meta_learner import PPOTrader

    if getattr(args, "autodiscover", False):
        from ..universe import run_autodiscover
        tickers = run_autodiscover(n=TRADE_LIST_SIZE)
        logger.info("Autodiscover tickers: %s", tickers)
    else:
        tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return
    interval_min = getattr(args, "interval", None) or CYCLE_INTERVAL_MIN
    interval_sec = interval_min * 60

    trader = PPOTrader(n_assets=n_assets)
    logger.info("Starting paper trading for %s (interval=%dm, ollama_every=%dm) [simulated capital $%.2f]",
                tickers, interval_min, OLLAMA_INTERVAL_MIN, STARTING_CAPITAL)

    last_ollama_time = 0.0
    cached_boardroom_summary: str | None = None
    from ..data.loader import load_ohlcv as _load_ohlcv_paper

    paper_cash = float(STARTING_CAPITAL)
    paper_shares: dict[str, float] = {t: 0.0 for t in tickers}

    while True:
        if should_stop():
            logger.info("Paper trading stop requested; exiting cleanly")
            break
        try:
            do_ollama = (time.time() - last_ollama_time) >= (OLLAMA_INTERVAL_MIN * 60)
            logger.info("Cycle ollama_refresh=%s (%s)", do_ollama, "sentiment+boardroom from LLM" if do_ollama else "cached sentiment/boardroom")
            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t, force_refresh_sentiment=do_ollama))
            if is_debug():
                _verify_specialist_outputs(specialist_list, tickers)

            prices: dict[str, float] = {}
            for t in tickers:
                df = _load_ohlcv_paper(t, lookback_years=1)
                if df.empty or "close" not in df.columns:
                    prices[t] = float("nan")
                else:
                    prices[t] = float(df["close"].iloc[-1])

            def _px(tt: str) -> float:
                p = prices.get(tt, float("nan"))
                return float(p) if np.isfinite(p) else 0.0

            m2m = paper_cash + sum(paper_shares.get(t, 0.0) * _px(t) for t in tickers)
            paper_equity = m2m
            if paper_equity < 1e-6:
                paper_equity = float(STARTING_CAPITAL)
                paper_cash = paper_equity
                paper_shares = {t: 0.0 for t in tickers}
            position_values = np.array(
                [paper_shares.get(t, 0.0) * _px(t) for t in tickers],
                dtype=np.float64,
            )
            current_holdings = position_values / (paper_equity + 1e-12)
            action = trader.predict(
                specialist_outputs_by_asset=specialist_list,
                holdings=current_holdings,
                debug=is_debug(),
            )
            action = np.array(action, dtype=np.float64)
            if not ENV_ALLOW_CASH_EXPOSURE:
                action = np.array([_position_used(float(a)) for a in action])
            exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
            target_holdings = exposure * weights  # match PPO env: actual fractional holdings (sum = exposure)
            rebalance_band = max(MIN_TRADE_NOTIONAL, paper_equity * REBALANCE_BAND_PCT)

            if BOARDROOM_SUMMARY_ENABLED:
                if do_ollama:
                    last_ollama_time = time.time()
                    try:
                        from ..utils.boardroom_summary import get_boardroom_summary
                        summary = get_boardroom_summary(tickers, specialist_list, weights, exposure, cash=None)
                        if summary:
                            cached_boardroom_summary = summary
                            logger.info("Boardroom: %s", summary)
                            if BOARDROOM_SUMMARY_DISCORD:
                                from ..broker.discord_notifier import send_boardroom_summary_embed
                                send_boardroom_summary_embed(summary, tickers, exposure)
                    except Exception as exc:
                        logger.debug("Boardroom summary skipped: %s", exc)
                elif cached_boardroom_summary:
                    logger.info("Boardroom (cached): %s", cached_boardroom_summary)

            for i, t in enumerate(tickers):
                position_pct = float(target_holdings[i])
                out = specialist_list[i]
                px = _px(t)
                target_value = paper_equity * target_holdings[i]
                current_value = paper_shares.get(t, 0.0) * px
                if target_value > current_value + rebalance_band:
                    notional = target_value - current_value
                    if paper_cash >= notional and px > 1e-12:
                        paper_cash -= notional
                        paper_shares[t] = paper_shares.get(t, 0.0) + notional / px
                        logger.info(
                            "Paper BUY %s | $%.2f | Sentiment=%.2f Quant=%.2f | equity=$%.2f",
                            t, notional, out["sentiment"], out["quant_prob"], paper_equity,
                        )
                        send_discord_notification(t, "PAPER", position_pct, specialist_outputs=out)
                    else:
                        logger.info(
                            "Paper | %s HOLD (insufficient cash for BUY $%.2f) | Weight=%.1f%%",
                            t, notional, position_pct * 100,
                        )
                elif current_value > target_value + rebalance_band:
                    notional = min(current_value - target_value, current_value)
                    if px > 1e-12:
                        paper_cash += notional
                        paper_shares[t] = max(0.0, paper_shares.get(t, 0.0) - notional / px)
                    logger.info(
                        "Paper SELL %s | $%.2f | Sentiment=%.2f Quant=%.2f | equity=$%.2f",
                        t, notional, out["sentiment"], out["quant_prob"], paper_equity,
                    )
                    send_discord_notification(t, "PAPER", position_pct, specialist_outputs=out)
                else:
                    logger.info(
                        "Paper HOLD %s | Sentiment=%.2f Quant=%.2f Fund=%.2f Regime=%s | Exposure=%.1f%% Weight=%.1f%% (equity=$%.2f)",
                        t,
                        out["sentiment"],
                        out["quant_prob"],
                        out["fund_score"],
                        out["regime_name"],
                        exposure * 100,
                        position_pct * 100,
                        paper_equity,
                    )
                    send_discord_notification(t, "PAPER", position_pct, specialist_outputs=out)

            final_m2m = paper_cash + sum(paper_shares.get(t, 0.0) * _px(t) for t in tickers)
            logger.info(
                "Cycle completed (type=%s) paper_equity=$%.2f",
                "ollama_refresh" if do_ollama else "cache_only",
                final_m2m,
            )
            if _interruptible_sleep(interval_sec):
                logger.info("Paper trading stop requested; exiting cleanly")
                break
        except KeyboardInterrupt:
            logger.info("Paper trading stopped")
            break
        except Exception as e:
            logger.exception("Paper loop error: %s", e)
            if _interruptible_sleep(60):
                logger.info("Paper trading stop requested; exiting cleanly")
                break


def cmd_live(args: argparse.Namespace) -> None:
    """Live trading with Alpaca: poll data, run specialists for all tickers, place orders."""
    import time
    from datetime import date

    from ..broker import AlpacaBroker, send_discord_notification
    from ..meta_learner import PPOTrader

    if getattr(args, "autodiscover", False):
        from ..universe import run_autodiscover
        tickers = run_autodiscover(n=TRADE_LIST_SIZE)
        logger.info("Autodiscover tickers: %s", tickers)
    else:
        tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return
    interval_min = getattr(args, "interval", None) or CYCLE_INTERVAL_MIN
    interval_sec = interval_min * 60

    broker = AlpacaBroker()
    if not broker.client:
        logger.error("Alpaca client not initialized. Check API keys in config.py or .env")
        return

    model_path = Path(LIVE_PPO_MODEL_PATH) if LIVE_PPO_MODEL_PATH else None
    trader = PPOTrader(n_assets=n_assets, model_path=model_path)
    logger.info("Starting LIVE trading for %s (interval=%dm, ollama_every=%dm) [fractional=%s] model=%s",
                tickers, interval_min, OLLAMA_INTERVAL_MIN, FRACTIONAL, model_path or "ppo_trader.zip")

    try:
        from ..broker.discord_notifier import send_live_start_embed
        account = broker.get_account()
        cash = float(account.get("cash", 0))
        if send_live_start_embed(tickers, args.interval, model_path, FRACTIONAL, cash):
            logger.info("Sent live-start summary to Discord")
    except Exception as exc:
        logger.debug("Discord live-start notification skipped: %s", exc)

    bought_today: set[str] = set()
    last_trading_date: date | None = None
    last_ollama_time = 0.0
    cached_boardroom_summary: str | None = None
    pdt_tracker = pdt_tracker_from_config()

    while True:
        if should_stop():
            logger.info("Live trading stop requested; exiting cleanly")
            break
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            tz = ZoneInfo("America/New_York")
            ny_today = datetime.now(tz).date()
            if last_trading_date is not None and ny_today != last_trading_date:
                bought_today.clear()
            last_trading_date = ny_today

            account = broker.get_account()
            cash = float(account.get("cash", 0))
            logger.info("Account cash: $%.2f", cash)

            # Log Alpaca's actual positions (source of truth)
            alpaca_positions = broker.get_all_positions()
            if alpaca_positions:
                for pos in alpaca_positions:
                    logger.info(
                        "Alpaca position %s | qty=%.6g market_value=$%.2f",
                        pos["symbol"],
                        pos["qty"],
                        pos["market_value"],
                    )
            else:
                logger.debug("Alpaca positions: none")

            position_symbols = [p["symbol"] for p in alpaca_positions]

            bought_today = broker.get_symbols_bought_on_date(
                ny_today, position_symbols, tz
            )

            do_ollama = (time.time() - last_ollama_time) >= (OLLAMA_INTERVAL_MIN * 60)
            logger.info("Cycle ollama_refresh=%s (%s)", do_ollama, "sentiment+boardroom from LLM" if do_ollama else "cached sentiment/boardroom")
            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t, force_refresh_sentiment=do_ollama))
            if is_debug():
                _verify_specialist_outputs(specialist_list, tickers)

            # Size targets from total portfolio value, not cash; pass current holdings into PPO
            portfolio_value = float(account.get("portfolio_value", 0))
            pos_by_symbol = {p["symbol"]: float(p.get("market_value", 0)) for p in alpaca_positions}
            current_holdings = np.array(
                [pos_by_symbol.get(t, 0.0) / (portfolio_value + 1e-12) for t in tickers],
                dtype=np.float64,
            )
            action = trader.predict(
                specialist_outputs_by_asset=specialist_list,
                holdings=current_holdings,
                debug=is_debug(),
            )
            action = np.array(action, dtype=np.float64)
            if not ENV_ALLOW_CASH_EXPOSURE:
                action = np.array([_position_used(float(a)) for a in action])
            exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
            target_holdings = exposure * weights  # match PPO env: actual fractional holdings (sum = exposure)
            if cash <= 0:
                logger.warning("Account on margin (cash=%.2f); not placing buy orders", cash)
            # Target values are fractions of total portfolio (consistent with backtest)
            rebalance_band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)

            if BOARDROOM_SUMMARY_ENABLED:
                if do_ollama:
                    last_ollama_time = time.time()
                    try:
                        from ..utils.boardroom_summary import get_boardroom_summary
                        portfolio_value = float(account.get("portfolio_value", 0))
                        summary = get_boardroom_summary(
                            tickers,
                            specialist_list,
                            weights,
                            exposure,
                            cash=cash,
                            current_positions=alpaca_positions,
                            portfolio_value=portfolio_value,
                        )
                        if summary:
                            cached_boardroom_summary = summary
                            logger.info("Boardroom: %s", summary)
                            if BOARDROOM_SUMMARY_DISCORD:
                                from ..broker.discord_notifier import send_boardroom_summary_embed
                                send_boardroom_summary_embed(summary, tickers, exposure)
                    except Exception as exc:
                        logger.debug("Boardroom summary skipped: %s", exc)
                elif cached_boardroom_summary:
                    logger.info("Boardroom (cached): %s", cached_boardroom_summary)

            if not broker.is_market_open():
                logger.info("Market closed (Alpaca clock); skipping orders this cycle")
                logger.info("Cycle completed (type=%s)", "ollama_refresh" if do_ollama else "cache_only")
                if _interruptible_sleep(interval_sec):
                    logger.info("Live trading stop requested; exiting cleanly")
                    break
                continue

            # Cooldowns: do not sell if bought recently; do not buy if sold recently
            from datetime import timedelta

            now_dt = datetime.now(tz)
            since_hold = now_dt - timedelta(minutes=MIN_HOLD_MINUTES_AFTER_BUY)
            since_reentry = now_dt - timedelta(minutes=MIN_REENTRY_MINUTES_AFTER_SELL)
            bought_since = broker.get_symbols_bought_since(since_hold, tickers, tz)
            sold_since = broker.get_symbols_sold_since(since_reentry, tickers, tz)

            # Fully locked: all capital in positions bought today; no buys or sells possible
            if alpaca_positions and cash < MIN_TRADE_NOTIONAL and all(
                pos.get("symbol", "") in bought_today for pos in alpaca_positions
            ):
                logger.info(
                    "All capital in positions bought today; no buys or sells possible. Sleeping until next trading day."
                )
                logger.info("Cycle completed (type=%s)", "ollama_refresh" if do_ollama else "cache_only")
                next_midnight = now_dt.date() + timedelta(days=1)
                next_day = datetime(next_midnight.year, next_midnight.month, next_midnight.day, 0, 0, 1, tzinfo=tz)
                sec_until = max(1, int((next_day - now_dt).total_seconds()))
                if _interruptible_sleep(sec_until):
                    logger.info("Live trading stop requested; exiting cleanly")
                    break
                continue

            # For whole-share orders we need latest price per ticker
            latest_prices: dict[str, float] = {}
            if not FRACTIONAL:
                from ..data.loader import load_ohlcv
                for t in tickers:
                    df = load_ohlcv(t, lookback_years=1)
                    if df is not None and not df.empty and "close" in df.columns:
                        latest_prices[t] = float(df["close"].iloc[-1])
                    else:
                        latest_prices[t] = 0.0

            for i, t in enumerate(tickers):
                target_value = portfolio_value * float(target_holdings[i])
                current_pos = broker.get_position(t)
                current_value = float(current_pos.get("market_value", 0))
                outputs = specialist_list[i]
                position_pct = float(target_holdings[i])

                if target_value > current_value + rebalance_band:
                    if t in sold_since:
                        logger.info("HOLD %s: skip buy (re-entry cooldown)", t)
                    elif ENV_PDT_RESTRICTIONS and pdt_tracker.should_block_new_opens(
                        portfolio_value, ny_today
                    ):
                        logger.info("HOLD %s: skip buy (PDT new-open freeze)", t)
                    elif cash <= 0:
                        logger.debug("HOLD %s: skip buy (no margin)", t)
                    else:
                        notional = target_value - current_value
                        if FRACTIONAL:
                            order = broker.place_order(t, notional=notional, side="buy")
                            if order:
                                bought_today.add(t)
                                logger.info("BUY %s: $%.2f", t, notional)
                                send_discord_notification(t, "BUY", position_pct, outputs, order)
                        else:
                            price = latest_prices.get(t) or 0.0
                            if price > 0:
                                qty = max(1, int(round(notional / price)))
                                order = broker.place_order(t, qty=qty, side="buy")
                                if order:
                                    bought_today.add(t)
                                    logger.info("BUY %s: %d shares @ ~$%.2f", t, qty, price)
                                    send_discord_notification(t, "BUY", position_pct, outputs, order)
                            else:
                                logger.warning("No price for %s; skipping buy", t)
                elif current_value > target_value + rebalance_band:
                    if t in bought_since:
                        logger.info("HOLD %s: skip sell (within hold period)", t)
                        continue
                    if t in bought_today:
                        if not ENV_PDT_RESTRICTIONS:
                            logger.info("HOLD %s: skip sell (bought today)", t)
                            continue
                        if portfolio_value < PDT_MIN_ACCOUNT_EQUITY and not pdt_tracker.can_open_new_day_trade(
                            ny_today
                        ):
                            logger.info("HOLD %s: skip sell (PDT rolling day-trade limit)", t)
                            continue
                    record_pdt_on_sell = (
                        ENV_PDT_RESTRICTIONS
                        and portfolio_value < PDT_MIN_ACCOUNT_EQUITY
                        and t in bought_today
                    )
                    if current_value <= 0:
                        logger.debug("HOLD %s: no long position to reduce (current_value=%.2f)", t, current_value)
                    else:
                        notional = min(current_value - target_value, current_value)
                        if notional < 1.0:
                            logger.debug("HOLD %s: sell amount below $1", t)
                        else:
                            # If selling most of position (>= 95%), close full position to avoid rounding issues
                            sell_all = notional >= current_value * 0.95
                            if sell_all:
                                if broker.close_position(t):
                                    logger.info("SELL ALL %s (was $%.2f)", t, current_value)
                                    if record_pdt_on_sell:
                                        pdt_tracker.record_day_trade(ny_today)
                                    send_discord_notification(
                                        t, "SELL", position_pct, outputs,
                                        {"close_position": True, "notional": current_value, "status": "closed"},
                                    )
                                else:
                                    logger.warning("Failed to close position %s", t)
                            elif FRACTIONAL:
                                order = broker.place_order(t, notional=notional, side="sell")
                                if order:
                                    logger.info("SELL %s: $%.2f", t, notional)
                                    if record_pdt_on_sell:
                                        pdt_tracker.record_day_trade(ny_today)
                                    send_discord_notification(t, "SELL", position_pct, outputs, order)
                            else:
                                current_qty = float(current_pos.get("qty", 0))
                                price = latest_prices.get(t) or 0.0
                                if price > 0 and current_qty > 0:
                                    would_sell_qty = min(int(current_qty), max(1, int(round(notional / price))))
                                    if would_sell_qty >= current_qty * 0.95:
                                        if broker.close_position(t):
                                            logger.info("SELL ALL %s (was %d shares)", t, int(current_qty))
                                            if record_pdt_on_sell:
                                                pdt_tracker.record_day_trade(ny_today)
                                            send_discord_notification(
                                                t, "SELL", position_pct, outputs,
                                                {"close_position": True, "notional": current_value, "status": "closed"},
                                            )
                                    else:
                                        order = broker.place_order(t, qty=would_sell_qty, side="sell")
                                        if order:
                                            logger.info("SELL %s: %d shares @ ~$%.2f", t, would_sell_qty, price)
                                            if record_pdt_on_sell:
                                                pdt_tracker.record_day_trade(ny_today)
                                            send_discord_notification(t, "SELL", position_pct, outputs, order)
                                else:
                                    logger.warning("No position or price for %s; skipping sell", t)
                else:
                    logger.info("HOLD %s | Position=%.1f%%", t, position_pct * 100)

            logger.info("Cycle completed (type=%s)", "ollama_refresh" if do_ollama else "cache_only")
            if _interruptible_sleep(interval_sec):
                logger.info("Live trading stop requested; exiting cleanly")
                break
        except KeyboardInterrupt:
            logger.info("Live trading stopped")
            break
        except Exception as e:
            logger.exception("Live trading error: %s", e)
            if _interruptible_sleep(60):
                logger.info("Live trading stop requested; exiting cleanly")
                break


def cmd_test(args: argparse.Namespace) -> None:
    """Run minimal verification across all modes: sentiment, backtest, paper, train, tune, live."""
    from ..data.loader import load_ohlcv
    from ..meta_learner import PPOTrader

    tickers = _tickers_from_args(args)
    if not tickers:
        tickers = list(TRADE_TICKERS)
    n_assets = len(tickers)
    modes = (args.modes or "sentiment,backtest,paper,train,tune,live").lower().split(",")
    modes = [m.strip() for m in modes if m.strip()]

    logger.info("Test run for modes: %s", modes)

    if "sentiment" in modes:
        logger.info("--- Testing SENTIMENT (real + proxy) ---")
        from ..agents import get_sentiment

        s, c, from_llm = get_sentiment(tickers[0])
        logger.info("  Real sentiment %s: score=%.3f confidence=%.3f from_llm=%s", tickers[0], s, c, from_llm)
        sample_ret = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01] * 5)
        proxy = _proxy_specialists_from_returns(sample_ret, start=len(sample_ret), window=21)
        logger.info("  Proxy sentiment: %.3f (quant=%.3f fund=%.3f regime=%.3f)", proxy["sentiment"], proxy["quant_prob"], proxy["fund_score"], proxy["regime_flag"])

    if "backtest" in modes:
        logger.info("--- Testing BACKTEST (2 windows) ---")
        aligned, returns_array = _load_aligned_returns(tickers)
        if aligned is None:
            logger.warning("  Skipped: no aligned data")
        else:
            return_cols = ["return_" + t for t in tickers]
            window = 21
            trader = PPOTrader(n_assets=n_assets)
            for start in list(range(0, min(len(aligned), window * 2), window))[:2]:
                specialist_list = [_proxy_specialists_from_returns(aligned[return_cols[i]], start, window) for i in range(n_assets)]
                action = trader.predict(specialist_outputs_by_asset=specialist_list)
                logger.info("  Window %d: positions=%s", start, [round(float(a), 2) for a in action])

    if "paper" in modes:
        logger.info("--- Testing PAPER (1 cycle) ---")
        specialist_list = [_run_specialists(t) for t in tickers]
        trader = PPOTrader(n_assets=n_assets)
        action = trader.predict(specialist_outputs_by_asset=specialist_list)
        for i, t in enumerate(tickers):
            logger.info("  %s: sentiment=%.2f position=%.1f%%", t, specialist_list[i]["sentiment"], _position_used(float(action[i])) * 100)

    if "train" in modes:
        logger.info("--- Testing TRAIN (specialist_fn once) ---")
        specialist_list = [_run_specialists(t) for t in tickers]
        logger.info("  Specialist outputs: sentiment range [%.2f, %.2f]", min(o["sentiment"] for o in specialist_list), max(o["sentiment"] for o in specialist_list))

    if "tune" in modes:
        logger.info("--- Testing TUNE (1 trial, 2000 steps) ---")
        aligned, returns_array = _load_aligned_returns(tickers)
        if returns_array is None:
            logger.warning("  Skipped: no aligned data")
        else:
            from ..meta_learner.tuner import run_tuning
            run_tuning(returns_array=returns_array, n_assets=n_assets, n_trials=1, n_timesteps=2000, show_progress=False)
            logger.info("  Tune trial completed")

    if "live" in modes:
        logger.info("--- Testing LIVE (1 cycle if Alpaca configured) ---")
        try:
            from ..broker import AlpacaBroker
        except ImportError:
            AlpacaBroker = None
        if AlpacaBroker is None:
            logger.warning("  Skipped: alpaca package not installed")
        else:
            broker = AlpacaBroker()
            if not broker.client:
                logger.warning("  Skipped: Alpaca not configured")
            else:
                specialist_list = [_run_specialists(t) for t in tickers]
                trader = PPOTrader(n_assets=n_assets)
                action = trader.predict(specialist_outputs_by_asset=specialist_list)
                action = np.array(action, dtype=np.float64)
                if not ENV_ALLOW_CASH_EXPOSURE:
                    action = np.array([_position_used(float(a)) for a in action])
                exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
                for i, t in enumerate(tickers):
                    logger.info("  %s: sentiment=%.2f exposure=%.1f%% target_weight=%.1f%%", t, specialist_list[i]["sentiment"], exposure * 100, weights[i] * 100)

    logger.info("--- Test complete ---")


def cmd_autodiscover(args: argparse.Namespace) -> None:
    """Run autodiscovery once: score universe, select top N, print and optionally save list."""
    from ..universe import run_autodiscover

    n = getattr(args, "n", None) or TRADE_LIST_SIZE
    tickers = run_autodiscover(n=n)
    logger.info("Autodiscover top %d: %s", len(tickers), tickers)
    if getattr(args, "save", False):
        path = OUTPUT_DIR / "current_trade_list.txt"
        path.write_text("\n".join(tickers))
        logger.info("Saved to %s", path)
    if not getattr(args, "quiet", False):
        for t in tickers:
            print(t)


def cmd_train_universe(args: argparse.Namespace) -> None:
    """Train universe ScorerModel from historical aligned returns (top quartile forward return = 1)."""
    from ..universe.train_scorer import train_scorer_from_returns

    tickers = _tickers_from_args(args)
    if not tickers:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS")
        return
    _, returns_array = _load_aligned_returns(tickers, lookback_years=3)
    if returns_array is None:
        logger.error("Could not load aligned returns for %s", tickers)
        return
    n_assets = len(tickers)
    logger.info("Training universe scorer on %d tickers, %d days", n_assets, returns_array.shape[0])
    train_scorer_from_returns(
        returns_array,
        n_assets=n_assets,
        forward_days=getattr(args, "forward_days", 21),
        n_estimators=getattr(args, "n_estimators", 100),
        max_depth=getattr(args, "max_depth", 4),
        learning_rate=getattr(args, "learning_rate", 0.05),
    )
    logger.info("Universe scorer saved. Run 'autodiscover' or paper/live --autodiscover to use it.")


def cmd_schedule(args: argparse.Namespace) -> None:
    """Dispatch schedule run / status / run-now / stop / install / uninstall."""
    action = getattr(args, "schedule_action", None) or "run"
    config_path = Path(args.config).resolve()
    if action == "run":
        from ..scheduler import run_scheduler
        run_scheduler(config_path)
    elif action == "status":
        from ..scheduler import status as schedule_status
        info = schedule_status(config_path)
        tz = info.get("timezone") or "?"
        print(f"Timezone: {tz}")
        for sid, s in info.get("services", {}).items():
            w = s.get("window", {})
            win_str = f"{w.get('start', '?')}-{w.get('end', '?')} weekdays {w.get('weekdays', [])}"
            in_win = "yes" if s.get("in_window") else "no"
            running = s.get("running")
            pid = s.get("pid")
            run_str = f"running (pid {pid})" if (running and pid) else "not running"
            log_path = s.get("log_path", "?")
            print(f"  {sid}: window {win_str}; in window: {in_win}; {run_str}; log: {log_path}")
    elif action == "run-now":
        from ..scheduler import run_job_now
        job_id = getattr(args, "job_id", None) or ""
        if not job_id:
            logger.error("schedule run-now requires a service id (e.g. weekday_live)")
            raise SystemExit(1)
        try:
            run_job_now(config_path, job_id)
        except ValueError as e:
            logger.error("%s", e)
            raise SystemExit(1)
    elif action == "stop":
        from ..scheduler import stop_service
        job_id = getattr(args, "job_id", None) or ""
        if not job_id:
            logger.error("schedule stop requires a service id (e.g. weekday_live)")
            raise SystemExit(1)
        stop_service(job_id)
        logger.info("Stopped service %s", job_id)
    elif action == "install":
        cmd_schedule_install(args)
    elif action == "uninstall":
        cmd_schedule_uninstall(args)
    else:
        logger.error("Unknown schedule action: %s", action)
        raise SystemExit(1)


def cmd_schedule_install(args: argparse.Namespace) -> None:
    """Install systemd user service so the scheduler runs on startup (Linux only)."""
    from ..scheduler_systemd import install as systemd_install
    config_path = Path(args.config).resolve()
    project_root = BASE_DIR.parent
    systemd_install(config_path, project_root, sys.executable)


def cmd_schedule_uninstall(args: argparse.Namespace) -> None:
    """Remove systemd user service (Linux only)."""
    from ..scheduler_systemd import uninstall as systemd_uninstall
    systemd_uninstall()

