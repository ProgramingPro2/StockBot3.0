"""
CLI entry point for The Boardroom trading system.
Modes: train, backtest, paper, live (multi-asset: TRADE_TICKERS), schedule (time-window live/paper + cron train/backtest).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meta_learner import PPOTrader

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

from .config import (
    BASE_DIR,
    ENV_ALLOW_CASH_EXPOSURE,
    ENV_OBS_ROLLING_FEATURES,
    FRACTIONAL,
    LIVE_PPO_MODEL_PATH,
    META_LEARNER_MODEL_DIR,
    MIN_TRADE_NOTIONAL,
    OUTPUT_DIR,
    STARTING_CAPITAL,
    TRAIN_HISTORICAL_RATIO,
    TRAIN_HISTORICAL_RATIO_OVERRIDE,
    TRADE_LIST_SIZE,
    TRADE_TICKERS,
)
from .utils.logger import get_logger, is_debug, set_debug, setup_logger

setup_logger()
logger = get_logger()


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
    from .data.loader import load_ohlcv

    df = load_ohlcv(ticker)
    if df.empty or "close" not in df.columns or len(df) < 21:
        return 0.0
    ret = df["close"].pct_change().dropna().tail(21)
    if ret.empty:
        return 0.0
    mean_ret = float(ret.mean())
    return float(max(-0.5, min(0.5, mean_ret * 12)))


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


def _run_specialists(ticker: str = "SPY") -> dict:
    """Run all specialists and return dict of outputs. Uses cache on network failure."""
    from .agents import (
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
    from .data.loader import load_ohlcv

    sentiment, sentiment_conf, from_llm = get_sentiment(ticker)
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


def _load_aligned_returns(tickers: list[str], lookback_years: int = 3):
    """Load and align OHLCV returns for tickers. Returns (aligned_df, returns_array)."""
    from .data.loader import load_ohlcv

    dfs: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=lookback_years)
        if df.empty or len(df) < 252:
            return None, None
        df = df[["close"]].copy()
        df.columns = [t]
        df["return_" + t] = df[t].pct_change()
        dfs.append(df)

    aligned = dfs[0]
    for df in dfs[1:]:
        aligned = aligned.join(df, how="inner")
    aligned = aligned.dropna(how="all").tail(252 * 2)
    if len(aligned) < 252:
        return None, None

    return_cols = ["return_" + t for t in tickers]
    returns_array = aligned[return_cols].fillna(0).values.astype(np.float32)
    return aligned, returns_array


def cmd_train(args: argparse.Namespace) -> None:
    """Train specialists and meta-learner on TRADE_TICKERS (5 stocks)."""
    import numpy as np

    from .agents.fundamental import _ensure_model as ensure_fundamental
    from .agents.momentum import _ensure_model as ensure_momentum
    from .agents.quantitative import _ensure_model as ensure_quantitative
    from .agents.regime import _ensure_model as ensure_regime
    from .meta_learner import PPOTrader

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    logger.info("Training specialists for %d tickers: %s", n_assets, tickers)
    ensure_quantitative(tickers, retrain=args.retrain)
    ensure_fundamental(tickers)
    ensure_regime(retrain=args.retrain)
    ensure_momentum(tickers, retrain=args.retrain)

    historical_ratio = getattr(args, "historical_ratio", None)
    if historical_ratio is not None:
        pass
    elif getattr(args, "use_historical", False):
        historical_ratio = 1.0
    else:
        historical_ratio = TRAIN_HISTORICAL_RATIO

    user_set_ratio = (getattr(args, "historical_ratio", None) is not None) or getattr(args, "use_historical", False) or TRAIN_HISTORICAL_RATIO_OVERRIDE
    if not user_set_ratio and getattr(args, "use_tuned", False):
        from .meta_learner.tuner import load_tuned_params
        tuned = load_tuned_params()
        if tuned and "historical_ratio" in tuned:
            historical_ratio = tuned["historical_ratio"]
            logger.info("Using historical_ratio from tuned params: %.2f", historical_ratio)

    historical_returns = None
    if historical_ratio > 0:
        _, returns_array = _load_aligned_returns(tickers)
        if returns_array is not None:
            historical_returns = returns_array
            logger.info("Using historical returns for PPO training (%d days)", len(returns_array))
        else:
            logger.warning("Could not load historical data; falling back to synthetic env")
            historical_ratio = 0.0

    logger.info("Training meta-learner (PPO)...")
    trader = PPOTrader(n_assets=n_assets, load=not args.retrain)

    def _proxy_specialist_fn_from_historical(
        returns_array: np.ndarray, n: int, window: int = 21
    ):
        """Return a callable that yields proxy specialist outputs from historical returns (no LLM)."""
        n_days = returns_array.shape[0]

        def fn() -> list[dict]:
            start = random.randint(window, n_days) if n_days > window else window
            return [
                _proxy_specialists_from_returns(pd.Series(returns_array[:, i]), start, window)
                for i in range(n)
            ]
        return fn

    ppo_kwargs = None
    if getattr(args, "use_tuned", False):
        from .config import PPO_N_STEPS
        from .meta_learner.tuner import load_tuned_params
        ppo_kwargs = load_tuned_params()
        if ppo_kwargs:
            ppo_kwargs = dict(ppo_kwargs)
            ppo_kwargs.pop("historical_ratio", None)
            ppo_kwargs.pop("best_value", None)
            tuned_n = ppo_kwargs.get("n_steps")
            if tuned_n is not None and tuned_n > PPO_N_STEPS:
                ppo_kwargs["n_steps"] = PPO_N_STEPS
            logger.info("Using tuned hyperparameters from previous tune run")
        else:
            logger.warning("No tuned params found; run 'tune' first")

    # Training uses proxy-from-returns only (no LLM sentiment). Matches backtest.
    specialist_fn_for_train = (
        _proxy_specialist_fn_from_historical(historical_returns, n_assets)
        if historical_returns is not None
        else None
    )
    dated_path = trader.train(
        total_timesteps=args.timesteps,
        specialist_fn=specialist_fn_for_train,
        historical_returns=historical_returns,
        historical_ratio=historical_ratio,
        ppo_kwargs=ppo_kwargs,
        n_envs=getattr(args, "n_envs", None),
    )
    logger.info("Training complete. Models saved.")

    if args.timesteps >= 1_000_000:
        from datetime import datetime

        from .broker.discord_notifier import send_training_complete_embed
        from .utils.plotting import save_performance_plot

        result = _run_backtest(tickers, trader)
        if result is not None:
            equity, returns, bh_returns, equity_dollars, rebalance_dates, rebalance_direction, rebalance_trades = result
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
            dated_name = dated_path.name if dated_path else None
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

    from .meta_learner.ppo_trader import benchmark_device

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

    logger.info("Running PPO for %d timesteps on each device...", timesteps)
    results = benchmark_device(
        n_assets=n_assets,
        timesteps=timesteps,
        historical_returns=historical_returns,
        specialist_fn=None,
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
) -> None:
    """Log final backtest statistics (P/L, drawdown, etc.). Optionally scale to starting_capital for dollar values."""
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


def _proxy_specialists_from_returns(
    returns: pd.Series, start: int, window: int = 21
) -> dict:
    """
    Derive proxy specialist outputs from rolling returns for backtest (no API).
    Returns dict with sentiment, quant_prob, fund_score, regime_flag, momentum_score,
    mean_reversion_score, volatility_score, event_risk_score, and
    optional rolling_return_5d, rolling_return_21d, rolling_vol_21d.
    Uses momentum/mean-reversion as proxies for sentiment and fundamental.
    """
    from .config import ENV_OBS_ROLLING_FEATURES

    lookback = returns.iloc[max(0, start - window) : start]
    if lookback.empty or lookback.isna().all():
        out = {
            "sentiment": 0.0,
            "quant_prob": 0.5,
            "fund_score": 0.0,
            "regime_flag": 0.5,
            "momentum_score": 0.0,
            "mean_reversion_score": 0.0,
            "volatility_score": 0.5,
            "event_risk_score": 0.0,
            "relative_strength_score": 0.0,
            "liquidity_score": 0.5,
        }
        if ENV_OBS_ROLLING_FEATURES:
            out.update({"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0})
        return out

    mean_ret = float(lookback.mean())
    std_ret = float(lookback.std())
    if std_ret < 1e-8:
        std_ret = 0.01

    trend = max(-0.5, min(0.5, mean_ret * 20))
    quant_prob = 0.5 + trend
    vol_norm = max(-0.5, min(0.5, (std_ret - 0.01) * 50))
    regime_flag = 0.5 + vol_norm

    sentiment_proxy = max(-0.5, min(0.5, mean_ret * 12))
    fund_score_proxy = max(-0.5, min(0.5, -mean_ret * 15))
    momentum_proxy = max(-1, min(1, mean_ret * 15))
    mean_reversion_proxy = max(-0.5, min(0.5, -mean_ret * 20))
    volatility_proxy = max(0.0, min(1.0, (vol_norm + 0.5)))
    event_risk_proxy = 0.0

    out = {
        "sentiment": sentiment_proxy,
        "quant_prob": quant_prob,
        "fund_score": fund_score_proxy,
        "regime_flag": regime_flag,
        "momentum_score": momentum_proxy,
        "mean_reversion_score": mean_reversion_proxy,
        "volatility_score": volatility_proxy,
        "event_risk_score": event_risk_proxy,
        "relative_strength_score": 0.0,
        "liquidity_score": 0.5,
    }
    if ENV_OBS_ROLLING_FEATURES:
        ret_5d = float(lookback.tail(5).mean()) if len(lookback) >= 5 else mean_ret
        ret_21d = float(lookback.tail(21).mean()) if len(lookback) >= 21 else mean_ret
        vol_21d = float(lookback.tail(21).std()) if len(lookback) >= 21 else std_ret
        if vol_21d < 1e-8:
            vol_21d = 0.01
        out["rolling_return_5d"] = max(-1, min(1, ret_5d * 50))
        out["rolling_return_21d"] = max(-1, min(1, ret_21d * 50))
        out["rolling_vol_21d"] = max(-1, min(1, (vol_21d - 0.01) * 50))
    return out


def _run_backtest(
    tickers: list[str],
    trader: PPOTrader,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, list, list, list[list[tuple[str, float]]]] | None:
    """
    Load aligned data, run backtest day-by-day with live-style threshold.
    Only rebalance an asset when dollar gap (target vs current) exceeds MIN_TRADE_NOTIONAL.
    Return (equity, strategy_returns, bh_returns, equity_dollars, rebalance_dates,
    rebalance_direction, rebalance_trades) or None if insufficient data.
    """
    from .data.loader import load_ohlcv

    dfs: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=3)
        if df.empty or len(df) < 252:
            return None
        df = df[["close"]].copy()
        df.columns = [t]
        df["return_" + t] = df[t].pct_change()
        dfs.append(df)
    aligned = dfs[0]
    for df in dfs[1:]:
        aligned = aligned.join(df, how="inner")
    aligned = aligned.dropna(how="all").tail(252 * 2)
    if len(aligned) < 252:
        return None

    n_assets = len(tickers)
    return_cols = ["return_" + t for t in tickers]
    n = len(aligned)
    window = 21
    strategy_returns = pd.Series(0.0, index=aligned.index, dtype=float)
    equity_arr = np.ones(n, dtype=np.float64)
    rebalance_dates: list = []
    rebalance_direction: list = []
    rebalance_trades: list[list[tuple[str, float]]] = []
    effective_weights: np.ndarray | None = None
    prev_position_sum: float | None = None
    last_buy_bar: np.ndarray = np.full(n_assets, -1, dtype=np.int64)

    for t in range(window, n):
        specialist_list = [
            _proxy_specialists_from_returns(aligned[ret_col], t, window)
            for ret_col in return_cols
        ]
        action = trader.predict(specialist_outputs_by_asset=specialist_list, debug=is_debug())
        action = np.array(action, dtype=np.float64)
        if not ENV_ALLOW_CASH_EXPOSURE:
            action = np.array([_position_used(float(a)) for a in action])
        exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
        portfolio_value = equity_arr[t - 1] * STARTING_CAPITAL

        if effective_weights is None:
            effective_weights = np.array(weights, dtype=np.float64)
            for i in range(n_assets):
                if weights[i] > 1e-6:
                    last_buy_bar[i] = t
            prev_position_sum = float(np.sum(effective_weights))
            rebalance_dates.append(aligned.index[t])
            rebalance_trades.append([(tickers[i], float(weights[i] * 100)) for i in range(n_assets) if weights[i] > 1e-6])
            rebalance_direction.append(1 if exposure > 0 else 0)
        else:
            new_effective = effective_weights.copy()
            for i in range(n_assets):
                current_value_i = portfolio_value * effective_weights[i]
                target_value_i = portfolio_value * weights[i]
                is_increase = target_value_i > current_value_i + MIN_TRADE_NOTIONAL
                is_decrease = current_value_i > target_value_i + MIN_TRADE_NOTIONAL
                if is_increase:
                    new_effective[i] = weights[i]
                    last_buy_bar[i] = t
                elif is_decrease and last_buy_bar[i] != t:
                    new_effective[i] = weights[i]
            weight_sum = float(np.sum(new_effective)) + 1e-8
            exposure_sum = float(np.sum(weights)) + 1e-8
            new_effective = new_effective / weight_sum * exposure_sum
            traded = np.any(np.abs(new_effective - effective_weights) > 1e-6)
            if traded:
                rebalance_dates.append(aligned.index[t])
                deltas = [(tickers[i], float((new_effective[i] - effective_weights[i]) * 100)) for i in range(n_assets) if abs(new_effective[i] - effective_weights[i]) > 1e-6]
                rebalance_trades.append(deltas)
                position_sum = float(np.sum(new_effective))
                rebalance_direction.append(1 if prev_position_sum is not None and position_sum > prev_position_sum else (-1 if prev_position_sum is not None and position_sum < prev_position_sum else 0))
                prev_position_sum = position_sum
            effective_weights = new_effective

        eff_sum = float(np.sum(effective_weights)) + 1e-8
        basket_ret = sum(effective_weights[i] * aligned[return_cols[i]].iloc[t] for i in range(n_assets)) / eff_sum
        strategy_returns.iloc[t] = basket_ret
        equity_arr[t] = equity_arr[t - 1] * (1.0 + basket_ret)

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
    )


def cmd_tune(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter tuning for PPO on historical data."""
    from .meta_learner.tuner import PPO_TUNE_PARAMS_PATH, run_tuning

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    _, returns_array = _load_aligned_returns(tickers)
    if returns_array is None:
        logger.error("Could not load historical data for tuning")
        return

    logger.info("Running hyperparameter tuning for %d tickers (%d trials, %d timesteps/trial)...", n_assets, args.n_trials, args.timesteps)
    best = run_tuning(
        returns_array=returns_array,
        n_assets=n_assets,
        n_trials=args.n_trials,
        n_timesteps=args.timesteps,
        timeout=args.timeout if args.timeout > 0 else None,
    )
    logger.info("Tuning complete. Best params saved to %s", PPO_TUNE_PARAMS_PATH)
    try:
        from .broker.discord_notifier import send_hpo_complete_embed
        if send_hpo_complete_embed(
            n_trials=args.n_trials,
            n_tickers=n_assets,
            timesteps_per_trial=args.timesteps,
            best_params=best,
            best_value=best.get("best_value"),
        ):
            logger.info("Sent HPO-complete notification to Discord")
    except Exception as exc:
        logger.debug("Discord webhook not configured or HPO notification failed: %s", exc)
    for k, v in best.items():
        if k != "policy_kwargs":
            logger.info("  %s: %s", k, v)
    logger.info("Run 'train --use-historical --use-tuned' to train with these params.")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest portfolio (5 tickers) with trained models; save performance plot."""
    from datetime import datetime

    from .meta_learner import PPOTrader
    from .utils.plotting import save_performance_plot

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
    equity, returns, bh_returns, equity_dollars, rebalance_dates, rebalance_direction, rebalance_trades = result

    _backtest_stats(equity, returns, bh_returns, starting_capital=STARTING_CAPITAL)
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


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: poll data, run specialists for all tickers, log decisions."""
    import time

    from .broker import send_discord_notification
    from .meta_learner import PPOTrader

    if getattr(args, "autodiscover", False):
        from .universe import run_autodiscover
        tickers = run_autodiscover(n=TRADE_LIST_SIZE)
        logger.info("Autodiscover tickers: %s", tickers)
    else:
        tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return
    interval_sec = args.interval * 60

    trader = PPOTrader(n_assets=n_assets)
    logger.info("Starting paper trading for %s (interval=%dm) [simulated capital $%.2f]", tickers, args.interval, STARTING_CAPITAL)

    while True:
        try:
            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t))
            if is_debug():
                _verify_specialist_outputs(specialist_list, tickers)
            action = trader.predict(specialist_outputs_by_asset=specialist_list, debug=is_debug())
            action = np.array(action, dtype=np.float64)
            if not ENV_ALLOW_CASH_EXPOSURE:
                action = np.array([_position_used(float(a)) for a in action])
            exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)

            for i, t in enumerate(tickers):
                position_pct = float(weights[i])
                out = specialist_list[i]
                logger.info(
                    "Paper | %s | Sentiment=%.2f Quant=%.2f Fund=%.2f Regime=%s | Exposure=%.1f%% Weight=%.1f%% (capital=$%.2f)",
                    t,
                    out["sentiment"],
                    out["quant_prob"],
                    out["fund_score"],
                    out["regime_name"],
                    exposure * 100,
                    position_pct * 100,
                    STARTING_CAPITAL,
                )
                send_discord_notification(t, "PAPER", position_pct, specialist_outputs=out)

            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Paper trading stopped")
            break
        except Exception as e:
            logger.exception("Paper loop error: %s", e)
            time.sleep(60)


def cmd_live(args: argparse.Namespace) -> None:
    """Live trading with Alpaca: poll data, run specialists for all tickers, place orders."""
    import time
    from datetime import date

    from .broker import AlpacaBroker, send_discord_notification
    from .meta_learner import PPOTrader

    if getattr(args, "autodiscover", False):
        from .universe import run_autodiscover
        tickers = run_autodiscover(n=TRADE_LIST_SIZE)
        logger.info("Autodiscover tickers: %s", tickers)
    else:
        tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return
    interval_sec = args.interval * 60

    broker = AlpacaBroker()
    if not broker.client:
        logger.error("Alpaca client not initialized. Check API keys in config.py or .env")
        return

    model_path = Path(LIVE_PPO_MODEL_PATH) if LIVE_PPO_MODEL_PATH else None
    trader = PPOTrader(n_assets=n_assets, model_path=model_path)
    logger.info("Starting LIVE trading for %s (interval=%dm) [fractional=%s] model=%s", tickers, args.interval, FRACTIONAL, model_path or "ppo_trader.zip")

    try:
        from .broker.discord_notifier import send_live_start_embed
        account = broker.get_account()
        cash = float(account.get("cash", 0))
        if send_live_start_embed(tickers, args.interval, model_path, FRACTIONAL, cash):
            logger.info("Sent live-start summary to Discord")
    except Exception as exc:
        logger.debug("Discord live-start notification skipped: %s", exc)

    bought_today: set[str] = set()
    last_trading_date: date | None = None

    while True:
        try:
            today = date.today()
            if last_trading_date is not None and today != last_trading_date:
                bought_today.clear()
            last_trading_date = today

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

            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t))
            if is_debug():
                _verify_specialist_outputs(specialist_list, tickers)
            action = trader.predict(specialist_outputs_by_asset=specialist_list, debug=is_debug())
            action = np.array(action, dtype=np.float64)
            if not ENV_ALLOW_CASH_EXPOSURE:
                action = np.array([_position_used(float(a)) for a in action])
            exposure, weights = _exposure_and_weights(action, ENV_ALLOW_CASH_EXPOSURE)
            # No margin: only invest when cash is positive; long-only so never target short
            if cash <= 0:
                logger.warning("Account on margin (cash=%.2f); not placing buy orders", cash)
            total_invested = max(0.0, cash * exposure) if cash > 0 else 0.0

            if not broker.is_market_open():
                logger.info("Market closed (Alpaca clock); skipping orders this cycle")
                time.sleep(interval_sec)
                continue

            # For whole-share orders we need latest price per ticker
            latest_prices: dict[str, float] = {}
            if not FRACTIONAL:
                from .data.loader import load_ohlcv
                for t in tickers:
                    df = load_ohlcv(t, lookback_years=1)
                    if df is not None and not df.empty and "close" in df.columns:
                        latest_prices[t] = float(df["close"].iloc[-1])
                    else:
                        latest_prices[t] = 0.0

            for i, t in enumerate(tickers):
                target_value = total_invested * weights[i]
                current_pos = broker.get_position(t)
                current_value = float(current_pos.get("market_value", 0))
                outputs = specialist_list[i]
                position_pct = float(weights[i])

                if target_value > current_value + MIN_TRADE_NOTIONAL:
                    if cash <= 0:
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
                elif current_value > target_value + MIN_TRADE_NOTIONAL:
                    if t in bought_today:
                        logger.info("HOLD %s: skip sell (bought today, avoid day trade)", t)
                    elif current_value <= 0:
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
                                    send_discord_notification(t, "SELL", position_pct, outputs, order)
                            else:
                                current_qty = float(current_pos.get("qty", 0))
                                price = latest_prices.get(t) or 0.0
                                if price > 0 and current_qty > 0:
                                    would_sell_qty = min(int(current_qty), max(1, int(round(notional / price))))
                                    if would_sell_qty >= current_qty * 0.95:
                                        if broker.close_position(t):
                                            logger.info("SELL ALL %s (was %d shares)", t, int(current_qty))
                                            send_discord_notification(
                                                t, "SELL", position_pct, outputs,
                                                {"close_position": True, "notional": current_value, "status": "closed"},
                                            )
                                    else:
                                        order = broker.place_order(t, qty=would_sell_qty, side="sell")
                                        if order:
                                            logger.info("SELL %s: %d shares @ ~$%.2f", t, would_sell_qty, price)
                                            send_discord_notification(t, "SELL", position_pct, outputs, order)
                                else:
                                    logger.warning("No position or price for %s; skipping sell", t)
                else:
                    logger.info("HOLD %s | Position=%.1f%%", t, position_pct * 100)

            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Live trading stopped")
            break
        except Exception as e:
            logger.exception("Live trading error: %s", e)
            time.sleep(60)


def cmd_test(args: argparse.Namespace) -> None:
    """Run minimal verification across all modes: sentiment, backtest, paper, train, tune, live."""
    from .data.loader import load_ohlcv
    from .meta_learner import PPOTrader

    tickers = _tickers_from_args(args)
    if not tickers:
        tickers = list(TRADE_TICKERS)
    n_assets = len(tickers)
    modes = (args.modes or "sentiment,backtest,paper,train,tune,live").lower().split(",")
    modes = [m.strip() for m in modes if m.strip()]

    logger.info("Test run for modes: %s", modes)

    if "sentiment" in modes:
        logger.info("--- Testing SENTIMENT (real + proxy) ---")
        from .agents import get_sentiment

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
            from .meta_learner.tuner import run_tuning
            run_tuning(returns_array=returns_array, n_assets=n_assets, n_trials=1, n_timesteps=2000)
            logger.info("  Tune trial completed")

    if "live" in modes:
        logger.info("--- Testing LIVE (1 cycle if Alpaca configured) ---")
        try:
            from .broker import AlpacaBroker
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
    from .universe import run_autodiscover

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
    from .universe.train_scorer import train_scorer_from_returns

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
        from .scheduler import run_scheduler
        run_scheduler(config_path)
    elif action == "status":
        from .scheduler import status as schedule_status
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
        from .scheduler import run_job_now
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
        from .scheduler import stop_service
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
    from .scheduler_systemd import install as systemd_install
    config_path = Path(args.config).resolve()
    project_root = BASE_DIR.parent
    systemd_install(config_path, project_root, sys.executable)


def cmd_schedule_uninstall(args: argparse.Namespace) -> None:
    """Remove systemd user service (Linux only)."""
    from .scheduler_systemd import uninstall as systemd_uninstall
    systemd_uninstall()


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

    tune_p = sub.add_parser("tune", help="Run Optuna hyperparameter tuning for PPO")
    _add_debug(tune_p)
    tune_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    tune_p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    tune_p.add_argument("--timesteps", type=int, default=100_000, help="PPO timesteps per trial")
    tune_p.add_argument("--timeout", type=int, default=0, help="Max seconds for tuning (0 = no limit, default)")
    tune_p.set_defaults(func=cmd_tune)

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


if __name__ == "__main__":
    main()
