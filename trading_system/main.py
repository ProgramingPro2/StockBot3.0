"""
CLI entry point for The Boardroom trading system.
Modes: train, backtest, paper, live (multi-asset: TRADE_TICKERS).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import N_ASSETS, OUTPUT_DIR, SPECIALIST_CACHE_DIR, TRADE_TICKERS
from .utils.logger import setup_logger, get_logger

setup_logger()
logger = get_logger()


def _run_specialists(ticker: str = "SPY") -> dict:
    """Run all specialists and return dict of outputs. Uses cache on network failure."""
    from .agents import get_sentiment, get_trend_probability, get_intrinsic_value_score, detect_regime
    from .data.loader import load_ohlcv

    sentiment, sentiment_conf = get_sentiment(ticker)
    quant_prob = get_trend_probability(ticker)
    fund_score, fund_conf = get_intrinsic_value_score(ticker)
    spy = load_ohlcv("SPY")
    regime_result = detect_regime(spy)
    regime_flag = regime_result["flag"]

    out = {
        "sentiment": sentiment,
        "quant_prob": quant_prob,
        "fund_score": fund_score,
        "regime_flag": regime_flag,
        "regime_name": regime_result["regime"],
    }
    return out


def _position_used(position_pct: float) -> float:
    """Use 50% when policy outputs ~0 so backtest/paper/live have meaningful exposure."""
    return 0.5 if position_pct < 0.01 else position_pct


def _tickers_from_args(args: argparse.Namespace) -> list[str]:
    """Resolve ticker list: --tickers override or config TRADE_TICKERS."""
    if getattr(args, "tickers", None):
        return [s.strip().upper() for s in args.tickers.split(",") if s.strip()]
    return list(TRADE_TICKERS)


def _load_aligned_returns(tickers: list[str], lookback_years: int = 3):
    """Load and align OHLCV returns for tickers. Returns (aligned_df, returns_array)."""
    import pandas as pd
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
    ensure_fundamental()
    ensure_regime(retrain=args.retrain)

    historical_returns = None
    if getattr(args, "use_historical", False):
        _, returns_array = _load_aligned_returns(tickers)
        if returns_array is not None:
            historical_returns = returns_array
            logger.info("Using historical returns for PPO training (%d days)", len(returns_array))
        else:
            logger.warning("Could not load historical data; falling back to synthetic env")

    logger.info("Training meta-learner (PPO)...")
    trader = PPOTrader(n_assets=n_assets, load=False)

    def _load_specialist_cache(t: str) -> dict | None:
        p = SPECIALIST_CACHE_DIR / f"{t.upper()}.json"
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_specialist_cache(t: str, data: dict) -> None:
        try:
            p = SPECIALIST_CACHE_DIR / f"{t.upper()}.json"
            with open(p, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def specialist_fn() -> list[dict]:
        out_list: list[dict] = []
        for t in tickers:
            cached = _load_specialist_cache(t)
            if cached:
                out_list.append(cached)
                continue
            try:
                out = _run_specialists(t)
                _save_specialist_cache(t, out)
                out_list.append(out)
            except Exception as e:
                logger.warning("Specialist fetch failed for %s: %s. Using fallback.", t, e)
                fallback = {
                    "sentiment": float(np.random.uniform(-1, 1)),
                    "quant_prob": float(np.random.uniform(0, 1)),
                    "fund_score": float(np.random.uniform(-1, 1)),
                    "regime_flag": float(np.random.uniform(0, 1)),
                    "regime_name": "Sideways",
                }
                _save_specialist_cache(t, fallback)
                out_list.append(fallback)
        return out_list

    specialist_fn()
    ppo_kwargs = None
    if getattr(args, "use_tuned", False):
        from .meta_learner.tuner import load_tuned_params
        ppo_kwargs = load_tuned_params()
        if ppo_kwargs:
            logger.info("Using tuned hyperparameters from previous tune run")
        else:
            logger.warning("No tuned params found; run 'tune' first")

    trader.train(
        total_timesteps=args.timesteps,
        specialist_fn=specialist_fn,
        historical_returns=historical_returns,
        ppo_kwargs=ppo_kwargs,
    )
    logger.info("Training complete. Models saved.")


def _backtest_stats(equity: "pd.Series", strategy_returns: "pd.Series", bh_returns: "pd.Series") -> None:
    """Log final backtest statistics (P/L, drawdown, etc.)."""
    total_return_pct = (float(equity.iloc[-1]) - 1.0) * 100
    n_days = len(equity)
    years = n_days / 252.0
    cagr = (float(equity.iloc[-1]) ** (1 / max(years, 0.01)) - 1.0) * 100 if years > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd_pct = float(drawdown.min()) * 100

    # Volatility (annualized)
    vol_ann = float(strategy_returns.std() * (252 ** 0.5)) * 100

    # Sharpe (annualized)
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * (252 ** 0.5)

    # Win rate (fraction of days with positive strategy return)
    win_rate = float((strategy_returns > 0).sum() / max(len(strategy_returns), 1)) * 100

    # Buy-and-hold comparison
    bh_equity = (1 + bh_returns.fillna(0)).cumprod()
    bh_return_pct = (float(bh_equity.iloc[-1]) - 1.0) * 100

    logger.info("--- Backtest final stats ---")
    logger.info("Total return (P/L): %.2f%%", total_return_pct)
    logger.info("CAGR: %.2f%%", cagr)
    logger.info("Max drawdown: %.2f%%", max_dd_pct)
    logger.info("Volatility (ann.): %.2f%%", vol_ann)
    logger.info("Sharpe ratio: %.3f", sharpe)
    logger.info("Win rate (days): %.1f%%", win_rate)
    logger.info("Trading days: %d", n_days)
    logger.info("Buy-and-hold return: %.2f%%", bh_return_pct)
    logger.info("Strategy vs B&H: %+.2f%%", total_return_pct - bh_return_pct)
    logger.info("----------------------------")


def _proxy_specialists_from_returns(
    returns: "pd.Series", start: int, window: int = 21
) -> dict:
    """
    Derive proxy specialist outputs from rolling returns for backtest (no API).
    Returns dict with sentiment, quant_prob, fund_score, regime_flag, and
    optional rolling_return_5d, rolling_return_21d, rolling_vol_21d.
    """
    from .config import ENV_OBS_ROLLING_FEATURES

    lookback = returns.iloc[max(0, start - window) : start]
    if lookback.empty or lookback.isna().all():
        out = {"sentiment": 0.0, "quant_prob": 0.5, "fund_score": 0.0, "regime_flag": 0.5}
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

    out = {"sentiment": 0.0, "quant_prob": quant_prob, "fund_score": 0.0, "regime_flag": regime_flag}
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
    for k, v in best.items():
        if k != "policy_kwargs":
            logger.info("  %s: %s", k, v)
    logger.info("Run 'train --use-historical --use-tuned' to train with these params.")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest portfolio (5 tickers) with trained models; save performance plot."""
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from .data.loader import load_ohlcv
    from .meta_learner import PPOTrader
    from .utils.plotting import save_performance_plot

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return

    logger.info("Running portfolio backtest for %s...", tickers)

    dfs: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=3)
        if df.empty or len(df) < 252:
            logger.error("Insufficient data for backtest: %s", t)
            return
        df = df[["close"]].copy()
        df.columns = [t]
        df["return_" + t] = df[t].pct_change()
        dfs.append(df)

    aligned = dfs[0]
    for df in dfs[1:]:
        aligned = aligned.join(df, how="inner")
    aligned = aligned.dropna(how="all").tail(252 * 2)
    if len(aligned) < 252:
        logger.error("Insufficient aligned data for backtest")
        return

    return_cols = ["return_" + t for t in tickers]
    n = len(aligned)
    window = 21
    trader = PPOTrader(n_assets=n_assets)

    strategy_returns = pd.Series(0.0, index=aligned.index, dtype=float)
    rebalance_dates: list = []
    rebalance_direction: list = []
    prev_position_sum: float | None = None

    for start in range(0, n, window):
        end = min(start + window, n)
        specialist_list: list[dict] = []
        for t, ret_col in zip(tickers, return_cols):
            ret_series = aligned[ret_col]
            specialist_list.append(_proxy_specialists_from_returns(ret_series, start, window))
        action = trader.predict(specialist_outputs_by_asset=specialist_list)
        action = np.array([_position_used(float(a)) for a in action])
        position_sum = float(np.sum(action))
        total = position_sum + 1e-8
        weights = action / total

        rebalance_dates.append(aligned.index[start])
        if prev_position_sum is not None:
            rebalance_direction.append(1 if position_sum > prev_position_sum else -1)
        else:
            rebalance_direction.append(0)
        prev_position_sum = position_sum

        for day in range(start, end):
            day_ret = sum(
                weights[i] * aligned[return_cols[i]].iloc[day]
                for i in range(n_assets)
            )
            strategy_returns.iloc[day] = day_ret
    strategy_returns = strategy_returns.fillna(0)

    equity = (1 + strategy_returns).cumprod()
    bh_returns = aligned[return_cols].mean(axis=1)
    bh_equity = (1 + bh_returns.fillna(0)).cumprod()
    returns = strategy_returns.dropna()

    _backtest_stats(equity, returns, bh_returns)
    plot_filename = f"backtest_portfolio_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    path = save_performance_plot(
        equity,
        returns,
        filename=plot_filename,
        rebalance_dates=rebalance_dates,
        rebalance_direction=rebalance_direction,
    )
    logger.info("Saved plot to %s", path)


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: poll data, run specialists for all tickers, log decisions."""
    import time

    from .broker import send_discord_notification
    from .meta_learner import PPOTrader

    tickers = _tickers_from_args(args)
    n_assets = len(tickers)
    if n_assets == 0:
        logger.error("No tickers specified; use --tickers or set TRADE_TICKERS in config")
        return
    interval_sec = args.interval * 60

    trader = PPOTrader(n_assets=n_assets)
    logger.info("Starting paper trading for %s (interval=%dm)", tickers, args.interval)

    while True:
        try:
            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t))
            action = trader.predict(specialist_outputs_by_asset=specialist_list)

            for i, t in enumerate(tickers):
                position_pct = _position_used(float(action[i]))
                out = specialist_list[i]
                logger.info(
                    "Paper | %s | Sentiment=%.2f Quant=%.2f Fund=%.2f Regime=%s | Position=%.1f%%",
                    t,
                    out["sentiment"],
                    out["quant_prob"],
                    out["fund_score"],
                    out["regime_name"],
                    position_pct * 100,
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

    from .broker import AlpacaBroker, send_discord_notification
    from .meta_learner import PPOTrader

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

    trader = PPOTrader(n_assets=n_assets)
    logger.info("Starting LIVE trading for %s (interval=%dm)", tickers, args.interval)

    while True:
        try:
            account = broker.get_account()
            cash = float(account.get("cash", 0))
            logger.info("Account cash: $%.2f", cash)

            specialist_list: list[dict] = []
            for t in tickers:
                specialist_list.append(_run_specialists(t))
            action = trader.predict(specialist_outputs_by_asset=specialist_list)
            action = np.array([_position_used(float(a)) for a in action])
            total = float(np.sum(action)) + 1e-8
            weights = action / total

            for i, t in enumerate(tickers):
                target_value = cash * weights[i]
                current_pos = broker.get_position(t)
                current_value = float(current_pos.get("market_value", 0))
                outputs = specialist_list[i]
                position_pct = weights[i]

                if target_value > current_value + 100:
                    notional = target_value - current_value
                    order = broker.place_order(t, notional=notional, side="buy")
                    if order:
                        logger.info("BUY %s: $%.2f", t, notional)
                        send_discord_notification(t, "BUY", position_pct, outputs, order)
                elif current_value > target_value + 100:
                    notional = current_value - target_value
                    order = broker.place_order(t, notional=notional, side="sell")
                    if order:
                        logger.info("SELL %s: $%.2f", t, notional)
                        send_discord_notification(t, "SELL", position_pct, outputs, order)
                else:
                    logger.info("HOLD %s | Position=%.1f%%", t, position_pct * 100)
                    send_discord_notification(t, "HOLD", position_pct, outputs)

            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Live trading stopped")
            break
        except Exception as e:
            logger.exception("Live trading error: %s", e)
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="The Boardroom - Hierarchical Multi-Agent Trading System")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train specialists and meta-learner on TRADE_TICKERS")
    train_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    train_p.add_argument("--retrain", action="store_true", help="Force retrain specialists")
    train_p.add_argument("--timesteps", type=int, default=500_000, help="PPO timesteps")
    train_p.add_argument("--use-historical", action="store_true", help="Train PPO on real historical returns instead of synthetic")
    train_p.add_argument("--use-tuned", action="store_true", help="Use hyperparameters from previous 'tune' run")
    train_p.set_defaults(func=cmd_train)

    tune_p = sub.add_parser("tune", help="Run Optuna hyperparameter tuning for PPO")
    tune_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    tune_p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    tune_p.add_argument("--timesteps", type=int, default=100_000, help="PPO timesteps per trial")
    tune_p.add_argument("--timeout", type=int, default=3600, help="Max seconds for tuning (None = no limit)")
    tune_p.set_defaults(func=cmd_tune)

    back_p = sub.add_parser("backtest", help="Backtest portfolio with trained models")
    back_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    back_p.set_defaults(func=cmd_backtest)

    paper_p = sub.add_parser("paper", help="Paper trading loop")
    paper_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    paper_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    paper_p.set_defaults(func=cmd_paper)

    live_p = sub.add_parser("live", help="Live trading with Alpaca")
    live_p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: config TRADE_TICKERS)")
    live_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    live_p.set_defaults(func=cmd_live)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
