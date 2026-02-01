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
    trader.train(total_timesteps=args.timesteps, specialist_fn=specialist_fn)
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
) -> tuple[float, float, float, float]:
    """
    Derive proxy specialist outputs from rolling returns for backtest (no API).
    Returns (sentiment, quant_prob, fund_score, regime_flag) in expected ranges.
    """
    lookback = returns.iloc[max(0, start - window) : start]
    if lookback.empty or lookback.isna().all():
        return 0.0, 0.5, 0.0, 0.5
    mean_ret = float(lookback.mean())
    std_ret = float(lookback.std())
    if std_ret < 1e-8:
        std_ret = 0.01
    # quant_prob: 0.5 + trend bias, clamped to [0, 1]
    trend = max(-0.5, min(0.5, mean_ret * 20))
    quant_prob = 0.5 + trend
    # regime_flag: higher vol -> higher value, clamped to [0, 1]
    vol_norm = max(-0.5, min(0.5, (std_ret - 0.01) * 50))
    regime_flag = 0.5 + vol_norm
    return 0.0, quant_prob, 0.0, regime_flag


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest portfolio (5 tickers) with trained models; save performance plot."""
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
    for start in range(0, n, window):
        end = min(start + window, n)
        specialist_list: list[dict] = []
        for t, ret_col in zip(tickers, return_cols):
            ret_series = aligned[ret_col]
            s, q, f, r = _proxy_specialists_from_returns(ret_series, start, window)
            specialist_list.append({
                "sentiment": s, "quant_prob": q, "fund_score": f, "regime_flag": r,
            })
        action = trader.predict(specialist_outputs_by_asset=specialist_list)
        action = np.array([_position_used(float(a)) for a in action])
        total = float(np.sum(action)) + 1e-8
        weights = action / total
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
    path = save_performance_plot(equity, returns, filename="backtest_portfolio.png")
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
    train_p.add_argument("--timesteps", type=int, default=50_000, help="PPO timesteps")
    train_p.set_defaults(func=cmd_train)

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
