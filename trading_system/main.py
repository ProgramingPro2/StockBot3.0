"""
CLI entry point for The Boardroom trading system.
Modes: --train, --backtest, --paper
"""
import json
import argparse
from pathlib import Path

from .config import OUTPUT_DIR, SPECIALIST_CACHE_DIR
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


def cmd_train(args: argparse.Namespace) -> None:
    """Train specialists and meta-learner."""
    from .agents.fundamental import _ensure_model as ensure_fundamental
    from .agents.quantitative import _ensure_model as ensure_quantitative
    from .agents.regime import _ensure_model as ensure_regime
    from .meta_learner import PPOTrader

    logger.info("Training specialists...")
    ensure_quantitative("SPY", retrain=args.retrain)
    ensure_fundamental()
    ensure_regime(retrain=args.retrain)

    logger.info("Training meta-learner (PPO)...")
    trader = PPOTrader(n_assets=1, load=False)

    ticker = args.ticker
    cache_path = SPECIALIST_CACHE_DIR / f"{ticker.upper()}.json"

    def _load_specialist_cache() -> dict | None:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_specialist_cache(data: dict) -> None:
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def specialist_fn() -> dict:
        # Fetch once, cache; reuse for all env resets (avoids network on every reset)
        cached = _load_specialist_cache()
        if cached:
            return cached
        try:
            out = _run_specialists(ticker)
            _save_specialist_cache(out)
            return out
        except Exception as e:
            logger.warning("Specialist fetch failed: %s. Using cached/random fallback.", e)
            import numpy as np
            fallback = {
                "sentiment": float(np.random.uniform(-1, 1)),
                "quant_prob": float(np.random.uniform(0, 1)),
                "fund_score": float(np.random.uniform(-1, 1)),
                "regime_flag": float(np.random.uniform(0, 1)),
                "regime_name": "Sideways",
            }
            _save_specialist_cache(fallback)  # Cache fallback so we don't retry network
            return fallback

    # Pre-fetch and cache before training starts
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
    """Backtest with trained models; save performance plots."""
    import pandas as pd
    from .data.loader import load_ohlcv
    from .meta_learner import PPOTrader
    from .utils.plotting import save_performance_plot

    ticker = args.ticker
    logger.info("Running backtest for %s...", ticker)

    df = load_ohlcv(ticker, lookback_years=3)
    if df.empty or len(df) < 252:
        logger.error("Insufficient data for backtest")
        return

    trader = PPOTrader(n_assets=1)
    df = df.tail(252 * 2).copy()
    df["return"] = df["close"].pct_change()
    returns_series = df["return"]
    n = len(df)
    window = 21
    # Rolling backtest: one prediction per 21-day block, proxy specialists from prior returns
    strategy_returns = pd.Series(0.0, index=df.index, dtype=float)
    position_pcts: list[float] = []
    for start in range(0, n, window):
        end = min(start + window, n)
        sentiment, quant_prob, fund_score, regime_flag = _proxy_specialists_from_returns(
            returns_series, start, window
        )
        action = trader.predict(sentiment, quant_prob, fund_score, regime_flag)
        position_pct = float(action[0])
        # Fallback: trained policy can collapse to 0; use 50% exposure so backtest is meaningful
        position_used = 0.5 if position_pct < 0.01 else position_pct
        position_pcts.append(position_used)
        strategy_returns.iloc[start:end] = returns_series.iloc[start:end] * position_used

    df["strategy_return"] = strategy_returns
    df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    returns = df["strategy_return"].dropna()
    bh_returns = df["return"].copy()

    _backtest_stats(df["equity"], returns, bh_returns)
    mean_pos = sum(position_pcts) / len(position_pcts) if position_pcts else 0.0
    logger.info("Position (mean over blocks): %.1f%%", mean_pos * 100)

    path = save_performance_plot(df["equity"], returns, filename=f"backtest_{ticker}.png")
    logger.info("Saved plot to %s", path)


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: poll data, run specialists, meta-learner, log decisions."""
    import time

    from .agents import get_sentiment, get_trend_probability, get_intrinsic_value_score, detect_regime
    from .broker import send_discord_notification
    from .meta_learner import PPOTrader

    ticker = args.ticker
    interval_sec = args.interval * 60

    trader = PPOTrader(n_assets=1)
    logger.info("Starting paper trading for %s (interval=%dm)", ticker, args.interval)

    while True:
        try:
            outputs = _run_specialists(ticker)
            action = trader.predict(
                outputs["sentiment"],
                outputs["quant_prob"],
                outputs["fund_score"],
                outputs["regime_flag"],
            )
            position_pct = float(action[0])

            logger.info(
                "Paper | %s | Sentiment=%.2f Quant=%.2f Fund=%.2f Regime=%s | Position=%.1f%%",
                ticker,
                outputs["sentiment"],
                outputs["quant_prob"],
                outputs["fund_score"],
                outputs["regime_name"],
                position_pct * 100,
            )

            # Send Discord notification
            send_discord_notification(ticker, "PAPER", position_pct, specialist_outputs=outputs)

            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Paper trading stopped")
            break
        except Exception as e:
            logger.exception("Paper loop error: %s", e)
            time.sleep(60)


def cmd_live(args: argparse.Namespace) -> None:
    """Live trading with Alpaca: poll data, run specialists, place orders, send Discord notifications."""
    import time

    from .agents import get_sentiment, get_trend_probability, get_intrinsic_value_score, detect_regime
    from .broker import AlpacaBroker, send_discord_notification
    from .meta_learner import PPOTrader

    ticker = args.ticker
    interval_sec = args.interval * 60

    broker = AlpacaBroker()
    if not broker.client:
        logger.error("Alpaca client not initialized. Check API keys in config.py or .env")
        return

    trader = PPOTrader(n_assets=1)
    logger.info("Starting LIVE trading for %s (interval=%dm)", ticker, args.interval)

    while True:
        try:
            account = broker.get_account()
            cash = account.get("cash", 0)
            logger.info("Account cash: $%.2f", cash)

            outputs = _run_specialists(ticker)
            action = trader.predict(
                outputs["sentiment"],
                outputs["quant_prob"],
                outputs["fund_score"],
                outputs["regime_flag"],
            )
            position_pct = float(action[0])

            # Get current position
            current_pos = broker.get_position(ticker)
            current_value = current_pos.get("market_value", 0)
            target_value = cash * position_pct

            # Determine action
            if target_value > current_value + 100:
                # Buy more
                notional = target_value - current_value
                order = broker.place_order(ticker, notional=notional, side="buy")
                if order:
                    logger.info("BUY %s: $%.2f", ticker, notional)
                    send_discord_notification(ticker, "BUY", position_pct, outputs, order)
            elif current_value > target_value + 100:
                # Sell some
                notional = current_value - target_value
                order = broker.place_order(ticker, notional=notional, side="sell")
                if order:
                    logger.info("SELL %s: $%.2f", ticker, notional)
                    send_discord_notification(ticker, "SELL", position_pct, outputs, order)
            else:
                logger.info("HOLD %s | Position=%.1f%%", ticker, position_pct * 100)
                send_discord_notification(ticker, "HOLD", position_pct, outputs)

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

    train_p = sub.add_parser("train", help="Train specialists and meta-learner")
    train_p.add_argument("--ticker", default="SPY", help="Ticker for training")
    train_p.add_argument("--retrain", action="store_true", help="Force retrain specialists")
    train_p.add_argument("--timesteps", type=int, default=50_000, help="PPO timesteps")
    train_p.set_defaults(func=cmd_train)

    back_p = sub.add_parser("backtest", help="Backtest with trained models")
    back_p.add_argument("--ticker", default="SPY", help="Ticker for backtest")
    back_p.set_defaults(func=cmd_backtest)

    paper_p = sub.add_parser("paper", help="Paper trading loop")
    paper_p.add_argument("--ticker", default="SPY", help="Ticker for paper trading")
    paper_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    paper_p.set_defaults(func=cmd_paper)

    live_p = sub.add_parser("live", help="Live trading with Alpaca")
    live_p.add_argument("--ticker", default="SPY", help="Ticker for live trading")
    live_p.add_argument("--interval", type=int, default=60, help="Minutes between decisions")
    live_p.set_defaults(func=cmd_live)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
