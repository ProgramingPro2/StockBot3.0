"""
Discord webhook notifications for trades.
"""
import json
import math
from pathlib import Path

import requests

from ..config import DISCORD_WEBHOOK_URL
from ..utils.logger import get_logger

logger = get_logger()


def send_discord_notification(
    ticker: str,
    action: str,
    position_pct: float,
    specialist_outputs: dict | None = None,
    order_details: dict | None = None,
) -> bool:
    """
    Send trade notification to Discord webhook.
    action: 'BUY', 'SELL', 'HOLD', 'PAPER'
    Returns True on success.
    """
    if not DISCORD_WEBHOOK_URL:
        return False

    specialist_outputs = specialist_outputs or {}
    order_details = order_details or {}

    # Build embed
    color = 0x00FF00 if action == "BUY" else 0xFF0000 if action == "SELL" else 0xFFFF00
    fields = [
        {"name": "Action", "value": action, "inline": True},
        {"name": "Position", "value": f"{position_pct * 100:.1f}%", "inline": True},
    ]

    if specialist_outputs:
        fields.append({"name": "Sentiment", "value": f"{specialist_outputs.get('sentiment', 0):.2f}", "inline": True})
        fields.append({"name": "Quant Prob", "value": f"{specialist_outputs.get('quant_prob', 0.5):.2f}", "inline": True})
        fields.append({"name": "Fund Score", "value": f"{specialist_outputs.get('fund_score', 0):.2f}", "inline": True})
        fields.append({"name": "Regime", "value": specialist_outputs.get("regime_name", "Unknown"), "inline": True})
        fields.append({"name": "MeanRev", "value": f"{specialist_outputs.get('mean_reversion_score', 0):.2f}", "inline": True})
        fields.append({"name": "Vol", "value": f"{specialist_outputs.get('volatility_score', 0.5):.2f}", "inline": True})
        fields.append({"name": "Event", "value": f"{specialist_outputs.get('event_risk_score', 0):.2f}", "inline": True})

    if order_details:
        if "qty" in order_details and order_details["qty"]:
            fields.append({"name": "Qty", "value": f"{order_details['qty']:.2f}", "inline": True})
        if "notional" in order_details and order_details["notional"]:
            fields.append({"name": "Notional", "value": f"${order_details['notional']:.2f}", "inline": True})
        if "status" in order_details:
            fields.append({"name": "Status", "value": order_details["status"], "inline": True})

    payload = {
        "embeds": [
            {
                "title": f"The Boardroom - {ticker}",
                "color": color,
                "fields": fields,
                "footer": {"text": "The Boardroom Trading System"},
            }
        ]
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Discord notification failed: %s", e)
        return False


def send_training_complete_embed(
    timesteps: int,
    stats: dict,
    plot_path: Path,
    dated_model_name: str | None = None,
    historical_ratio: float | None = None,
) -> bool:
    """
    Send an embedded Discord message for training complete (runs over 1M steps).
    stats: dict from _compute_backtest_stats (total_return_pct, cagr, max_dd_pct, sharpe, etc.).
    plot_path: path to the performance plot PNG.
    dated_model_name: optional filename of the dated model saved (e.g. ppo_trader_2026-02-07_14-30-00.zip).
    """
    if not DISCORD_WEBHOOK_URL:
        return False

    fields = [
        {"name": "Timesteps", "value": f"{timesteps:,}", "inline": True},
        {"name": "Total return", "value": f"{stats.get('total_return_pct', 0):.2f}%", "inline": True},
        {"name": "CAGR", "value": f"{stats.get('cagr', 0):.2f}%", "inline": True},
        {"name": "Max drawdown", "value": f"{stats.get('max_dd_pct', 0):.2f}%", "inline": True},
        {"name": "Sharpe", "value": f"{stats.get('sharpe', 0):.3f}", "inline": True},
        {"name": "Win rate (days)", "value": f"{stats.get('win_rate', 0):.1f}%", "inline": True},
        {"name": "Trading days", "value": str(stats.get("n_days", 0)), "inline": True},
        {"name": "B&H return", "value": f"{stats.get('bh_return_pct', 0):.2f}%", "inline": True},
        {"name": "Strategy vs B&H", "value": f"{stats.get('vs_bh', 0):+.2f}%", "inline": True},
    ]
    if historical_ratio is not None:
        fields.append({"name": "Historical ratio", "value": f"{historical_ratio:.2f}", "inline": True})
    if dated_model_name:
        fields.append({"name": "Dated model", "value": dated_model_name, "inline": False})

    filename = plot_path.name
    payload = {
        "embeds": [
            {
                "title": "Training complete",
                "color": 0x00FF00,
                "fields": fields,
                "image": {"url": f"attachment://{filename}"},
                "footer": {"text": "The Boardroom Trading System"},
            }
        ]
    }

    try:
        with open(plot_path, "rb") as f:
            file_data = f.read()
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            data={"payload_json": json.dumps(payload)},
            files={"file": (filename, file_data, "image/png")},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Discord training-complete notification failed: %s", e)
        return False


def send_live_start_embed(
    tickers: list[str],
    interval_min: int,
    model_path: Path | str | None,
    fractional: bool,
    cash: float | None = None,
) -> bool:
    """
    Send an embedded Discord message when the live trading program starts.
    """
    if not DISCORD_WEBHOOK_URL:
        return False

    ticker_str = ", ".join(tickers) if len(tickers) <= 10 else ", ".join(tickers[:8]) + f"... (+{len(tickers) - 8} more)"
    model_str = str(model_path) if model_path else "ppo_trader.zip"

    fields = [
        {"name": "Tickers", "value": ticker_str, "inline": False},
        {"name": "Interval", "value": f"{interval_min} min", "inline": True},
        {"name": "Model", "value": model_str, "inline": True},
        {"name": "Fractional", "value": "Yes" if fractional else "No", "inline": True},
    ]
    if cash is not None:
        fields.append({"name": "Account cash", "value": f"${cash:,.2f}", "inline": True})

    payload = {
        "embeds": [
            {
                "title": "Live trading started",
                "color": 0x0066CC,
                "fields": fields,
                "footer": {"text": "The Boardroom Trading System"},
            }
        ]
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Discord live-start notification failed: %s", e)
        return False


def send_hpo_complete_embed(
    n_trials: int,
    n_tickers: int,
    timesteps_per_trial: int,
    best_params: dict,
    best_value: float | None = None,
) -> bool:
    """
    Send an embedded Discord message for HPO (hyperparameter optimization) complete.
    best_params: dict from run_tuning (learning_rate, n_steps, gamma, etc.).
    best_value: best eval reward from Optuna study, if available.
    """
    if not DISCORD_WEBHOOK_URL:
        return False

    pk = best_params.get("policy_kwargs") or {}
    net_arch = pk.get("net_arch") or {}
    arch_str = str(net_arch) if net_arch else "default"

    fields = [
        {"name": "Trials", "value": str(n_trials), "inline": True},
        {"name": "Tickers", "value": str(n_tickers), "inline": True},
        {"name": "Timesteps/trial", "value": f"{timesteps_per_trial:,}", "inline": True},
        {"name": "Best eval reward", "value": f"{best_value:.4f}" if best_value is not None and not math.isnan(best_value) else "N/A", "inline": True},
        {"name": "Learning rate", "value": f"{best_params.get('learning_rate', 0):.2e}", "inline": True},
        {"name": "n_steps", "value": str(best_params.get("n_steps", "")), "inline": True},
        {"name": "gamma", "value": f"{best_params.get('gamma', 0):.4f}", "inline": True},
        {"name": "Clip range", "value": f"{best_params.get('clip_range', 0):.3f}", "inline": True},
        {"name": "Historical ratio", "value": f"{best_params.get('historical_ratio', 1.0):.2f}", "inline": True},
        {"name": "Net arch", "value": arch_str, "inline": False},
    ]

    payload = {
        "embeds": [
            {
                "title": "HPO complete",
                "color": 0x9B59B6,
                "fields": fields,
                "footer": {"text": "The Boardroom Trading System"},
            }
        ]
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Discord HPO notification failed: %s", e)
        return False
