"""
Discord webhook notifications for trades.
"""
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
