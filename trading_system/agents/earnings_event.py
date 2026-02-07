"""
Earnings/Event Risk Specialist: Calendar-based event risk score.
Higher when closer to next earnings date.
Rule-based; no training required.
"""
from datetime import datetime, timezone

import yfinance as yf

from ..config import EVENT_RISK_LOOKBACK_DAYS
from ..utils.logger import get_logger

logger = get_logger()


def get_event_risk_score(ticker: str) -> float:
    """
    Get event risk score for a ticker based on days until next earnings.
    Higher = closer to earnings (more event risk, consider reducing size).
    Returns float in [0, 1].
    """
    try:
        t = yf.Ticker(ticker.upper())
        dates_df = t.get_earnings_dates(limit=4)
        if dates_df is None or dates_df.empty:
            return 0.0

        now = datetime.now(timezone.utc)
        future_dates = []
        for idx in dates_df.index:
            if hasattr(idx, "to_pydatetime"):
                dt = idx.to_pydatetime()
            else:
                dt = idx
            if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > now:
                future_dates.append(dt)

        if not future_dates:
            return 0.0

        next_earnings = min(future_dates)
        delta = (next_earnings - now).days

        if delta > EVENT_RISK_LOOKBACK_DAYS:
            return 0.0

        score = max(0.0, 1.0 - delta / EVENT_RISK_LOOKBACK_DAYS)
        return float(min(1.0, score))

    except Exception as e:
        logger.debug("Earnings date fetch failed for %s: %s", ticker, e)
        return 0.0
