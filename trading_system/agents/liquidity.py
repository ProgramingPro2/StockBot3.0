"""
Liquidity Specialist: Rule-based normalized score [0,1] from volume percentile over 20d/60d.
Helps the meta-learner avoid sizing too large in illiquid names.
"""
import numpy as np

from ..data.loader import load_ohlcv
from ..utils.logger import get_logger

logger = get_logger()


def get_liquidity_score(
    ticker: str,
    window_short: int = 20,
    window_long: int = 60,
) -> float:
    """
    Get liquidity score for a ticker from volume percentile over the lookback window.
    Returns float in [0, 1]: higher = more liquid (current volume high vs recent history).
    """
    ticker = ticker.strip().upper()
    df = load_ohlcv(ticker)
    if df.empty or "volume" not in df.columns:
        logger.debug("Liquidity: no volume data for %s", ticker)
        return 0.5

    vol = df["volume"].dropna()
    if len(vol) < window_short:
        return 0.5

    # Use window_long if enough history, else window_short
    w = min(window_long, len(vol))
    if w < 5:
        return 0.5

    recent = vol.tail(w).values.astype(float)
    current = float(vol.iloc[-1])
    if current <= 0 or np.nanmax(recent) <= 0:
        return 0.5

    # Percentile of current volume within the recent series (0-100)
    pct = float(np.sum(recent <= current) / len(recent)) * 100.0
    score = np.clip(pct / 100.0, 0.0, 1.0)
    return float(score)
