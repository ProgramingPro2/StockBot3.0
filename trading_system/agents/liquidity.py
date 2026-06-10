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

    The reference window excludes the current bar so the current volume is compared
    against a purely historical baseline (self-inclusion inflates the percentile toward
    the middle because the current bar always counts as at least one match).
    """
    ticker = ticker.strip().upper()
    df = load_ohlcv(ticker)
    if df.empty or "volume" not in df.columns:
        logger.debug("Liquidity: no volume data for %s", ticker)
        return 0.5

    vol = df["volume"].dropna()
    if len(vol) < window_short + 1:
        return 0.5

    # Use window_long if enough history, else window_short; always exclude current bar
    w = min(window_long, len(vol) - 1)
    if w < 5:
        return 0.5

    # Reference window: the w bars immediately before the current bar
    reference = vol.iloc[-(w + 1):-1].values.astype(float)
    current = float(vol.iloc[-1])
    if current <= 0 or np.nanmax(reference) <= 0:
        return 0.5

    pct = float(np.sum(reference <= current) / len(reference))
    score = float(np.clip(pct, 0.0, 1.0))
    return score
