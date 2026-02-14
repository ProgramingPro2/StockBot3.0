"""
Relative Strength Specialist: Per-ticker score from performance vs benchmark (SPY or sector ETF) over 20d/60d.
Complements momentum by making outperforming the market explicit; useful for rotation.
"""
import numpy as np

from ..config import RELATIVE_STRENGTH_BENCHMARK
from ..data.loader import load_ohlcv
from ..utils.logger import get_logger

logger = get_logger()

MIN_DAYS = 61  # Need at least 60d of history for 60d return


def get_relative_strength_score(ticker: str, benchmark: str | None = None) -> float:
    """
    Get relative strength score for a ticker vs a benchmark over 20d and 60d.
    Returns float in [-1, 1]: negative = underperforming, positive = outperforming.
    """
    bench = (benchmark or RELATIVE_STRENGTH_BENCHMARK).strip().upper()
    ticker = ticker.strip().upper()

    df_ticker = load_ohlcv(ticker)
    df_bench = load_ohlcv(bench)
    if df_ticker.empty or "close" not in df_ticker.columns or len(df_ticker) < MIN_DAYS:
        logger.debug("Relative strength: insufficient ticker data for %s", ticker)
        return 0.0
    if df_bench.empty or "close" not in df_bench.columns or len(df_bench) < MIN_DAYS:
        logger.debug("Relative strength: insufficient benchmark data for %s", bench)
        return 0.0

    # Align to common index (ticker dates)
    common = df_ticker[["close"]].join(df_bench[["close"]], how="inner", rsuffix="_bench")
    if len(common) < MIN_DAYS:
        return 0.0

    close_t = common["close"]
    close_b = common["close_bench"]
    ret_20_t = (close_t.iloc[-1] / close_t.iloc[-21]) - 1.0 if len(close_t) >= 21 else 0.0
    ret_60_t = (close_t.iloc[-1] / close_t.iloc[-61]) - 1.0 if len(close_t) >= 61 else 0.0
    ret_20_b = (close_b.iloc[-1] / close_b.iloc[-21]) - 1.0 if len(close_b) >= 21 else 0.0
    ret_60_b = (close_b.iloc[-1] / close_b.iloc[-61]) - 1.0 if len(close_b) >= 61 else 0.0

    rel_20 = ret_20_t - ret_20_b
    rel_60 = ret_60_t - ret_60_b
    avg_rel = 0.5 * (rel_20 + rel_60)

    # Scale: e.g. ±0.2 relative return -> ±1; use linear cap
    scale = 5.0  # 0.2 -> 1.0
    score = float(np.clip(avg_rel * scale, -1.0, 1.0))
    return score
