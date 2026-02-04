"""
Selector: rank scored symbols and return top N for the trade list.
"""
from ..config import TRADE_LIST_SIZE
from ..utils.logger import get_logger

logger = get_logger()


def select_top_n(
    scored_tickers: list[tuple[str, float]],
    n: int | None = None,
) -> list[str]:
    """
    Sort by score descending and return top n ticker symbols.
    scored_tickers: list of (ticker, score).
    n: number to select (default TRADE_LIST_SIZE).
    """
    n = n or TRADE_LIST_SIZE
    if not scored_tickers:
        return []
    sorted_list = sorted(scored_tickers, key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_list[:n]]
