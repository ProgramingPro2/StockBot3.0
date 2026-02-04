"""
Autodiscover pipeline: get candidates, extract features, score, select top N.
"""
from ..config import TRADE_LIST_SIZE, UNIVERSE_SOURCE
from ..utils.logger import get_logger
from .features import extract_features_for_ticker
from .provider import get_candidate_tickers
from .scorer import ScorerModel
from .selector import select_top_n

logger = get_logger()


def run_autodiscover(n: int | None = None) -> list[str]:
    """
    Run autodiscovery: get candidate tickers, score each, return top n.
    When UNIVERSE_SOURCE=config, returns TRADE_TICKERS (no discovery).
    Otherwise returns list of n tickers (default TRADE_LIST_SIZE).
    """
    candidates = get_candidate_tickers()
    if UNIVERSE_SOURCE == "config":
        return candidates
    n = n or TRADE_LIST_SIZE
    scorer = ScorerModel(load=True)
    scored: list[tuple[str, float]] = []
    for ticker in candidates:
        try:
            features = extract_features_for_ticker(ticker)
            score = scorer.score_features(features)
            scored.append((ticker, score))
        except Exception as e:
            logger.debug("Skip %s: %s", ticker, e)
    return select_top_n(scored, n=n)
