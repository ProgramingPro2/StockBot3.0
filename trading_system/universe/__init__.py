"""
Universe provider, feature extraction, scorer model, and selector for autodiscovery.
When UNIVERSE_SOURCE is config, trade list is TRADE_TICKERS (no discovery).
When file or sp500, get_candidate_tickers() returns the candidate list; scorer + selector pick top N.
"""
from .autodiscover import run_autodiscover
from .features import extract_features_for_ticker
from .provider import get_candidate_tickers
from .scorer import ScorerModel
from .selector import select_top_n

__all__ = [
    "run_autodiscover",
    "get_candidate_tickers",
    "ScorerModel",
    "select_top_n",
    "extract_features_for_ticker",
]
