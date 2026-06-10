"""
Fundamental Specialist: cross-sectional composite z-score for intrinsic value.

Design choice: deterministic composite z-score instead of XGBoost classifier.

The previous XGBoost classifier used P/E tercile as the label while including
trailingPE as a feature - circular by construction. With yfinance providing only
current-snapshot fundamentals (no historical per-date values), a forward-return
label is not feasible either. A cross-sectional relative-value approach is the
honest alternative: z-score each value/quality metric for a ticker vs a reference
universe, combine them, and return the composite. This is:
- Non-circular (no metric is both the label and a feature)
- Deterministic and interpretable
- Properly calibrated: 0 = average, positive = relatively attractive

Universe statistics (mean and std per metric) are persisted as JSON so the score
is stable across calls and does not retrain on each request.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import CACHE_MAX_AGE_DAYS, FUNDAMENTAL_CACHE_DIR, FUNDAMENTAL_MODEL_DIR
from ..utils.logger import get_logger

logger = get_logger()

FEATURE_KEYS = [
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "debtToEquity",
    "returnOnEquity",
    "returnOnAssets",
    "freeCashflow",
    "operatingCashflow",
    "revenueGrowth",
    "earningsGrowth",
    "profitMargins",
    "currentRatio",
    "quickRatio",
    "dividendYield",
]

# Sign convention: +1 means higher value = more undervalued/attractive
METRIC_DIRECTIONS: dict[str, int] = {
    "trailingPE": -1,
    "forwardPE": -1,
    "priceToBook": -1,
    "debtToEquity": -1,
    "returnOnEquity": +1,
    "returnOnAssets": +1,
    "profitMargins": +1,
    "revenueGrowth": +1,
    "earningsGrowth": +1,
    "currentRatio": +1,
}


def _cache_path(ticker: str) -> Path:
    """Cache file path for fundamental features."""
    FUNDAMENTAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FUNDAMENTAL_CACHE_DIR / f"{ticker.upper()}.json"


def _extract_features(ticker: str) -> dict[str, float]:
    """Extract fundamental metrics from yfinance. Uses cache on network failure."""
    ticker = ticker.upper()
    cache_path = _cache_path(ticker)

    def _load_cache() -> dict[str, float] | None:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(data: dict[str, float]) -> None:
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug("Failed to cache fundamentals for %s: %s", ticker, e)

    # Try yfinance with retries for transient SSL/network errors
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info is None:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                break
            out = {}
            for k in FEATURE_KEYS:
                v = info.get(k)
                if v is not None and not isinstance(v, str):
                    out[k] = float(v)
            if out:
                _save_cache(out)
                return out
            if attempt < 2:
                time.sleep(1 + attempt)
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                logger.warning("yfinance fundamental fetch failed for %s after 3 attempts: %s. Trying cache.", ticker, e)

    # Fallback to cache (even if stale)
    cached = _load_cache()
    if cached:
        return cached

    return {}


_universe_stats: dict[str, dict[str, float]] | None = None  # {metric: {"mean": float, "std": float}}
_STATS_PATH = FUNDAMENTAL_MODEL_DIR / "universe_stats.json"


def _ensure_model(tickers: list[str] | None = None) -> bool:
    """
    Compute and cache cross-sectional universe statistics (mean/std per metric).
    If stats already exist on disk, load them. Otherwise fetch and compute.
    """
    global _universe_stats

    if _universe_stats is not None:
        return True

    if _STATS_PATH.exists():
        try:
            with open(_STATS_PATH) as f:
                _universe_stats = json.load(f)
            return True
        except Exception as e:
            logger.warning("Failed to load universe stats: %s", e)

    if tickers is None:
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "V", "WMT", "PG",
            "XOM", "UNH", "JNJ", "PFE", "HD", "BAC", "MA", "TSLA", "NFLX", "DIS",
        ]

    rows: list[dict[str, float]] = []
    for ticker in tickers:
        feat = _extract_features(ticker)
        if len(feat) >= 3:
            rows.append(feat)

    if len(rows) < 5:
        logger.warning("Insufficient universe data for fundamental scoring (%d tickers)", len(rows))
        return False

    df = pd.DataFrame(rows)
    stats: dict[str, dict[str, float]] = {}
    for metric in METRIC_DIRECTIONS:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) < 3:
            continue
        m, s = float(vals.mean()), float(vals.std())
        if s < 1e-10:
            continue
        stats[metric] = {"mean": m, "std": s}

    if not stats:
        logger.warning("No valid metrics found in universe fundamentals")
        return False

    _universe_stats = stats
    FUNDAMENTAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_STATS_PATH, "w") as f:
            json.dump(_universe_stats, f)
    except Exception as e:
        logger.warning("Failed to save universe stats: %s", e)
    return True


def get_intrinsic_value_score(ticker: str, retrain_tickers: list[str] | None = None) -> tuple[float, float]:
    """
    Get intrinsic value score for a ticker.
    Returns (score, confidence) in [-1, 1] and [0, 1].
    Undervalued=1, Fair=0, Overvalued=-1.

    Score is a cross-sectional composite z-score: compares this ticker's value and
    quality metrics to a reference universe. Positive = relatively cheap/high quality.
    Uses tanh scaling so that a mean z-score of ~1.5 std devs maps to ±0.9.
    Confidence reflects the fraction of metrics available for this ticker.
    """
    if not _ensure_model(retrain_tickers):
        return 0.0, 0.5

    feat = _extract_features(ticker)
    if len(feat) < 3:
        return 0.0, 0.5

    z_scores: list[float] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in feat or metric not in _universe_stats:
            continue
        v = feat[metric]
        if v is None or not np.isfinite(v):
            continue
        m = _universe_stats[metric]["mean"]
        s = _universe_stats[metric]["std"]
        z = float((v - m) / s) * direction
        z_scores.append(float(np.clip(z, -3.0, 3.0)))

    if not z_scores:
        return 0.0, 0.5

    mean_z = float(np.mean(z_scores))
    # tanh maps ±1.5 avg z-score to ±0.9; naturally bounded in (-1, 1)
    score = float(np.clip(np.tanh(mean_z / 1.5), -1.0, 1.0))
    confidence = float(np.clip(len(z_scores) / len(METRIC_DIRECTIONS), 0.0, 1.0))
    return score, confidence
