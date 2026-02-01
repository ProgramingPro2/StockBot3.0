"""
yfinance wrapper with robust caching (parquet) and rate-limit handling.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..config import CACHE_DIR, CACHE_MAX_AGE_HOURS, RATE_LIMIT_REQUESTS_PER_MIN, YFINANCE_LOOKBACK_YEARS
from ..utils.logger import get_logger

logger = get_logger()
_last_request_time = 0.0
_request_count = 0
_REQUEST_WINDOW = 60.0  # seconds


def _rate_limit() -> None:
    """Throttle requests to avoid yfinance blocking."""
    global _last_request_time, _request_count
    now = time.time()
    if now - _last_request_time > _REQUEST_WINDOW:
        _request_count = 0
        _last_request_time = now
    _request_count += 1
    if _request_count > RATE_LIMIT_REQUESTS_PER_MIN:
        sleep_time = _REQUEST_WINDOW - (now - _last_request_time)
        if sleep_time > 0:
            logger.debug("Rate limit: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)
        _request_count = 0
        _last_request_time = time.time()
    else:
        time.sleep(0.5)  # Small delay between requests


def _cache_path(ticker: str) -> Path:
    """Cache file path for ticker OHLCV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _cache_valid(path: Path) -> bool:
    """Check if cache is fresh within max age."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=CACHE_MAX_AGE_HOURS)


def load_ohlcv(
    ticker: str,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    use_cache: bool = True,
    lookback_years: int | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from yfinance with retry and caching.
    Returns DataFrame with columns: open, high, low, close, volume.
    If lookback_years is set, fetches that many years (ignores cache).
    """
    ticker = ticker.upper()
    if lookback_years is not None:
        end = end or datetime.now()
        start = start or (end - timedelta(days=lookback_years * 365))
        use_cache = False
    cache_path = _cache_path(ticker)

    if use_cache and start is None and end is None:
        if _cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug("Loaded %s from cache (%d rows)", ticker, len(df))
                return df
            except Exception as e:
                logger.warning("Cache read failed for %s: %s", ticker, e)

    for attempt in range(3):
        try:
            _rate_limit()
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(start=start, end=end, auto_adjust=True)
            if df.empty:
                logger.warning("No data returned for %s", ticker)
                return pd.DataFrame()

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = [c.lower() for c in df.columns]

            if use_cache and start is None and end is None:
                df.to_parquet(cache_path)

            return df
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("yfinance attempt %d failed for %s: %s. Retrying in %ds", attempt + 1, ticker, e, wait)
            time.sleep(wait)

    logger.warning("Failed to load %s after 3 attempts. Trying stale cache.", ticker)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info("Using stale cache for %s (%d rows)", ticker, len(df))
            return df
        except Exception as e:
            logger.warning("Stale cache read failed for %s: %s", ticker, e)
    return pd.DataFrame()
