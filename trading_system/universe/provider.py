"""
Universe provider: returns candidate tickers from config, file, or S&P 500.
"""
from pathlib import Path

from ..config import BASE_DIR, TRADE_TICKERS, UNIVERSE_FILE, UNIVERSE_MODEL_DIR, UNIVERSE_SOURCE
from ..utils.logger import get_logger

logger = get_logger()

SP500_CACHE_PATH = UNIVERSE_MODEL_DIR / "sp500_tickers.txt"
SP500_CACHE_MAX_AGE_DAYS = 7


def _sp500_tickers_from_wikipedia() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia. Returns list of ticker symbols."""
    try:
        import pandas as pd

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        if "Symbol" in df.columns:
            tickers = df["Symbol"].str.strip().str.upper().tolist()
        elif "Ticker" in df.columns:
            tickers = df["Ticker"].str.strip().str.upper().tolist()
        else:
            logger.warning("S&P 500 table format changed; using first column")
            tickers = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
        return [t for t in tickers if t and len(t) <= 6 and t.isalpha()]
    except Exception as e:
        logger.warning("Failed to fetch S&P 500 from Wikipedia: %s", e)
        return []


def _read_sp500_cache() -> list[str] | None:
    if not SP500_CACHE_PATH.exists():
        return None
    try:
        from datetime import datetime, timedelta

        if datetime.now().timestamp() - SP500_CACHE_PATH.stat().st_mtime > SP500_CACHE_MAX_AGE_DAYS * 86400:
            return None
        text = SP500_CACHE_PATH.read_text()
        return [s.strip().upper() for s in text.splitlines() if s.strip()]
    except Exception:
        return None


def _write_sp500_cache(tickers: list[str]) -> None:
    try:
        UNIVERSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        SP500_CACHE_PATH.write_text("\n".join(tickers))
    except Exception as e:
        logger.warning("Could not write S&P 500 cache: %s", e)


def get_candidate_tickers() -> list[str]:
    """
    Return candidate tickers for autodiscovery.
    - config: return TRADE_TICKERS (no discovery).
    - file: read one ticker per line from UNIVERSE_FILE.
    - sp500: fetch S&P 500 list (cached), return tickers.
    """
    if UNIVERSE_SOURCE == "config":
        return list(TRADE_TICKERS)
    if UNIVERSE_SOURCE == "file":
        if not UNIVERSE_FILE or not UNIVERSE_FILE.strip():
            logger.warning("UNIVERSE_SOURCE=file but UNIVERSE_FILE is empty")
            return list(TRADE_TICKERS)
        path = Path(UNIVERSE_FILE.strip())
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        if not path.is_file():
            logger.warning("UNIVERSE_FILE does not exist or is not a file: %s", path)
            return list(TRADE_TICKERS)
        try:
            text = path.read_text()
            tickers = [s.strip().upper() for s in text.splitlines() if s.strip()]
            return tickers if tickers else list(TRADE_TICKERS)
        except Exception as e:
            logger.warning("Could not read UNIVERSE_FILE: %s", e)
            return list(TRADE_TICKERS)
    if UNIVERSE_SOURCE == "sp500":
        # Today's constituent list reflects survivorship bias: companies that were delisted,
        # merged, or went bankrupt are absent. Use UNIVERSE_SOURCE=file with a fixed
        # point-in-time ticker list to avoid this bias when training or running the scorer.
        logger.warning(
            "UNIVERSE_SOURCE=sp500 fetches today's S&P 500 constituent list, which induces "
            "survivorship bias in scorer training and backtesting. Use UNIVERSE_SOURCE=file "
            "with a fixed point-in-time list as the documented mitigation."
        )
        cached = _read_sp500_cache()
        if cached:
            return cached
        tickers = _sp500_tickers_from_wikipedia()
        if tickers:
            _write_sp500_cache(tickers)
            return tickers
        logger.warning("S&P 500 fetch failed; falling back to TRADE_TICKERS")
        return list(TRADE_TICKERS)
    logger.warning("Unknown UNIVERSE_SOURCE=%s; using TRADE_TICKERS", UNIVERSE_SOURCE)
    return list(TRADE_TICKERS)
