"""
Logic to fetch text data for the Narrative Specialist LLM.
Primary: yfinance news. Uses cache on network failure.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import yfinance as yf

from ..config import NEWS_CACHE_DIR, RATE_LIMIT_REQUESTS_PER_MIN
from ..utils.logger import get_logger

logger = get_logger()
_last_request_time = 0.0


def _throttle() -> None:
    """Simple throttle for news requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 60 / RATE_LIMIT_REQUESTS_PER_MIN:
        time.sleep(60 / RATE_LIMIT_REQUESTS_PER_MIN - elapsed)
    _last_request_time = time.time()


@dataclass
class NewsItem:
    """News headline and summary for LLM."""
    title: str
    summary: str
    published: str | None


def _news_cache_path(ticker: str) -> Path:
    """Cache file path for news."""
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NEWS_CACHE_DIR / f"{ticker.upper()}.json"


def fetch_news(ticker: str, max_items: int = 20) -> list[NewsItem]:
    """
    Fetch recent news headlines and summaries for a ticker.
    Primary: yfinance Ticker.news. Uses cache on network failure.
    """
    ticker = ticker.upper()
    cache_path = _news_cache_path(ticker)

    def _load_cache() -> list[NewsItem]:
        if not cache_path.exists():
            return []
        try:
            with open(cache_path) as f:
                raw = json.load(f)
            return [NewsItem(**item) for item in raw]
        except Exception:
            return []

    def _save_cache(items: list[NewsItem]) -> None:
        try:
            with open(cache_path, "w") as f:
                json.dump([{"title": i.title, "summary": i.summary, "published": i.published} for i in items], f)
        except Exception as e:
            logger.debug("Failed to cache news for %s: %s", ticker, e)

    _throttle()

    try:
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.news
        if not raw:
            return _load_cache()

        items = []
        for item in raw[:max_items]:
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("link", "")
            pub = item.get("providerPublishTime")
            published = str(pub) if pub else None
            items.append(NewsItem(title=title, summary=summary, published=published))

        if items:
            _save_cache(items)
        return items
    except Exception as e:
        logger.warning("Failed to fetch news for %s: %s. Using cache.", ticker, e)
        return _load_cache()


def news_to_text(news: list[NewsItem]) -> str:
    """Concatenate news items into a single string for LLM prompt."""
    if not news:
        return "No recent news available for this ticker."
    lines = []
    for i, n in enumerate(news, 1):
        lines.append(f"[{i}] {n.title}\n{n.summary}")
    return "\n\n".join(lines)
