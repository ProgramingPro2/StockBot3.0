"""
Logic to fetch text data for the Narrative Specialist LLM.
Sources: Alpaca (when configured), then yfinance Search(ticker) and Ticker.news; merges and ranks by relevance.
"""
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta

import yfinance as yf

from ..config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    NEWS_CACHE_DIR,
    RATE_LIMIT_REQUESTS_PER_MIN,
)
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


def _parse_one(item: dict) -> NewsItem | None:
    """Parse one raw item (nested content or flat) into NewsItem. Returns None if no title."""
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    title = content.get("title") or item.get("title", "")
    summary = content.get("summary") or content.get("description") or item.get("summary", "") or item.get("link", "")
    pub = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    published = str(pub) if pub else None
    if not isinstance(title, str):
        title = str(title) if title else ""
    if not isinstance(summary, str):
        summary = str(summary) if summary else ""
    title = (title or "").strip()
    summary = (summary or "").strip()
    if not title and not summary:
        return None
    return NewsItem(title=title, summary=summary, published=published)


def _parse_news(raw: list, max_items: int) -> list[NewsItem]:
    """Parse raw list from yfinance (Search or Ticker) into NewsItems."""
    items = []
    for item in raw[: max_items * 2]:
        if len(items) >= max_items:
            break
        parsed = _parse_one(item)
        if parsed:
            items.append(parsed)
    return items


def _fetch_alpaca_news(ticker: str, max_items: int) -> list[NewsItem]:
    """Fetch news for ticker from Alpaca Data API (Benzinga). Returns [] if disabled or error."""
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        return []
    try:
        from alpaca.data.historical import NewsClient
        from alpaca.data.requests import NewsRequest
    except ImportError:
        logger.debug("Alpaca NewsClient not available; skip Alpaca news.")
        return []
    try:
        _throttle()
        client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        req = NewsRequest(symbols=ticker.upper(), start=start, end=end, limit=max_items)
        news_set = client.get_news(req)
        articles = getattr(news_set, "data", {}).get("news", []) or []
        out = []
        for a in articles[:max_items]:
            title = (getattr(a, "headline", None) or "").strip()
            summary = (getattr(a, "summary", None) or getattr(a, "content", None) or "").strip()
            if isinstance(summary, str) and len(summary) > 500:
                summary = summary[:500] + "..."
            pub = getattr(a, "created_at", None)
            published = str(pub) if pub else None
            if title or summary:
                out.append(NewsItem(title=title, summary=summary, published=published))
        if out:
            logger.debug("Alpaca news for %s: %d items", ticker, len(out))
        return out
    except Exception as e:
        logger.debug("Alpaca news for %s: %s", ticker, e)
        return []


# Stricter relevance: item must mention the ticker symbol (word boundary) or an ETF/sector phrase.
# Phrases used for ranking and as fallback when symbol alone would drop too many ETF stories.
_TICKER_PHRASES: dict[str, list[str]] = {
    "IWM": ["Russell 2000", "iShares Russell", "small cap", "small-cap"],
    "XLE": ["Energy Select", "energy sector", "energy ETF", "XLE"],
    "XLF": ["Financial Select", "financials sector", "XLF", "financials ETF"],
    "EEM": ["emerging markets", "iShares MSCI Emerging", "EEM", "emerging market ETF"],
    "DIA": ["Dow Jones", "Dow 30", "Diamonds ETF", "SPDR Dow", "DIA"],
    "SPY": ["S&P 500", "SPDR S&P 500", "SPY"],
    "QQQ": ["Nasdaq 100", "Invesco QQQ", "QQQ"],
}


def _ticker_or_phrase_in_title(item: NewsItem, ticker: str) -> bool:
    """True if title contains the ticker symbol (word) or an ETF/sector phrase for this ticker."""
    title = (item.title or "").strip().upper()
    ticker_upper = ticker.upper()
    if re.search(r"\b" + re.escape(ticker_upper) + r"\b", title, re.IGNORECASE):
        return True
    for phrase in _TICKER_PHRASES.get(ticker_upper, []):
        if phrase.upper() in title:
            return True
    return False


def _relevance_score(item: NewsItem, ticker: str) -> int:
    """2 = ticker in title, 1 = ticker in summary only. Used for ranking."""
    title = (item.title or "")
    summary = (item.summary or "")
    ticker_upper = ticker.upper()
    pat = re.compile(r"\b" + re.escape(ticker_upper) + r"\b", re.IGNORECASE)
    if pat.search(title):
        return 2
    if pat.search(summary):
        return 1
    return 0


def _phrase_rank_boost(item: NewsItem, ticker: str) -> int:
    """Extra boost when ETF/sector phrase in title (for tiebreak)."""
    title = (item.title or "").upper()
    for phrase in _TICKER_PHRASES.get(ticker.upper(), []):
        if phrase.upper() in title:
            return 1
    return 0


def _dedupe_and_rank(news: list[NewsItem], ticker: str, strict: bool = True) -> list[NewsItem]:
    """Dedupe by title; when strict keep only items that mention the ticker symbol; sort by relevance."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for n in news:
        key = re.sub(r"\s+", " ", (n.title or "").lower().strip())[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(n)
    if strict:
        relevant = [n for n in unique if _ticker_or_phrase_in_title(n, ticker)]
        if not relevant:
            relevant = unique
    else:
        relevant = unique
    relevant.sort(
        key=lambda n: (
            -_relevance_score(n, ticker),
            -_phrase_rank_boost(n, ticker),
            (n.title or ""),
        )
    )
    return relevant


def fetch_news(ticker: str, max_items: int = 20) -> list[NewsItem]:
    """
    Fetch recent news for a ticker. Ticker-specific: uses Search(ticker) first (best for
    single-name tickers like AAPL, TSLA), then Ticker.news; merges, dedupes, and ranks
    items that mention the ticker higher so the LLM sees the most relevant content first.
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

    for attempt in range(3):
        try:
            from_alpaca = _fetch_alpaca_news(ticker, max_items)
            _throttle()
            search_raw: list = []
            ticker_raw: list = []
            try:
                search = yf.Search(ticker, news_count=max_items)
                search_raw = list(search.news) if hasattr(search, "news") and search.news else []
            except Exception as e:
                logger.debug("Search news for %s: %s", ticker, e)
            _throttle()
            try:
                yf_ticker = yf.Ticker(ticker)
                ticker_raw = list(yf_ticker.news) if yf_ticker.news else []
            except Exception as e:
                logger.debug("Ticker.news for %s: %s", ticker, e)
            from_search = _parse_news(search_raw, max_items)
            from_ticker = _parse_news(ticker_raw, max_items)
            combined = from_alpaca + from_search + [n for n in from_ticker if n not in from_search]
            if not combined:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                return _load_cache()
            merged = _dedupe_and_rank(combined, ticker, strict=True)[:max_items]
            if merged:
                _save_cache(merged)
                logger.debug(
                    "News for %s: %d Alpaca, %d Search, %d Ticker, %d merged",
                    ticker,
                    len(from_alpaca),
                    len(from_search),
                    len(from_ticker),
                    len(merged),
                )
            return merged
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                logger.warning("Failed to fetch news for %s after 3 attempts: %s. Using cache.", ticker, e)
                return _load_cache()
    return _load_cache()


def news_to_text(news: list[NewsItem]) -> str:
    """Concatenate news items into a single string for LLM prompt."""
    if not news:
        return "No recent news available for this ticker."
    lines = []
    for i, n in enumerate(news, 1):
        lines.append(f"[{i}] {n.title}\n{n.summary}")
    return "\n\n".join(lines)
