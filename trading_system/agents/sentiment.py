"""
Narrative Specialist: LLM-based sentiment via Ollama.
"""
import json
import os
import re
import time

from pydantic import BaseModel

from ..config import OLLAMA_FALLBACK_MODELS, OLLAMA_MODEL, OLLAMA_URL
from ..data.news_scraper import fetch_news, news_to_text
from ..utils.logger import get_logger

logger = get_logger()

# How long a cached sentiment score remains valid before the next call re-fetches.
# Configurable via SENTIMENT_CACHE_TTL_HOURS env var; default 4 hours.
_SENTIMENT_CACHE_TTL_SECONDS: float = float(os.getenv("SENTIMENT_CACHE_TTL_HOURS", "4")) * 3600.0


class SentimentOutput(BaseModel):
    """Structured output from LLM."""
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str = ""


def _extract_json(text: str) -> str | None:
    """Extract JSON object from LLM response (handles markdown, extra text)."""
    if not text or not text.strip():
        return None
    text = text.strip()
    for candidate in [text, text.split("\n")[0]]:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        raw = match.group(1).strip()
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            try:
                json.loads(raw)
                return raw
            except json.JSONDecodeError:
                pass
    match = re.search(r"\{[^{}]*\"sentiment_score\"[^{}]*\"confidence\"[^{}]*\}", text)
    if match:
        return match.group(0)
    match = re.search(r"\{[^{}]+\}", text)
    if match:
        raw = match.group(0)
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        raw = text[start : end + 1]
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass
    return None


def _parse_sentiment_response(json_str: str) -> tuple[float, float] | None:
    """Parse JSON to (score, confidence). Returns None on failure."""
    try:
        parsed = SentimentOutput.model_validate_json(json_str)
        return max(-1, min(1, parsed.sentiment_score)), max(0, min(1, parsed.confidence))
    except Exception:
        pass
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            score = float(data.get("sentiment_score", 0.0))
            conf = float(data.get("confidence", 0.5))
            return max(-1, min(1, score)), max(0, min(1, conf))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _models_to_try() -> list[str]:
    """Primary model first, then fallbacks, no duplicates."""
    seen = set()
    out = []
    for m in [OLLAMA_MODEL] + OLLAMA_FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# Per-ticker cache: (score, confidence, timestamp). TTL controlled by _SENTIMENT_CACHE_TTL_SECONDS.
_sentiment_cache: dict[str, tuple[float, float, float]] = {}


def get_sentiment(ticker: str, force_refresh: bool = False) -> tuple[float, float, bool]:
    """
    Get sentiment score and confidence for a ticker.
    Returns (sentiment_score, confidence, from_llm) in [-1, 1], [0, 1], and bool.
    When from_llm is False, caller should use proxy sentiment instead.
    When force_refresh is False, returns a cached value if it is still within TTL.
    """
    if not force_refresh and ticker in _sentiment_cache:
        score, conf, ts = _sentiment_cache[ticker]
        if time.time() - ts < _SENTIMENT_CACHE_TTL_SECONDS:
            return score, conf, True

    if "OLLAMA_HOST" not in os.environ and OLLAMA_URL:
        host = OLLAMA_URL.replace("https://", "").replace("http://", "").strip()
        if host:
            os.environ.setdefault("OLLAMA_HOST", host)

    news = fetch_news(ticker)
    text = news_to_text(news)
    num_items = len(news)
    no_news_placeholder = "No recent news available for this ticker."
    # Consider "real news" only if we have headline/summary text beyond the "[N]" labels
    text_without_labels = re.sub(r"\[\d+\]\s*", "", text).strip()
    has_real_news = num_items > 0 and text.strip() != no_news_placeholder and len(text_without_labels) > 20
    first_headline = ""
    for n in news:
        if (n.title or "").strip() or (n.summary or "").strip():
            first_headline = ((n.title or "") + " " + (n.summary or "")).strip()[:100]
            break
    preview = first_headline or ("(no headline text)" if num_items else "no items")
    logger.info(
        "Sentiment input for %s: %d news items, has_content=%s, preview=%s",
        ticker,
        num_items,
        has_real_news,
        preview[:100] + "..." if len(preview) > 100 else preview,
    )

    prompt = f"""Analyze the following news about {ticker} and output a JSON object with exactly these keys:
- sentiment_score: float in the range -1 to 1 (-1=very bearish, 0=neutral, 1=very bullish). Use 0 only when there is no news or the news is truly neutral. When headlines have a clear positive or negative tone, use a non-zero value (e.g. 0.3 for mildly bullish, -0.5 for bearish).
- confidence: float from 0 to 1. Use 0 only when there is no news. When you have real headlines, set confidence based on how much relevant news there is (e.g. 0.5 for one headline, 0.8 for several clear items).
- reasoning: brief explanation (one sentence)

News:
{text}

Output valid JSON only. Example: {{"sentiment_score": 0.2, "confidence": 0.7, "reasoning": "Mixed signals"}}"""

    models = _models_to_try()
    last_error: Exception | None = None

    for model in models:
        for attempt in range(2):
            for use_format in [False, True]:
                try:
                    from ollama import chat

                    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                    if use_format:
                        kwargs["format"] = SentimentOutput.model_json_schema()

                    response = chat(**kwargs)

                    if isinstance(response, str):
                        content = response.strip()
                    elif hasattr(response, "message"):
                        content = (response.message.content or "").strip()
                    elif isinstance(response, dict):
                        content = (response.get("message", {}).get("content", "") or "").strip()
                    else:
                        content = ""

                    if not content:
                        if use_format:
                            continue
                        raise ValueError("Empty response from Ollama")

                    json_str = _extract_json(content)
                    if not json_str:
                        if use_format:
                            continue
                        raise ValueError("Empty or invalid JSON response")

                    result = _parse_sentiment_response(json_str)
                    if result is None:
                        if use_format:
                            continue
                        raise ValueError("Could not parse sentiment_score and confidence from JSON")

                    score, conf = result
                    if has_real_news and score == 0.0 and conf == 0.0:
                        logger.info(
                            "Sentiment from LLM for %s: score=%.2f confidence=%.2f (model returned neutral despite %d news items)",
                            ticker,
                            score,
                            conf,
                            num_items,
                        )
                    else:
                        logger.info(
                            "Sentiment from LLM for %s: score=%.2f confidence=%.2f model=%s",
                            ticker,
                            score,
                            conf,
                            model,
                        )
                    _sentiment_cache[ticker] = (score, conf, time.time())
                    return score, conf, True

                except Exception as e:
                    last_error = e
                    if not use_format and attempt < 1:
                        time.sleep(1)
                    if model != models[-1]:
                        logger.debug("Ollama model %s failed for %s: %s, trying next.", model, ticker, e)
                    continue
            if attempt < 1:
                time.sleep(1)

    logger.warning(
        "Ollama sentiment failed for %s (empty/invalid JSON). Last error: %s. Using proxy.",
        ticker,
        last_error,
    )
    return 0.0, 0.5, False
