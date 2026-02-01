"""
Narrative Specialist: LLM-based sentiment via Ollama.
"""
import json
import re

from pydantic import BaseModel

from ..config import OLLAMA_MODEL
from ..data.news_scraper import fetch_news, news_to_text
from ..utils.logger import get_logger

logger = get_logger()


class SentimentOutput(BaseModel):
    """Structured output from LLM."""
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str


def _extract_json(text: str) -> str | None:
    """Extract JSON object from LLM response (handles markdown, extra text)."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try raw parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Try to extract from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # Try to find {...} object
    match = re.search(r"\{[^{}]*\"sentiment_score\"[^{}]*\}", text)
    if match:
        return match.group(0)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def get_sentiment(ticker: str) -> tuple[float, float]:
    """
    Get sentiment score and confidence for a ticker.
    Returns (sentiment_score, confidence) in [-1, 1] and [0, 1].
    Fallback: (0, 0.5) if Ollama unreachable.
    """
    news = fetch_news(ticker)
    text = news_to_text(news)

    prompt = f"""Analyze the following news about {ticker} and output a JSON object with exactly these keys:
- sentiment_score: float from -1 (very bearish) to 1 (very bullish)
- confidence: float from 0 (uncertain) to 1 (very confident)
- reasoning: brief explanation (one sentence)

News:
{text}

Output valid JSON only. Example: {{"sentiment_score": 0.2, "confidence": 0.7, "reasoning": "Mixed signals"}}"""

    for use_format in [True, False]:
        try:
            from ollama import chat

            kwargs: dict = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}]}
            if use_format:
                kwargs["format"] = SentimentOutput.model_json_schema()

            response = chat(**kwargs)

            if hasattr(response, "message"):
                content = response.message.content or ""
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content", "") or ""
            else:
                content = ""

            raw = content.strip() if content else ""
            json_str = _extract_json(raw) if raw else None

            if not json_str:
                if use_format:
                    continue
                raise ValueError("Empty or invalid JSON response")

            parsed = SentimentOutput.model_validate_json(json_str)
            score = max(-1, min(1, parsed.sentiment_score))
            conf = max(0, min(1, parsed.confidence))
            return score, conf

        except Exception as e:
            if not use_format:
                logger.warning("Ollama sentiment failed for %s: %s. Returning neutral.", ticker, e)
                return 0.0, 0.5
            continue

    logger.warning("Ollama sentiment failed for %s (empty/invalid JSON). Returning neutral.", ticker)
    return 0.0, 0.5
