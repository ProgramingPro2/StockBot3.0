"""
LLM-generated one-sentence summary of the boardroom's stance each cycle (paper/live).
Uses Ollama with same model/fallback pattern as Narrative Specialist; fails gracefully.
"""
import os
import time
from typing import TYPE_CHECKING

from ..config import OLLAMA_FALLBACK_MODELS, OLLAMA_MODEL, OLLAMA_URL
from .logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger()


def _models_to_try() -> list[str]:
    """Primary model first, then fallbacks, no duplicates."""
    seen = set()
    out = []
    for m in [OLLAMA_MODEL] + OLLAMA_FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _build_context(
    tickers: list[str],
    specialist_list: list[dict],
    weights: "np.ndarray",
    exposure: float,
    cash: float | None,
) -> str:
    """Build a short readable context string for the LLM."""
    lines = [
        "Tickers: " + ", ".join(tickers),
        f"Portfolio: exposure={exposure * 100:.0f}% (rest in cash)",
    ]
    if cash is not None:
        lines.append(f"Cash: ${cash:,.0f}")
    lines.append("")
    for i, ticker in enumerate(tickers):
        out = specialist_list[i] if i < len(specialist_list) else {}
        w = float(weights[i]) * 100 if i < len(weights) else 0
        lines.append(
            f"  {ticker}: weight={w:.0f}% | sentiment={out.get('sentiment', 0):.2f} "
            f"quant_prob={out.get('quant_prob', 0.5):.2f} fund={out.get('fund_score', 0):.2f} "
            f"regime={out.get('regime_name', '?')} momentum={out.get('momentum_score', 0):.2f}"
        )
    return "\n".join(lines)


def get_boardroom_summary(
    tickers: list[str],
    specialist_list: list[dict],
    weights: "np.ndarray",
    exposure: float,
    cash: float | None = None,
) -> str | None:
    """
    Ask the LLM for one short, friendly sentence summarizing the boardroom's stance.
    Returns the sentence or None on failure (never raises; trading loop continues).
    """
    import numpy as np

    if not tickers or not specialist_list or weights is None:
        return None
    weights = np.asarray(weights)
    if len(weights) != len(tickers) or len(specialist_list) != len(tickers):
        return None

    if "OLLAMA_HOST" not in os.environ and OLLAMA_URL:
        host = OLLAMA_URL.replace("https://", "").replace("http://", "").strip()
        if host:
            os.environ.setdefault("OLLAMA_HOST", host)

    context = _build_context(tickers, specialist_list, weights, exposure, cash)
    prompt = f"""You are summarizing a trading system's current stance for a human. Below is the boardroom state: which symbols we track, how much we allocate to each, and key specialist scores (sentiment, trend probability, fundamental score, regime, momentum).

Write exactly one short, friendly sentence that says what the boardroom decided and why in plain language (e.g. which names we favor, whether we're holding cash). Be approachable and non-technical. No JSON, no bullet points, no emojis. One sentence only.

Boardroom state:
{context}

One-sentence summary:"""

    models = _models_to_try()
    last_error: Exception | None = None

    for model in models:
        for attempt in range(2):
            try:
                from ollama import chat

                response = chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )

                if isinstance(response, str):
                    content = response.strip()
                elif hasattr(response, "message"):
                    content = (response.message.content or "").strip()
                elif isinstance(response, dict):
                    content = (response.get("message", {}).get("content", "") or "").strip()
                else:
                    content = ""

                if not content or len(content) > 500:
                    raise ValueError("Empty or too long response from Ollama")
                # Take first sentence if model added more
                first = content.split(".")[0].strip()
                if first:
                    content = first + "." if not first.endswith(".") else first
                return content[:400].strip()

            except Exception as e:
                last_error = e
                if attempt < 1:
                    time.sleep(1)
                if model != models[-1]:
                    logger.debug(
                        "Boardroom summary Ollama model %s failed: %s, trying next.",
                        model,
                        e,
                    )
                continue

    logger.warning(
        "Boardroom summary failed (Ollama). Last error: %s. Continuing without summary.",
        last_error,
    )
    return None
