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
    current_positions: list[dict] | None = None,
    portfolio_value: float | None = None,
) -> str:
    """Build a short readable context string for the LLM."""
    lines = [
        "Tickers: " + ", ".join(tickers),
        f"Target: exposure={exposure * 100:.0f}% (rest in cash)",
    ]
    if cash is not None:
        lines.append(f"Cash: ${cash:,.0f}")
    lines.append("")

    # Current portfolio (actual holdings) when live broker data is provided
    if current_positions is not None and portfolio_value is not None:
        if portfolio_value <= 0:
            lines.append("Current portfolio (actual holdings): no positions / $0")
        else:
            pos_by_symbol = {p["symbol"]: float(p.get("market_value", 0)) for p in current_positions}
            lines.append(f"Current portfolio (actual holdings): Total ${portfolio_value:,.0f}.")
            for ticker in tickers:
                mv = pos_by_symbol.get(ticker, 0.0)
                pct = (mv / portfolio_value * 100) if portfolio_value > 0 else 0
                lines.append(f"  {ticker}: ${mv:,.0f} ({pct:.0f}%)")
            others = [p for p in current_positions if p.get("symbol") not in tickers]
            if others:
                for p in others:
                    sym = p.get("symbol", "?")
                    mv = float(p.get("market_value", 0))
                    lines.append(f"  Other: {sym} ${mv:,.0f}")
        lines.append("")

    for i, ticker in enumerate(tickers):
        out = specialist_list[i] if i < len(specialist_list) else {}
        w = float(weights[i]) * 100 if i < len(weights) else 0
        lines.append(
            f"  {ticker}: target weight={w:.0f}% | sentiment={out.get('sentiment', 0):.2f} "
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
    current_positions: list[dict] | None = None,
    portfolio_value: float | None = None,
) -> str | None:
    """
    Ask the LLM for one short, friendly sentence summarizing the boardroom's stance.
    Returns the sentence or None on failure (never raises; trading loop continues).
    When current_positions and portfolio_value are provided (live mode), the summary
    describes actual holdings first, then target/rebalancing if relevant.
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

    context = _build_context(
        tickers,
        specialist_list,
        weights,
        exposure,
        cash,
        current_positions=current_positions,
        portfolio_value=portfolio_value,
    )
    has_current = (
        current_positions is not None
        and portfolio_value is not None
        and portfolio_value > 0
    )
    if has_current:
        instruction = (
            "Summarize the CURRENT portfolio in plain language: what we actually hold and in what proportion. "
            "You may add one short phrase about target or rebalancing if relevant (e.g. 'rebalancing to X'). "
            "Be accurate to the actual holdings listed above, not only the target weights."
        )
    else:
        instruction = (
            "Write exactly one short, friendly sentence that says what the boardroom decided and why "
            "in plain language (e.g. which names we favor, whether we're holding cash)."
        )
    prompt = f"""You are summarizing a trading system's current stance for a human. Below is the boardroom state: symbols we track, target allocation, and key specialist scores. When "Current portfolio (actual holdings)" is present, it shows what we actually hold.

{instruction}
Be approachable and non-technical. No JSON, no bullet points, no emojis. One sentence only.

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
