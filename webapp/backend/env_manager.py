"""Read and write the project's ``.env`` file while preserving comments.

The web UI exposes a friendly grouped-form view; under the hood we keep the
file's structure intact so users can still edit it by hand without losing
notes/sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import ENV_EXAMPLE_FILE, ENV_FILE

ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
COMMENT_RE = re.compile(r"^\s*#\s?(.*)$")


SECRET_KEYS = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "DISCORD_WEBHOOK_URL",
}


GROUP_PREFIXES: list[tuple[str, tuple[str, ...]]] = [
    ("Trading", ("TRADE_", "STARTING_CAPITAL", "FRACTIONAL", "MIN_TRADE_NOTIONAL", "REBALANCE_BAND_PCT", "MIN_HOLD_MINUTES_AFTER_BUY", "MIN_REENTRY_MINUTES_AFTER_SELL", "TRANSACTION_COST_BPS", "SLIPPAGE_BPS")),
    ("Walk-forward", ("TEST_WINDOW_DAYS", "TRAIN_WINDOW_YEARS")),
    ("PPO", ("PPO_", "TRAIN_HISTORICAL_RATIO")),
    ("Environment", ("ENV_",)),
    ("PDT", ("PDT_",)),
    ("Universe", ("UNIVERSE_", "AUTODISCOVER_", "TRADE_LIST_SIZE")),
    ("Ollama", ("OLLAMA_", "BOARDROOM_SUMMARY_")),
    ("Cycle", ("CYCLE_", "BACKTEST_OLLAMA_")),
    ("Alpaca", ("ALPACA_", "LIVE_PPO_MODEL_PATH")),
    ("Discord", ("DISCORD_",)),
    ("Cache", ("CACHE_",)),
    ("Relative strength", ("RELATIVE_STRENGTH_",)),
]


def _group_for(key: str) -> str:
    for label, prefixes in GROUP_PREFIXES:
        if any(key.startswith(p) or key == p for p in prefixes):
            return label
    return "Other"


@dataclass
class EnvLine:
    raw: str
    key: str | None = None
    value: str | None = None
    comment: str | None = None  # most recent leading comment for this key


def _read(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def parse(text: str) -> tuple[list[EnvLine], dict[str, str]]:
    """Parse env content; return (lines, key->value).

    Lines preserve original content. Each assignment carries the most recent
    contiguous comment block as ``comment`` for use as a hint in the UI.
    """
    lines: list[EnvLine] = []
    pending_comment: list[str] = []
    keymap: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip():
            lines.append(EnvLine(raw=raw))
            pending_comment.clear()
            continue
        m_comment = COMMENT_RE.match(raw)
        if m_comment:
            pending_comment.append(m_comment.group(1).strip())
            lines.append(EnvLine(raw=raw))
            continue
        m_assign = ASSIGN_RE.match(raw)
        if m_assign:
            key = m_assign.group(1)
            value = m_assign.group(2).strip()
            if value.startswith(('"', "'")) and value.endswith(value[0]):
                value = value[1:-1]
            comment = " ".join(pending_comment) if pending_comment else None
            lines.append(EnvLine(raw=raw, key=key, value=value, comment=comment))
            keymap[key] = value
            pending_comment.clear()
        else:
            lines.append(EnvLine(raw=raw))
            pending_comment.clear()
    return lines, keymap


def serialize(lines: list[EnvLine]) -> str:
    return "\n".join(line.raw for line in lines).rstrip() + "\n"


def _example_documentation() -> dict[str, str]:
    """Pull short comments from .env.example to enrich the UI form."""
    if not ENV_EXAMPLE_FILE.exists():
        return {}
    _, _ = parse(_read(ENV_EXAMPLE_FILE))
    docs: dict[str, str] = {}
    pending: list[str] = []
    for raw in _read(ENV_EXAMPLE_FILE).splitlines():
        m_comment = COMMENT_RE.match(raw)
        if m_comment:
            pending.append(m_comment.group(1).strip())
            continue
        m_assign = ASSIGN_RE.match(raw)
        if m_assign:
            key = m_assign.group(1)
            if pending:
                docs[key] = " ".join(pending)
            pending = []
        else:
            pending = []
    return docs


def load() -> tuple[list[EnvLine], dict[str, str], dict[str, str]]:
    text = _read(ENV_FILE)
    lines, kv = parse(text)
    return lines, kv, _example_documentation()


def variables_for_ui() -> list[dict[str, str | bool]]:
    """Return enriched variables for the UI form."""
    lines, kv, docs = load()
    out: list[dict[str, str | bool]] = []
    seen: set[str] = set()
    for line in lines:
        if not line.key or line.key in seen:
            continue
        seen.add(line.key)
        out.append({
            "key": line.key,
            "value": kv.get(line.key, ""),
            "description": line.comment or docs.get(line.key) or None,
            "secret": line.key in SECRET_KEYS,
            "group": _group_for(line.key),
        })
    # Append documented keys missing from the user's .env so the UI can suggest them.
    for key, doc in docs.items():
        if key in seen:
            continue
        out.append({
            "key": key,
            "value": "",
            "description": doc,
            "secret": key in SECRET_KEYS,
            "group": _group_for(key),
        })
    return out


def update(updates: dict[str, str]) -> str:
    """Apply ``updates`` to the env file (preserving order/comments) and return the new content."""
    text = _read(ENV_FILE)
    lines, _ = parse(text)
    handled: set[str] = set()
    for line in lines:
        if line.key and line.key in updates:
            new_value = updates[line.key]
            quoted = _format_value(new_value)
            line.raw = f"{line.key}={quoted}"
            line.value = new_value
            handled.add(line.key)
    leftover = {k: v for k, v in updates.items() if k not in handled}
    if leftover:
        if lines and lines[-1].raw.strip():
            lines.append(EnvLine(raw=""))
        lines.append(EnvLine(raw="# Added by web UI"))
        for key, value in leftover.items():
            quoted = _format_value(value)
            lines.append(EnvLine(raw=f"{key}={quoted}", key=key, value=value))
    new_text = serialize(lines)
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(new_text)
    return new_text


def _format_value(value: str) -> str:
    """Quote when needed; keep simple scalars unquoted."""
    if value == "":
        return ""
    if any(ch in value for ch in (" ", "#", "\t")) or value.startswith(("\"", "'")):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{escaped}"'
    return value


def mask_secrets(items: Iterable[dict[str, str | bool]]) -> list[dict[str, str | bool]]:
    out = []
    for it in items:
        if it.get("secret") and it.get("value"):
            value = str(it["value"])
            it = {**it, "value": value[:3] + "***" + value[-2:] if len(value) > 6 else "***"}
        out.append(it)
    return out
