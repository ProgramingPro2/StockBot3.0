"""Read and write the project's ``.env`` file while preserving comments.

The web UI exposes a friendly grouped-form view backed by ``env_registry.json``.
On each env load, missing registry keys are written to ``.env`` with canonical defaults.

UI hint text merge order (first non-empty wins): (1) leading comment in the
user's ``.env`` for that key, (2) ``env_var_hints.json``, (3) ``.env.example``,
(4) registry ``description``, (5) generic fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .paths import ENV_EXAMPLE_FILE, ENV_FILE, ENV_HINTS_FILE, ENV_REGISTRY_FILE

ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
COMMENT_RE = re.compile(r"^\s*#\s?(.*)$")
COMMENTED_ASSIGN_RE = re.compile(
    r"^\s*#\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$"
)

SECRET_KEYS = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "DISCORD_WEBHOOK_URL",
}

GROUP_SECTION_HEADERS: dict[str, str] = {
    "Web": "# --- Web UI ---",
    "Universe": "# --- Universe ---",
    "Ollama": "# --- Ollama ---",
    "Relative strength": "# --- Relative strength ---",
    "Cache": "# --- Cache ---",
    "PPO": "# --- PPO training ---",
    "Alpaca": "# --- Alpaca ---",
    "Discord": "# --- Discord ---",
    "Cycle": "# --- Paper/live cycle ---",
    "Trading": "# --- Trading and execution ---",
    "Intraday margin": "# --- Intraday margin framework (IMF) ---",
    "Walk-forward": "# --- Walk-forward evaluation ---",
    "Environment": "# --- PPO environment ---",
}


class EnvValidationError(ValueError):
    """Raised when an env value is not in the allowed set for a registry key."""


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    default: str
    group: str
    secret: bool = False
    allowed_values: tuple[str, ...] = ()
    description: str | None = None


@dataclass
class EnvLine:
    raw: str
    key: str | None = None
    value: str | None = None
    comment: str | None = None


def _read(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def parse(text: str) -> tuple[list[EnvLine], dict[str, str]]:
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


def _strip_example_value_tail(value: str) -> str:
    v = value.strip()
    prev = None
    while prev != v:
        prev = v
        v = re.sub(r"\s+\([^)]*\)\s*$", "", v)
    return v.strip()


def _unquote_env_value(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _example_documentation() -> dict[str, str]:
    if not ENV_EXAMPLE_FILE.exists():
        return {}
    docs: dict[str, str] = {}
    pending: list[str] = []
    for raw in _read(ENV_EXAMPLE_FILE).splitlines():
        if not raw.strip():
            pending.clear()
            continue
        m_commented = COMMENTED_ASSIGN_RE.match(raw)
        if m_commented:
            key = m_commented.group(1)
            if pending:
                docs[key] = " ".join(pending)
            continue
        m_comment = COMMENT_RE.match(raw)
        if m_comment:
            pending.append(m_comment.group(1).strip())
            continue
        m_assign = ASSIGN_RE.match(raw)
        if m_assign:
            key = m_assign.group(1)
            if pending:
                docs[key] = " ".join(pending)
            pending.clear()
        else:
            pending.clear()
    return docs


_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@lru_cache(maxsize=1)
def _load_env_hints() -> dict[str, str]:
    if not ENV_HINTS_FILE.exists():
        return {}
    try:
        data = json.loads(ENV_HINTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if k.startswith("_") or not _KEY_RE.match(k):
            continue
        t = v.strip()
        if t:
            out[k] = t
    return out


def clear_env_hints_cache() -> None:
    _load_env_hints.cache_clear()
    clear_registry_cache()


@lru_cache(maxsize=1)
def _load_registry() -> tuple[RegistryEntry, ...]:
    if not ENV_REGISTRY_FILE.exists():
        return ()
    try:
        data = json.loads(ENV_REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid env registry: {ENV_REGISTRY_FILE}") from exc
    raw_vars = data.get("variables") if isinstance(data, dict) else data
    if not isinstance(raw_vars, list):
        raise RuntimeError("env_registry.json must contain a 'variables' array")
    entries: list[RegistryEntry] = []
    seen: set[str] = set()
    for item in raw_vars:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str) or not _KEY_RE.match(key) or key in seen:
            continue
        seen.add(key)
        default = item.get("default", "")
        if default is None:
            default = ""
        default = str(default)
        group = item.get("group", "Other")
        if not isinstance(group, str) or not group.strip():
            group = "Other"
        secret = bool(item.get("secret")) or key in SECRET_KEYS
        allowed_raw = item.get("allowed_values")
        allowed: tuple[str, ...] = ()
        if isinstance(allowed_raw, list):
            allowed = tuple(str(v) for v in allowed_raw)
        desc = item.get("description")
        description = desc.strip() if isinstance(desc, str) and desc.strip() else None
        entries.append(
            RegistryEntry(
                key=key,
                default=default,
                group=group.strip(),
                secret=secret,
                allowed_values=allowed,
                description=description,
            )
        )
    return tuple(entries)


def clear_registry_cache() -> None:
    _load_registry.cache_clear()


def _registry_by_key() -> dict[str, RegistryEntry]:
    return {e.key: e for e in _load_registry()}


def _fallback_hint(key: str) -> str:
    return f"Configuration option {key}. See .env.example."


def _resolve_description(
    entry: RegistryEntry,
    user_comment: str | None,
    docs: dict[str, str],
    hints: dict[str, str],
) -> str:
    if user_comment and user_comment.strip():
        return user_comment.strip()
    if entry.key in hints:
        return hints[entry.key]
    if entry.description:
        return entry.description
    if entry.key in docs:
        return docs[entry.key]
    return _fallback_hint(entry.key)


def _validate_value(entry: RegistryEntry, value: str) -> None:
    if not entry.allowed_values:
        return
    if value in entry.allowed_values:
        return
    allowed = ", ".join(entry.allowed_values)
    raise EnvValidationError(
        f"{entry.key}: invalid value {value!r}. Allowed: {allowed}"
    )


def load() -> tuple[list[EnvLine], dict[str, str], dict[str, str]]:
    text = _read(ENV_FILE)
    lines, kv = parse(text)
    return lines, kv, _example_documentation()


def _user_first_key_comments(lines: list[EnvLine]) -> dict[str, str | None]:
    comments: dict[str, str | None] = {}
    for line in lines:
        if not line.key or line.key in comments:
            continue
        comments[line.key] = line.comment
    return comments


def ensure_defaults_in_env() -> bool:
    """Append missing registry keys to ``.env`` with canonical defaults."""
    registry = _load_registry()
    if not registry:
        return False
    text = _read(ENV_FILE)
    lines, kv = parse(text)
    missing = [e for e in registry if e.key not in kv]
    if not missing:
        return False

    if lines and lines[-1].raw.strip():
        lines.append(EnvLine(raw=""))
    lines.append(EnvLine(raw="# Defaults synced from env registry (Boardroom UI)"))

    last_group: str | None = None
    for entry in missing:
        if entry.group != last_group:
            header = GROUP_SECTION_HEADERS.get(entry.group)
            if header:
                lines.append(EnvLine(raw=header))
            last_group = entry.group
        quoted = _format_value(entry.default)
        lines.append(
            EnvLine(
                raw=f"{entry.key}={quoted}",
                key=entry.key,
                value=entry.default,
            )
        )

    new_text = serialize(lines)
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(new_text)
    return True


def variables_for_ui() -> list[dict[str, str | bool | list[str] | None]]:
    """Return registry-backed variables for the UI form."""
    ensure_defaults_in_env()
    lines, kv, docs = load()
    hints = _load_env_hints()
    user_comments = _user_first_key_comments(lines)
    out: list[dict[str, str | bool | list[str] | None]] = []

    for entry in _load_registry():
        desc = _resolve_description(entry, user_comments.get(entry.key), docs, hints)
        value = kv.get(entry.key, entry.default)
        item: dict[str, str | bool | list[str] | None] = {
            "key": entry.key,
            "value": value,
            "description": desc,
            "secret": entry.secret,
            "group": entry.group,
            "allowed_values": list(entry.allowed_values) if entry.allowed_values else None,
        }
        out.append(item)
    return out


def update(updates: dict[str, str]) -> str:
    """Apply updates to the env file; validate enum keys against the registry."""
    registry = _registry_by_key()
    for key, value in updates.items():
        entry = registry.get(key)
        if entry is not None:
            _validate_value(entry, value)

    text = _read(ENV_FILE)
    lines, kv = parse(text)
    handled: set[str] = set()
    for line in lines:
        if line.key and line.key in updates:
            new_value = updates[line.key]
            entry = registry.get(line.key)
            if entry is not None:
                _validate_value(entry, new_value)
            quoted = _format_value(new_value)
            line.raw = f"{line.key}={quoted}"
            line.value = new_value
            handled.add(line.key)
    leftover = {k: v for k, v in updates.items() if k not in handled}
    if leftover:
        if lines and lines[-1].raw.strip():
            lines.append(EnvLine(raw=""))
        lines.append(EnvLine(raw="# Updated by web UI"))
        for key, value in sorted(leftover.items()):
            entry = registry.get(key)
            if entry is not None:
                _validate_value(entry, value)
            quoted = _format_value(value)
            lines.append(EnvLine(raw=f"{key}={quoted}", key=key, value=value))
    new_text = serialize(lines)
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(new_text)
    return new_text


def _format_value(value: str) -> str:
    if value == "":
        return ""
    if any(ch in value for ch in (" ", "#", "\t")) or value.startswith(("\"", "'")):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{escaped}"'
    return value


def mask_secrets(items: Iterable[dict[str, str | bool | list[str] | None]]) -> list[dict[str, str | bool | list[str] | None]]:
    out = []
    for it in items:
        row = dict(it)
        if row.get("secret") and row.get("value"):
            value = str(row["value"])
            row["value"] = value[:3] + "***" + value[-2:] if len(value) > 6 else "***"
        out.append(row)
    return out
