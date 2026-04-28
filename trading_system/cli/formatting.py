"""
Rich-based output helpers for the Boardroom CLI.

These helpers format command results as colored panels and tables when
running in a TTY, and stay quiet in JSON mode (so machine-readable output
can flow through stdout cleanly).
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


_console = Console(highlight=False, soft_wrap=True)
_quiet = False


def get_console() -> Console:
    return _console


def set_quiet(quiet: bool) -> None:
    """When True, formatted helpers become no-ops (e.g. for --json mode)."""
    global _quiet
    _quiet = bool(quiet)


def is_quiet() -> bool:
    return _quiet


@contextmanager
def quiet_block(quiet: bool):
    """Temporarily enable/disable formatted output."""
    prev = _quiet
    set_quiet(quiet)
    try:
        yield
    finally:
        set_quiet(prev)


def banner(title: str, subtitle: str | None = None, style: str = "cyan") -> None:
    if _quiet:
        return
    body = subtitle or ""
    _console.print(Panel.fit(body, title=title, border_style=style))


def kv_panel(title: str, items: dict[str, Any], style: str = "cyan") -> None:
    if _quiet:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    for k, v in items.items():
        table.add_row(str(k), _fmt_value(v))
    _console.print(Panel(table, title=title, border_style=style, expand=False))


def stats_table(stats: dict[str, Any], starting_capital: float | None = None) -> None:
    """Render backtest statistics as a Rich table."""
    if _quiet:
        return
    table = Table(title="Backtest Stats", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")

    def add(label: str, value: str, style: str | None = None) -> None:
        if style:
            table.add_row(label, f"[{style}]{value}[/{style}]")
        else:
            table.add_row(label, value)

    if starting_capital and "final_value" in stats:
        add("Starting capital", f"${starting_capital:,.2f}")
        final_val = stats["final_value"]
        pl = stats["pl_dollars"]
        pl_style = "green" if pl >= 0 else "red"
        add("Final value", f"${final_val:,.2f}")
        add("P/L ($)", f"${pl:+,.2f}", pl_style)
        add("Peak value", f"${stats['peak_value']:,.2f}")
        add("Max drawdown ($)", f"${stats['max_dd_dollars']:,.2f}", "red")

    ret = stats.get("total_return_pct", 0.0)
    bh = stats.get("bh_return_pct", 0.0)
    vs_bh = stats.get("vs_bh", ret - bh)
    ret_style = "green" if ret >= 0 else "red"
    bh_style = "green" if bh >= 0 else "red"
    vs_style = "green" if vs_bh >= 0 else "red"

    add("Total return", f"{ret:+.2f}%", ret_style)
    add("CAGR", f"{stats.get('cagr', 0.0):+.2f}%")
    add("Max drawdown", f"{stats.get('max_dd_pct', 0.0):.2f}%", "red")
    add("Annual volatility", f"{stats.get('vol_ann', 0.0):.2f}%")
    add("Sharpe ratio", f"{stats.get('sharpe', 0.0):.3f}")
    add("Win rate (days)", f"{stats.get('win_rate', 0.0):.1f}%")
    add("Trading days", str(stats.get("n_days", 0)))
    add("Buy-and-hold return", f"{bh:+.2f}%", bh_style)
    add("Strategy vs B&H", f"{vs_bh:+.2f}%", vs_style)

    _console.print(table)


def models_table(rows: Iterable[dict[str, Any]]) -> None:
    if _quiet:
        return
    table = Table(title="PPO Models", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", justify="right")
    table.add_column("File", style="cyan")
    table.add_column("Modified")
    table.add_column("Size", justify="right")
    for i, row in enumerate(rows, 1):
        table.add_row(str(i), row.get("name", ""), row.get("mtime", ""), row.get("size", ""))
    _console.print(table)


def trade_list(tickers: Iterable[str], title: str = "Tickers") -> None:
    if _quiet:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    for t in tickers:
        table.add_row(t)
    _console.print(Panel(table, title=title, border_style="cyan", expand=False))


def emit_json(payload: dict[str, Any]) -> None:
    """Emit a single-line JSON object on stdout for machine consumption."""
    sys.stdout.write(json.dumps(payload, default=_json_default) + "\n")
    sys.stdout.flush()


def write_result_path(payload: dict[str, Any]) -> Path | None:
    """When BOARDROOM_RESULT_PATH is set, dump the payload to that file."""
    p = os.environ.get("BOARDROOM_RESULT_PATH", "").strip()
    if not p:
        return None
    out = Path(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, default=_json_default, indent=2))
    return out


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _json_default(obj: Any):
    try:
        from pathlib import Path as _P

        if isinstance(obj, _P):
            return str(obj)
    except Exception:
        pass
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)
