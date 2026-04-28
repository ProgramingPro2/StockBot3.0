"""
Plotting utilities for The Boardroom.
Headless only: save to disk, no plt.show().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import OUTPUT_DIR
from .output_cleanup import rotate_output_plots


def save_performance_plot(
    equity_curve: pd.Series,
    returns: pd.Series | None = None,
    filename: str = "performance.png",
    rebalance_dates: list | None = None,
    rebalance_direction: list | None = None,
    rebalance_trades: list[list[tuple[str, float]]] | None = None,
) -> Path:
    """
    Save equity curve and optional returns to PNG.
    Optionally overlay rebalance markers (green = increase, red = decrease).
    When rebalance_trades is provided, show ticker and +/- % for each trade above markers.
    No plt.show() - headless only.
    """
    n_axes = 2 if returns is not None else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(16, 5 * n_axes))
    if returns is None:
        axes = [axes]

    axes[0].plot(equity_curve.index, equity_curve.values, label="Equity")
    axes[0].set_title("Equity Curve")
    axes[0].set_xlabel("Date")
    axes[0].grid(True, alpha=0.3)

    def _fmt_trades(trades: list[tuple[str, float]]) -> str:
        parts = []
        for ticker, pct in trades:
            sign = "+" if pct >= 0 else ""
            parts.append(f"{ticker} {sign}{pct:.1f}%")
        return ", ".join(parts)

    if rebalance_dates:
        directions = rebalance_direction if rebalance_direction and len(rebalance_direction) == len(rebalance_dates) else None
        trades_per_date = rebalance_trades if rebalance_trades and len(rebalance_trades) == len(rebalance_dates) else None
        increase_dates, increase_vals, increase_trades = [], [], []
        decrease_dates, decrease_vals, decrease_trades = [], [], []
        neutral_dates, neutral_vals, neutral_trades = [], [], []
        for i, d in enumerate(rebalance_dates):
            if d not in equity_curve.index:
                continue
            val = equity_curve.loc[d]
            trades_here = trades_per_date[i] if trades_per_date else []
            if directions is not None:
                di = directions[i]
                if di == 1 or di == "buy" or (isinstance(di, (int, float)) and di > 0):
                    increase_dates.append(d)
                    increase_vals.append(val)
                    increase_trades.append(trades_here)
                elif di == -1 or di == "sell" or (isinstance(di, (int, float)) and di < 0):
                    decrease_dates.append(d)
                    decrease_vals.append(val)
                    decrease_trades.append(trades_here)
                else:
                    neutral_dates.append(d)
                    neutral_vals.append(val)
                    neutral_trades.append(trades_here)
            else:
                neutral_dates.append(d)
                neutral_vals.append(val)
                neutral_trades.append(trades_here)
        if increase_dates:
            axes[0].scatter(increase_dates, increase_vals, c="green", marker="^", s=40, zorder=5, label="Rebalance (increase)")
            # for j, (d, v) in enumerate(zip(increase_dates, increase_vals)):
            #     if j < len(increase_trades) and increase_trades[j]:
            #         label = _fmt_trades(increase_trades[j])
            #         axes[0].annotate(
            #             label,
            #             xy=(d, v),
            #             xytext=(0, 14),
            #             textcoords="offset points",
            #             ha="center",
            #             fontsize=8,
            #             alpha=0.9,
            #         )
        if decrease_dates:
            axes[0].scatter(decrease_dates, decrease_vals, c="red", marker="v", s=40, zorder=5, label="Rebalance (decrease)")
            # for j, (d, v) in enumerate(zip(decrease_dates, decrease_vals)):
            #     if j < len(decrease_trades) and decrease_trades[j]:
            #         label = _fmt_trades(decrease_trades[j])
            #         axes[0].annotate(
            #             label,
            #             xy=(d, v),
            #             xytext=(0, -16),
            #             textcoords="offset points",
            #             ha="center",
            #             fontsize=8,
            #             alpha=0.9,
            #         )
        if neutral_dates:
            axes[0].scatter(neutral_dates, neutral_vals, c="gray", marker="|", s=80, zorder=5, label="Rebalance")
            # for j, (d, v) in enumerate(zip(neutral_dates, neutral_vals)):
            #     if j < len(neutral_trades) and neutral_trades[j]:
            #         label = _fmt_trades(neutral_trades[j])
            #         axes[0].annotate(
            #             label,
            #             xy=(d, v),
            #             xytext=(0, 14),
            #             textcoords="offset points",
            #             ha="center",
            #             fontsize=8,
            #             alpha=0.9,
            #         )
        axes[0].legend(loc="best", fontsize=8)
        # Leave vertical room so rebalance labels (increase above, decrease below) are not clipped
        ymin, ymax = axes[0].get_ylim()
        pad = max((ymax - ymin) * 0.12, (equity_curve.max() - equity_curve.min()) * 0.05)
        axes[0].set_ylim(ymin - pad, ymax + pad)

    if returns is not None:
        axes[1].hist(returns.dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_title("Returns Distribution")
        axes[1].set_xlabel("Return")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    rotate_output_plots()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    rotate_output_plots()
    return out_path


def _index_to_iso(idx: pd.Index) -> list[str]:
    out: list[str] = []
    for d in idx:
        if hasattr(d, "isoformat"):
            out.append(d.isoformat())  # type: ignore[union-attr]
        else:
            out.append(str(d))
    return out


def save_performance_series(
    equity_dollars: pd.Series,
    bh_returns: pd.Series,
    rebalance_dates: list | None,
    rebalance_direction: list | None,
    rebalance_trades: list[list[tuple[str, float]]] | None,
    tickers: list[str],
    weights_history: np.ndarray,
    starting_capital: float,
    filename: str,
    kind: str,
    weights_max_points: int = 250,
) -> Path:
    """
    Write interactive-chart payload (JSON) next to the PNG performance plot.
    Includes equity, buy-and-hold equity, daily returns, drawdown, rebalance events,
    and sub-sampled per-ticker weights.
    """
    idx = equity_dollars.index
    eq = equity_dollars.astype(float)
    bh_eq = ((1.0 + bh_returns.reindex(idx).fillna(0.0)).cumprod() * float(starting_capital)).astype(float)
    cummax = eq.cummax()
    drawdown = (eq / cummax.replace(0.0, np.nan) - 1.0).fillna(0.0)
    daily_ret = eq.pct_change().fillna(0.0)

    rebalances: list[dict[str, Any]] = []
    if rebalance_dates:
        directions = (
            rebalance_direction
            if rebalance_direction and len(rebalance_direction) == len(rebalance_dates)
            else None
        )
        trades_per = (
            rebalance_trades
            if rebalance_trades and len(rebalance_trades) == len(rebalance_dates)
            else None
        )
        for i, d in enumerate(rebalance_dates):
            if d not in eq.index:
                continue
            di = int(directions[i]) if directions is not None else 0
            trades_here = trades_per[i] if trades_per else []
            rebalances.append(
                {
                    "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                    "direction": di,
                    "trades": [{"ticker": t, "pct_delta": float(p)} for t, p in trades_here],
                }
            )

    n = len(weights_history)
    if n <= weights_max_points:
        row_ix = np.arange(n, dtype=np.int64)
    else:
        row_ix = np.unique(np.linspace(0, n - 1, num=weights_max_points).astype(np.int64))

    weight_rows: list[dict[str, Any]] = []
    for ri in row_ix:
        d = idx[ri]
        row: dict[str, Any] = {
            "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
        }
        for j, sym in enumerate(tickers):
            row[sym] = float(weights_history[ri, j])
        weight_rows.append(row)

    payload: dict[str, Any] = {
        "kind": kind,
        "starting_capital": float(starting_capital),
        "tickers": list(tickers),
        "dates": _index_to_iso(idx),
        "equity": [float(x) for x in eq.values],
        "bh_equity": [float(x) for x in bh_eq.values],
        "returns": [float(x) for x in daily_ret.values],
        "drawdown": [float(x) for x in drawdown.values],
        "rebalances": rebalances,
        "weights": weight_rows,
    }

    out_path = OUTPUT_DIR / filename
    rotate_output_plots()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rotate_output_plots()
    return out_path


def save_regime_plot(
    spy_data: pd.DataFrame,
    regime_labels: list[int] | list[str],
    filename: str = "regime.png",
) -> Path:
    """
    Save SPY price with regime annotations.
    No plt.show() - headless only.
    regime_labels: numeric (0-3) or strings; strings are mapped to indices.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(spy_data.index, spy_data["Close"], label="SPY Close", alpha=0.8)
    if len(regime_labels) == len(spy_data):
        if isinstance(regime_labels[0], str):
            uniq = list(dict.fromkeys(regime_labels))
            c = [uniq.index(l) for l in regime_labels]
        else:
            c = regime_labels
        ax.scatter(
            spy_data.index,
            spy_data["Close"],
            c=c,
            cmap="RdYlGn",
            alpha=0.5,
            s=10,
        )
    ax.set_title("SPY with Regime Labels")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
