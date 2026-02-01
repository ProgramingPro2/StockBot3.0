"""
Plotting utilities for The Boardroom.
Headless only: save to disk, no plt.show().
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import OUTPUT_DIR


def save_performance_plot(
    equity_curve: pd.Series,
    returns: pd.Series | None = None,
    filename: str = "performance.png",
    rebalance_dates: list | None = None,
    rebalance_direction: list | None = None,
) -> Path:
    """
    Save equity curve and optional returns to PNG.
    Optionally overlay rebalance markers (green = increase, red = decrease).
    No plt.show() - headless only.
    """
    fig, axes = plt.subplots(2 if returns is not None else 1, 1, figsize=(12, 6))
    if returns is None:
        axes = [axes]

    axes[0].plot(equity_curve.index, equity_curve.values, label="Equity")
    axes[0].set_title("Equity Curve")
    axes[0].set_xlabel("Date")
    axes[0].grid(True, alpha=0.3)

    if rebalance_dates:
        directions = rebalance_direction if rebalance_direction and len(rebalance_direction) == len(rebalance_dates) else None
        increase_dates, increase_vals = [], []
        decrease_dates, decrease_vals = [], []
        neutral_dates, neutral_vals = [], []
        for i, d in enumerate(rebalance_dates):
            if d not in equity_curve.index:
                continue
            val = equity_curve.loc[d]
            if directions is not None:
                di = directions[i]
                if di == 1 or di == "buy" or (isinstance(di, (int, float)) and di > 0):
                    increase_dates.append(d)
                    increase_vals.append(val)
                elif di == -1 or di == "sell" or (isinstance(di, (int, float)) and di < 0):
                    decrease_dates.append(d)
                    decrease_vals.append(val)
                else:
                    neutral_dates.append(d)
                    neutral_vals.append(val)
            else:
                neutral_dates.append(d)
                neutral_vals.append(val)
        if increase_dates:
            axes[0].scatter(increase_dates, increase_vals, c="green", marker="^", s=40, zorder=5, label="Rebalance (increase)")
        if decrease_dates:
            axes[0].scatter(decrease_dates, decrease_vals, c="red", marker="v", s=40, zorder=5, label="Rebalance (decrease)")
        if neutral_dates:
            axes[0].scatter(neutral_dates, neutral_vals, c="gray", marker="|", s=80, zorder=5, label="Rebalance")
        axes[0].legend(loc="best", fontsize=8)

    if returns is not None:
        axes[1].hist(returns.dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_title("Returns Distribution")
        axes[1].set_xlabel("Return")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
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
