"""Prune old checkpoints and rotate plot outputs to keep workspace tidy."""
from pathlib import Path

from ..config import META_LEARNER_MODEL_DIR, OUTPUT_DIR, PPO_CHECKPOINT_KEEP_LAST, OUTPUT_PLOT_KEEP_LAST


def prune_checkpoints(
    checkpoint_dir: Path | None = None,
    pattern: str = "ppo_trader_*_steps.zip",
    keep: int | None = None,
) -> int:
    """
    Remove oldest checkpoint zips matching pattern, keeping the `keep` most recent by mtime.
    Returns number of files deleted.
    """
    keep = keep if keep is not None else PPO_CHECKPOINT_KEEP_LAST
    d = checkpoint_dir or (META_LEARNER_MODEL_DIR / "checkpoints")
    if not d.is_dir():
        return 0
    files = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for p in files[keep:]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def rotate_output_plots(
    output_dir: Path | None = None,
    prefixes: tuple[str, ...] = ("backtest_portfolio_", "training_complete_"),
    keep: int | None = None,
) -> int:
    """
    Delete oldest PNGs for known prefixes, keeping `keep` newest per prefix.
    Returns total files removed.
    """
    keep = keep if keep is not None else OUTPUT_PLOT_KEEP_LAST
    od = output_dir or OUTPUT_DIR
    if not od.is_dir():
        return 0
    removed = 0
    for prefix in prefixes:
        files = sorted(od.glob(prefix + "*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep:]:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed
