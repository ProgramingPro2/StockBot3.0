"""Lightning callbacks for specialist training visibility (non-TTY logs + JobManager metrics)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from .logger import get_logger

logger = get_logger()


def _float_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        try:
            if hasattr(v, "detach"):
                v = v.detach().cpu().item()
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _append_jsonl(record: dict[str, Any]) -> None:
    raw = os.environ.get("BOARDROOM_METRICS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


class EpochProgressCallback(Callback):
    """Log one line per training epoch for web UI logs and optional ``metrics.jsonl``."""

    def __init__(self, phase: str = "specialist") -> None:
        super().__init__()
        self.phase = phase

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        metrics = dict(trainer.callback_metrics or {})
        loss = _float_metric(metrics, "train_loss_epoch", "train_loss")
        epoch_1based = int(trainer.current_epoch) + 1
        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        loss_str = f"{loss:.6f}" if loss is not None else "n/a"
        logger.info(
            "Specialist %s | epoch %d/%d | train_loss=%s",
            self.phase,
            epoch_1based,
            max_epochs,
            loss_str,
        )
        rec: dict[str, Any] = {
            "ts": time.time(),
            "event": "specialist",
            "phase": self.phase,
            "epoch": epoch_1based,
            "max_epochs": max_epochs,
            "train_loss": loss,
        }
        _append_jsonl(rec)


__all__ = ["EpochProgressCallback"]
