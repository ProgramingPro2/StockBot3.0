"""Lightweight metrics callback: emit one JSON line per rollout for the web UI.

Activated when ``BOARDROOM_METRICS_PATH`` is set in the environment (the web
JobManager does this automatically). When unset the callback is a no-op so
CLI runs keep their existing behavior.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


def metrics_path() -> Path | None:
    raw = os.environ.get("BOARDROOM_METRICS_PATH", "").strip()
    if not raw:
        return None
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class JsonlMetricsCallback(BaseCallback):
    """Append a JSON line summarizing each rollout to ``metrics_path``.

    Schema (one line per rollout):

    ``{"ts": <unix>, "timesteps": <int>, "iter": <int>,
       "ep_rew_mean": <float|null>, "ep_len_mean": <float|null>,
       "fps": <int|null>,
       "policy_loss": <float|null>, "value_loss": <float|null>,
       "entropy_loss": <float|null>, "approx_kl": <float|null>,
       "explained_variance": <float|null>, "learning_rate": <float|null>}``

    Numbers come from SB3's ``logger.name_to_value`` snapshot and from the
    rollout buffer. Missing keys are written as ``null``.
    """

    def __init__(self, path: Path | None = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.path = path or metrics_path()
        self._iter = 0
        self._t0 = time.time()

    def _on_training_start(self) -> None:  # type: ignore[override]
        if self.path is None:
            return
        self._t0 = time.time()
        self._iter = 0
        record = {
            "ts": self._t0,
            "event": "start",
            "total_timesteps": int(getattr(self.model, "_total_timesteps", 0) or 0),
            "n_envs": int(getattr(self.model, "n_envs", 1) or 1),
        }
        self._append(record)

    def _on_step(self) -> bool:  # type: ignore[override]
        return True

    def _on_rollout_end(self) -> None:  # type: ignore[override]
        if self.path is None:
            return
        self._iter += 1
        values = self._snapshot_logger()
        record: dict[str, Any] = {
            "ts": time.time(),
            "event": "rollout",
            "iter": self._iter,
            "timesteps": int(self.num_timesteps),
            "elapsed": time.time() - self._t0,
            "ep_rew_mean": values.get("rollout/ep_rew_mean"),
            "ep_len_mean": values.get("rollout/ep_len_mean"),
            "fps": values.get("time/fps"),
            "policy_loss": values.get("train/policy_gradient_loss"),
            "value_loss": values.get("train/value_loss"),
            "entropy_loss": values.get("train/entropy_loss"),
            "approx_kl": values.get("train/approx_kl"),
            "explained_variance": values.get("train/explained_variance"),
            "learning_rate": values.get("train/learning_rate"),
            "clip_fraction": values.get("train/clip_fraction"),
            "loss": values.get("train/loss"),
            "eval_mean_reward": values.get("eval/mean_reward"),
        }
        self._append(record)

    def _on_training_end(self) -> None:  # type: ignore[override]
        if self.path is None:
            return
        self._append({
            "ts": time.time(),
            "event": "end",
            "iter": self._iter,
            "timesteps": int(self.num_timesteps),
            "elapsed": time.time() - self._t0,
        })

    def _snapshot_logger(self) -> dict[str, float]:
        """Read SB3's logger.name_to_value at the end of a rollout."""
        out: dict[str, float] = {}
        sb3_logger = getattr(self.model, "logger", None)
        n2v = getattr(sb3_logger, "name_to_value", None) if sb3_logger else None
        if not n2v:
            return out
        for k, v in dict(n2v).items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


__all__ = ["JsonlMetricsCallback", "metrics_path"]
