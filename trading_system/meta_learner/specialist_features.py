"""
Walk-forward historical specialist output generator for Phase 3 Mode B training.

Produces a walk-forward matrix of REAL specialist outputs for PPO training,
with the same shape/slot-ordering contract as the proxy feature matrix from
proxy_specialists.precompute_proxy_feature_matrix().

Output shape: (n_days, n_assets, N_SLOTS=13)  dtype=float32

Slot layout (matches proxy_specialists.py slot contract exactly):
  Slot  0  sentiment          [-1,  1]  PROXY FALLBACK  (LLM unavailable historically)
  Slot  1  quant_prob         [ 0,  1]  TFT walk-forward; proxy fallback when insufficient data
  Slot  2  fund_score         [-1,  1]  PROXY FALLBACK  (yfinance: current-snapshot only)
  Slot  3  regime_flag        [ 0,  1]  VAE+KMeans walk-forward on SPY
  Slot  4  momentum_score     [-1,  1]  LightGBM walk-forward
  Slot  5  mean_reversion     [-1,  1]  rule-based: RSI + price z-score vs SMA
  Slot  6  volatility         [ 0,  1]  rule-based: short/long vol ratio
  Slot  7  event_risk         [ 0,  1]  PROXY FALLBACK  (earnings calendar: no historical point-in-time)
  Slot  8  relative_strength  [-1,  1]  rule-based: ticker vs benchmark return comparison
  Slot  9  liquidity          [ 0,  1]  rule-based: volume percentile
  Slot 10  ret_5d_norm        [-1,  1]  returns-based (identical to proxy)
  Slot 11  ret_21d_norm       [-1,  1]  returns-based (identical to proxy)
  Slot 12  vol_21d_norm       [-1,  1]  returns-based (identical to proxy)

Walk-forward no-lookahead convention:
  Features at index t use data from bars [0, t-1] only — no data from bar t.
  This matches the proxy path where the window ends at t (exclusive): returns[t-21:t].
  Rule-based agents apply pandas .shift(1) to rolling statistics so that row t
  contains stats computed with close prices up to bar t-1.
  Trained models are refit on data[:refit_date] and applied to subsequent bars.

Proxy fallback reasons:
  Slot 0  – Ollama/LLM sentiment unavailable for past dates; news_scraper.as_of not
             sufficient to reconstruct point-in-time news for arbitrary historical dates.
  Slot 2  – yfinance provides only current-snapshot fundamentals; no historical per-date
             balance-sheet / income-statement series via the free API.
  Slot 7  – yfinance earnings calendar is forward-looking; no point-in-time future-earnings
             dates are available for arbitrary past bars.
  Slot 1  – TFT requires TFT_MAX_ENCODER_LENGTH+TFT_MAX_PREDICTION_LENGTH+100 ≈ 382 bars
             warmup; proxy hit-rate is used if the data window is smaller.

Public API:
  get_or_build_specialist_matrix(tickers, start, end, ...) -> (matrix, dates, metadata)
      Cache-aware entry point.  Loads OHLCV data, delegates to precompute, saves .npz cache.

  precompute_specialist_feature_matrix(aligned_close, tickers, dates, ...) -> (matrix, meta)
      Low-level function that takes pre-loaded DataFrames.  Mirrors
      precompute_proxy_feature_matrix's signature as closely as practical.

Cache layout:
  {SPECIALIST_FEATURES_CACHE_DIR}/{hex24}.npz         — matrix + dates + tickers arrays
  {SPECIALIST_FEATURES_CACHE_DIR}/{hex24}_meta.json   — full metadata dict
  Cache key: SHA-256 of (sorted_tickers, start_date, end_date, SCHEMA_VERSION,
                          refit_every, window)

CLI pre-generation:
  python -m trading_system.meta_learner.specialist_features \\
      --tickers SPY,IWM,XLE,XLF,EEM --start 2015-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    DEVICE,
    MOMENTUM_FORWARD_DAYS,
    MOMENTUM_LEARNING_RATE,
    MOMENTUM_MAX_DEPTH,
    MOMENTUM_N_ESTIMATORS,
    MOMENTUM_WINDOW_DAYS,
    RELATIVE_STRENGTH_BENCHMARK,
    VAE_HIDDEN_DIMS,
    VAE_LATENT_DIM,
    VAE_REGIME_CLUSTERS,
    VOLATILITY_REF_WINDOW_DAYS,
    VOLATILITY_WINDOW_DAYS,
)
from ..data.loader import load_ohlcv
from ..data.processor import add_momentum_features, prepare_regime_features
from ..utils.logger import get_logger
from .proxy_specialists import proxy_specialists_vectorized_batch

logger = get_logger()


def _quiet_thirdparty_logging() -> None:
    """Silence noisy third-party output during walk-forward generation.

    Lightning re-logs hardware banners and upsell tips on every Trainer
    construction (~90 refits per run), and sklearn warns on every LGB predict
    call; none of it is actionable here.
    """
    import logging as _logging
    import os as _os
    import warnings as _warnings

    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning"):
        _logging.getLogger(name).setLevel(_logging.WARNING)
    _warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    _warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    _warnings.filterwarnings("ignore", message=".*you defined a `validation_step`.*")
    _warnings.filterwarnings("ignore", message=".*ReduceLROnPlateau conditioned on metric.*")
    _warnings.filterwarnings("ignore", message=".*litmodels.*")
    # Suppress Lightning's stdout tip about litmodels / cloud uploads.
    # It's written via rich's Console.print(), not through Python logging,
    # so we patch it out at the env-var level that Lightning checks first.
    _os.environ.setdefault("LIGHTNING_DISABLE_SUPERFLUOUS_WARNINGS", "1")
    _os.environ.setdefault("LT_DISABLE_LIGHTNING_TIPS", "1")


def _emit_progress(stage: str, frac: float, detail: str = "") -> None:
    """Append a progress row to the web UI metrics stream (no-op outside jobs).

    The Boardroom job manager tails BOARDROOM_METRICS_PATH and broadcasts each
    line over WebSocket; the Train page renders these rows as a progress bar.
    """
    path = os.environ.get("BOARDROOM_METRICS_PATH", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "event": "progress",
                        "stage": stage,
                        "progress": round(max(0.0, min(1.0, float(frac))), 4),
                        "detail": detail,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Config constants (with graceful fallback if not yet in config.py)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ..config import SPECIALIST_FEATURES_CACHE_DIR  # type: ignore[attr-defined]
except ImportError:
    from ..config import CACHE_DIR
    SPECIALIST_FEATURES_CACHE_DIR = CACHE_DIR / "specialist_features"

try:
    from ..config import SPECIALIST_REFIT_EVERY_BARS  # type: ignore[attr-defined]
except ImportError:
    SPECIALIST_REFIT_EVERY_BARS = 63

SPECIALIST_FEATURES_CACHE_DIR = Path(SPECIALIST_FEATURES_CACHE_DIR)
SPECIALIST_FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Slot constants
# ─────────────────────────────────────────────────────────────────────────────

N_SLOTS = 13
SCHEMA_VERSION = "v1.1"

SLOT_SENTIMENT = 0
SLOT_QUANT_PROB = 1
SLOT_FUND_SCORE = 2
SLOT_REGIME_FLAG = 3
SLOT_MOMENTUM = 4
SLOT_MEAN_REVERSION = 5
SLOT_VOLATILITY = 6
SLOT_EVENT_RISK = 7
SLOT_RELATIVE_STRENGTH = 8
SLOT_LIQUIDITY = 9
SLOT_RET_5D = 10
SLOT_RET_21D = 11
SLOT_VOL_21D = 12

# "specialist" = real computed output; "proxy" = same formula as proxy_specialists.py;
# "specialist_or_proxy" = TFT when data sufficient, else proxy; "real" = directly from returns
SLOT_SOURCES: dict[int, str] = {
    SLOT_SENTIMENT: "proxy",
    SLOT_QUANT_PROB: "specialist_or_proxy",
    SLOT_FUND_SCORE: "proxy",
    SLOT_REGIME_FLAG: "specialist",
    SLOT_MOMENTUM: "specialist",
    SLOT_MEAN_REVERSION: "specialist",
    SLOT_VOLATILITY: "specialist",
    SLOT_EVENT_RISK: "proxy",
    SLOT_RELATIVE_STRENGTH: "specialist",
    SLOT_LIQUIDITY: "specialist",
    SLOT_RET_5D: "real",
    SLOT_RET_21D: "real",
    SLOT_VOL_21D: "real",
}

SLOT_SOURCE_REASONS: dict[int, str] = {
    SLOT_SENTIMENT: (
        "LLM/Ollama unavailable for historical dates; sentiment.py requires live Ollama and "
        "news_scraper.as_of is insufficient to reconstruct point-in-time news for arbitrary bars"
    ),
    SLOT_FUND_SCORE: (
        "yfinance provides only current-snapshot fundamentals; no historical per-date "
        "balance-sheet/income-statement series available via the free API"
    ),
    SLOT_EVENT_RISK: (
        "yfinance earnings calendar is forward-looking; no point-in-time future-earnings "
        "dates are available for arbitrary past bars"
    ),
    SLOT_QUANT_PROB: (
        "TFT requires TFT_MAX_ENCODER_LENGTH+TFT_MAX_PREDICTION_LENGTH+100 bars warmup (~382); "
        "proxy hit-rate fallback used when the available history is smaller"
    ),
}

MOMENTUM_FEATURE_COLS = ["return_5d", "return_20d", "rsi", "atr"]

# Minimum bars before first model refit (per model type)
_VAE_MIN_WARMUP = 200
_LGB_MIN_WARMUP = max(MOMENTUM_WINDOW_DAYS + MOMENTUM_FORWARD_DAYS + 10, 50)

# Mean-reversion agent constants (match agents/mean_reversion.py)
_MR_RSI_PERIOD = 14
_MR_SMA_WINDOW = 20

# ─────────────────────────────────────────────────────────────────────────────
# Cache utilities
# ─────────────────────────────────────────────────────────────────────────────


def _cache_key(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    refit_every: int,
    window: int,
) -> str:
    payload = json.dumps(
        {
            "tickers": sorted(tickers),
            "start_date": start_date,
            "end_date": end_date,
            "schema_version": SCHEMA_VERSION,
            "refit_every": refit_every,
            "window": window,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _save_cache(
    key: str,
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    metadata: dict[str, Any],
    elapsed_s: float,
) -> None:
    npz_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}.npz"
    meta_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}_meta.json"
    np.savez_compressed(
        str(npz_path),
        matrix=matrix.astype(np.float32),
        dates=np.array([str(d.date()) for d in dates], dtype="U10"),
        tickers=np.array(tickers),
    )
    full_meta = {
        **metadata,
        "cache_key": key,
        "generation_time_seconds": round(elapsed_s, 1),
        "schema_version": SCHEMA_VERSION,
        "n_days": int(matrix.shape[0]),
        "n_assets": int(matrix.shape[1]),
        "shape": list(matrix.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2, default=str)
    logger.info("Specialist matrix cached: %s (%.1fs)", npz_path, elapsed_s)


def _load_cache(
    key: str,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]] | None:
    npz_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}.npz"
    meta_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=False)
        matrix = data["matrix"].astype(np.float32)
        dates = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
        with open(meta_path) as f:
            metadata = json.load(f)
        logger.info(
            "Loaded specialist matrix from cache %s (shape=%s)", npz_path, matrix.shape
        )
        return matrix, dates, metadata
    except Exception as exc:
        logger.warning("Cache load failed for %s: %s — will regenerate", npz_path, exc)
        return None


def _tft_checkpoint_meta_path(ckpt_path: Path) -> Path:
    return ckpt_path.parent / f"{ckpt_path.stem}_meta.json"


def _save_tft_checkpoint(
    ckpt_path: Path,
    out: np.ndarray,
    last_bar: int,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    refit_every: int,
) -> None:
    """Persist partial TFT walk-forward output plus a JSON sidecar for the web UI."""
    np.savez_compressed(ckpt_path, out=out, last_bar=np.int64(last_bar))
    meta = {
        "kind": "tft_walkforward_checkpoint",
        "tickers": sorted(tickers),
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "refit_every": int(refit_every),
        "last_bar": int(last_bar),
        "n_days": int(out.shape[0]),
        "n_tickers": int(out.shape[1]),
    }
    with open(_tft_checkpoint_meta_path(ckpt_path), "w") as f:
        json.dump(meta, f, indent=2)


def _remove_tft_checkpoint(ckpt_path: Path) -> None:
    try:
        if ckpt_path.exists():
            ckpt_path.unlink()
    except OSError:
        pass
    meta_path = _tft_checkpoint_meta_path(ckpt_path)
    try:
        if meta_path.exists():
            meta_path.unlink()
    except OSError:
        pass


def list_specialist_cache_status() -> dict[str, Any]:
    """Summarize completed caches and partial TFT checkpoints for the web UI."""
    from ..config import SPECIALIST_REFIT_EVERY_BARS  # type: ignore[attr-defined]

    cache_dir = SPECIALIST_FEATURES_CACHE_DIR
    caches: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    seen_ckpt_ids: set[str] = set()

    for meta_path in sorted(cache_dir.glob("*_meta.json")):
        stem = meta_path.name.replace("_meta.json", "")
        if stem.startswith("tft_wf_ckpt_"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read checkpoint meta %s: %s", meta_path, exc)
                continue
            npz_path = cache_dir / f"{stem}.npz"
            last_bar = int(meta.get("last_bar", 0))
            n_days = int(meta.get("n_days", 0))
            progress = (last_bar / n_days) if n_days > 0 else 0.0
            checkpoints.append(
                {
                    "checkpoint_id": stem,
                    "tickers": meta.get("tickers") or [],
                    "start_date": meta.get("start_date"),
                    "end_date": meta.get("end_date"),
                    "refit_every": meta.get("refit_every"),
                    "last_bar": last_bar,
                    "n_days": n_days,
                    "n_tickers": int(meta.get("n_tickers", 0)),
                    "progress_pct": round(min(1.0, max(0.0, progress)), 4),
                    "size_bytes": int(npz_path.stat().st_size) if npz_path.exists() else 0,
                    "mtime": float(npz_path.stat().st_mtime) if npz_path.exists() else float(meta_path.stat().st_mtime),
                }
            )
            seen_ckpt_ids.add(stem)
            continue

        npz_path = cache_dir / f"{stem}.npz"
        if not npz_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read cache meta %s: %s", meta_path, exc)
            continue
        tickers_raw = meta.get("tickers", [])
        tickers_list = [str(t).upper() for t in tickers_raw]
        caches.append(
            {
                "cache_key": stem,
                "start_date": str(meta.get("start_date", "")),
                "end_date": str(meta.get("end_date", "")),
                "tickers": tickers_list,
                "n_days": int(meta.get("n_days", 0)),
                "n_tickers": int(meta.get("n_assets", meta.get("n_tickers", 0))),
                "refit_every": int(meta.get("refit_every", 0)),
                "window": int(meta.get("window", 21)),
                "size_bytes": int(npz_path.stat().st_size),
                "mtime": float(npz_path.stat().st_mtime),
                "generation_seconds": meta.get("generation_time_seconds"),
            }
        )

    for npz_path in sorted(cache_dir.glob("tft_wf_ckpt_*.npz")):
        stem = npz_path.stem
        if stem in seen_ckpt_ids:
            continue
        try:
            ckpt = np.load(npz_path)
            last_bar = int(ckpt["last_bar"])
            out = ckpt["out"]
            n_days, n_tickers = int(out.shape[0]), int(out.shape[1])
            progress = (last_bar / n_days) if n_days > 0 else 0.0
            checkpoints.append(
                {
                    "checkpoint_id": stem,
                    "tickers": [],
                    "start_date": None,
                    "end_date": None,
                    "refit_every": None,
                    "last_bar": last_bar,
                    "n_days": n_days,
                    "n_tickers": n_tickers,
                    "progress_pct": round(min(1.0, max(0.0, progress)), 4),
                    "size_bytes": int(npz_path.stat().st_size),
                    "mtime": float(npz_path.stat().st_mtime),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read orphan checkpoint %s: %s", npz_path, exc)

    caches.sort(key=lambda c: c.get("mtime", 0), reverse=True)
    checkpoints.sort(key=lambda c: c.get("mtime", 0), reverse=True)
    return {
        "cache_dir": str(cache_dir),
        "default_refit_every": int(SPECIALIST_REFIT_EVERY_BARS),
        "caches": caches,
        "checkpoints": checkpoints,
    }


def delete_specialist_cache(cache_key: str) -> bool:
    """Remove a completed specialist matrix cache (.npz + _meta.json)."""
    key = cache_key.strip()
    if not key or "/" in key or "\\" in key:
        return False
    npz_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}.npz"
    meta_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}_meta.json"
    removed = False
    for path in (npz_path, meta_path):
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)
    return removed


def delete_tft_checkpoint(checkpoint_id: str) -> bool:
    """Remove a partial TFT walk-forward checkpoint (.npz + sidecar meta)."""
    cid = checkpoint_id.strip()
    if not cid.startswith("tft_wf_ckpt_") or "/" in cid or "\\" in cid:
        return False
    ckpt_path = SPECIALIST_FEATURES_CACHE_DIR / f"{cid}.npz"
    if not ckpt_path.exists():
        return False
    _remove_tft_checkpoint(ckpt_path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based specialist functions (vectorized; shift(1) ensures no-lookahead)
# ─────────────────────────────────────────────────────────────────────────────


def _rsi_df(close: pd.DataFrame, period: int = _MR_RSI_PERIOD) -> pd.DataFrame:
    """Compute RSI for all assets at once. Returns DataFrame in [0, 100]."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - 100 / (1 + rs)).clip(0, 100)
    return rsi.astype(np.float32)


def _mean_reversion_df(close: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-reversion score per asset; mirrors agents/mean_reversion.py.
    Applies shift(1) so row t uses data up to bar t-1.
    Returns values in [-1, 1].
    """
    rsi = _rsi_df(close, period=_MR_RSI_PERIOD)
    sma = close.rolling(_MR_SMA_WINDOW, min_periods=1).mean()
    std_price = close.rolling(_MR_SMA_WINDOW, min_periods=2).std().fillna(1e-10)
    std_price = std_price.replace(0, 1e-10)
    z_score = (close - sma) / std_price
    z_term = np.clip(-z_score * 0.3, -0.3, 0.3)
    rsi_term = (50.0 - rsi) / 50.0
    score = np.clip(rsi_term + z_term, -1.0, 1.0)
    # Shift so features at t use data up to t-1
    return score.shift(1).fillna(0.0).astype(np.float32)


def _volatility_df(close: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility score (short/long vol ratio); mirrors agents/volatility.py.
    Returns values in [0, 1]. Applies shift(1) for no-lookahead.
    """
    rets = close.pct_change().fillna(0.0)
    vol_short = rets.rolling(VOLATILITY_WINDOW_DAYS, min_periods=5).std().fillna(1e-10)
    vol_long = rets.rolling(
        VOLATILITY_REF_WINDOW_DAYS, min_periods=VOLATILITY_WINDOW_DAYS
    ).std().fillna(1e-10)
    ratio = (vol_short / vol_long.replace(0, 1e-10)).clip(0.0, 1.0)
    score = ratio.fillna(0.5)
    return score.shift(1).fillna(0.5).astype(np.float32)


def _relative_strength_df(
    close: pd.DataFrame, bench_close: pd.Series
) -> pd.DataFrame:
    """
    Relative strength vs benchmark; mirrors agents/relative_strength.py.
    Returns values in [-1, 1]. Applies shift(1) for no-lookahead.
    """
    ret_20_t = close.pct_change(20).shift(1)
    ret_60_t = close.pct_change(60).shift(1)
    ret_20_b = bench_close.pct_change(20).shift(1)
    ret_60_b = bench_close.pct_change(60).shift(1)
    rel_20 = ret_20_t.subtract(ret_20_b, axis=0)
    rel_60 = ret_60_t.subtract(ret_60_b, axis=0)
    avg_rel = 0.5 * (rel_20 + rel_60)
    score = np.clip(avg_rel * 5.0, -1.0, 1.0).fillna(0.0)
    return score.astype(np.float32)


def _liquidity_df(
    volume: pd.DataFrame, ref_window: int = 60
) -> pd.DataFrame:
    """
    Liquidity score from volume percentile; mirrors agents/liquidity.py.
    At each row t, compares volume[t-1] (current) vs volume[t-ref_window-1:t-1] (reference).
    Returns values in [0, 1]. shift(1) applied so row t uses bar t-1 as current.
    """

    def _rolling_pct(col: pd.Series) -> pd.Series:
        # Window includes the current bar plus ref_window prior bars.
        # arr[-1] = current, arr[:-1] = reference
        def _pct(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.5
            cur = arr[-1]
            ref = arr[:-1]
            if cur <= 0 or np.nanmax(ref) <= 0:
                return 0.5
            return float(np.sum(ref <= cur) / len(ref))

        return col.rolling(ref_window + 1, min_periods=5).apply(_pct, raw=True)

    scores = volume.apply(_rolling_pct).fillna(0.5)
    # shift(1): row t uses volume[t-1] as "current" and volume[t-ref_window-1:t-1] as reference
    return scores.shift(1).fillna(0.5).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward refit schedule
# ─────────────────────────────────────────────────────────────────────────────


def _refit_indices(n_days: int, min_warmup: int, refit_every: int) -> list[int]:
    """Return sorted list of bar indices where a model should be refit."""
    idxs: list[int] = []
    idx = min_warmup
    while idx < n_days:
        idxs.append(idx)
        idx += refit_every
    return idxs


# ─────────────────────────────────────────────────────────────────────────────
# Regime VAE walk-forward
# ─────────────────────────────────────────────────────────────────────────────


def _walkforward_regime_flag(
    spy_ohlcv: pd.DataFrame,
    dates: pd.DatetimeIndex,
    refit_every: int,
) -> np.ndarray:
    """
    Walk-forward regime flag using SPY VAE+KMeans.
    Returns (n_days,) float32 in [0, 1]; 0=Bull, 0.5=Sideways, 0.9=Crash.
    Falls back to neutral 0.5 where insufficient data.
    """
    try:
        import torch
        import torch.nn as nn
        from sklearn.cluster import KMeans

        from ..agents.regime import VAE, _assign_regime_labels
    except ImportError as exc:
        logger.warning("Regime VAE unavailable: %s — using proxy 0.5", exc)
        return np.full(len(dates), 0.5, dtype=np.float32)

    n_days = len(dates)
    flags = np.full(n_days, 0.5, dtype=np.float32)

    def _get_device() -> str:
        return "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"

    device = _get_device()
    DEFENSIVE_REGIMES = {"Bear", "Crash"}

    # Align SPY to dates, forward-fill gaps
    spy_aligned = spy_ohlcv.reindex(dates, method="ffill")
    if spy_aligned.empty or len(spy_aligned.dropna()) < _VAE_MIN_WARMUP:
        logger.warning("Insufficient SPY data for regime VAE; using proxy 0.5")
        return flags

    # Precompute regime features for the entire date range.
    # Each row uses data up to that row (rolling stats), then shift(1) for no-lookahead.
    all_feats_raw = prepare_regime_features(spy_aligned)
    if all_feats_raw.empty:
        return flags
    # Reindex to dates so alignment is consistent
    all_feats = all_feats_raw.reindex(dates).shift(1).bfill().ffill().fillna(0.0)

    refits = _refit_indices(n_days, _VAE_MIN_WARMUP, refit_every)
    if not refits:
        logger.info("Regime VAE: no refit windows in %d bars (min_warmup=%d)", n_days, _VAE_MIN_WARMUP)
        return flags

    trained_windows: list[tuple[int, int, Any, Any, np.ndarray, np.ndarray, list[str]]] = []

    for i, r_idx in enumerate(refits):
        # Training data: regime features from dates[0..r_idx-1]
        train_feats = all_feats_raw.reindex(dates[:r_idx]).dropna()
        if len(train_feats) < _VAE_MIN_WARMUP:
            logger.debug("Skipping VAE refit at bar %d: only %d rows", r_idx, len(train_feats))
            continue

        raw = train_feats.values.astype(np.float64)
        s_mean = np.nanmean(raw, axis=0)
        s_std = np.nanstd(raw, axis=0)
        s_std[s_std < 1e-10] = 1.0
        X = np.nan_to_num((raw - s_mean) / s_std, nan=0.0)

        _bs = 64
        drop_last = _bs > 1 and len(X) > _bs
        loader = torch.utils.data.DataLoader(
            torch.FloatTensor(X), batch_size=_bs, shuffle=True, drop_last=drop_last
        )
        n_feat = X.shape[1]
        vae = VAE(n_feat, VAE_LATENT_DIM, VAE_HIDDEN_DIMS).to(device)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        for _ in range(30):
            vae.train()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                recon, mu_b, logvar_b = vae(batch)
                mse = nn.functional.mse_loss(recon, batch, reduction="mean")
                kld = -0.5 * torch.mean(1 + logvar_b - mu_b.pow(2) - logvar_b.exp())
                (mse + 0.1 * kld).backward()
                opt.step()

        vae.eval()
        with torch.no_grad():
            _, mu_all, _ = vae(torch.FloatTensor(X).to(device))
            z_all = mu_all.cpu().numpy()

        km = KMeans(n_clusters=VAE_REGIME_CLUSTERS, random_state=42, n_init=10)
        labels = km.fit_predict(z_all)
        regime_labels = _assign_regime_labels(raw, labels, VAE_REGIME_CLUSTERS)

        next_r = refits[i + 1] if i + 1 < len(refits) else n_days
        trained_windows.append((r_idx, next_r, vae, km, s_mean, s_std, regime_labels))
        logger.info("Regime VAE refit at bar %d/%d (window [%d, %d))", r_idx, n_days, r_idx, next_r)
        _emit_progress("regime_vae", 0.05 + 0.20 * (r_idx / max(n_days, 1)), f"bar {r_idx}/{n_days}")

    if not trained_windows:
        logger.warning("No regime VAE windows trained; using proxy 0.5")
        return flags

    for (start_i, end_i, vae, km, s_mean, s_std, regime_labels) in trained_windows:
        vae.eval()
        window_feats = all_feats.iloc[start_i:end_i]
        n_feat = len(s_mean)
        X_inf = window_feats.values[:, :n_feat].astype(np.float64)
        if X_inf.shape[1] < n_feat:
            X_inf = np.hstack([X_inf, np.zeros((len(X_inf), n_feat - X_inf.shape[1]))])
        X_inf = np.nan_to_num((X_inf - s_mean) / s_std, nan=0.0)

        with torch.no_grad():
            t_inf = torch.FloatTensor(X_inf).to(next(vae.parameters()).device)
            recon_inf, mu_inf, _ = vae(t_inf)
            recon_errs = torch.mean((recon_inf - t_inf) ** 2, dim=1).cpu().numpy()
            z_inf = mu_inf.cpu().numpy()

        clusters = km.predict(z_inf)
        for j in range(len(clusters)):
            regime = regime_labels[clusters[j]] if clusters[j] < len(regime_labels) else "Sideways"
            if regime in DEFENSIVE_REGIMES:
                flag = 0.9 if regime == "Crash" else 0.7
            elif regime == "Bull":
                flag = 0.0
            else:
                flag = 0.5
            if float(recon_errs[j]) > 0.5:
                flag = min(1.0, flag + 0.2)
            flags[start_i + j] = float(np.clip(flag, 0.0, 1.0))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Momentum LightGBM walk-forward
# ─────────────────────────────────────────────────────────────────────────────


def _fit_momentum_model(
    ohlcv_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    cutoff_date: pd.Timestamp,
) -> tuple[Any, Any, Any] | None:
    """Fit LightGBM momentum on OHLCV data up to cutoff_date. Returns (model, imputer, scaler) or None."""
    try:
        import lightgbm as lgb
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        logger.warning("LightGBM unavailable: %s", exc)
        return None

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for ticker in tickers:
        df = ohlcv_dict.get(ticker)
        if df is None or df.empty:
            continue
        df_slice = df[df.index <= cutoff_date].copy()
        if len(df_slice) < MOMENTUM_WINDOW_DAYS + MOMENTUM_FORWARD_DAYS + 10:
            continue
        df_feat = add_momentum_features(df_slice, window=20)
        if not all(c in df_feat.columns for c in MOMENTUM_FEATURE_COLS):
            continue
        df_feat["fwd_ret"] = df_feat["close"].shift(-MOMENTUM_FORWARD_DAYS) / df_feat["close"] - 1
        df_feat = df_feat.dropna(subset=["fwd_ret"] + MOMENTUM_FEATURE_COLS)
        if len(df_feat) < 30:
            continue
        n = len(df_feat)
        train_end = max(0, int(n * 0.8) - MOMENTUM_FORWARD_DAYS)
        if train_end < 20:
            continue
        X_parts.append(df_feat.iloc[:train_end][MOMENTUM_FEATURE_COLS].values.astype(np.float32))
        y_parts.append(np.clip(df_feat.iloc[:train_end]["fwd_ret"].values, -1.0, 1.0).astype(np.float32))

    if not X_parts or sum(len(x) for x in X_parts) < 30:
        return None

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=MOMENTUM_N_ESTIMATORS,
        max_depth=MOMENTUM_MAX_DEPTH,
        learning_rate=MOMENTUM_LEARNING_RATE,
        verbosity=-1,
        force_col_wise=True,
        n_jobs=-1,
    )
    model.fit(X_sc, y_all)
    return model, imputer, scaler


def _walkforward_momentum_scores(
    ohlcv_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: pd.DatetimeIndex,
    refit_every: int,
) -> np.ndarray:
    """
    Walk-forward LightGBM momentum scores.
    Returns (n_days, n_assets) float32 in [-1, 1].
    Proxy fallback (0.0) wherever model unavailable.
    """
    n_days = len(dates)
    n_assets = len(tickers)
    out = np.zeros((n_days, n_assets), dtype=np.float32)

    # Precompute rolling momentum features for all tickers; shift(1) for no-lookahead
    feat_cache: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = ohlcv_dict.get(ticker)
        if df is None or df.empty:
            continue
        df_feat = add_momentum_features(df.copy(), window=20)
        # shift(1): features at bar t use data up to bar t-1
        feat_cache[ticker] = (
            df_feat[MOMENTUM_FEATURE_COLS]
            .shift(1)
            .reindex(dates)
            .fillna(0.0)
        )

    refits = _refit_indices(n_days, _LGB_MIN_WARMUP, refit_every)
    if not refits:
        logger.info("Momentum LGB: no refit windows (min_warmup=%d)", _LGB_MIN_WARMUP)
        return out

    for i, r_idx in enumerate(refits):
        cutoff = dates[r_idx - 1]
        result = _fit_momentum_model(ohlcv_dict, tickers, cutoff)
        if result is None:
            logger.debug("Momentum model fit failed at bar %d/%d", r_idx, n_days)
            continue
        model, imputer, scaler = result
        next_r = refits[i + 1] if i + 1 < len(refits) else n_days
        logger.info("Momentum LGB refit at bar %d/%d (window [%d, %d))", r_idx, n_days, r_idx, next_r)
        _emit_progress("momentum_lgb", 0.25 + 0.10 * (r_idx / max(n_days, 1)), f"bar {r_idx}/{n_days}")

        # Batch inference for all tickers in window [r_idx, next_r)
        X_stack = np.stack(
            [
                feat_cache.get(t, pd.DataFrame(0.0, index=dates, columns=MOMENTUM_FEATURE_COLS))
                .iloc[r_idx:next_r]
                .values
                for t in tickers
            ],
            axis=0,
        ).astype(np.float32)  # (n_assets, n_window_bars, n_features)

        n_a, n_b, n_f = X_stack.shape
        if n_b == 0:
            continue
        X_flat = X_stack.reshape(n_a * n_b, n_f)
        X_imp = imputer.transform(np.nan_to_num(X_flat, nan=0.0))
        X_sc = scaler.transform(X_imp)
        preds = np.clip(model.predict(X_sc), -1.0, 1.0).astype(np.float32)
        # Reshape: (n_assets, n_bars) -> (n_bars, n_assets)
        out[r_idx:next_r, :] = preds.reshape(n_a, n_b).T

    return out


# ─────────────────────────────────────────────────────────────────────────────
# TFT quant_prob walk-forward (best-effort; proxy fallback when insufficient data)
# ─────────────────────────────────────────────────────────────────────────────


def _tft_checkpoint_path(dates: pd.DatetimeIndex, tickers: list[str], refit_every: int) -> Path:
    """Deterministic path for a TFT walk-forward checkpoint .npz file."""
    raw = f"tft_ckpt:{sorted(tickers)}:{str(dates[0].date())}:{str(dates[-1].date())}:{refit_every}"
    h = hashlib.md5(raw.encode()).hexdigest()[:16]
    return SPECIALIST_FEATURES_CACHE_DIR / f"tft_wf_ckpt_{h}.npz"


def _walkforward_tft_scores(
    ohlcv_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: pd.DatetimeIndex,
    refit_every: int,
) -> np.ndarray | None:
    """
    Walk-forward TFT quant_prob scores.
    Returns (n_days, n_assets) float32 in [0, 1], or None to trigger proxy fallback.
    Requires at least TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 100 warmup bars.
    """
    try:
        import math
        import tempfile

        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping
        from lightning.pytorch.loggers import CSVLogger

        from ..config import (
            TFT_ATTENTION_HEAD_SIZE,
            TFT_DROPOUT,
            TFT_HIDDEN_SIZE,
            TFT_LSTM_LAYERS,
            TFT_MAX_ENCODER_LENGTH,
            TFT_MAX_PREDICTION_LENGTH,
        )
        from ..data.processor import prepare_tft_data, prepare_tft_data_inference
    except ImportError as exc:
        logger.warning("TFT walkforward unavailable: %s — using proxy for quant_prob", exc)
        return None

    tft_min_warmup = TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 100
    n_days = len(dates)
    n_assets = len(tickers)
    out = np.full((n_days, n_assets), 0.5, dtype=np.float32)

    refits = _refit_indices(n_days, tft_min_warmup, refit_every)
    if not refits:
        logger.info(
            "TFT quant_prob: no refit windows (%d bars < min_warmup %d) — using proxy fallback",
            n_days,
            tft_min_warmup,
        )
        return None

    # ── Checkpoint / resume ──────────────────────────────────────────────────
    # After each successful refit we save the partial `out` array to a .npz.
    # On restart we reload it and skip windows that are already filled in
    # (detected by the last filled bar being past the window start).
    ckpt_path = _tft_checkpoint_path(dates, tickers, refit_every)
    resume_from_bar: int = 0
    if ckpt_path.exists():
        try:
            ckpt = np.load(ckpt_path)
            ckpt_out = ckpt["out"]
            ckpt_bar = int(ckpt["last_bar"])
            if ckpt_out.shape == (n_days, n_assets):
                out = ckpt_out
                resume_from_bar = ckpt_bar
                logger.info(
                    "TFT checkpoint found — resuming from bar %d/%d (%d/%d refits already done)",
                    resume_from_bar,
                    n_days,
                    sum(1 for r in refits if r <= resume_from_bar),
                    len(refits),
                )
            else:
                logger.warning("TFT checkpoint shape mismatch (%s vs %s) — ignoring", ckpt_out.shape, (n_days, n_assets))
        except Exception as exc:
            logger.warning("TFT checkpoint load failed (%s) — starting fresh", exc)
    # ─────────────────────────────────────────────────────────────────────────

    any_trained = resume_from_bar > 0  # already have data if resuming

    def _add_log_vol(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "volume" in df.columns:
            df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
        return df

    def _fix_time_idx(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["ticker", "time_idx"]).reset_index(drop=True)
        df["time_idx"] = df.groupby("ticker").cumcount()
        return df

    for i, r_idx in enumerate(refits):
        next_r = refits[i + 1] if i + 1 < len(refits) else n_days

        # Skip windows already computed in a prior (interrupted) run
        if r_idx <= resume_from_bar:
            logger.debug("TFT checkpoint skip bar %d (already done)", r_idx)
            continue

        cutoff = dates[r_idx - 1]

        # Build training data sliced to refit cutoff
        train_parts: list[pd.DataFrame] = []
        for ticker in tickers:
            df = ohlcv_dict.get(ticker)
            if df is None or df.empty:
                continue
            df_slice = df[df.index <= cutoff]
            if len(df_slice) < TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 50:
                continue
            prep = prepare_tft_data(df_slice, ticker=ticker)
            if not prep.empty:
                train_parts.append(_add_log_vol(prep))

        if not train_parts:
            logger.debug("TFT: insufficient training data at bar %d", r_idx)
            continue

        combined = _fix_time_idx(pd.concat(train_parts, ignore_index=True))
        candidate_feats = [c for c in ["return_1d", "vol_20d", "log_volume"] if c in combined.columns]
        if not candidate_feats:
            candidate_feats = ["return_1d"]

        try:
            train_ds = TimeSeriesDataSet(
                combined,
                time_idx="time_idx",
                target="forward_return_30d",
                group_ids=["ticker"],
                max_encoder_length=TFT_MAX_ENCODER_LENGTH,
                max_prediction_length=TFT_MAX_PREDICTION_LENGTH,
                static_categoricals=[],
                static_reals=[],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=candidate_feats,
                target_normalizer=GroupNormalizer(groups=["ticker"]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
        except Exception as exc:
            logger.warning("TFT dataset build failed at bar %d: %s", r_idx, exc)
            continue

        train_dl = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
        tft_model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=1e-3,
            hidden_size=TFT_HIDDEN_SIZE,
            attention_head_size=TFT_ATTENTION_HEAD_SIZE,
            dropout=TFT_DROPOUT,
            lstm_layers=TFT_LSTM_LAYERS,
            output_size=7,
            reduce_on_plateau_patience=2,
        )

        with tempfile.TemporaryDirectory(prefix="tft_wf_") as tmpdir:
            trainer = Trainer(
                max_epochs=10,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=CSVLogger(tmpdir, name="tft_wf"),
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[EarlyStopping(monitor="train_loss", patience=3, mode="min")],
            )
            try:
                trainer.fit(tft_model, train_dataloaders=train_dl)
            except Exception as exc:
                logger.warning("TFT training failed at bar %d: %s", r_idx, exc)
                continue

        # Per-ticker inference: build one encoder context per ticker (data up to
        # the refit cutoff), run a single forward pass, and use the last window's
        # first-step prediction as a constant probability score for the whole
        # window [r_idx, next_r).
        #
        # This avoids the two bugs in the previous batch approach:
        #   1. preds[t_i] indexes by flat window order, not by ticker.
        #   2. Date mapping failed on tz-aware yfinance timestamps vs tz-naive
        #      aligned_close index.
        train_tickers = set(combined["ticker"].unique())
        tft_model.eval()

        for ticker in tickers:
            if ticker not in train_tickers:
                continue
            t_col = tickers.index(ticker) if ticker in tickers else -1
            if t_col < 0:
                continue
            df = ohlcv_dict.get(ticker)
            if df is None or df.empty:
                continue
            # Use only data up to (and including) the refit cutoff so the model
            # has no look-ahead into [r_idx, next_r).
            df_enc = df[df.index <= cutoff].tail(
                TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 10
            )
            if len(df_enc) < TFT_MAX_ENCODER_LENGTH + 1:
                continue

            prep = prepare_tft_data_inference(df_enc, ticker=ticker)
            if prep.empty:
                continue
            prep = _add_log_vol(prep)
            prep = _fix_time_idx(prep)

            try:
                inf_ds = TimeSeriesDataSet.from_dataset(
                    train_ds, prep, stop_randomization=True
                )
                inf_dl = inf_ds.to_dataloader(
                    train=False, batch_size=len(inf_ds), num_workers=0
                )
                with torch.no_grad():
                    ticker_preds = tft_model.predict(inf_dl)
                # ticker_preds shape: (n_windows, max_pred_len) or (n_windows,)
                # Use the last window's first-step prediction as the score.
                if ticker_preds.numel() == 0:
                    continue
                if ticker_preds.dim() >= 2:
                    score = float(ticker_preds[-1, 0])
                else:
                    score = float(ticker_preds[-1])
                prob = float(1.0 / (1.0 + math.exp(-20.0 * score)))
                prob = float(np.clip(prob, 0.0, 1.0))
                out[r_idx:next_r, t_col] = prob
            except Exception as exc:
                logger.debug("TFT per-ticker inference failed (%s, bar %d): %s", ticker, r_idx, exc)
                continue

        any_trained = True
        logger.info("TFT refit at bar %d/%d (window [%d, %d))", r_idx, n_days, r_idx, next_r)
        _emit_progress("tft_quant_prob", 0.35 + 0.63 * (r_idx / max(n_days, 1)), f"bar {r_idx}/{n_days}")

        # Save checkpoint so a restart can resume from this window
        try:
            _save_tft_checkpoint(ckpt_path, out, r_idx, tickers, dates, refit_every)
        except OSError as exc:
            logger.warning("TFT checkpoint save failed: %s", exc)

    if not any_trained:
        logger.info("TFT: no successful refits — proxy fallback for quant_prob")
        return None
    # If all predictions are still at the neutral init value, the date mapping failed; use proxy
    if np.allclose(out, 0.5, atol=1e-6):
        logger.info(
            "TFT: trained but all predictions remained at 0.5 (date mapping mismatch) "
            "— proxy fallback for quant_prob"
        )
        return None

    # Full run complete — remove checkpoint file
    _remove_tft_checkpoint(ckpt_path)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────


def precompute_specialist_feature_matrix(
    aligned_close: pd.DataFrame,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    *,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
    aligned_volume: pd.DataFrame | None = None,
    benchmark_close: pd.Series | None = None,
    spy_ohlcv: pd.DataFrame | None = None,
    window: int = 21,
    refit_every: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute walk-forward specialist feature matrix.

    Args:
        aligned_close:   (n_days, n_assets) DataFrame of close prices, columns = tickers.
        tickers:         List of ticker symbols matching aligned_close columns.
        dates:           DatetimeIndex of length n_days for the date axis.
        ohlcv_dict:      Optional dict ticker -> full OHLCV DataFrame. Enables trained models
                         (momentum LightGBM, TFT) and volume-based liquidity. If None, those
                         slots fall back to proxy.
        aligned_volume:  Optional (n_days, n_assets) volume DataFrame. Used for liquidity.
                         Ignored when ohlcv_dict is provided (volume extracted from there).
        benchmark_close: Optional (n_days,) Series for benchmark returns (relative strength).
                         If None, loads RELATIVE_STRENGTH_BENCHMARK from disk.
        spy_ohlcv:       Optional full SPY OHLCV DataFrame for regime VAE.
                         If None, uses aligned_close["SPY"] (if present) or loads SPY.
        window:          Proxy feature lookback in days (default 21; matches proxy path).
        refit_every:     Walk-forward refit cadence in trading days (default from config).

    Returns:
        (matrix, metadata) where:
            matrix   – np.ndarray of shape (n_days, n_assets, 13) float32
            metadata – dict documenting slot_sources, refit_cadence, coverage, etc.
    """
    t0 = time.time()

    if refit_every is None:
        refit_every = int(SPECIALIST_REFIT_EVERY_BARS)

    aligned_close = aligned_close.copy()
    if not isinstance(aligned_close.columns, pd.Index):
        aligned_close.columns = pd.Index(tickers)

    n_days, n_assets = aligned_close.shape
    returns_np = (
        aligned_close.pct_change().fillna(0.0).values.astype(np.float32)
    )

    _quiet_thirdparty_logging()
    _emit_progress("proxy_base", 0.01, f"{n_days} days x {n_assets} assets")

    # Step 1: proxy baseline for all 13 slots using sliding window (no z-score)
    out = np.zeros((n_days, n_assets, N_SLOTS), dtype=np.float32)
    if n_days > window:
        from numpy.lib.stride_tricks import sliding_window_view

        wins = sliding_window_view(returns_np, window_shape=window, axis=0)
        wins = np.transpose(wins, (0, 2, 1))  # (n_days-window+1, n_assets, window)
        batch = wins[: n_days - window]        # (n_days-window, n_assets, window)
        out[window:] = proxy_specialists_vectorized_batch(batch)

    # Track which slots ended up as real vs proxy
    actual_sources = dict(SLOT_SOURCES)

    # Step 2: rule-based slots from OHLCV (override proxy)
    logger.info("Computing rule-based specialist slots (mean_reversion, volatility, rel_strength, liquidity)...")
    _emit_progress("rule_based", 0.03, "mean_reversion / volatility / rel_strength / liquidity")

    close_df = aligned_close.reindex(dates).ffill()
    if close_df.shape[1] != n_assets:
        close_df = aligned_close.copy()
    close_df.columns = tickers

    mr = _mean_reversion_df(close_df)  # (n_days, n_assets)
    out[:, :, SLOT_MEAN_REVERSION] = mr.reindex(dates).fillna(0.0).values

    vol_scores = _volatility_df(close_df)  # (n_days, n_assets)
    out[:, :, SLOT_VOLATILITY] = vol_scores.reindex(dates).fillna(0.5).values

    # Benchmark for relative strength
    bench = benchmark_close
    if bench is None:
        bench_ticker = RELATIVE_STRENGTH_BENCHMARK
        if bench_ticker in tickers:
            bench = close_df[bench_ticker]
        elif spy_ohlcv is not None and "close" in spy_ohlcv.columns:
            bench = spy_ohlcv["close"].reindex(dates, method="ffill")
        else:
            logger.info("Loading benchmark %s for relative strength", bench_ticker)
            bench_df = load_ohlcv(bench_ticker)
            bench = bench_df["close"].reindex(dates, method="ffill") if not bench_df.empty else None

    if bench is not None:
        rs = _relative_strength_df(close_df, bench.reindex(dates).ffill())
        out[:, :, SLOT_RELATIVE_STRENGTH] = rs.fillna(0.0).values
    else:
        logger.warning("No benchmark close for relative strength; using proxy")
        actual_sources[SLOT_RELATIVE_STRENGTH] = "proxy"

    # Liquidity from volume
    vol_df: pd.DataFrame | None = None
    if ohlcv_dict is not None:
        vol_cols = {t: ohlcv_dict[t]["volume"].reindex(dates, method="ffill") for t in tickers if t in ohlcv_dict and "volume" in ohlcv_dict[t].columns}
        if vol_cols:
            vol_df = pd.DataFrame(vol_cols)[tickers]
    elif aligned_volume is not None:
        vol_df = aligned_volume.copy()
        vol_df.columns = tickers

    if vol_df is not None and not vol_df.empty:
        liq = _liquidity_df(vol_df)  # (n_days, n_assets)
        out[:, :, SLOT_LIQUIDITY] = liq.reindex(dates).fillna(0.5).values
    else:
        logger.info("No volume data for liquidity; slot uses proxy (0.5 neutral)")
        actual_sources[SLOT_LIQUIDITY] = "proxy"

    # Step 3: regime VAE walk-forward
    logger.info("Running walk-forward regime VAE...")
    _spy_ohlcv = spy_ohlcv
    if _spy_ohlcv is None:
        if "SPY" in (ohlcv_dict or {}):
            _spy_ohlcv = ohlcv_dict["SPY"]
        else:
            spy_loaded = load_ohlcv("SPY")
            _spy_ohlcv = spy_loaded if not spy_loaded.empty else None

    regime_flags = _walkforward_regime_flag(_spy_ohlcv, dates, refit_every)  # (n_days,)
    out[:, :, SLOT_REGIME_FLAG] = regime_flags[:, np.newaxis]  # broadcast to all assets

    # Step 4: momentum LightGBM walk-forward
    if ohlcv_dict is not None:
        logger.info("Running walk-forward momentum LightGBM...")
        mom_scores = _walkforward_momentum_scores(ohlcv_dict, tickers, dates, refit_every)
        out[:, :, SLOT_MOMENTUM] = mom_scores
    else:
        logger.info("No ohlcv_dict provided; momentum uses proxy fallback")
        actual_sources[SLOT_MOMENTUM] = "proxy"

    # Step 5: TFT quant_prob walk-forward (best-effort)
    if ohlcv_dict is not None:
        logger.info("Attempting walk-forward TFT quant_prob...")
        tft_result = _walkforward_tft_scores(ohlcv_dict, tickers, dates, refit_every)
        if tft_result is not None:
            out[:, :, SLOT_QUANT_PROB] = tft_result
            actual_sources[SLOT_QUANT_PROB] = "specialist"
        else:
            actual_sources[SLOT_QUANT_PROB] = "proxy"
    else:
        actual_sources[SLOT_QUANT_PROB] = "proxy"

    # Sanity clamp: ensure all slots stay within documented ranges
    out[:, :, [SLOT_SENTIMENT, SLOT_FUND_SCORE, SLOT_MEAN_REVERSION,
               SLOT_RELATIVE_STRENGTH, SLOT_MOMENTUM]] = np.clip(
        out[:, :, [SLOT_SENTIMENT, SLOT_FUND_SCORE, SLOT_MEAN_REVERSION,
                   SLOT_RELATIVE_STRENGTH, SLOT_MOMENTUM]], -1.0, 1.0
    )
    out[:, :, [SLOT_QUANT_PROB, SLOT_REGIME_FLAG, SLOT_VOLATILITY,
               SLOT_EVENT_RISK, SLOT_LIQUIDITY]] = np.clip(
        out[:, :, [SLOT_QUANT_PROB, SLOT_REGIME_FLAG, SLOT_VOLATILITY,
                   SLOT_EVENT_RISK, SLOT_LIQUIDITY]], 0.0, 1.0
    )

    elapsed = time.time() - t0
    logger.info(
        "Specialist matrix built in %.1fs | shape=%s | real slots: %s",
        elapsed,
        out.shape,
        sorted(k for k, v in actual_sources.items() if v != "proxy"),
    )

    metadata: dict[str, Any] = {
        "tickers": tickers,
        "n_days": n_days,
        "n_assets": n_assets,
        "window": window,
        "refit_every": refit_every,
        "slot_sources": {str(k): v for k, v in actual_sources.items()},
        "slot_source_reasons": {str(k): v for k, v in SLOT_SOURCE_REASONS.items()},
        "elapsed_seconds": round(elapsed, 1),
    }
    return out.astype(np.float32), metadata


# ─────────────────────────────────────────────────────────────────────────────
# Cache-aware entry point
# ─────────────────────────────────────────────────────────────────────────────


def _slice_from_superset_cache(
    tickers: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    refit_every: int,
    window: int,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]] | None:
    """
    Fallback cache lookup: find any cached matrix with identical tickers,
    schema_version, refit_every, and window whose date range covers the
    requested [start_ts, end_ts] window, and slice the rows out.

    Valid because rows are walk-forward (row t uses only data from bars < t),
    so a slice of a superset matrix is identical to a fresh build over the
    sub-range with the same warmup history.
    """
    want_tickers = sorted(t.upper() for t in tickers)
    for meta_path in sorted(SPECIALIST_FEATURES_CACHE_DIR.glob("*_meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("schema_version") != SCHEMA_VERSION:
                continue
            if int(meta.get("refit_every", -1)) != refit_every:
                continue
            if int(meta.get("window", -1)) != window:
                continue
            c_start = pd.Timestamp(meta.get("start_date", "2200-01-01"))
            c_end = pd.Timestamp(meta.get("end_date", "1900-01-01"))
            if c_start > start_ts or c_end < end_ts:
                continue
            key = meta.get("cache_key") or meta_path.name.replace("_meta.json", "")
            npz_path = SPECIALIST_FEATURES_CACHE_DIR / f"{key}.npz"
            if not npz_path.exists():
                continue
            data = np.load(str(npz_path), allow_pickle=False)
            cached_tickers = sorted(str(t).upper() for t in data["tickers"])
            if cached_tickers != want_tickers:
                continue
            dates = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
            mask = (dates >= start_ts) & (dates <= end_ts)
            if not mask.any():
                continue
            matrix = data["matrix"].astype(np.float32)[mask]
            sliced_meta = dict(meta)
            sliced_meta["sliced_from_cache_key"] = key
            sliced_meta["start_date"] = str(start_ts.date())
            sliced_meta["end_date"] = str(end_ts.date())
            logger.info(
                "Sliced specialist matrix from superset cache %s: %d rows (%s to %s)",
                npz_path.name, int(mask.sum()), start_ts.date(), end_ts.date(),
            )
            return matrix, dates[mask], sliced_meta
        except Exception as exc:  # noqa: BLE001
            logger.warning("Superset cache check failed for %s: %s", meta_path, exc)
    return None


def get_or_build_specialist_matrix(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    window: int = 21,
    refit_every: int | None = None,
    force_rebuild: bool = False,
    benchmark: str | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    """
    Cache-aware entry point for the specialist feature matrix.

    Loads aligned OHLCV data for all tickers + SPY benchmark, then calls
    precompute_specialist_feature_matrix. Results are cached under
    SPECIALIST_FEATURES_CACHE_DIR and reused on subsequent calls with identical
    (tickers, start, end, refit_every, window) parameters.

    Args:
        tickers:       List of tickers (matched to TRADE_TICKERS in training).
        start:         Start date string or Timestamp (inclusive).
        end:           End date string or Timestamp (inclusive).
        window:        Proxy feature lookback (default 21).
        refit_every:   Walk-forward refit cadence; uses SPECIALIST_REFIT_EVERY_BARS if None.
        force_rebuild: Skip cache and regenerate even if a compatible cache exists.
        benchmark:     Benchmark ticker for relative strength (default RELATIVE_STRENGTH_BENCHMARK).

    Returns:
        (matrix, dates, metadata)
            matrix  – (n_days, n_assets, 13) float32 ready for HistoricalTradingEnv
            dates   – pd.DatetimeIndex aligned to matrix rows
            metadata – dict with slot_sources, generation info, etc.
    """
    if refit_every is None:
        refit_every = int(SPECIALIST_REFIT_EVERY_BARS)

    tickers_upper = [t.strip().upper() for t in tickers]
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    start_str = str(start_ts.date())
    end_str = str(end_ts.date())

    key = _cache_key(tuple(tickers_upper), start_str, end_str, refit_every, window)

    if not force_rebuild:
        cached = _load_cache(key)
        if cached is not None:
            return cached
        sliced = _slice_from_superset_cache(
            tickers_upper, start_ts, end_ts, refit_every, window
        )
        if sliced is not None:
            return sliced

    logger.info(
        "Generating specialist matrix: tickers=%s start=%s end=%s refit_every=%d",
        tickers_upper, start_str, end_str, refit_every,
    )

    t_gen_start = time.time()

    bench_ticker = (benchmark or RELATIVE_STRENGTH_BENCHMARK).strip().upper()

    # How many extra bars to load before start_ts for warmup (VAE needs 200+, TFT needs 382+)
    _extra_days = 400  # calendar days to load before start for model warmup
    load_start = start_ts - pd.Timedelta(days=_extra_days)

    # Load OHLCV for all tickers + SPY for regime
    ohlcv_dict: dict[str, pd.DataFrame] = {}
    all_load_tickers = list(dict.fromkeys(tickers_upper + ["SPY", bench_ticker]))
    for ticker in all_load_tickers:
        df = load_ohlcv(ticker, start=load_start, end=end_ts + pd.Timedelta(days=1))
        if not df.empty:
            idx = pd.to_datetime(df.index)
            # Strip timezone so comparisons with tz-naive start/end timestamps work
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df.index = idx.normalize()
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            ohlcv_dict[ticker] = df

    if not any(t in ohlcv_dict for t in tickers_upper):
        raise RuntimeError(f"No OHLCV data loaded for any of {tickers_upper}")

    # Build aligned close matrix including warmup bars BEFORE start_ts so that:
    #   - rolling stats (vol, MR, RS, liquidity) have enough history
    #   - walk-forward trained models have training data before the output window
    close_frames = {t: ohlcv_dict[t]["close"] for t in tickers_upper if t in ohlcv_dict}
    aligned_full = pd.DataFrame(close_frames).sort_index()
    aligned_full = aligned_full.dropna(how="all").ffill().bfill()
    dates_full = pd.DatetimeIndex(aligned_full.index)

    if len(dates_full) < window + 10:
        raise RuntimeError(
            f"Only {len(dates_full)} trading days in loaded range; "
            f"need at least {window + 10}"
        )

    bench_close: pd.Series | None = None
    if bench_ticker in ohlcv_dict:
        bench_close = ohlcv_dict[bench_ticker]["close"].reindex(dates_full, method="ffill")

    spy_ohlcv = ohlcv_dict.get("SPY")

    # Compute on full extended range (includes warmup before start_ts)
    matrix_full, metadata = precompute_specialist_feature_matrix(
        aligned_close=aligned_full,
        tickers=tickers_upper,
        dates=dates_full,
        ohlcv_dict=ohlcv_dict,
        benchmark_close=bench_close,
        spy_ohlcv=spy_ohlcv,
        window=window,
        refit_every=refit_every,
    )

    # Slice output to the requested [start_ts, end_ts] window
    out_mask = (dates_full >= start_ts) & (dates_full <= end_ts)
    matrix = matrix_full[out_mask]
    dates = dates_full[out_mask]

    if len(dates) < 1:
        raise RuntimeError(
            f"No trading days in requested window [{start_str}, {end_str}] "
            f"after slicing extended range [{dates_full[0].date()}, {dates_full[-1].date()}]"
        )

    elapsed = time.time() - t_gen_start
    metadata["start_date"] = start_str
    metadata["end_date"] = end_str
    metadata["cache_key"] = key
    metadata["warmup_bars"] = int(out_mask.sum() - int(np.sum(~out_mask[:int(np.argmax(out_mask))])))

    _save_cache(key, matrix, dates, tickers_upper, metadata, elapsed)
    _emit_progress("done", 1.0, f"cached {matrix.shape[0]} days in {elapsed:.0f}s")

    return matrix, dates, metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI pre-generation entry point
# ─────────────────────────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate specialist feature matrix cache for Mode B PPO training."
    )
    parser.add_argument(
        "--tickers",
        default="SPY,IWM,XLE,XLF,EEM",
        help="Comma-separated ticker list (default: SPY,IWM,XLE,XLF,EEM)",
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=int(SPECIALIST_REFIT_EVERY_BARS),
        help=f"Walk-forward refit cadence in trading days (default: {SPECIALIST_REFIT_EVERY_BARS})",
    )
    parser.add_argument("--window", type=int, default=21, help="Proxy lookback window (default: 21)")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Regenerate even if a compatible cache exists",
    )
    parser.add_argument(
        "--benchmark",
        default=RELATIVE_STRENGTH_BENCHMARK,
        help=f"Benchmark ticker for relative strength (default: {RELATIVE_STRENGTH_BENCHMARK})",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logger.info(
        "Specialist feature pre-generation: %d tickers %s -> %s refit_every=%d",
        len(tickers), args.start, args.end, args.refit_every,
    )

    t0 = time.time()
    matrix, dates, meta = get_or_build_specialist_matrix(
        tickers=tickers,
        start=args.start,
        end=args.end,
        window=args.window,
        refit_every=args.refit_every,
        force_rebuild=args.force_rebuild,
        benchmark=args.benchmark,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | matrix shape: {matrix.shape}")
    print(f"Date range: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} trading days)")
    print("Slot sources (real vs proxy):")
    for slot_idx, source in sorted(meta.get("slot_sources", {}).items(), key=lambda x: int(x[0])):
        slot_name = [
            "sentiment", "quant_prob", "fund_score", "regime_flag", "momentum_score",
            "mean_reversion", "volatility", "event_risk", "relative_strength", "liquidity",
            "ret_5d_norm", "ret_21d_norm", "vol_21d_norm",
        ][int(slot_idx)]
        print(f"  Slot {int(slot_idx):2d} {slot_name:<20s} -> {source}")
    cache_dir = SPECIALIST_FEATURES_CACHE_DIR
    print(f"Cache: {cache_dir}/{meta.get('cache_key', '?')}.npz")


if __name__ == "__main__":
    _main()
