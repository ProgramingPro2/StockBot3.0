"""
Quantitative Specialist: TFT for trend probability.
"""
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from ..config import (
    DEVICE,
    QUANTITATIVE_MODEL_DIR,
    TFT_ATTENTION_HEAD_SIZE,
    TFT_DROPOUT,
    TFT_HIDDEN_SIZE,
    TFT_LSTM_LAYERS,
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_PREDICTION_LENGTH,
    TFT_TARGET,
    YFINANCE_LOOKBACK_YEARS,
)
from ..data.loader import load_ohlcv
from ..data.processor import prepare_tft_data, prepare_tft_data_inference
from ..meta_learner.runs_index import TENSORBOARD_DIR, current_run_id, record_subphase
from ..utils.lightning_progress import EpochProgressCallback
from ..utils.logger import get_logger

logger = get_logger()

_model: TemporalFusionTransformer | None = None
_training_dataset: TimeSeriesDataSet | None = None


def _get_device() -> str:
    """Return cuda if available, else cpu."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-volume and ensure return-based features are present.
    Raw price and volume columns are not added as encoder inputs because
    cross-ticker magnitude differences make them non-comparable.
    """
    df = df.copy()
    if "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    return df


def _build_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    """
    Build TimeSeriesDataSet from prepared OHLCV DataFrame.

    Only return-based and log-volume features are used as encoder inputs so that
    cross-ticker magnitude differences (e.g. $3 stock vs $1000 stock) do not
    inflate or suppress gradients. Raw open/high/low/close/volume are excluded.
    """
    # Use only scale-free features; exclude raw OHLCV prices/volume
    candidate_features = ["return_1d", "vol_20d", "log_volume"]
    available = [c for c in candidate_features if c in df.columns]
    if not available:
        available = ["return_1d"]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TFT_TARGET,
        group_ids=["ticker"],
        max_encoder_length=TFT_MAX_ENCODER_LENGTH,
        max_prediction_length=TFT_MAX_PREDICTION_LENGTH,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=available,
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def _forecast_to_probability(forecast: float, scale: float = 20.0) -> float:
    """Map TFT forecast (30d return) to trend probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-scale * forecast))


def _fix_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reassign time_idx per group as a contiguous 0-based integer sequence.

    prepare_tft_data assigns time_idx before dropna, which can leave gaps when
    rows with NaN targets are removed.  This ensures a monotonically increasing
    index with no gaps within each group, as required by TimeSeriesDataSet.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()
    return df


def _ensure_model(
    ticker: str | list[str] | None = None,
    retrain: bool = False,
    end_date: str | None = None,
) -> bool:
    """Load or train TFT model. ticker: single symbol, list of symbols (train on all), or None for SPY."""
    global _model, _training_dataset

    ckpt = QUANTITATIVE_MODEL_DIR / "tft.ckpt"
    if ckpt.exists() and not retrain:
        try:
            _model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
            _model.to(_get_device())
            return True
        except Exception as e:
            logger.warning("Failed to load TFT: %s", e)

    tickers: list[str] = []
    if isinstance(ticker, list):
        tickers = [t.upper() for t in ticker if t]
    else:
        tickers = [ticker.upper() if ticker else "SPY"]

    if not tickers:
        tickers = ["SPY"]

    parts: list[pd.DataFrame] = []
    for t in tickers:
        df = load_ohlcv(t, lookback_years=YFINANCE_LOOKBACK_YEARS, end=end_date)
        if df.empty or len(df) < TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 100:
            logger.warning("Insufficient data for TFT training: %s", t)
            continue
        prepared = prepare_tft_data(df, ticker=t)
        if not prepared.empty:
            prepared = _add_derived_features(prepared)
            parts.append(prepared)

    if not parts:
        logger.warning("No valid TFT training data for tickers: %s", tickers)
        return False

    combined = pd.concat(parts, ignore_index=True)
    # Ensure monotonically increasing time_idx per group with no gaps after dropna
    combined = _fix_time_idx(combined)

    # Chronological 80/20 train/val split per group
    # Each group's time_idx runs 0..N-1; split at 80th percentile per group.
    min_val_rows = TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH

    def _split_group(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(g)
        split_idx = int(n * 0.8)
        # Val needs encoder context: include trailing encoder-length rows from train
        val_start = max(0, split_idx - TFT_MAX_ENCODER_LENGTH)
        return g.iloc[:split_idx], g.iloc[val_start:]

    train_parts, val_parts = [], []
    for _, grp in combined.groupby("ticker"):
        tr, va = _split_group(grp)
        if len(tr) >= TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH:
            train_parts.append(tr)
        if len(va) >= min_val_rows:
            val_parts.append(va)

    if not train_parts:
        logger.warning("No valid train split for TFT")
        return False

    train_df = pd.concat(train_parts, ignore_index=True)
    train_df = _fix_time_idx(train_df)

    _bs = int(os.getenv("TFT_BATCH_SIZE", "32").strip() or "32")
    _nw = int(os.getenv("TFT_DATALOADER_NUM_WORKERS", "0").strip() or "0")
    _persist = os.getenv("TFT_DATALOADER_PERSISTENT", "").strip().lower() in ("1", "true", "yes")

    _training_dataset = _build_dataset(train_df)
    train_dl = _training_dataset.to_dataloader(
        train=True,
        batch_size=_bs,
        num_workers=_nw,
        persistent_workers=bool(_nw and _persist),
    )

    val_dl = None
    if val_parts:
        try:
            val_df = pd.concat(val_parts, ignore_index=True)
            val_df = _fix_time_idx(val_df)
            val_dataset = TimeSeriesDataSet.from_dataset(_training_dataset, val_df, stop_randomization=True)
            val_dl = val_dataset.to_dataloader(
                train=False,
                batch_size=_bs,
                num_workers=_nw,
                persistent_workers=bool(_nw and _persist),
            )
        except Exception as e:
            logger.warning("Could not build TFT validation dataloader: %s", e)
            val_dl = None

    model = TemporalFusionTransformer.from_dataset(
        _training_dataset,
        learning_rate=1e-3,
        hidden_size=TFT_HIDDEN_SIZE,
        lstm_layers=TFT_LSTM_LAYERS,
        dropout=TFT_DROPOUT,
        attention_head_size=TFT_ATTENTION_HEAD_SIZE,
    )

    run_id = current_run_id() or "manual"
    tb_subdir = f"{run_id}_specialist_quant"
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=str(TENSORBOARD_DIR),
        name="",
        version=tb_subdir,
    )
    try:
        record_subphase(run_id, "specialist_quant", tb_path=str(tb_logger.log_dir))
    except Exception:  # noqa: BLE001
        pass

    _epochs = int(os.getenv("TFT_MAX_EPOCHS", "10").strip() or "10")
    callbacks = [EpochProgressCallback(phase="specialist_quant")]
    if val_dl is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                verbose=False,
            )
        )

    trainer_kw: dict = {
        "max_epochs": _epochs,
        "accelerator": _get_device(),
        "devices": 1,
        "logger": tb_logger,
        "callbacks": callbacks,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    _prec = os.getenv("TFT_PRECISION", "").strip()
    if _prec in ("16-mixed", "16", "bf16-mixed", "32"):
        trainer_kw["precision"] = _prec
    trainer = Trainer(**trainer_kw)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*is an instance of `nn.Module` and is already saved during checkpointing.*",
            module="lightning.pytorch.utilities.parsing",
        )
        fit_kwargs: dict = {"model": model, "train_dataloaders": train_dl}
        if val_dl is not None:
            fit_kwargs["val_dataloaders"] = val_dl
        trainer.fit(**fit_kwargs)
    QUANTITATIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(ckpt))
    _model = model
    _model.to(_get_device())
    return True


def get_trend_probability(ticker: str, retrain: bool = False) -> float:
    """
    Get probability that trend continues over next 30 days for ticker.
    Uses inference-only preparation (no target drop) so the current bar is scored.
    Returns float in [0, 1].
    """
    if not _ensure_model(ticker, retrain=retrain):
        return 0.5

    t = ticker.upper()
    df = load_ohlcv(ticker)
    if df.empty or len(df) < TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH:
        return 0.5

    prepared = prepare_tft_data_inference(df, ticker=t)
    if prepared.empty:
        return 0.5

    prepared = _add_derived_features(prepared)

    if _training_dataset is not None:
        try:
            dataset = TimeSeriesDataSet.from_dataset(_training_dataset, prepared)
        except Exception:
            dataset = _build_dataset(prepared)
    else:
        dataset = _build_dataset(prepared)
    dl = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    _model.eval()
    with torch.no_grad():
        preds = _model.predict(dl)
    # preds: (n_samples, prediction_length) or (n_samples,) — use last row, first horizon step
    if preds is None or preds.numel() == 0:
        forecast = 0.0
    elif preds.dim() == 2:
        forecast = float(preds[-1, 0])
    elif preds.dim() == 1:
        forecast = float(preds[-1])
    else:
        forecast = float(preds.flatten()[-1])

    if not np.isfinite(forecast):
        return 0.5
    return float(np.clip(_forecast_to_probability(forecast), 0.0, 1.0))
