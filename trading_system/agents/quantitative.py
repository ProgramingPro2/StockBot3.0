"""
Quantitative Specialist: TFT for trend probability.
"""
import math

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

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
from ..data.processor import prepare_tft_data
from ..utils.logger import get_logger

logger = get_logger()

_model: TemporalFusionTransformer | None = None
_training_dataset: TimeSeriesDataSet | None = None


def _get_device() -> str:
    """Return cuda if available, else cpu."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    """Build TimeSeriesDataSet from prepared OHLCV DataFrame."""
    time_varying_unknown_reals = ["open", "high", "low", "close", "volume", "return_1d", "vol_20d"]
    available = [c for c in time_varying_unknown_reals if c in df.columns]
    if not available:
        available = ["close", "return_1d", "vol_20d"]

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


def _ensure_model(ticker: str | None = None, retrain: bool = False) -> bool:
    """Load or train TFT model."""
    global _model, _training_dataset

    ckpt = QUANTITATIVE_MODEL_DIR / "tft.ckpt"
    if ckpt.exists() and not retrain:
        try:
            _model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
            _model.to(_get_device())
            return True
        except Exception as e:
            logger.warning("Failed to load TFT: %s", e)

    # Train on ticker or SPY as proxy
    train_ticker = ticker or "SPY"
    df = load_ohlcv(train_ticker, lookback_years=YFINANCE_LOOKBACK_YEARS)
    if df.empty or len(df) < TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH + 100:
        logger.warning("Insufficient data for TFT training: %s", train_ticker)
        return False

    prepared = prepare_tft_data(df)
    if prepared.empty:
        return False

    _training_dataset = _build_dataset(prepared)
    train_dl = _training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        _training_dataset,
        learning_rate=1e-3,
        hidden_size=TFT_HIDDEN_SIZE,
        lstm_layers=TFT_LSTM_LAYERS,
        dropout=TFT_DROPOUT,
        attention_head_size=TFT_ATTENTION_HEAD_SIZE,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator=_get_device(),
        devices=1,
        logger=CSVLogger(QUANTITATIVE_MODEL_DIR, name="logs"),
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dl)
    QUANTITATIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(ckpt))
    _model = model
    _model.to(_get_device())
    return True


def get_trend_probability(ticker: str, retrain: bool = False) -> float:
    """
    Get probability that trend continues over next 30 days for ticker.
    Returns float in [0, 1].
    """
    if not _ensure_model(ticker, retrain=retrain):
        return 0.5

    df = load_ohlcv(ticker)
    if df.empty or len(df) < TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH:
        return 0.5

    prepared = prepare_tft_data(df)
    if prepared.empty:
        return 0.5

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
    forecast = float(preds[0]) if preds.numel() >= 1 else 0.0
    return _forecast_to_probability(forecast)
