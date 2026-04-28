"""Read/write the project's .env and expose static config values."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .. import env_manager
from ..paths import ENV_FILE
from ..schemas import EnvDocument, EnvUpdateRequest, EnvVar, StaticConfig

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/env", response_model=EnvDocument)
def get_env(reveal: bool = False) -> EnvDocument:
    items = env_manager.variables_for_ui()
    if not reveal:
        items = env_manager.mask_secrets(items)
    raw = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    return EnvDocument(
        variables=[EnvVar(**item) for item in items],
        raw=raw,
    )


@router.put("/env", response_model=EnvDocument)
def update_env(payload: EnvUpdateRequest) -> EnvDocument:
    env_manager.update(payload.updates)
    items = env_manager.mask_secrets(env_manager.variables_for_ui())
    raw = ENV_FILE.read_text()
    return EnvDocument(
        variables=[EnvVar(**item) for item in items],
        raw=raw,
    )


@router.get("/static", response_model=StaticConfig)
def get_static() -> StaticConfig:
    """Expose runtime constants that were resolved at process start."""
    from trading_system import config as cfg

    keys = [
        "TRADE_TICKERS",
        "N_ASSETS",
        "TRADE_LIST_SIZE",
        "STARTING_CAPITAL",
        "FRACTIONAL",
        "MIN_TRADE_NOTIONAL",
        "REBALANCE_BAND_PCT",
        "TRANSACTION_COST_BPS",
        "SLIPPAGE_BPS",
        "MIN_HOLD_MINUTES_AFTER_BUY",
        "MIN_REENTRY_MINUTES_AFTER_SELL",
        "TRAIN_HISTORICAL_RATIO",
        "TEST_WINDOW_DAYS",
        "TRAIN_WINDOW_YEARS",
        "PPO_DEVICE",
        "PPO_N_ENVS",
        "PPO_N_STEPS",
        "PPO_BATCH_SIZE",
        "PPO_LEARNING_RATE",
        "PPO_TOTAL_TIMESTEPS",
        "PPO_CHECKPOINT_FREQ",
        "PPO_CHECKPOINT_KEEP_LAST",
        "PPO_EARLY_STOPPING_PATIENCE",
        "PPO_EARLY_STOPPING_EVAL_FREQ",
        "PPO_EARLY_STOPPING_MIN_DELTA",
        "ENV_EPISODE_DAYS",
        "ENV_REWARD_DRAWDOWN_PENALTY",
        "ENV_OBS_ROLLING_FEATURES",
        "ENV_ALLOW_CASH_EXPOSURE",
        "ENV_PDT_RESTRICTIONS",
        "PDT_MIN_ACCOUNT_EQUITY",
        "PDT_MAX_DAY_TRADES",
        "PDT_ROLLING_BUSINESS_DAYS",
        "OLLAMA_URL",
        "OLLAMA_MODEL",
        "OLLAMA_FALLBACK_MODELS",
        "ALPACA_BASE_URL",
        "DEVICE",
        "BOARDROOM_SUMMARY_ENABLED",
        "BOARDROOM_SUMMARY_DISCORD",
        "CYCLE_INTERVAL_MIN",
        "OLLAMA_INTERVAL_MIN",
        "UNIVERSE_SOURCE",
        "UNIVERSE_FILE",
        "RELATIVE_STRENGTH_BENCHMARK",
        "LIVE_PPO_MODEL_PATH",
    ]
    values: dict[str, Any] = {}
    for key in keys:
        try:
            values[key] = getattr(cfg, key)
        except AttributeError:
            continue
    return StaticConfig(values=values)
