"""
Configuration for The Boardroom trading system.
Ollama URL, CUDA device, hyperparameters, cache paths.
"""
import os
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "cache"
FUNDAMENTAL_CACHE_DIR = CACHE_DIR / "fundamentals"
NEWS_CACHE_DIR = CACHE_DIR / "news"
SPECIALIST_CACHE_DIR = CACHE_DIR / "specialist"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Model subdirectories
QUANTITATIVE_MODEL_DIR = MODEL_DIR / "quantitative"
FUNDAMENTAL_MODEL_DIR = MODEL_DIR / "fundamental"
REGIME_MODEL_DIR = MODEL_DIR / "regime"
META_LEARNER_MODEL_DIR = MODEL_DIR / "meta_learner"

# Ensure directories exist
for d in [CACHE_DIR, FUNDAMENTAL_CACHE_DIR, NEWS_CACHE_DIR, SPECIALIST_CACHE_DIR, OUTPUT_DIR, QUANTITATIVE_MODEL_DIR, FUNDAMENTAL_MODEL_DIR, REGIME_MODEL_DIR, META_LEARNER_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Alpaca (live trading)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# Discord webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Trade tickers (5 assets; override via TRADE_TICKERS=SPY,IWM,... in .env)
# Default: ETFs chosen for active-trading fit (volatility, dispersion) vs strong-trend names.
_tickers_env = os.getenv("TRADE_TICKERS", "").strip()
TRADE_TICKERS: list[str] = (
    [s.strip().upper() for s in _tickers_env.split(",") if s.strip()]
    if _tickers_env
    else ["SPY", "IWM", "XLE", "XLF", "EEM"]
)
N_ASSETS = len(TRADE_TICKERS)

# Data
CACHE_MAX_AGE_HOURS = 1
CACHE_MAX_AGE_DAYS = 7  # Stale cache OK when network fails
SPECIALIST_CACHE_MAX_AGE_HOURS = 1  # Refetch when older (sentiment/news change)
RATE_LIMIT_REQUESTS_PER_MIN = 30
YFINANCE_LOOKBACK_YEARS = 10

# Device
DEVICE = "cuda"  # Fallback to cpu if unavailable

# TFT (Quantitative Specialist)
TFT_HIDDEN_SIZE = 64
TFT_LSTM_LAYERS = 2
TFT_DROPOUT = 0.1
TFT_ATTENTION_HEAD_SIZE = 4
TFT_MAX_ENCODER_LENGTH = 252
TFT_MAX_PREDICTION_LENGTH = 30
TFT_TARGET = "forward_return_30d"

# XGBoost (Fundamental Specialist)
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

# VAE (Regime Specialist)
VAE_INPUT_DIM = 10  # Rolling returns, vol, volume features
VAE_LATENT_DIM = 4
VAE_HIDDEN_DIMS = [32, 16]
VAE_REGIME_CLUSTERS = 4  # Bull, Bear, Sideways, Crash

# PPO (Meta-Learner)
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_NET_ARCH: dict | list = None  # None = default; or {"pi": [64, 64], "vf": [64, 64]}
PPO_TOTAL_TIMESTEPS = int(os.getenv("PPO_TOTAL_TIMESTEPS", "500000"))  # Try 1000000+ with --use-historical --use-tuned

# Environment
ENV_ROLLING_SHARPE_DAYS = 252
ENV_EPISODE_DAYS = int(os.getenv("ENV_EPISODE_DAYS", "42"))  # ~2 months; was 21
ENV_REWARD_DRAWDOWN_PENALTY = float(os.getenv("ENV_REWARD_DRAWDOWN_PENALTY", "0.7"))  # 0.3-1.0; stronger = more defensive
ENV_OBS_ROLLING_FEATURES = True  # Add rolling return 5d/21d and vol 21d per asset

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
