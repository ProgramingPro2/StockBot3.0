"""
Configuration for The Boardroom trading system.
Ollama URL, CUDA device, hyperparameters, cache paths.
"""
import os
from pathlib import Path

# Paths (BASE_DIR first so we can load .env from project root)
BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = BASE_DIR.parent

# Load .env from project root so it is found regardless of CWD
try:
    from dotenv import load_dotenv
    for _env_path in (_PROJECT_ROOT / ".env", Path.cwd() / ".env", BASE_DIR / ".env"):
        if _env_path.exists():
            load_dotenv(_env_path, override=True)
            break
    else:
        load_dotenv(override=True)
except ImportError:
    pass

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
MOMENTUM_MODEL_DIR = MODEL_DIR / "momentum"
META_LEARNER_MODEL_DIR = MODEL_DIR / "meta_learner"
UNIVERSE_MODEL_DIR = MODEL_DIR / "universe"

# Universe (autodiscovery)
UNIVERSE_SOURCE = os.getenv("UNIVERSE_SOURCE", "config").lower()  # config | file | sp500
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "")  # Path to ticker list file when UNIVERSE_SOURCE=file
TRADE_LIST_SIZE = int(os.getenv("TRADE_LIST_SIZE", "5"))  # Number of symbols autodiscovery selects (must match PPO n_assets)
AUTODISCOVER_INTERVAL_DAYS = int(os.getenv("AUTODISCOVER_INTERVAL_DAYS", "7"))  # Run discovery every N days in paper/live

# Mean-Reversion, Volatility, Earnings/Event Risk Specialists (rule-based)
EVENT_RISK_LOOKBACK_DAYS = 21  # Days before earnings to ramp event_risk_score
VOLATILITY_WINDOW_DAYS = 21  # Short window for realized vol
VOLATILITY_REF_WINDOW_DAYS = 252  # Long window for vol normalization

# Momentum Specialist (short-horizon)
MOMENTUM_WINDOW_DAYS = 60  # Rolling window for features and training
MOMENTUM_FORWARD_DAYS = 5  # Forward return horizon for label
MOMENTUM_N_ESTIMATORS = 100
MOMENTUM_MAX_DEPTH = 4
MOMENTUM_LEARNING_RATE = 0.05

# Relative Strength Specialist (benchmark for outperformance score)
RELATIVE_STRENGTH_BENCHMARK = os.getenv("RELATIVE_STRENGTH_BENCHMARK", "SPY").strip().upper() or "SPY"

# Ensure directories exist
for d in [CACHE_DIR, FUNDAMENTAL_CACHE_DIR, NEWS_CACHE_DIR, SPECIALIST_CACHE_DIR, OUTPUT_DIR, QUANTITATIVE_MODEL_DIR, FUNDAMENTAL_MODEL_DIR, REGIME_MODEL_DIR, MOMENTUM_MODEL_DIR, META_LEARNER_MODEL_DIR, UNIVERSE_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ollama (first model tried; fallbacks used if primary fails, e.g. not pulled)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("OLLAMA_FALLBACK_MODELS", "gpt-oss:latest,llama3:latest,llama2:7b").split(",")
    if m.strip()
]

# Alpaca (live trading)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# PPO model for live: path to .zip (absolute or relative to META_LEARNER_MODEL_DIR). Empty = ppo_trader.zip
LIVE_PPO_MODEL_PATH = os.getenv("LIVE_PPO_MODEL_PATH", "").strip()

# Discord webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Boardroom summary: LLM one-sentence summary each cycle (paper/live). Default off.
BOARDROOM_SUMMARY_ENABLED = os.getenv("BOARDROOM_SUMMARY_ENABLED", "").strip().lower() in ("1", "true", "yes")
BOARDROOM_SUMMARY_DISCORD = os.getenv("BOARDROOM_SUMMARY_DISCORD", "").strip().lower() in ("1", "true", "yes")

# Paper/live: normal run interval (minutes) and Ollama refresh interval (minutes).
# Normal run every CYCLE_INTERVAL_MIN; sentiment + boardroom summary from Ollama every OLLAMA_INTERVAL_MIN (cached between).
try:
    _cycle = int(os.getenv("CYCLE_INTERVAL_MIN", "5").strip() or "5")
    CYCLE_INTERVAL_MIN = max(1, _cycle)
except (ValueError, TypeError):
    CYCLE_INTERVAL_MIN = 5
try:
    _ollama = int(os.getenv("OLLAMA_INTERVAL_MIN", "60").strip() or "60")
    OLLAMA_INTERVAL_MIN = max(1, _ollama)
except (ValueError, TypeError):
    OLLAMA_INTERVAL_MIN = 60

# Backtest / training: bars between proxy-sentiment "refresh" (simulates live Ollama cadence without LLM).
# Unset = max(1, OLLAMA_INTERVAL_MIN // CYCLE_INTERVAL_MIN) to mirror paper/live ratio.
try:
    _be = os.getenv("BACKTEST_OLLAMA_EVERY_N_BARS", "").strip()
    BACKTEST_OLLAMA_EVERY_N_BARS = max(1, int(_be)) if _be else None
except (ValueError, TypeError):
    BACKTEST_OLLAMA_EVERY_N_BARS = None


def sentiment_refresh_every_n_bars() -> int:
    """Trading days between proxy sentiment recomputes in backtest and training stale cadence."""
    if BACKTEST_OLLAMA_EVERY_N_BARS is not None:
        return BACKTEST_OLLAMA_EVERY_N_BARS
    return max(1, OLLAMA_INTERVAL_MIN // CYCLE_INTERVAL_MIN)


# Trading: fractional shares and starting capital
FRACTIONAL = os.getenv("FRACTIONAL", "yes").lower() in ("1", "true", "yes")
try:
    STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000").strip() or "100000")
except (ValueError, TypeError):
    STARTING_CAPITAL = 100000.0
# Minimum notional (dollars) to trigger a trade; used by live, backtest, and training envs
try:
    MIN_TRADE_NOTIONAL = float(os.getenv("MIN_TRADE_NOTIONAL", "100").strip() or "100")
except (ValueError, TypeError):
    MIN_TRADE_NOTIONAL = 100.0
# Rebalance band: only rebalance when |target - current| > max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
try:
    REBALANCE_BAND_PCT = float(os.getenv("REBALANCE_BAND_PCT", "0.03").strip() or "0.03")
except (ValueError, TypeError):
    REBALANCE_BAND_PCT = 0.03
# Execution cooldowns (minutes): min hold after buy before allowing sell; min wait after sell before allowing re-buy
try:
    MIN_HOLD_MINUTES_AFTER_BUY = int(os.getenv("MIN_HOLD_MINUTES_AFTER_BUY", "60").strip() or "60")
except (ValueError, TypeError):
    MIN_HOLD_MINUTES_AFTER_BUY = 60
try:
    MIN_REENTRY_MINUTES_AFTER_SELL = int(os.getenv("MIN_REENTRY_MINUTES_AFTER_SELL", "60").strip() or "60")
except (ValueError, TypeError):
    MIN_REENTRY_MINUTES_AFTER_SELL = 60
# Transaction cost and slippage (basis points) for backtest and training reward
try:
    TRANSACTION_COST_BPS = float(os.getenv("TRANSACTION_COST_BPS", "10").strip() or "10")
except (ValueError, TypeError):
    TRANSACTION_COST_BPS = 10.0
try:
    SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5").strip() or "5")
except (ValueError, TypeError):
    SLIPPAGE_BPS = 5.0

# Pattern day trader (PDT) simulation: backtest, training envs, and live Alpaca loop
ENV_PDT_RESTRICTIONS = os.getenv("ENV_PDT_RESTRICTIONS", "").strip().lower() in ("1", "true", "yes")
try:
    PDT_MIN_ACCOUNT_EQUITY = float(os.getenv("PDT_MIN_ACCOUNT_EQUITY", "25000").strip() or "25000")
except (ValueError, TypeError):
    PDT_MIN_ACCOUNT_EQUITY = 25000.0
try:
    PDT_MAX_DAY_TRADES = int(os.getenv("PDT_MAX_DAY_TRADES", "3").strip() or "3")
except (ValueError, TypeError):
    PDT_MAX_DAY_TRADES = 3
try:
    PDT_ROLLING_BUSINESS_DAYS = int(os.getenv("PDT_ROLLING_BUSINESS_DAYS", "5").strip() or "5")
except (ValueError, TypeError):
    PDT_ROLLING_BUSINESS_DAYS = 5
PDT_MAX_DAY_TRADES = max(1, PDT_MAX_DAY_TRADES)
PDT_ROLLING_BUSINESS_DAYS = max(1, PDT_ROLLING_BUSINESS_DAYS)

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
# When OHLCV parquet cache is older than this many days, refetch on next load (0 = disable, use hours-only)
try:
    CACHE_STALE_FULL_REFRESH_DAYS = int(os.getenv("CACHE_STALE_FULL_REFRESH_DAYS", "90").strip() or "90")
except (ValueError, TypeError):
    CACHE_STALE_FULL_REFRESH_DAYS = 90
SPECIALIST_CACHE_MAX_AGE_HOURS = 1  # Refetch when older (sentiment/news change)
RATE_LIMIT_REQUESTS_PER_MIN = 30
YFINANCE_LOOKBACK_YEARS = 10

# Device
DEVICE = "cuda"  # Specialists (TFT, VAE): GPU when available
PPO_DEVICE = os.getenv("PPO_DEVICE", "cpu")  # PPO meta-learner: CPU recommended by SB3 for MLP

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
# When using --use-tuned, n_steps is capped at this value for faster training (tune often picks 4096)
PPO_N_STEPS = int(os.getenv("PPO_N_STEPS", "2048"))
PPO_BATCH_SIZE = int(os.getenv("PPO_BATCH_SIZE", "256"))
PPO_N_EPOCHS = int(os.getenv("PPO_N_EPOCHS", "10"))
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_NET_ARCH: dict | list = None  # None = default; or {"pi": [64, 64], "vf": [64, 64]}
PPO_TOTAL_TIMESTEPS = int(os.getenv("PPO_TOTAL_TIMESTEPS", "500000"))  # Try 1000000+ with --use-historical --use-tuned
# Save checkpoint every N timesteps during training (0 = disable)
PPO_CHECKPOINT_FREQ = int(os.getenv("PPO_CHECKPOINT_FREQ", "100000"))
PPO_CHECKPOINT_KEEP_LAST = int(os.getenv("PPO_CHECKPOINT_KEEP_LAST", "5"))
OUTPUT_PLOT_KEEP_LAST = int(os.getenv("OUTPUT_PLOT_KEEP_LAST", "5"))
# Early stopping: stop if eval reward has no improvement for this many consecutive evaluations (0 = disabled)
PPO_EARLY_STOPPING_PATIENCE = int(os.getenv("PPO_EARLY_STOPPING_PATIENCE", "5"))
# Eval every N timesteps for early stopping; 0 = no early stopping
PPO_EARLY_STOPPING_EVAL_FREQ = int(os.getenv("PPO_EARLY_STOPPING_EVAL_FREQ", "50000"))
# Minimum improvement in eval reward to count as improvement
PPO_EARLY_STOPPING_MIN_DELTA = float(os.getenv("PPO_EARLY_STOPPING_MIN_DELTA", "0.01"))
# L2 regularization: Adam weight_decay (0 = disabled). Reduces overfitting.
PPO_WEIGHT_DECAY = float(os.getenv("PPO_WEIGHT_DECAY", "0"))
# CPU threads for PPO when device=cpu; 0 = use all available
PPO_CPU_THREADS = int(os.getenv("PPO_CPU_THREADS", "0")) or (os.cpu_count() or 4)
# Parallel envs for SubprocVecEnv; 1 = single env (no subprocess). With n_envs>1, total rollout = n_steps * n_envs
PPO_N_ENVS = int(os.getenv("PPO_N_ENVS", "4"))

# PPO training: fraction of episodes on historical vs synthetic data (0.0 = synthetic only, 1.0 = historical only)
# Default 1.0 so training learns from market data, not self-fulfilling synthetic simulator
TRAIN_HISTORICAL_RATIO = float(os.getenv("TRAIN_HISTORICAL_RATIO", "1.0"))
# When set in .env, use this ratio for both training and tuning; tuner will not search for historical_ratio
TRAIN_HISTORICAL_RATIO_OVERRIDE = "TRAIN_HISTORICAL_RATIO" in os.environ

# Walk-forward: hold out last TEST_WINDOW_DAYS for out-of-sample evaluation; tune/train on the rest
TEST_WINDOW_DAYS = int(os.getenv("TEST_WINDOW_DAYS", "252"))  # 1 year default
TRAIN_WINDOW_YEARS = int(os.getenv("TRAIN_WINDOW_YEARS", "5"))  # years of history to load

# Environment
# Reward: use rolling Sharpe only when we have enough steps; keep window <= episode length so reward is reachable
ENV_EPISODE_DAYS = int(os.getenv("ENV_EPISODE_DAYS", "42"))  # ~2 months
ENV_ROLLING_SHARPE_DAYS = min(252, max(5, ENV_EPISODE_DAYS - 5))  # reachable within episode
ENV_REWARD_DRAWDOWN_PENALTY = float(os.getenv("ENV_REWARD_DRAWDOWN_PENALTY", "0.7"))  # 0.3-1.0; stronger = more defensive
ENV_OBS_ROLLING_FEATURES = True  # Add rolling return 5d/21d and vol 21d per asset
ENV_ALLOW_CASH_EXPOSURE = os.getenv("ENV_ALLOW_CASH_EXPOSURE", "true").lower() in ("1", "true", "yes")  # When True, sum(action) = total exposure (capped at 1); rest in cash

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
