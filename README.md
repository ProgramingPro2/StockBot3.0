# The Boardroom - Hierarchical Multi-Agent Trading System

A hierarchical multi-agent algorithmic trading system for long-term equity strategies. Four specialist agents (Narrative, Quantitative, Fundamental, Regime) feed into a PPO meta-learner for position sizing. The system trades a configurable list of symbols (default: 5 ETFs chosen for active-trading fit: SPY, IWM, XLE, XLF, EEM).

## Requirements

- Ubuntu Server (headless)
- Python 3.10+
- NVIDIA GPU (RTX 3090 recommended) for CUDA
- [Ollama](https://ollama.ai) installed and running (for Narrative Specialist LLM)

## Setup

```bash
cd StockBot3.0
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# or: pip install -e .
```

### Ollama Setup

```bash
# Install Ollama, then pull a model:
ollama pull gpt-oss:20b
```

### Alpaca Setup (for live trading)

1. Sign up at [Alpaca](https://alpaca.markets) and get API keys
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your Alpaca keys:
   ```
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```

### Discord Notifications (optional)

1. Create a webhook in your Discord server (Server Settings → Integrations → Webhooks)
2. Add the webhook URL to `.env`:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

Discord notifications are sent for: trades (paper/live), training complete (runs >1M steps), and HPO complete (after `tune` finishes).

## Usage

Run from the **project root** (`StockBot3.0`), not from inside `trading_system/`:

```bash
cd StockBot3.0
source .venv/bin/activate
python -m trading_system train --timesteps 500000
```

By default the system uses the ticker list in config (`TRADE_TICKERS`: SPY, IWM, XLE, XLF, EEM). Override with `--tickers`:

```bash
python -m trading_system train --tickers SPY,IWM,XLE,XLF,EEM --timesteps 500000
```

### Ticker selection

The default tickers (SPY, IWM, XLE, XLF, EEM) are chosen for active-trading fit: more volatility and dispersion across assets, where the RL strategy can add value over buy-and-hold. Strong uptrend names (e.g. NVDA, TSLA) tend to favor buy-and-hold. Override via `TRADE_TICKERS` in `.env` or `--tickers` on the command line.

Or use the helper script: `./run.sh train --timesteps 500000`

### Train Specialists and Meta-Learner

```bash
python -m trading_system train --timesteps 500000
```

Options:

- `--tickers`: Comma-separated tickers (default: config `TRADE_TICKERS`)
- `--retrain`: Force retrain specialists (default: false)
- `--timesteps`: PPO training timesteps (default: 500000)
- `--use-historical`: Train PPO on real historical returns only (equivalent to `--historical-ratio 1.0`)
- `--historical-ratio`: Fraction of episodes on historical vs synthetic data. For mixed mode use 0.1-0.9 (HPO samples this range). 0 = synthetic only, 1 = historical only.
- `--use-tuned`: Use hyperparameters from a previous `tune` run

Training runs specialists (TFT on all tickers, fundamental/regime as before) and the PPO meta-learner. Use `--use-historical` for pure historical data, or `--historical-ratio 0.7` for 70% historical / 30% synthetic. Set `TRAIN_HISTORICAL_RATIO` in `.env` for a default ratio.

### Hyperparameter Tuning (Optuna)

Run automatic hyperparameter tuning before training to find better PPO settings:

```bash
python -m trading_system tune --n-trials 50 --timesteps 100000
```

Options:

- `--tickers`: Comma-separated tickers (default: config `TRADE_TICKERS`)
- `--n-trials`: Number of Optuna trials (default: 50)
- `--timesteps`: PPO timesteps per trial (default: 100000)
- `--timeout`: Max seconds for tuning; 0 = no limit (default)

Best parameters are saved to `trading_system/models/meta_learner/ppo_tune_best.json`, including `historical_ratio` (tuned). Then train with them:

```bash
python -m trading_system train --use-historical --use-tuned --timesteps 500000
# Or mixed training (70% historical, 30% synthetic):
python -m trading_system train --historical-ratio 0.7 --use-tuned --timesteps 500000
```

### Backtest

```bash
python -m trading_system backtest
```

Portfolio backtest over all tickers; saves performance plot to `trading_system/outputs/backtest_portfolio.png`. Use `--tickers SPY,AAPL,...` to override the list.

### Paper Trading

```bash
python -m trading_system paper --interval 60
```

Runs a loop that polls data, runs specialists for all tickers, and logs position sizes every `--interval` minutes. No real orders are placed. Sends Discord notifications per ticker if webhook is configured.

### Live Trading (Alpaca)

```bash
python -m trading_system live --interval 60
```

Places real orders for all tickers; when `ENV_ALLOW_CASH_EXPOSURE` is true (default), only `cash * exposure` is invested (exposure = min(1, sum of PPO actions)); the rest stays in cash. Use `--tickers` to override the list, or `--autodiscover` to resolve the list via universe autodiscovery.

The system avoids day trading: it never sells a position in the same period it was bought (same calendar day in live, same bar in backtest, same step in training). See **Day-trading prevention** in `docs/ARCHITECTURE.md`.

**WARNING**: This places REAL orders via Alpaca. Use paper trading URL in `.env` for testing:

```
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

For live trading, change to:

```
ALPACA_BASE_URL=https://api.alpaca.markets
```

### Schedule (run on startup)

The scheduler is a **systemd-like supervisor**: it keeps live/paper processes running only during a time window (e.g. 9:30 AM–5 PM weekdays), restarts them if they exit in-window, and stops them outside the window. One-shot jobs (train, backtest) run at cron times. Config: `config/schedule.yaml`.

**Config schema** (`config/schedule.yaml`):

- **timezone**: e.g. `America/New_York` (used for window and cron).
- **services**: Time-window "units" (live or paper). Each has `id`, `type`, `window` (`start`, `end`, `weekdays`), `options`, and optional `log` path. Example: `weekday_live` with window 09:30–17:00 Mon–Fri; child output goes to `trading_system/outputs/scheduler_weekday_live.log`.
- **jobs**: Cron one-shot jobs (train, backtest) with `id`, `type`, `when` (cron), `options`.

**Commands**:

```bash
# Run scheduler daemon in foreground (time-window services + cron one-shots)
python -m trading_system schedule run
# Or: schedule run --config /path/to/schedule.yaml

# Show service status (in window, running, pid, log path)
python -m trading_system schedule status

# Start a service immediately, ignoring the window (for testing)
python -m trading_system schedule run-now weekday_live

# Stop a running service by id
python -m trading_system schedule stop weekday_live

# Install as systemd user service (run on login; Linux only)
python -m trading_system schedule install --config config/schedule.yaml

# Remove the service
python -m trading_system schedule uninstall
```

For start at boot without being logged in, run once: `loginctl enable-linger $USER`. The install command writes a user unit to `~/.config/systemd/user/trading_system_schedule.service` and enables/starts it. Daemon stdout/stderr go to `trading_system/outputs/schedule.log`; each service's child process logs to its `log` path (e.g. `scheduler_weekday_live.log`).

## Data Sources

Market data sources are **free**:

- **yfinance**: OHLCV, fundamentals, news headlines
- **Ollama**: Local LLM for sentiment (no API keys)

Trading integration:

- **Alpaca**: Commission-free trading API (requires account)

## Architecture

See `docs/ARCHITECTURE.md` for the full agent flow and data pipeline.

## Configuration

Edit `trading_system/config.py` for:

- `TRADE_TICKERS`: list of symbols to trade (default: SPY, AAPL, MSFT, GOOGL, NVDA). Can be overridden by env `TRADE_TICKERS=SPY,AAPL,...` in `.env`.
- `OLLAMA_URL`, `OLLAMA_MODEL`
- `DEVICE` (cuda/cpu)
- TFT, XGBoost, VAE, PPO hyperparameters
- `TRAIN_HISTORICAL_RATIO`: Fraction of PPO episodes on historical vs synthetic data (0.0-1.0; default: 0.0). Overridden by `--historical-ratio` or `--use-historical`. When set in `.env`, this value is used as the fixed ratio for both training and tuning; the tuner will not search for a new value.
- `ENV_REWARD_DRAWDOWN_PENALTY`: Penalty for drawdown in PPO reward (default: 0.5)
- `ENV_OBS_ROLLING_FEATURES`: Include rolling return/volatility in observations (default: true)
- `ENV_ALLOW_CASH_EXPOSURE`: When true (default), sum of PPO actions = total market exposure (capped at 1); rest stays in cash. When false, weights are normalized to 100% invested.
- `RELATIVE_STRENGTH_BENCHMARK`: Benchmark symbol for relative strength specialist (default: SPY). Override in `.env` for sector-relative strength.
- `MIN_TRADE_NOTIONAL`: Minimum dollar gap to trigger a trade (default: 100). Used in live, backtest, and training envs; trades below this size are skipped (HOLD).
- **Universe (autodiscovery)**: `UNIVERSE_SOURCE` (config | file | sp500), `UNIVERSE_FILE` (path when source=file), `TRADE_LIST_SIZE` (default 5), `AUTODISCOVER_INTERVAL_DAYS` (default 7)

Use `python -m trading_system tune` for automatic PPO hyperparameter optimization instead of manual config edits. Changing the number of tickers requires retraining the PPO (saved policy has fixed input/output dimensions).

### Universe Autodiscovery

To let the system pick the trade list from a larger universe:

1. Set `UNIVERSE_SOURCE=file` and `UNIVERSE_FILE` to a path with one ticker per line, or `UNIVERSE_SOURCE=sp500` to use S&P 500 (cached).
2. Train the universe ScorerModel from historical returns (top quartile forward return = 1):
   ```bash
   python -m trading_system train-universe --tickers SPY,IWM,XLE,XLF,EEM,AAPL,MSFT,...
   ```
3. Run discovery once: `python -m trading_system autodiscover --save` (saves to `outputs/current_trade_list.txt`).
4. Use `--autodiscover` with paper or live to resolve the trade list at start:
   ```bash
   python -m trading_system paper --autodiscover --interval 60
   python -m trading_system live --autodiscover --interval 60
   ```
   When `UNIVERSE_SOURCE=config`, behavior is unchanged: trade list = TRADE_TICKERS or `--tickers`.
