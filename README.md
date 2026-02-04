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
pip install -r trading_system/requirements.txt
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
- `--use-historical`: Train PPO on real historical returns instead of synthetic (recommended)
- `--use-tuned`: Use hyperparameters from a previous `tune` run

Training runs specialists (TFT on all tickers, fundamental/regime as before) and the PPO meta-learner. Use `--use-historical` to align training with backtest (real market dynamics instead of synthetic returns).

### Hyperparameter Tuning (Optuna)

Run automatic hyperparameter tuning before training to find better PPO settings:

```bash
python -m trading_system tune --n-trials 50 --timesteps 100000
```

Options:

- `--tickers`: Comma-separated tickers (default: config `TRADE_TICKERS`)
- `--n-trials`: Number of Optuna trials (default: 50)
- `--timesteps`: PPO timesteps per trial (default: 100000)
- `--timeout`: Max seconds for tuning; use 0 for no limit (default: 3600)

Best parameters are saved to `trading_system/models/meta_learner/ppo_tune_best.json`. Then train with them:

```bash
python -m trading_system train --use-historical --use-tuned --timesteps 500000
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

**WARNING**: This places REAL orders via Alpaca. Use paper trading URL in `.env` for testing:

```
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

For live trading, change to:

```
ALPACA_BASE_URL=https://api.alpaca.markets
```

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
- `ENV_REWARD_DRAWDOWN_PENALTY`: Penalty for drawdown in PPO reward (default: 0.5)
- `ENV_OBS_ROLLING_FEATURES`: Include rolling return/volatility in observations (default: true)
- `ENV_ALLOW_CASH_EXPOSURE`: When true (default), sum of PPO actions = total market exposure (capped at 1); rest stays in cash. When false, weights are normalized to 100% invested.
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
