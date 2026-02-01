# The Boardroom - Hierarchical Multi-Agent Trading System

A hierarchical multi-agent algorithmic trading system for long-term equity strategies. Four specialist agents (Narrative, Quantitative, Fundamental, Regime) feed into a PPO meta-learner for position sizing. The system trades a configurable list of symbols (default: 5 stocks: SPY, AAPL, MSFT, GOOGL, NVDA).

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
python -m trading_system train --timesteps 50000
```

By default the system uses the ticker list in config (`TRADE_TICKERS`: SPY, AAPL, MSFT, GOOGL, NVDA). Override with `--tickers`:

```bash
python -m trading_system train --tickers SPY,AAPL,MSFT,GOOGL,AMD --timesteps 50000
```

Or use the helper script: `./run.sh train --timesteps 50000`

### Train Specialists and Meta-Learner

```bash
python -m trading_system train --timesteps 50000
```

Options:

- `--tickers`: Comma-separated tickers (default: config `TRADE_TICKERS`)
- `--retrain`: Force retrain specialists (default: false)
- `--timesteps`: PPO training timesteps (default: 50000)

Training runs specialists (TFT on all tickers, fundamental/regime as before) and the PPO meta-learner on the full ticker list.

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

Places real orders for all tickers; cash is allocated by normalizing PPO position weights. Use `--tickers` to override the list.

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

Changing the number of tickers requires retraining the PPO (saved policy has fixed input/output dimensions).
