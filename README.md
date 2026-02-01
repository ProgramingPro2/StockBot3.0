# The Boardroom - Hierarchical Multi-Agent Trading System

A hierarchical multi-agent algorithmic trading system for long-term equity strategies. Four specialist agents (Narrative, Quantitative, Fundamental, Regime) feed into a PPO meta-learner for position sizing.

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
python -m trading_system train --ticker SPY --timesteps 50000
```

Or use the helper script: `./run.sh train --ticker SPY --timesteps 50000`

### Train Specialists and Meta-Learner

```bash
python -m trading_system train --ticker SPY --timesteps 50000
```

Options:
- `--ticker`: Ticker for training (default: SPY)
- `--retrain`: Force retrain specialists (default: false)
- `--timesteps`: PPO training timesteps (default: 50000)

### Backtest

```bash
python -m trading_system backtest --ticker SPY
```

Saves performance plot to `trading_system/outputs/backtest_SPY.png`.

### Paper Trading

```bash
python -m trading_system paper --ticker SPY --interval 60
```

Runs a loop that polls data, runs specialists, and logs decisions every `--interval` minutes. No real orders are placed. Sends Discord notifications if webhook is configured.

### Live Trading (Alpaca)

```bash
python -m trading_system live --ticker SPY --interval 60
```

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
- `OLLAMA_URL`, `OLLAMA_MODEL`
- `DEVICE` (cuda/cpu)
- TFT, XGBoost, VAE, PPO hyperparameters
