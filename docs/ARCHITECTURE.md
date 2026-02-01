# The Boardroom - Architecture

## Overview

The Boardroom is a hierarchical multi-agent trading system. Four specialist agents analyze the market independently (per ticker). Their outputs are fed into a PPO meta-learner that decides position sizing for a configurable list of symbols (default: 5 stocks).

The system trades a configurable list of tickers (see `TRADE_TICKERS` in config; default: SPY, AAPL, MSFT, GOOGL, NVDA). Override via `--tickers SPY,AAPL,...` on the CLI. Training, backtest, paper, and live all use this list.

## Data Flow

```
yfinance (OHLCV, fundamentals, news) per ticker
    |
    +---> Narrative Specialist (Ollama LLM) --> sentiment_score, confidence
    +---> Quantitative Specialist (TFT) --> trend_probability (trained on all tickers)
    +---> Fundamental Specialist (XGBoost) --> intrinsic_value_score
    +---> Regime Specialist (VAE on SPY) --> regime, regime_flag (shared)
    |
    v
Meta-Learner (PPO) --> position_size % per asset (n_assets outputs)
```

## Specialists

### Narrative Specialist

- **Model**: Local LLM via Ollama (e.g. llama3.1)
- **Input**: News headlines/summaries from yfinance
- **Output**: sentiment_score [-1, 1], confidence [0, 1]
- **Logic**: Zero-shot; no training required

### Quantitative Specialist

- **Model**: Temporal Fusion Transformer (pytorch-forecasting)
- **Input**: 10 years daily OHLCV from yfinance
- **Output**: trend_probability [0, 1] (probability trend continues 30 days)
- **Logic**: Train on all TRADE_TICKERS (one TFT with group_ids per ticker); forecast 30d return, map to probability via sigmoid

### Fundamental Specialist

- **Model**: XGBoost classifier
- **Input**: P/E, Debt/Equity, ROE, FCF, etc. from yfinance
- **Output**: intrinsic_value_score [-1, 1] (Undervalued=1, Fair=0, Overvalued=-1)
- **Logic**: Train on heuristic labels (P/E percentile); 3-class classification

### Regime Specialist

- **Model**: VAE + KMeans on SPY
- **Input**: Rolling returns, volatility, volume features
- **Output**: regime (Bull/Bear/Sideways/Crash), regime_flag [0, 1] (1 = defensive)
- **Logic**: Unsupervised; cluster latent space; reconstruction error for anomaly (Crash)

## Meta-Learner

- **Model**: PPO (stable-baselines3)
- **Observation**: Per-asset specialist features + holdings. Length = 5 \* n_assets: for each asset (sentiment, quant_prob, fund_score, regime_flag), then n_assets holdings. Example: 25-dim for 5 assets.
- **Action**: Position size % per asset [0, 1]; n_assets outputs (e.g. 5).
- **Reward**: Portfolio rolling Sharpe ratio (portfolio return = position-weighted sum of asset returns).

## File Structure

```
trading_system/
  config.py         - Configuration (loads from .env)
  main.py           - CLI: train, backtest, paper, live
  data/             - Loader, news scraper, processor
  agents/           - sentiment, quantitative, fundamental, regime
  meta_learner/     - Gym env, PPO trader
  broker/           - Alpaca integration, Discord notifications
  utils/            - Logger, plotting (headless)
  models/           - Saved checkpoints
  outputs/          - Plots, logs
```

## Trading Modes

**Train**: Train specialists (TFT on all TRADE_TICKERS, fundamental/regime as before) and meta-learner (PPO) on TRADE_TICKERS. Use `--tickers` to override the list.

**Backtest**: Portfolio backtest over all tickers; aligned OHLCV, proxy specialists per ticker, one portfolio equity curve and plot (`backtest_portfolio.png`).

**Paper**: Run specialists for all tickers each interval; one PPO prediction (5 position sizes); log and Discord per ticker.

**Live**: Same as paper; allocate cash by normalizing position weights (sum to 1); place buy/sell per ticker via Alpaca; Discord per ticker.

Switching the number of assets (e.g. changing TRADE_TICKERS length) requires retraining the PPO; the saved policy has fixed input/output dimensions.
