# The Boardroom - Architecture

## Overview

The Boardroom is a hierarchical multi-agent trading system. Four specialist agents analyze the market independently. Their outputs are fed into a PPO meta-learner that decides position sizing.

## Data Flow

```
yfinance (OHLCV, fundamentals, news)
    |
    +---> Narrative Specialist (Ollama LLM) --> sentiment_score, confidence
    +---> Quantitative Specialist (TFT) --> trend_probability
    +---> Fundamental Specialist (XGBoost) --> intrinsic_value_score
    +---> Regime Specialist (VAE on SPY) --> regime, regime_flag
    |
    v
Meta-Learner (PPO) --> position_size %
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
- **Logic**: Train on ticker or SPY; forecast 30d return, map to probability via sigmoid

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
- **Observation**: [sentiment, quant_prob, fund_score, regime_flag, holdings]
- **Action**: Position size % per asset [0, 1]
- **Reward**: Rolling Sharpe ratio

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

**Train**: Train specialists and meta-learner on historical data
**Backtest**: Simulate strategy on historical data
**Paper**: Log decisions without placing orders (+ Discord notifications)
**Live**: Place real orders via Alpaca (+ Discord notifications)
