# The Boardroom - Architecture

## Overview

The Boardroom is a hierarchical multi-agent trading system. Five specialist agents analyze the market independently (per ticker). Their outputs are fed into a PPO meta-learner that decides position sizing for a configurable list of symbols (default: 5 stocks).

The system trades a configurable list of tickers (see `TRADE_TICKERS` in config; default: SPY, AAPL, MSFT, GOOGL, NVDA). Override via `--tickers SPY,AAPL,...` on the CLI. Training, backtest, paper, and live all use this list.

## Data Flow

```
yfinance (OHLCV, fundamentals, news) per ticker
    |
    +---> Narrative Specialist (Ollama LLM) --> sentiment_score, confidence
    +---> Quantitative Specialist (TFT) --> trend_probability (trained on all tickers)
    +---> Fundamental Specialist (XGBoost) --> intrinsic_value_score
    +---> Regime Specialist (VAE on SPY) --> regime, regime_flag (shared)
    +---> Momentum Specialist (LightGBM) --> momentum_score (short-horizon)
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

### Momentum Specialist

- **Model**: LightGBM regressor
- **Input**: OHLCV-derived short-horizon features (return_5d, return_20d, RSI, ATR)
- **Output**: momentum_score [-1, 1] (short-term trend strength/direction)
- **Logic**: Train on TRADE_TICKERS; target = forward 5d return clipped to [-1, 1]; one scalar per ticker

## Meta-Learner

- **Model**: PPO (stable-baselines3)
- **Observation**: Per-asset specialist features + holdings + optional rolling features. Length = 9 \* n_assets when `ENV_OBS_ROLLING_FEATURES` is true: for each asset (sentiment, quant_prob, fund_score, regime_flag, momentum_score, holdings, rolling_return_5d, rolling_return_21d, rolling_vol_21d). Example: 45-dim for 5 assets.
- **Action**: Position size % per asset [0, 1]; n_assets outputs (e.g. 5). When `ENV_ALLOW_CASH_EXPOSURE` is true, the **sum** of actions is interpreted as total market exposure (capped at 1); the rest stays in cash (0 return).
- **Reward**: Portfolio rolling Sharpe ratio minus drawdown penalty (configurable via `ENV_REWARD_DRAWDOWN_PENALTY`).
- **Training**: Two modes: synthetic `TradingEnv` (random returns) or `HistoricalTradingEnv` (real OHLCV returns). Use `--use-historical` for real data.
- **Tuning**: Run `tune` for Optuna-based hyperparameter optimization; use `--use-tuned` when training.

### Variable Exposure and Cash

When `ENV_ALLOW_CASH_EXPOSURE` is true (default): allocation uses raw action sum as total market exposure (capped at 1.0). Strategy return = exposure _ (weighted basket return); remainder of capital stays in cash. In live trading, only `cash _ exposure` is invested; the rest remains in the account. When false, behavior is unchanged: weights are normalized to sum to 1 (100% invested).

## File Structure

```
trading_system/
  config.py         - Configuration (loads from .env)
  main.py           - CLI: train, tune, backtest, paper, live
  data/             - Loader, news scraper, processor
  agents/           - sentiment, quantitative, fundamental, regime, momentum
  meta_learner/     - Gym env (TradingEnv, HistoricalTradingEnv), PPO trader, Optuna tuner
  universe/         - Universe provider, features, ScorerModel, selector, autodiscover pipeline
  broker/           - Alpaca integration, Discord notifications
  utils/            - Logger, plotting (headless)
  models/           - Saved checkpoints (including models/universe/ for scorer)
  outputs/          - Plots, logs, current_trade_list.txt (when autodiscover --save)
```

## Trading Modes

**Train**: Train specialists (TFT on all TRADE_TICKERS, fundamental/regime as before) and meta-learner (PPO) on TRADE_TICKERS. Use `--tickers` to override. Use `--use-historical` to train on real returns, `--use-tuned` to apply tuned hyperparameters.

**Tune**: Run Optuna hyperparameter tuning on historical data. Best params saved to `models/meta_learner/ppo_tune_best.json`.

**Backtest**: Portfolio backtest over all tickers; aligned OHLCV, proxy specialists per ticker, one portfolio equity curve and plot (`backtest_portfolio.png`).

**Paper**: Run specialists for all tickers each interval; one PPO prediction (5 position sizes); log exposure and weights; Discord per ticker. Use `--autodiscover` to resolve the trade list via universe autodiscovery (top N by scorer) at start.

**Live**: Same as paper; allocate `cash * exposure` to the weighted basket; place buy/sell per ticker via Alpaca; remainder stays in cash. Use `--autodiscover` to resolve the trade list via universe autodiscovery at start.

Switching the number of assets (e.g. changing TRADE_TICKERS length) requires retraining the PPO; the saved policy has fixed input/output dimensions.

## Universe (Autodiscovery)

A separate pipeline scores a candidate universe and selects the top N symbols for the trade list. The PPO still runs on a fixed N; autodiscovery only changes **which** N symbols are traded.

- **UniverseProvider**: Candidate tickers from `UNIVERSE_SOURCE`: `config` (use TRADE_TICKERS only, no discovery), `file` (one ticker per line from `UNIVERSE_FILE`), or `sp500` (fetch S&P 500 list, cached).
- **Feature extraction**: Per-symbol specialist outputs (sentiment, quant_prob, fund_score, regime_flag, optional rolling features) via `universe.features.extract_features_for_ticker`.
- **ScorerModel**: GradientBoostingClassifier trained on historical proxy features; label = 1 if forward return in top quartile, else 0. Saves to `models/universe/scorer_model.pkl`. When no model exists, a simple rule score is used.
- **Selector**: Rank by score descending; take top N (`TRADE_LIST_SIZE`, default 5).
- **CLI**: `autodiscover` runs discovery once and prints or saves the list (`--save` to `outputs/current_trade_list.txt`). `train-universe` trains the ScorerModel from aligned historical returns. Paper and live accept `--autodiscover` to resolve the trade list at start.
