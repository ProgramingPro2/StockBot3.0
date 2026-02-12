# The Boardroom - Architecture

## Overview

The Boardroom is a hierarchical multi-agent trading system. Eight specialist agents analyze the market independently (per ticker). Their outputs are fed into a PPO meta-learner that decides position sizing for a configurable list of symbols (default: 5 stocks).

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
    +---> Mean-Reversion Specialist (rule-based) --> mean_reversion_score (oversold/overbought)
    +---> Volatility Specialist (rule-based) --> volatility_score (per-asset risk)
    +---> Earnings/Event Risk Specialist (rule-based) --> event_risk_score (days to earnings)
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

### Mean-Reversion Specialist

- **Model**: Rule-based (RSI + Z-score)
- **Input**: OHLCV from yfinance
- **Output**: mean_reversion_score [-1, 1] (oversold=1, overbought=-1)
- **Logic**: RSI and Z-score of price vs 20d SMA; no training required

### Volatility Specialist

- **Model**: Rule-based (rolling realized vol)
- **Input**: OHLCV from yfinance
- **Output**: volatility_score [0, 1] (higher = higher recent volatility)
- **Logic**: 21d vol vs 252d vol ratio; no training required

### Earnings/Event Risk Specialist

- **Model**: Rule-based (calendar)
- **Input**: Earnings dates from yfinance get_earnings_dates()
- **Output**: event_risk_score [0, 1] (higher = closer to next earnings)
- **Logic**: Ramp from 0 when >21 days away to 1 when 0-3 days before; no training required

## Meta-Learner

- **Model**: PPO (stable-baselines3)
- **Observation**: Per-asset specialist features + holdings + optional rolling features. Length = 12 \* n_assets when `ENV_OBS_ROLLING_FEATURES` is true: for each asset (sentiment, quant_prob, fund_score, regime_flag, momentum_score, mean_reversion_score, volatility_score, event_risk_score, holdings, rolling_return_5d, rolling_return_21d, rolling_vol_21d). Example: 60-dim for 5 assets.
- **Action**: Position size % per asset [0, 1]; n_assets outputs (e.g. 5). When `ENV_ALLOW_CASH_EXPOSURE` is true, the **sum** of actions is interpreted as total market exposure (capped at 1); the rest stays in cash (0 return).
- **Reward**: Portfolio rolling Sharpe ratio minus drawdown penalty (configurable via `ENV_REWARD_DRAWDOWN_PENALTY`).
- **Training**: Three modes: synthetic `TradingEnv` (random returns), `HistoricalTradingEnv` (real OHLCV returns), or `MixedDataEnv` (configurable blend). Use `--use-historical` for 100% historical, `--historical-ratio 0.7` for 70% historical / 30% synthetic, or `TRAIN_HISTORICAL_RATIO` in config. Mixed training lets the model learn both generalizable strategies and real market dynamics.
- **Tuning**: Run `tune` for Optuna-based hyperparameter optimization; use `--use-tuned` when training.

### Variable Exposure and Cash

When `ENV_ALLOW_CASH_EXPOSURE` is true (default): allocation uses raw action sum as total market exposure (capped at 1.0). Strategy return = exposure _ (weighted basket return); remainder of capital stays in cash. In live trading, only `cash _ exposure` is invested; the rest remains in the account. When false, behavior is unchanged: weights are normalized to sum to 1 (100% invested).

## Scheduler

The scheduler daemon behaves like a small service manager:

- **Time-window services**: Each configured service (e.g. `weekday_live`) has a window (`start`, `end`, `weekdays`) in a given timezone. The daemon loop (every 60s) checks the current time: if in window and the process is not running, it starts it (or restarts after exit, with a short delay); if outside window and the process is running, it stops it. Child stdout/stderr are written to the service's `log` path (e.g. `trading_system/outputs/scheduler_weekday_live.log`).
- **One-shot jobs**: Train and backtest run at cron times via a background APScheduler thread; they are subprocess runs that complete and exit.
- **CLI**: `schedule run` (daemon), `schedule status` (in-window, running, pid, log path), `schedule run-now <id>`, `schedule stop <id>`, `schedule install` / `uninstall` (systemd user service).

Config lives in `config/schedule.yaml` (timezone, services, jobs). See README "Schedule (run on startup)" for the YAML schema and commands.

## File Structure

```
trading_system/
  config.py         - Configuration (loads from .env)
  main.py           - CLI: train, tune, backtest, paper, live, schedule
  scheduler.py      - Time-window supervisor + cron one-shot jobs
  scheduler_systemd.py - Install/uninstall systemd user service (Linux)
  data/             - Loader, news scraper, processor
  agents/           - sentiment, quantitative, fundamental, regime, momentum, mean_reversion, volatility, earnings_event
  meta_learner/     - Gym env (TradingEnv, HistoricalTradingEnv), PPO trader, Optuna tuner
  universe/         - Universe provider, features, ScorerModel, selector, autodiscover pipeline
  broker/           - Alpaca integration, Discord notifications
  utils/            - Logger, plotting (headless)
  models/           - Saved checkpoints (including models/universe/ for scorer)
  outputs/          - Plots, logs, schedule.log (daemon), scheduler_<id>.log (per service), current_trade_list.txt (when autodiscover --save)
config/
  schedule.yaml     - Scheduler config (timezone, services, jobs)
```

## Day-trading prevention

The system enforces a single rule everywhere: **do not reduce a position in the same period it was increased**.

- **Live**: A set `bought_today` records tickers that were bought on the current calendar day. When the PPO signals a sell (target value below current), the live loop skips the sell for that ticker if it is in `bought_today`. The set is cleared when the date changes.
- **Backtest**: For each bar, a per-asset `last_buy_bar[i]` stores the bar index where asset `i` was last increased. A decrease (sell) is applied only when the dollar gap exceeds `MIN_TRADE_NOTIONAL` and `last_buy_bar[i] != current_bar`, so we never sell on the same bar we bought.
- **Training (TradingEnv and HistoricalTradingEnv)**: In each `step()`, updates are applied in two phases: (1) apply all increases that meet the notional threshold and record those assets in `increased_this_step`; (2) apply decreases only for assets that meet the threshold and are not in `increased_this_step`. So the agent never reduces a position in the same step it increased it.

This keeps behavior consistent across live, backtest, and training and avoids day-trade patterns.

## Trading Modes

**Train**: Train specialists (TFT on all TRADE_TICKERS, fundamental/regime as before) and meta-learner (PPO) on TRADE_TICKERS. Use `--tickers` to override. Use `--use-historical` for pure historical returns, `--historical-ratio 0.7` for mixed (70% historical, 30% synthetic), or `TRAIN_HISTORICAL_RATIO` in config. Use `--use-tuned` to apply tuned hyperparameters.

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
