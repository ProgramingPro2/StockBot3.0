# The Boardroom - Architecture

## Overview

The Boardroom is a hierarchical multi-agent trading system. Ten specialist agents analyze the market independently (per ticker). Their outputs are fed into a PPO meta-learner that decides position sizing for a configurable list of symbols (default: 5 stocks).

The system trades a configurable list of tickers (see `TRADE_TICKERS` in config; default: SPY, IWM, XLE, XLF, EEM). Override via `--tickers SPY,AAPL,...` on the CLI. Training, backtest, paper, and live all use this list.

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
    +---> Relative Strength Specialist (rule-based) --> relative_strength_score (vs benchmark 20d/60d)
    +---> Liquidity Specialist (rule-based) --> liquidity_score [0,1] (volume percentile)
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

### Relative Strength Specialist

- **Model**: Rule-based (price returns vs benchmark)
- **Input**: OHLCV for ticker and benchmark (default SPY; configurable via `RELATIVE_STRENGTH_BENCHMARK`)
- **Output**: relative_strength_score [-1, 1] (negative = underperforming, positive = outperforming)
- **Logic**: 20d and 60d total returns vs benchmark; average relative return scaled/clipped to [-1, 1]; complements momentum for rotation

### Liquidity Specialist

- **Model**: Rule-based (volume percentile)
- **Input**: OHLCV volume from yfinance
- **Output**: liquidity_score [0, 1] (higher = more liquid)
- **Logic**: Percentile of current volume within last 20d/60d; helps meta-learner avoid oversizing illiquid names

## Meta-Learner

- **Model**: PPO (stable-baselines3)
- **Observation**: Per-asset specialist features + holdings + optional rolling features. Length = 14 \* n_assets when `ENV_OBS_ROLLING_FEATURES` is true: for each asset (sentiment, quant_prob, fund_score, regime_flag, momentum_score, mean_reversion_score, volatility_score, event_risk_score, relative_strength_score, liquidity_score, holdings, rolling_return_5d, rolling_return_21d, rolling_vol_21d). Without rolling: 11 \* n_assets. Example: 70-dim for 5 assets with rolling.
- **Action**: Position size % per asset [0, 1]; n_assets outputs (e.g. 5). When `ENV_ALLOW_CASH_EXPOSURE` is true, the **sum** of actions is interpreted as total market exposure (capped at 1); the rest stays in cash (0 return).
- **Reward**: Full-portfolio return (weights \* returns, cash earns 0) with rolling Sharpe when enough history; drawdown penalty (`ENV_REWARD_DRAWDOWN_PENALTY`) and turnover cost (`TRANSACTION_COST_BPS`, `SLIPPAGE_BPS`).
- **Training**: Default is historical-first (`TRAIN_HISTORICAL_RATIO=1.0`). Three modes: synthetic `TradingEnv`, `HistoricalTradingEnv` (real OHLCV returns), or mixed. Use `--use-historical` for 100% historical or `TRAIN_HISTORICAL_RATIO` in config. When set in `.env`, it is used as the fixed ratio for both training and tuning. See [Training Data and Evaluation](training_data_and_evaluation.md) for walk-forward splits and data contract.
- **Tuning**: Run `tune` for Optuna-based hyperparameter optimization; use `--use-tuned` when training.

### Variable Exposure and Cash

When `ENV_ALLOW_CASH_EXPOSURE` is true (default): allocation uses raw action sum as total market exposure (capped at 1.0). Strategy return = exposure \* (weighted basket return); remainder of capital stays in cash. **Live, paper, and backtest size targets from total portfolio value** (not cash), so target value per asset = portfolio_value \* weight. This avoids sell-then-rebuy churn when cash is temporarily low. When false, weights are normalized to sum to 1 (100% invested).

## Scheduler

The scheduler daemon behaves like a small service manager:

- **Time-window services**: Each configured service (e.g. `weekday_live`) has a window (`start`, `end`, `weekdays`) in a given timezone. The daemon loop (every 60s) checks the current time: if in window and the process is not running, it starts it (or restarts after exit, with a short delay); if outside window and the process is running, it stops it. Child stdout/stderr are written to the service's `log` path (e.g. `trading_system/outputs/scheduler_weekday_live.log`).
- **One-shot jobs**: Train and backtest run at cron times via a background APScheduler thread; they are subprocess runs that complete and exit.
- **CLI**: `schedule run` (daemon), `schedule status` (in-window, running, pid, log path), `schedule run-now <id>`, `schedule stop <id>`, `schedule install` / `uninstall` (systemd user service).

Config lives in `config/schedule.yaml` (timezone, services, jobs). See README "Schedule (run on startup)" for the YAML schema and commands.

## File Structure

```
trading_system/
  config.py             - Configuration (loads from .env)
  main.py               - Re-exports the Typer CLI + helpers used by tests
  cli/
    app.py              - Typer app (replaces argparse parser); rich-formatted help
    parser.py           - Backwards-compat shim re-exporting cli/app
    formatting.py       - Rich helpers (panels, stats tables, --json)
    commands/           - Per-command Typer modules
    run.py              - Implementations (cmd_train, cmd_backtest, ...)
  scheduler.py          - Time-window supervisor + cron one-shot jobs
  scheduler_systemd.py  - Install/uninstall systemd user service (Linux)
  data/                 - Loader, news scraper, processor
  agents/               - sentiment, quantitative, fundamental, regime, momentum, ...
  meta_learner/
    env.py / env_helpers.py
    ppo_trader.py       - PPO trainer; tensorboard_log + JsonlMetricsCallback
    metrics_callback.py - Per-rollout JSONL writer for the web UI charts
    runs_index.py       - models/meta_learner/runs.json bookkeeping
    tuner.py            - Optuna tuning (also writes TB logs per trial)
    eval_helpers.py / proxy_specialists.py
  universe/             - Universe provider, features, ScorerModel, selector
  broker/               - Alpaca integration, Discord notifications
  utils/                - Logger, plotting (headless), pdt
  models/               - Saved checkpoints
    meta_learner/
      ppo_trader*.zip
      tensorboard/<run_id>/  (PPO logs)
      runs.json
  outputs/              - Plots, logs, schedule.log
    jobs/<job_id>/      - log.txt, metrics.jsonl, result.json, params.json
config/
  schedule.yaml         - Scheduler config (timezone, services, jobs)

webapp/
  backend/              - FastAPI (api/, job_manager.py, env_manager.py, schedule_manager.py,
                          system_info.py, tensorboard_proxy.py, schemas.py)
  frontend/             - Vite + React + Tailwind (src/pages, src/components, lib/api.ts, lib/ws.ts)

docker/
  Dockerfile            - Multi-stage: node:20-alpine + nvidia/cuda runtime
  docker-compose.yml    - webapp + tensorboard sidecar + optional ollama profile
  entrypoint.sh

requirements-web.txt    - FastAPI/uvicorn/websockets/psutil/nvidia-ml-py
Makefile                - web/web-dev/web-prod, frontend-build, docker-up*, test, cli
```

## Day-trading prevention

By default the system avoids same-period buy-then-sell on the same asset using `last_buy_bar` / `last_sell_bar` and min-hold / min-reentry rules aligned with live cooldowns.

- **Live**: A set `bought_today` records tickers bought on the current America/New_York session date. When `ENV_PDT_RESTRICTIONS` is off, sells are skipped if the symbol is in `bought_today` (same as before). When PDT mode is on and portfolio value is below `PDT_MIN_ACCOUNT_EQUITY`, same-day sells are allowed only up to `PDT_MAX_DAY_TRADES` within a rolling `PDT_ROLLING_BUSINESS_DAYS` window (each qualifying sell is recorded); at the limit, new buys are skipped until the window rolls forward. Above the equity threshold, PDT limits do not apply.
- **Backtest**: Shared `rebalance_multi_asset` logic applies the same band, cooldown bars, and optional PDT rules (`trading_system/utils/pdt.py`) per bar.
- **Training (`TradingEnv`, `HistoricalTradingEnv`)**: Each step uses the same rebalance helper as the backtest so policy training matches execution semantics. `HistoricalTradingEnv` accepts optional `trading_dates` (aligned index) for calendar-based PDT windows; otherwise an integer cursor is used.

Alpaca may enforce PDT independently; local flags mirror policy for simulation and logging.

## Trading Modes

**Train**: Train specialists (TFT on all TRADE_TICKERS, fundamental/regime as before) and meta-learner (PPO) on TRADE_TICKERS. Use `--tickers` to override. Use `--use-historical` for pure historical returns, `--historical-ratio 0.7` for mixed (70% historical, 30% synthetic), or `TRAIN_HISTORICAL_RATIO` in config. When `TRAIN_HISTORICAL_RATIO` is set in `.env`, it is used as the fixed ratio for both training and tuning. Use `--use-tuned` to apply tuned hyperparameters. Random historical episodes from `ProxySpecialistFn` apply the same **proxy sentiment staleness** expectation as backtest (probabilistic refresh vs carry-over, no Ollama); other proxy features still vary each sampled day.

**Tune**: Run Optuna hyperparameter tuning on historical data. Best params saved to `models/meta_learner/ppo_tune_best.json`.

**Backtest**: Portfolio backtest over all tickers; aligned OHLCV, proxy specialists per ticker, one portfolio equity curve and plot (`backtest_portfolio.png`). Sentiment uses the same **OHLCV momentum proxy** as live when the LLM is not used (`sentiment_proxy_from_close_prices`), with **staleness** between refreshes: recomputed every `N` trading days where `N` = `BACKTEST_OLLAMA_EVERY_N_BARS` if set, else `max(1, OLLAMA_INTERVAL_MIN // CYCLE_INTERVAL_MIN)` so the bar cadence matches the paper/live Ollama-to-cycle ratio without any Ollama requests. Other proxy specialist fields still update every bar from the precomputed return window.

**Paper**: Run specialists for all tickers each interval; one PPO prediction (5 position sizes); log exposure and weights; Discord per ticker. Use `--autodiscover` to resolve the trade list via universe autodiscovery (top N by scorer) at start.

**Live**: Same as paper; target value per ticker = portfolio_value \* weight (sized from total portfolio value, not cash); place buy/sell per ticker via Alpaca. Rebalance band and cooldowns (min hold after buy, min re-entry after sell) apply. Use `--autodiscover` to resolve the trade list via universe autodiscovery at start.

Switching the number of assets (e.g. changing TRADE_TICKERS length) requires retraining the PPO; the saved policy has fixed input/output dimensions.

## Universe (Autodiscovery)

A separate pipeline scores a candidate universe and selects the top N symbols for the trade list. The PPO still runs on a fixed N; autodiscovery only changes **which** N symbols are traded.

- **UniverseProvider**: Candidate tickers from `UNIVERSE_SOURCE`: `config` (use TRADE_TICKERS only, no discovery), `file` (one ticker per line from `UNIVERSE_FILE`), or `sp500` (fetch S&P 500 list, cached).
- **Feature extraction**: Per-symbol specialist outputs (sentiment, quant_prob, fund_score, regime_flag, optional rolling features) via `universe.features.extract_features_for_ticker`.
- **ScorerModel**: GradientBoostingClassifier trained on historical proxy features; label = 1 if forward return in top quartile, else 0. Saves to `models/universe/scorer_model.pkl`. When no model exists, a simple rule score is used.
- **Selector**: Rank by score descending; take top N (`TRADE_LIST_SIZE`, default 5).
- **CLI**: `autodiscover` runs discovery once and prints or saves the list (`--save` to `outputs/current_trade_list.txt`). `train-universe` trains the ScorerModel from aligned historical returns. Paper and live accept `--autodiscover` to resolve the trade list at start.

## Web Layer

The CLI (`trading_system.cli.app`) is wrapped by a FastAPI + React control plane under `webapp/`.

```
Browser (React + Tailwind, Vite build)
    | HTTP / WS via /api/*
    v
FastAPI (webapp/backend)
    +---> JobManager: subprocess.Popen("python -m trading_system <cmd>", env=BOARDROOM_*)
    |       +-> tees stdout to outputs/jobs/<id>/log.txt (broadcast over WS)
    |       +-> tails outputs/jobs/<id>/metrics.jsonl (broadcast over WS)
    |       +-> reads outputs/jobs/<id>/result.json when done
    +---> EnvManager: parse/write .env preserving comments
    +---> ScheduleManager: read/write config/schedule.yaml + run-now/stop/status
    +---> SystemInfo: psutil + nvidia-ml-py for CPU/GPU pills
    +---> RunsIndex: models/meta_learner/runs.json (TensorBoard run list)
    +---> /tb/* reverse-proxy to TensorBoard (port 6006; web UI default HTTP port 28473)
```

The Boardroom CLI accepts `--json` and reads `BOARDROOM_RESULT_PATH`/`BOARDROOM_METRICS_PATH`/`BOARDROOM_RUN_ID`/`BOARDROOM_RUN_KIND` from the environment. The PPO trainer in `trading_system/meta_learner/ppo_trader.py` writes TensorBoard logs to `models/meta_learner/tensorboard/<run_id>/` and emits one `metrics.jsonl` line per rollout via `JsonlMetricsCallback` (only when `BOARDROOM_METRICS_PATH` is set, so direct CLI use is unaffected).

`docker/` packages the web layer as a multi-stage build (Node frontend builder + CUDA Python runtime). `docker/docker-compose.yml` defines:

- `webapp` (this image, port 28473, GPU reservations, mounts the project's `.env`/`config/`/`outputs/`/`models/`/`data/cache/`).
- `tensorboard` reading the same logs volume on port 6006; `webapp` proxies it under `/tb/`.
- `ollama` (opt-in via the `ollama` profile) for fully containerized deployments.
