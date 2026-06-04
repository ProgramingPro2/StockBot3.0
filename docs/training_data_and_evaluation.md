# Training Data and Evaluation

## Data Contract

### Historical Feature Store

The price/volume-first feature store (`trading_system/data/feature_store.py`) is the primary training dataset. It provides:

- **Point-in-time only**: No feature uses future data (no `shift(-n)` or lookahead). All features are computed from OHLCV up to and including the current bar.
- **Schema**: Per (date, ticker): `return_1d`, `return_5d`, `return_20d`, `vol_20d`, `atr_pct`, `momentum_score`, `mean_reversion_score`, `volatility_score`, `volume_ratio`, and optional rolling features (`rolling_return_5d`, `rolling_return_21d`, `rolling_vol_21d`).
- **Usage**: `build_feature_dataset(tickers, lookback_years)` returns a DataFrame with MultiIndex (date, ticker). `get_features_for_date(feature_df, as_of_date, tickers)` returns one dict per ticker for that date, compatible with the PPO observation schema (with sentiment/quant/fund set to neutral when using price-only).

### Walk-Forward Evaluation

- **Splits**: Chronological train/test. Config: `TEST_WINDOW_DAYS` (default 252) and `TEST_WINDOW_PLACEMENT` (`recent` = hold out newest days, `oldest` = hold out oldest days so training sees recent regimes, `none` = no holdout). No shuffle.
- **Tuning**: Optuna tuning uses only the train window. The test window is never seen during hyperparameter search.
- **Training**: When a split is available, PPO is trained on the train window only. The "training complete" backtest and reported stats use the held-out test window (out-of-sample).
- **Leakage**: Target construction and rolling features use only past data. Use `get_features_for_date(..., as_of_date)` so that inference at date `t` only sees rows with index <= t.

## Execution Safeguards

- **Portfolio sizing**: Live, paper, and backtest size targets from **total portfolio value**, not cash. This prevents sell-then-rebuy churn when cash is temporarily low.
- **Holdings**: Current portfolio weights (holdings) are passed into `PPOTrader.predict(..., holdings=...)` in live, paper, and backtest so the policy sees the same observation structure as in training.
- **Rebalance band**: Trades occur only when `|target_value - current_value| > max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)`.
- **Cooldowns**: After a buy, no sell for `MIN_HOLD_MINUTES_AFTER_BUY` minutes. After a sell, no buy for `MIN_REENTRY_MINUTES_AFTER_SELL` minutes (live only; backtest uses 1-bar equivalents).
- **Transaction costs**: Backtest and training env apply `TRANSACTION_COST_BPS` and `SLIPPAGE_BPS` so strategies that rely on friction-free trading are penalized.

## Specialists

Specialists (sentiment, quantitative, fundamental, regime, momentum, etc.) are **optional overlays**. The system can run with the price/volume feature store as the baseline. Specialists add signal only when they have been validated (e.g. out-of-sample) or when explicitly enabled.

## Configuration reference

| Purpose | Config / .env |
|--------|----------------|
| Rebalance threshold | `MIN_TRADE_NOTIONAL`, `REBALANCE_BAND_PCT` |
| Live cooldowns | `MIN_HOLD_MINUTES_AFTER_BUY`, `MIN_REENTRY_MINUTES_AFTER_SELL` |
| Transaction friction | `TRANSACTION_COST_BPS`, `SLIPPAGE_BPS` |
| Walk-forward test window | `TEST_WINDOW_DAYS` (default 252), `TEST_WINDOW_PLACEMENT` (`recent` / `oldest` / `none`) |
| History length for data load | `TRAIN_WINDOW_YEARS` (default 5) |
| Historical vs synthetic training | `TRAIN_HISTORICAL_RATIO` (default 1.0) |
| Reward window reachable in episode | `ENV_EPISODE_DAYS`, `ENV_ROLLING_SHARPE_DAYS` (derived) |
| Early stopping | `PPO_EARLY_STOPPING_PATIENCE`, `PPO_EARLY_STOPPING_EVAL_FREQ`, `PPO_EARLY_STOPPING_MIN_DELTA` (0 = disabled) |
| L2 regularization | `PPO_WEIGHT_DECAY` (Adam weight_decay; 0 = disabled) |

See `.env.example` for all optional variables and comments.

## Early stopping and regularization

To reduce overfitting:

- **Early stopping**: When training with historical data, the trainer evaluates the policy every `PPO_EARLY_STOPPING_EVAL_FREQ` timesteps (5 episodes per eval). If the mean eval reward does not improve by at least `PPO_EARLY_STOPPING_MIN_DELTA` for `PPO_EARLY_STOPPING_PATIENCE` consecutive evaluations, training stops. Set `PPO_EARLY_STOPPING_EVAL_FREQ=0` or `PPO_EARLY_STOPPING_PATIENCE=0` to disable.
- **Weight decay**: Set `PPO_WEIGHT_DECAY` (e.g. `1e-5`) to add L2 regularization via the Adam optimizer. This shrinks policy weights and can improve generalization.
