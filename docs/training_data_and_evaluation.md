# Training Data and Evaluation

## Data Contract

### Historical Feature Store

The price/volume-first feature store (`trading_system/data/feature_store.py`) is the primary training dataset. It provides:

- **Point-in-time only**: No feature uses future data (no `shift(-n)` or lookahead). All features are computed from OHLCV up to and including the current bar. Cache keys include a hash of feature names and window params to prevent stale-feature silent serving.
- **Schema**: Per (date, ticker): `return_1d`, `return_5d`, `return_20d`, `vol_20d`, `atr_pct`, `momentum_score`, `mean_reversion_score`, `volatility_score`, `volume_ratio`, and optional rolling features (`rolling_return_5d`, `rolling_return_21d`, `rolling_vol_21d`).
- **Usage**: `build_feature_dataset(tickers, lookback_years)` returns a DataFrame with MultiIndex (date, ticker). `get_features_for_date(feature_df, as_of_date, tickers)` returns one dict per ticker for that date, compatible with the PPO observation schema (with sentiment/quant/fund set to neutral when using price-only).

### Walk-Forward Evaluation

- **Splits**: Chronological train/test. Config: `TEST_WINDOW_DAYS` (default 252) and `TEST_WINDOW_PLACEMENT` (`recent` = hold out newest days, `oldest` = hold out oldest days so training sees recent regimes, `none` = no holdout). No shuffle.
- **Tuning**: Optuna tuning uses only the train window. The test window is never seen during hyperparameter search.
- **Training**: When a split is available, PPO is trained on the train window only. The "training complete" backtest and reported stats use the held-out test window (out-of-sample).
- **Backtest**: The `backtest` command is **out-of-sample by default** — it evaluates on the held-out test window. Use `--in-sample` to evaluate on all available data.
- **Leakage**: Target construction and rolling features use only past data. Use `get_features_for_date(..., as_of_date)` so that inference at date `t` only sees rows with index <= t.
- **Specialist training cutoff**: Specialists (TFT, LightGBM, VAE) are retrained only on data prior to the test window boundary; they never see test-window bars during their own training pass.

## Feature Sources and Observation Schema

The PPO observation has 14 × n_assets dimensions: 13 feature slots × n_assets, followed by n_assets holdings values. The 13 slots are:

| Slot | Name | Range | Mode A (proxy) source | Mode B (specialist) source |
|------|------|-------|----------------------|----------------------------|
| 0 | sentiment | [-1, 1] | OHLCV momentum heuristic | proxy fallback (no historical LLM) |
| 1 | quant_prob | [0, 1] | return-window proxy | proxy fallback (TFT date-mapping limitation, see below) |
| 2 | fund_score | [-1, 1] | OHLCV heuristic | proxy fallback (no historical fundamentals) |
| 3 | regime_flag | [0, 1] | rule-based vol/return proxy | VAE+KMeans walk-forward on SPY |
| 4 | momentum_score | [-1, 1] | return-window proxy | LightGBM walk-forward |
| 5 | mean_reversion | [-1, 1] | return z-score proxy | rule-based: RSI + price z-score vs SMA |
| 6 | volatility | [0, 1] | return-window proxy | rule-based: short/long vol ratio |
| 7 | event_risk | [0, 1] | neutral proxy | proxy fallback (no point-in-time earnings calendar) |
| 8 | relative_strength | [-1, 1] | return-window proxy | rule-based: ticker vs benchmark return |
| 9 | liquidity | [0, 1] | return-window proxy | rule-based: volume percentile |
| 10 | ret_5d_norm | [-1, 1] | returns-based | identical to proxy |
| 11 | ret_21d_norm | [-1, 1] | returns-based | identical to proxy |
| 12 | vol_21d_norm | [-1, 1] | returns-based | identical to proxy |

### FEATURE_SOURCE=proxy (Mode A)

All slots use the precomputed proxy feature matrix from `meta_learner/proxy_specialists.py`. No specialist model inference occurs during training or backtest. Suitable for production where specialist model dependencies (Ollama, yfinance snapshot, historical earnings data) are unavailable.

### FEATURE_SOURCE=specialist (Mode B)

Slots that have real walk-forward historical data (3, 4, 5, 6, 8, 9, 10, 11, 12) use real specialist outputs. Slots 0, 2, 7 fall back to proxy (see reasons below). Slot 1 (quant_prob) currently always falls back to proxy: a known deferred TFT date-mapping limitation in `specialist_features.py` means walk-forward TFT predictions do not map back onto output bars (predictions stay at the neutral 0.5), so the generator substitutes the proxy hit-rate. The backtest also evaluates specialist-mode models through the same specialist matrix path (`_run_backtest` checks the model's `feature_source` metadata).

Specialist features are precomputed via the walk-forward generator before PPO training:
```
python -m trading_system.meta_learner.specialist_features \
    --tickers <TRADE_TICKERS> --start <start_date> --end <end_date>
```
This produces a cached `.npz` matrix and `_meta.json` sidecar in `SPECIALIST_FEATURES_CACHE_DIR`. The PPO trainer loads this cache when `FEATURE_SOURCE=specialist`.

Proxy fallback reasons in Mode B:
- Slot 0 (sentiment): Ollama/LLM unavailable for past dates; news point-in-time reconstruction not feasible.
- Slot 2 (fund_score): yfinance provides only current-snapshot fundamentals, not historical per-date balance-sheet series.
- Slot 7 (event_risk): yfinance earnings calendar is forward-looking; no point-in-time future-earnings dates for arbitrary past bars.
- Slot 1 (quant_prob): deferred TFT date-mapping limitation — the walk-forward TFT trains but its predictions do not map back to output bar indices, leaving all values at neutral 0.5; the generator detects this and substitutes the proxy hit-rate. Fixing the prediction-to-date mapping (or using a shorter-encoder TFT) is a known follow-up.

## Expanding Z-Score Normalization (warm-start)

Both proxy and specialist paths pass all 13 slots through an expanding z-score normalizer. The normalizer accumulates running mean and variance over the training window (Welford online algorithm). At slice boundaries (train → val, train → test), the normalizer state is **warm-started** from the final statistics of the preceding slice to eliminate distribution shift at the boundary. The final running statistics are saved in the model's `.meta.json` sidecar and restored at inference time, so live/paper observations are on the same scale as training observations.

## Model Metadata Sidecar

Every trained `ppo_trader_<timestamp>.zip` is accompanied by `ppo_trader_<timestamp>.zip.meta.json`:

```json
{
  "feature_source": "proxy",
  "tickers": ["IWM", "XLE", ...],
  "norm_running_stats": { "mean": [...], "var": [...], "count": N },
  "n_assets": 30
}
```

Live/paper paths load this sidecar and verify that `feature_source` matches the configured `FEATURE_SOURCE`. A mismatch raises an error before any orders are placed. Legacy models without a sidecar are assumed to be proxy-mode with a WARNING.

## Specialist Feature Generator

`trading_system/meta_learner/specialist_features.py` provides `get_or_build_specialist_matrix(tickers, start, end, ...)` which:

1. Checks the cache for an existing `.npz` file keyed by `SHA-256(sorted_tickers, start, end, SCHEMA_VERSION, refit_every, window)`. On exact-key miss, it also scans for a cached superset matrix (same tickers, refit cadence, window, and schema) whose date range covers the requested window, and slices the rows out — valid because every row is walk-forward (uses only data from earlier bars).
2. On full cache miss, runs the walk-forward loop: for each refit window, fits trained models (VAE, LightGBM, TFT) on data up to the refit date, then computes specialist outputs for subsequent bars.
3. Rule-based agents (mean_reversion, volatility, relative_strength, liquidity) are applied directly with `.shift(1)` to enforce no-lookahead.
4. Saves the result as `{cache_dir}/{hex24}.npz` + `{hex24}_meta.json` under `data/cache/specialist_features/`.

The cache can be pre-generated via CLI before PPO training to avoid the generation overhead (roughly an hour for ~1000 trading days x 30 tickers, dominated by walk-forward TFT refits) during each training run. Thanks to superset slicing, one wide pre-generated matrix serves any narrower train/validation/test window with the same parameters.

## Execution Safeguards

- **Portfolio sizing**: Live, paper, and backtest size targets from **total portfolio value**, not cash. This prevents sell-then-rebuy churn when cash is temporarily low.
- **Holdings**: Current portfolio weights (holdings) are passed into `PPOTrader.predict(..., holdings=...)` in live, paper, and backtest so the policy sees the same observation structure as in training.
- **Rebalance band**: Trades occur only when `|target_value - current_value| > max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)`.
- **Two-pass rebalance (live/paper)**: All sells execute in the first pass; buys in the second pass against the updated `remaining_capital`. Capital is decremented per filled order so late tickers in the list are not silently starved of buying power. This mirrors the training environment's single-bar cost accounting.
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
