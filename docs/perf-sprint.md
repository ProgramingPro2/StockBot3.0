# Performance sprint: training and HPO

This document summarizes optimizations applied and how to reproduce baseline vs optimized wall-clock measurements.

## Environment variables

| Variable | Effect |
|----------|--------|
| `BENCH_TRACE=1` | Log `boardroom.bench` timing blocks around PPO `learn()` and env construction. |
| `PPO_USE_DUMMY_VEC=1` | Use `DummyVecEnv` instead of `SubprocVecEnv` (lower IPC overhead when env step is cheap). |
| `PPO_TRAIN_USE_FULL_TUNED_N_STEPS=1` | Do not clamp tuned `n_steps` to `PPO_N_STEPS` when loading `ppo_tune_best.json`. |
| `PPO_TORCH_COMPILE=1` | Apply `torch.compile` to the PPO policy when `PPO_DEVICE=cuda`. |
| `TRAIN_SPECIALISTS_PARALLEL=1` | Run specialist `ensure_*` in parallel threads (default on). Set to `0` for sequential. |
| `OPTUNA_PRUNER=median` | Use `MedianPruner` instead of default `HyperbandPruner`. |
| `XGB_TREE_METHOD` | XGBoost `tree_method` (default `hist`). |
| `XGB_DEVICE` | `cpu` or `cuda` for fundamental XGBoost. |
| `XGB_N_JOBS` | Thread count for XGBoost (`0` = omit, use XGBoost default). |
| `LIGHTGBM_N_JOBS` | `n_jobs` for momentum LightGBM (default `-1`). |
| `TFT_BATCH_SIZE` | TFT dataloader batch size (default `32`). |
| `TFT_DATALOADER_NUM_WORKERS` | DataLoader workers (default `0`). |
| `TFT_DATALOADER_PERSISTENT=1` | Enable `persistent_workers` when `num_workers>0`. |
| `TFT_MAX_EPOCHS` | TFT trainer max epochs (default `10`). |
| `TFT_PRECISION` | Lightning precision: `16-mixed`, `bf16-mixed`, or `32` if set. |
| `REGIME_VAE_BATCH_SIZE` | Regime VAE batch size (default `64`). |
| `REGIME_VAE_EPOCHS` | Regime VAE epochs (default `50`). |

## HPO behavior changes

- **Eval frequency**: Sharpe eval runs `n_evaluations` times per trial: `eval_freq = n_timesteps // (n_evaluations * n_envs)`.
- **Optuna pruning**: Intermediate `trial.report(sharpe, step)` after each eval; default pruner is **Hyperband** (`OPTUNA_PRUNER=median` restores median).
- **OMP threads**: `OMP_NUM_THREADS=1` is set when `n_envs>1` in the tuning objective (matches training).
- **Proxy matrix**: Precomputed once per study for train and validation slices; passed into envs and eval callback to avoid repeated O(n_days) Python loops per worker/eval.

## Validation eval

`SharpeEvalCallback` keeps one `HistoricalTradingEnv` for the validation slice and resets cursor each eval instead of reconstructing the env (avoids repeated proxy matrix allocation).

## Baseline commands (wall-clock)

From repo root with `.venv` activated:

```bash
# Training (includes specialists + PPO)
time python -m trading_system train --timesteps 50000 --use-historical

# HPO
time python -m trading_system tune --n-trials 5 --timesteps 50000

# Device comparison (optional tuned hyperparameters)
python -m trading_system benchmark-device --timesteps 20000 --use-historical --use-tuned
```

## Sweep ideas (manual)

- `PPO_USE_DUMMY_VEC=1` vs unset with your `PPO_N_ENVS`.
- `PPO_DEVICE=cpu` vs `cuda` for meta-learner only (`.env`).
- `TRAIN_SPECIALISTS_PARALLEL=0` vs `1` when debugging GPU OOM.

## Results table

Fill in locally after runs:

| Experiment | Wall-clock (s) | Notes |
|------------|----------------|-------|
| B-train baseline | | |
| B-tune baseline | | |

## Implemented in code (sprint summary)

- Optuna: intermediate `trial.report` from `SharpeEvalCallback`, fixed `eval_freq`, `OMP_NUM_THREADS=1` when `n_envs>1`, default **Hyperband** pruner, train/val proxy matrix precomputed once per study.
- Envs: optional shared `proxy_feature_cache` for `HistoricalTradingEnv`; no per-step `.copy()` when aux is absent; vectorized `merge_daily_aux_into_proxy_matrix`; vectorized sliding-window `precompute_proxy_feature_matrix`.
- `SharpeEvalCallback`: one persistent validation env + `close_eval_env` cleanup on prune/failure.
- Training: `DummyVecEnv` when `PPO_USE_DUMMY_VEC=1`; `torch.compile` when `PPO_TORCH_COMPILE=1`; train proxy cache from CLI; optional full tuned `n_steps` via `PPO_TRAIN_USE_FULL_TUNED_N_STEPS=1`.
- Specialists: parallel `ensure_*` (default on, `TRAIN_SPECIALISTS_PARALLEL=0` off); TFT / XGB / LightGBM / regime VAE knobs via env vars above.
- `benchmark-device`: `--use-tuned` passes loaded hyperparameters into the benchmark.
