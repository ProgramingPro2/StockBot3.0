"""Parity between HistoricalTradingEnv and BatchedHistoricalTorchVecEnv (CPU)."""
from __future__ import annotations

import numpy as np

from trading_system.meta_learner.env import HistoricalTradingEnv
from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix
from trading_system.meta_learner.torch_vec_env import BatchedHistoricalTorchVecEnv


def _random_actions(rng: np.random.Generator, n_assets: int) -> np.ndarray:
    a = rng.random(n_assets, dtype=np.float32)
    return a


def test_batched_torch_vec_matches_historical_many_steps() -> None:
    rng = np.random.default_rng(0)
    n_days, n_assets = 500, 3
    returns = rng.normal(0.0, 0.012, size=(n_days, n_assets)).astype(np.float32)
    proxy = precompute_proxy_feature_matrix(returns)

    hist = HistoricalTradingEnv(returns, n_assets, proxy_feature_cache=proxy)
    vec = BatchedHistoricalTorchVecEnv(returns, proxy, n_assets, 1, device="cpu")

    obs_h, _ = hist.reset(seed=123)
    vec._rng = np.random.default_rng(123)
    vec._reset_one(0)
    obs_v = vec._build_obs_batch()[0]
    np.testing.assert_allclose(obs_h, obs_v, rtol=0, atol=1e-4)

    step_rng = np.random.default_rng(1)
    for _ in range(200):
        a = _random_actions(step_rng, n_assets)
        obs_h, r_h, term_h, trunc_h, _ = hist.step(a)
        done_h = term_h or trunc_h

        vec.step_async(a.reshape(1, -1))
        obs_v, r_v, done_v, _ = vec.step_wait()
        assert abs(float(r_h) - float(r_v[0])) < 1e-4
        assert bool(done_h) == bool(done_v[0])
        if not done_h:
            np.testing.assert_allclose(obs_h, obs_v[0], rtol=0, atol=1e-4)
        if done_h:
            break
