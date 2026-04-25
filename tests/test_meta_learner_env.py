"""Tests for meta-learner env: action normalization and reward stability."""
import unittest

import numpy as np

from trading_system.meta_learner.env import HistoricalTradingEnv, TradingEnv
from trading_system.meta_learner.proxy_specialists import normalize_action_max_unit_exposure


class TestNormalizeAction(unittest.TestCase):
    def test_sum_over_one_scales_down(self) -> None:
        a = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        n = normalize_action_max_unit_exposure(a)
        self.assertAlmostEqual(float(np.sum(n)), 1.0, places=5)
        self.assertTrue(np.all(n <= 1.0) and np.all(n >= 0.0))

    def test_sum_under_one_unchanged(self) -> None:
        a = np.array([0.2, 0.3, 0.1], dtype=np.float32)
        n = normalize_action_max_unit_exposure(a)
        np.testing.assert_allclose(n, a, rtol=1e-5)


class TestHistoricalEnvDeterminism(unittest.TestCase):
    def test_same_seed_same_episode_start(self) -> None:
        rng = np.random.default_rng(42)
        ret = rng.normal(0, 0.01, size=(400, 2)).astype(np.float32)
        e1 = HistoricalTradingEnv(returns_array=ret, n_assets=2)
        e2 = HistoricalTradingEnv(returns_array=ret, n_assets=2)
        o1, _ = e1.reset(seed=123)
        o2, _ = e2.reset(seed=123)
        np.testing.assert_allclose(o1, o2, rtol=1e-5)


class TestTradingEnvBootstrap(unittest.TestCase):
    def test_bootstrap_returns_used(self) -> None:
        env = TradingEnv(n_assets=2, episode_days=3)
        block = np.array([[0.01, 0.02], [-0.01, 0.0], [0.0, 0.01], [0.02, -0.01]], dtype=np.float32)
        env.reset(seed=0, options={"bootstrap_returns": block, "specialist_outputs": [{}, {}]})
        _, r, term, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        self.assertIn("return", info)
        self.assertFalse(term)
