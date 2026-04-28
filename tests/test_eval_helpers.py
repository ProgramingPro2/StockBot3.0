"""Tests for deterministic validation evaluation helpers."""
import unittest

import numpy as np

from trading_system.meta_learner.eval_helpers import deterministic_validation_walk


class _FixedWeightModel:
    """Minimal predict-compatible model used for deterministic eval tests."""

    def __init__(self, n_assets: int):
        self._action = np.ones(n_assets, dtype=np.float32) / float(n_assets)

    def predict(self, obs, deterministic: bool = True):
        return self._action.copy(), None


class TestEvalHelpers(unittest.TestCase):
    def test_deterministic_validation_walk_returns_finite_sharpe(self) -> None:
        rng = np.random.default_rng(7)
        val_returns = rng.normal(0.0005, 0.01, size=(320, 3)).astype(np.float32)
        model = _FixedWeightModel(n_assets=3)

        sharpe, daily = deterministic_validation_walk(model, val_returns, n_assets=3)

        self.assertTrue(np.isfinite(sharpe))
        self.assertGreater(len(daily), 0)
        self.assertTrue(np.all(np.isfinite(daily)))

    def test_deterministic_validation_walk_is_reproducible(self) -> None:
        rng = np.random.default_rng(123)
        val_returns = rng.normal(0.0, 0.01, size=(300, 2)).astype(np.float32)
        model = _FixedWeightModel(n_assets=2)

        sharpe_a, daily_a = deterministic_validation_walk(model, val_returns, n_assets=2)
        sharpe_b, daily_b = deterministic_validation_walk(model, val_returns, n_assets=2)

        self.assertAlmostEqual(sharpe_a, sharpe_b, places=10)
        np.testing.assert_allclose(daily_a, daily_b, rtol=0.0, atol=0.0)

