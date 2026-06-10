"""Proxy feature matrix shape, dtype, and property tests.

Verifies:
- Shape is (n_days, n_assets, 13) with dtype float32
- Scalar (single asset) vs multi-asset batch parity
- Distinct slot values on trending data
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np


class TestProxyFeatureMatrixShape(unittest.TestCase):
    """precompute_proxy_feature_matrix shape and dtype contract."""

    def _build(self, returns, zscore=False):
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix
        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", zscore):
            return precompute_proxy_feature_matrix(returns, window=21)

    def test_shape_matches_n_days_n_assets_13(self) -> None:
        rng = np.random.default_rng(0)
        n_days, n_assets = 150, 4
        returns = rng.normal(0.0, 0.01, size=(n_days, n_assets)).astype(np.float32)
        out = self._build(returns)
        self.assertEqual(out.shape, (n_days, n_assets, 13))

    def test_dtype_float32(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0, 0.01, size=(100, 2)).astype(np.float32)
        out = self._build(returns)
        self.assertEqual(out.dtype, np.float32)

    def test_warmup_rows_are_zero(self) -> None:
        rng = np.random.default_rng(2)
        window = 21
        returns = rng.normal(0.0, 0.01, size=(120, 3)).astype(np.float32)
        out = self._build(returns)
        np.testing.assert_array_equal(
            out[:window], np.zeros((window, 3, 13), dtype=np.float32),
            err_msg="Warmup rows (t < window) must be zero",
        )

    def test_single_asset_shape(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0, 0.01, size=(100, 1)).astype(np.float32)
        out = self._build(returns)
        self.assertEqual(out.shape, (100, 1, 13))

    def test_all_finite_after_warmup(self) -> None:
        rng = np.random.default_rng(4)
        window = 21
        returns = rng.normal(0.0, 0.01, size=(150, 3)).astype(np.float32)
        out = self._build(returns)
        self.assertTrue(np.all(np.isfinite(out[window:])),
                        "All post-warmup features must be finite")


class TestProxyFeatureMatrixScalarVsBatch(unittest.TestCase):
    """Single-asset (scalar) must match the same column from multi-asset (vectorized batch)."""

    def _build_no_zscore(self, returns):
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix
        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", False):
            return precompute_proxy_feature_matrix(returns, window=21)

    def test_scalar_vs_batch_col0(self) -> None:
        rng = np.random.default_rng(10)
        n_days = 120
        returns_3 = rng.normal(0.0, 0.01, size=(n_days, 3)).astype(np.float32)
        returns_1 = returns_3[:, :1]

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_XSRANK_ENABLED", False):
            out_3 = self._build_no_zscore(returns_3)
            out_1 = self._build_no_zscore(returns_1)

        # Cross-sectional slot 8 (relative_strength) depends on multi-asset context;
        # for single asset it returns 0.  All other slots must match col 0.
        non_cross_slots = [i for i in range(13) if i != 8]
        np.testing.assert_allclose(
            out_3[:, 0:1, non_cross_slots],
            out_1[:, 0:1, non_cross_slots],
            rtol=1e-5, atol=1e-5,
            err_msg="Single-asset and first-column of multi-asset batch differ for non-cross-sectional slots",
        )


class TestProxyFeatureDistinctOnTrendingData(unittest.TestCase):
    """Slot values must be meaningfully distinct on trending data (not all the same)."""

    def _build_no_zscore(self, returns):
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix
        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", False):
            return precompute_proxy_feature_matrix(returns, window=21)

    def test_distinct_slot_values_on_trending_data(self) -> None:
        n_days = 120
        # Strong upward trend: mean daily return > 0
        returns = np.full((n_days, 2), 0.005, dtype=np.float32)
        returns[50:70, 0] = -0.01  # brief reversal in first asset
        out = self._build_no_zscore(returns)

        # Check that at least some rows past warmup have non-zero distinct slot values
        post_warmup = out[21:]
        n_slots = post_warmup.shape[2]
        for slot_idx in range(n_slots):
            slot_vals = post_warmup[:, :, slot_idx].ravel()
            # Each slot should have some variation (not all constant 0.5 or 0)
            std = float(np.std(slot_vals))
            # At least some slots should be non-trivial
            # (liquidity slot 9 is always 0.5 in proxy path — skip)
            if slot_idx == 9:
                continue
            # Don't require ALL slots to vary; just check the output is non-trivially shaped
        # Confirm at least half the slots have some variation
        varying_slots = sum(
            1 for s in range(n_slots) if s != 9 and float(np.std(post_warmup[:, :, s].ravel())) > 1e-6
        )
        self.assertGreaterEqual(varying_slots, 8,
                                f"Expected at least 8 varying slots but got {varying_slots}")

    def test_slots_within_documented_ranges(self) -> None:
        """Verify that all slot values for non-warmup rows stay within [-1, 1] or [0, 1]."""
        rng = np.random.default_rng(20)
        n_days, n_assets = 200, 3
        returns = rng.normal(0.0, 0.01, size=(n_days, n_assets)).astype(np.float32)
        out = self._build_no_zscore(returns)
        post = out[21:]

        # All slots clipped to [-1, 1] by convention
        self.assertLessEqual(float(post.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(post.min()), -1.0 - 1e-5)

        # Slots documented as [0, 1]: 1, 3, 6, 7, 9
        for slot in [1, 3, 6, 7, 9]:
            vals = post[:, :, slot]
            self.assertGreaterEqual(float(vals.min()), -1e-5,
                                    f"Slot {slot} has negative values")
