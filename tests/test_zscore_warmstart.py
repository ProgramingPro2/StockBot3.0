"""Z-score warm-start correctness tests.

1. Full-sequence z-score vs train-slice + warm-started val-slice must produce
   identical val rows (no cold-start distribution shift at the boundary).

2. Stats round-trip through JSON serialization: serialize -> deserialize ->
   continue updating must give the same results as uninterrupted computation.
"""
from __future__ import annotations

import json
import unittest

import numpy as np


class TestExpandingZscoreWarmstart(unittest.TestCase):
    """_expanding_zscore_3d warm-start produces identical val rows to full-sequence run."""

    @staticmethod
    def _zscore(mat, warmstart_stats=None, return_stats=False):
        from trading_system.meta_learner.proxy_specialists import _expanding_zscore_3d
        return _expanding_zscore_3d(mat, warmstart_stats=warmstart_stats, return_stats=return_stats)

    def test_warmstart_val_matches_full_sequence(self) -> None:
        rng = np.random.default_rng(0)
        n, m, f = 200, 3, 13
        mat = rng.normal(0.0, 1.0, size=(n, m, f)).astype(np.float32)
        split = 140

        # Full sequence
        full_out = self._zscore(mat)

        # Split: train slice, then warm-start val slice
        train_out, train_stats = self._zscore(mat[:split], return_stats=True)
        val_out = self._zscore(mat[split:], warmstart_stats=train_stats)

        # val rows must match the corresponding rows of the full-sequence output
        np.testing.assert_allclose(
            full_out[split:], val_out,
            rtol=1e-5, atol=1e-5,
            err_msg="Warm-started val slice does not match full-sequence output",
        )

    def test_warmstart_train_rows_match_full_sequence(self) -> None:
        rng = np.random.default_rng(1)
        n, m, f = 150, 2, 13
        mat = rng.normal(0.0, 0.5, size=(n, m, f)).astype(np.float32)
        split = 100

        full_out = self._zscore(mat)
        train_out, _ = self._zscore(mat[:split], return_stats=True)

        # Train rows must also match
        np.testing.assert_allclose(
            full_out[:split], train_out,
            rtol=1e-5, atol=1e-5,
            err_msg="Train-slice output does not match full-sequence train rows",
        )

    def test_stats_json_roundtrip(self) -> None:
        """Stats serialized to JSON and deserialized must continue updating identically."""
        rng = np.random.default_rng(2)
        n, m, f = 180, 3, 13
        mat = rng.normal(0.0, 1.0, size=(n, m, f)).astype(np.float32)
        split = 120

        _, stats_before_json = self._zscore(mat[:split], return_stats=True)

        # Simulate JSON round-trip (as stored in the model metadata sidecar)
        json_str = json.dumps(stats_before_json)
        stats_after_json = json.loads(json_str)

        # Ensure numpy deserialization works
        stats_after_json["running_sum"] = np.asarray(
            stats_after_json["running_sum"], dtype=np.float64
        ).tolist()
        stats_after_json["running_sq"] = np.asarray(
            stats_after_json["running_sq"], dtype=np.float64
        ).tolist()

        # Continue from both stat dicts
        val_direct = self._zscore(mat[split:], warmstart_stats=stats_before_json)
        val_roundtrip = self._zscore(mat[split:], warmstart_stats=stats_after_json)

        np.testing.assert_allclose(
            val_direct, val_roundtrip,
            rtol=1e-5, atol=1e-5,
            err_msg="JSON round-tripped stats do not produce identical output",
        )

    def test_stats_roundtrip_matches_full_sequence(self) -> None:
        """Full pipeline: compute train stats, round-trip through JSON, apply to val, check vs full."""
        rng = np.random.default_rng(3)
        n, m, f = 200, 4, 13
        mat = rng.normal(0.0, 0.8, size=(n, m, f)).astype(np.float32)
        split = 130

        full_out = self._zscore(mat)
        _, stats = self._zscore(mat[:split], return_stats=True)

        # JSON round-trip
        stats = json.loads(json.dumps(stats))
        val_out = self._zscore(mat[split:], warmstart_stats=stats)

        np.testing.assert_allclose(
            full_out[split:], val_out,
            rtol=1e-5, atol=1e-5,
            err_msg="JSON round-trip stats warm-start does not match full sequence",
        )

    def test_no_warmstart_is_cold_start(self) -> None:
        """Without warmstart, val slice is cold — must differ from full-sequence val rows."""
        rng = np.random.default_rng(4)
        n, m, f = 200, 2, 13
        mat = rng.normal(0.0, 1.0, size=(n, m, f)).astype(np.float32)
        split = 100

        full_out = self._zscore(mat)
        cold_val_out = self._zscore(mat[split:])

        # Cold val must be different (distribution shift from cold start)
        # We check that at least some rows differ significantly within min_periods
        diff = np.abs(full_out[split:] - cold_val_out)
        # The first ~63 rows of cold val run from count=0, full run from count=split;
        # so the outputs must differ
        self.assertGreater(
            float(diff[:63].max()), 0.01,
            "Cold start val unexpectedly matches full sequence — warm-start might be misapplied",
        )


class TestZscorePrecomputeWarmstart(unittest.TestCase):
    """precompute_proxy_feature_matrix warm-start via return_stats parameter.

    The warmstart mechanism seeds the expanding z-score stats from the end of the
    train slice.  Raw features in the val slice are computed solely from val data,
    so the first `window` rows of the val slice remain zero (warmup) even with warmstart —
    identical to a cold-start for raw features.  The benefit of warmstart is that the
    z-score normalization does not reset to cold (count=0) at the slice boundary; instead
    it continues from the train-slice statistics.

    This test verifies the following achievable guarantees:
    1. The returned stats dict correctly reflects the full accumulated count.
    2. After the warmup window, the val-slice z-scored features are self-consistent
       and not NaN/Inf.
    3. The stat round-trip (serialize -> deserialize -> use) gives identical output
       as using the in-memory stats dict.
    """

    def test_return_stats_count_is_correct(self) -> None:
        """Returned stats count equals the number of rows processed by precompute."""
        from unittest.mock import patch
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        rng = np.random.default_rng(10)
        n, m = 200, 3
        returns = rng.normal(0.0, 0.01, size=(n, m)).astype(np.float32)

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", True):
            _, stats = precompute_proxy_feature_matrix(returns, return_stats=True)

        self.assertEqual(stats["count"], n)

    def test_warmstart_val_finite_after_window(self) -> None:
        """Val output past warmup rows must be finite with warmstart stats."""
        from unittest.mock import patch
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        rng = np.random.default_rng(11)
        n, m = 300, 3
        returns = rng.normal(0.0, 0.01, size=(n, m)).astype(np.float32)
        split = 200
        window = 21

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", True):
            _, train_stats = precompute_proxy_feature_matrix(returns[:split], return_stats=True)
            val_mat = precompute_proxy_feature_matrix(returns[split:], warmstart_stats=train_stats)

        post_warmup = val_mat[window:]
        self.assertTrue(
            np.all(np.isfinite(post_warmup)),
            "Val features after warmup window must all be finite with warmstart",
        )

    def test_warmstart_vs_coldstart_stat_continuity(self) -> None:
        """Warmstart count must be higher than cold-start count at same val bar."""
        from unittest.mock import patch
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        rng = np.random.default_rng(12)
        n, m = 300, 3
        returns = rng.normal(0.0, 0.01, size=(n, m)).astype(np.float32)
        split = 200

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", True):
            _, train_stats = precompute_proxy_feature_matrix(returns[:split], return_stats=True)
            _, val_stats_warm = precompute_proxy_feature_matrix(
                returns[split:], warmstart_stats=train_stats, return_stats=True
            )
            _, val_stats_cold = precompute_proxy_feature_matrix(
                returns[split:], warmstart_stats=None, return_stats=True
            )

        self.assertEqual(val_stats_warm["count"], split + (n - split))
        self.assertEqual(val_stats_cold["count"], n - split)
        self.assertGreater(val_stats_warm["count"], val_stats_cold["count"])

    def test_precompute_warmstart_stat_roundtrip(self) -> None:
        """JSON round-tripped stats produce identical val output as in-memory stats."""
        import json
        from unittest.mock import patch
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        rng = np.random.default_rng(13)
        n, m = 300, 3
        returns = rng.normal(0.0, 0.01, size=(n, m)).astype(np.float32)
        split = 200

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", True):
            _, train_stats = precompute_proxy_feature_matrix(returns[:split], return_stats=True)
            train_stats_roundtrip = json.loads(json.dumps(train_stats))

            val_direct = precompute_proxy_feature_matrix(
                returns[split:], warmstart_stats=train_stats
            )
            val_roundtrip = precompute_proxy_feature_matrix(
                returns[split:], warmstart_stats=train_stats_roundtrip
            )

        np.testing.assert_allclose(
            val_direct, val_roundtrip, rtol=1e-5, atol=1e-5,
            err_msg="JSON round-tripped stats produce different val output than in-memory stats",
        )
