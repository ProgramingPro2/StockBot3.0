"""precompute_proxy_feature_matrix: vectorized path matches legacy semantics."""
import unittest

import numpy as np

from trading_system.meta_learner.proxy_specialists import (
    precompute_proxy_feature_matrix,
    proxy_specialists_vectorized,
)


class TestProxyPrecomputeVectorized(unittest.TestCase):
    def test_batch_matches_row_loop(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.012, size=(180, 4)).astype(np.float32)
        window = 21
        out = precompute_proxy_feature_matrix(returns, window=window)
        for t in range(window, returns.shape[0]):
            w = returns[t - window : t, :]
            expected = proxy_specialists_vectorized(w)
            np.testing.assert_allclose(out[t], expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(np.all(out[:window] == 0))
