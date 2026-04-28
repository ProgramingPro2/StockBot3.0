"""Proxy specialists and precomputed backtest path consistency."""
import unittest

import numpy as np
import pandas as pd

from trading_system.meta_learner.proxy_specialists import (
    precompute_proxy_feature_matrix,
    proxy_specialists_from_returns_series,
    sentiment_proxy_from_close_prices,
    tuple_to_specialist_dict,
)


class TestProxyConsistency(unittest.TestCase):
    def test_precompute_matches_series(self) -> None:
        n = 80
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, size=n).astype(np.float64)
        s = pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        window = 21
        mat = np.asarray(s.values, dtype=np.float32).reshape(-1, 1)
        pre = precompute_proxy_feature_matrix(mat, window=window)
        t = 50
        d_series = proxy_specialists_from_returns_series(s, t, window)
        d_mat = tuple_to_specialist_dict(tuple(float(x) for x in pre[t, 0]))
        for k in ("sentiment", "quant_prob", "fund_score", "momentum_score"):
            self.assertAlmostEqual(d_series[k], d_mat[k], places=5)

    def test_sentiment_proxy_from_close_matches_tail_mean(self) -> None:
        rng = np.random.default_rng(1)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=30))
        s = pd.Series(close)
        ret = s.pct_change().dropna().tail(21)
        mean_ret = float(ret.mean())
        expected = float(max(-0.5, min(0.5, mean_ret * 12)))
        self.assertAlmostEqual(sentiment_proxy_from_close_prices(s), expected, places=10)

    def test_sentiment_proxy_short_series_zero(self) -> None:
        self.assertEqual(sentiment_proxy_from_close_prices(pd.Series([100.0, 101.0])), 0.0)
