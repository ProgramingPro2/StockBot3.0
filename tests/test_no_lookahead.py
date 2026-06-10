"""No-lookahead invariant tests for proxy feature pipeline and specialist_features slots.

For proxy features: features at index t must depend only on returns before t (i.e.
returns[0:t]), so perturbing returns[t:] must not change features[0:t].

For specialist_features rule-based slots (mean_reversion slot 5, volatility slot 6,
relative_strength slot 8, liquidity slot 9): the same invariant must hold on a
synthetic OHLCV frame processed through the walk-forward pipeline.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestProxyNoLookahead(unittest.TestCase):
    """Proxy feature matrix must be causally correct: features[t] use only returns[0:t]."""

    def _build_proxy_matrix(self, returns: np.ndarray) -> np.ndarray:
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", False):
            mat = precompute_proxy_feature_matrix(returns, window=10)
        return mat

    def test_perturb_future_returns_does_not_change_past_features(self) -> None:
        rng = np.random.default_rng(0)
        n_days, n_assets = 80, 3
        returns = rng.normal(0.0, 0.01, size=(n_days, n_assets)).astype(np.float32)

        probe_t = 40  # check that features at t < probe_t are unchanged
        returns_perturbed = returns.copy()
        # Replace all returns from probe_t onwards with random noise
        returns_perturbed[probe_t:] = rng.normal(9.0, 1.0, size=(n_days - probe_t, n_assets)).astype(np.float32)

        mat_orig = self._build_proxy_matrix(returns)
        mat_pert = self._build_proxy_matrix(returns_perturbed)

        # Features at indices < probe_t must be identical (causal)
        np.testing.assert_array_equal(
            mat_orig[:probe_t],
            mat_pert[:probe_t],
            err_msg="Proxy features at t<probe_t changed when only returns[probe_t:] were perturbed",
        )

    def test_perturb_single_future_bar_leaves_prior_features_unchanged(self) -> None:
        rng = np.random.default_rng(7)
        n_days, n_assets = 60, 2
        returns = rng.normal(0.0, 0.01, size=(n_days, n_assets)).astype(np.float32)

        perturbed = returns.copy()
        perturbed[50:] = 100.0  # extreme values far in the future

        mat_orig = self._build_proxy_matrix(returns)
        mat_pert = self._build_proxy_matrix(perturbed)

        np.testing.assert_array_equal(mat_orig[:50], mat_pert[:50])

    def test_zscore_causal_with_warmstart(self) -> None:
        """With expanding z-score enabled, features[t] must still only use stats from rows <= t."""
        from trading_system.meta_learner.proxy_specialists import precompute_proxy_feature_matrix

        rng = np.random.default_rng(42)
        n_days, n_assets = 100, 2
        returns = rng.normal(0.0, 0.01, size=(n_days, n_assets)).astype(np.float32)

        probe_t = 60
        perturbed = returns.copy()
        perturbed[probe_t:] = rng.normal(5.0, 0.5, size=(n_days - probe_t, n_assets)).astype(np.float32)

        # z-score is expanding and forward-causal: stats at t use only rows 0..t
        with patch("trading_system.meta_learner.proxy_specialists.PROXY_EXPANDING_ZSCORE", True):
            mat_orig = precompute_proxy_feature_matrix(returns, window=10)
            mat_pert = precompute_proxy_feature_matrix(perturbed, window=10)

        np.testing.assert_array_equal(mat_orig[:probe_t], mat_pert[:probe_t])


class TestSpecialistFeaturesNoLookahead(unittest.TestCase):
    """Rule-based specialist slots must be shift(1)-safe: bar t uses close prices <= t-1."""

    def _make_ohlcv(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
        volumes = rng.integers(100_000, 1_000_000, n).astype(float)
        highs = closes * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
        lows = closes * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
        opens = closes * (1.0 + rng.normal(0.0, 0.002, n))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )

    def test_rule_based_mean_reversion_slot_causal(self) -> None:
        """specialist_features mean-reversion slot uses shift(1) so row t uses close[t-1]."""
        from trading_system.meta_learner.specialist_features import _mean_reversion_df

        rng = np.random.default_rng(1)
        n = 80
        df = self._make_ohlcv(n, rng)

        probe_t = 50
        df_pert = df.copy()
        df_pert.iloc[probe_t:, df_pert.columns.get_loc("close")] = 9999.0

        # Build close DataFrames (single-column)
        close = df[["close"]]
        close_pert = df_pert[["close"]]

        slot_orig = _mean_reversion_df(close)
        slot_pert = _mean_reversion_df(close_pert)

        # Rows strictly before probe_t must be identical due to shift(1)
        np.testing.assert_array_equal(
            slot_orig.values[:probe_t],
            slot_pert.values[:probe_t],
            err_msg="mean_reversion slot at t<probe_t changed when close[probe_t:] was perturbed",
        )

    def test_rule_based_volatility_slot_causal(self) -> None:
        from trading_system.meta_learner.specialist_features import _volatility_df

        rng = np.random.default_rng(2)
        n = 80
        df = self._make_ohlcv(n, rng)
        probe_t = 50
        df_pert = df.copy()
        df_pert.iloc[probe_t:, df_pert.columns.get_loc("close")] = 9999.0

        slot_orig = _volatility_df(df[["close"]])
        slot_pert = _volatility_df(df_pert[["close"]])
        np.testing.assert_array_equal(slot_orig.values[:probe_t], slot_pert.values[:probe_t])

    def test_rule_based_relative_strength_slot_causal(self) -> None:
        from trading_system.meta_learner.specialist_features import _relative_strength_df

        rng = np.random.default_rng(3)
        n = 80
        ticker_df = self._make_ohlcv(n, rng)
        bench_df = self._make_ohlcv(n, rng)
        probe_t = 50

        ticker_pert = ticker_df.copy()
        ticker_pert.iloc[probe_t:, ticker_pert.columns.get_loc("close")] = 9999.0

        close = ticker_df[["close"]]
        close_pert = ticker_pert[["close"]]
        bench = bench_df["close"]

        slot_orig = _relative_strength_df(close, bench)
        slot_pert = _relative_strength_df(close_pert, bench)
        np.testing.assert_array_equal(slot_orig.values[:probe_t], slot_pert.values[:probe_t])

    def test_rule_based_liquidity_slot_causal(self) -> None:
        from trading_system.meta_learner.specialist_features import _liquidity_df

        rng = np.random.default_rng(4)
        n = 80
        df = self._make_ohlcv(n, rng)
        probe_t = 50
        df_pert = df.copy()
        df_pert.iloc[probe_t:, df_pert.columns.get_loc("volume")] = 1e12

        slot_orig = _liquidity_df(df[["volume"]])
        slot_pert = _liquidity_df(df_pert[["volume"]])
        np.testing.assert_array_equal(slot_orig.values[:probe_t], slot_pert.values[:probe_t])
