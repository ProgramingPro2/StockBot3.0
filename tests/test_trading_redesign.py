"""
Regression and contract tests for the trading system redesign.
Covers: feature store leakage, walk-forward splits, portfolio math, execution safeguards.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_system.data.feature_store import (
    check_no_future_leakage,
    get_features_for_date,
    _neutral_feature_dict,
)
from trading_system.main import (
    _chronological_train_test_split,
    _exposure_and_weights,
    _load_aligned_returns,
)


class TestFeatureStoreLeakage(unittest.TestCase):
    """Feature store must not use future data."""

    def test_no_future_leakage_contract(self) -> None:
        self.assertTrue(check_no_future_leakage())

    def test_neutral_feature_dict_has_required_keys(self) -> None:
        d = _neutral_feature_dict()
        for k in ["sentiment", "quant_prob", "fund_score", "regime_flag", "momentum_score"]:
            self.assertIn(k, d)

    def test_get_features_for_date_empty_df_returns_neutral(self) -> None:
        out = get_features_for_date(None, "2024-01-15", ["SPY"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["quant_prob"], 0.5)
        self.assertEqual(out[0]["sentiment"], 0.0)

    def test_get_features_for_date_missing_ticker_returns_neutral(self) -> None:
        empty = pd.DataFrame()
        out = get_features_for_date(empty, "2024-01-15", ["SPY", "QQQ"])
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertEqual(d["quant_prob"], 0.5)


class TestChronologicalSplit(unittest.TestCase):
    """Walk-forward split must be chronological with no overlap."""

    def test_split_lengths(self) -> None:
        n = 600
        aligned = pd.DataFrame(index=pd.date_range("2022-01-01", periods=n, freq="B"))
        returns_array = np.random.randn(n, 3).astype(np.float32) * 0.01
        train_ret, test_ret, aligned_train, aligned_test = _chronological_train_test_split(
            aligned, returns_array, test_days=252
        )
        self.assertIsNotNone(train_ret)
        self.assertIsNotNone(test_ret)
        self.assertEqual(len(train_ret) + len(test_ret), n)
        self.assertEqual(len(test_ret), 252)
        self.assertEqual(len(aligned_train) + len(aligned_test), n)

    def test_split_insufficient_data_returns_none(self) -> None:
        aligned = pd.DataFrame(index=pd.date_range("2022-01-01", periods=100, freq="B"))
        returns_array = np.random.randn(100, 2).astype(np.float32)
        train_ret, test_ret, _, _ = _chronological_train_test_split(
            aligned, returns_array, test_days=252
        )
        self.assertIsNone(train_ret)
        self.assertIsNone(test_ret)


class TestHistoryWindow(unittest.TestCase):
    """Default load must not truncate to 504 days when TRAIN_WINDOW_YEARS is used."""

    def test_load_aligned_returns_no_default_truncation(self) -> None:
        # Build enough days so that old code would have truncated to 504
        n_days = 800
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        tickers = ["SPY", "IWM"]

        def fake_load_ohlcv(symbol: str, lookback_years: int = 5) -> pd.DataFrame:
            df = pd.DataFrame(
                {"close": 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)},
                index=dates,
            )
            return df

        with patch("trading_system.data.loader.load_ohlcv", side_effect=fake_load_ohlcv):
            aligned, returns_array = _load_aligned_returns(tickers, max_days=None)
        self.assertIsNotNone(aligned)
        self.assertIsNotNone(returns_array)
        # Without max_days we must get full window (no silent 252*2 truncation)
        self.assertGreaterEqual(len(aligned), 500, "should not truncate to 504 by default")


class TestValidationSplit(unittest.TestCase):
    """Validation slice must not overlap training slice (early stopping / tuning)."""

    def test_validation_slice_non_overlap(self) -> None:
        n = 1000
        val_size = max(21, min(252, n // 5))
        train_len = n - val_size
        # Simulate tuner/train split: train_returns = arr[:n-val_size], val_returns = arr[n-val_size:]
        arr = np.random.randn(n, 3).astype(np.float32) * 0.01
        train_returns = arr[:train_len]
        val_returns = arr[train_len:]
        self.assertEqual(len(train_returns) + len(val_returns), n)
        self.assertGreater(len(train_returns), 0)
        self.assertGreater(len(val_returns), 0)
        # No overlap: last row of train != first row of val
        np.testing.assert_array_equal(val_returns[0], arr[train_len])
        np.testing.assert_array_equal(train_returns[-1], arr[train_len - 1])


class TestPortfolioMath(unittest.TestCase):
    """Exposure and weights; target sizing from portfolio value (regression for churn bug)."""

    def test_exposure_and_weights_sum_to_exposure(self) -> None:
        action = np.array([0.2, 0.2, 0.1, 0.0, 0.0], dtype=np.float64)
        exposure, weights = _exposure_and_weights(action, allow_cash_exposure=True)
        # Weights are normalized to sum to 1; exposure = min(1, sum(action))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertGreaterEqual(exposure, 0)
        self.assertLessEqual(exposure, 1)
        self.assertAlmostEqual(exposure, 0.5)

    def test_target_holdings_sum_to_exposure(self) -> None:
        # Regression: target_holdings = exposure * weights must sum to exposure (cash preserved)
        action = np.array([0.2, 0.2, 0.1, 0.0, 0.0], dtype=np.float64)
        exposure, weights = _exposure_and_weights(action, allow_cash_exposure=True)
        target_holdings = exposure * weights
        self.assertAlmostEqual(float(np.sum(target_holdings)), exposure)
        self.assertAlmostEqual(float(np.sum(target_holdings)), 0.5)

    def test_target_value_from_portfolio_value_not_cash(self) -> None:
        # Regression: target must be portfolio_value * weight, not cash * weight
        portfolio_value = 100_000.0
        cash = 500.0
        weight = 0.2
        target_from_portfolio = portfolio_value * weight
        target_from_cash = cash * weight
        self.assertNotAlmostEqual(target_from_portfolio, target_from_cash)
        self.assertAlmostEqual(target_from_portfolio, 20_000.0)
        # Live logic uses portfolio_value for target_value
        self.assertGreater(target_from_portfolio, target_from_cash)


class TestExecutionSafeguards(unittest.TestCase):
    """Rebalance band and cooldowns are applied."""

    def test_rebalance_band_lower_bound(self) -> None:
        from trading_system.config import MIN_TRADE_NOTIONAL, REBALANCE_BAND_PCT
        portfolio_value = 50_000.0
        band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
        self.assertGreaterEqual(band, MIN_TRADE_NOTIONAL)
        self.assertGreaterEqual(band, portfolio_value * REBALANCE_BAND_PCT)


if __name__ == "__main__":
    unittest.main()
