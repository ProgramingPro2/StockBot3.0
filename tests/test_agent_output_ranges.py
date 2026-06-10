"""Agent output range tests.

Each rule-based agent must return finite values within documented ranges
on synthetic OHLCV data including edge cases (constant prices, NaN rows,
very short history).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    volumes = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": closes * (1.0 + rng.normal(0.0, 0.002, n)),
            "high": closes * (1.0 + np.abs(rng.normal(0.0, 0.003, n))),
            "low": closes * (1.0 - np.abs(rng.normal(0.0, 0.003, n))),
            "close": closes,
            "volume": volumes,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _make_flat_ohlcv_df(n: int) -> pd.DataFrame:
    """All closes equal (constant prices — extreme edge case for vol and z-score agents)."""
    closes = np.full(n, 100.0)
    volumes = np.full(n, 500_000.0)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _make_short_ohlcv_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    closes = 100.0 + rng.normal(0.0, 1.0, n)
    volumes = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes, "volume": volumes},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _make_nan_ohlcv_df(n: int = 100) -> pd.DataFrame:
    df = _make_ohlcv_df(n, seed=5)
    rng = np.random.default_rng(5)
    nan_rows = rng.choice(range(20, n), size=10, replace=False)
    df.iloc[nan_rows, df.columns.get_loc("close")] = np.nan
    return df


class TestVolatilityAgentOutputRange(unittest.TestCase):
    """get_volatility_score must return float in [0, 1] for all inputs."""

    def _score(self, df: pd.DataFrame) -> float:
        from trading_system.agents.volatility import get_volatility_score
        with patch("trading_system.agents.volatility.load_ohlcv", return_value=df):
            return get_volatility_score("FAKE")

    def test_normal_prices_in_range(self) -> None:
        score = self._score(_make_ohlcv_df(200))
        self.assertTrue(np.isfinite(score), "score must be finite")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_constant_prices_returns_half(self) -> None:
        score = self._score(_make_flat_ohlcv_df(200))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_very_short_history_returns_neutral(self) -> None:
        score = self._score(_make_short_ohlcv_df(5))
        self.assertEqual(score, 0.5)

    def test_empty_df_returns_neutral(self) -> None:
        from trading_system.agents.volatility import get_volatility_score
        with patch("trading_system.agents.volatility.load_ohlcv", return_value=pd.DataFrame()):
            score = get_volatility_score("FAKE")
        self.assertEqual(score, 0.5)


class TestMeanReversionAgentOutputRange(unittest.TestCase):
    """get_mean_reversion_score must return float in [-1, 1]."""

    def _score(self, df: pd.DataFrame) -> float:
        from trading_system.agents.mean_reversion import get_mean_reversion_score
        from trading_system.data.processor import add_momentum_features
        with patch("trading_system.agents.mean_reversion.load_ohlcv", return_value=df):
            return get_mean_reversion_score("FAKE")

    def test_normal_prices_in_range(self) -> None:
        score = self._score(_make_ohlcv_df(200))
        self.assertTrue(np.isfinite(score), "score must be finite")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_constant_prices_returns_zero(self) -> None:
        score = self._score(_make_flat_ohlcv_df(200))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_short_history_returns_zero(self) -> None:
        score = self._score(_make_short_ohlcv_df(5))
        self.assertEqual(score, 0.0)

    def test_empty_df_returns_zero(self) -> None:
        from trading_system.agents.mean_reversion import get_mean_reversion_score
        with patch("trading_system.agents.mean_reversion.load_ohlcv", return_value=pd.DataFrame()):
            score = get_mean_reversion_score("FAKE")
        self.assertEqual(score, 0.0)

    def test_nan_rows_in_range(self) -> None:
        score = self._score(_make_nan_ohlcv_df(100))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


class TestRelativeStrengthAgentOutputRange(unittest.TestCase):
    """get_relative_strength_score must return float in [-1, 1]."""

    def _score(self, ticker_df: pd.DataFrame, bench_df: pd.DataFrame) -> float:
        from trading_system.agents.relative_strength import get_relative_strength_score
        def _load(sym, **kwargs):
            return bench_df if sym.upper() == "SPY" else ticker_df
        with patch("trading_system.agents.relative_strength.load_ohlcv", side_effect=_load):
            return get_relative_strength_score("FAKE", benchmark="SPY")

    def test_normal_prices_in_range(self) -> None:
        score = self._score(_make_ohlcv_df(200, seed=1), _make_ohlcv_df(200, seed=2))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_short_history_returns_zero(self) -> None:
        score = self._score(_make_short_ohlcv_df(5), _make_short_ohlcv_df(5))
        self.assertEqual(score, 0.0)

    def test_empty_df_returns_zero(self) -> None:
        from trading_system.agents.relative_strength import get_relative_strength_score
        empty = pd.DataFrame()
        with patch("trading_system.agents.relative_strength.load_ohlcv", return_value=empty):
            score = get_relative_strength_score("FAKE")
        self.assertEqual(score, 0.0)


class TestLiquidityAgentOutputRange(unittest.TestCase):
    """get_liquidity_score must return float in [0, 1]."""

    def _score(self, df: pd.DataFrame) -> float:
        from trading_system.agents.liquidity import get_liquidity_score
        with patch("trading_system.agents.liquidity.load_ohlcv", return_value=df):
            return get_liquidity_score("FAKE")

    def test_normal_volume_in_range(self) -> None:
        score = self._score(_make_ohlcv_df(200))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_short_history_returns_neutral(self) -> None:
        score = self._score(_make_short_ohlcv_df(5))
        self.assertEqual(score, 0.5)

    def test_empty_df_returns_neutral(self) -> None:
        from trading_system.agents.liquidity import get_liquidity_score
        with patch("trading_system.agents.liquidity.load_ohlcv", return_value=pd.DataFrame()):
            score = get_liquidity_score("FAKE")
        self.assertEqual(score, 0.5)

    def test_zero_volume_returns_neutral(self) -> None:
        df = _make_ohlcv_df(100)
        df["volume"] = 0.0
        score = self._score(df)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertTrue(np.isfinite(score))
