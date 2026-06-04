"""Intraday margin framework (IMF) guards for backtest, training envs, and live trading.

Replaces legacy pattern-day-trader (PDT) day-trade counting with buying-power and
same-day round-trip rules aligned with Alpaca's intraday margin model (Jun 2026).
"""
from __future__ import annotations

from datetime import date
from typing import Union

import numpy as np
import pandas as pd

SessionKey = Union[date, int]


class IntradayMarginGuard:
    """
    Simulates intraday buying power and optional same-day round-trip limits.

    When restrictions are disabled, backtest/train/live keep the conservative rule:
    no sell on the same session if a buy occurred that session (live ``bought_today``).
    """

    def __init__(
        self,
        *,
        restrictions_enabled: bool,
        min_intraday_equity: float,
        intraday_multiplier: float,
    ) -> None:
        self.restrictions_enabled = bool(restrictions_enabled)
        self.min_intraday_equity = float(min_intraday_equity)
        self.intraday_multiplier = max(1.0, float(intraday_multiplier))

    def effective_multiplier(self, equity_dollars: float) -> float:
        if equity_dollars >= self.min_intraday_equity:
            return self.intraday_multiplier
        return 1.0

    def intraday_buying_power(self, equity_dollars: float) -> float:
        """Simulated intraday buying power (equity * multiplier when above minimum)."""
        if equity_dollars <= 0:
            return 0.0
        return equity_dollars * self.effective_multiplier(equity_dollars)

    def should_block_buy(self, equity_dollars: float, buy_notional: float) -> bool:
        """Block a buy when it would exceed simulated intraday buying power."""
        if not self.restrictions_enabled or buy_notional <= 0:
            return False
        return buy_notional > self.intraday_buying_power(equity_dollars) + 1e-6

    def allow_same_day_sell(self, equity_dollars: float, bought_this_session: bool) -> bool:
        """
        Whether a sell is allowed when the symbol was bought earlier this session.

        IMF off: block same-day sells (conservative, avoids accidental day trades).
        IMF on + equity at/above minimum: allow (no day-trade count under new rules).
        IMF on + equity below minimum: block (small accounts: no intraday round trips).
        """
        if not bought_this_session:
            return True
        if not self.restrictions_enabled:
            return False
        return equity_dollars >= self.min_intraday_equity


def session_date_from_index(index: pd.Index, t: int) -> date:
    """Normalize backtest / aligned index position to a calendar date."""
    ts = index[t]
    if hasattr(ts, "date"):
        return ts.date()
    return pd.Timestamp(ts).date()


def imf_guard_from_config() -> IntradayMarginGuard:
    from ..config import (
        ENV_IMF_RESTRICTIONS,
        IMF_INTRADAY_MULTIPLIER,
        IMF_MIN_INTRADAY_EQUITY,
    )

    return IntradayMarginGuard(
        restrictions_enabled=ENV_IMF_RESTRICTIONS,
        min_intraday_equity=IMF_MIN_INTRADAY_EQUITY,
        intraday_multiplier=IMF_INTRADAY_MULTIPLIER,
    )


def _can_trade_after_buy(bar: int, last_buy: int, min_hold: int) -> bool:
    if last_buy < 0:
        return True
    if min_hold <= 0:
        return bar > last_buy
    return (bar - last_buy) >= min_hold


def _can_trade_after_sell(bar: int, last_sell: int, min_re: int) -> bool:
    if last_sell < 0:
        return True
    if min_re <= 0:
        return bar > last_sell
    return (bar - last_sell) >= min_re


def rebalance_multi_asset(
    effective_weights: np.ndarray,
    target_holdings: np.ndarray,
    portfolio_value: float,
    rebalance_band: float,
    bar_index: int,
    last_buy_bar: np.ndarray,
    last_sell_bar: np.ndarray,
    min_hold_bars: int,
    min_reentry_bars: int,
    imf_guard: IntradayMarginGuard,
    session_key: SessionKey,
    equity_dollars: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One row of portfolio backtest-style rebalancing with optional IMF rules.
    Returns (new_effective, new_last_buy_bar, new_last_sell_bar).
    """
    n_assets = len(effective_weights)
    new_effective = effective_weights.copy()
    new_buy = last_buy_bar.copy()
    new_sell = last_sell_bar.copy()
    for i in range(n_assets):
        current_value_i = portfolio_value * new_effective[i]
        target_value_i = portfolio_value * float(target_holdings[i])
        is_increase = target_value_i > current_value_i + rebalance_band
        is_decrease = current_value_i > target_value_i + rebalance_band
        if is_increase:
            buy_notional = target_value_i - current_value_i
            if imf_guard.should_block_buy(equity_dollars, buy_notional):
                continue
            if _can_trade_after_sell(bar_index, int(new_sell[i]), min_reentry_bars):
                new_effective[i] = float(target_holdings[i])
                new_buy[i] = bar_index
        elif is_decrease:
            bought_this_session = int(new_buy[i]) == bar_index
            cand_normal = _can_trade_after_buy(bar_index, int(new_buy[i]), min_hold_bars)
            if not cand_normal and bought_this_session:
                if not imf_guard.allow_same_day_sell(equity_dollars, True):
                    continue
            elif not cand_normal:
                continue
            new_effective[i] = float(target_holdings[i])
            new_sell[i] = bar_index
    return new_effective, new_buy, new_sell
