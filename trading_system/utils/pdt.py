"""Pattern day trader (PDT) rolling window for backtest, training envs, and live trading."""
from __future__ import annotations

from collections import deque
from datetime import date
from typing import Deque, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

SessionKey = Union[date, int]


class PDTTracker:
    """
    Tracks day-trade events in a rolling business-day (calendar) or integer-step window.
    Use ``date`` keys with live/backtest; use ``int`` keys (episode step/cursor) for synthetic envs.
    """

    def __init__(
        self,
        *,
        max_day_trades: int,
        rolling_business_days: int,
        restrictions_enabled: bool,
        min_account_equity: float,
    ) -> None:
        self.max_day_trades = max(1, int(max_day_trades))
        self.rolling_business_days = max(1, int(rolling_business_days))
        self.restrictions_enabled = bool(restrictions_enabled)
        self.min_account_equity = float(min_account_equity)
        self._events: Deque[SessionKey] = deque()

    def clear(self) -> None:
        self._events.clear()

    def _prune(self, session_end: SessionKey) -> None:
        if isinstance(session_end, int):
            cutoff = session_end - self.rolling_business_days
            while self._events and self._events[0] <= cutoff:
                self._events.popleft()
            return
        end = pd.Timestamp(session_end).normalize()
        start = end - BDay(self.rolling_business_days)
        while self._events and pd.Timestamp(self._events[0]).normalize() <= start:
            self._events.popleft()

    def record_day_trade(self, session_key: SessionKey) -> None:
        self._events.append(session_key)
        self._prune(session_key)

    def count_in_window(self, session_key: SessionKey) -> int:
        if not self._events:
            return 0
        if isinstance(session_key, int):
            lo = session_key - self.rolling_business_days
            return sum(1 for x in self._events if lo < x <= session_key)
        end = pd.Timestamp(session_key).normalize()
        start = end - BDay(self.rolling_business_days)
        n = 0
        for d in self._events:
            td = pd.Timestamp(d).normalize()
            if start < td <= end:
                n += 1
        return n

    def can_open_new_day_trade(self, session_key: SessionKey) -> bool:
        return self.count_in_window(session_key) < self.max_day_trades

    def should_block_new_opens(self, equity_dollars: float, session_key: SessionKey) -> bool:
        if not self.restrictions_enabled:
            return False
        if equity_dollars >= self.min_account_equity:
            return False
        return self.count_in_window(session_key) >= self.max_day_trades


def session_date_from_index(index: pd.Index, t: int) -> date:
    """Normalize backtest / aligned index position to a calendar date."""
    ts = index[t]
    if hasattr(ts, "date"):
        return ts.date()
    return pd.Timestamp(ts).date()


def pdt_tracker_from_config() -> PDTTracker:
    from ..config import (
        ENV_PDT_RESTRICTIONS,
        PDT_MAX_DAY_TRADES,
        PDT_MIN_ACCOUNT_EQUITY,
        PDT_ROLLING_BUSINESS_DAYS,
    )

    return PDTTracker(
        max_day_trades=PDT_MAX_DAY_TRADES,
        rolling_business_days=PDT_ROLLING_BUSINESS_DAYS,
        restrictions_enabled=ENV_PDT_RESTRICTIONS,
        min_account_equity=PDT_MIN_ACCOUNT_EQUITY,
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
    pdt_tracker: PDTTracker,
    session_key: SessionKey,
    equity_dollars: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One row of portfolio backtest-style rebalancing with optional PDT rules.
    Returns (new_effective, new_last_buy_bar, new_last_sell_bar).
    """
    from ..config import ENV_PDT_RESTRICTIONS, PDT_MIN_ACCOUNT_EQUITY

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
            if ENV_PDT_RESTRICTIONS and pdt_tracker.should_block_new_opens(equity_dollars, session_key):
                continue
            if _can_trade_after_sell(bar_index, int(new_sell[i]), min_reentry_bars):
                new_effective[i] = float(target_holdings[i])
                new_buy[i] = bar_index
        elif is_decrease:
            cand_normal = _can_trade_after_buy(bar_index, int(new_buy[i]), min_hold_bars)
            allowed = cand_normal
            record_pdt = False
            if (
                not cand_normal
                and ENV_PDT_RESTRICTIONS
                and int(new_buy[i]) == bar_index
            ):
                if equity_dollars >= PDT_MIN_ACCOUNT_EQUITY:
                    allowed = True
                elif pdt_tracker.can_open_new_day_trade(session_key):
                    allowed = True
                    record_pdt = True
            if allowed:
                new_effective[i] = float(target_holdings[i])
                new_sell[i] = bar_index
                if record_pdt:
                    pdt_tracker.record_day_trade(session_key)
    return new_effective, new_buy, new_sell
