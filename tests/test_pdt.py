"""Tests for PDTTracker rolling window and freeze logic."""
from __future__ import annotations

from datetime import date, timedelta

from trading_system.utils.pdt import PDTTracker


def test_count_int_window() -> None:
    tr = PDTTracker(
        max_day_trades=3,
        rolling_business_days=5,
        restrictions_enabled=True,
        min_account_equity=25_000.0,
    )
    for k in (1, 2, 3):
        tr.record_day_trade(k)
    assert tr.count_in_window(5) == 3
    assert tr.can_open_new_day_trade(5) is False
    assert tr.should_block_new_opens(10_000.0, 5) is True


def test_prune_old_int_events() -> None:
    tr = PDTTracker(
        max_day_trades=3,
        rolling_business_days=3,
        restrictions_enabled=True,
        min_account_equity=25_000.0,
    )
    tr.record_day_trade(10)
    tr.record_day_trade(11)
    tr.record_day_trade(12)
    assert tr.count_in_window(12) == 3
    tr.record_day_trade(20)
    assert tr.count_in_window(20) <= 3


def test_date_window_count() -> None:
    tr = PDTTracker(
        max_day_trades=3,
        rolling_business_days=5,
        restrictions_enabled=True,
        min_account_equity=25_000.0,
    )
    d0 = date(2024, 1, 2)
    for i in range(3):
        tr.record_day_trade(d0 + timedelta(days=i))
    assert tr.count_in_window(d0 + timedelta(days=2)) == 3


def test_should_block_respects_equity_and_flag() -> None:
    tr = PDTTracker(
        max_day_trades=1,
        rolling_business_days=5,
        restrictions_enabled=True,
        min_account_equity=25_000.0,
    )
    tr.record_day_trade(1)
    assert tr.should_block_new_opens(10_000.0, 2) is True
    assert tr.should_block_new_opens(30_000.0, 2) is False
    tr2 = PDTTracker(
        max_day_trades=1,
        rolling_business_days=5,
        restrictions_enabled=False,
        min_account_equity=25_000.0,
    )
    tr2.record_day_trade(1)
    assert tr2.should_block_new_opens(10_000.0, 2) is False


def test_clear() -> None:
    tr = PDTTracker(
        max_day_trades=2,
        rolling_business_days=5,
        restrictions_enabled=True,
        min_account_equity=25_000.0,
    )
    tr.record_day_trade(1)
    tr.clear()
    assert tr.count_in_window(5) == 0
