"""Tests for IntradayMarginGuard (IMF) buying power and same-day sell rules."""
from datetime import date, timedelta

from trading_system.utils.imf import IntradayMarginGuard


def test_intraday_buying_power_below_minimum_uses_1x() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=True,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.intraday_buying_power(1500.0) == 1500.0


def test_intraday_buying_power_at_minimum_uses_multiplier() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=True,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.intraday_buying_power(10000.0) == 40000.0


def test_should_block_buy_when_exceeds_buying_power() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=True,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.should_block_buy(5000.0, 25000.0) is True
    assert g.should_block_buy(5000.0, 10000.0) is False


def test_same_day_sell_imf_off_blocks() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=False,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.allow_same_day_sell(50000.0, True) is False


def test_same_day_sell_imf_on_high_equity_allows() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=True,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.allow_same_day_sell(10000.0, True) is True


def test_same_day_sell_imf_on_low_equity_blocks() -> None:
    g = IntradayMarginGuard(
        restrictions_enabled=True,
        min_intraday_equity=2000.0,
        intraday_multiplier=4.0,
    )
    assert g.allow_same_day_sell(1500.0, True) is False
