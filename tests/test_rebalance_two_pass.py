"""Two-pass rebalance logic tests for _execute_rebalance_passes.

Tests verify:
  - Sells execute before buys (pass ordering)
  - remaining_capital decremented per buy
  - Buys skipped when capital insufficient
  - Pending-order symbols skipped
  - Qty rounding to 0 skips instead of forcing 1 share
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from trading_system.cli.run import _execute_rebalance_passes


def _make_imf_guard(block_buy=False, allow_sell=True):
    guard = MagicMock()
    guard.should_block_buy.return_value = block_buy
    guard.allow_same_day_sell.return_value = allow_sell
    return guard


def _make_broker(sell_returns=True, buy_returns=True):
    broker = MagicMock()
    broker.place_order.return_value = {"id": "mock_order"} if buy_returns else None
    broker.close_position.return_value = sell_returns
    return broker


class TestSellsBeforeBuys(unittest.TestCase):
    """Sell pass must execute before buy pass."""

    def test_sell_order_placed_before_buy_order(self) -> None:
        call_order = []
        broker = MagicMock()
        # close_position used for large sells; track it as a "sell" action
        broker.close_position.side_effect = lambda t: call_order.append(("close_sell", t)) or True
        broker.place_order.side_effect = lambda *a, **kw: call_order.append(("order", kw.get("side"))) or {"id": "x"}

        target_holdings = np.array([0.0, 0.5], dtype=np.float32)
        # AAPL: current=500, target=0 -> sell_all path (notional >= 0.95 * current_value)
        sell_candidates = [(0, "AAPL", 500.0, 0.0, 5.0)]
        # MSFT: buy 200 notional
        buy_candidates = [(1, "MSFT", 200.0, 0.0)]

        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=sell_candidates,
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=target_holdings,
            specialist_list=[{}, {}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )

        # Sells (close_position or sell order) must come before buys
        event_types = [kind for kind, *_ in call_order]
        sell_indices = [i for i, k in enumerate(event_types) if k in ("close_sell", "order") and
                        (k == "close_sell" or call_order[i][1] == "sell")]
        buy_indices = [i for i, k in enumerate(event_types) if k == "order" and call_order[i][1] == "buy"]
        self.assertTrue(len(sell_indices) > 0, "No sell action recorded")
        self.assertTrue(len(buy_indices) > 0, "No buy action recorded")
        self.assertLess(sell_indices[0], buy_indices[0],
                        f"Sell at index {sell_indices[0]} was not before buy at {buy_indices[0]}")


class TestRemainingCapitalDecrement(unittest.TestCase):
    """remaining_capital must be decremented after each successful buy."""

    def test_capital_decremented_per_buy(self) -> None:
        broker = _make_broker(buy_returns=True)
        target_holdings = np.array([0.4, 0.3], dtype=np.float32)
        buy_candidates = [
            (0, "AAPL", 400.0, 0.0),
            (1, "MSFT", 300.0, 0.0),
        ]
        remaining_cap, _ = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=target_holdings,
            specialist_list=[{}, {}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        # 1000 - 400 - 300 = 300
        self.assertAlmostEqual(remaining_cap, 300.0, places=2)

    def test_capital_not_decremented_on_failed_order(self) -> None:
        broker = _make_broker(buy_returns=False)
        buy_candidates = [(0, "AAPL", 400.0, 0.0)]
        remaining_cap, _ = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.4], dtype=np.float32),
            specialist_list=[{}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        self.assertAlmostEqual(remaining_cap, 1000.0, places=2)


class TestBuySkippedWhenCapitalInsufficient(unittest.TestCase):
    """Buys must be skipped when notional exceeds remaining capital."""

    def test_buy_skipped_no_capital(self) -> None:
        broker = _make_broker()
        buy_candidates = [(0, "AAPL", 1500.0, 0.0)]  # need 1500; only 500 available
        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=500.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.5], dtype=np.float32),
            specialist_list=[{}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_not_called()
        self.assertAlmostEqual(remaining_cap, 500.0, places=2)
        self.assertNotIn("AAPL", bought)

    def test_first_buy_consumes_capital_for_second(self) -> None:
        """First buy eats capital; second buy should be skipped if insufficient."""
        broker = _make_broker(buy_returns=True)
        buy_candidates = [
            (0, "AAPL", 800.0, 0.0),
            (1, "MSFT", 500.0, 0.0),  # only 200 left after AAPL buy
        ]
        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.4, 0.3], dtype=np.float32),
            specialist_list=[{}, {}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        self.assertIn("AAPL", bought)
        self.assertNotIn("MSFT", bought)
        self.assertAlmostEqual(remaining_cap, 200.0, places=2)

    def test_sells_proceeds_fund_buy(self) -> None:
        """Sell proceeds must be added to remaining_capital before buy pass."""
        broker = _make_broker(sell_returns=True, buy_returns=True)
        # available_capital is 100, but sell adds 600, so buy of 400 must succeed
        sell_candidates = [(0, "AAPL", 600.0, 0.0, 6.0)]
        buy_candidates = [(1, "MSFT", 400.0, 0.0)]
        target_holdings = np.array([0.0, 0.4], dtype=np.float32)

        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=sell_candidates,
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=100.0,
            portfolio_value=10000.0,
            target_holdings=target_holdings,
            specialist_list=[{}, {}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        self.assertIn("MSFT", bought)
        # 100 + 600 (sell proceeds) - 400 (buy) = 300
        self.assertAlmostEqual(remaining_cap, 300.0, places=2)


class TestPendingOrdersSkipped(unittest.TestCase):
    """Pending-order symbols must be skipped in both passes."""

    def test_pending_buy_skipped(self) -> None:
        broker = _make_broker()
        buy_candidates = [(0, "AAPL", 500.0, 0.0)]
        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.4], dtype=np.float32),
            specialist_list=[{}],
            fractional=True,
            latest_prices={},
            pending_sells=set(),
            pending_buys={"AAPL"},  # AAPL has pending buy
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_not_called()
        self.assertNotIn("AAPL", bought)

    def test_pending_sell_skipped(self) -> None:
        broker = _make_broker()
        sell_candidates = [(0, "AAPL", 500.0, 100.0, 5.0)]
        remaining_cap, _ = _execute_rebalance_passes(
            sell_candidates=sell_candidates,
            buy_candidates=[],
            broker=broker,
            available_capital=500.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.1], dtype=np.float32),
            specialist_list=[{}],
            fractional=True,
            latest_prices={},
            pending_sells={"AAPL"},  # AAPL has pending sell
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_not_called()
        broker.close_position.assert_not_called()


class TestQtyRoundingToZeroSkipped(unittest.TestCase):
    """Whole-share mode: if qty rounds to 0, skip order (no forced min=1)."""

    def test_buy_qty_zero_skipped(self) -> None:
        broker = _make_broker()
        # notional=5.0, price=1000.0 -> qty = round(5/1000) = 0 -> skip
        buy_candidates = [(0, "GOOG", 5.0, 0.0)]
        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.1], dtype=np.float32),
            specialist_list=[{}],
            fractional=False,  # whole-share mode
            latest_prices={"GOOG": 1000.0},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_not_called()
        self.assertNotIn("GOOG", bought)

    def test_buy_qty_nonzero_placed(self) -> None:
        broker = _make_broker(buy_returns=True)
        # notional=200.0, price=50.0 -> qty=4 -> should place order
        buy_candidates = [(0, "XYZ", 200.0, 0.0)]
        remaining_cap, bought = _execute_rebalance_passes(
            sell_candidates=[],
            buy_candidates=buy_candidates,
            broker=broker,
            available_capital=1000.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.2], dtype=np.float32),
            specialist_list=[{}],
            fractional=False,
            latest_prices={"XYZ": 50.0},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_called_once()
        call_kwargs = broker.place_order.call_args
        self.assertEqual(call_kwargs[1].get("qty") or call_kwargs[0][1], 4)
        self.assertIn("XYZ", bought)

    def test_sell_qty_zero_skipped(self) -> None:
        broker = _make_broker()
        # notional=2.0 (current=102, target=100), price=1000 -> qty=round(2/1000)=0 -> skip
        sell_candidates = [(0, "GOOG", 102.0, 100.0, 0.1)]  # 0.1 shares
        _execute_rebalance_passes(
            sell_candidates=sell_candidates,
            buy_candidates=[],
            broker=broker,
            available_capital=0.0,
            portfolio_value=10000.0,
            target_holdings=np.array([0.01], dtype=np.float32),
            specialist_list=[{}],
            fractional=False,
            latest_prices={"GOOG": 1000.0},
            pending_sells=set(),
            pending_buys=set(),
            bought_since=set(),
            sold_since=set(),
            bought_today=set(),
            imf_guard=_make_imf_guard(),
        )
        broker.place_order.assert_not_called()
        broker.close_position.assert_not_called()
