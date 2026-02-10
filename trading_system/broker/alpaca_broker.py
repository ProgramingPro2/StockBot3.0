"""
Alpaca broker integration for live trading.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ..config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from ..utils.logger import get_logger

logger = get_logger()


class AlpacaBroker:
    """Alpaca API wrapper for placing orders."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or ALPACA_API_KEY
        self.secret_key = secret_key or ALPACA_SECRET_KEY
        self.base_url = base_url or ALPACA_BASE_URL
        self.client: TradingClient | None = None

        if self.api_key and self.secret_key:
            try:
                self.client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper="paper" in self.base_url.lower(),
                )
            except Exception as e:
                logger.error("Failed to initialize Alpaca client: %s", e)

    def get_account(self) -> dict:
        """Get account info."""
        if not self.client:
            return {}
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
            }
        except Exception as e:
            logger.error("Failed to get account: %s", e)
            return {}

    def is_market_open(self) -> bool:
        """Return True if the market is open (via Alpaca clock API)."""
        if not self.client:
            return False
        try:
            clock = self.client.get_clock()
            return bool(clock.is_open)
        except Exception as e:
            logger.error("Failed to get market clock: %s", e)
            return False

    def get_all_positions(self) -> list[dict]:
        """Get all open positions from Alpaca (source of truth for account holdings)."""
        if not self.client:
            return []
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Failed to get all positions: %s", e)
            return []

    def get_position(self, ticker: str) -> dict:
        """Get current position for ticker."""
        if not self.client:
            return {}
        try:
            position = self.client.get_open_position(ticker.upper())
            return {
                "qty": float(position.qty),
                "market_value": float(position.market_value),
                "avg_entry_price": float(position.avg_entry_price),
            }
        except Exception:
            return {}

    def place_order(
        self,
        ticker: str,
        qty: float | None = None,
        notional: float | None = None,
        side: str = "buy",
        order_type: str = "market",
    ) -> dict:
        """
        Place an order.
        qty: number of shares (for qty-based orders)
        notional: dollar amount (for notional orders)
        side: 'buy' or 'sell'
        order_type: 'market' or 'limit'
        Returns order details or empty dict on failure.
        """
        if not self.client:
            logger.error("Alpaca client not initialized. Check API keys.")
            return {}

        ticker = ticker.upper()
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # For sell + notional, cap notional to slightly below position value so Alpaca's
        # notional->shares conversion never exceeds available qty (avoids 40310000 errors).
        if side.lower() == "sell" and notional is not None and notional > 0:
            pos = self.get_position(ticker)
            pos_value = float(pos.get("market_value", 0))
            if pos_value > 0:
                notional = min(notional, pos_value * 0.999)

        try:
            if order_type == "market":
                # Alpaca requires notional to be limited to 2 decimal places
                notional_rounded = round(notional, 2) if notional is not None else None
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    notional=notional_rounded,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.client.submit_order(req)
            else:
                logger.warning("Limit orders not yet implemented")
                return {}

            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty) if order.qty else None,
                "notional": float(order.notional) if order.notional else None,
                "side": order.side.value,
                "status": order.status.value,
            }
        except Exception as e:
            logger.error("Failed to place order for %s: %s", ticker, e)
            return {}

    def close_position(self, ticker: str) -> bool:
        """Close position for ticker."""
        if not self.client:
            return False
        try:
            self.client.close_position(ticker.upper())
            return True
        except Exception as e:
            logger.error("Failed to close position for %s: %s", ticker, e)
            return False
