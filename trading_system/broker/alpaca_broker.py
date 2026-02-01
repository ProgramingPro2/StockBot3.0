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

        try:
            if order_type == "market":
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    notional=notional,
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
