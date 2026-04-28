"""Ticker info: validate symbols and return a sparkline + last close."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import TickerInfo

router = APIRouter(prefix="/api/tickers", tags=["tickers"])


@router.get("/{symbol}", response_model=TickerInfo)
def get_ticker(symbol: str, points: int = 60) -> TickerInfo:
    from trading_system.data.loader import load_ohlcv

    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="empty symbol")
    df = load_ohlcv(sym, lookback_years=1)
    if df is None or df.empty or "close" not in df.columns:
        return TickerInfo(symbol=sym)
    close = df["close"].dropna()
    if close.empty:
        return TickerInfo(symbol=sym)
    last = float(close.iloc[-1])
    spark = close.tail(max(2, points)).round(4).tolist()
    return TickerInfo(symbol=sym, last_close=last, sparkline=spark)
