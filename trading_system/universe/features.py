"""
Feature extraction per symbol for the universe scorer.
Reuses specialist outputs (sentiment, quant, fundamental, regime) and optional rolling features.
"""
import numpy as np

from ..config import ENV_OBS_ROLLING_FEATURES
from ..utils.logger import get_logger

logger = get_logger()


def _sentiment_proxy_from_ohlcv(ticker: str) -> float:
    """Momentum-based sentiment proxy when real sentiment unavailable."""
    from ..data.loader import load_ohlcv

    df = load_ohlcv(ticker)
    if df.empty or "close" not in df.columns or len(df) < 21:
        return 0.0
    ret = df["close"].pct_change().dropna().tail(21)
    if ret.empty:
        return 0.0
    mean_ret = float(ret.mean())
    return float(max(-0.5, min(0.5, mean_ret * 12)))


def _rolling_features_from_ohlcv(close_series) -> dict:
    """Rolling return 5d/21d and vol 21d from close."""
    if close_series.empty or len(close_series) < 5:
        return {"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0}
    ret = close_series.pct_change().dropna()
    if ret.empty or len(ret) < 5:
        return {"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0}
    ret_5d = float(ret.tail(5).mean()) if len(ret) >= 5 else 0.0
    ret_21d = float(ret.tail(21).mean()) if len(ret) >= 21 else float(ret.mean())
    vol_21d = float(ret.tail(21).std()) if len(ret) >= 21 else float(ret.std())
    if vol_21d < 1e-8:
        vol_21d = 0.01
    return {
        "rolling_return_5d": max(-1, min(1, ret_5d * 50)),
        "rolling_return_21d": max(-1, min(1, ret_21d * 50)),
        "rolling_vol_21d": max(-1, min(1, (vol_21d - 0.01) * 50)),
    }


def extract_features_for_ticker(ticker: str) -> dict:
    """
    Run specialists (or proxies) for one ticker and return a feature dict.
    Keys: sentiment, quant_prob, fund_score, regime_flag; optionally rolling_return_5d, rolling_return_21d, rolling_vol_21d.
    Used by the universe scorer.
    """
    from ..agents import get_sentiment, get_trend_probability, get_intrinsic_value_score, detect_regime
    from ..data.loader import load_ohlcv

    ticker = ticker.upper()
    sentiment, _, from_llm = get_sentiment(ticker)
    if not from_llm:
        sentiment = _sentiment_proxy_from_ohlcv(ticker)
    quant_prob = get_trend_probability(ticker)
    fund_score, _ = get_intrinsic_value_score(ticker)
    spy = load_ohlcv("SPY")
    regime_result = detect_regime(spy)
    regime_flag = regime_result["flag"]

    out = {
        "sentiment": float(sentiment),
        "quant_prob": float(quant_prob),
        "fund_score": float(fund_score),
        "regime_flag": float(regime_flag),
    }
    if ENV_OBS_ROLLING_FEATURES:
        ohlcv = load_ohlcv(ticker)
        if not ohlcv.empty and "close" in ohlcv.columns:
            out.update(_rolling_features_from_ohlcv(ohlcv["close"]))
        else:
            out.update({"rolling_return_5d": 0.0, "rolling_return_21d": 0.0, "rolling_vol_21d": 0.0})
    return out


def feature_dict_to_vector(features: dict) -> np.ndarray:
    """
    Convert feature dict to fixed-length vector for ScorerModel.
    Order: sentiment, quant_prob, fund_score, regime_flag [, rolling_return_5d, rolling_return_21d, rolling_vol_21d].
    """
    base = [
        float(features.get("sentiment", 0.0)),
        float(features.get("quant_prob", 0.5)),
        float(features.get("fund_score", 0.0)),
        float(features.get("regime_flag", 0.5)),
    ]
    if ENV_OBS_ROLLING_FEATURES:
        base.extend([
            float(features.get("rolling_return_5d", 0.0)),
            float(features.get("rolling_return_21d", 0.0)),
            float(features.get("rolling_vol_21d", 0.0)),
        ])
    return np.array(base, dtype=np.float32)
