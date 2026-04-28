"""
Custom Gymnasium environment for the RL meta-learner.
Observation: per-asset [sentiment, quant_prob, fund_score, regime_flag, momentum_score, holdings]
  + optional [rolling_return_5d, rolling_return_21d, rolling_vol_21d].
Action: position size % per asset (normalized so sum <= 1 before execution, matching live/backtest).
Reward: log utility of portfolio return minus drawdown and turnover costs.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..config import (
    ENV_EPISODE_DAYS,
    ENV_OBS_ROLLING_FEATURES,
    ENV_REWARD_DRAWDOWN_PENALTY,
    ENV_ROLLING_SHARPE_DAYS,
    MIN_HOLD_MINUTES_AFTER_BUY,
    MIN_REENTRY_MINUTES_AFTER_SELL,
    MIN_TRADE_NOTIONAL,
    REBALANCE_BAND_PCT,
    SLIPPAGE_BPS,
    STARTING_CAPITAL,
    TRANSACTION_COST_BPS,
)
from ..utils.pdt import pdt_tracker_from_config, rebalance_multi_asset, session_date_from_index
from .proxy_specialists import (
    merge_daily_aux_into_proxy_row,
    normalize_action_max_unit_exposure,
)


def _sanitize_obs(obs: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip to [-1, 1] so the policy never sees invalid values."""
    out = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(out, -1.0, 1.0)


def _obs_features_per_asset() -> int:
    """Number of observation features per asset."""
    base = 11  # 10 specialist scores (incl. relative_strength, liquidity) + holdings
    if ENV_OBS_ROLLING_FEATURES:
        return base + 3  # rolling_return_5d, rolling_return_21d, rolling_vol_21d
    return base


def _effective_holdings_min_trade(
    action: np.ndarray,
    holdings: np.ndarray,
    equity: float,
    min_trade_equity: float,
) -> np.ndarray:
    """Vectorized min-notional execution (same semantics as prior nested loops)."""
    a = np.asarray(action, dtype=np.float64)
    h = np.asarray(holdings, dtype=np.float64)
    notional = np.abs(a - h) * equity
    trade = notional >= min_trade_equity
    buy = (a > h) & trade
    sell = (a < h) & trade
    eff = h.copy()
    eff = np.where(buy, a, eff)
    eff = np.where(sell, a, eff)
    return eff.astype(np.float32)


def _compute_step_reward(
    portfolio_return: float,
    turnover: float,
    drawdown: float,
    drawdown_penalty: float,
) -> float:
    """Log utility minus drawdown and proportional transaction cost (no hidden 10x multiplier)."""
    utility = float(np.log(max(1e-8, 1.0 + portfolio_return)))
    cost_frac = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
    reward = utility - drawdown_penalty * drawdown - turnover * cost_frac
    return float(np.clip(np.nan_to_num(reward, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0))


class TradingEnv(gym.Env):
    """
    Trading environment for PPO meta-learner.
    Multi-asset: n_assets sets of specialist features + n_assets holdings.
    Optional bootstrap_returns in reset(options): real return rows for synthetic branch (no self-fulfilling signal).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_assets: int = 1,
        rolling_sharpe_days: int | None = None,
        episode_days: int | None = None,
        drawdown_penalty: float | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.rolling_sharpe_days = rolling_sharpe_days or ENV_ROLLING_SHARPE_DAYS
        self.episode_days = episode_days or ENV_EPISODE_DAYS
        self.drawdown_penalty = (
            drawdown_penalty if drawdown_penalty is not None else ENV_REWARD_DRAWDOWN_PENALTY
        )
        self.render_mode = render_mode
        self._obs_per_asset = _obs_features_per_asset()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_per_asset * n_assets,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        self._returns_history: list[float] = []
        self._equity_history: list[float] = []
        self._equity_peak: float = 1.0
        self._holdings: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self._step_count = 0
        self._specialist_outputs: list[dict] | None = None
        self._rolling_returns: np.ndarray | None = None  # (n_assets, 21) per-asset
        self._rolling_vols: np.ndarray | None = None
        self._bootstrap_returns: np.ndarray | None = None
        self._bootstrap_idx: int = 0
        self._pdt_tracker = pdt_tracker_from_config()
        self._last_buy_bar = np.full(self.n_assets, -999, dtype=np.int64)
        self._last_sell_bar = np.full(self.n_assets, -999, dtype=np.int64)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._returns_history = []
        self._equity_history = [1.0]
        self._equity_peak = 1.0
        self._holdings = np.zeros(self.n_assets, dtype=np.float32)
        self._step_count = 0
        self._pdt_tracker = pdt_tracker_from_config()
        self._last_buy_bar = np.full(self.n_assets, -999, dtype=np.int64)
        self._last_sell_bar = np.full(self.n_assets, -999, dtype=np.int64)
        specialist = options.get("specialist_outputs", []) if options else []
        if isinstance(specialist, dict):
            specialist = [specialist] * self.n_assets
        self._specialist_outputs = (specialist + [{}] * self.n_assets)[: self.n_assets]
        self._rolling_returns = np.full((self.n_assets, 21), np.nan, dtype=np.float32)
        self._rolling_vols = np.full(self.n_assets, 0.01, dtype=np.float32)
        br = options.get("bootstrap_returns") if options else None
        if br is not None:
            self._bootstrap_returns = np.asarray(br, dtype=np.float32)
            self._bootstrap_idx = 0
        else:
            self._bootstrap_returns = None
            self._bootstrap_idx = 0

        obs = self._get_obs(holdings=self._holdings)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = normalize_action_max_unit_exposure(action)

        equity = float(self._equity_history[-1])
        portfolio_value = equity * STARTING_CAPITAL
        rebalance_band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
        _day_m = 24 * 60
        min_hold_bars = 1 if MIN_HOLD_MINUTES_AFTER_BUY >= _day_m else 0
        min_reentry_bars = 1 if MIN_REENTRY_MINUTES_AFTER_SELL >= _day_m else 0
        bar = int(self._step_count)
        sess = int(bar)
        h = self._holdings.astype(np.float64)
        target = np.asarray(action, dtype=np.float64)
        eh, lb, ls = rebalance_multi_asset(
            h,
            target,
            portfolio_value,
            rebalance_band,
            bar,
            self._last_buy_bar,
            self._last_sell_bar,
            min_hold_bars,
            min_reentry_bars,
            self._pdt_tracker,
            sess,
            portfolio_value,
        )
        self._last_buy_bar, self._last_sell_bar = lb, ls
        effective_holdings = eh.astype(np.float32)
        turnover = float(np.sum(np.abs(effective_holdings - self._holdings)))
        self._holdings = effective_holdings

        if self._bootstrap_returns is not None and self._bootstrap_idx < len(self._bootstrap_returns):
            returns_per_asset = np.asarray(
                self._bootstrap_returns[self._bootstrap_idx], dtype=np.float32
            ).reshape(self.n_assets)
            self._bootstrap_idx += 1
        else:
            outputs = self._specialist_outputs or [{}] * self.n_assets
            returns_per_asset = np.zeros(self.n_assets, dtype=np.float32)
            for i in range(self.n_assets):
                s = outputs[i] if i < len(outputs) else {}
                sentiment = s.get("sentiment", 0.0)
                quant_prob = s.get("quant_prob", 0.5)
                fund_score = s.get("fund_score", 0.0)
                regime_flag = s.get("regime_flag", 0.5)
                signal = (sentiment + (quant_prob - 0.5) * 2 + fund_score - regime_flag) / 4
                ret = signal * 0.01 + float(self.np_random.normal(0, 0.01))
                returns_per_asset[i] = ret
                if self._rolling_returns is not None:
                    self._rolling_returns[i] = np.roll(self._rolling_returns[i], 1)
                    self._rolling_returns[i, 0] = ret
                    valid = self._rolling_returns[i][~np.isnan(self._rolling_returns[i])]
                    if len(valid) >= 5:
                        self._rolling_vols[i] = max(0.01, float(np.std(valid)))

        portfolio_return = float(np.dot(effective_holdings, returns_per_asset))
        self._returns_history.append(portfolio_return)
        new_eq = self._equity_history[-1] * (1 + portfolio_return)
        self._equity_history.append(new_eq)
        self._equity_peak = max(self._equity_peak, new_eq)

        obs = self._get_obs(holdings=self._holdings)

        current = self._equity_history[-1]
        drawdown = (self._equity_peak - current) / (self._equity_peak + 1e-8) if len(self._equity_history) > 1 else 0.0
        reward = _compute_step_reward(portfolio_return, turnover, drawdown, self.drawdown_penalty)

        terminated = self._step_count >= self.episode_days
        truncated = False

        return _sanitize_obs(obs), reward, terminated, truncated, {"return": portfolio_return}

    def _get_obs(self, holdings: np.ndarray) -> np.ndarray:
        outputs = self._specialist_outputs or [{}] * self.n_assets
        parts: list[float] = []
        for i in range(self.n_assets):
            s = outputs[i] if i < len(outputs) else {}
            parts.append(float(s.get("sentiment", 0.0)))
            parts.append(float(s.get("quant_prob", 0.5)))
            parts.append(float(s.get("fund_score", 0.0)))
            parts.append(float(s.get("regime_flag", 0.5)))
            parts.append(float(s.get("momentum_score", 0.0)))
            parts.append(float(s.get("mean_reversion_score", 0.0)))
            parts.append(float(s.get("volatility_score", 0.5)))
            parts.append(float(s.get("event_risk_score", 0.0)))
            parts.append(float(s.get("relative_strength_score", 0.0)))
            parts.append(float(s.get("liquidity_score", 0.5)))
            parts.append(float(holdings[i]))
            if ENV_OBS_ROLLING_FEATURES:
                if self._rolling_returns is not None and self._rolling_vols is not None:
                    valid = self._rolling_returns[i][~np.isnan(self._rolling_returns[i])]
                    ret_5d = float(np.mean(valid[-5:])) if len(valid) >= 5 else 0.0
                    ret_21d = float(np.mean(valid)) if len(valid) > 0 else 0.0
                    vol_21d = self._rolling_vols[i]
                    parts.append(np.clip(ret_5d * 50, -1, 1))
                    parts.append(np.clip(ret_21d * 50, -1, 1))
                    parts.append(np.clip((vol_21d - 0.01) * 50, -1, 1))
                else:
                    parts.extend([0.0, 0.0, 0.0])
        obs = np.array(parts, dtype=np.float32)
        return _sanitize_obs(obs)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class HistoricalTradingEnv(gym.Env):
    """
    Trading environment using real historical returns.
    Steps through aligned return data; derives specialist proxies from lookback.
    Optional daily_aux_features (n_days, n_assets, 2): atr_pct, volume_ratio mapped into obs slots.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_array: np.ndarray,
        n_assets: int,
        rolling_sharpe_days: int | None = None,
        episode_days: int | None = None,
        drawdown_penalty: float | None = None,
        render_mode: str | None = None,
        daily_aux_features: np.ndarray | None = None,
        trading_dates: np.ndarray | pd.DatetimeIndex | list | None = None,
    ):
        """
        Args:
            returns_array: Shape (n_days, n_assets), daily returns per asset.
            n_assets: Number of assets.
            daily_aux_features: Optional (n_days, n_assets, 2) for atr_pct, volume_ratio.
            trading_dates: Optional length n_days calendar index for PDT session keys.
        """
        super().__init__()
        self.returns_array = np.asarray(returns_array, dtype=np.float32)
        self.n_days, self.n_assets = self.returns_array.shape
        if self.n_assets != n_assets:
            raise ValueError("returns_array n_assets mismatch")
        self.daily_aux_features = (
            np.asarray(daily_aux_features, dtype=np.float32) if daily_aux_features is not None else None
        )
        if self.daily_aux_features is not None and self.daily_aux_features.shape[:2] != (
            self.n_days,
            self.n_assets,
        ):
            raise ValueError("daily_aux_features shape must match (n_days, n_assets, K)")
        self.rolling_sharpe_days = rolling_sharpe_days or ENV_ROLLING_SHARPE_DAYS
        self.episode_days = episode_days or ENV_EPISODE_DAYS
        self.drawdown_penalty = (
            drawdown_penalty if drawdown_penalty is not None else ENV_REWARD_DRAWDOWN_PENALTY
        )
        self.render_mode = render_mode
        self._obs_per_asset = _obs_features_per_asset()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_per_asset * n_assets,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        self._cursor: int = 0
        self._step_in_episode: int = 0
        self._returns_history: list[float] = []
        self._equity_history: list[float] = []
        self._equity_peak: float = 1.0
        self._holdings: np.ndarray | None = None
        self.np_random = np.random.default_rng()

        # Pre-compute proxy features once; _get_obs() becomes a cheap array lookup.
        # Shape (n_days, n_assets, 13) — row t uses returns_array[t-21:t].
        from .proxy_specialists import precompute_proxy_feature_matrix
        self._proxy_cache = precompute_proxy_feature_matrix(self.returns_array, window=21)

        self._trading_dates: pd.DatetimeIndex | None = None
        if trading_dates is not None:
            td = pd.DatetimeIndex(trading_dates)
            if len(td) != self.n_days:
                raise ValueError("trading_dates length must match returns_array n_days")
            self._trading_dates = td

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        min_start = 21 + self.rolling_sharpe_days
        max_start = self.n_days - self.episode_days - 1
        if max_start <= min_start:
            self._cursor = min_start
        else:
            self._cursor = int(self.np_random.integers(min_start, max_start + 1))

        self._step_in_episode = 0
        self._returns_history = []
        self._equity_history = [1.0]
        self._equity_peak = 1.0
        self._holdings = np.zeros(self.n_assets, dtype=np.float32)
        self._pdt_tracker = pdt_tracker_from_config()
        self._last_buy_bar = np.full(self.n_assets, -999, dtype=np.int64)
        self._last_sell_bar = np.full(self.n_assets, -999, dtype=np.int64)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_in_episode += 1
        action = normalize_action_max_unit_exposure(action)

        equity = float(self._equity_history[-1])
        portfolio_value = equity * STARTING_CAPITAL
        rebalance_band = max(MIN_TRADE_NOTIONAL, portfolio_value * REBALANCE_BAND_PCT)
        _day_m = 24 * 60
        min_hold_bars = 1 if MIN_HOLD_MINUTES_AFTER_BUY >= _day_m else 0
        min_reentry_bars = 1 if MIN_REENTRY_MINUTES_AFTER_SELL >= _day_m else 0
        bar = int(self._cursor)
        if self._trading_dates is not None:
            sess = session_date_from_index(self._trading_dates, bar)
        else:
            sess = int(bar)
        h = self._holdings.astype(np.float64)
        target = np.asarray(action, dtype=np.float64)
        eh, lb, ls = rebalance_multi_asset(
            h,
            target,
            portfolio_value,
            rebalance_band,
            bar,
            self._last_buy_bar,
            self._last_sell_bar,
            min_hold_bars,
            min_reentry_bars,
            self._pdt_tracker,
            sess,
            portfolio_value,
        )
        self._last_buy_bar, self._last_sell_bar = lb, ls
        effective_holdings = eh.astype(np.float32)
        turnover = float(np.sum(np.abs(effective_holdings - self._holdings)))
        self._holdings = effective_holdings

        if self._cursor >= self.n_days:
            obs = self._get_obs()
            return _sanitize_obs(obs), 0.0, True, False, {"return": 0.0}

        day_returns = self.returns_array[self._cursor]
        portfolio_return = float(np.dot(effective_holdings, day_returns))
        self._returns_history.append(portfolio_return)
        new_eq = self._equity_history[-1] * (1 + portfolio_return)
        self._equity_history.append(new_eq)
        self._equity_peak = max(self._equity_peak, new_eq)

        self._cursor += 1
        obs = self._get_obs()

        current = self._equity_history[-1]
        drawdown = (self._equity_peak - current) / (self._equity_peak + 1e-8) if len(self._equity_history) > 1 else 0.0
        reward = _compute_step_reward(portfolio_return, turnover, drawdown, self.drawdown_penalty)

        terminated = self._step_in_episode >= self.episode_days or self._cursor >= self.n_days
        truncated = False

        return _sanitize_obs(obs), reward, terminated, truncated, {"return": portfolio_return}

    def _get_obs(self) -> np.ndarray:
        if self._cursor == 0:
            features = np.zeros((self.n_assets, 13), dtype=np.float32)
            features[:, 1] = 0.5
            features[:, 3] = 0.5
            features[:, 6] = 0.5
            features[:, 9] = 0.5
            features[:, 12] = 0.01
        else:
            features = self._proxy_cache[self._cursor].copy()  # (n_assets, 13)
            if self.daily_aux_features is not None:
                aux_row = self.daily_aux_features[self._cursor - 1]
                for j in range(self.n_assets):
                    features[j] = merge_daily_aux_into_proxy_row(
                        features[j], aux_row[j]
                    )

        holdings = self._holdings if self._holdings is not None else np.zeros(self.n_assets, dtype=np.float32)
        if ENV_OBS_ROLLING_FEATURES:
            per_asset = np.column_stack(
                (features[:, :10], holdings, features[:, 10:13])
            )  # (n_assets, 14)
        else:
            per_asset = np.column_stack((features[:, :10], holdings))  # (n_assets, 11)
        obs = per_asset.flatten().astype(np.float32)
        return _sanitize_obs(obs)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class MixedDataEnv(gym.Env):
    """
    Trading environment that mixes historical and synthetic data.
    On each reset(), randomly selects HistoricalTradingEnv or TradingEnv (with specialist_fn)
    based on historical_ratio. Allows the model to learn both real market dynamics and
    generalizable strategies from synthetic data.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        historical_env: HistoricalTradingEnv,
        synthetic_env: TradingEnv,
        historical_ratio: float,
    ):
        super().__init__()
        if not 0 <= historical_ratio <= 1:
            raise ValueError("historical_ratio must be in [0, 1]")
        self.historical_env = historical_env
        self.synthetic_env = synthetic_env
        self.historical_ratio = historical_ratio
        self._current = historical_env

        self.observation_space = historical_env.observation_space
        self.action_space = historical_env.action_space
        self.np_random = np.random.default_rng()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self.np_random.random() < self.historical_ratio:
            self._current = self.historical_env
        else:
            self._current = self.synthetic_env
        return self._current.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        return self._current.step(action)

    def render(self) -> None:
        pass

    def close(self) -> None:
        self.historical_env.close()
        self.synthetic_env.close()
