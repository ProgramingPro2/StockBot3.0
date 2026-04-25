"""
Custom Gymnasium environment for the RL meta-learner.
Observation: per-asset [sentiment, quant_prob, fund_score, regime_flag, momentum_score, holdings]
  + optional [rolling_return_5d, rolling_return_21d, rolling_vol_21d].
Action: position size % per asset
Reward: portfolio Sharpe ratio over rolling window, minus drawdown penalty
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import (
    ENV_EPISODE_DAYS,
    ENV_OBS_ROLLING_FEATURES,
    ENV_REWARD_DRAWDOWN_PENALTY,
    ENV_ROLLING_SHARPE_DAYS,
    MIN_TRADE_NOTIONAL,
    SLIPPAGE_BPS,
    STARTING_CAPITAL,
    TRANSACTION_COST_BPS,
)


def _sanitize_obs(obs: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip to [-1, 1] so the policy never sees invalid values."""
    out = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(out, -1.0, 1.0)


def _proxy_specialists_vectorized(returns_2d: np.ndarray) -> np.ndarray:
    """
    Vectorized version: returns_2d shape (window, n_assets).
    Returns shape (n_assets, 13): sentiment, quant_prob, fund_score, regime_flag,
    momentum, mean_reversion, volatility, event_risk, relative_strength, liquidity,
    ret_5d_norm, ret_21d_norm, vol_21d_norm per asset.
    """
    # returns_2d may have NaNs; use nanmean/nanstd for robustness
    mean_ret = np.nanmean(returns_2d, axis=0).astype(np.float32)
    std_ret = np.nanstd(returns_2d, axis=0).astype(np.float32)
    std_ret = np.where(std_ret < 1e-8, 0.01, std_ret)

    sentiment = np.clip(mean_ret * 12, -0.5, 0.5)
    trend = np.clip(mean_ret * 20, -0.5, 0.5)
    quant_prob = 0.5 + trend
    fund_score = np.clip(-mean_ret * 15, -0.5, 0.5)
    vol_norm = np.clip((std_ret - 0.01) * 50, -0.5, 0.5)
    regime_flag = 0.5 + vol_norm
    momentum = np.clip(mean_ret * 15, -1, 1)
    mean_reversion = np.clip(-mean_ret * 20, -0.5, 0.5)
    volatility = np.clip((vol_norm + 1.0) / 2.0, 0.0, 1.0)
    event_risk = np.zeros_like(mean_ret, dtype=np.float32)
    relative_strength = np.zeros_like(mean_ret, dtype=np.float32)
    liquidity = np.full_like(mean_ret, 0.5, dtype=np.float32)

    n = returns_2d.shape[0]
    ret_5d = np.nanmean(returns_2d[-5:] if n >= 5 else returns_2d, axis=0)
    ret_21d = np.nanmean(returns_2d[-21:] if n >= 21 else returns_2d, axis=0)
    vol_21d = np.nanstd(returns_2d[-21:] if n >= 21 else returns_2d, axis=0)
    vol_21d = np.where(vol_21d < 1e-8, 0.01, vol_21d)

    ret_5d_norm = np.clip(ret_5d * 50, -1, 1)
    ret_21d_norm = np.clip(ret_21d * 50, -1, 1)
    vol_21d_norm = np.clip((vol_21d - 0.01) * 50, -1, 1)

    return np.column_stack(
        (
            sentiment,
            quant_prob,
            fund_score,
            regime_flag,
            momentum,
            mean_reversion,
            volatility,
            event_risk,
            relative_strength,
            liquidity,
            ret_5d_norm,
            ret_21d_norm,
            vol_21d_norm,
        )
    )


def _proxy_specialists_from_returns_np(
    returns: np.ndarray, start: int, window: int = 21
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Derive proxy specialist outputs from rolling returns (numpy, no pandas).
    Returns (sentiment, quant_prob, fund_score, regime_flag, momentum_score,
             mean_reversion, volatility, event_risk, relative_strength, liquidity,
             rolling_return_5d, rolling_return_21d, rolling_vol_21d).
    """
    lookback = returns[max(0, start - window) : start]
    lookback = lookback[~np.isnan(lookback)]
    if len(lookback) == 0:
        return 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.01

    mean_ret = float(np.mean(lookback))
    std_ret = float(np.std(lookback))
    if std_ret < 1e-8:
        std_ret = 0.01

    trend = np.clip(mean_ret * 20, -0.5, 0.5)
    quant_prob = 0.5 + trend
    vol_norm = np.clip((std_ret - 0.01) * 50, -0.5, 0.5)
    regime_flag = 0.5 + vol_norm
    sentiment_proxy = float(np.clip(mean_ret * 12, -0.5, 0.5))
    fund_score_proxy = float(np.clip(-mean_ret * 15, -0.5, 0.5))
    momentum_proxy = float(np.clip(mean_ret * 15, -1, 1))
    mean_reversion_proxy = float(np.clip(-mean_ret * 20, -0.5, 0.5))
    volatility_proxy = float(np.clip((vol_norm + 1.0) / 2.0, 0.0, 1.0))
    event_risk_proxy = 0.0
    relative_strength_proxy = 0.0
    liquidity_proxy = 0.5

    ret_5d = float(np.mean(lookback[-5:])) if len(lookback) >= 5 else mean_ret
    ret_21d = float(np.mean(lookback[-21:])) if len(lookback) >= 21 else mean_ret
    vol_21d = float(np.std(lookback[-21:])) if len(lookback) >= 21 else std_ret
    if vol_21d < 1e-8:
        vol_21d = 0.01

    ret_5d_norm = np.clip(ret_5d * 50, -1, 1)
    ret_21d_norm = np.clip(ret_21d * 50, -1, 1)
    vol_21d_norm = np.clip((vol_21d - 0.01) * 50, -1, 1)

    return (
        sentiment_proxy,
        quant_prob,
        fund_score_proxy,
        regime_flag,
        momentum_proxy,
        mean_reversion_proxy,
        volatility_proxy,
        event_risk_proxy,
        relative_strength_proxy,
        liquidity_proxy,
        ret_5d_norm,
        ret_21d_norm,
        vol_21d_norm,
    )


def _obs_features_per_asset() -> int:
    """Number of observation features per asset."""
    base = 11  # 10 specialist scores (incl. relative_strength, liquidity) + holdings
    if ENV_OBS_ROLLING_FEATURES:
        return base + 3  # rolling_return_5d, rolling_return_21d, rolling_vol_21d
    return base


class TradingEnv(gym.Env):
    """
    Trading environment for PPO meta-learner.
    Multi-asset: n_assets sets of specialist features + n_assets holdings.
    Uses synthetic returns during training (for backward compatibility).
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
        self._holdings: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self._step_count = 0
        self._specialist_outputs: list[dict] | None = None
        self._rolling_returns: np.ndarray | None = None  # (n_assets, 21) per-asset
        self._rolling_vols: np.ndarray | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._returns_history = []
        self._equity_history = [1.0]
        self._holdings = np.zeros(self.n_assets, dtype=np.float32)
        self._step_count = 0
        specialist = options.get("specialist_outputs", []) if options else []
        if isinstance(specialist, dict):
            specialist = [specialist] * self.n_assets
        self._specialist_outputs = (specialist + [{}] * self.n_assets)[: self.n_assets]
        self._rolling_returns = np.full((self.n_assets, 21), np.nan, dtype=np.float32)
        self._rolling_vols = np.full(self.n_assets, 0.01, dtype=np.float32)

        obs = self._get_obs(holdings=self._holdings)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = np.clip(action, 0, 1).astype(np.float32)

        min_trade_equity = MIN_TRADE_NOTIONAL / STARTING_CAPITAL
        equity = self._equity_history[-1]
        effective_holdings = self._holdings.copy()
        increased_this_step: set[int] = set()
        for i in range(self.n_assets):
            notional = abs(action[i] - self._holdings[i]) * equity
            if action[i] > self._holdings[i] and notional >= min_trade_equity:
                effective_holdings[i] = action[i]
                increased_this_step.add(i)
        for i in range(self.n_assets):
            if action[i] < self._holdings[i] and i not in increased_this_step:
                notional = abs(action[i] - self._holdings[i]) * equity
                if notional >= min_trade_equity:
                    effective_holdings[i] = action[i]
        turnover = float(np.sum(np.abs(effective_holdings - self._holdings)))
        self._holdings = effective_holdings

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

        # Full-portfolio return: weights sum to <= 1, unallocated is cash (0 return)
        portfolio_return = float(np.dot(effective_holdings, returns_per_asset))
        self._returns_history.append(portfolio_return)
        self._equity_history.append(self._equity_history[-1] * (1 + portfolio_return))

        obs = self._get_obs(holdings=self._holdings)

        if len(self._returns_history) >= self.rolling_sharpe_days:
            recent = np.array(self._returns_history[-self.rolling_sharpe_days :])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            reward = float(np.clip(sharpe, -2, 2))
        else:
            reward = portfolio_return * 100

        if self.drawdown_penalty > 0 and len(self._equity_history) > 1:
            peak = max(self._equity_history)
            current = self._equity_history[-1]
            drawdown = (peak - current) / (peak + 1e-8)
            reward -= self.drawdown_penalty * drawdown
        cost_bps = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
        reward -= turnover * cost_bps * 10.0
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0))

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
    Episodes are rolling windows over the data.
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
    ):
        """
        Args:
            returns_array: Shape (n_days, n_assets), daily returns per asset.
            n_assets: Number of assets.
        """
        super().__init__()
        self.returns_array = np.asarray(returns_array, dtype=np.float32)
        self.n_days, self.n_assets = self.returns_array.shape
        if self.n_assets != n_assets:
            raise ValueError("returns_array n_assets mismatch")
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
        self._holdings: np.ndarray | None = None
        self.np_random = np.random.default_rng()

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
        self._holdings = np.zeros(self.n_assets, dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_in_episode += 1
        action = np.clip(action, 0, 1).astype(np.float32)

        min_trade_equity = MIN_TRADE_NOTIONAL / STARTING_CAPITAL
        equity = self._equity_history[-1]
        effective_holdings = self._holdings.copy()
        increased_this_step: set[int] = set()
        for i in range(self.n_assets):
            notional = abs(action[i] - self._holdings[i]) * equity
            if action[i] > self._holdings[i] and notional >= min_trade_equity:
                effective_holdings[i] = action[i]
                increased_this_step.add(i)
        for i in range(self.n_assets):
            if action[i] < self._holdings[i] and i not in increased_this_step:
                notional = abs(action[i] - self._holdings[i]) * equity
                if notional >= min_trade_equity:
                    effective_holdings[i] = action[i]
        turnover = float(np.sum(np.abs(effective_holdings - self._holdings)))
        self._holdings = effective_holdings

        if self._cursor >= self.n_days:
            obs = self._get_obs()
            return _sanitize_obs(obs), 0.0, True, False, {"return": 0.0}

        day_returns = self.returns_array[self._cursor]
        # Full-portfolio return: weights sum to <= 1, unallocated is cash (0 return)
        portfolio_return = float(np.dot(effective_holdings, day_returns))
        self._returns_history.append(portfolio_return)
        self._equity_history.append(self._equity_history[-1] * (1 + portfolio_return))

        self._cursor += 1
        obs = self._get_obs()

        if len(self._returns_history) >= self.rolling_sharpe_days:
            recent = np.array(self._returns_history[-self.rolling_sharpe_days :])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            reward = float(np.clip(sharpe, -2, 2))
        else:
            reward = portfolio_return * 100

        if self.drawdown_penalty > 0 and len(self._equity_history) > 1:
            peak = max(self._equity_history)
            current = self._equity_history[-1]
            drawdown = (peak - current) / (peak + 1e-8)
            reward -= self.drawdown_penalty * drawdown
        cost_bps = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
        reward -= turnover * cost_bps * 10.0
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0))

        terminated = self._step_in_episode >= self.episode_days or self._cursor >= self.n_days
        truncated = False

        return _sanitize_obs(obs), reward, terminated, truncated, {"return": portfolio_return}

    def _get_obs(self) -> np.ndarray:
        window = 21
        start_idx = max(0, self._cursor - window)
        returns_window = self.returns_array[start_idx : self._cursor, :]  # (window, n_assets)
        if returns_window.size == 0:
            features = np.zeros((self.n_assets, 13), dtype=np.float32)
            features[:, 1] = 0.5
            features[:, 3] = 0.5
            features[:, 6] = 0.5
            features[:, 9] = 0.5  # liquidity
            features[:, 12] = 0.01
        else:
            features = _proxy_specialists_vectorized(returns_window)  # (n_assets, 13)

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
