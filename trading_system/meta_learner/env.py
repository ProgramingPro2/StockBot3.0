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
)


def _proxy_specialists_from_returns_np(
    returns: np.ndarray, start: int, window: int = 21
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Derive proxy specialist outputs from rolling returns (numpy, no pandas).
    Returns (sentiment, quant_prob, fund_score, regime_flag, momentum_score,
             rolling_return_5d, rolling_return_21d, rolling_vol_21d).
    """
    lookback = returns[max(0, start - window) : start]
    lookback = lookback[~np.isnan(lookback)]
    if len(lookback) == 0:
        return 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.01

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

    ret_5d = float(np.mean(lookback[-5:])) if len(lookback) >= 5 else mean_ret
    ret_21d = float(np.mean(lookback[-21:])) if len(lookback) >= 21 else mean_ret
    vol_21d = float(np.std(lookback[-21:])) if len(lookback) >= 21 else std_ret
    if vol_21d < 1e-8:
        vol_21d = 0.01

    # Normalize for obs: clip to roughly [-1, 1]
    ret_5d_norm = np.clip(ret_5d * 50, -1, 1)
    ret_21d_norm = np.clip(ret_21d * 50, -1, 1)
    vol_21d_norm = np.clip((vol_21d - 0.01) * 50, -1, 1)

    return sentiment_proxy, quant_prob, fund_score_proxy, regime_flag, momentum_proxy, ret_5d_norm, ret_21d_norm, vol_21d_norm


def _obs_features_per_asset() -> int:
    """Number of observation features per asset."""
    base = 6  # sentiment, quant_prob, fund_score, regime_flag, momentum_score, holdings
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
        self._step_count = 0
        self._specialist_outputs: list[dict] | None = None
        self._rolling_returns: np.ndarray | None = None  # (n_assets, 21) per-asset
        self._rolling_vols: np.ndarray | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._returns_history = []
        self._equity_history = [1.0]
        self._step_count = 0
        specialist = options.get("specialist_outputs", []) if options else []
        if isinstance(specialist, dict):
            specialist = [specialist] * self.n_assets
        self._specialist_outputs = (specialist + [{}] * self.n_assets)[: self.n_assets]
        self._rolling_returns = np.full((self.n_assets, 21), np.nan, dtype=np.float32)
        self._rolling_vols = np.full(self.n_assets, 0.01, dtype=np.float32)

        obs = self._get_obs(holdings=np.zeros(self.n_assets, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = np.clip(action, 0, 1).astype(np.float32)

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

        action_sum = float(np.sum(action)) + 1e-8
        portfolio_return = float(np.dot(action, returns_per_asset) / action_sum)
        self._returns_history.append(portfolio_return)
        self._equity_history.append(self._equity_history[-1] * (1 + portfolio_return))

        holdings = action
        obs = self._get_obs(holdings=holdings)

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

        terminated = self._step_count >= self.episode_days
        truncated = False

        return obs, reward, terminated, truncated, {"return": portfolio_return}

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
        return np.clip(obs, -1, 1)

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
        self._holdings = action

        if self._cursor >= self.n_days:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"return": 0.0}

        day_returns = self.returns_array[self._cursor]
        action_sum = float(np.sum(action)) + 1e-8
        portfolio_return = float(np.dot(action, day_returns) / action_sum)
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

        terminated = self._step_in_episode >= self.episode_days or self._cursor >= self.n_days
        truncated = False

        return obs, reward, terminated, truncated, {"return": portfolio_return}

    def _get_obs(self) -> np.ndarray:
        parts: list[float] = []
        for i in range(self.n_assets):
            ret_series = self.returns_array[: self._cursor, i]
            s, q, f, r, mom, ret_5d, ret_21d, vol_21d = _proxy_specialists_from_returns_np(
                ret_series, len(ret_series), window=21
            )
            parts.append(s)
            parts.append(q)
            parts.append(f)
            parts.append(r)
            parts.append(mom)
            parts.append(float(self._holdings[i]) if self._holdings is not None else 0.0)
            if ENV_OBS_ROLLING_FEATURES:
                parts.extend([ret_5d, ret_21d, vol_21d])
        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, -1, 1)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
