"""
Custom Gymnasium environment for the RL meta-learner.
Observation: per-asset [sentiment, quant_prob, fund_score, regime_flag] then holdings.
Action: position size % per asset
Reward: portfolio Sharpe ratio over rolling window
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Trading environment for PPO meta-learner.
    Multi-asset: n_assets sets of specialist features + n_assets holdings.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_assets: int = 1,
        rolling_sharpe_days: int = 252,
        episode_days: int = 21,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.rolling_sharpe_days = rolling_sharpe_days
        self.episode_days = episode_days
        self.render_mode = render_mode

        # Observation: 4 specialist features per asset + holdings per asset = 5 * n_assets
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5 * n_assets,),
            dtype=np.float32,
        )

        # Action: position size % per asset (0 to 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        self._returns_history: list[float] = []
        self._step_count = 0
        self._specialist_outputs: list[dict] | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._returns_history = []
        self._step_count = 0
        specialist = options.get("specialist_outputs", []) if options else []
        # Accept list of n_assets dicts; pad or truncate to n_assets
        if isinstance(specialist, dict):
            specialist = [specialist] * self.n_assets
        self._specialist_outputs = (specialist + [{}] * self.n_assets)[: self.n_assets]

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
            returns_per_asset[i] = signal * 0.01 + np.random.normal(0, 0.01)

        action_sum = float(np.sum(action)) + 1e-8
        portfolio_return = float(np.dot(action, returns_per_asset) / action_sum)
        self._returns_history.append(portfolio_return)

        holdings = action
        obs = self._get_obs(holdings=holdings)

        if len(self._returns_history) >= self.rolling_sharpe_days:
            recent = np.array(self._returns_history[-self.rolling_sharpe_days:])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            reward = float(np.clip(sharpe, -2, 2))
        else:
            reward = portfolio_return * 100

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
        parts.extend(holdings.tolist())
        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, -1, 1)

    def render(self) -> None:
        pass  # Headless

    def close(self) -> None:
        pass
