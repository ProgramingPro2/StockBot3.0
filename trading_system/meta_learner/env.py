"""
Custom Gymnasium environment for the RL meta-learner.
Observation: [sentiment, quant_prob, fund_score, regime_flag, current_holdings_pct]
Action: position size % per asset
Reward: Sharpe ratio over rolling window
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Trading environment for PPO meta-learner.
    Single-asset for simplicity; extensible to multi-asset.
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

        # Observation: sentiment, quant_prob, fund_score, regime_flag, holdings
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4 + n_assets,),
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
        self._specialist_outputs: dict | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._returns_history = []
        self._step_count = 0
        specialist = options.get("specialist_outputs", {}) if options else {}
        self._specialist_outputs = specialist

        obs = self._get_obs(holdings=np.zeros(self.n_assets, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = np.clip(action, 0, 1).astype(np.float32)

        # Simulate return based on action and specialist outputs
        specialist = self._specialist_outputs or {}
        sentiment = specialist.get("sentiment", 0.0)
        quant_prob = specialist.get("quant_prob", 0.5)
        fund_score = specialist.get("fund_score", 0.0)
        regime_flag = specialist.get("regime_flag", 0.5)

        # Combined signal: positive = bullish
        signal = (sentiment + (quant_prob - 0.5) * 2 + fund_score - regime_flag) / 4
        # Simulated daily return (scaled by action)
        daily_return = signal * 0.01 * np.mean(action) + np.random.normal(0, 0.01)
        self._returns_history.append(daily_return)

        holdings = action
        obs = self._get_obs(holdings=holdings)

        # Reward: rolling Sharpe (or simple return if not enough history)
        if len(self._returns_history) >= self.rolling_sharpe_days:
            recent = np.array(self._returns_history[-self.rolling_sharpe_days:])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            reward = float(np.clip(sharpe, -2, 2))
        else:
            reward = daily_return * 100

        terminated = self._step_count >= self.episode_days
        truncated = False

        return obs, reward, terminated, truncated, {"return": daily_return}

    def _get_obs(self, holdings: np.ndarray) -> np.ndarray:
        specialist = self._specialist_outputs or {}
        sentiment = specialist.get("sentiment", 0.0)
        quant_prob = specialist.get("quant_prob", 0.5)
        fund_score = specialist.get("fund_score", 0.0)
        regime_flag = specialist.get("regime_flag", 0.5)

        obs = np.array(
            [sentiment, quant_prob, fund_score, regime_flag, *holdings],
            dtype=np.float32,
        )
        return np.clip(obs, -1, 1)

    def render(self) -> None:
        pass  # Headless

    def close(self) -> None:
        pass
