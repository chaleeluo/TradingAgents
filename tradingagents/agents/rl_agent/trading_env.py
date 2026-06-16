"""Gymnasium trading environment for reinforcement learning.

State space: price-derived features + technical indicators.
Action space: Discrete 3 (Buy / Hold / Sell).
Reward: risk-adjusted log return.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from tradingagents.default_config import DEFAULT_CONFIG


class TradingEnv(gym.Env):
    """A discrete-action trading environment.

    Observations
    ------------
    Normalised feature vector derived from OHLCV data and technical
    indicators.  The exact feature set is controlled by ``feature_columns``.

    Actions
    -------
    0 → Hold, 1 → Buy (go long 1 unit), 2 → Sell (go short 1 unit).

    Reward
    ------
    Log return of the portfolio, scaled by a volatility penalty so the agent
    learns risk-adjusted behaviour instead of raw absolute return.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Action labels – used by the agent bridge & tests
    ACTIONS = ("Hold", "Buy", "Sell")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        window_size: int = 20,
        risk_penalty: float = 0.5,
        trade_cost: float = 0.001,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Store raw prices for return calculation
        self._raw_prices = None
        if "close" in df.columns and "high" in df.columns and "low" in df.columns:
            # Raw OHLCV data - compute features
            self._raw_prices = df["close"].values
            df = TradingEnv.compute_features(df)

        self.df = df.reset_index(drop=True)

        self.window_size = window_size
        self.risk_penalty = risk_penalty
        self.trade_cost = trade_cost
        self.render_mode = render_mode

        # Infer feature columns: exclude known non-feature columns.
        _exclude = {"open", "high", "low", "close", "volume", "date", "timestamp"}
        self.feature_columns = (
            feature_columns
            if feature_columns is not None
            else [c for c in self.df.select_dtypes(include=[np.number]).columns
                  if c.lower() not in _exclude]
        )

        # Observation: window_size x num_features (flattened)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * len(self.feature_columns),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"{action} ∉ {self.action_space}"

        # Get prices from raw data if available
        if self._raw_prices is not None:
            prev_price = float(self._raw_prices[self._idx])
            self._idx += 1
            done = self._idx >= len(self._raw_prices) - 1
            curr_price = float(self._raw_prices[self._idx])
        else:
            # For feature dataframes, we need to estimate price changes
            # Use a simple random walk for testing
            import random
            prev_price = 100.0  # Base price
            self._idx += 1
            done = self._idx >= len(self.df) - 1
            # Random price change for testing
            curr_price = prev_price * (1 + random.uniform(-0.02, 0.02))

        # Portfolio return for this step
        step_return = self._compute_return(action, prev_price, curr_price)
        self._portfolio_value *= np.exp(step_return)
        self._returns_history.append(step_return)

        reward = self._compute_reward(step_return)
        self._action_history.append(action)

        return self._observe(), reward, done, False, self._info()

    def render(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            prices = self.df["close"].values[: self._idx + 1]
            ax1.plot(prices)
            ax1.set_title("Price")
            ax2.plot(np.exp(np.cumsum(self._returns_history)))
            ax2.set_title("Portfolio Value")
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._idx = self.window_size
        self._portfolio_value = 1.0
        self._returns_history: list[float] = []
        self._action_history: list[int] = []
        self._last_action = 0  # start flat

    def _observe(self) -> np.ndarray:
        window = self.df.iloc[self._idx - self.window_size : self._idx]
        obs = window[self.feature_columns].values.astype(np.float32)
        # Normalise each feature column to [0, 1] within the window
        col_min = obs.min(axis=0, keepdims=True)
        col_max = obs.max(axis=0, keepdims=True)
        obs = (obs - col_min) / (col_max - col_min + 1e-8)
        return obs.flatten()

    def _compute_return(self, action: int, prev_price: float, curr_price: float) -> float:
        price_return = (curr_price - prev_price) / prev_price
        if action == 1:  # Buy
            net = price_return - self.trade_cost
        elif action == 2:  # Sell
            net = -price_return - self.trade_cost
        else:  # Hold
            net = 0.0
        return net

    def _compute_reward(self, step_return: float) -> float:
        self._returns_history.append(step_return)
        if len(self._returns_history) < 2:
            return step_return
        recent = self._returns_history[-min(20, len(self._returns_history)):
                                       ]
        volatility = np.std(recent) + 1e-8
        return step_return - self.risk_penalty * volatility

    def _info(self) -> dict[str, Any]:
        return {
            "idx": self._idx,
            "portfolio_value": float(self._portfolio_value),
            "total_return": float(np.exp(np.sum(self._returns_history)) - 1),
        }

    # ------------------------------------------------------------------
    # Utility: build features from a raw OHLCV DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Augment a raw OHLCV DataFrame with technical indicator columns.

        Expects columns: ``open, high, low, close, volume`` (case-insensitive).
        Returns a new DataFrame with additional feature columns; the original
        columns are dropped so they don't leak into the observation space.
        """
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Price-derived features
        data["returns"] = np.diff(close, prepend=close[0]) / close[0]
        data["log_returns"] = np.log(close / np.roll(close, 1))
        data["high_low_pct"] = (high - low) / close
        data["volume_change"] = np.diff(volume, prepend=volume[0]) / (volume[0] + 1e-8)

        # Simple Moving Averages
        data["sma_5"] = _sma(close, 5)
        data["sma_10"] = _sma(close, 10)
        data["sma_20"] = _sma(close, 20)
        data["close_sma_5"] = close / (data["sma_5"] + 1e-8)

        # RSI
        data["rsi"] = _rsi(close, 14)

        # Bollinger Bands
        bb_mid = _sma(close, 20)
        bb_std = _rolling_std(close, 20)
        data["bb_width"] = 2 * bb_std / (bb_mid + 1e-8)
        data["bb_position"] = (close - bb_mid) / (2 * bb_std + 1e-8)

        # ATR
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        data["atr"] = _sma(tr, 14)

        # MACD
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        data["macd"] = ema12 - ema26
        data["macd_signal"] = _ema(data["macd"].values, 9)
        data["macd_hist"] = data["macd"] - data["macd_signal"]

        # Drop original price columns
        drop_cols = {"open", "high", "low", "close", "volume"}
        data = data.drop(columns=[c for c in drop_cols if c in data.columns])

        # Drop rows with NaN from indicator computation
        data = data.dropna().reset_index(drop=True)
        return data


# ------------------------------------------------------------------
# Standalone helpers (no pandas dependency in the critical path)
# ------------------------------------------------------------------


def _sma(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        out[i] = np.mean(values[i - window + 1 : i + 1])
    return out


def _ema(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    multiplier = 2.0 / (window + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (values[i] - out[i - 1]) * multiplier + out[i - 1]
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        out[i] = np.std(values[i - window + 1 : i + 1])
    return out


def _rsi(values: np.ndarray, window: int = 14) -> np.ndarray:
    deltas = np.diff(values, prepend=values[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = _sma(gains, window)
    avg_loss = _sma(losses, window)
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))
