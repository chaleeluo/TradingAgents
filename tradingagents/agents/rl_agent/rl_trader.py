"""Reinforcement Learning trading agent using PPO.

Provides:
- ``RLTrader``: train a PPO model on historical data, then use it for
  inference.
- ``create_rl_trader``: factory that returns a callable compatible with the
  existing LangGraph agent pipeline (returns ``{"rl_signal": ..., "messages": ...}``).

Usage (standalone)::

    trader = RLTrader()
    trader.train(df_train, total_timesteps=10_000)
    signal = trader.predict(df_test)
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.agents.rl_agent.trading_env import TradingEnv
from tradingagents.default_config import DEFAULT_CONFIG

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

    class PPO:  # type: ignore[no-redef]
        pass

    class DummyVecEnv:  # type: ignore[no-redef]
        pass

    class EvalCallback:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(DEFAULT_CONFIG.get("data_cache_dir", ""), "rl_models")
os.makedirs(_MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# RL Trader
# ---------------------------------------------------------------------------


class RLTrader:
    """PPO-based trading agent.

    Parameters
    ----------
    model_path : str, optional
        Path to a pre-trained SB3 model zip file.  When provided,
        ``load()`` is called automatically.
    policy : str
        SB3 policy network type (``"MlpPolicy"`` by default).
    learning_rate : float
    """

    def __init__(
        self,
        model_path: str | None = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        device: str = "auto",
    ):
        self.policy = policy
        self.learning_rate = learning_rate
        self.device = device
        self._model: Any = None
        self._env_spec: dict[str, Any] = {}  # saved alongside model

        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        *,
        total_timesteps: int = 20_000,
        eval_frac: float = 0.2,
        feature_columns: list[str] | None = None,
        window_size: int = 20,
        risk_penalty: float = 0.5,
        trade_cost: float = 0.001,
        progress: bool = True,
    ) -> dict[str, float]:
        """Train the PPO agent on historical price data.

        Returns
        -------
        dict with keys ``mean_reward``, ``mean_ep_len`` from the evaluation
        run after training completes.
        """
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL training. "
                "Install it with: pip install tradingagents[rl]"
            )

        # Split into train / eval
        n = len(df)
        split = int(n * (1 - eval_frac))
        df_train, df_eval = df.iloc[:split].copy(), df.iloc[split:].copy()

        train_features = TradingEnv.compute_features(df_train)
        eval_features = TradingEnv.compute_features(df_eval)

        # Ensure eval has enough data (at least window_size + 2 rows)
        min_eval_rows = window_size + 2
        if len(eval_features) < min_eval_rows:
            # Adjust split to give eval more data
            split = max(0, n - min_eval_rows)
            df_train, df_eval = df.iloc[:split].copy(), df.iloc[split:].copy()
            train_features = TradingEnv.compute_features(df_train)
            eval_features = TradingEnv.compute_features(df_eval)

        train_env = DummyVecEnv([
            lambda: TradingEnv(
                train_features,
                window_size=window_size,
                risk_penalty=risk_penalty,
                trade_cost=trade_cost,
            )
        ])
        # Use non-vectorized env for evaluation (simpler step/reset interface)
        eval_env = TradingEnv(
            eval_features,
            window_size=window_size,
            risk_penalty=risk_penalty,
            trade_cost=trade_cost,
        )

        self._model = PPO(
            self.policy,
            train_env,
            learning_rate=self.learning_rate,
            device=self.device,
            verbose=1 if progress else 0,
        )

        callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(_MODEL_DIR, "best"),
            log_path=os.path.join(_MODEL_DIR, "logs"),
            eval_freq=max(1_000, total_timesteps // 10),
            deterministic=True,
            render=False,
        )

        self._model.learn(total_timesteps=total_timesteps, progress_bar=progress, callback=callback)

        # Final evaluation
        rewards, ep_lens = self._evaluate(eval_env, n_eval_episodes=5)
        self._env_spec = {
            "feature_columns": feature_columns,
            "window_size": window_size,
            "risk_penalty": risk_penalty,
            "trade_cost": trade_cost,
        }

        # eval_env is non-vectorized, no close() needed
        train_env.close()

        return {"mean_reward": float(np.mean(rewards)), "mean_ep_len": float(np.mean(ep_lens))}

    # ------------------------------------------------------------------
    # Prediction / Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run the trained model on a feature-augmented DataFrame.

        Returns an array of integer actions (0=Hold, 1=Buy, 2=Sell) of the
        same length as ``df``.
        """
        if self._model is None:
            raise RuntimeError("No trained model. Call train() or load() first.")

        env = TradingEnv(
            df,
            window_size=self._env_spec.get("window_size", 20),
            risk_penalty=self._env_spec.get("risk_penalty", 0.5),
            trade_cost=self._env_spec.get("trade_cost", 0.001),
        )
        obs, _ = env.reset()
        actions: list[int] = []
        done = False
        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            actions.append(int(action))
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        return np.array(actions, dtype=np.int32)

    def predict_signal(self, df: pd.DataFrame) -> str:
        """Convenience: returns ``"Buy"`` / ``"Hold"`` / ``"Sell"`` for the
        *last* time step (the most recent signal)."""
        actions = self.predict(df)
        return TradingEnv.ACTIONS[int(actions[-1])]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save the trained model and its environment spec to disk.

        Returns the path of the saved model.
        """
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        if path is None:
            path = os.path.join(_MODEL_DIR, "rl_trader_model.zip")
        self._model.save(path)
        # Environment spec
        spec_path = Path(path).with_suffix(".pkl")
        with open(spec_path, "wb") as f:
            pickle.dump(self._env_spec, f)
        return str(path)

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required to load RL models.")
        self._model = PPO.load(path, device=self.device)
        spec_path = Path(path).with_suffix(".pkl")
        if spec_path.exists():
            with open(spec_path, "rb") as f:
                self._env_spec = pickle.load(f)
        self._model.set_env(DummyVecEnv([lambda: TradingEnv(
            pd.DataFrame(),
            window_size=self._env_spec.get("window_size", 20),
        )]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(env, n_eval_episodes: int = 5) -> tuple[list[float], list[int]]:
        rewards: list[float] = []
        ep_lens: list[int] = []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
            rewards.append(total)
            ep_lens.append(steps)
        return rewards, ep_lens


# ---------------------------------------------------------------------------
# LangGraph-compatible node factory
# ---------------------------------------------------------------------------


def create_rl_trader(model_path: str | None = None):
    """Factory: returns an agent node callable compatible with the LangGraph
    ``AgentState`` message-passing convention.

    The returned function signature matches ``create_trader()`` from the
    existing pipeline::

        def rl_node(state: AgentState) -> dict[str, Any]:

    It reads ``company_of_interest`` from state, fetches OHLCV data via
    yfinance, runs the RL model, and returns ``{"rl_signal": signal,
    "messages": [AIMessage(content=...)]}``.

    When ``model_path`` is omitted the node raises a clear error at runtime.
    """
    _trader = RLTrader(model_path) if model_path else None

    def rl_node(state, name: str = "RLTrader"):
        from langchain_core.messages import AIMessage

        ticker = state["company_of_interest"]

        if _trader is None or _trader._model is None:
            msg = (
                f"RLTrader node has no trained model for {ticker}. "
                "Train or load a model before running inference."
            )
            return {"rl_signal": "Hold", "messages": [AIMessage(content=msg)]}

        # Fetch recent price data
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            if len(hist) < 30:
                return {"rl_signal": "Hold", "messages": [AIMessage(content=f"Insufficient data for {ticker}")]}

            features = TradingEnv.compute_features(hist)
            signal = _trader.predict_signal(features)
        except Exception as exc:
            signal = "Hold"

        return {
            "rl_signal": signal,
            "messages": [AIMessage(content=f"RL Signal: {signal}")],
        }

    return rl_node
