"""Look-ahead-safe Gymnasium environment for one PSX symbol.

At step t the agent observes row t, trades at row t+1 open, and the portfolio is
marked at row t+1 close before reward is calculated. The next open is never part
of the row-t observation.
"""

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .action_validity import (
    ACTION_NAMES,
    ActionOutcome,
    action_mask,
    action_name,
    evaluate_action_validity,
    finalize_action_outcome,
    portfolio_state_from_shares,
    require_supported_action_mode,
    valid_actions,
)
from .config import DYNAMIC_PORTFOLIO_FEATURES, SingleSymbolEnvConfig
from .reward import RewardComponents, calculate_reward_components
from .validation import prepare_single_symbol_data


PORTFOLIO_OBSERVATION_FEATURES = DYNAMIC_PORTFOLIO_FEATURES


class SingleSymbolTradingEnv(gym.Env[np.ndarray, int]):
    """Deterministic long-only, all-in/all-out trading environment v1."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(
        self,
        data: pd.DataFrame,
        config: SingleSymbolEnvConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "ansi", "human"}:
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.config = config or SingleSymbolEnvConfig()
        require_supported_action_mode(self.config.invalid_action_mode)
        self.render_mode = render_mode
        self._data = prepare_single_symbol_data(
            data,
            feature_columns=self.config.feature_columns,
        )
        if len(self._data) < 2:
            raise ValueError("At least two trading observations are required")
        self.symbol = str(self._data["symbol"].iloc[0])
        self.action_space = spaces.Discrete(3)
        self.observation_feature_names = (
            *self.config.feature_columns,
            *PORTFOLIO_OBSERVATION_FEATURES,
        )
        float32_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-float32_limit,
            high=float32_limit,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )
        self._history: list[dict[str, object]] = []
        self._terminated = False
        self._truncated = False
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.initial_cash = float(self.config.initial_cash)
        self.cash = self.initial_cash
        self.shares_held = 0
        self.average_entry_price = 0.0
        self.current_position_value = 0.0
        self.total_portfolio_value = self.initial_cash
        self.realized_profit_loss = 0.0
        self.unrealized_profit_loss = 0.0
        self.total_transaction_costs = 0.0
        self.peak_portfolio_value = self.initial_cash
        self.current_drawdown = 0.0
        self.number_of_trades = 0
        self.current_step = 0
        self.current_date = self._data["date"].iloc[0]
        self._episode_steps = 0
        self._history = []
        self._terminated = False
        self._truncated = False
        observation = self._observation()
        return observation, self._info(
            action=None,
            observation_index=0,
            execution_index=None,
            execution_price=None,
            shares_traded=0,
            transaction_cost=0.0,
            reward_components=RewardComponents.zero(),
            action_outcome=None,
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode completion; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be one of {tuple(ACTION_NAMES)}")
        action = int(action)
        state_before_action = portfolio_state_from_shares(self.shares_held)
        validity = evaluate_action_validity(action, state_before_action)
        previous_value = self.total_portfolio_value
        previous_drawdown = self.current_drawdown
        observation_index = self.current_step
        execution_index = observation_index + 1
        if execution_index >= len(self._data):
            raise RuntimeError("No next trading observation is available")
        next_row = self._data.iloc[execution_index]
        raw_open = float(next_row["open"])
        execution_price: float | None = None
        shares_traded = 0
        commission = 0.0
        slippage_cost = 0.0
        execution_failure_reason: str | None = None

        if action == 1 and validity.state_valid:
            execution_price = raw_open * (1 + self.config.slippage_rate)
            affordable = math.floor(
                self.cash
                / (execution_price * (1 + self.config.commission_rate))
            )
            if affordable < 1:
                execution_failure_reason = "Insufficient cash to buy one whole share"
            else:
                shares_traded = affordable
                transaction_value = shares_traded * execution_price
                commission = transaction_value * self.config.commission_rate
                slippage_cost = shares_traded * (execution_price - raw_open)
                total_outflow = transaction_value + commission
                existing_cost = self.average_entry_price * self.shares_held
                self.cash -= total_outflow
                self.shares_held += shares_traded
                self.average_entry_price = (
                    existing_cost + total_outflow
                ) / self.shares_held
                self.number_of_trades += 1
        elif action == 2 and validity.state_valid:
            execution_price = raw_open * (1 - self.config.slippage_rate)
            quantity = self.shares_held
            shares_traded = -quantity
            transaction_value = quantity * execution_price
            commission = transaction_value * self.config.commission_rate
            slippage_cost = quantity * (raw_open - execution_price)
            net_proceeds = transaction_value - commission
            self.cash += net_proceeds
            self.realized_profit_loss += (
                net_proceeds - self.average_entry_price * quantity
            )
            self.shares_held = 0
            self.average_entry_price = 0.0
            self.number_of_trades += 1

        action_outcome = finalize_action_outcome(
            validity,
            execution_failure_reason=execution_failure_reason,
            trade_executed=shares_traded != 0,
        )

        transaction_cost = commission + slippage_cost
        self.total_transaction_costs += transaction_cost
        self.current_step = execution_index
        self.current_date = next_row["date"]
        close_price = float(next_row["close"])
        self.current_position_value = self.shares_held * close_price
        self.unrealized_profit_loss = (
            self.current_position_value
            - self.average_entry_price * self.shares_held
            if self.shares_held
            else 0.0
        )
        self.total_portfolio_value = self.cash + self.current_position_value
        self.peak_portfolio_value = max(
            self.peak_portfolio_value,
            self.total_portfolio_value,
        )
        self.current_drawdown = (
            (self.peak_portfolio_value - self.total_portfolio_value)
            / self.peak_portfolio_value
        )
        if previous_value <= 0 or self.total_portfolio_value <= 0:
            raise RuntimeError("Portfolio value must remain positive")
        reward_components = calculate_reward_components(
            config=self.config.reward_config,
            previous_portfolio_value=previous_value,
            current_portfolio_value=self.total_portfolio_value,
            transaction_cost=transaction_cost,
            previous_drawdown=previous_drawdown,
            current_drawdown=self.current_drawdown,
            action_invalid=not action_outcome.action_valid,
        )
        reward = reward_components.total_reward
        if not math.isfinite(reward):
            raise RuntimeError("Reward must remain finite")
        self._episode_steps += 1
        self._terminated = self.current_step == len(self._data) - 1
        self._truncated = bool(
            not self._terminated
            and self.config.max_episode_steps is not None
            and self._episode_steps >= self.config.max_episode_steps
        )
        info = self._info(
            action=action,
            observation_index=observation_index,
            execution_index=execution_index,
            execution_price=execution_price,
            shares_traded=shares_traded,
            transaction_cost=transaction_cost,
            reward_components=reward_components,
            action_outcome=action_outcome,
        )
        self._history.append(
            {
                "initial_portfolio_value": self.initial_cash,
                "observation_date": self._data["date"].iloc[observation_index],
                "execution_date": next_row["date"],
                "action": action,
                "action_name": action_name(action),
                "action_valid": action_outcome.action_valid,
                "action_executed": action_outcome.action_executed,
                "trade_executed": action_outcome.trade_executed,
                "invalid_action_reason": (
                    action_outcome.invalid_reason
                    or action_outcome.execution_failure_reason
                ),
                "semantic_invalid_action_reason": action_outcome.invalid_reason,
                "execution_failure_reason": (
                    action_outcome.execution_failure_reason
                ),
                "execution_price": execution_price,
                "shares_traded": shares_traded,
                "transaction_cost": transaction_cost,
                "cash": self.cash,
                "shares_held": self.shares_held,
                "portfolio_value": self.total_portfolio_value,
                "realized_profit_loss": self.realized_profit_loss,
                "unrealized_profit_loss": self.unrealized_profit_loss,
                "drawdown": self.current_drawdown,
                "reward": reward,
                "reward_breakdown": reward_components.to_dict(),
            }
        )
        if self.render_mode == "human":
            self.render()
        return (
            self._observation(),
            reward,
            self._terminated,
            self._truncated,
            info,
        )

    def _observation(self) -> np.ndarray:
        row = self._data.iloc[self.current_step]
        market = [float(row[column]) for column in self.config.feature_columns]
        value = self.total_portfolio_value
        cash_ratio = self.cash / value
        position_ratio = self.current_position_value / value
        invested_cost = self.average_entry_price * self.shares_held
        unrealized_ratio = (
            self.unrealized_profit_loss / invested_cost if invested_cost else 0.0
        )
        values = np.asarray(
            [
                *market,
                cash_ratio,
                position_ratio,
                float(self.shares_held > 0),
                unrealized_ratio,
                self.current_drawdown,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise RuntimeError("Observation contains NaN or infinite values")
        return values

    def _info(
        self,
        *,
        action: int | None,
        observation_index: int,
        execution_index: int | None,
        execution_price: float | None,
        shares_traded: int,
        transaction_cost: float,
        reward_components: RewardComponents,
        action_outcome: ActionOutcome | None,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "environment_version": self.config.environment_version,
            "date": self._data["date"].iloc[observation_index],
            "next_date": (
                self._data["date"].iloc[execution_index]
                if execution_index is not None
                else (
                    self._data["date"].iloc[1]
                    if len(self._data) > 1
                    else None
                )
            ),
            "action": action,
            "action_name": action_name(action) if action is not None else None,
            "execution_price": execution_price,
            "shares_traded": shares_traded,
            "transaction_cost": transaction_cost,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.total_portfolio_value,
            "realized_profit_loss": self.realized_profit_loss,
            "unrealized_profit_loss": self.unrealized_profit_loss,
            "drawdown": self.current_drawdown,
            "reward_components": reward_components.to_legacy_dict(),
            "reward_breakdown": reward_components.to_dict(),
            "reward_version": reward_components.reward_version,
            "action_validity_version": self.config.action_validity_version,
            "invalid_action_mode": self.config.invalid_action_mode,
        }
        if action_outcome is None:
            state = portfolio_state_from_shares(self.shares_held)
            info.update(
                {
                    "selected_action": None,
                    "selected_action_name": None,
                    "portfolio_state_before_action": state.value,
                    "action_state_valid": None,
                    "action_valid": None,
                    "action_executed": None,
                    "trade_executed": False,
                    "invalid_action_reason": None,
                    "semantic_invalid_action_reason": None,
                    "execution_failure_reason": None,
                    "valid_action_ids": valid_actions(state),
                    "valid_action_mask": action_mask(state),
                }
            )
        else:
            info.update(action_outcome.as_dict())
        return info

    def get_history(self) -> pd.DataFrame:
        """Return an isolated copy of structured episode transitions."""
        return pd.DataFrame(self._history).copy(deep=True)

    def render(self) -> str | None:
        summary = (
            f"{self.symbol} {self.current_date.date()} | cash={self.cash:,.2f} | "
            f"shares={self.shares_held} | value={self.total_portfolio_value:,.2f} | "
            f"drawdown={self.current_drawdown:.2%}"
        )
        if self.render_mode == "human":
            print(summary)
            return None
        return summary

    def close(self) -> None:
        """Release environment resources; v1 owns no external handles."""
        return None
