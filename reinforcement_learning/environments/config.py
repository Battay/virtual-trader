"""Configuration contract for the single-symbol trading environment."""

from dataclasses import dataclass


ENVIRONMENT_VERSION = "single_symbol_env_v1"

DEFAULT_OBSERVATION_FEATURES = (
    "simple_return",
    "log_return",
    "high_low_range",
    "open_close_return",
    "rolling_volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "obv",
    "volume_ma_20",
)


@dataclass(frozen=True)
class SingleSymbolEnvConfig:
    """Explicit simulation and reward parameters.

    Commission and slippage defaults are conservative development assumptions,
    not a representation of an exact PSX broker fee schedule.
    """

    environment_version: str = ENVIRONMENT_VERSION
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    feature_columns: tuple[str, ...] = DEFAULT_OBSERVATION_FEATURES
    transaction_cost_penalty_weight: float = 0.0
    drawdown_penalty_weight: float = 0.1
    invalid_action_penalty: float = 0.0001
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        if self.environment_version != ENVIRONMENT_VERSION:
            raise ValueError(f"environment_version must be {ENVIRONMENT_VERSION!r}")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        for name in (
            "commission_rate",
            "slippage_rate",
            "transaction_cost_penalty_weight",
            "drawdown_penalty_weight",
            "invalid_action_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.commission_rate >= 1 or self.slippage_rate >= 1:
            raise ValueError("commission_rate and slippage_rate must be below 1")
        if not self.feature_columns or len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise ValueError("feature_columns must be non-empty and unique")
        forbidden = {"symbol", "date"}.intersection(self.feature_columns)
        if forbidden:
            raise ValueError("symbol and date cannot be observation features")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive when configured")
