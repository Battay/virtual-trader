"""Trading environments will be implemented in Milestone 3B."""
"""Gymnasium trading environments."""

from .action_validity import (
    MASKING_STATUS,
    SECTOR_ACTION_VALIDITY_VERSION,
    ActionOutcome,
    ActionValidityDecision,
    PortfolioState,
    action_mask,
    action_name,
    action_validity_metadata,
    evaluate_action_validity,
    is_action_valid,
    valid_actions,
)
from .config import ENVIRONMENT_VERSION, SingleSymbolEnvConfig
from .reward import (
    SECTOR_REWARD_VERSION,
    RewardComponents,
    RewardDiagnosticsAccumulator,
    SectorRewardConfig,
    calculate_reward_components,
)


def __getattr__(name: str):
    """Load the environment lazily so validation remains CLI-safe."""
    if name in {"SingleSymbolTradingEnv", "SingleSymbolEnv"}:
        from .single_symbol_env import SingleSymbolTradingEnv

        values = {
            "SingleSymbolTradingEnv": SingleSymbolTradingEnv,
            "SingleSymbolEnv": SingleSymbolTradingEnv,
        }
        return values[name]
    if name in {
        "EqualSymbolEpisodeSampler",
        "SectorEnvironmentError",
        "SectorTrainingEnv",
    }:
        from .sector_training_env import (
            EqualSymbolEpisodeSampler,
            SectorEnvironmentError,
            SectorTrainingEnv,
        )

        return {
            "EqualSymbolEpisodeSampler": EqualSymbolEpisodeSampler,
            "SectorEnvironmentError": SectorEnvironmentError,
            "SectorTrainingEnv": SectorTrainingEnv,
        }[name]
    raise AttributeError(name)

__all__ = (
    "ActionOutcome",
    "ActionValidityDecision",
    "ENVIRONMENT_VERSION",
    "EqualSymbolEpisodeSampler",
    "MASKING_STATUS",
    "PortfolioState",
    "RewardComponents",
    "RewardDiagnosticsAccumulator",
    "SECTOR_ACTION_VALIDITY_VERSION",
    "SECTOR_REWARD_VERSION",
    "SectorEnvironmentError",
    "SectorRewardConfig",
    "SectorTrainingEnv",
    "SingleSymbolEnv",
    "SingleSymbolEnvConfig",
    "SingleSymbolTradingEnv",
    "action_mask",
    "action_name",
    "action_validity_metadata",
    "calculate_reward_components",
    "evaluate_action_validity",
    "is_action_valid",
    "valid_actions",
)
