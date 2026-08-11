"""Trading environments will be implemented in Milestone 3B."""
"""Gymnasium trading environments."""

from .config import ENVIRONMENT_VERSION, SingleSymbolEnvConfig


def __getattr__(name: str):
    """Load the environment lazily so validation remains CLI-safe."""
    if name in {"SingleSymbolTradingEnv", "SingleSymbolEnv", "action_name"}:
        from .single_symbol_env import SingleSymbolTradingEnv, action_name

        values = {
            "SingleSymbolTradingEnv": SingleSymbolTradingEnv,
            "SingleSymbolEnv": SingleSymbolTradingEnv,
            "action_name": action_name,
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
    "ENVIRONMENT_VERSION",
    "EqualSymbolEpisodeSampler",
    "SectorEnvironmentError",
    "SectorTrainingEnv",
    "SingleSymbolEnvConfig",
    "SingleSymbolEnv",
    "SingleSymbolTradingEnv",
    "action_name",
)
