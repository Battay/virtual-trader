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
    raise AttributeError(name)

__all__ = (
    "ENVIRONMENT_VERSION",
    "SingleSymbolEnvConfig",
    "SingleSymbolEnv",
    "SingleSymbolTradingEnv",
    "action_name",
)
