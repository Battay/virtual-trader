"""Stable-Baselines3 PPO training contracts for one PSX symbol."""

from .callbacks import ProgressHandler, TrainingProgress
from .config import PPO_CONFIG_VERSION, PPOConfig
from .devices import (
    SUPPORTED_TORCH_DEVICES,
    TorchDeviceError,
    TorchDeviceResolution,
    resolve_torch_device,
)
from .results import PPOTrainingResult


def __getattr__(name: str):
    """Load the trainer lazily so module CLI execution remains warning-free."""
    if name in {
        "PPOTrainerError",
        "create_training_vector_environment",
        "train_single_symbol",
    }:
        from .ppo_trainer import (
            PPOTrainerError,
            create_training_vector_environment,
            train_single_symbol,
        )

        values = {
            "PPOTrainerError": PPOTrainerError,
            "create_training_vector_environment": create_training_vector_environment,
            "train_single_symbol": train_single_symbol,
        }
        return values[name]
    raise AttributeError(name)


__all__ = (
    "PPO_CONFIG_VERSION",
    "PPOConfig",
    "PPOTrainingResult",
    "PPOTrainerError",
    "ProgressHandler",
    "SUPPORTED_TORCH_DEVICES",
    "TorchDeviceError",
    "TorchDeviceResolution",
    "TrainingProgress",
    "create_training_vector_environment",
    "resolve_torch_device",
    "train_single_symbol",
)
