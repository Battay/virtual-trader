"""Stable-Baselines3 PPO training contracts for one PSX symbol."""

from .callbacks import ProgressHandler, TrainingProgress
from .config import PPO_CONFIG_VERSION, PPOConfig
from .devices import (
    SUPPORTED_TORCH_DEVICES,
    TorchDeviceError,
    TorchDeviceResolution,
    resolve_torch_device,
)
from .results import PPOTrainingDiagnostics, PPOTrainingResult
from .sector_recurrent_config import (
    SECTOR_RECURRENT_TRAINER_VERSION,
    SectorRecurrentPPOConfig,
)
from .sector_recurrent_results import (
    SectorRecurrentTrainingResult,
    SectorValidationResult,
)


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
    if name in {
        "SectorRecurrentTrainerError",
        "load_sector_training_universe",
        "train_sector_recurrent_ppo",
    }:
        from .sector_recurrent_trainer import (
            SectorRecurrentTrainerError,
            load_sector_training_universe,
            train_sector_recurrent_ppo,
        )

        return {
            "SectorRecurrentTrainerError": SectorRecurrentTrainerError,
            "load_sector_training_universe": load_sector_training_universe,
            "train_sector_recurrent_ppo": train_sector_recurrent_ppo,
        }[name]
    raise AttributeError(name)


__all__ = (
    "PPO_CONFIG_VERSION",
    "PPOConfig",
    "PPOTrainingDiagnostics",
    "PPOTrainingResult",
    "PPOTrainerError",
    "ProgressHandler",
    "SUPPORTED_TORCH_DEVICES",
    "TorchDeviceError",
    "TorchDeviceResolution",
    "TrainingProgress",
    "SECTOR_RECURRENT_TRAINER_VERSION",
    "SectorRecurrentPPOConfig",
    "SectorRecurrentTrainerError",
    "SectorRecurrentTrainingResult",
    "SectorValidationResult",
    "create_training_vector_environment",
    "resolve_torch_device",
    "train_single_symbol",
    "load_sector_training_universe",
    "train_sector_recurrent_ppo",
)
