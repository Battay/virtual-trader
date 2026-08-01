"""Central paths for future PPO models and their companion artifacts."""

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.src.config import MASTER_MODELS_DIR, SYMBOL_MODELS_DIR
from feature_engineering.storage import safe_path_component


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Reserved paths for one immutable future model version."""

    model: Path
    scaler: Path
    metrics: Path


def symbol_model_paths(symbol: str, version: int) -> ModelArtifactPaths:
    """Return deterministic artifact paths for one symbol-model version."""
    if version < 1:
        raise ValueError("model version must be at least 1")
    component = safe_path_component(symbol)
    directory = Path(SYMBOL_MODELS_DIR) / component / f"v{version:04d}"
    return ModelArtifactPaths(
        model=directory / "ppo_model.zip",
        scaler=directory / "standard_scaler.joblib",
        metrics=directory / "metrics.json",
    )


def master_model_paths(version: int) -> ModelArtifactPaths:
    """Return deterministic artifact paths for one master-model version."""
    if version < 1:
        raise ValueError("model version must be at least 1")
    directory = Path(MASTER_MODELS_DIR) / f"v{version:04d}"
    return ModelArtifactPaths(
        model=directory / "ppo_model.zip",
        scaler=directory / "standard_scaler.joblib",
        metrics=directory / "metrics.json",
    )
