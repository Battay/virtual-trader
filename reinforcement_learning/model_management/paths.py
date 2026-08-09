"""Central paths for immutable PPO models and their companion artifacts."""

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.src.config import (
    MASTER_MODELS_DIR,
    SAVED_MODELS_DIR,
    SYMBOL_MODELS_DIR,
)
from feature_engineering.storage import safe_path_component


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Reserved paths for one immutable future model version."""

    model: Path
    scaler: Path
    metrics: Path


@dataclass(frozen=True)
class PPOArtifactBundlePaths:
    """Every path in one versioned, self-contained PPO artifact bundle."""

    directory: Path
    model: Path
    metadata: Path
    ppo_config: Path
    validation_metrics: Path
    baseline_metrics: Path
    rl_contract: Path
    scaler: Path
    scaler_metadata: Path
    registry_record: Path
    manifest: Path


def ppo_bundle_paths(
    model_scope: str,
    symbol: str,
    version: int,
    saved_models_dir: Path = SAVED_MODELS_DIR,
) -> PPOArtifactBundlePaths:
    """Return canonical paths for one symbol or master PPO bundle.

    ``saved_models_dir`` is injectable so persistence tests and developer smoke
    runs never need to touch the production model directories.
    """
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("model version must be a positive integer")
    root = Path(saved_models_dir)
    if model_scope == "symbol":
        symbol_text = str(symbol).strip()
        if not symbol_text:
            raise ValueError("symbol scope requires a symbol")
        directory = (
            root
            / "symbol_models"
            / safe_path_component(symbol_text)
            / f"v{version:04d}"
        )
    elif model_scope == "master":
        if str(symbol).strip():
            raise ValueError("master scope cannot specify a symbol")
        directory = root / "master_models" / f"v{version:04d}"
    else:
        raise ValueError(f"Unsupported model scope: {model_scope}")
    return PPOArtifactBundlePaths(
        directory=directory,
        model=directory / "ppo_model.zip",
        metadata=directory / "model_metadata.json",
        ppo_config=directory / "ppo_config.json",
        validation_metrics=directory / "validation_metrics.json",
        baseline_metrics=directory / "baseline_comparison_metrics.json",
        rl_contract=directory / "rl_contract.json",
        scaler=directory / "rl_observation_scaler.joblib",
        scaler_metadata=directory / "rl_observation_scaler.json",
        registry_record=directory / "registry_record.json",
        manifest=directory / "artifact_manifest.json",
    )


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
