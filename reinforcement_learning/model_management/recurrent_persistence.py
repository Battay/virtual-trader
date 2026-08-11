"""Temporary-only RecurrentPPO save/reload compatibility proof."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
    SAVED_MODELS_DIR,
)
from feature_engineering.storage import atomic_write_json
from reinforcement_learning.recurrent_data_contract import load_recurrent_partition
from reinforcement_learning.training.recurrent_results import (
    RecurrentPPOTrainingResult,
    TemporaryRecurrentPersistenceResult,
)
from reinforcement_learning.training.recurrent_trainer import count_model_parameters


class RecurrentPersistenceError(RuntimeError):
    """Raised when a temporary recurrent round trip is unsafe or incompatible."""


def _bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _architecture(model: RecurrentPPO) -> dict[str, object]:
    policy = model.policy
    return {
        "policy_class": type(policy).__name__,
        "lstm_hidden_size": int(policy.lstm_actor.hidden_size),
        "n_lstm_layers": int(policy.lstm_actor.num_layers),
        "shared_lstm": bool(policy.shared_lstm),
        "enable_critic_lstm": bool(policy.enable_critic_lstm),
    }


def verify_temporary_recurrent_round_trip(
    training: RecurrentPPOTrainingResult,
    *,
    temporary_root: Path,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> TemporaryRecurrentPersistenceResult:
    """Save/reload one model outside the project and never register it."""

    if not training.succeeded or not isinstance(training.model, RecurrentPPO):
        raise RecurrentPersistenceError("completed RecurrentPPO training is required")
    root = Path(temporary_root).expanduser().resolve(strict=False)
    project = Path(PROJECT_ROOT).resolve(strict=False)
    production_models = Path(SAVED_MODELS_DIR).resolve(strict=False)
    if (
        root == project
        or project in root.parents
        or root == production_models
        or production_models in root.parents
    ):
        raise RecurrentPersistenceError(
            "temporary recurrent persistence must be outside the project"
        )
    root.mkdir(parents=True, exist_ok=False)
    registry_before = _bytes(Path(registry_path))
    model_path = root / "recurrent_model.zip"
    metadata_path = root / "recurrent_metadata.json"
    architecture = _architecture(training.model)
    metadata = {
        "artifact_purpose": "temporary_developer_round_trip",
        "production_candidate": False,
        "registered": False,
        "symbol": training.symbol,
        "algorithm": training.algorithm,
        "policy": training.policy,
        "trainer_version": training.trainer_version,
        "recurrent_contract_version": training.recurrent_contract_version,
        "environment_version": training.environment_version,
        "feature_version": training.feature_version,
        "seed": training.seed,
        "actual_timesteps": training.actual_timesteps,
        "architecture": architecture,
        "config": dict(training.config),
        "test_evaluation_performed": False,
    }
    training.model.save(model_path)
    if not model_path.is_file():
        raise RecurrentPersistenceError("RecurrentPPO.save did not create a zip")
    atomic_write_json(metadata, metadata_path)
    loaded = RecurrentPPO.load(model_path, device=training.resolved_device or "cpu")
    loaded_architecture = _architecture(loaded)
    architecture_match = loaded_architecture == architecture
    if not architecture_match or type(loaded.policy) is not type(training.model.policy):
        raise RecurrentPersistenceError("reloaded recurrent architecture differs")

    validation = load_recurrent_partition(
        training.symbol,
        "validation",
        splits_dir=Path(splits_dir),
    )
    from reinforcement_learning.environments import SingleSymbolTradingEnv

    environment = SingleSymbolTradingEnv(validation.data)
    try:
        observation_array, _ = environment.reset(seed=training.seed)
    finally:
        environment.close()
    episode_start = np.asarray([True], dtype=bool)
    action_a, state_a = training.model.predict(
        observation_array,
        state=None,
        episode_start=episode_start,
        deterministic=True,
    )
    action_b, state_b = loaded.predict(
        observation_array,
        state=None,
        episode_start=episode_start,
        deterministic=True,
    )
    action_match = bool(np.array_equal(action_a, action_b))
    state_match = bool(
        state_a is not None
        and state_b is not None
        and all(np.array_equal(left, right) for left, right in zip(state_a, state_b))
    )
    if not action_match or not state_match:
        raise RecurrentPersistenceError(
            "reloaded deterministic recurrent prediction differs"
        )
    registry_touched = _bytes(Path(registry_path)) != registry_before
    if registry_touched:
        raise RecurrentPersistenceError("temporary round trip changed model registry")
    return TemporaryRecurrentPersistenceResult(
        model_path=model_path,
        metadata_path=metadata_path,
        policy_class=type(loaded.policy).__name__,
        saved_parameter_count=count_model_parameters(training.model),
        loaded_parameter_count=count_model_parameters(loaded),
        deterministic_action_match=action_match,
        recurrent_state_match=state_match,
        architecture_match=architecture_match,
        registry_touched=registry_touched,
    )
