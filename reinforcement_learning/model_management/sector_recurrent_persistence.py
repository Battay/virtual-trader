"""Temporary-only persistence proof for a sector RecurrentPPO foundation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sb3_contrib import RecurrentPPO

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
    SAVED_MODELS_DIR,
)
from feature_engineering.storage import atomic_write_json
from reinforcement_learning.environments import SingleSymbolTradingEnv
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    LoadedRecurrentPartition,
    load_recurrent_partition,
)
from reinforcement_learning.training.sector_recurrent_results import (
    SectorRecurrentTrainingResult,
    SectorValidationResult,
    TemporarySectorPersistenceResult,
)


class SectorRecurrentPersistenceError(RuntimeError):
    """Raised when temporary sector persistence would be misleading."""


def _bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _tree_snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_temporary_sector_round_trip(
    training: SectorRecurrentTrainingResult,
    validation: SectorValidationResult,
    *,
    temporary_root: Path,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    registry_path: Path = MODEL_REGISTRY_PATH,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
) -> TemporarySectorPersistenceResult:
    """Save/reload outside the project; never register or promote the model."""

    if not training.succeeded or not isinstance(training.model, RecurrentPPO):
        raise SectorRecurrentPersistenceError("completed sector training is required")
    if validation.universe_hash != training.sector_universe_hash:
        raise SectorRecurrentPersistenceError("training/validation universe differs")
    if not validation.model_parameters_unchanged:
        raise SectorRecurrentPersistenceError("validation changed the sector model")
    root = Path(temporary_root).expanduser().resolve(strict=False)
    project = Path(PROJECT_ROOT).resolve(strict=False)
    protected = tuple(
        Path(path).resolve(strict=False)
        for path in (SAVED_MODELS_DIR, MODELS_DATA_DIR)
    )
    if root == project or project in root.parents or any(
        root == path or path in root.parents for path in protected
    ):
        raise SectorRecurrentPersistenceError(
            "temporary sector persistence must be outside the project"
        )
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SectorRecurrentPersistenceError("temporary sector root must be empty")
    registry_before = _bytes(Path(registry_path))
    model_trees_before = {
        str(path): _tree_snapshot(Path(path))
        for path in (SAVED_MODELS_DIR, MODELS_DATA_DIR)
    }
    model_path = root / "commercial_banks_sector_foundation_recurrent_ppo.zip"
    metadata_path = root / "research_metadata.json"
    metadata: dict[str, object] = {
        "artifact_schema_version": "temporary_sector_recurrent_bundle_v1",
        "artifact_purpose": "temporary_research_round_trip",
        "production_candidate": False,
        "registered": False,
        "promoted": False,
        "model_scope": "sector",
        "sector_id": training.sector_id,
        "sector_name": training.sector_name,
        "trainer_version": training.trainer_version,
        "algorithm": training.algorithm,
        "policy": training.policy,
        "taxonomy_version": training.taxonomy_version,
        "sector_universe_hash": training.sector_universe_hash,
        "constituent_symbols": list(training.constituent_symbols),
        "constituent_count": len(training.constituent_symbols),
        "sampling_strategy": training.sampling_strategy,
        "normalization_scope": training.normalization_scope,
        "recurrent_contract_version": training.recurrent_contract_version,
        "environment_version": training.environment_version,
        "feature_version": training.feature_version,
        "seed": training.seed,
        "training": training.to_dict(),
        "validation_summary": {
            "evaluation_partition": "validation",
            "independent_symbol_capital": validation.independent_symbol_capital,
            "symbols_evaluated": len(validation.symbol_results),
            "failures": dict(validation.failures),
            "aggregate_metrics": dict(validation.aggregate_metrics),
            "collapse_diagnostics": dict(validation.collapse_diagnostics),
            "test_evaluated": False,
        },
        "test_evaluation_performed": False,
        "model_registry_v3_future_fields": {
            "model_scope": "sector",
            "sector": training.sector_name,
            "taxonomy_version": training.taxonomy_version,
            "universe_hash": training.sector_universe_hash,
            "constituent_count": len(training.constituent_symbols),
            "manifest_path": None,
            "sampling_strategy": training.sampling_strategy,
            "normalization_scope": training.normalization_scope,
            "recurrent_contract_version": training.recurrent_contract_version,
            "recurrent_trainer_version": training.trainer_version,
            "parent_model_id": None,
            "target_symbol": None,
            "target_excluded": False,
            "transfer_lineage": None,
        },
    }
    training.model.save(model_path)
    if not model_path.is_file():
        raise SectorRecurrentPersistenceError("sector model save did not create a zip")
    atomic_write_json(metadata, metadata_path)
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    integrity = loaded_metadata == metadata
    if not integrity or loaded_metadata.get("test_evaluation_performed") is not False:
        raise SectorRecurrentPersistenceError("sector metadata round trip differs")
    loaded_model = RecurrentPPO.load(
        model_path, device=training.resolved_device or "cpu"
    )
    symbol = training.constituent_symbols[0]
    train = partition_loader(symbol, "train", splits_dir=Path(splits_dir))
    environment = SingleSymbolTradingEnv(train.data)
    try:
        observation, _ = environment.reset(seed=training.seed)
    finally:
        environment.close()
    episode_start = np.asarray([True], dtype=bool)
    action_a, state_a = training.model.predict(
        observation, state=None, episode_start=episode_start, deterministic=True
    )
    action_b, state_b = loaded_model.predict(
        observation, state=None, episode_start=episode_start, deterministic=True
    )
    action_match = bool(np.array_equal(action_a, action_b))
    state_match = bool(
        state_a is not None
        and state_b is not None
        and all(np.array_equal(left, right) for left, right in zip(state_a, state_b))
    )
    if not action_match or not state_match:
        raise SectorRecurrentPersistenceError("reloaded recurrent prediction differs")
    registry_touched = _bytes(Path(registry_path)) != registry_before
    model_trees_after = {
        str(path): _tree_snapshot(Path(path))
        for path in (SAVED_MODELS_DIR, MODELS_DATA_DIR)
    }
    if registry_touched or model_trees_after != model_trees_before:
        raise SectorRecurrentPersistenceError("production model state changed")
    return TemporarySectorPersistenceResult(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
        model_sha256=sha256_file(model_path),
        deterministic_action_match=action_match,
        recurrent_state_match=state_match,
        metadata_integrity_verified=integrity,
        registry_touched=registry_touched,
    )


__all__ = (
    "SectorRecurrentPersistenceError",
    "verify_temporary_sector_round_trip",
)
