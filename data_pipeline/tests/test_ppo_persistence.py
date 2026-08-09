"""Deterministic offline tests for atomic, versioned PPO persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.config import MODEL_REGISTRY_PATH, SAVED_MODELS_DIR
from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.evaluation.comparison import (
    CANDIDATE_CRITERIA_VERSION,
    compare_candidate_on_validation,
)
from reinforcement_learning.evaluation.results import CandidateValidationDecision
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.model_management import persistence as persistence_module
from reinforcement_learning.model_management.paths import ppo_bundle_paths
from reinforcement_learning.model_management.persistence import (
    ArtifactCompatibilityError,
    ModelVersionError,
    PPOPersistenceError,
    RegistryCommitPendingError,
    audit_registry_filesystem_consistency,
    check_promotion_eligibility,
    load_persisted_ppo,
    next_persisted_model_version,
    persist_developer_smoke_bundle,
    persist_ppo_candidate,
    reconcile_persisted_bundle,
    verify_artifact_bundle,
)
from reinforcement_learning.model_management.registry import load_model_registry
from reinforcement_learning.training.config import PPO_CONFIG_VERSION, PPOConfig
from reinforcement_learning.training.ppo_trainer import train_single_symbol


def _processed(rows: int = 48, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    wave = 1.75 * np.sin(index / 2.5)
    opens = 100.0 + 0.3 * index + wave
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2024-01-01", periods=rows),
        "open": opens,
        "high": opens + 2.0,
        "low": opens - 2.0,
        "close": opens + np.cos(index / 3.0),
        "volume": 10_000.0 + 100.0 * index,
    }
    for position, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = np.sin(index / (position + 2)) + position * 0.01
    return pd.DataFrame(data)


def _tiny_config(seed: int = 37) -> PPOConfig:
    return replace(
        PPOConfig(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        total_timesteps=16,
        seed=seed,
    )


def _file_tree(path: Path) -> dict[str, tuple[int, str]]:
    root = Path(path)
    if not root.exists():
        return {}
    return {
        str(item.relative_to(root)): (item.stat().st_size, sha256_file(item))
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


@dataclass(frozen=True)
class _CandidateFixture:
    splits_dir: Path
    source_directory: Path
    training: object
    validation_pass: object
    validation_fail: object


@pytest.fixture(scope="module")
def candidate_fixture(tmp_path_factory: pytest.TempPathFactory) -> _CandidateFixture:
    splits_dir = tmp_path_factory.mktemp("ppo-persistence-contract")
    source = _processed()
    split = chronological_split(source, scope="symbol")
    source_directory = splits_dir / "symbols" / "MCB"
    persist_split_artifacts(split, source_directory)
    training = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert training.succeeded and training.model is not None
    comparison = compare_candidate_on_validation(
        training.model,
        "MCB",
        trainer_result=training,
        deterministic_seed=37,
        random_seed=91,
        splits_dir=splits_dir,
    )
    pass_decision = CandidateValidationDecision(
        status="validation_pass",
        passed=True,
        reasons=("Deterministic persistence fixture passed validation.",),
        criteria_version=CANDIDATE_CRITERIA_VERSION,
        thresholds=dict(comparison.candidate_decision.thresholds),
    )
    fail_decision = CandidateValidationDecision(
        status="validation_fail",
        passed=False,
        reasons=("Deterministic persistence fixture failed validation.",),
        criteria_version=CANDIDATE_CRITERIA_VERSION,
        thresholds=dict(comparison.candidate_decision.thresholds),
    )
    return _CandidateFixture(
        splits_dir=splits_dir,
        source_directory=source_directory,
        training=training,
        validation_pass=replace(comparison, candidate_decision=pass_decision),
        validation_fail=replace(comparison, candidate_decision=fail_decision),
    )


def _persist_candidate(
    fixture: _CandidateFixture,
    root: Path,
):
    return persist_ppo_candidate(
        fixture.training,
        fixture.validation_pass,
        symbol="MCB",
        notes="Offline deterministic candidate.",
        registry_path=root / "model_registry.csv",
        saved_models_dir=root / "saved_models",
        splits_dir=fixture.splits_dir,
        created_at=datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc),
    )


def _rewrite_manifest_entry(bundle: Path, artifact_name: str) -> None:
    manifest_path = bundle / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = bundle / artifact_name
    manifest["files"][artifact_name] = {
        "sha256": sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_candidate_bundle_is_complete_attributable_and_validation_only(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_before = _file_tree(candidate_fixture.source_directory)

    def forbidden_partition_loader(*args, **kwargs):
        raise AssertionError("Persistence must not load TRAIN, VALIDATION, or TEST data")

    monkeypatch.setattr(
        "reinforcement_learning.data_contract.load_rl_partition",
        forbidden_partition_loader,
    )
    persisted = _persist_candidate(candidate_fixture, tmp_path)

    assert persisted.model_id == "ppo-symbol-MCB-v0001"
    assert persisted.model_version == 1
    assert persisted.model_status == "candidate"
    assert persisted.validation_status == "validation_pass"
    assert persisted.promotion_status == "candidate"
    paths = ppo_bundle_paths(
        "symbol", "MCB", 1, saved_models_dir=tmp_path / "saved_models"
    )
    assert persisted.bundle_path == paths.directory
    expected_names = {
        "ppo_model.zip",
        "model_metadata.json",
        "ppo_config.json",
        "validation_metrics.json",
        "baseline_comparison_metrics.json",
        "rl_contract.json",
        "rl_observation_scaler.joblib",
        "rl_observation_scaler.json",
        "registry_record.json",
        "artifact_manifest.json",
    }
    assert {path.name for path in paths.directory.iterdir()} == expected_names
    verification = verify_artifact_bundle(paths.directory, load_model=True)
    metadata = verification.metadata
    assert metadata["identity"] == {
        "model_id": persisted.model_id,
        "model_scope": "symbol",
        "symbol": "MCB",
        "algorithm": "PPO",
        "model_version": 1,
    }
    assert metadata["versions"]["ppo_config"] == PPO_CONFIG_VERSION
    assert metadata["versions"]["environment"] == ENVIRONMENT_VERSION
    assert metadata["versions"]["rl_contract"] == "rl_partition_v1"
    assert metadata["observation"]["features"] == list(
        DEFAULT_OBSERVATION_FEATURES
    )
    assert metadata["observation"]["shape"] == [17]
    assert metadata["training"]["seed"] == candidate_fixture.training.seed
    assert metadata["training"]["rows"] == candidate_fixture.training.training_rows
    assert metadata["validation"]["rows"] == (
        candidate_fixture.validation_pass.validation_rows
    )
    assert metadata["validation"]["partition"] == "validation"
    assert metadata["test_evaluation_performed"] is False
    test_metadata = metadata["data_availability"]["partitions"]["test"]
    assert set(test_metadata) == {"rows", "start", "end", "evaluation_status"}
    assert test_metadata["evaluation_status"] == "sealed_not_evaluated"
    assert verification.validation_metrics["evaluation_partition"] == "validation"
    assert verification.validation_metrics["test_evaluation_performed"] is False
    assert verification.baseline_metrics["evaluation_partition"] == "validation"
    assert "test_metrics" not in json.dumps(verification.validation_metrics).lower()
    assert "test_metrics" not in json.dumps(verification.baseline_metrics).lower()
    assert "artifact_manifest.json" not in verification.manifest["files"]
    assert sha256_file(paths.rl_contract) == (
        candidate_fixture.training.source_rl_contract_sha256
    )
    assert sha256_file(paths.scaler) == (
        candidate_fixture.training.source_observation_scaler_sha256
    )
    assert sha256_file(paths.scaler_metadata) == (
        candidate_fixture.training.source_observation_scaler_metadata_sha256
    )
    registry = load_model_registry(tmp_path / "model_registry.csv")
    assert len(registry) == 1
    assert registry.iloc[0]["model_status"] == "candidate"
    assert registry.iloc[0]["validation_status"] == "validation_pass"
    assert registry.iloc[0]["promotion_status"] == "candidate"
    assert registry.iloc[0]["manifest_sha256"] == persisted.manifest_sha256
    assert _file_tree(candidate_fixture.source_directory) == source_before


def test_exact_load_by_id_and_version_reloads_and_predicts(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
) -> None:
    persisted = _persist_candidate(candidate_fixture, tmp_path)
    by_id = load_persisted_ppo(
        model_id=persisted.model_id,
        registry_path=persisted.registry_path,
        saved_models_dir=tmp_path / "saved_models",
    )
    by_version = load_persisted_ppo(
        symbol="MCB",
        model_version=1,
        registry_path=persisted.registry_path,
        saved_models_dir=tmp_path / "saved_models",
    )
    observation = np.zeros((17,), dtype=np.float32)
    action_by_id, _ = by_id.model.predict(observation, deterministic=True)
    action_by_version, _ = by_version.model.predict(observation, deterministic=True)
    assert int(action_by_id) in {0, 1, 2}
    assert int(action_by_id) == int(action_by_version)
    assert by_id.model.num_timesteps == candidate_fixture.training.actual_timesteps
    assert by_id.model_id == by_version.model_id == persisted.model_id

    common = {
        "registry_path": persisted.registry_path,
        "saved_models_dir": tmp_path / "saved_models",
    }
    with pytest.raises(PPOPersistenceError, match="exactly one selection"):
        load_persisted_ppo(**common)
    with pytest.raises(PPOPersistenceError, match="explicit model_version"):
        load_persisted_ppo(symbol="MCB", **common)
    with pytest.raises(PPOPersistenceError, match="exactly one selection"):
        load_persisted_ppo(
            model_id=persisted.model_id,
            symbol="MCB",
            model_version=1,
            **common,
        )
    with pytest.raises(PPOPersistenceError, match="resolved 0 registry rows"):
        load_persisted_ppo(symbol="MCB", model_version=99, **common)


def test_versions_are_monotonic_collision_safe_and_never_overwrite(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
) -> None:
    first = _persist_candidate(candidate_fixture, tmp_path)
    first_before = _file_tree(first.bundle_path)
    second = _persist_candidate(candidate_fixture, tmp_path)
    assert (first.model_version, second.model_version) == (1, 2)
    assert first.model_id != second.model_id
    assert first.bundle_path != second.bundle_path
    assert _file_tree(first.bundle_path) == first_before
    assert next_persisted_model_version(
        registry_path=first.registry_path,
        saved_models_dir=tmp_path / "saved_models",
        symbol="MCB",
    ) == 3
    registry = load_model_registry(first.registry_path)
    audit = audit_registry_filesystem_consistency(
        registry=registry,
        model_scope="symbol",
        symbol="MCB",
        saved_models_dir=tmp_path / "saved_models",
    )
    assert audit.consistent
    assert audit.registry_versions == audit.filesystem_versions == (1, 2)

    reserved_root = tmp_path / "reserved-failed-version"
    reserved = _persist_candidate(candidate_fixture, reserved_root)
    reserved_registry = pd.read_csv(reserved.registry_path)
    reserved_registry.loc[0, "model_status"] = "failed"
    reserved_registry.to_csv(reserved.registry_path, index=False)
    shutil.rmtree(reserved.bundle_path)
    with pytest.raises(ModelVersionError, match="reconcile before allocating"):
        next_persisted_model_version(
            registry_path=reserved.registry_path,
            saved_models_dir=reserved_root / "saved_models",
            symbol="MCB",
        )

    malformed_root = tmp_path / "malformed"
    malformed = malformed_root / "saved_models/symbol_models/MCB/not-a-version"
    malformed.mkdir(parents=True)
    with pytest.raises(ModelVersionError, match="Malformed model version"):
        _persist_candidate(candidate_fixture, malformed_root)
    assert not (malformed_root / "model_registry.csv").exists()

    occupied_root = tmp_path / "occupied"
    occupied = occupied_root / "saved_models/symbol_models/MCB/v0001"
    occupied.mkdir(parents=True)
    sentinel = occupied / "do-not-overwrite.txt"
    sentinel.write_text("immutable", encoding="utf-8")
    with pytest.raises(ModelVersionError, match="reconcile before allocating"):
        _persist_candidate(candidate_fixture, occupied_root)
    assert sentinel.read_text(encoding="utf-8") == "immutable"


def test_eligibility_rejects_bad_runs_and_validation_fail_is_experiment_only(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    persisted = _persist_candidate(candidate_fixture, candidate_root)
    eligibility = check_promotion_eligibility(
        model_id=persisted.model_id,
        registry_path=persisted.registry_path,
        saved_models_dir=candidate_root / "saved_models",
    )
    assert eligibility.eligible
    assert eligibility.reasons == ("Candidate is eligible for explicit promotion.",)

    failed_training = replace(
        candidate_fixture.training,
        status="failed",
        model=None,
        error="fixture failure",
        message="Training failed.",
    )
    interrupted_training = replace(
        candidate_fixture.training,
        status="interrupted",
        model=None,
        error="KeyboardInterrupt",
        message="Training interrupted.",
    )
    for training in (failed_training, interrupted_training):
        rejected_root = tmp_path / training.status
        with pytest.raises(PPOPersistenceError, match="only completed"):
            persist_ppo_candidate(
                training,
                candidate_fixture.validation_pass,
                symbol="MCB",
                registry_path=rejected_root / "registry.csv",
                saved_models_dir=rejected_root / "models",
                splits_dir=candidate_fixture.splits_dir,
            )
        assert not (rejected_root / "models").exists()
        assert not (rejected_root / "registry.csv").exists()

    failed_root = tmp_path / "validation-fail"
    with pytest.raises(PPOPersistenceError, match="validation_fail"):
        persist_ppo_candidate(
            candidate_fixture.training,
            candidate_fixture.validation_fail,
            symbol="MCB",
            registry_path=failed_root / "registry.csv",
            saved_models_dir=failed_root / "models",
            splits_dir=candidate_fixture.splits_dir,
        )
    assert not (failed_root / "models").exists()
    assert not (failed_root / "registry.csv").exists()

    experiment_root = tmp_path / "experiment"
    experiment = persist_developer_smoke_bundle(
        candidate_fixture.training,
        candidate_fixture.validation_fail,
        symbol="MCB",
        registry_path=experiment_root / "registry.csv",
        saved_models_dir=experiment_root / "models",
        splits_dir=candidate_fixture.splits_dir,
    )
    assert experiment.model_status == "experiment"
    assert experiment.validation_status == "validation_fail"
    assert experiment.promotion_status == "not_eligible"
    not_eligible = check_promotion_eligibility(
        model_id=experiment.model_id,
        registry_path=experiment.registry_path,
        saved_models_dir=experiment_root / "models",
    )
    assert not not_eligible.eligible
    assert any("not candidate" in reason for reason in not_eligible.reasons)
    assert any("not validation_pass" in reason for reason in not_eligible.reasons)

    incompatible_root = tmp_path / "incompatible"
    incompatible = replace(candidate_fixture.validation_pass, feature_version="other")
    with pytest.raises(ArtifactCompatibilityError, match="feature versions differ"):
        persist_ppo_candidate(
            candidate_fixture.training,
            incompatible,
            symbol="MCB",
            registry_path=incompatible_root / "registry.csv",
            saved_models_dir=incompatible_root / "models",
            splits_dir=candidate_fixture.splits_dir,
        )


def test_model_save_failure_removes_staging_and_never_registers(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "save-failure"

    def failing_save(destination, *args, **kwargs):
        del args, kwargs
        Path(destination).write_bytes(b"partial PPO artifact")
        raise OSError("injected model save failure")

    monkeypatch.setattr(candidate_fixture.training.model, "save", failing_save)
    with pytest.raises(OSError, match="injected model save failure"):
        _persist_candidate(candidate_fixture, root)
    identity = root / "saved_models/symbol_models/MCB"
    assert not identity.exists() or list(identity.iterdir()) == []
    assert not (root / "model_registry.csv").exists()
    assert not list(root.rglob("*.staging-*"))


def test_registry_failure_leaves_verifiable_orphan_then_reconciles_once(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry-failure"
    original_append = persistence_module._append_model_version_unlocked

    def failing_append(*args, **kwargs):
        raise OSError("injected registry replacement failure")

    monkeypatch.setattr(
        persistence_module, "_append_model_version_unlocked", failing_append
    )
    with pytest.raises(RegistryCommitPendingError) as raised:
        _persist_candidate(candidate_fixture, root)
    orphan = raised.value.bundle_path
    assert orphan.is_dir()
    verification = verify_artifact_bundle(orphan, load_model=True)
    assert verification.metadata["identity"]["model_id"] == raised.value.model_id
    assert not (root / "model_registry.csv").exists()

    monkeypatch.setattr(
        persistence_module, "_append_model_version_unlocked", original_append
    )
    reconciled = reconcile_persisted_bundle(
        orphan,
        registry_path=root / "model_registry.csv",
        saved_models_dir=root / "saved_models",
    )
    assert reconciled.reconciled
    assert reconciled.model_id == raised.value.model_id
    assert len(load_model_registry(root / "model_registry.csv")) == 1
    second = reconcile_persisted_bundle(
        orphan,
        registry_path=root / "model_registry.csv",
        saved_models_dir=root / "saved_models",
    )
    assert second.reconciled
    assert len(load_model_registry(root / "model_registry.csv")) == 1
    loaded = load_persisted_ppo(
        model_id=reconciled.model_id,
        registry_path=root / "model_registry.csv",
        saved_models_dir=root / "saved_models",
    )
    assert loaded.model_id == reconciled.model_id


def test_registry_failure_preserves_previous_version_and_registry_bytes(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registry-failure-after-existing"
    first = _persist_candidate(candidate_fixture, root)
    first_before = _file_tree(first.bundle_path)
    registry_before = first.registry_path.read_bytes()

    def failing_append(*args, **kwargs):
        raise OSError("injected second-version registry failure")

    monkeypatch.setattr(
        persistence_module, "_append_model_version_unlocked", failing_append
    )
    with pytest.raises(RegistryCommitPendingError) as raised:
        _persist_candidate(candidate_fixture, root)
    assert raised.value.model_id == "ppo-symbol-MCB-v0002"
    assert first.registry_path.read_bytes() == registry_before
    assert _file_tree(first.bundle_path) == first_before
    assert verify_artifact_bundle(raised.value.bundle_path, load_model=True)


@pytest.mark.parametrize("damage", ("missing_manifest", "metadata_hash", "model_hash"))
def test_missing_or_tampered_artifacts_fail_before_ppo_load(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    root = tmp_path / damage
    persisted = _persist_candidate(candidate_fixture, root)
    if damage == "missing_manifest":
        (persisted.bundle_path / "artifact_manifest.json").unlink()
    elif damage == "metadata_hash":
        with (persisted.bundle_path / "model_metadata.json").open("ab") as handle:
            handle.write(b"\n")
    else:
        with (persisted.bundle_path / "ppo_model.zip").open("ab") as handle:
            handle.write(b"corruption")
    load_calls: list[object] = []

    def forbidden_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("PPO.load must follow manifest verification")

    monkeypatch.setattr(persistence_module.PPO, "load", forbidden_load)
    with pytest.raises(ArtifactCompatibilityError):
        load_persisted_ppo(
            model_id=persisted.model_id,
            registry_path=persisted.registry_path,
            saved_models_dir=root / "saved_models",
        )
    assert load_calls == []


def test_hash_valid_but_incompatible_observation_order_is_rejected(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
) -> None:
    persisted = _persist_candidate(candidate_fixture, tmp_path)
    metadata_path = persisted.bundle_path / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observation"]["features"] = list(
        reversed(metadata["observation"]["features"])
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_manifest_entry(persisted.bundle_path, metadata_path.name)
    with pytest.raises(ArtifactCompatibilityError, match="feature order"):
        verify_artifact_bundle(persisted.bundle_path)


def test_temp_smoke_cleanup_and_production_locations_remain_byte_identical(
    candidate_fixture: _CandidateFixture,
    tmp_path: Path,
) -> None:
    registry_before = Path(MODEL_REGISTRY_PATH).read_bytes()
    production_models_before = _file_tree(Path(SAVED_MODELS_DIR))
    source_before = _file_tree(candidate_fixture.source_directory)
    with pytest.raises(PPOPersistenceError, match="outside the project"):
        persist_developer_smoke_bundle(
            candidate_fixture.training,
            candidate_fixture.validation_fail,
            symbol="MCB",
            registry_path=Path(MODEL_REGISTRY_PATH),
            saved_models_dir=tmp_path / "otherwise-safe-models",
            splits_dir=candidate_fixture.splits_dir,
        )
    with pytest.raises(PPOPersistenceError, match="outside the project"):
        persist_developer_smoke_bundle(
            candidate_fixture.training,
            candidate_fixture.validation_fail,
            symbol="MCB",
            registry_path=tmp_path / "otherwise-safe-registry.csv",
            saved_models_dir=Path(SAVED_MODELS_DIR),
            splits_dir=candidate_fixture.splits_dir,
        )
    temporary_parent = tmp_path / "temporary-smoke"
    temporary_parent.mkdir()
    with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary_name:
        temporary_root = Path(temporary_name)
        experiment = persist_developer_smoke_bundle(
            candidate_fixture.training,
            candidate_fixture.validation_fail,
            symbol="MCB",
            registry_path=temporary_root / "registry.csv",
            saved_models_dir=temporary_root / "models",
            splits_dir=candidate_fixture.splits_dir,
        )
        loaded = load_persisted_ppo(
            model_id=experiment.model_id,
            registry_path=experiment.registry_path,
            saved_models_dir=temporary_root / "models",
        )
        action, _ = loaded.model.predict(
            np.zeros((17,), dtype=np.float32), deterministic=True
        )
        assert int(action) in {0, 1, 2}
        assert experiment.bundle_path.is_dir()
    assert not Path(temporary_name).exists()
    assert Path(MODEL_REGISTRY_PATH).read_bytes() == registry_before
    assert _file_tree(Path(SAVED_MODELS_DIR)) == production_models_before
    assert _file_tree(candidate_fixture.source_directory) == source_before
