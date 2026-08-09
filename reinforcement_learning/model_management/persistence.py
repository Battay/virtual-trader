"""Atomic, versioned PPO candidate persistence and exact artifact loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile

import joblib
import pandas as pd
from stable_baselines3 import PPO

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
    SAVED_MODELS_DIR,
)
from feature_engineering.storage import safe_path_component
from reinforcement_learning.data_contract import (
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
    RL_PARTITION_SCHEMA_VERSION,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.evaluation.comparison import (
    CANDIDATE_CRITERIA_VERSION,
)
from reinforcement_learning.evaluation.ppo_evaluator import policy_parameter_hash
from reinforcement_learning.evaluation.results import ValidationComparisonResult
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.training.config import PPO_CONFIG_VERSION
from reinforcement_learning.training.results import PPOTrainingResult

from .paths import PPOArtifactBundlePaths, ppo_bundle_paths
from .registry import (
    MODEL_REGISTRY_COLUMNS,
    MODEL_REGISTRY_SCHEMA_VERSION,
    ModelRegistryError,
    _append_model_version_unlocked,
    load_model_registry,
    model_registry_lock,
    validate_model_record,
)


PPO_ARTIFACT_BUNDLE_VERSION = "ppo_artifact_bundle_v1"
ALGORITHM = "PPO"
MODEL_SCOPE = "symbol"
MANIFEST_HASH_ALGORITHM = "sha256"
VERSION_DIRECTORY_PATTERN = re.compile(r"v(?P<version>[0-9]{4,})")


class PPOPersistenceError(RuntimeError):
    """Raised when a PPO bundle cannot be persisted or trusted safely."""


class ModelVersionError(PPOPersistenceError):
    """Raised when registry/filesystem versions are malformed or inconsistent."""


class ArtifactCompatibilityError(PPOPersistenceError):
    """Raised when training, validation, or persisted metadata are incompatible."""


class RegistryCommitPendingError(PPOPersistenceError):
    """A complete bundle exists but its atomic registry append did not finish."""

    def __init__(
        self,
        *,
        model_id: str,
        bundle_path: Path,
        registry_path: Path,
        cause: BaseException,
    ) -> None:
        self.model_id = model_id
        self.bundle_path = Path(bundle_path)
        self.registry_path = Path(registry_path)
        self.cause = cause
        super().__init__(
            "Bundle committed but registry update is pending for "
            f"{model_id}; reconcile {self.bundle_path}: {type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True)
class RegistryFilesystemAudit:
    """Consistency of persisted versions for one exact model identity."""

    model_scope: str
    symbol: str
    registry_versions: tuple[int, ...]
    filesystem_versions: tuple[int, ...]
    filesystem_only_versions: tuple[int, ...]
    registry_only_versions: tuple[int, ...]

    @property
    def consistent(self) -> bool:
        return not self.filesystem_only_versions and not self.registry_only_versions


@dataclass(frozen=True)
class BundleVerification:
    """Validated JSON payloads and paths for one immutable bundle."""

    paths: PPOArtifactBundlePaths
    metadata: Mapping[str, object]
    ppo_config: Mapping[str, object]
    validation_metrics: Mapping[str, object]
    baseline_metrics: Mapping[str, object]
    contract: Mapping[str, object]
    scaler_metadata: Mapping[str, object]
    planned_registry_record: Mapping[str, object]
    manifest: Mapping[str, object]
    manifest_sha256: str


@dataclass(frozen=True)
class PersistedPPOBundle:
    """Successful filesystem and registry result for one immutable version."""

    model_id: str
    model_version: int
    symbol: str
    model_status: str
    validation_status: str
    promotion_status: str
    bundle_path: Path
    registry_path: Path
    manifest_sha256: str
    registry_record: Mapping[str, object]
    reconciled: bool = False


@dataclass(frozen=True)
class LoadedPPOBundle:
    """Explicitly selected and fully verified PPO artifact bundle."""

    model: PPO
    model_id: str
    model_version: int
    symbol: str
    registry_record: Mapping[str, object]
    verification: BundleVerification


@dataclass(frozen=True)
class PromotionEligibility:
    """Read-only eligibility result; it never mutates registry state."""

    model_id: str
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _SourceArtifacts:
    contract_path: Path
    scaler_path: Path
    scaler_metadata_path: Path
    contract: Mapping[str, object]
    scaler_metadata: Mapping[str, object]


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactCompatibilityError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError(f"{label} must contain a JSON object")
    return payload


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _write_json_fsynced(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with destination.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory_best_effort(path: Path) -> None:
    try:
        descriptor = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _paths_from_directory(directory: Path) -> PPOArtifactBundlePaths:
    root = Path(directory)
    return PPOArtifactBundlePaths(
        directory=root,
        model=root / "ppo_model.zip",
        metadata=root / "model_metadata.json",
        ppo_config=root / "ppo_config.json",
        validation_metrics=root / "validation_metrics.json",
        baseline_metrics=root / "baseline_comparison_metrics.json",
        rl_contract=root / "rl_contract.json",
        scaler=root / "rl_observation_scaler.joblib",
        scaler_metadata=root / "rl_observation_scaler.json",
        registry_record=root / "registry_record.json",
        manifest=root / "artifact_manifest.json",
    )


def _bundle_payload_paths(paths: PPOArtifactBundlePaths) -> tuple[Path, ...]:
    return (
        paths.model,
        paths.metadata,
        paths.ppo_config,
        paths.validation_metrics,
        paths.baseline_metrics,
        paths.rl_contract,
        paths.scaler,
        paths.scaler_metadata,
        paths.registry_record,
    )


def _normalized_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise PPOPersistenceError("creation timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc).isoformat()


def _git_provenance() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "", None
    return commit, bool(status.strip())


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for label, distribution in (
        ("stable_baselines3", "stable-baselines3"),
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
    ):
        try:
            versions[label] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def _require_canonical_symbol(symbol: str) -> str:
    symbol_text = str(symbol).strip()
    if not symbol_text:
        raise PPOPersistenceError("symbol is required")
    if safe_path_component(symbol_text) != symbol_text:
        raise PPOPersistenceError(
            "symbol must already be a collision-safe filesystem component"
        )
    return symbol_text


def _validate_candidate_inputs(
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    *,
    symbol: str,
    allow_validation_failure: bool,
) -> None:
    if not isinstance(training, PPOTrainingResult):
        raise PPOPersistenceError("training_result must be a PPOTrainingResult")
    if not training.succeeded or training.status != "completed":
        raise PPOPersistenceError("only completed PPO training results can persist")
    if training.model is None:
        raise PPOPersistenceError("completed training result has no in-memory model")
    if training.symbol != symbol or comparison.symbol != symbol:
        raise ArtifactCompatibilityError("training/validation symbol mismatch")
    if training.algorithm != ALGORITHM:
        raise ArtifactCompatibilityError("only PPO artifacts are supported")
    if comparison.status != "completed" or comparison.evaluation_partition != "validation":
        raise ArtifactCompatibilityError("comparison must be completed on validation")
    if not comparison.ppo_model_unchanged:
        raise ArtifactCompatibilityError("validation changed PPO model state")
    if int(training.model.num_timesteps) != training.actual_timesteps:
        raise ArtifactCompatibilityError("trainer timestep metadata is stale")
    if comparison.ppo_model_timesteps_after != training.actual_timesteps:
        raise ArtifactCompatibilityError("validation and training timesteps differ")
    current_policy_hash = policy_parameter_hash(training.model)
    if current_policy_hash != comparison.ppo_parameter_hash_after:
        raise ArtifactCompatibilityError("PPO policy changed after validation")
    version_pairs = (
        (training.environment_version, comparison.environment_version, "environment"),
        (training.rl_contract_version, comparison.rl_contract_version, "RL contract"),
        (training.feature_version, comparison.feature_version, "feature"),
    )
    for training_value, validation_value, label in version_pairs:
        if training_value != validation_value:
            raise ArtifactCompatibilityError(f"{label} versions differ")
    if training.ppo_config_version != PPO_CONFIG_VERSION:
        raise ArtifactCompatibilityError("PPO configuration version is incompatible")
    if tuple(training.observation_features) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("training observation feature order is incompatible")
    if tuple(comparison.observation_features) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("validation observation feature order is incompatible")
    expected_shape = (
        len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
    )
    if tuple(training.observation_shape or ()) != expected_shape:
        raise ArtifactCompatibilityError("training observation shape is incompatible")
    if training.model.observation_space.shape != expected_shape:
        raise ArtifactCompatibilityError("PPO observation space is incompatible")
    if getattr(training.model.action_space, "n", None) != 3:
        raise ArtifactCompatibilityError("PPO action space is incompatible")
    decision = comparison.candidate_decision
    if decision.criteria_version != CANDIDATE_CRITERIA_VERSION:
        raise ArtifactCompatibilityError("candidate criteria version is incompatible")
    allowed = {"validation_pass"}
    if allow_validation_failure:
        allowed.add("validation_fail")
    if decision.status not in allowed:
        raise PPOPersistenceError(
            f"validation status {decision.status!r} is not persistable"
        )
    if decision.passed != (decision.status == "validation_pass"):
        raise ArtifactCompatibilityError("candidate decision is internally inconsistent")


def _validate_partition_metadata(
    metadata: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    try:
        rows_value = metadata["rows"]
        start = str(metadata["start"])
        end = str(metadata["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"RL contract {label} partition metadata is invalid"
        ) from exc
    if isinstance(rows_value, bool) or not isinstance(rows_value, Integral):
        raise ArtifactCompatibilityError(
            f"RL contract {label} partition rows must be a positive integer"
        )
    rows = int(rows_value)
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            f"RL contract {label} partition dates must use YYYY-MM-DD"
        ) from exc
    if rows < 1 or start_date > end_date:
        raise ArtifactCompatibilityError(
            f"RL contract {label} partition metadata is invalid"
        )
    return {"rows": rows, "start": start, "end": end}


def _load_and_validate_source_artifacts(
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    *,
    symbol: str,
    splits_dir: Path,
) -> _SourceArtifacts:
    source_directory = (
        Path(splits_dir) / "symbols" / safe_path_component(symbol)
    ).resolve()
    contract_path = (source_directory / RL_CONTRACT_FILENAME).resolve()
    scaler_path = (source_directory / RL_OBSERVATION_SCALER_FILENAME).resolve()
    scaler_metadata_path = scaler_path.with_suffix(".json")
    path_pairs = (
        (training.source_rl_contract_path, comparison.source_rl_contract_path, contract_path),
        (
            training.source_observation_scaler_path,
            comparison.source_observation_scaler_path,
            scaler_path,
        ),
        (
            training.source_observation_scaler_metadata_path,
            comparison.source_observation_scaler_metadata_path,
            scaler_metadata_path,
        ),
    )
    for training_path, validation_path, expected in path_pairs:
        if not training_path or not validation_path:
            raise ArtifactCompatibilityError("RL source provenance path is missing")
        if Path(training_path).resolve() != expected or Path(validation_path).resolve() != expected:
            raise ArtifactCompatibilityError("RL source provenance path is incompatible")
    current_hashes = (
        sha256_file(contract_path),
        sha256_file(scaler_path),
        sha256_file(scaler_metadata_path),
    )
    expected_hash_pairs = (
        (
            training.source_rl_contract_sha256,
            comparison.source_rl_contract_sha256,
        ),
        (
            training.source_observation_scaler_sha256,
            comparison.source_observation_scaler_sha256,
        ),
        (
            training.source_observation_scaler_metadata_sha256,
            comparison.source_observation_scaler_metadata_sha256,
        ),
    )
    for current, (training_hash, validation_hash) in zip(
        current_hashes, expected_hash_pairs, strict=True
    ):
        if not training_hash or current != training_hash or current != validation_hash:
            raise ArtifactCompatibilityError(
                "RL source artifact changed between training, validation, and persistence"
            )
    contract = _read_json_object(contract_path, label="RL contract")
    scaler_metadata = _read_json_object(
        scaler_metadata_path, label="RL observation scaler metadata"
    )
    if contract.get("artifact_schema_version") != RL_PARTITION_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("RL contract version is incompatible")
    if contract.get("environment_version") != ENVIRONMENT_VERSION:
        raise ArtifactCompatibilityError("RL environment version is incompatible")
    if contract.get("feature_version") != training.feature_version:
        raise ArtifactCompatibilityError("RL feature version is incompatible")
    if tuple(contract.get("observation_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("RL observation feature order is incompatible")
    if contract.get("scaler_fit_partition") != "train":
        raise ArtifactCompatibilityError("RL scaler was not fitted on train")
    if tuple(scaler_metadata.get("scaled_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("RL scaler feature order is incompatible")
    partitions = contract.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ArtifactCompatibilityError("RL contract partition metadata is missing")
    train = _validate_partition_metadata(dict(partitions.get("train", {})), label="train")
    validation = _validate_partition_metadata(
        dict(partitions.get("validation", {})), label="validation"
    )
    _validate_partition_metadata(dict(partitions.get("test", {})), label="test")
    if (
        train["rows"] != training.training_rows
        or train["start"] != training.training_start
        or train["end"] != training.training_end
    ):
        raise ArtifactCompatibilityError("training result differs from RL contract")
    if (
        validation["rows"] != comparison.validation_rows
        or validation["start"] != comparison.validation_start
        or validation["end"] != comparison.validation_end
    ):
        raise ArtifactCompatibilityError("validation result differs from RL contract")
    try:
        scaler_training_rows = int(scaler_metadata.get("training_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("RL scaler training rows are invalid") from exc
    if scaler_training_rows != training.training_rows:
        raise ArtifactCompatibilityError("RL scaler training rows are stale")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        raise ArtifactCompatibilityError(f"RL observation scaler is unreadable: {exc}") from exc
    if int(getattr(scaler, "n_features_in_", -1)) != len(DEFAULT_OBSERVATION_FEATURES):
        raise ArtifactCompatibilityError("RL observation scaler width is incompatible")
    return _SourceArtifacts(
        contract_path=contract_path,
        scaler_path=scaler_path,
        scaler_metadata_path=scaler_metadata_path,
        contract=contract,
        scaler_metadata=scaler_metadata,
    )


def _identity_directory(
    *,
    model_scope: str,
    symbol: str,
    saved_models_dir: Path,
) -> Path:
    return ppo_bundle_paths(
        model_scope,
        symbol,
        1,
        saved_models_dir=Path(saved_models_dir),
    ).directory.parent


def _filesystem_versions(identity_directory: Path) -> tuple[int, ...]:
    if not identity_directory.exists():
        return ()
    if not identity_directory.is_dir() or identity_directory.is_symlink():
        raise ModelVersionError(f"Model identity path is unsafe: {identity_directory}")
    versions: list[int] = []
    for child in sorted(identity_directory.iterdir(), key=lambda path: path.name):
        if child.name.startswith("."):
            continue
        match = VERSION_DIRECTORY_PATTERN.fullmatch(child.name)
        if match is None or not child.is_dir() or child.is_symlink():
            raise ModelVersionError(
                f"Malformed model version entry blocks allocation: {child}"
            )
        version = int(match.group("version"))
        if version < 1 or child.name != f"v{version:04d}":
            raise ModelVersionError(f"Malformed model version directory: {child}")
        versions.append(version)
    if len(versions) != len(set(versions)):
        raise ModelVersionError("Duplicate filesystem model versions detected")
    return tuple(sorted(versions))


def audit_registry_filesystem_consistency(
    *,
    registry: pd.DataFrame,
    model_scope: str,
    symbol: str,
    saved_models_dir: Path,
) -> RegistryFilesystemAudit:
    """Audit exact persisted versions without mutating either side."""
    symbol_text = str(symbol).strip() if model_scope == "symbol" else ""
    matches = registry.loc[registry["model_scope"].astype(str).eq(model_scope)]
    matches = matches.loc[matches["symbol"].astype("string").fillna("").eq(symbol_text)]
    registry_versions = tuple(
        sorted(int(value) for value in matches["model_version"].tolist())
    )
    filesystem_versions = _filesystem_versions(
        _identity_directory(
            model_scope=model_scope,
            symbol=symbol_text,
            saved_models_dir=saved_models_dir,
        )
    )
    for _, row in matches.iterrows():
        expected = ppo_bundle_paths(
            model_scope,
            symbol_text,
            int(row["model_version"]),
            saved_models_dir=Path(saved_models_dir),
        )
        try:
            _validate_registry_bundle_paths(row.to_dict(), expected)
        except ArtifactCompatibilityError as exc:
            raise ModelVersionError(
                f"Registry artifact paths disagree with version {row['model_version']}"
            ) from exc
    registry_set = set(registry_versions)
    filesystem_set = set(filesystem_versions)
    return RegistryFilesystemAudit(
        model_scope=model_scope,
        symbol=symbol_text,
        registry_versions=registry_versions,
        filesystem_versions=filesystem_versions,
        filesystem_only_versions=tuple(sorted(filesystem_set - registry_set)),
        registry_only_versions=tuple(sorted(registry_set - filesystem_set)),
    )


def _next_persisted_model_version_unlocked(
    *,
    registry_path: Path,
    saved_models_dir: Path,
    model_scope: str,
    symbol: str,
) -> int:
    registry = load_model_registry(registry_path)
    audit = audit_registry_filesystem_consistency(
        registry=registry,
        model_scope=model_scope,
        symbol=symbol,
        saved_models_dir=saved_models_dir,
    )
    if not audit.consistent:
        raise ModelVersionError(
            "Registry/filesystem model versions disagree; reconcile before allocating: "
            f"filesystem_only={audit.filesystem_only_versions}, "
            f"registry_only={audit.registry_only_versions}"
        )
    versions = (*audit.registry_versions, *audit.filesystem_versions)
    return max(versions, default=0) + 1


def next_persisted_model_version(
    *,
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
    model_scope: str = MODEL_SCOPE,
    symbol: str,
) -> int:
    """Allocate the next collision-safe version under the registry lock."""
    symbol_text = _require_canonical_symbol(symbol) if model_scope == "symbol" else ""
    with model_registry_lock(registry_path):
        return _next_persisted_model_version_unlocked(
            registry_path=Path(registry_path),
            saved_models_dir=Path(saved_models_dir),
            model_scope=model_scope,
            symbol=symbol_text,
        )


def _build_registry_record(
    *,
    model_id: str,
    version: int,
    symbol: str,
    paths: PPOArtifactBundlePaths,
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    source: _SourceArtifacts,
    created_at: str,
    model_status: str,
    promotion_status: str,
    notes: str,
    source_git_commit: str,
    source_worktree_dirty: bool | None,
) -> dict[str, object]:
    partitions = dict(source.contract["partitions"])
    train = _validate_partition_metadata(dict(partitions["train"]), label="train")
    validation = _validate_partition_metadata(
        dict(partitions["validation"]), label="validation"
    )
    test = _validate_partition_metadata(dict(partitions["test"]), label="test")
    record: dict[str, object] = {column: "" for column in MODEL_REGISTRY_COLUMNS}
    record.update(
        {
            "model_id": model_id,
            "model_scope": MODEL_SCOPE,
            "symbol": symbol,
            "algorithm": ALGORITHM,
            "model_version": version,
            "model_status": model_status,
            "training_status": "completed",
            "feature_version": training.feature_version,
            "environment_version": training.environment_version,
            "created_at": created_at,
            "last_trained_at": training.completed_at,
            "complete_available_history_start": train["start"],
            "complete_available_history_end": test["end"],
            "training_data_start": train["start"],
            "training_data_end": train["end"],
            "validation_data_start": validation["start"],
            "validation_data_end": validation["end"],
            "test_data_start": test["start"],
            "test_data_end": test["end"],
            "training_rows": train["rows"],
            "validation_rows": validation["rows"],
            "test_rows": test["rows"],
            "dataset_latest_date": test["end"],
            "new_data_days": 0,
            "needs_retraining": False,
            "model_path": str(paths.model.resolve()),
            "scaler_path": str(paths.scaler.resolve()),
            "metrics_path": str(paths.validation_metrics.resolve()),
            "training_duration_seconds": training.duration_seconds,
            "random_seed": training.seed,
            "notes": notes,
            "registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
            "artifact_schema_version": PPO_ARTIFACT_BUNDLE_VERSION,
            "rl_contract_version": training.rl_contract_version,
            "ppo_config_version": training.ppo_config_version,
            "validation_status": comparison.candidate_decision.status,
            "promotion_status": promotion_status,
            "rl_contract_path": str(paths.rl_contract.resolve()),
            "scaler_metadata_path": str(paths.scaler_metadata.resolve()),
            "metadata_path": str(paths.metadata.resolve()),
            "config_path": str(paths.ppo_config.resolve()),
            "validation_metrics_path": str(paths.validation_metrics.resolve()),
            "baseline_metrics_path": str(paths.baseline_metrics.resolve()),
            "registry_record_path": str(paths.registry_record.resolve()),
            "manifest_path": str(paths.manifest.resolve()),
            "manifest_sha256": "",
            "observation_shape": json.dumps(list(training.observation_shape or ())),
            "observation_features": json.dumps(list(training.observation_features)),
            "source_git_commit": source_git_commit,
            "source_worktree_dirty": (
                "" if source_worktree_dirty is None else source_worktree_dirty
            ),
        }
    )
    return record


def _build_metadata(
    *,
    model_id: str,
    version: int,
    symbol: str,
    paths: PPOArtifactBundlePaths,
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    source: _SourceArtifacts,
    created_at: str,
    artifact_purpose: str,
    model_status: str,
    promotion_status: str,
    source_git_commit: str,
    source_worktree_dirty: bool | None,
) -> dict[str, object]:
    partitions = {
        name: _validate_partition_metadata(dict(metadata), label=name)
        for name, metadata in dict(source.contract["partitions"]).items()
        if name in {"train", "validation", "test"}
    }
    partitions["test"]["evaluation_status"] = "sealed_not_evaluated"
    complete_rows = sum(int(item["rows"]) for item in partitions.values())
    return {
        "bundle_schema_version": PPO_ARTIFACT_BUNDLE_VERSION,
        "identity": {
            "model_id": model_id,
            "model_scope": MODEL_SCOPE,
            "symbol": symbol,
            "algorithm": ALGORITHM,
            "model_version": version,
        },
        "lifecycle": {
            "artifact_purpose": artifact_purpose,
            "model_status": model_status,
            "training_status": training.status,
            "validation_status": comparison.candidate_decision.status,
            "promotion_status": promotion_status,
        },
        "created_at": created_at,
        "last_trained_at": training.completed_at,
        "source": {
            "git_commit": source_git_commit or None,
            "worktree_dirty": source_worktree_dirty,
        },
        "dependencies": _package_versions(),
        "versions": {
            "ppo_config": training.ppo_config_version,
            "environment": training.environment_version,
            "rl_contract": training.rl_contract_version,
            "feature": training.feature_version,
            "candidate_criteria": comparison.candidate_decision.criteria_version,
        },
        "training": {
            "start": training.training_start,
            "end": training.training_end,
            "rows": training.training_rows,
            "requested_timesteps": training.requested_timesteps,
            "actual_timesteps": training.actual_timesteps,
            "duration_seconds": training.duration_seconds,
            "seed": training.seed,
            "device": training.device,
            "policy_parameter_sha256": comparison.ppo_parameter_hash_after,
        },
        "validation": {
            "partition": "validation",
            "start": comparison.validation_start,
            "end": comparison.validation_end,
            "rows": comparison.validation_rows,
            "duration_seconds": comparison.evaluation_duration_seconds,
            "deterministic_seed": comparison.deterministic_seed,
            "random_seed": comparison.random_seed,
            "model_parameter_sha256_before": comparison.ppo_parameter_hash_before,
            "model_parameter_sha256_after": comparison.ppo_parameter_hash_after,
            "model_timesteps_before": comparison.ppo_model_timesteps_before,
            "model_timesteps_after": comparison.ppo_model_timesteps_after,
            "decision": comparison.candidate_decision.to_dict(),
        },
        "data_availability": {
            "complete_start": partitions["train"]["start"],
            "complete_end": partitions["test"]["end"],
            "complete_rows": complete_rows,
            "dataset_latest_date": partitions["test"]["end"],
            "partitions": partitions,
        },
        "observation": {
            "features": list(training.observation_features),
            "dynamic_portfolio_features": list(DYNAMIC_PORTFOLIO_FEATURES),
            "shape": list(training.observation_shape or ()),
            "action_count": 3,
            "actions": {"0": "Hold", "1": "Buy", "2": "Sell"},
        },
        "environment_config": _json_ready(comparison.environment_config),
        "scaler": {
            "fit_partition": "train",
            "training_rows": training.training_rows,
            "source_path": str(source.scaler_path),
            "source_sha256": training.source_observation_scaler_sha256,
            "source_metadata_path": str(source.scaler_metadata_path),
            "source_metadata_sha256": (
                training.source_observation_scaler_metadata_sha256
            ),
            "bundled_path": paths.scaler.name,
            "bundled_metadata_path": paths.scaler_metadata.name,
        },
        "rl_contract": {
            "source_path": str(source.contract_path),
            "source_sha256": training.source_rl_contract_sha256,
            "bundled_path": paths.rl_contract.name,
        },
        "artifacts": {
            "model": paths.model.name,
            "metadata": paths.metadata.name,
            "ppo_config": paths.ppo_config.name,
            "validation_metrics": paths.validation_metrics.name,
            "baseline_metrics": paths.baseline_metrics.name,
            "rl_contract": paths.rl_contract.name,
            "scaler": paths.scaler.name,
            "scaler_metadata": paths.scaler_metadata.name,
            "registry_record": paths.registry_record.name,
            "manifest": paths.manifest.name,
        },
        "test_evaluation_performed": False,
    }


def _write_staged_bundle(
    *,
    stage_paths: PPOArtifactBundlePaths,
    final_paths: PPOArtifactBundlePaths,
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    source: _SourceArtifacts,
    metadata: Mapping[str, object],
    planned_record: Mapping[str, object],
    model_id: str,
) -> str:
    if training.model is None:
        raise PPOPersistenceError("training model is unavailable")
    training.model.save(str(stage_paths.model))
    if not stage_paths.model.is_file():
        raise PPOPersistenceError("Stable-Baselines3 did not create the model zip")
    _fsync_file(stage_paths.model)
    shutil.copyfile(source.contract_path, stage_paths.rl_contract)
    shutil.copyfile(source.scaler_path, stage_paths.scaler)
    shutil.copyfile(source.scaler_metadata_path, stage_paths.scaler_metadata)
    for copied in (
        stage_paths.rl_contract,
        stage_paths.scaler,
        stage_paths.scaler_metadata,
    ):
        _fsync_file(copied)
    config_payload = {
        "bundle_schema_version": PPO_ARTIFACT_BUNDLE_VERSION,
        "model_id": model_id,
        "ppo_config_version": training.ppo_config_version,
        "configuration": dict(training.ppo_config),
    }
    validation_payload = {
        "bundle_schema_version": PPO_ARTIFACT_BUNDLE_VERSION,
        "model_id": model_id,
        "evaluation_partition": "validation",
        "validation_start": comparison.validation_start,
        "validation_end": comparison.validation_end,
        "validation_rows": comparison.validation_rows,
        "candidate_decision": comparison.candidate_decision.to_dict(),
        "ppo": comparison.ppo.to_dict(include_history=False),
        "test_evaluation_performed": False,
    }
    baseline_payload = comparison.to_dict(include_history=False)
    if baseline_payload.get("evaluation_partition") != "validation":
        raise ArtifactCompatibilityError("baseline comparison is not validation-only")
    _write_json_fsynced(stage_paths.metadata, metadata)
    _write_json_fsynced(stage_paths.ppo_config, config_payload)
    _write_json_fsynced(stage_paths.validation_metrics, validation_payload)
    _write_json_fsynced(stage_paths.baseline_metrics, baseline_payload)
    _write_json_fsynced(stage_paths.registry_record, planned_record)
    entries = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(_bundle_payload_paths(stage_paths), key=lambda item: item.name)
    }
    manifest = {
        "bundle_schema_version": PPO_ARTIFACT_BUNDLE_VERSION,
        "model_id": model_id,
        "hash_algorithm": MANIFEST_HASH_ALGORITHM,
        "files": entries,
    }
    _write_json_fsynced(stage_paths.manifest, manifest)
    _fsync_directory_best_effort(stage_paths.directory)
    verification = verify_artifact_bundle(stage_paths.directory, load_model=True)
    if verification.metadata.get("identity", {}).get("model_id") != model_id:
        raise ArtifactCompatibilityError("staged model identity validation failed")
    expected_final = str(final_paths.model.resolve())
    if str(planned_record.get("model_path")) != expected_final:
        raise ArtifactCompatibilityError("planned registry path is inconsistent")
    return verification.manifest_sha256


def _publish_bundle_atomically(
    *,
    final_paths: PPOArtifactBundlePaths,
    training: PPOTrainingResult,
    comparison: ValidationComparisonResult,
    source: _SourceArtifacts,
    metadata: Mapping[str, object],
    planned_record: Mapping[str, object],
    model_id: str,
) -> str:
    final_paths.directory.parent.mkdir(parents=True, exist_ok=True)
    if final_paths.directory.exists():
        raise ModelVersionError(
            f"Model version directory already exists: {final_paths.directory}"
        )
    stage_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{final_paths.directory.name}.staging-",
            dir=final_paths.directory.parent,
        )
    )
    stage_paths = _paths_from_directory(stage_directory)
    published = False
    try:
        manifest_sha256 = _write_staged_bundle(
            stage_paths=stage_paths,
            final_paths=final_paths,
            training=training,
            comparison=comparison,
            source=source,
            metadata=metadata,
            planned_record=planned_record,
            model_id=model_id,
        )
        if final_paths.directory.exists():
            raise ModelVersionError(
                f"Model version collision detected: {final_paths.directory}"
            )
        os.rename(stage_directory, final_paths.directory)
        published = True
        _fsync_directory_best_effort(final_paths.directory.parent)
        return manifest_sha256
    finally:
        if not published and stage_directory.exists():
            shutil.rmtree(stage_directory)


def _record_values_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    integer_columns = {
        "model_version",
        "training_rows",
        "validation_rows",
        "test_rows",
        "new_data_days",
        "random_seed",
    }
    float_columns = {"training_duration_seconds"}
    boolean_columns = {"needs_retraining", "source_worktree_dirty"}

    def empty(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def boolean(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1"}

    for column in MODEL_REGISTRY_COLUMNS:
        left_value = left.get(column, "")
        right_value = right.get(column, "")
        if empty(left_value) and empty(right_value):
            continue
        if empty(left_value) != empty(right_value):
            return False
        if column in integer_columns:
            try:
                if int(left_value) != int(right_value):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
            continue
        if column in float_columns:
            try:
                if not math.isclose(
                    float(left_value),
                    float(right_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
            continue
        if column in boolean_columns:
            if boolean(left_value) != boolean(right_value):
                return False
            continue
        if str(left_value) != str(right_value):
            return False
    return True


def _validate_registry_bundle_paths(
    record: Mapping[str, object],
    paths: PPOArtifactBundlePaths,
) -> None:
    expected = {
        "model_path": paths.model,
        "scaler_path": paths.scaler,
        "metrics_path": paths.validation_metrics,
        "rl_contract_path": paths.rl_contract,
        "scaler_metadata_path": paths.scaler_metadata,
        "metadata_path": paths.metadata,
        "config_path": paths.ppo_config,
        "validation_metrics_path": paths.validation_metrics,
        "baseline_metrics_path": paths.baseline_metrics,
        "registry_record_path": paths.registry_record,
        "manifest_path": paths.manifest,
    }
    for field, expected_path in expected.items():
        try:
            actual_path = Path(str(record[field])).resolve()
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"Registry artifact path is invalid: {field}"
            ) from exc
        if actual_path != expected_path.resolve():
            raise ArtifactCompatibilityError(
                f"Registry artifact path differs from bundle identity: {field}"
            )


def _persist(
    training_result: PPOTrainingResult,
    validation_result: ValidationComparisonResult,
    *,
    symbol: str,
    notes: str,
    registry_path: Path,
    saved_models_dir: Path,
    splits_dir: Path,
    created_at: datetime | None,
    artifact_purpose: str,
    allow_validation_failure: bool,
) -> PersistedPPOBundle:
    symbol_text = _require_canonical_symbol(symbol)
    notes_text = str(notes).strip()
    if len(notes_text) > 1_000:
        raise PPOPersistenceError("notes cannot exceed 1,000 characters")
    _validate_candidate_inputs(
        training_result,
        validation_result,
        symbol=symbol_text,
        allow_validation_failure=allow_validation_failure,
    )
    source = _load_and_validate_source_artifacts(
        training_result,
        validation_result,
        symbol=symbol_text,
        splits_dir=Path(splits_dir),
    )
    created_timestamp = _normalized_timestamp(created_at)
    git_commit, worktree_dirty = _git_provenance()
    registry_destination = Path(registry_path)
    model_root = Path(saved_models_dir)
    with model_registry_lock(registry_destination):
        version = _next_persisted_model_version_unlocked(
            registry_path=registry_destination,
            saved_models_dir=model_root,
            model_scope=MODEL_SCOPE,
            symbol=symbol_text,
        )
        model_id = f"ppo-symbol-{symbol_text}-v{version:04d}"
        paths = ppo_bundle_paths(
            MODEL_SCOPE,
            symbol_text,
            version,
            saved_models_dir=model_root,
        )
        if artifact_purpose == "production_candidate":
            model_status = "candidate"
            promotion_status = "candidate"
        else:
            model_status = "experiment"
            promotion_status = "not_eligible"
        planned_record = _build_registry_record(
            model_id=model_id,
            version=version,
            symbol=symbol_text,
            paths=paths,
            training=training_result,
            comparison=validation_result,
            source=source,
            created_at=created_timestamp,
            model_status=model_status,
            promotion_status=promotion_status,
            notes=notes_text,
            source_git_commit=git_commit,
            source_worktree_dirty=worktree_dirty,
        )
        planned_record = validate_model_record(planned_record)
        metadata = _build_metadata(
            model_id=model_id,
            version=version,
            symbol=symbol_text,
            paths=paths,
            training=training_result,
            comparison=validation_result,
            source=source,
            created_at=created_timestamp,
            artifact_purpose=artifact_purpose,
            model_status=model_status,
            promotion_status=promotion_status,
            source_git_commit=git_commit,
            source_worktree_dirty=worktree_dirty,
        )
        manifest_sha256 = _publish_bundle_atomically(
            final_paths=paths,
            training=training_result,
            comparison=validation_result,
            source=source,
            metadata=metadata,
            planned_record=planned_record,
            model_id=model_id,
        )
        final_record: dict[str, object] | None = None
        try:
            final_record = validate_model_record(
                dict(planned_record, manifest_sha256=manifest_sha256)
            )
            _append_model_version_unlocked(final_record, registry_destination)
        except Exception as exc:
            try:
                current = load_model_registry(registry_destination)
                matches = current.loc[current["model_id"].astype(str).eq(model_id)]
            except Exception:
                matches = pd.DataFrame()
            registry_has_exact_record = (
                final_record is not None
                and len(matches) == 1
                and _record_values_equal(final_record, matches.iloc[0].to_dict())
            )
            if not registry_has_exact_record:
                raise RegistryCommitPendingError(
                    model_id=model_id,
                    bundle_path=paths.directory,
                    registry_path=registry_destination,
                    cause=exc,
                ) from exc
        assert final_record is not None
    return PersistedPPOBundle(
        model_id=model_id,
        model_version=version,
        symbol=symbol_text,
        model_status=model_status,
        validation_status=validation_result.candidate_decision.status,
        promotion_status=promotion_status,
        bundle_path=paths.directory,
        registry_path=registry_destination,
        manifest_sha256=manifest_sha256,
        registry_record=final_record,
    )


def persist_ppo_candidate(
    training_result: PPOTrainingResult,
    validation_result: ValidationComparisonResult,
    *,
    symbol: str,
    notes: str = "",
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    created_at: datetime | None = None,
) -> PersistedPPOBundle:
    """Persist a validation-pass candidate; never promote it automatically."""
    return _persist(
        training_result,
        validation_result,
        symbol=symbol,
        notes=notes,
        registry_path=Path(registry_path),
        saved_models_dir=Path(saved_models_dir),
        splits_dir=Path(splits_dir),
        created_at=created_at,
        artifact_purpose="production_candidate",
        allow_validation_failure=False,
    )


def _ensure_nonproduction_smoke_targets(
    *,
    registry_path: Path,
    saved_models_dir: Path,
) -> None:
    registry = Path(registry_path).resolve()
    models = Path(saved_models_dir).resolve()
    production_registry = Path(MODEL_REGISTRY_PATH).resolve()
    production_models = Path(SAVED_MODELS_DIR).resolve()
    production_data_models = Path(MODELS_DATA_DIR).resolve()
    project_root = Path(PROJECT_ROOT).resolve()
    if (
        registry == production_registry
        or production_data_models in registry.parents
        or project_root in registry.parents
    ):
        raise PPOPersistenceError(
            "developer smoke registry must be outside the project production tree"
        )
    if (
        models == production_models
        or production_models in models.parents
        or models in production_models.parents
        or project_root in models.parents
        or models == project_root
    ):
        raise PPOPersistenceError(
            "developer smoke models must be outside the project production tree"
        )


def persist_developer_smoke_bundle(
    training_result: PPOTrainingResult,
    validation_result: ValidationComparisonResult,
    *,
    symbol: str,
    registry_path: Path,
    saved_models_dir: Path,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    notes: str = "Developer smoke artifact; never eligible for promotion.",
    created_at: datetime | None = None,
) -> PersistedPPOBundle:
    """Persist pass/fail validation only to explicit non-production temp roots."""
    _ensure_nonproduction_smoke_targets(
        registry_path=registry_path,
        saved_models_dir=saved_models_dir,
    )
    return _persist(
        training_result,
        validation_result,
        symbol=symbol,
        notes=notes,
        registry_path=Path(registry_path),
        saved_models_dir=Path(saved_models_dir),
        splits_dir=Path(splits_dir),
        created_at=created_at,
        artifact_purpose="developer_smoke_experiment",
        allow_validation_failure=True,
    )


def _validate_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ArtifactCompatibilityError(f"{label} is not a SHA-256 digest")
    return text


def _contains_sealed_test_metrics(value: object) -> bool:
    forbidden = {
        "test_metrics",
        "test_performance",
        "test_result",
        "test_evaluation_result",
        "test_returns",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in forbidden
            or _contains_sealed_test_metrics(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sealed_test_metrics(item) for item in value)
    return False


def verify_artifact_bundle(
    bundle_path: Path,
    *,
    load_model: bool = False,
) -> BundleVerification:
    """Verify exact files, hashes, metadata compatibility, and optional PPO load."""
    directory = Path(bundle_path)
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactCompatibilityError(f"Artifact bundle is missing or unsafe: {directory}")
    paths = _paths_from_directory(directory)
    expected_names = {path.name for path in _bundle_payload_paths(paths)} | {
        paths.manifest.name
    }
    actual_names: set[str] = set()
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ArtifactCompatibilityError(f"Unsafe artifact bundle entry: {entry}")
        actual_names.add(entry.name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ArtifactCompatibilityError(
            f"Artifact bundle file set differs; missing={missing}, extra={extra}"
        )
    manifest = _read_json_object(paths.manifest, label="artifact manifest")
    if manifest.get("bundle_schema_version") != PPO_ARTIFACT_BUNDLE_VERSION:
        raise ArtifactCompatibilityError("Artifact manifest schema is incompatible")
    if manifest.get("hash_algorithm") != MANIFEST_HASH_ALGORITHM:
        raise ArtifactCompatibilityError("Artifact manifest hash algorithm is incompatible")
    entries = manifest.get("files")
    if not isinstance(entries, Mapping):
        raise ArtifactCompatibilityError("Artifact manifest files are missing")
    payload_names = {path.name for path in _bundle_payload_paths(paths)}
    if set(entries) != payload_names:
        raise ArtifactCompatibilityError("Artifact manifest file set is incomplete")
    for path in _bundle_payload_paths(paths):
        item = entries.get(path.name)
        if not isinstance(item, Mapping):
            raise ArtifactCompatibilityError(f"Manifest entry is invalid: {path.name}")
        expected_hash = _validate_sha256(
            item.get("sha256"), label=f"Manifest hash for {path.name}"
        )
        try:
            expected_bytes = int(item.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"Manifest byte size is invalid: {path.name}"
            ) from exc
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
            raise ArtifactCompatibilityError(f"Artifact hash/size mismatch: {path.name}")
    metadata = _read_json_object(paths.metadata, label="model metadata")
    config = _read_json_object(paths.ppo_config, label="PPO configuration")
    validation_metrics = _read_json_object(
        paths.validation_metrics, label="validation metrics"
    )
    baseline_metrics = _read_json_object(
        paths.baseline_metrics, label="baseline comparison metrics"
    )
    contract = _read_json_object(paths.rl_contract, label="bundled RL contract")
    scaler_metadata = _read_json_object(
        paths.scaler_metadata, label="bundled scaler metadata"
    )
    planned_record = _read_json_object(
        paths.registry_record, label="planned registry record"
    )
    if metadata.get("bundle_schema_version") != PPO_ARTIFACT_BUNDLE_VERSION:
        raise ArtifactCompatibilityError("Model metadata schema is incompatible")
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise ArtifactCompatibilityError("Model identity metadata is missing")
    model_id = str(identity.get("model_id", ""))
    symbol = str(identity.get("symbol", "")).strip()
    try:
        version = int(identity.get("model_version"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactCompatibilityError("Model version metadata is invalid") from exc
    try:
        safe_symbol = safe_path_component(symbol)
    except ValueError as exc:
        raise ArtifactCompatibilityError("Model symbol identity is unsafe") from exc
    if version < 1 or safe_symbol != symbol:
        raise ArtifactCompatibilityError("Model symbol/version identity is unsafe")
    expected_model_id = f"ppo-{MODEL_SCOPE}-{symbol}-v{version:04d}"
    if model_id != expected_model_id:
        raise ArtifactCompatibilityError("Model identity metadata is inconsistent")
    if manifest.get("model_id") != model_id or config.get("model_id") != model_id:
        raise ArtifactCompatibilityError("Artifact model identities differ")
    if str(planned_record.get("model_id", "")) != model_id:
        raise ArtifactCompatibilityError("Planned registry identity differs")
    if identity.get("algorithm") != ALGORITHM or identity.get("model_scope") != MODEL_SCOPE:
        raise ArtifactCompatibilityError("Artifact algorithm or scope is incompatible")
    try:
        normalized_record = validate_model_record(planned_record)
    except (ModelRegistryError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"Planned registry record is invalid: {exc}"
        ) from exc
    planned_final_directory = Path(str(normalized_record["model_path"])).resolve().parent
    planned_paths = _paths_from_directory(planned_final_directory)
    _validate_registry_bundle_paths(normalized_record, planned_paths)
    if (
        planned_final_directory.name != f"v{version:04d}"
        or planned_final_directory.parent.name != symbol
        or planned_final_directory.parent.parent.name != "symbol_models"
    ):
        raise ArtifactCompatibilityError(
            "Planned registry paths do not match model identity"
        )
    identity_record_fields = {
        "model_id": model_id,
        "model_scope": MODEL_SCOPE,
        "symbol": symbol,
        "algorithm": ALGORITHM,
        "model_version": version,
    }
    for field, expected_value in identity_record_fields.items():
        if str(normalized_record[field]) != str(expected_value):
            raise ArtifactCompatibilityError(
                f"Planned registry identity differs: {field}"
            )
    if contract.get("artifact_schema_version") != RL_PARTITION_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("Bundled RL contract version is incompatible")
    if contract.get("environment_version") != ENVIRONMENT_VERSION:
        raise ArtifactCompatibilityError("Bundled environment version is incompatible")
    if tuple(contract.get("observation_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("Bundled observation feature order is incompatible")
    if contract.get("scaler_fit_partition") != "train":
        raise ArtifactCompatibilityError("Bundled scaler provenance is incompatible")
    if tuple(scaler_metadata.get("scaled_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("Bundled scaler feature order is incompatible")
    versions = metadata.get("versions")
    observation = metadata.get("observation")
    training_metadata = metadata.get("training")
    scaler_provenance = metadata.get("scaler")
    lifecycle = metadata.get("lifecycle")
    if not all(
        isinstance(item, Mapping)
        for item in (versions, observation, training_metadata, scaler_provenance, lifecycle)
    ):
        raise ArtifactCompatibilityError("Required model metadata sections are missing")
    if versions.get("ppo_config") != PPO_CONFIG_VERSION:
        raise ArtifactCompatibilityError("Persisted PPO configuration version is incompatible")
    if versions.get("environment") != ENVIRONMENT_VERSION:
        raise ArtifactCompatibilityError("Persisted environment version is incompatible")
    if versions.get("rl_contract") != RL_PARTITION_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("Persisted RL contract version is incompatible")
    if versions.get("feature") != contract.get("feature_version"):
        raise ArtifactCompatibilityError("Persisted feature versions differ")
    version_record_fields = {
        "feature_version": versions.get("feature"),
        "environment_version": versions.get("environment"),
        "rl_contract_version": versions.get("rl_contract"),
        "ppo_config_version": versions.get("ppo_config"),
    }
    for field, expected_value in version_record_fields.items():
        if str(normalized_record[field]) != str(expected_value):
            raise ArtifactCompatibilityError(
                f"Persisted registry/version metadata differs: {field}"
            )
    if tuple(observation.get("features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise ArtifactCompatibilityError("Persisted observation feature order is incompatible")
    expected_shape = [
        len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES)
    ]
    if observation.get("shape") != expected_shape or observation.get("action_count") != 3:
        raise ArtifactCompatibilityError("Persisted observation/action spaces are incompatible")
    if scaler_provenance.get("fit_partition") != "train":
        raise ArtifactCompatibilityError("Persisted scaler is not train-fitted")
    try:
        metadata_training_rows = int(training_metadata.get("rows", -1))
        scaler_source_rows = int(scaler_provenance.get("training_rows", -2))
        scaler_metadata_rows = int(scaler_metadata.get("training_rows", -3))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactCompatibilityError("Persisted scaler row provenance is invalid") from exc
    if not (
        metadata_training_rows == scaler_source_rows == scaler_metadata_rows
    ):
        raise ArtifactCompatibilityError("Persisted scaler row provenance differs")
    if sha256_file(paths.rl_contract) != _validate_sha256(
        metadata.get("rl_contract", {}).get("source_sha256")
        if isinstance(metadata.get("rl_contract"), Mapping)
        else "",
        label="RL contract source hash",
    ):
        raise ArtifactCompatibilityError("Bundled RL contract differs from its source")
    if sha256_file(paths.scaler) != _validate_sha256(
        scaler_provenance.get("source_sha256"),
        label="RL scaler source hash",
    ):
        raise ArtifactCompatibilityError("Bundled RL scaler differs from its source")
    if sha256_file(paths.scaler_metadata) != _validate_sha256(
        scaler_provenance.get("source_metadata_sha256"),
        label="RL scaler metadata source hash",
    ):
        raise ArtifactCompatibilityError(
            "Bundled RL scaler metadata differs from its source"
        )
    if config.get("ppo_config_version") != PPO_CONFIG_VERSION:
        raise ArtifactCompatibilityError("PPO configuration artifact is incompatible")
    configuration = config.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ArtifactCompatibilityError("PPO configuration payload is missing")
    if configuration.get("config_version") != PPO_CONFIG_VERSION:
        raise ArtifactCompatibilityError("PPO configuration payload is incompatible")
    if int(configuration.get("seed", -1)) != int(training_metadata.get("seed", -2)):
        raise ArtifactCompatibilityError("Persisted PPO seed metadata differs")
    try:
        if int(normalized_record["random_seed"]) != int(training_metadata.get("seed")):
            raise ArtifactCompatibilityError("Persisted registry seed metadata differs")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactCompatibilityError("Persisted registry seed is invalid") from exc
    availability = metadata.get("data_availability")
    contract_partitions = contract.get("partitions")
    if not isinstance(availability, Mapping) or not isinstance(
        contract_partitions, Mapping
    ):
        raise ArtifactCompatibilityError("Persisted partition metadata is missing")
    availability_partitions = availability.get("partitions")
    if not isinstance(availability_partitions, Mapping):
        raise ArtifactCompatibilityError("Persisted availability partitions are missing")
    normalized_partitions: dict[str, dict[str, object]] = {}
    for name in ("train", "validation", "test"):
        contract_partition = _validate_partition_metadata(
            dict(contract_partitions.get(name, {})), label=name
        )
        availability_partition = availability_partitions.get(name)
        if not isinstance(availability_partition, Mapping):
            raise ArtifactCompatibilityError(
                f"Persisted {name} availability metadata is missing"
            )
        normalized_availability = _validate_partition_metadata(
            dict(availability_partition), label=name
        )
        if normalized_availability != contract_partition:
            raise ArtifactCompatibilityError(
                f"Persisted {name} partition metadata differs"
            )
        normalized_partitions[name] = contract_partition
    if availability_partitions["test"].get("evaluation_status") != "sealed_not_evaluated":
        raise ArtifactCompatibilityError("Persisted TEST partition is not sealed")
    if (
        training_metadata.get("start") != normalized_partitions["train"]["start"]
        or training_metadata.get("end") != normalized_partitions["train"]["end"]
        or int(training_metadata.get("rows", -1))
        != normalized_partitions["train"]["rows"]
    ):
        raise ArtifactCompatibilityError("Persisted training partition metadata differs")
    if validation_metrics.get("evaluation_partition") != "validation":
        raise ArtifactCompatibilityError("Persisted validation metrics use wrong partition")
    if baseline_metrics.get("evaluation_partition") != "validation":
        raise ArtifactCompatibilityError("Persisted baseline metrics use wrong partition")
    if validation_metrics.get("model_id") != model_id:
        raise ArtifactCompatibilityError("Validation metrics model identity differs")
    for payload_name, payload in (
        ("validation metrics", validation_metrics),
        ("baseline metrics", baseline_metrics),
    ):
        if (
            str(payload.get("validation_start"))
            != str(normalized_partitions["validation"]["start"])
            or str(payload.get("validation_end"))
            != str(normalized_partitions["validation"]["end"])
            or int(payload.get("validation_rows", -1))
            != int(normalized_partitions["validation"]["rows"])
        ):
            raise ArtifactCompatibilityError(
                f"Persisted {payload_name} partition metadata differs"
            )
    if (
        baseline_metrics.get("symbol") != symbol
        or baseline_metrics.get("environment_version") != ENVIRONMENT_VERSION
        or baseline_metrics.get("rl_contract_version") != RL_PARTITION_SCHEMA_VERSION
        or baseline_metrics.get("feature_version") != versions.get("feature")
        or tuple(baseline_metrics.get("observation_features", ()))
        != DEFAULT_OBSERVATION_FEATURES
    ):
        raise ArtifactCompatibilityError("Persisted baseline provenance differs")
    validation_metadata = metadata.get("validation")
    if not isinstance(validation_metadata, Mapping):
        raise ArtifactCompatibilityError("Persisted validation metadata is missing")
    candidate_decision = validation_metadata.get("decision")
    if not isinstance(candidate_decision, Mapping):
        raise ArtifactCompatibilityError("Persisted candidate decision is missing")
    if (
        validation_metadata.get("start")
        != normalized_partitions["validation"]["start"]
        or validation_metadata.get("end")
        != normalized_partitions["validation"]["end"]
        or int(validation_metadata.get("rows", -1))
        != normalized_partitions["validation"]["rows"]
    ):
        raise ArtifactCompatibilityError(
            "Persisted validation partition metadata differs"
        )
    record_partition_fields = {
        "training_data_start": normalized_partitions["train"]["start"],
        "training_data_end": normalized_partitions["train"]["end"],
        "validation_data_start": normalized_partitions["validation"]["start"],
        "validation_data_end": normalized_partitions["validation"]["end"],
        "test_data_start": normalized_partitions["test"]["start"],
        "test_data_end": normalized_partitions["test"]["end"],
        "training_rows": normalized_partitions["train"]["rows"],
        "validation_rows": normalized_partitions["validation"]["rows"],
        "test_rows": normalized_partitions["test"]["rows"],
        "dataset_latest_date": normalized_partitions["test"]["end"],
    }
    for field, expected_value in record_partition_fields.items():
        if str(normalized_record[field]) != str(expected_value):
            raise ArtifactCompatibilityError(
                f"Persisted registry partition metadata differs: {field}"
            )
    if (
        validation_metrics.get("candidate_decision") != candidate_decision
        or baseline_metrics.get("candidate_decision") != candidate_decision
    ):
        raise ArtifactCompatibilityError("Persisted validation decisions differ")
    if (
        lifecycle.get("validation_status") != candidate_decision.get("status")
        or normalized_record["validation_status"] != candidate_decision.get("status")
        or lifecycle.get("model_status") != normalized_record["model_status"]
        or lifecycle.get("promotion_status") != normalized_record["promotion_status"]
    ):
        raise ArtifactCompatibilityError("Persisted lifecycle metadata differs")
    if candidate_decision.get("passed") is not (
        candidate_decision.get("status") == "validation_pass"
    ):
        raise ArtifactCompatibilityError("Persisted candidate decision is inconsistent")
    purpose = lifecycle.get("artifact_purpose")
    if purpose == "production_candidate":
        if (
            lifecycle.get("model_status") != "candidate"
            or lifecycle.get("promotion_status") != "candidate"
            or candidate_decision.get("status") != "validation_pass"
        ):
            raise ArtifactCompatibilityError(
                "Production candidate lifecycle metadata is inconsistent"
            )
    elif purpose == "developer_smoke_experiment":
        if (
            lifecycle.get("model_status") != "experiment"
            or lifecycle.get("promotion_status") != "not_eligible"
            or candidate_decision.get("status")
            not in {"validation_pass", "validation_fail"}
        ):
            raise ArtifactCompatibilityError(
                "Developer-smoke lifecycle metadata is inconsistent"
            )
    else:
        raise ArtifactCompatibilityError("Persisted artifact purpose is unsupported")
    if normalized_record.get("manifest_sha256") not in {"", None} and not pd.isna(
        normalized_record.get("manifest_sha256")
    ):
        raise ArtifactCompatibilityError(
            "Planned registry record must not contain a manifest self-reference"
        )
    if metadata.get("test_evaluation_performed") is not False:
        raise ArtifactCompatibilityError("Persisted metadata does not keep TEST sealed")
    if validation_metrics.get("test_evaluation_performed") is not False:
        raise ArtifactCompatibilityError("Validation metrics do not keep TEST sealed")
    if any(
        _contains_sealed_test_metrics(payload)
        for payload in (metadata, validation_metrics, baseline_metrics)
    ):
        raise ArtifactCompatibilityError("Persisted bundle contains sealed TEST metrics")
    try:
        scaler = joblib.load(paths.scaler)
    except Exception as exc:
        raise ArtifactCompatibilityError(f"Bundled scaler is unreadable: {exc}") from exc
    if int(getattr(scaler, "n_features_in_", -1)) != len(DEFAULT_OBSERVATION_FEATURES):
        raise ArtifactCompatibilityError("Bundled scaler width is incompatible")
    try:
        with zipfile.ZipFile(paths.model) as archive:
            if archive.testzip() is not None:
                raise ArtifactCompatibilityError("PPO model zip has a corrupt member")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactCompatibilityError(f"PPO model zip is corrupt: {exc}") from exc
    if load_model:
        try:
            model = PPO.load(str(paths.model), device="cpu")
        except Exception as exc:
            raise ArtifactCompatibilityError(f"PPO model cannot be loaded: {exc}") from exc
        expected_timesteps = int(training_metadata.get("actual_timesteps", -1))
        expected_policy_hash = str(training_metadata.get("policy_parameter_sha256", ""))
        if int(model.num_timesteps) != expected_timesteps:
            raise ArtifactCompatibilityError("Loaded PPO timestep metadata differs")
        if policy_parameter_hash(model) != expected_policy_hash:
            raise ArtifactCompatibilityError("Loaded PPO policy hash differs")
        if model.observation_space.shape != tuple(expected_shape):
            raise ArtifactCompatibilityError("Loaded PPO observation shape differs")
        if getattr(model.action_space, "n", None) != 3:
            raise ArtifactCompatibilityError("Loaded PPO action space differs")
        if int(getattr(model, "seed", -1)) != int(training_metadata.get("seed", -2)):
            raise ArtifactCompatibilityError("Loaded PPO seed differs from metadata")
    return BundleVerification(
        paths=paths,
        metadata=metadata,
        ppo_config=config,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        contract=contract,
        scaler_metadata=scaler_metadata,
        planned_registry_record=planned_record,
        manifest=manifest,
        manifest_sha256=sha256_file(paths.manifest),
    )


def _record_from_bundle_verification(
    verification: BundleVerification,
) -> dict[str, object]:
    record = dict(verification.planned_registry_record)
    record["manifest_sha256"] = verification.manifest_sha256
    return validate_model_record(record)


def reconcile_persisted_bundle(
    bundle_path: Path,
    *,
    registry_path: Path,
    saved_models_dir: Path,
) -> PersistedPPOBundle:
    """Register one explicit, complete filesystem-only bundle after validation."""
    verification = verify_artifact_bundle(bundle_path, load_model=True)
    record = _record_from_bundle_verification(verification)
    identity = dict(verification.metadata["identity"])
    symbol = str(identity["symbol"])
    version = int(identity["model_version"])
    expected = ppo_bundle_paths(
        MODEL_SCOPE,
        symbol,
        version,
        saved_models_dir=Path(saved_models_dir),
    )
    if verification.paths.directory.resolve() != expected.directory.resolve():
        raise ModelVersionError("Bundle path does not match its model identity")
    _validate_registry_bundle_paths(record, expected)
    with model_registry_lock(registry_path):
        registry = load_model_registry(registry_path)
        existing = registry.loc[
            registry["model_id"].astype(str).eq(str(record["model_id"]))
        ]
        if len(existing) == 1:
            if not _record_values_equal(record, existing.iloc[0].to_dict()):
                raise ModelVersionError("Registry has a conflicting model record")
        elif len(existing) > 1:
            raise ModelVersionError("Registry has ambiguous model records")
        else:
            conflicts = registry.loc[
                registry["model_scope"].astype(str).eq(MODEL_SCOPE)
                & registry["symbol"].astype("string").eq(symbol)
                & registry["model_version"].astype(int).eq(version)
            ]
            if not conflicts.empty:
                raise ModelVersionError("Registry version identity is already occupied")
            _append_model_version_unlocked(record, Path(registry_path))
    return PersistedPPOBundle(
        model_id=str(record["model_id"]),
        model_version=version,
        symbol=symbol,
        model_status=str(record["model_status"]),
        validation_status=str(record["validation_status"]),
        promotion_status=str(record["promotion_status"]),
        bundle_path=verification.paths.directory,
        registry_path=Path(registry_path),
        manifest_sha256=verification.manifest_sha256,
        registry_record=record,
        reconciled=True,
    )


def _resolve_registry_record(
    *,
    registry_path: Path,
    model_id: str | None,
    symbol: str | None,
    model_version: int | None,
) -> dict[str, object]:
    by_id = model_id is not None
    by_version = symbol is not None or model_version is not None
    if by_id == by_version:
        raise PPOPersistenceError(
            "Specify exactly one selection: model_id or symbol plus model_version"
        )
    registry = load_model_registry(registry_path)
    if by_id:
        matches = registry.loc[registry["model_id"].astype(str).eq(str(model_id))]
    else:
        if symbol is None or model_version is None:
            raise PPOPersistenceError("symbol selection requires an explicit model_version")
        if isinstance(model_version, bool) or not isinstance(model_version, int) or model_version < 1:
            raise PPOPersistenceError("model_version must be a positive integer")
        symbol_text = _require_canonical_symbol(symbol)
        matches = registry.loc[
            registry["model_scope"].astype(str).eq(MODEL_SCOPE)
            & registry["symbol"].astype("string").eq(symbol_text)
            & registry["model_version"].astype(int).eq(model_version)
        ]
    if len(matches) != 1:
        raise PPOPersistenceError(
            f"Explicit model selection resolved {len(matches)} registry rows"
        )
    return matches.iloc[0].to_dict()


def load_persisted_ppo(
    *,
    model_id: str | None = None,
    symbol: str | None = None,
    model_version: int | None = None,
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
) -> LoadedPPOBundle:
    """Load one exact registry version; implicit latest selection is forbidden."""
    record = _resolve_registry_record(
        registry_path=Path(registry_path),
        model_id=model_id,
        symbol=symbol,
        model_version=model_version,
    )
    symbol_text = str(record["symbol"])
    version = int(record["model_version"])
    expected = ppo_bundle_paths(
        str(record["model_scope"]),
        symbol_text,
        version,
        saved_models_dir=Path(saved_models_dir),
    )
    _validate_registry_bundle_paths(record, expected)
    verification = verify_artifact_bundle(expected.directory, load_model=False)
    bundle_record = _record_from_bundle_verification(verification)
    if not _record_values_equal(record, bundle_record):
        raise ArtifactCompatibilityError("Registry and artifact metadata differ")
    expected_manifest_hash = _validate_sha256(
        record.get("manifest_sha256"), label="Registry manifest hash"
    )
    if verification.manifest_sha256 != expected_manifest_hash:
        raise ArtifactCompatibilityError("Registry manifest hash differs")
    try:
        model = PPO.load(str(expected.model), device="cpu")
    except Exception as exc:
        raise ArtifactCompatibilityError(f"PPO model cannot be loaded: {exc}") from exc
    training_metadata = dict(verification.metadata["training"])
    observation = dict(verification.metadata["observation"])
    if int(model.num_timesteps) != int(training_metadata["actual_timesteps"]):
        raise ArtifactCompatibilityError("Loaded PPO timesteps differ from metadata")
    if policy_parameter_hash(model) != training_metadata["policy_parameter_sha256"]:
        raise ArtifactCompatibilityError("Loaded PPO policy hash differs from metadata")
    if model.observation_space.shape != tuple(observation["shape"]):
        raise ArtifactCompatibilityError("Loaded PPO observation shape differs")
    if getattr(model.action_space, "n", None) != int(observation["action_count"]):
        raise ArtifactCompatibilityError("Loaded PPO action space differs")
    if int(getattr(model, "seed", -1)) != int(training_metadata["seed"]):
        raise ArtifactCompatibilityError("Loaded PPO seed differs from metadata")
    return LoadedPPOBundle(
        model=model,
        model_id=str(record["model_id"]),
        model_version=version,
        symbol=symbol_text,
        registry_record=record,
        verification=verification,
    )


def check_promotion_eligibility(
    *,
    model_id: str,
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
) -> PromotionEligibility:
    """Validate promotion prerequisites without changing any status or file."""
    try:
        loaded = load_persisted_ppo(
            model_id=model_id,
            registry_path=registry_path,
            saved_models_dir=saved_models_dir,
        )
    except Exception as exc:
        return PromotionEligibility(
            model_id=model_id,
            eligible=False,
            reasons=(f"Artifact validation failed: {type(exc).__name__}: {exc}",),
        )
    record = loaded.registry_record
    reasons: list[str] = []
    if str(record["model_status"]) != "candidate":
        reasons.append("Model status is not candidate.")
    if str(record["validation_status"]) != "validation_pass":
        reasons.append("Validation status is not validation_pass.")
    if str(record["promotion_status"]) != "candidate":
        reasons.append("Promotion status is not candidate.")
    decision = dict(loaded.verification.metadata["validation"])["decision"]
    if not isinstance(decision, Mapping) or decision.get("passed") is not True:
        reasons.append("Persisted candidate decision did not pass.")
    return PromotionEligibility(
        model_id=model_id,
        eligible=not reasons,
        reasons=tuple(reasons) if reasons else ("Candidate is eligible for explicit promotion.",),
    )
