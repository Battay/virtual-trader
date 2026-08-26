"""Persistent, restart-safe orchestration for independent recurrent agents.

Version 1 deliberately runs one worker at a time. It creates one job for every
member of the frozen identity universe, retains explicit ineligible records,
loads TRAIN for optimization, and optionally loads VALIDATION only after
training. TEST is not exposed by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Sequence

import pandas as pd

from data_pipeline.src.clustering_market_mode import (
    load_authoritative_current_equity_identity,
)
from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    SAVED_MODELS_DIR,
    TRAINING_RUNS_DIR,
)
from data_pipeline.src.equity_universe import (
    CLASSIFICATION_POLICY_VERSION,
    EQUITY_UNIVERSE_VERSION,
    IDENTITY_POLICY,
    deterministic_universe_identity,
)
from data_pipeline.src.instrument_audit import COMMON_EQUITY, OFFICIAL_LISTING_DERIVED
from feature_engineering.schemas import FEATURE_VERSION
from feature_engineering.storage import atomic_write_json, safe_path_component
from reinforcement_learning.canonical_recurrent_artifacts import (
    CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
    SUPPORTED_RECURRENT_TRAIN_CONTRACT_VERSIONS,
    load_canonical_recovery_evidence,
    load_training_recurrent_contract_metadata,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.evaluation.recurrent_evaluator import (
    evaluate_recurrent_on_validation,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.history_policy import MATURE_MINIMUM_USABLE_OBSERVATIONS
from reinforcement_learning.recurrent_data_contract import (
    RecurrentDataContractError,
)

from .callbacks import TrainingProgress
from .devices import (
    TorchDeviceResolution,
    resolve_torch_device,
    torch_devices_equivalent,
)
from .job_state import (
    COMPLETED,
    FAILED,
    INELIGIBLE,
    INTERRUPTED,
    PENDING,
    QUEUED,
    STALE,
    TRAINING,
    TRAINING_JOB_SCHEMA_VERSION,
    TRAINING_ORCHESTRATOR_VERSION,
    TRAINING_RUN_SCHEMA_VERSION,
    VALIDATING,
    TrainingJobRecord,
    TrainingJobStateError,
    TrainingRunManifest,
    canonical_hash,
    progress_snapshot,
    transition_job,
    utc_now,
)
from .recurrent_config import RecurrentPPOConfig
from .recurrent_results import RecurrentPPOTrainingResult
from .recurrent_trainer import (
    RECURRENT_TRAINER_VERSION,
    train_recurrent_single_symbol,
)


DEFAULT_READINESS_EVIDENCE_PATH = (
    PROCESSED_DATA_DIR / "sector_universes" / "current_verified_symbols.csv"
)
RUN_MANIFEST_FILENAME = "run_manifest.json"
JOBS_DIRECTORY_NAME = "jobs"
MODELS_DIRECTORY_NAME = "models"
CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
VALIDATION_DIRECTORY_NAME = "validation"
LOGS_DIRECTORY_NAME = "logs"
RESUME_CAPABILITY = "restart_from_zero_only_no_optimizer_checkpoint_v1"

ELIGIBLE_TRAINABLE = "eligible_trainable"
INSUFFICIENT_DATA = "insufficient_data"
INCOMPATIBLE_FEATURE_CONTRACT = "incompatible_feature_contract"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"
UNSUPPORTED = "unsupported"
INCOMPATIBLE_CONTRACT = "incompatible_contract"


class RecurrentOrchestratorError(RuntimeError):
    """Raised when a run cannot proceed without corrupting its state contract."""


@dataclass(frozen=True)
class RecurrentUniverseDiscovery:
    records: pd.DataFrame
    universe_version: str
    universe_hash: str
    identity_count: int
    category_counts: Mapping[str, int]
    source_inventory_hash: str

    @property
    def eligible_count(self) -> int:
        return int(self.category_counts.get(ELIGIBLE_TRAINABLE, 0))

    @property
    def ineligible_count(self) -> int:
        return self.identity_count - self.eligible_count


def _identity_universe_hash(identity: pd.DataFrame) -> str:
    required = {"symbol", "sector", "security_type", "source", "snapshot_date"}
    missing = sorted(required.difference(identity.columns))
    if missing:
        raise RecurrentOrchestratorError(
            "identity is missing provenance columns: " + ", ".join(missing)
        )
    ordered = identity.sort_values("symbol", kind="mergesort")
    snapshots = sorted(set(ordered["snapshot_date"].astype(str)))
    if len(snapshots) != 1:
        raise RecurrentOrchestratorError("identity must have one snapshot date")
    members = [
        {
            "symbol": str(row.symbol),
            "instrument_category": COMMON_EQUITY,
            "classification_basis": OFFICIAL_LISTING_DERIVED,
            "security_type": str(row.security_type),
            "sector": str(row.sector),
            "authoritative_source": str(row.source),
        }
        for row in ordered.itertuples(index=False)
    ]
    return deterministic_universe_identity(
        {
            "universe_version": EQUITY_UNIVERSE_VERSION,
            "identity_policy": IDENTITY_POLICY,
            "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
            "listing_snapshot_date": snapshots[0],
            "members": members,
        }
    )


def _readiness_evidence(path: Path | None) -> dict[str, Mapping[str, object]]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        frame = pd.read_csv(path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return {}
    if "symbol" not in frame or frame["symbol"].duplicated().any():
        return {}
    return {
        str(row["symbol"]): row
        for row in frame.to_dict(orient="records")
        if str(row.get("symbol", "")).strip()
    }


def _missing_contract_category(
    evidence: Mapping[str, object] | None,
    recovery_evidence: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    if recovery_evidence and recovery_evidence.get("recovery_valid") is False:
        usable = int(recovery_evidence.get("final_usable_feature_rows", 0))
        return (
            INSUFFICIENT_DATA,
            "insufficient_canonical_train_history_v2:"
            f"{usable}<{MATURE_MINIMUM_USABLE_OBSERVATIONS}",
        )
    if not evidence:
        return MISSING_REQUIRED_ARTIFACTS, "recurrent_contract_missing"
    history = str(evidence.get("history_class", "")).strip().upper()
    exclusion = str(evidence.get("exclusion_reason", "")).strip()
    if history == "INSUFFICIENT":
        return INSUFFICIENT_DATA, "insufficient_usable_history"
    if history == "COLD_START":
        return INSUFFICIENT_DATA, "cold_start_not_independent_training"
    if exclusion.startswith("unsupported_security_type"):
        return UNSUPPORTED, exclusion
    if exclusion == "not_active_recently_traded":
        return UNSUPPORTED, exclusion
    return MISSING_REQUIRED_ARTIFACTS, exclusion or "recurrent_contract_missing"


def _metadata_compatibility(metadata: object) -> None:
    if metadata.recurrent_contract_version not in SUPPORTED_RECURRENT_TRAIN_CONTRACT_VERSIONS:
        raise RecurrentOrchestratorError("recurrent contract version is incompatible")
    if metadata.feature_version not in {
        FEATURE_VERSION,
        CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
    }:
        raise RecurrentOrchestratorError("feature version is incompatible")
    if metadata.environment_version != ENVIRONMENT_VERSION:
        raise RecurrentOrchestratorError("environment version is incompatible")
    if metadata.observation_features != DEFAULT_OBSERVATION_FEATURES:
        raise RecurrentOrchestratorError("observation feature ordering is incompatible")
    if not metadata.history.independent_recurrent_ready:
        raise RecurrentOrchestratorError("history is not independent-training ready")
    if metadata.training_scope != "symbol" or metadata.constituent_symbols != (
        metadata.symbol,
    ):
        raise RecurrentOrchestratorError("recurrent contract is not symbol-scoped")


def discover_recurrent_training_universe(
    *,
    identity: pd.DataFrame | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    readiness_evidence_path: Path | None = DEFAULT_READINESS_EVIDENCE_PATH,
    metadata_loader: Callable[..., object] = (
        load_training_recurrent_contract_metadata
    ),
) -> RecurrentUniverseDiscovery:
    """Account for every frozen identity without loading a market partition."""

    universe = (
        load_authoritative_current_equity_identity()
        if identity is None
        else identity.copy(deep=True)
    )
    required = {
        "symbol",
        "company_name",
        "sector",
        "security_type",
        "source",
        "snapshot_date",
    }
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise RecurrentOrchestratorError(
            "identity universe is missing fields: " + ", ".join(missing)
        )
    universe["symbol"] = universe["symbol"].astype("string").str.strip()
    universe = universe.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    if universe["symbol"].eq("").any() or universe["symbol"].duplicated().any():
        raise RecurrentOrchestratorError("identity symbols must be unique and nonempty")
    evidence = _readiness_evidence(readiness_evidence_path)
    recovery_evidence = load_canonical_recovery_evidence()
    records: list[dict[str, object]] = []
    for row in universe.to_dict(orient="records"):
        symbol = str(row["symbol"])
        metadata: object | None = None
        category = ELIGIBLE_TRAINABLE
        reason = "canonical_mature_recurrent_contract"
        error = ""
        try:
            metadata = metadata_loader(symbol, splits_dir=Path(splits_dir))
            _metadata_compatibility(metadata)
        except FileNotFoundError as exc:
            category, reason = _missing_contract_category(
                evidence.get(symbol), recovery_evidence.get(symbol)
            )
            error = f"{type(exc).__name__}: {exc}"
        except RecurrentDataContractError as exc:
            text = str(exc).lower()
            if "missing" in text:
                category, reason = _missing_contract_category(
                    evidence.get(symbol), recovery_evidence.get(symbol)
                )
            elif "feature" in text:
                category, reason = INCOMPATIBLE_FEATURE_CONTRACT, str(exc)
            elif "history" in text:
                category, reason = INSUFFICIENT_DATA, str(exc)
            else:
                category, reason = INCOMPATIBLE_CONTRACT, str(exc)
            error = f"{type(exc).__name__}: {exc}"
        except (OSError, ValueError, RecurrentOrchestratorError) as exc:
            text = str(exc).lower()
            category = (
                INCOMPATIBLE_FEATURE_CONTRACT
                if "feature" in text
                else INCOMPATIBLE_CONTRACT
            )
            reason = str(exc)
            error = f"{type(exc).__name__}: {exc}"
        contract_hash = sha256_file(metadata.contract_path) if metadata else ""
        records.append(
            {
                "symbol": symbol,
                "company_name": str(row["company_name"]),
                "sector": str(row["sector"]),
                "security_type": str(row["security_type"]),
                "category": category,
                "reason": reason,
                "compatibility_error": error,
                "recurrent_contract_version": (
                    metadata.recurrent_contract_version if metadata else ""
                ),
                "environment_version": metadata.environment_version if metadata else "",
                "feature_version": metadata.feature_version if metadata else "",
                "source_data_hash": contract_hash,
                "train_rows": metadata.train.rows if metadata else 0,
                "train_start": metadata.train.start if metadata else "",
                "train_end": metadata.train.end if metadata else "",
                "validation_available": (
                    bool(getattr(metadata, "validation_available", True))
                    if metadata
                    else False
                ),
            }
        )
    result = pd.DataFrame(records).sort_values("symbol", kind="mergesort").reset_index(
        drop=True
    )
    if len(result) != len(universe) or result["symbol"].duplicated().any():
        raise RecurrentOrchestratorError("training discovery lost identity members")
    counts = result["category"].value_counts().sort_index().to_dict()
    source_inventory = canonical_hash(
        {
            "universe_hash": _identity_universe_hash(universe),
            "contracts": [
                {
                    "symbol": str(row.symbol),
                    "category": str(row.category),
                    "source_data_hash": str(row.source_data_hash),
                }
                for row in result.itertuples(index=False)
            ],
        }
    )
    return RecurrentUniverseDiscovery(
        records=result,
        universe_version=EQUITY_UNIVERSE_VERSION,
        universe_hash=_identity_universe_hash(universe),
        identity_count=len(universe),
        category_counts=counts,
        source_inventory_hash=source_inventory,
    )


def _run_identity(
    discovery: RecurrentUniverseDiscovery,
    config: RecurrentPPOConfig,
    *,
    validation_enabled: bool,
) -> tuple[str, str, str]:
    configuration = config.to_dict()
    hyperparameters_hash = canonical_hash(configuration)
    identity = {
        "orchestrator_version": TRAINING_ORCHESTRATOR_VERSION,
        "universe_version": discovery.universe_version,
        "universe_hash": discovery.universe_hash,
        "source_inventory_hash": discovery.source_inventory_hash,
        "agent_version": RECURRENT_TRAINER_VERSION,
        "hyperparameters_hash": hyperparameters_hash,
        "requested_timesteps": config.total_timesteps,
        "seed": config.seed,
        "requested_device": config.device,
        "validation_enabled": validation_enabled,
        "worker_limit": 1,
    }
    fingerprint = canonical_hash(identity)
    run_id = f"rppo-symbols-{fingerprint[:20]}"
    return run_id, fingerprint, hyperparameters_hash


def _artifact_relative_path(symbol: str, kind: str, *, attempt: int = 0) -> str:
    safe = safe_path_component(symbol)
    if attempt < 0:
        raise ValueError("artifact attempt cannot be negative")
    attempt_name = f"attempt_{attempt:03d}"
    if kind == "model":
        return f"{MODELS_DIRECTORY_NAME}/{safe}/{attempt_name}/model.zip"
    if kind == "checkpoint":
        return f"{CHECKPOINTS_DIRECTORY_NAME}/{safe}/{attempt_name}"
    if kind == "validation":
        return f"{VALIDATION_DIRECTORY_NAME}/{safe}/{attempt_name}.json"
    raise ValueError(f"unsupported artifact kind: {kind}")


def build_training_run(
    discovery: RecurrentUniverseDiscovery,
    *,
    config: RecurrentPPOConfig,
    created_at: str | None = None,
    validation_enabled: bool = True,
) -> tuple[TrainingRunManifest, tuple[TrainingJobRecord, ...]]:
    """Create deterministic run/job identities with all universe members present."""

    when = created_at or utc_now()
    run_id, fingerprint, hyperparameters_hash = _run_identity(
        discovery, config, validation_enabled=validation_enabled
    )
    jobs: list[TrainingJobRecord] = []
    for row in discovery.records.itertuples(index=False):
        eligible = str(row.category) == ELIGIBLE_TRAINABLE
        job_identity = canonical_hash(
            {
                "run_fingerprint": fingerprint,
                "symbol": str(row.symbol),
                "source_data_hash": str(row.source_data_hash),
            }
        )
        status = QUEUED if eligible else INELIGIBLE
        validation_available = bool(getattr(row, "validation_available", True))
        jobs.append(
            TrainingJobRecord(
                schema_version=TRAINING_JOB_SCHEMA_VERSION,
                job_id=f"job-{safe_path_component(str(row.symbol))}-{job_identity[:16]}",
                run_id=run_id,
                symbol=str(row.symbol),
                trainability="eligible" if eligible else "ineligible",
                trainability_reason=(
                    "canonical_mature_recurrent_contract"
                    if eligible
                    else f"{row.category}:{row.reason}"
                ),
                agent_version=RECURRENT_TRAINER_VERSION,
                environment_version=str(row.environment_version),
                data_contract_version=str(row.recurrent_contract_version),
                feature_version=str(row.feature_version),
                universe_version=discovery.universe_version,
                universe_hash=discovery.universe_hash,
                source_data_hash=str(row.source_data_hash),
                requested_timesteps=config.total_timesteps,
                completed_timesteps=0,
                seed=config.seed,
                hyperparameters_hash=hyperparameters_hash,
                requested_device=config.device,
                effective_device=None,
                device_name=None,
                status=status,
                created_at=when,
                started_at=None,
                updated_at=when,
                completed_at=(when if not eligible else None),
                wall_clock_duration_seconds=0.0,
                checkpoint_path=_artifact_relative_path(str(row.symbol), "checkpoint"),
                model_path=None,
                model_sha256=None,
                validation_status=(
                    "pending"
                    if validation_enabled and eligible and validation_available
                    else (
                        "not_available_train_only_contract"
                        if validation_enabled and eligible and not validation_available
                        else "not_requested"
                    )
                ),
                validation_metrics_reference=None,
                failure_error_message=None,
                retry_count=0,
                state_history=(
                    {
                        "from": None,
                        "to": status,
                        "timestamp": when,
                        "message": "deterministic_job_created",
                    },
                ),
            )
        )
    jobs.sort(key=lambda item: item.symbol)
    manifest = TrainingRunManifest(
        schema_version=TRAINING_RUN_SCHEMA_VERSION,
        orchestrator_version=TRAINING_ORCHESTRATOR_VERSION,
        run_id=run_id,
        run_fingerprint=fingerprint,
        universe_version=discovery.universe_version,
        universe_hash=discovery.universe_hash,
        identity_count=discovery.identity_count,
        eligible_count=discovery.eligible_count,
        ineligible_count=discovery.ineligible_count,
        agent_version=RECURRENT_TRAINER_VERSION,
        requested_timesteps=config.total_timesteps,
        seed=config.seed,
        requested_device=config.device,
        hyperparameters_hash=hyperparameters_hash,
        source_inventory_hash=discovery.source_inventory_hash,
        validation_enabled=validation_enabled,
        worker_limit=1,
        resume_capability=RESUME_CAPABILITY,
        created_at=when,
    )
    return manifest, tuple(jobs)


class TrainingRunStore:
    """Atomic per-job JSON persistence under one isolated run directory."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory).expanduser().resolve(strict=False)
        production = Path(SAVED_MODELS_DIR).resolve(strict=False)
        if self.run_directory == production or production in self.run_directory.parents:
            raise RecurrentOrchestratorError(
                "orchestration runs cannot live in production saved-model directories"
            )
        self.manifest_path = self.run_directory / RUN_MANIFEST_FILENAME
        self.jobs_directory = self.run_directory / JOBS_DIRECTORY_NAME
        self.lock_path = self.run_directory / ".state.lock"

    def initialize(
        self,
        manifest: TrainingRunManifest,
        jobs: Sequence[TrainingJobRecord],
    ) -> None:
        if self.run_directory.exists() and any(self.run_directory.iterdir()):
            raise FileExistsError(f"training run already exists: {self.run_directory}")
        for name in (
            JOBS_DIRECTORY_NAME,
            MODELS_DIRECTORY_NAME,
            CHECKPOINTS_DIRECTORY_NAME,
            VALIDATION_DIRECTORY_NAME,
            LOGS_DIRECTORY_NAME,
        ):
            (self.run_directory / name).mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest.to_dict(), self.manifest_path)
        for job in jobs:
            self.write_job(job)

    def _job_path(self, symbol: str) -> Path:
        return self.jobs_directory / f"{safe_path_component(symbol)}.json"

    def read_manifest(self) -> TrainingRunManifest:
        return TrainingRunManifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )

    def read_job(self, symbol: str) -> TrainingJobRecord:
        path = self._job_path(symbol)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RecurrentOrchestratorError(f"job is missing: {symbol}") from exc
        return TrainingJobRecord.from_dict(payload)

    def write_job(self, job: TrainingJobRecord) -> None:
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(job.to_dict(), self._job_path(job.symbol))

    def update_job(
        self,
        symbol: str,
        updater: Callable[[TrainingJobRecord], TrainingJobRecord],
    ) -> TrainingJobRecord:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                current = self.read_job(symbol)
                updated = updater(current)
                if updated.symbol != current.symbol or updated.run_id != current.run_id:
                    raise RecurrentOrchestratorError("job updater changed identity")
                self.write_job(updated)
                return updated
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def list_jobs(self) -> tuple[TrainingJobRecord, ...]:
        jobs = [
            TrainingJobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.jobs_directory.glob("*.json"))
        ]
        manifest = self.read_manifest()
        if len(jobs) != manifest.identity_count or len({job.symbol for job in jobs}) != len(jobs):
            raise RecurrentOrchestratorError("persisted job inventory does not reconcile")
        return tuple(sorted(jobs, key=lambda item: item.symbol))

    def resolve_artifact(self, relative_path: str) -> Path:
        value = Path(relative_path)
        if value.is_absolute() or ".." in value.parts:
            raise RecurrentOrchestratorError("job artifact path must be run-relative")
        resolved = (self.run_directory / value).resolve(strict=False)
        if self.run_directory not in resolved.parents:
            raise RecurrentOrchestratorError("job artifact escaped run directory")
        return resolved


def create_training_run(
    discovery: RecurrentUniverseDiscovery,
    *,
    config: RecurrentPPOConfig,
    runs_root: Path = TRAINING_RUNS_DIR,
    validation_enabled: bool = True,
    created_at: str | None = None,
) -> TrainingRunStore:
    manifest, jobs = build_training_run(
        discovery,
        config=config,
        created_at=created_at,
        validation_enabled=validation_enabled,
    )
    root = Path(runs_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / manifest.run_id
    if destination.exists():
        raise FileExistsError(f"training run already exists: {destination}")
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{manifest.run_id}."))
    try:
        TrainingRunStore(staging).initialize(manifest, jobs)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return TrainingRunStore(destination)


def recover_interrupted_jobs(
    store: TrainingRunStore,
    *,
    timestamp: str | None = None,
) -> tuple[str, ...]:
    """Convert abandoned in-flight markers to honest INTERRUPTED states."""

    recovered: list[str] = []
    for job in store.list_jobs():
        if job.status not in {TRAINING, VALIDATING}:
            continue
        store.update_job(
            job.symbol,
            lambda current: transition_job(
                current,
                INTERRUPTED,
                timestamp=timestamp,
                message="stale_inflight_state_recovered_after_process_restart",
                failure_error_message=(
                    "Process ended while job was in flight; optimizer state was not "
                    "checkpointed and an explicit restart is required."
                ),
            ),
        )
        recovered.append(job.symbol)
    return tuple(recovered)


def explicitly_requeue_job(
    store: TrainingRunStore,
    symbol: str,
    *,
    timestamp: str | None = None,
) -> TrainingJobRecord:
    """Explicitly restart FAILED/INTERRUPTED/STALE work from timestep zero."""

    def restart(current: TrainingJobRecord) -> TrainingJobRecord:
        queued = transition_job(
            current,
            QUEUED,
            timestamp=timestamp,
            message="explicit_restart_from_zero_not_checkpoint_resume",
        )
        return replace(
            queued,
            checkpoint_path=_artifact_relative_path(
                queued.symbol, "checkpoint", attempt=queued.retry_count
            ),
        )

    return store.update_job(symbol, restart)


def job_contract_compatibility(
    job: TrainingJobRecord,
    *,
    metadata_loader: Callable[..., object] = (
        load_training_recurrent_contract_metadata
    ),
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> tuple[bool, str]:
    """Verify that current training inputs still match the persisted job."""

    try:
        metadata = metadata_loader(job.symbol, splits_dir=Path(splits_dir))
        _metadata_compatibility(metadata)
    except Exception as exc:
        return False, f"contract_unavailable_or_incompatible:{type(exc).__name__}:{exc}"
    current_hash = sha256_file(metadata.contract_path)
    comparisons = (
        (job.agent_version, RECURRENT_TRAINER_VERSION, "agent_version_changed"),
        (job.environment_version, metadata.environment_version, "environment_version_changed"),
        (job.data_contract_version, metadata.recurrent_contract_version, "data_contract_changed"),
        (job.feature_version, metadata.feature_version, "feature_version_changed"),
        (job.source_data_hash, current_hash, "source_contract_hash_changed"),
    )
    for recorded, current, reason in comparisons:
        if recorded != current:
            return False, reason
    return True, "job_contract_compatible"


def completed_job_compatibility(
    store: TrainingRunStore,
    job: TrainingJobRecord,
    *,
    metadata_loader: Callable[..., object] = (
        load_training_recurrent_contract_metadata
    ),
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> tuple[bool, str]:
    """Verify contract and artifact identity before a completed job is skipped."""

    if job.status != COMPLETED:
        return False, "job_not_completed"
    compatible, reason = job_contract_compatibility(
        job,
        metadata_loader=metadata_loader,
        splits_dir=Path(splits_dir),
    )
    if not compatible:
        return False, reason
    if not job.model_path or not job.model_sha256:
        return False, "model_reference_missing"
    try:
        model_path = store.resolve_artifact(job.model_path)
    except RecurrentOrchestratorError as exc:
        return False, f"model_path_invalid:{exc}"
    if not model_path.is_file():
        return False, "model_artifact_missing"
    if sha256_file(model_path) != job.model_sha256:
        return False, "model_artifact_hash_changed"
    return True, "completed_compatible_model"


def mark_stale_jobs(
    store: TrainingRunStore,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    metadata_loader: Callable[..., object] = (
        load_training_recurrent_contract_metadata
    ),
) -> tuple[str, ...]:
    """Mark completed jobs STALE when current contracts/artifacts no longer match."""

    stale: list[str] = []
    for job in store.list_jobs():
        if job.status in {INELIGIBLE, STALE, TRAINING, VALIDATING}:
            continue
        if job.status == COMPLETED:
            compatible, reason = completed_job_compatibility(
                store,
                job,
                metadata_loader=metadata_loader,
                splits_dir=Path(splits_dir),
            )
        else:
            compatible, reason = job_contract_compatibility(
                job,
                metadata_loader=metadata_loader,
                splits_dir=Path(splits_dir),
            )
        if compatible:
            continue
        store.update_job(
            job.symbol,
            lambda current: transition_job(
                current,
                STALE,
                message="completed_job_contract_or_artifact_became_stale",
                failure_error_message=reason,
            ),
        )
        stale.append(job.symbol)
    return tuple(stale)


def _file_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _save_model_atomically(model: object, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"model path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.stem}.", suffix=".zip"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RecurrentOrchestratorError("trainer did not create a model archive")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def validate_recurrent_device_resolution(
    resolution: TorchDeviceResolution,
) -> None:
    """Fail closed on silent fallback and on AUTO selecting unstable MPS."""

    requested = resolution.requested_device
    effective = resolution.resolved_device
    if requested == "auto":
        if effective not in {"cpu", "cuda"}:
            raise RecurrentOrchestratorError(
                "AUTO recurrent scheduling may select CUDA or CPU, never MPS"
            )
        return
    if not torch_devices_equivalent(requested, effective):
        raise RecurrentOrchestratorError(
            f"explicit {requested} request resolved to {effective}; silent fallback denied"
        )


def _elapsed(started_at: str | None, now: float, wall_start: float) -> float:
    del started_at
    return max(0.0, now - wall_start)


def execute_queued_jobs(
    store: TrainingRunStore,
    *,
    config: RecurrentPPOConfig,
    max_jobs: int = 1,
    symbols: Sequence[str] | None = None,
    fail_fast: bool = False,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    trainer: Callable[..., RecurrentPPOTrainingResult] = (
        train_recurrent_single_symbol
    ),
    evaluator: Callable[..., object] = evaluate_recurrent_on_validation,
    device_resolver: Callable[[str], TorchDeviceResolution] = resolve_torch_device,
    metadata_loader: Callable[..., object] = (
        load_training_recurrent_contract_metadata
    ),
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> tuple[TrainingJobRecord, ...]:
    """Run at most ``max_jobs`` queued symbols sequentially and persist each stage."""

    if max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    manifest = store.read_manifest()
    if config.total_timesteps != manifest.requested_timesteps or config.seed != manifest.seed:
        raise RecurrentOrchestratorError("runtime configuration differs from run manifest")
    if config.device != manifest.requested_device:
        raise RecurrentOrchestratorError("runtime device differs from run manifest")
    if canonical_hash(config.to_dict()) != manifest.hyperparameters_hash:
        raise RecurrentOrchestratorError("runtime hyperparameters differ from run manifest")
    mark_stale_jobs(
        store,
        splits_dir=Path(splits_dir),
        metadata_loader=metadata_loader,
    )
    requested_symbols = None if symbols is None else {str(value).strip() for value in symbols}
    jobs = [job for job in store.list_jobs() if job.status == QUEUED]
    if requested_symbols is not None:
        unknown = requested_symbols.difference(job.symbol for job in store.list_jobs())
        if unknown:
            raise RecurrentOrchestratorError(
                "requested symbols are absent from run: " + ", ".join(sorted(unknown))
            )
        jobs = [job for job in jobs if job.symbol in requested_symbols]
    jobs.sort(key=lambda item: item.symbol)
    registry_before = _file_bytes(Path(registry_path))
    outcomes: list[TrainingJobRecord] = []
    for queued in jobs[:max_jobs]:
        wall_start = time.perf_counter()
        try:
            training_job = store.update_job(
                queued.symbol,
                lambda current: transition_job(
                    current,
                    TRAINING,
                    message="single_worker_training_started",
                ),
            )
            resolution = device_resolver(config.device)
            validate_recurrent_device_resolution(resolution)
            training_job = store.update_job(
                queued.symbol,
                lambda current: replace(
                    current,
                    effective_device=resolution.resolved_device,
                    device_name=resolution.device_name,
                    updated_at=utc_now(),
                ),
            )

            def progress(event: TrainingProgress) -> bool:
                if event.symbol != training_job.symbol:
                    raise RecurrentOrchestratorError("trainer emitted progress for another symbol")
                store.update_job(
                    event.symbol,
                    lambda current: replace(
                        current,
                        completed_timesteps=max(
                            current.completed_timesteps, int(event.current_timesteps)
                        ),
                        updated_at=event.timestamp,
                        wall_clock_duration_seconds=_elapsed(
                            current.started_at, time.perf_counter(), wall_start
                        ),
                    ),
                )
                return True

            result = trainer(
                queued.symbol,
                config=config,
                device=resolution.resolved_device,
                total_timesteps=config.total_timesteps,
                seed=config.seed,
                progress_callback=progress,
                splits_dir=Path(splits_dir),
                smoke_test=config.total_timesteps <= 1_024,
            )
            duration = max(0.0, time.perf_counter() - wall_start)
            if result.status == "interrupted":
                outcome = store.update_job(
                    queued.symbol,
                    lambda current: transition_job(
                        current,
                        INTERRUPTED,
                        message=result.message,
                        completed_timesteps=result.actual_timesteps,
                        wall_clock_duration_seconds=duration,
                        failure_error_message=result.error,
                    ),
                )
                outcomes.append(outcome)
                continue
            if not result.succeeded or result.model is None:
                raise RecurrentOrchestratorError(
                    result.error or result.message or "trainer returned failure"
                )
            if not torch_devices_equivalent(
                result.device, resolution.resolved_device
            ):
                raise RecurrentOrchestratorError(
                    "trainer effective device differs from scheduled device"
                )
            relative_model = _artifact_relative_path(
                queued.symbol, "model", attempt=queued.retry_count
            )
            model_path = store.resolve_artifact(relative_model)
            model_hash = _save_model_atomically(result.model, model_path)
            validation_available = (
                queued.validation_status != "not_available_train_only_contract"
            )
            if manifest.validation_enabled and validation_available:
                store.update_job(
                    queued.symbol,
                    lambda current: transition_job(
                        current,
                        VALIDATING,
                        message="validation_only_evaluation_started",
                        completed_timesteps=result.actual_timesteps,
                        wall_clock_duration_seconds=duration,
                        model_path=relative_model,
                        model_sha256=model_hash,
                        validation_status="running",
                    ),
                )
                validation = evaluator(
                    result.model,
                    queued.symbol,
                    trainer_result=result,
                    seed=config.seed,
                    splits_dir=Path(splits_dir),
                )
                relative_validation = _artifact_relative_path(
                    queued.symbol, "validation", attempt=queued.retry_count
                )
                validation_path = store.resolve_artifact(relative_validation)
                payload = (
                    validation.to_dict(include_history=False)
                    if hasattr(validation, "to_dict")
                    else dict(validation)
                )
                if payload.get("test_evaluated") is True:
                    raise RecurrentOrchestratorError("evaluator reported TEST access")
                atomic_write_json(payload, validation_path)
                validation_status = "completed"
            else:
                relative_validation = None
                validation_status = (
                    "not_available_train_only_contract"
                    if manifest.validation_enabled and not validation_available
                    else "not_requested"
                )
            duration = max(0.0, time.perf_counter() - wall_start)
            outcome = store.update_job(
                queued.symbol,
                lambda current: transition_job(
                    current,
                    COMPLETED,
                    message="training_and_requested_validation_completed",
                    completed_timesteps=result.actual_timesteps,
                    wall_clock_duration_seconds=duration,
                    model_path=relative_model,
                    model_sha256=model_hash,
                    validation_status=validation_status,
                    validation_metrics_reference=relative_validation,
                    failure_error_message=None,
                ),
            )
            outcomes.append(outcome)
        except KeyboardInterrupt:
            current = store.read_job(queued.symbol)
            if current.status in {TRAINING, VALIDATING}:
                current = store.update_job(
                    queued.symbol,
                    lambda value: transition_job(
                        value,
                        INTERRUPTED,
                        message="orchestrator_keyboard_interrupt",
                        wall_clock_duration_seconds=max(
                            value.wall_clock_duration_seconds,
                            time.perf_counter() - wall_start,
                        ),
                        failure_error_message=(
                            "Interrupted without optimizer checkpoint; explicit "
                            "restart from zero is required."
                        ),
                    ),
                )
            outcomes.append(current)
            break
        except Exception as exc:
            current = store.read_job(queued.symbol)
            if current.status in {TRAINING, VALIDATING}:
                current = store.update_job(
                    queued.symbol,
                    lambda value: transition_job(
                        value,
                        FAILED,
                        message="symbol_failed_without_affecting_other_jobs",
                        wall_clock_duration_seconds=max(
                            value.wall_clock_duration_seconds,
                            time.perf_counter() - wall_start,
                        ),
                        failure_error_message=f"{type(exc).__name__}: {exc}",
                    ),
                )
            outcomes.append(current)
            if fail_fast:
                break
        finally:
            if _file_bytes(Path(registry_path)) != registry_before:
                raise RecurrentOrchestratorError("model registry changed during run")
    return tuple(outcomes)


def training_status_table(store: TrainingRunStore) -> pd.DataFrame:
    """Return deterministic download-style job progress for every identity."""

    rows = [progress_snapshot(job) for job in store.list_jobs()]
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(
        drop=True
    )


def _print_discovery(discovery: RecurrentUniverseDiscovery) -> None:
    print(f"Universe: {discovery.universe_version}")
    print(f"Universe hash: {discovery.universe_hash}")
    print(f"Identity symbols: {discovery.identity_count}")
    for category, count in sorted(discovery.category_counts.items()):
        print(f"{category}: {count}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track sequential recurrent PPO jobs")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--create-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--runs-root", type=Path, default=TRAINING_RUNS_DIR)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--no-validation", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    actions = sum(bool(value) for value in (args.discover, args.create_run, args.status, args.run))
    if actions != 1:
        print("Select exactly one of --discover, --create-run, --status, or --run")
        return 2
    if args.discover:
        _print_discovery(discover_recurrent_training_universe())
        return 0
    config = RecurrentPPOConfig(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
    )
    if args.create_run:
        discovery = discover_recurrent_training_universe()
        store = create_training_run(
            discovery,
            config=config,
            runs_root=args.runs_root,
            validation_enabled=not args.no_validation,
        )
        print(store.run_directory)
        return 0
    if args.run_directory is None:
        print("--run-directory is required for --status or --run")
        return 2
    store = TrainingRunStore(args.run_directory)
    if args.status:
        print(training_status_table(store).to_string(index=False))
        return 0
    outcomes = execute_queued_jobs(
        store,
        config=config,
        max_jobs=args.max_jobs,
        symbols=args.symbol,
    )
    print(pd.DataFrame([progress_snapshot(job) for job in outcomes]).to_string(index=False))
    return 0 if all(job.status == COMPLETED for job in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ELIGIBLE_TRAINABLE",
    "INCOMPATIBLE_CONTRACT",
    "INCOMPATIBLE_FEATURE_CONTRACT",
    "INSUFFICIENT_DATA",
    "MISSING_REQUIRED_ARTIFACTS",
    "RESUME_CAPABILITY",
    "UNSUPPORTED",
    "RecurrentOrchestratorError",
    "RecurrentUniverseDiscovery",
    "TrainingRunStore",
    "build_training_run",
    "create_training_run",
    "completed_job_compatibility",
    "discover_recurrent_training_universe",
    "execute_queued_jobs",
    "explicitly_requeue_job",
    "job_contract_compatibility",
    "mark_stale_jobs",
    "recover_interrupted_jobs",
    "training_status_table",
    "validate_recurrent_device_resolution",
]
