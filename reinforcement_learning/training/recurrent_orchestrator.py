"""Persistent, restart-safe orchestration for independent recurrent agents.

The v1 job and run schemas remain stable. Execution defaults to the validated
sequential path and additionally supports explicit bounded CPU process workers.
Workers own only symbol-local temporary artifacts; the parent owns every shared
state transition and promotion. TEST is not exposed by this module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import json
import math
import multiprocessing
from multiprocessing.connection import wait as wait_for_connections
import os
from pathlib import Path
import resource
import shutil
import sys
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
from data_pipeline.src.identity_universe_policy import (
    CURRENT_OPERATIONAL_IDENTITY,
    CURRENT_OPERATIONAL_TRAINING_POLICY,
)
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
WORKSPACES_DIRECTORY_NAME = "workspaces"
ACTIVE_WORKERS_FILENAME = "active_workers.json"
EXECUTION_LOCK_FILENAME = ".execution.lock"
RESUME_CAPABILITY = "restart_from_zero_only_no_optimizer_checkpoint_v1"
SUPPORTED_PROCESS_WORKERS = (1, 2, 4)

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
    identity_policy: str = CURRENT_OPERATIONAL_IDENTITY
    identity_snapshot: str = "current"
    execution_training_policy: str = CURRENT_OPERATIONAL_TRAINING_POLICY

    @property
    def eligible_count(self) -> int:
        return int(self.category_counts.get(ELIGIBLE_TRAINABLE, 0))

    @property
    def ineligible_count(self) -> int:
        return self.identity_count - self.eligible_count

    @property
    def trainable_symbol_hash(self) -> str:
        symbols = sorted(
            self.records.loc[
                self.records["category"].eq(ELIGIBLE_TRAINABLE), "symbol"
            ].astype(str)
        )
        return canonical_hash({"symbols": symbols})


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
    identity_policy: str | None = None,
    execution_training_policy: str | None = None,
) -> RecurrentUniverseDiscovery:
    """Account for every identity without loading a market partition."""

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
    snapshots = sorted(set(universe["snapshot_date"].astype(str)))
    if len(snapshots) != 1:
        raise RecurrentOrchestratorError("identity must have one snapshot date")
    resolved_identity_policy = (
        str(identity_policy).strip()
        if identity_policy is not None
        else str(universe.attrs.get("identity_role", CURRENT_OPERATIONAL_IDENTITY))
    )
    resolved_training_policy = (
        str(execution_training_policy).strip()
        if execution_training_policy is not None
        else str(
            universe.attrs.get(
                "execution_training_policy", CURRENT_OPERATIONAL_TRAINING_POLICY
            )
        )
    )
    if not resolved_identity_policy or not resolved_training_policy:
        raise RecurrentOrchestratorError("identity/training policy cannot be blank")
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
        identity_policy=resolved_identity_policy,
        identity_snapshot=snapshots[0],
        execution_training_policy=resolved_training_policy,
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
        "identity_policy": discovery.identity_policy,
        "identity_snapshot": discovery.identity_snapshot,
        "execution_training_policy": discovery.execution_training_policy,
        "trainable_symbol_count": discovery.eligible_count,
        "trainable_symbol_hash": discovery.trainable_symbol_hash,
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
        identity_policy=discovery.identity_policy,
        identity_snapshot=discovery.identity_snapshot,
        execution_training_policy=discovery.execution_training_policy,
        trainable_symbol_count=discovery.eligible_count,
        trainable_symbol_hash=discovery.trainable_symbol_hash,
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
        self.execution_lock_path = self.run_directory / EXECUTION_LOCK_FILENAME
        self.active_workers_path = self.run_directory / ACTIVE_WORKERS_FILENAME

    def initialize(
        self,
        manifest: TrainingRunManifest,
        jobs: Sequence[TrainingJobRecord],
    ) -> None:
        if self.run_directory.exists() and any(self.run_directory.iterdir()):
            raise FileExistsError(f"training run already exists: {self.run_directory}")
        self._validate_inventory(manifest, jobs)
        for name in (
            JOBS_DIRECTORY_NAME,
            MODELS_DIRECTORY_NAME,
            CHECKPOINTS_DIRECTORY_NAME,
            VALIDATION_DIRECTORY_NAME,
            LOGS_DIRECTORY_NAME,
            WORKSPACES_DIRECTORY_NAME,
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
        self._validate_inventory(manifest, jobs)
        return tuple(sorted(jobs, key=lambda item: item.symbol))

    @staticmethod
    def _validate_inventory(
        manifest: TrainingRunManifest,
        jobs: Sequence[TrainingJobRecord],
    ) -> None:
        """Fail closed when persisted jobs no longer match run identity provenance."""

        if len(jobs) != manifest.identity_count or len({job.symbol for job in jobs}) != len(jobs):
            raise RecurrentOrchestratorError("persisted job inventory does not reconcile")
        if any(
            job.run_id != manifest.run_id
            or job.universe_version != manifest.universe_version
            or job.universe_hash != manifest.universe_hash
            for job in jobs
        ):
            raise RecurrentOrchestratorError(
                "persisted job identity hash/version is incompatible with the run"
            )
        trainable = sorted(job.symbol for job in jobs if job.trainability == "eligible")
        if len(trainable) != manifest.eligible_count:
            raise RecurrentOrchestratorError(
                "persisted trainable-symbol count is incompatible with the run"
            )
        if manifest.trainable_symbol_hash and canonical_hash(
            {"symbols": trainable}
        ) != manifest.trainable_symbol_hash:
            raise RecurrentOrchestratorError(
                "persisted trainable-symbol hash is incompatible with the run"
            )

    def resolve_artifact(self, relative_path: str) -> Path:
        value = Path(relative_path)
        if value.is_absolute() or ".." in value.parts:
            raise RecurrentOrchestratorError("job artifact path must be run-relative")
        resolved = (self.run_directory / value).resolve(strict=False)
        if self.run_directory not in resolved.parents:
            raise RecurrentOrchestratorError("job artifact escaped run directory")
        return resolved

    @contextmanager
    def execution_lock(self):
        """Prevent overlapping orchestrator invocations for one run."""

        self.run_directory.mkdir(parents=True, exist_ok=True)
        with self.execution_lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RecurrentOrchestratorError(
                    "another orchestrator invocation already owns this run"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read_active_workers(self) -> dict[str, Mapping[str, object]]:
        if not self.active_workers_path.is_file():
            return {}
        try:
            payload = json.loads(self.active_workers_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise RecurrentOrchestratorError(
                "active-worker state is unreadable"
            ) from exc
        workers = payload.get("workers") if isinstance(payload, dict) else None
        if not isinstance(workers, dict):
            raise RecurrentOrchestratorError("active-worker state is malformed")
        return {str(key): dict(value) for key, value in workers.items()}

    def write_active_workers(
        self, workers: Mapping[str, Mapping[str, object]]
    ) -> None:
        atomic_write_json(
            {
                "schema_version": "recurrent_active_workers_v1",
                "workers": {
                    key: dict(value) for key, value in sorted(workers.items())
                },
            },
            self.active_workers_path,
        )


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
    if store.active_workers_path.is_file():
        store.write_active_workers({})
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


def _execute_queued_jobs_sequential(
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
    """Run the original sequential path without acquiring the invocation lock."""

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


def _peak_parent_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _parallel_workspace(store: TrainingRunStore, job: TrainingJobRecord) -> Path:
    return (
        store.run_directory
        / WORKSPACES_DIRECTORY_NAME
        / safe_path_component(job.symbol)
        / f"attempt_{job.retry_count:03d}"
    )


def _parallel_log_path(store: TrainingRunStore, job: TrainingJobRecord) -> Path:
    return (
        store.run_directory
        / LOGS_DIRECTORY_NAME
        / safe_path_component(job.symbol)
        / f"attempt_{job.retry_count:03d}.json"
    )


def _parallel_invocation_log_path(
    store: TrainingRunStore, invocation_id: str
) -> Path:
    return store.run_directory / LOGS_DIRECTORY_NAME / "invocations" / f"{invocation_id}.json"


def _validate_worker_message(
    payload: object, *, symbol: str
) -> Mapping[str, object]:
    from .parallel_worker import PARALLEL_WORKER_PROTOCOL_VERSION

    if not isinstance(payload, Mapping):
        raise RecurrentOrchestratorError("parallel worker returned a malformed payload")
    if payload.get("protocol_version") != PARALLEL_WORKER_PROTOCOL_VERSION:
        raise RecurrentOrchestratorError("parallel worker protocol is incompatible")
    if payload.get("symbol") != symbol:
        raise RecurrentOrchestratorError("parallel worker returned another symbol")
    if payload.get("type") not in {"progress", "stage", "result"}:
        raise RecurrentOrchestratorError("parallel worker message type is invalid")
    return payload


def _validate_worker_result(
    payload: Mapping[str, object],
    *,
    validation_expected: bool,
    requested_timesteps: int,
    maximum_actual_timesteps: int,
    expected_worker_pid: int,
    expected_cpu_threads: int | None,
) -> None:
    if payload.get("type") != "result":
        raise RecurrentOrchestratorError("parallel worker result marker is missing")
    status = payload.get("status")
    if status not in {"completed", "failed", "interrupted"}:
        raise RecurrentOrchestratorError("parallel worker terminal status is invalid")
    if payload.get("test_partition_loaded") is not False:
        raise RecurrentOrchestratorError("parallel worker reported TEST access")
    timesteps = payload.get("actual_timesteps")
    duration = payload.get("duration_seconds")
    worker_pid = payload.get("worker_pid")
    if (
        isinstance(timesteps, bool)
        or not isinstance(timesteps, int)
        or timesteps < 0
    ):
        raise RecurrentOrchestratorError("parallel worker timesteps are invalid")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration < 0
        or not math.isfinite(float(duration))
    ):
        raise RecurrentOrchestratorError("parallel worker duration is invalid")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid < 1:
        raise RecurrentOrchestratorError("parallel worker PID is invalid")
    if worker_pid != expected_worker_pid:
        raise RecurrentOrchestratorError("parallel worker PID does not match its process")
    if status == "completed":
        if timesteps < requested_timesteps or timesteps > maximum_actual_timesteps:
            raise RecurrentOrchestratorError(
                "parallel worker completed outside the bounded rollout budget"
            )
        if payload.get("requested_timesteps") != requested_timesteps:
            raise RecurrentOrchestratorError(
                "parallel worker requested-timestep telemetry is inconsistent"
            )
        if not torch_devices_equivalent(payload.get("effective_device"), "cpu"):
            raise RecurrentOrchestratorError(
                "parallel CPU worker reported an unexpected device"
            )
        if payload.get("model_file") != "model.zip":
            raise RecurrentOrchestratorError("parallel worker model reference is invalid")
        model_hash = payload.get("model_sha256")
        if (
            not isinstance(model_hash, str)
            or len(model_hash) != 64
            or any(value not in "0123456789abcdef" for value in model_hash)
        ):
            raise RecurrentOrchestratorError("parallel worker model hash is invalid")
        expected_validation = "validation.json" if validation_expected else None
        if payload.get("validation_file") != expected_validation:
            raise RecurrentOrchestratorError(
                "parallel worker validation reference is inconsistent"
            )
        thread_policy = payload.get("cpu_thread_policy")
        if not isinstance(thread_policy, Mapping):
            raise RecurrentOrchestratorError(
                "parallel worker CPU thread telemetry is missing"
            )
        if thread_policy.get("requested_threads_per_worker") != expected_cpu_threads:
            raise RecurrentOrchestratorError(
                "parallel worker CPU thread policy differs from its request"
            )
        if expected_cpu_threads is not None and thread_policy.get(
            "torch_intraop_threads"
        ) != expected_cpu_threads:
            raise RecurrentOrchestratorError(
                "parallel worker did not enforce the Torch CPU thread limit"
            )


def _terminate_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _write_parallel_log(
    store: TrainingRunStore,
    job: TrainingJobRecord,
    payload: Mapping[str, object],
) -> None:
    atomic_write_json(dict(payload), _parallel_log_path(store, job))


def _finalize_parallel_payload(
    store: TrainingRunStore,
    job: TrainingJobRecord,
    *,
    payload: Mapping[str, object],
    workspace: Path,
    validation_expected: bool,
    maximum_actual_timesteps: int,
    expected_worker_pid: int,
    expected_cpu_threads: int | None,
) -> TrainingJobRecord:
    """Validate/promote one worker result and perform parent-owned transitions."""

    _validate_worker_result(
        payload,
        validation_expected=validation_expected,
        requested_timesteps=job.requested_timesteps,
        maximum_actual_timesteps=maximum_actual_timesteps,
        expected_worker_pid=expected_worker_pid,
        expected_cpu_threads=expected_cpu_threads,
    )
    status = str(payload["status"])
    duration = float(payload["duration_seconds"])
    actual_timesteps = int(payload["actual_timesteps"])
    if status == "interrupted":
        current = store.read_job(job.symbol)
        if current.status not in {TRAINING, VALIDATING}:
            raise RecurrentOrchestratorError("interrupted worker job is not in flight")
        return store.update_job(
            job.symbol,
            lambda value: transition_job(
                value,
                INTERRUPTED,
                message="parallel_worker_interrupted",
                completed_timesteps=actual_timesteps,
                wall_clock_duration_seconds=duration,
                failure_error_message=str(payload.get("error") or "worker interrupted"),
            ),
        )
    if status == "failed":
        current = store.read_job(job.symbol)
        if current.status not in {TRAINING, VALIDATING}:
            raise RecurrentOrchestratorError("failed worker job is not in flight")
        return store.update_job(
            job.symbol,
            lambda value: transition_job(
                value,
                FAILED,
                message="parallel_worker_failed_without_affecting_peers",
                completed_timesteps=actual_timesteps,
                wall_clock_duration_seconds=duration,
                failure_error_message=str(payload.get("error") or "worker failed"),
            ),
        )

    model_source = workspace / "model.zip"
    if not model_source.is_file() or sha256_file(model_source) != payload["model_sha256"]:
        raise RecurrentOrchestratorError("parallel worker model artifact is missing or changed")
    relative_model = _artifact_relative_path(
        job.symbol, "model", attempt=job.retry_count
    )
    model_destination = store.resolve_artifact(relative_model)
    relative_validation: str | None = None
    validation_destination: Path | None = None
    promoted: list[Path] = []
    try:
        if model_destination.exists():
            raise FileExistsError(f"model path already exists: {model_destination}")
        model_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(model_source, model_destination)
        promoted.append(model_destination)

        current = store.read_job(job.symbol)
        if validation_expected:
            if current.status == TRAINING:
                current = store.update_job(
                    job.symbol,
                    lambda value: transition_job(
                        value,
                        VALIDATING,
                        message="parent_observed_validation_after_training",
                        completed_timesteps=actual_timesteps,
                        wall_clock_duration_seconds=duration,
                    ),
                )
            if current.status != VALIDATING:
                raise RecurrentOrchestratorError("validation state is inconsistent")
            validation_source = workspace / "validation.json"
            try:
                validation_payload = json.loads(
                    validation_source.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise RecurrentOrchestratorError(
                    "parallel validation artifact is unreadable"
                ) from exc
            if not isinstance(validation_payload, dict):
                raise RecurrentOrchestratorError(
                    "parallel validation artifact is malformed"
                )
            if validation_payload.get("test_evaluated") is True:
                raise RecurrentOrchestratorError("parallel evaluator reported TEST access")
            relative_validation = _artifact_relative_path(
                job.symbol, "validation", attempt=job.retry_count
            )
            validation_destination = store.resolve_artifact(relative_validation)
            if validation_destination.exists():
                raise FileExistsError(
                    f"validation path already exists: {validation_destination}"
                )
            validation_destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(validation_source, validation_destination)
            promoted.append(validation_destination)
            validation_status = "completed"
        else:
            if current.status != TRAINING:
                raise RecurrentOrchestratorError("training state is inconsistent")
            validation_status = (
                "not_available_train_only_contract"
                if job.validation_status == "not_available_train_only_contract"
                else "not_requested"
            )
        return store.update_job(
            job.symbol,
            lambda value: transition_job(
                value,
                COMPLETED,
                message="parallel_training_and_requested_validation_completed",
                completed_timesteps=actual_timesteps,
                wall_clock_duration_seconds=duration,
                model_path=relative_model,
                model_sha256=str(payload["model_sha256"]),
                validation_status=validation_status,
                validation_metrics_reference=relative_validation,
                failure_error_message=None,
            ),
        )
    except Exception:
        for path in promoted:
            path.unlink(missing_ok=True)
        raise


def _execute_queued_jobs_processes(
    store: TrainingRunStore,
    *,
    config: RecurrentPPOConfig,
    max_jobs: int,
    workers: int,
    cpu_threads_per_worker: int | None,
    symbols: Sequence[str] | None,
    fail_fast: bool,
    splits_dir: Path,
    metadata_loader: Callable[..., object],
    registry_path: Path,
    process_worker: Callable[[Mapping[str, object], object], None] | None,
    cancellation_requested: Callable[[], bool] | None,
    stop_after_current_requested: Callable[[], bool] | None,
    device_resolver: Callable[[str], TorchDeviceResolution],
) -> tuple[TrainingJobRecord, ...]:
    """Run bounded spawned CPU workers while the parent owns shared state."""

    from .parallel_worker import (
        CPU_THREAD_ENVIRONMENT_VARIABLES,
        PARALLEL_WORKER_PROTOCOL_VERSION,
        run_recurrent_process_worker,
    )

    if config.device != "cpu":
        raise RecurrentOrchestratorError(
            "parallel recurrent execution requires explicit device=cpu"
        )
    if workers not in SUPPORTED_PROCESS_WORKERS:
        raise ValueError("workers must be one of: 1, 2, 4")
    if workers > 1 and cpu_threads_per_worker is None:
        raise RecurrentOrchestratorError(
            "parallel CPU runs require an explicit --cpu-threads-per-worker policy"
        )
    if cpu_threads_per_worker is not None and (
        isinstance(cpu_threads_per_worker, bool) or cpu_threads_per_worker < 1
    ):
        raise ValueError("cpu_threads_per_worker must be positive")
    manifest = store.read_manifest()
    if config.total_timesteps != manifest.requested_timesteps or config.seed != manifest.seed:
        raise RecurrentOrchestratorError("runtime configuration differs from run manifest")
    if config.device != manifest.requested_device:
        raise RecurrentOrchestratorError("runtime device differs from run manifest")
    if canonical_hash(config.to_dict()) != manifest.hyperparameters_hash:
        raise RecurrentOrchestratorError("runtime hyperparameters differ from run manifest")
    resolution = device_resolver("cpu")
    validate_recurrent_device_resolution(resolution)
    if not torch_devices_equivalent(resolution.resolved_device, "cpu"):
        raise RecurrentOrchestratorError("parallel CPU resolution is not CPU")
    mark_stale_jobs(store, splits_dir=Path(splits_dir), metadata_loader=metadata_loader)
    all_jobs = store.list_jobs()
    requested_symbols = None if symbols is None else {str(value).strip() for value in symbols}
    if requested_symbols is not None:
        unknown = requested_symbols.difference(job.symbol for job in all_jobs)
        if unknown:
            raise RecurrentOrchestratorError(
                "requested symbols are absent from run: " + ", ".join(sorted(unknown))
            )
    queued = [job for job in all_jobs if job.status == QUEUED]
    if requested_symbols is not None:
        queued = [job for job in queued if job.symbol in requested_symbols]
    queued.sort(key=lambda item: item.symbol)
    selected = queued[:max_jobs]
    if not selected:
        store.write_active_workers({})
        return ()

    context = multiprocessing.get_context("spawn")
    target = process_worker or run_recurrent_process_worker
    registry_before = _file_bytes(Path(registry_path))
    pending = list(selected)
    active: dict[str, dict[str, object]] = {}
    outcomes: dict[str, TrainingJobRecord] = {}
    stop_launching = False
    interrupted_by_user = False
    invocation_error: str | None = None
    invocation_started_at = utc_now()
    invocation_wall_start = time.perf_counter()
    invocation_id = (
        f"cpu-process-{time.time_ns()}-"
        + canonical_hash(
            {
                "symbols": [job.symbol for job in selected],
                "workers": workers,
                "cpu_threads_per_worker": cpu_threads_per_worker,
            }
        )[:12]
    )
    invocation_log_path = _parallel_invocation_log_path(store, invocation_id)
    atomic_write_json(
        {
            "schema_version": "recurrent_parallel_invocation_v1",
            "invocation_id": invocation_id,
            "status": "running",
            "started_at": invocation_started_at,
            "workers": workers,
            "max_jobs": max_jobs,
            "selected_symbols": [job.symbol for job in selected],
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "requested_device": "cpu",
            "test_partition_loaded": False,
        },
        invocation_log_path,
    )

    def persist_active() -> None:
        store.write_active_workers(
            {
                symbol: {
                    "worker_pid": int(item["process"].pid),
                    "status": store.read_job(symbol).status,
                    "requested_timesteps": store.read_job(symbol).requested_timesteps,
                    "started_at": store.read_job(symbol).started_at,
                    "device": "cpu",
                }
                for symbol, item in active.items()
            }
        )

    def mark_protocol_failure(symbol: str, message: str) -> None:
        item = active[symbol]
        current = store.read_job(symbol)
        if current.status in {TRAINING, VALIDATING}:
            current = store.update_job(
                symbol,
                lambda value: transition_job(
                    value,
                    FAILED,
                    message="parallel_worker_protocol_failure",
                    wall_clock_duration_seconds=max(
                        value.wall_clock_duration_seconds,
                        time.perf_counter() - float(item["wall_start"]),
                    ),
                    failure_error_message=message,
                ),
            )
        outcomes[symbol] = current

    def launch(job: TrainingJobRecord) -> None:
        nonlocal stop_launching
        workspace = _parallel_workspace(store, job)
        if workspace.exists():
            store.update_job(
                job.symbol,
                lambda value: transition_job(
                    value,
                    TRAINING,
                    message="parallel_workspace_collision_detected",
                ),
            )
            outcomes[job.symbol] = store.update_job(
                job.symbol,
                lambda value: transition_job(
                    value,
                    FAILED,
                    message="parallel_worker_not_launched",
                    failure_error_message=(
                        f"isolated workspace already exists: {workspace}"
                    ),
                ),
            )
            if fail_fast:
                stop_launching = True
            return
        workspace.parent.mkdir(parents=True, exist_ok=True)
        training_job = store.update_job(
            job.symbol,
            lambda current: transition_job(
                current,
                TRAINING,
                message="parallel_process_training_started",
            ),
        )
        training_job = store.update_job(
            job.symbol,
            lambda current: replace(
                current,
                effective_device="cpu",
                device_name=resolution.device_name,
                updated_at=utc_now(),
            ),
        )
        validation_expected = bool(
            manifest.validation_enabled
            and training_job.validation_status
            != "not_available_train_only_contract"
        )
        receive_connection, send_connection = context.Pipe(duplex=False)
        request = {
            "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
            "symbol": job.symbol,
            "config": asdict(config),
            "requested_device": "cpu",
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "splits_dir": str(Path(splits_dir).resolve(strict=False)),
            "workspace": str(workspace),
            "validation_enabled": validation_expected,
            "test_partition_loaded": False,
        }
        process = context.Process(
            target=target,
            args=(request, send_connection),
            name=f"rppo-{safe_path_component(job.symbol)}",
        )
        inherited_thread_environment = {
            name: os.environ.get(name)
            for name in CPU_THREAD_ENVIRONMENT_VARIABLES
        }
        try:
            if cpu_threads_per_worker is not None:
                for name in CPU_THREAD_ENVIRONMENT_VARIABLES:
                    os.environ[name] = str(cpu_threads_per_worker)
            process.start()
        except Exception as exc:
            send_connection.close()
            receive_connection.close()
            current = store.update_job(
                job.symbol,
                lambda value: transition_job(
                    value,
                    FAILED,
                    message="parallel_worker_launch_failed",
                    failure_error_message=f"{type(exc).__name__}: {exc}",
                ),
            )
            outcomes[job.symbol] = current
            if fail_fast:
                stop_launching = True
            return
        finally:
            for name, value in inherited_thread_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        send_connection.close()
        active[job.symbol] = {
            "job": job,
            "process": process,
            "connection": receive_connection,
            "workspace": workspace,
            "wall_start": time.perf_counter(),
            "validation_expected": validation_expected,
            "result": None,
        }
        persist_active()

    def finish(symbol: str, payload: Mapping[str, object]) -> None:
        nonlocal stop_launching
        item = active[symbol]
        job = item["job"]
        process = item["process"]
        connection = item["connection"]
        workspace = item["workspace"]
        process.join(timeout=5)
        if process.is_alive():
            _terminate_process(process)
            payload = {
                "type": "result",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "status": "failed",
                "error": "worker reported a result but did not exit",
                "actual_timesteps": int(payload.get("actual_timesteps", 0)),
                "effective_device": payload.get("effective_device"),
                "duration_seconds": time.perf_counter() - float(item["wall_start"]),
                "worker_pid": int(process.pid or 1),
                "test_partition_loaded": False,
            }
        try:
            outcome = _finalize_parallel_payload(
                store,
                job,
                payload=payload,
                workspace=workspace,
                validation_expected=bool(item["validation_expected"]),
                maximum_actual_timesteps=(
                    math.ceil(job.requested_timesteps / config.n_steps)
                    * config.n_steps
                ),
                expected_worker_pid=int(process.pid),
                expected_cpu_threads=cpu_threads_per_worker,
            )
        except Exception as exc:
            mark_protocol_failure(symbol, f"{type(exc).__name__}: {exc}")
            outcome = outcomes[symbol]
        log_payload = {
            **dict(payload),
            "parent_final_status": outcome.status,
            "parent_peak_rss_bytes": _peak_parent_rss_bytes(),
        }
        _write_parallel_log(store, job, log_payload)
        outcomes[symbol] = outcome
        if outcome.status == FAILED and fail_fast:
            stop_launching = True
        connection.close()
        if process.is_alive():
            _terminate_process(process)
        try:
            process.close()
        except ValueError:
            pass
        shutil.rmtree(workspace, ignore_errors=True)
        del active[symbol]
        persist_active()

    store.write_active_workers({})
    try:
        while pending or active:
            if cancellation_requested is not None and cancellation_requested():
                raise KeyboardInterrupt
            if (
                stop_after_current_requested is not None
                and stop_after_current_requested()
            ):
                stop_launching = True
            while pending and len(active) < workers and not stop_launching:
                launch(pending.pop(0))
            if not active:
                break
            connections = [item["connection"] for item in active.values()]
            ready = wait_for_connections(connections, timeout=0.1)
            for connection in ready:
                symbol = next(
                    key
                    for key, item in active.items()
                    if item["connection"] is connection
                )
                try:
                    raw = connection.recv()
                    payload = _validate_worker_message(raw, symbol=symbol)
                    message_type = payload["type"]
                    if message_type == "progress":
                        timesteps = payload.get("actual_timesteps")
                        elapsed = payload.get("elapsed_seconds")
                        if (
                            isinstance(timesteps, bool)
                            or not isinstance(timesteps, int)
                            or timesteps < 0
                            or timesteps
                            > math.ceil(
                                store.read_job(symbol).requested_timesteps
                                / config.n_steps
                            )
                            * config.n_steps
                            or not isinstance(elapsed, (int, float))
                            or isinstance(elapsed, bool)
                            or float(elapsed) < 0
                            or not math.isfinite(float(elapsed))
                        ):
                            raise RecurrentOrchestratorError(
                                "parallel progress payload is invalid"
                            )
                        store.update_job(
                            symbol,
                            lambda current: replace(
                                current,
                                completed_timesteps=max(
                                    current.completed_timesteps, timesteps
                                ),
                                updated_at=str(payload.get("timestamp") or utc_now()),
                                wall_clock_duration_seconds=max(
                                    current.wall_clock_duration_seconds,
                                    float(elapsed),
                                ),
                            ),
                        )
                    elif message_type == "stage":
                        if payload.get("stage") != "validating" or not bool(
                            active[symbol]["validation_expected"]
                        ):
                            raise RecurrentOrchestratorError(
                                "parallel worker stage is invalid"
                            )
                        stage_timesteps = payload.get("actual_timesteps")
                        stage_elapsed = payload.get("elapsed_seconds")
                        if (
                            isinstance(stage_timesteps, bool)
                            or not isinstance(stage_timesteps, int)
                            or stage_timesteps
                            < store.read_job(symbol).requested_timesteps
                            or stage_timesteps
                            > math.ceil(
                                store.read_job(symbol).requested_timesteps
                                / config.n_steps
                            )
                            * config.n_steps
                            or isinstance(stage_elapsed, bool)
                            or not isinstance(stage_elapsed, (int, float))
                            or float(stage_elapsed) < 0
                            or not math.isfinite(float(stage_elapsed))
                        ):
                            raise RecurrentOrchestratorError(
                                "parallel validation-stage telemetry is invalid"
                            )
                        current = store.read_job(symbol)
                        if current.status == TRAINING:
                            store.update_job(
                                symbol,
                                lambda value: transition_job(
                                    value,
                                    VALIDATING,
                                    message="parallel_validation_started_after_training",
                                    completed_timesteps=int(
                                        stage_timesteps
                                    ),
                                    wall_clock_duration_seconds=float(
                                        stage_elapsed
                                    ),
                                    validation_status="running",
                                ),
                            )
                            persist_active()
                    else:
                        finish(symbol, payload)
                except EOFError:
                    process = active[symbol]["process"]
                    process.join(timeout=1)
                    synthetic = {
                        "type": "result",
                        "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                        "symbol": symbol,
                        "status": "failed",
                        "error": f"worker exited without a result (exit={process.exitcode})",
                        "actual_timesteps": store.read_job(symbol).completed_timesteps,
                        "effective_device": None,
                        "duration_seconds": time.perf_counter()
                        - float(active[symbol]["wall_start"]),
                        "worker_pid": int(process.pid or 1),
                        "test_partition_loaded": False,
                    }
                    finish(symbol, synthetic)
                except Exception as exc:
                    process = active[symbol]["process"]
                    _terminate_process(process)
                    mark_protocol_failure(symbol, f"{type(exc).__name__}: {exc}")
                    _write_parallel_log(
                        store,
                        active[symbol]["job"],
                        {
                            "type": "protocol_failure",
                            "symbol": symbol,
                            "error": f"{type(exc).__name__}: {exc}",
                            "parent_final_status": outcomes[symbol].status,
                        },
                    )
                    active[symbol]["connection"].close()
                    shutil.rmtree(active[symbol]["workspace"], ignore_errors=True)
                    del active[symbol]
                    persist_active()
                    if fail_fast:
                        stop_launching = True
            # A child that closed unexpectedly may need one final pipe read; if
            # no data remain, the next wait iteration yields EOF deterministically.
    except KeyboardInterrupt:
        stop_launching = True
        interrupted_by_user = True
        for symbol, item in list(active.items()):
            _terminate_process(item["process"])
            current = store.read_job(symbol)
            if current.status in {TRAINING, VALIDATING}:
                current = store.update_job(
                    symbol,
                    lambda value: transition_job(
                        value,
                        INTERRUPTED,
                        message="parallel_orchestrator_keyboard_interrupt",
                        wall_clock_duration_seconds=max(
                            value.wall_clock_duration_seconds,
                            time.perf_counter() - float(item["wall_start"]),
                        ),
                        failure_error_message=(
                            "Interrupted without optimizer checkpoint; explicit "
                            "restart from zero is required."
                        ),
                    ),
                )
            outcomes[symbol] = current
            item["connection"].close()
            shutil.rmtree(item["workspace"], ignore_errors=True)
            del active[symbol]
        persist_active()
    except BaseException as exc:
        invocation_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        had_active_workers = bool(active)
        for item in active.values():
            _terminate_process(item["process"])
            item["connection"].close()
            shutil.rmtree(item["workspace"], ignore_errors=True)
        active.clear()
        if had_active_workers:
            recover_interrupted_jobs(store)
        store.write_active_workers({})
        registry_changed = _file_bytes(Path(registry_path)) != registry_before
        final_jobs = {
            job.symbol: store.read_job(job.symbol) for job in selected
        }
        statuses = {symbol: job.status for symbol, job in final_jobs.items()}
        if interrupted_by_user:
            invocation_status = "interrupted"
        elif invocation_error is not None or registry_changed:
            invocation_status = "failed"
        elif any(status == FAILED for status in statuses.values()):
            invocation_status = "completed_with_failures"
        else:
            invocation_status = "completed"
        atomic_write_json(
            {
                "schema_version": "recurrent_parallel_invocation_v1",
                "invocation_id": invocation_id,
                "status": invocation_status,
                "started_at": invocation_started_at,
                "completed_at": utc_now(),
                "duration_seconds": time.perf_counter() - invocation_wall_start,
                "workers": workers,
                "max_jobs": max_jobs,
                "selected_symbols": [job.symbol for job in selected],
                "cpu_threads_per_worker": cpu_threads_per_worker,
                "requested_device": "cpu",
                "job_statuses": statuses,
                "error": invocation_error,
                "registry_unchanged": not registry_changed,
                "test_partition_loaded": False,
            },
            invocation_log_path,
        )
        if registry_changed:
            raise RecurrentOrchestratorError("model registry changed during run")
    return tuple(outcomes[job.symbol] for job in selected if job.symbol in outcomes)


def execute_queued_jobs(
    store: TrainingRunStore,
    *,
    config: RecurrentPPOConfig,
    max_jobs: int = 1,
    workers: int = 1,
    cpu_threads_per_worker: int | None = None,
    symbols: Sequence[str] | None = None,
    fail_fast: bool = False,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    trainer: Callable[..., RecurrentPPOTrainingResult] = train_recurrent_single_symbol,
    evaluator: Callable[..., object] = evaluate_recurrent_on_validation,
    device_resolver: Callable[[str], TorchDeviceResolution] = resolve_torch_device,
    metadata_loader: Callable[..., object] = load_training_recurrent_contract_metadata,
    registry_path: Path = MODEL_REGISTRY_PATH,
    process_worker: Callable[[Mapping[str, object], object], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    stop_after_current_requested: Callable[[], bool] | None = None,
    force_process_workers: bool = False,
) -> tuple[TrainingJobRecord, ...]:
    """Process at most ``max_jobs`` with a distinct bounded worker count."""

    if max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    if workers not in SUPPORTED_PROCESS_WORKERS:
        raise ValueError("workers must be one of: 1, 2, 4")
    with store.execution_lock():
        if workers == 1 and not force_process_workers:
            if cpu_threads_per_worker is not None:
                raise RecurrentOrchestratorError(
                    "CPU thread policy applies to spawned workers; omit it for "
                    "the legacy sequential path"
                )
            return _execute_queued_jobs_sequential(
                store,
                config=config,
                max_jobs=max_jobs,
                symbols=symbols,
                fail_fast=fail_fast,
                splits_dir=splits_dir,
                trainer=trainer,
                evaluator=evaluator,
                device_resolver=device_resolver,
                metadata_loader=metadata_loader,
                registry_path=registry_path,
            )
        if trainer is not train_recurrent_single_symbol or evaluator is not evaluate_recurrent_on_validation:
            raise RecurrentOrchestratorError(
                "spawned workers use the production trainer/evaluator; inject a "
                "bounded process_worker for offline tests"
            )
        return _execute_queued_jobs_processes(
            store,
            config=config,
            max_jobs=max_jobs,
            workers=workers,
            cpu_threads_per_worker=cpu_threads_per_worker,
            symbols=symbols,
            fail_fast=fail_fast,
            splits_dir=Path(splits_dir),
            metadata_loader=metadata_loader,
            registry_path=Path(registry_path),
            process_worker=process_worker,
            cancellation_requested=cancellation_requested,
            stop_after_current_requested=stop_after_current_requested,
            device_resolver=device_resolver,
        )


def training_status_table(store: TrainingRunStore) -> pd.DataFrame:
    """Return deterministic download-style job progress for every identity."""

    active_workers = store.read_active_workers()
    rows = []
    for job in store.list_jobs():
        row = progress_snapshot(job)
        active = active_workers.get(job.symbol, {})
        row["worker_process_id"] = active.get("worker_pid")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(
        drop=True
    )


def training_status_summary(store: TrainingRunStore) -> dict[str, int]:
    """Return reconciled global counts without changing persisted state."""

    jobs = store.list_jobs()
    counts = pd.Series([job.status for job in jobs], dtype="string").value_counts()
    active = int(counts.get(TRAINING, 0) + counts.get(VALIDATING, 0))
    return {
        "total": len(jobs),
        "eligible": sum(job.trainability == "eligible" for job in jobs),
        "queued": int(counts.get(QUEUED, 0)),
        "active": active,
        "completed": int(counts.get(COMPLETED, 0)),
        "failed": int(counts.get(FAILED, 0)),
        "interrupted": int(counts.get(INTERRUPTED, 0)),
    }


def _print_discovery(discovery: RecurrentUniverseDiscovery) -> None:
    print(f"Universe: {discovery.universe_version}")
    print(f"Universe hash: {discovery.universe_hash}")
    print(f"Identity symbols: {discovery.identity_count}")
    for category, count in sorted(discovery.category_counts.items()):
        print(f"{category}: {count}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track and run bounded recurrent PPO symbol jobs"
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        choices=SUPPORTED_PROCESS_WORKERS,
        default=1,
        help="Concurrent processes; --max-jobs remains the invocation job limit.",
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        help="Explicit Torch/BLAS thread limit for spawned CPU workers.",
    )
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
        print(json.dumps(training_status_summary(store), sort_keys=True))
        print(training_status_table(store).to_string(index=False))
        return 0
    outcomes = execute_queued_jobs(
        store,
        config=config,
        max_jobs=args.max_jobs,
        workers=args.workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
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
    "SUPPORTED_PROCESS_WORKERS",
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
    "training_status_summary",
    "validate_recurrent_device_resolution",
]
