"""Persistent per-symbol RecurrentPPO job state and transition contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from feature_engineering.storage import safe_path_component


TRAINING_JOB_SCHEMA_VERSION = "recurrent_training_job_v1"
TRAINING_RUN_SCHEMA_VERSION = "recurrent_training_run_v1"
TRAINING_ORCHESTRATOR_VERSION = "recurrent_multi_symbol_orchestrator_v1"

PENDING = "PENDING"
QUEUED = "QUEUED"
TRAINING = "TRAINING"
VALIDATING = "VALIDATING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
INTERRUPTED = "INTERRUPTED"
INELIGIBLE = "INELIGIBLE"
STALE = "STALE"

JOB_STATUSES = frozenset(
    {
        PENDING,
        QUEUED,
        TRAINING,
        VALIDATING,
        COMPLETED,
        FAILED,
        INTERRUPTED,
        INELIGIBLE,
        STALE,
    }
)
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, INTERRUPTED, INELIGIBLE, STALE})
IN_FLIGHT_STATUSES = frozenset({TRAINING, VALIDATING})

LEGAL_TRANSITIONS: Mapping[str, frozenset[str]] = {
    PENDING: frozenset({QUEUED, STALE}),
    QUEUED: frozenset({TRAINING, INTERRUPTED, STALE}),
    TRAINING: frozenset({VALIDATING, COMPLETED, FAILED, INTERRUPTED}),
    VALIDATING: frozenset({COMPLETED, FAILED, INTERRUPTED}),
    COMPLETED: frozenset({STALE}),
    FAILED: frozenset({QUEUED, STALE}),
    INTERRUPTED: frozenset({QUEUED, STALE}),
    STALE: frozenset({QUEUED}),
    INELIGIBLE: frozenset(),
}


class TrainingJobStateError(ValueError):
    """Raised when persisted training state would become ambiguous."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingJobRecord:
    """Complete persistent state for one identity-universe symbol."""

    schema_version: str
    job_id: str
    run_id: str
    symbol: str
    trainability: str
    trainability_reason: str
    agent_version: str
    environment_version: str
    data_contract_version: str
    feature_version: str
    universe_version: str
    universe_hash: str
    source_data_hash: str
    requested_timesteps: int
    completed_timesteps: int
    seed: int
    hyperparameters_hash: str
    requested_device: str
    effective_device: str | None
    device_name: str | None
    status: str
    created_at: str
    started_at: str | None
    updated_at: str
    completed_at: str | None
    wall_clock_duration_seconds: float
    checkpoint_path: str | None
    model_path: str | None
    model_sha256: str | None
    validation_status: str
    validation_metrics_reference: str | None
    failure_error_message: str | None
    retry_count: int
    state_history: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_JOB_SCHEMA_VERSION:
            raise TrainingJobStateError("training job schema version is incompatible")
        if self.status not in JOB_STATUSES:
            raise TrainingJobStateError(f"unknown job status: {self.status}")
        if not self.symbol.strip() or safe_path_component(self.symbol) != self.symbol:
            raise TrainingJobStateError("job symbol is not a safe canonical component")
        if min(
            self.requested_timesteps,
            self.completed_timesteps,
            self.seed,
            self.retry_count,
        ) < 0 or self.wall_clock_duration_seconds < 0:
            raise TrainingJobStateError("job counters/duration cannot be negative")
        if self.requested_timesteps < 1:
            raise TrainingJobStateError("requested timesteps must be positive")
        if self.trainability not in {"eligible", "ineligible"}:
            raise TrainingJobStateError("trainability must be eligible or ineligible")
        if (self.status == INELIGIBLE) != (self.trainability == "ineligible"):
            raise TrainingJobStateError("INELIGIBLE status must match trainability")
        for label, value in (
            ("universe_hash", self.universe_hash),
            ("hyperparameters_hash", self.hyperparameters_hash),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise TrainingJobStateError(f"{label} must be a lowercase SHA-256")
        if self.source_data_hash and (
            len(self.source_data_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.source_data_hash)
        ):
            raise TrainingJobStateError("source_data_hash must be blank or SHA-256")
        if self.status == COMPLETED and not self.model_path:
            raise TrainingJobStateError("completed job must reference an isolated model")

    @property
    def progress_percent(self) -> float:
        return min(100.0, 100.0 * self.completed_timesteps / self.requested_timesteps)

    @property
    def model_available(self) -> bool:
        return bool(self.model_path and self.model_sha256)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state_history"] = [dict(item) for item in self.state_history]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainingJobRecord":
        values = dict(payload)
        history = values.get("state_history", ())
        if not isinstance(history, (list, tuple)):
            raise TrainingJobStateError("state_history must be a sequence")
        values["state_history"] = tuple(dict(item) for item in history)
        return cls(**values)


@dataclass(frozen=True)
class TrainingRunManifest:
    """Stable run fingerprint plus its non-identity creation timestamp."""

    schema_version: str
    orchestrator_version: str
    run_id: str
    run_fingerprint: str
    universe_version: str
    universe_hash: str
    identity_count: int
    eligible_count: int
    ineligible_count: int
    agent_version: str
    requested_timesteps: int
    seed: int
    requested_device: str
    hyperparameters_hash: str
    source_inventory_hash: str
    validation_enabled: bool
    worker_limit: int
    resume_capability: str
    created_at: str
    test_partition_loaded: bool = False
    identity_policy: str = ""
    identity_snapshot: str = ""
    execution_training_policy: str = ""
    trainable_symbol_count: int = 0
    trainable_symbol_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RUN_SCHEMA_VERSION:
            raise TrainingJobStateError("training run schema version is incompatible")
        if self.orchestrator_version != TRAINING_ORCHESTRATOR_VERSION:
            raise TrainingJobStateError("training orchestrator version is incompatible")
        if self.identity_count != self.eligible_count + self.ineligible_count:
            raise TrainingJobStateError("run identity accounting does not reconcile")
        if self.worker_limit != 1:
            raise TrainingJobStateError(
                "v1 run identity retains worker_limit=1; bounded runtime CPU "
                "concurrency is invocation metadata and does not change model identity"
            )
        if self.test_partition_loaded:
            raise TrainingJobStateError("TEST cannot enter orchestration metadata")
        provenance = (
            self.identity_policy,
            self.identity_snapshot,
            self.execution_training_policy,
            self.trainable_symbol_hash,
        )
        if any(provenance) or self.trainable_symbol_count:
            if not all(provenance):
                raise TrainingJobStateError(
                    "training run identity provenance must be complete"
                )
            if self.trainable_symbol_count != self.eligible_count:
                raise TrainingJobStateError(
                    "trainable-symbol count must match eligible job count"
                )
            if len(self.trainable_symbol_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.trainable_symbol_hash
            ):
                raise TrainingJobStateError(
                    "trainable_symbol_hash must be a lowercase SHA-256"
                )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainingRunManifest":
        return cls(**dict(payload))


def transition_job(
    job: TrainingJobRecord,
    target: str,
    *,
    timestamp: str | None = None,
    message: str | None = None,
    **changes: object,
) -> TrainingJobRecord:
    """Apply one explicit legal state transition and append its audit event."""

    if target not in LEGAL_TRANSITIONS.get(job.status, frozenset()):
        raise TrainingJobStateError(
            f"illegal training-job transition: {job.status} -> {target}"
        )
    when = timestamp or utc_now()
    event = {
        "from": job.status,
        "to": target,
        "timestamp": when,
        "message": message or "",
    }
    values: dict[str, object] = {
        "status": target,
        "updated_at": when,
        "state_history": (*job.state_history, event),
    }
    if target == TRAINING and job.started_at is None:
        values["started_at"] = when
    if target in TERMINAL_STATUSES:
        values["completed_at"] = when
    if target == QUEUED:
        values["completed_at"] = None
        if job.status in {FAILED, INTERRUPTED, STALE}:
            values.update(
                completed_timesteps=0,
                started_at=None,
                wall_clock_duration_seconds=0.0,
                effective_device=None,
                device_name=None,
                checkpoint_path=None,
                model_path=None,
                model_sha256=None,
                validation_status="not_requested",
                validation_metrics_reference=None,
                failure_error_message=None,
                retry_count=job.retry_count + 1,
            )
    values.update(changes)
    return replace(job, **values)


def progress_snapshot(
    job: TrainingJobRecord,
    *,
    observed_elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Return display-safe progress and an ETA only from observed progress."""

    elapsed = (
        job.wall_clock_duration_seconds
        if observed_elapsed_seconds is None
        else max(0.0, float(observed_elapsed_seconds))
    )
    eta: float | None = None
    if (
        job.status == TRAINING
        and job.completed_timesteps > 0
        and elapsed > 0
        and job.completed_timesteps < job.requested_timesteps
    ):
        remaining = job.requested_timesteps - job.completed_timesteps
        eta = elapsed * remaining / job.completed_timesteps
        if not math.isfinite(eta):
            eta = None
    return {
        "symbol": job.symbol,
        "status": job.status,
        "completed_timesteps": job.completed_timesteps,
        "requested_timesteps": job.requested_timesteps,
        "progress_percent": job.progress_percent,
        "effective_device": job.effective_device,
        "started_at": job.started_at,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": eta,
        "validation_status": job.validation_status,
        "model_available": job.model_available,
        "error_summary": job.failure_error_message,
    }


__all__ = [
    "COMPLETED",
    "FAILED",
    "INELIGIBLE",
    "INTERRUPTED",
    "JOB_STATUSES",
    "LEGAL_TRANSITIONS",
    "PENDING",
    "QUEUED",
    "STALE",
    "TRAINING",
    "TRAINING_JOB_SCHEMA_VERSION",
    "TRAINING_ORCHESTRATOR_VERSION",
    "TRAINING_RUN_SCHEMA_VERSION",
    "VALIDATING",
    "TrainingJobRecord",
    "TrainingJobStateError",
    "TrainingRunManifest",
    "canonical_hash",
    "progress_snapshot",
    "transition_job",
    "utc_now",
]
