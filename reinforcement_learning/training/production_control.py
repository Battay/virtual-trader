"""Persistent control-plane services for full recurrent production training.

Streamlit and other clients call this module to read state or issue explicit
commands.  The detached controller owns the long-running orchestration call;
browser sessions never own worker lifetimes and never receive TEST data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
from numbers import Integral
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import pandas as pd

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    PROJECT_ROOT,
    PROCESSED_SPLITS_DIR,
    TRAINING_RUNS_DIR,
)
from data_pipeline.src.identity_universe_policy import (
    FROZEN_RESEARCH_IDENTITY_COUNT,
    FROZEN_RESEARCH_IDENTITY_MANIFEST_VERSION,
    FROZEN_RESEARCH_IDENTITY_SNAPSHOT,
    FROZEN_RESEARCH_TRAINING_POLICY,
    FROZEN_RESEARCH_UNIVERSE,
    FROZEN_RESEARCH_UNIVERSE_HASH,
    load_frozen_research_identity,
)
from feature_engineering.storage import atomic_write_json
from reinforcement_learning.canonical_recurrent_artifacts import (
    load_training_recurrent_contract_metadata,
)
from reinforcement_learning.environments.config import ENVIRONMENT_VERSION
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.model_management.registry import (
    ModelRegistryError,
    load_model_registry,
)

from .job_state import (
    COMPLETED,
    FAILED,
    INELIGIBLE,
    INTERRUPTED,
    QUEUED,
    STALE,
    TRAINING,
    VALIDATING,
    TrainingJobRecord,
    TrainingRunManifest,
    utc_now,
)
from .recurrent_config import RECURRENT_PPO_CONFIG_VERSION, RecurrentPPOConfig
from .recurrent_orchestrator import (
    RecurrentOrchestratorError,
    TrainingRunStore,
    create_training_run,
    discover_recurrent_training_universe,
    execute_queued_jobs,
    explicitly_requeue_job,
    recover_interrupted_jobs,
)
from .recurrent_trainer import RECURRENT_TRAINER_VERSION


PRODUCTION_CONTROL_VERSION = "recurrent_production_control_v1"
PRODUCTION_RUN_KIND = "FULL_PRODUCTION"
BENCHMARK_RUN_KIND = "BENCHMARK"
SMOKE_RUN_KIND = "SMOKE"
LEGACY_RUN_KIND = "LEGACY"
PRODUCTION_TIMESTEPS = 100_000
PRODUCTION_SEED = 42
PRODUCTION_DEVICE = "cpu"
PRODUCTION_WORKERS = 4
PRODUCTION_CPU_THREADS_PER_WORKER = 2
PRODUCTION_TRAINABLE_COUNT = 435
PRODUCTION_EXCLUDED_COUNT = 73
QUALIFIED_AGENTS_PER_HOUR = 46.81
CONTROLLER_STATE_FILENAME = "production_controller.json"
CONTROLLER_LOCK_FILENAME = ".production_controller.lock"
STOP_AFTER_CURRENT_FILENAME = "stop_after_current.json"
CONTROLLER_LOG_FILENAME = "production_controller.log"
CONTROLLER_ACTIVE_STATES = frozenset(
    {"STARTING", "RUNNING", "STOP_AFTER_CURRENT_REQUESTED", "INTERRUPT_REQUESTED"}
)
RUNNING_STATUS = "RUNNING"
COMPLETED_STATUS = "COMPLETED"
STOPPED_AFTER_CURRENT_STATUS = "STOPPED_AFTER_CURRENT"
STOPPING_AFTER_CURRENT_STATUS = "STOPPING_AFTER_CURRENT"
INTERRUPTED_STATUS = "INTERRUPTED"
FAILED_STATUS = "FAILED"
PAUSED_STATUS = "PAUSED"
BLOCKED_STATUS = "BLOCKED"
NOT_STARTED_STATUS = "NOT_STARTED"


class ProductionControlError(RuntimeError):
    """Raised when a production command would violate the frozen run contract."""


@dataclass(frozen=True)
class ProductionPlan:
    control_version: str
    identity_policy: str
    identity_snapshot: str
    frozen_universe_version: str
    universe_version: str
    universe_hash: str
    identity_count: int
    trainable_count: int
    excluded_count: int
    trainable_symbol_hash: str
    execution_training_policy: str
    algorithm: str
    policy: str
    requested_timesteps: int
    seed: int
    requested_device: str
    worker_count: int
    cpu_threads_per_worker: int
    recurrent_config_version: str
    environment_version: str
    trainer_version: str
    validation_policy: str
    test_status: str
    qualified_agents_per_hour: float


@dataclass(frozen=True)
class ControllerStatus:
    state: str
    pid: int | None
    alive: bool
    started_at: str | None
    updated_at: str | None
    completed_at: str | None
    message: str
    log_path: str | None


@dataclass(frozen=True)
class RunCatalogEntry:
    run_id: str
    run_directory: Path
    run_kind: str
    created_at: str
    status: str
    identity_count: int
    eligible_count: int
    universe_hash: str
    selected_count: int | None = None
    selected_symbol_hash: str | None = None


@dataclass(frozen=True)
class RunProgress:
    system_status: str
    eligible: int
    completed: int
    active: int
    training: int
    validating: int
    queued: int
    failed: int
    interrupted: int
    stale: int
    ineligible: int
    completed_training_timesteps: int
    requested_training_timesteps: int
    progress_percent: float
    elapsed_seconds: float | None
    agents_per_hour: float | None
    estimated_remaining_seconds: float | None


@dataclass(frozen=True)
class RunSnapshot:
    store: TrainingRunStore
    manifest: TrainingRunManifest
    run_kind: str
    controller: ControllerStatus
    progress: RunProgress
    jobs: pd.DataFrame


@dataclass(frozen=True)
class TrainingProgressValue:
    """Validated persisted training work for one eligible symbol."""

    actual_timesteps: int
    requested_timesteps: int
    clamped_timesteps: int
    fraction: float
    percent: float


@dataclass(frozen=True)
class AggregateTrainingProgress:
    """Aggregate eligible-only training work for a persisted run."""

    eligible_jobs: int
    completed_timesteps: int
    requested_timesteps: int
    fraction: float
    percent: float


def calculate_training_progress(
    actual_timesteps: object, requested_timesteps: object
) -> TrainingProgressValue:
    """Validate and clamp one persisted actual/requested timestep pair."""

    if (
        isinstance(actual_timesteps, bool)
        or not isinstance(actual_timesteps, Integral)
        or isinstance(requested_timesteps, bool)
        or not isinstance(requested_timesteps, Integral)
    ):
        raise ProductionControlError("training progress fields must be integers")
    actual = int(actual_timesteps)
    requested = int(requested_timesteps)
    if requested <= 0:
        raise ProductionControlError("requested training timesteps must be positive")
    if actual < 0:
        raise ProductionControlError("actual training timesteps cannot be negative")
    clamped = min(actual, requested)
    fraction = clamped / requested
    return TrainingProgressValue(
        actual_timesteps=actual,
        requested_timesteps=requested,
        clamped_timesteps=clamped,
        fraction=fraction,
        percent=fraction * 100.0,
    )


def job_training_progress(job: TrainingJobRecord) -> TrainingProgressValue:
    """Return one eligible job's honest persisted training progress."""

    if job.trainability != "eligible" or job.status == INELIGIBLE:
        raise ProductionControlError("ineligible jobs have no training progress")
    progress = calculate_training_progress(
        job.completed_timesteps, job.requested_timesteps
    )
    if job.status in {VALIDATING, COMPLETED} and progress.fraction < 1.0:
        raise ProductionControlError(
            f"{job.symbol} is {job.status} before its training budget is complete"
        )
    return progress


def aggregate_training_progress(
    jobs: Sequence[TrainingJobRecord],
) -> AggregateTrainingProgress:
    """Aggregate persisted work across eligible jobs only."""

    eligible_jobs = [job for job in jobs if job.trainability == "eligible"]
    if not eligible_jobs:
        return AggregateTrainingProgress(0, 0, 0, 0.0, 0.0)
    values = [job_training_progress(job) for job in eligible_jobs]
    completed = sum(value.clamped_timesteps for value in values)
    requested = sum(value.requested_timesteps for value in values)
    if requested <= 0:
        raise ProductionControlError("eligible run has no positive training budget")
    fraction = completed / requested
    return AggregateTrainingProgress(
        eligible_jobs=len(values),
        completed_timesteps=completed,
        requested_timesteps=requested,
        fraction=fraction,
        percent=fraction * 100.0,
    )


def production_config() -> RecurrentPPOConfig:
    """Return the immutable qualified recurrent configuration."""

    return RecurrentPPOConfig(
        total_timesteps=PRODUCTION_TIMESTEPS,
        seed=PRODUCTION_SEED,
        device=PRODUCTION_DEVICE,
    )


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _frozen_discovery():
    identity = load_frozen_research_identity()
    discovery = discover_recurrent_training_universe(
        identity=identity,
        identity_policy=FROZEN_RESEARCH_UNIVERSE,
        execution_training_policy=FROZEN_RESEARCH_TRAINING_POLICY,
    )
    if (
        discovery.identity_count != FROZEN_RESEARCH_IDENTITY_COUNT
        or discovery.eligible_count != PRODUCTION_TRAINABLE_COUNT
        or discovery.ineligible_count != PRODUCTION_EXCLUDED_COUNT
        or discovery.universe_hash != FROZEN_RESEARCH_UNIVERSE_HASH
    ):
        raise ProductionControlError("frozen production discovery no longer reconciles")
    return discovery


def production_plan() -> ProductionPlan:
    """Return immutable display metadata without reading any market partition."""

    return ProductionPlan(
        control_version=PRODUCTION_CONTROL_VERSION,
        identity_policy=FROZEN_RESEARCH_UNIVERSE,
        identity_snapshot=FROZEN_RESEARCH_IDENTITY_SNAPSHOT,
        frozen_universe_version=FROZEN_RESEARCH_IDENTITY_MANIFEST_VERSION,
        universe_version="current_common_equity_universe_v1",
        universe_hash=FROZEN_RESEARCH_UNIVERSE_HASH,
        identity_count=FROZEN_RESEARCH_IDENTITY_COUNT,
        trainable_count=PRODUCTION_TRAINABLE_COUNT,
        excluded_count=PRODUCTION_EXCLUDED_COUNT,
        trainable_symbol_hash=(
            "44efa67c6c1aa5ac27d559f85835493206617a63fa24c25648e2da0d9f38a4a2"
        ),
        execution_training_policy=FROZEN_RESEARCH_TRAINING_POLICY,
        algorithm="RecurrentPPO",
        policy="MlpLstmPolicy",
        requested_timesteps=PRODUCTION_TIMESTEPS,
        seed=PRODUCTION_SEED,
        requested_device=PRODUCTION_DEVICE,
        worker_count=PRODUCTION_WORKERS,
        cpu_threads_per_worker=PRODUCTION_CPU_THREADS_PER_WORKER,
        recurrent_config_version=RECURRENT_PPO_CONFIG_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        trainer_version=RECURRENT_TRAINER_VERSION,
        validation_policy="VALIDATION_ONLY_AFTER_TRAIN_TEST_SEALED_V1",
        test_status="SEALED",
        qualified_agents_per_hour=QUALIFIED_AGENTS_PER_HOUR,
    )


def _validate_production_manifest(manifest: TrainingRunManifest) -> None:
    plan = production_plan()
    checks = {
        "identity policy": manifest.identity_policy == plan.identity_policy,
        "identity snapshot": manifest.identity_snapshot == plan.identity_snapshot,
        "universe hash": manifest.universe_hash == plan.universe_hash,
        "identity count": manifest.identity_count == plan.identity_count,
        "trainable count": manifest.eligible_count == plan.trainable_count,
        "excluded count": manifest.ineligible_count == plan.excluded_count,
        "trainable symbol hash": manifest.trainable_symbol_hash == plan.trainable_symbol_hash,
        "execution policy": manifest.execution_training_policy == plan.execution_training_policy,
        "timesteps": manifest.requested_timesteps == plan.requested_timesteps,
        "seed": manifest.seed == plan.seed,
        "device": manifest.requested_device == plan.requested_device,
        "trainer": manifest.agent_version == plan.trainer_version,
        "validation": manifest.validation_enabled,
        "TEST seal": not manifest.test_partition_loaded,
    }
    failed = [label for label, valid in checks.items() if not valid]
    if failed:
        raise ProductionControlError(
            "run is not the frozen production contract: " + ", ".join(failed)
        )


def prepare_production_run(
    *, runs_root: Path = TRAINING_RUNS_DIR, created_at: str | None = None
) -> tuple[TrainingRunStore, bool]:
    """Create one production job store, or return the compatible existing store."""

    discovery = _frozen_discovery()
    config = production_config()
    from .recurrent_orchestrator import build_training_run

    manifest, _ = build_training_run(
        discovery,
        config=config,
        created_at=created_at,
        validation_enabled=True,
    )
    destination = Path(runs_root).expanduser().resolve(strict=False) / manifest.run_id
    if destination.exists():
        store = TrainingRunStore(destination)
        existing = store.read_manifest()
        _validate_production_manifest(existing)
        store.list_jobs()
        return store, False
    store = create_training_run(
        discovery,
        config=config,
        runs_root=Path(runs_root),
        validation_enabled=True,
        created_at=created_at,
    )
    _validate_production_manifest(store.read_manifest())
    return store, True


def classify_run(manifest: TrainingRunManifest, run_directory: Path) -> str:
    """Distinguish production from research runs without guessing from status."""

    try:
        _validate_production_manifest(manifest)
        return PRODUCTION_RUN_KIND
    except ProductionControlError:
        from .selective_training import (
            SELECTED_RUN_KIND,
            selected_metadata_path,
            validate_selected_run,
        )

        if selected_metadata_path(run_directory).is_file():
            try:
                validate_selected_run(TrainingRunStore(run_directory))
            except (OSError, ValueError, RuntimeError):
                return LEGACY_RUN_KIND
            return SELECTED_RUN_KIND
        name = Path(run_directory).name.lower()
        if manifest.requested_timesteps <= 1_024 or "smoke" in name:
            return SMOKE_RUN_KIND
        if "benchmark" in name or manifest.requested_timesteps < PRODUCTION_TIMESTEPS:
            return BENCHMARK_RUN_KIND
        return LEGACY_RUN_KIND


def _controller_paths(run_directory: Path) -> tuple[Path, Path, Path, Path]:
    root = Path(run_directory)
    return (
        root / CONTROLLER_STATE_FILENAME,
        root / CONTROLLER_LOCK_FILENAME,
        root / STOP_AFTER_CURRENT_FILENAME,
        root / "logs" / CONTROLLER_LOG_FILENAME,
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionControlError(f"control state is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ProductionControlError(f"control state is malformed: {path}")
    return payload


def _pid_alive(pid: int | None) -> bool:
    if pid is None or isinstance(pid, bool) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def controller_status(run_directory: Path) -> ControllerStatus:
    """Inspect persisted controller state without starting or signalling anything."""

    state_path, _, _, log_path = _controller_paths(Path(run_directory))
    payload = _read_json(state_path)
    if not payload:
        return ControllerStatus(
            state="NOT_STARTED",
            pid=None,
            alive=False,
            started_at=None,
            updated_at=None,
            completed_at=None,
            message="No detached controller has been launched.",
            log_path=str(log_path.relative_to(run_directory)),
        )
    raw_pid = payload.get("pid")
    pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) else None
    state = str(payload.get("state") or "UNKNOWN")
    alive = _pid_alive(pid) if state in CONTROLLER_ACTIVE_STATES else False
    if state in CONTROLLER_ACTIVE_STATES and not alive:
        state = "DEAD_CONTROLLER"
    return ControllerStatus(
        state=state,
        pid=pid,
        alive=alive,
        started_at=str(payload.get("started_at") or "") or None,
        updated_at=str(payload.get("updated_at") or "") or None,
        completed_at=str(payload.get("completed_at") or "") or None,
        message=str(payload.get("message") or ""),
        log_path=str(payload.get("log_path") or log_path.relative_to(run_directory)),
    )


def _write_controller_state(
    store: TrainingRunStore,
    *,
    state: str,
    pid: int | None,
    message: str,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    state_path, _, _, log_path = _controller_paths(store.run_directory)
    previous = _read_json(state_path)
    atomic_write_json(
        {
            "schema_version": PRODUCTION_CONTROL_VERSION,
            "run_id": store.read_manifest().run_id,
            "state": state,
            "pid": pid,
            "started_at": started_at or previous.get("started_at"),
            "updated_at": utc_now(),
            "completed_at": completed_at,
            "message": message,
            "log_path": str(log_path.relative_to(store.run_directory)),
            "test_partition_loaded": False,
        },
        state_path,
    )


def recover_dead_controller(
    store: TrainingRunStore,
    *,
    kill: Callable[[int, int], None] = os.kill,
) -> tuple[str, ...]:
    """Terminate recorded orphans and make abandoned in-flight jobs honest."""

    status = controller_status(store.run_directory)
    if status.state != "DEAD_CONTROLLER":
        return ()
    for worker in store.read_active_workers().values():
        pid = worker.get("worker_pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            continue
        try:
            kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    recovered = recover_interrupted_jobs(store)
    _write_controller_state(
        store,
        state="RECOVERED_INTERRUPTED",
        pid=status.pid,
        message=(
            "Dead controller state recovered; abandoned active jobs require "
            "explicit restart from zero."
        ),
        started_at=status.started_at,
        completed_at=utc_now(),
    )
    return recovered


def _elapsed_bounds(jobs: Sequence[TrainingJobRecord], controller: ControllerStatus) -> tuple[float | None, float | None]:
    starts = [job.started_at for job in jobs if job.started_at]
    if not starts:
        return None, None
    start = min(datetime.fromisoformat(value) for value in starts)
    ends = [job.completed_at for job in jobs if job.completed_at]
    end = (
        datetime.now(timezone.utc)
        if controller.alive
        else max((datetime.fromisoformat(value) for value in ends), default=start)
    )
    return start.timestamp(), max(0.0, (end - start).total_seconds())


def normalize_run_status(
    jobs: Sequence[TrainingJobRecord], controller: ControllerStatus
) -> str:
    """Return one persisted execution status for catalog, history, and UI."""

    counts = pd.Series([job.status for job in jobs], dtype="string").value_counts()
    eligible = sum(job.trainability == "eligible" for job in jobs)
    completed = int(counts.get(COMPLETED, 0))
    training = int(counts.get(TRAINING, 0))
    validating = int(counts.get(VALIDATING, 0))
    failed = int(counts.get(FAILED, 0))
    interrupted = int(counts.get(INTERRUPTED, 0))
    stale = int(counts.get(STALE, 0))
    controller_state = str(controller.state or "").upper()

    if controller.alive:
        if controller_state == "STOP_AFTER_CURRENT_REQUESTED":
            return STOPPING_AFTER_CURRENT_STATUS
        if controller_state == "INTERRUPT_REQUESTED":
            return INTERRUPTED_STATUS
        return RUNNING_STATUS
    if eligible > 0 and completed == eligible:
        return COMPLETED_STATUS
    if controller_state in {
        "STOPPED_AFTER_CURRENT",
        "STOPPING_AFTER_CURRENT",
    }:
        return STOPPED_AFTER_CURRENT_STATUS
    if "INTERRUPT" in controller_state or interrupted:
        return INTERRUPTED_STATUS
    if controller_state in {"FAILED", "PARTIAL_FAILURE"} or failed:
        return FAILED_STATUS
    if stale or training or validating:
        return BLOCKED_STATUS
    if completed:
        return PAUSED_STATUS
    return NOT_STARTED_STATUS


def _historical_agents_per_hour(
    jobs: Sequence[TrainingJobRecord],
) -> float | None:
    completed_jobs = [
        job
        for job in jobs
        if job.status == COMPLETED and job.started_at and job.completed_at
    ]
    if len(completed_jobs) < 2:
        return None
    start = min(datetime.fromisoformat(str(job.started_at)) for job in completed_jobs)
    end = max(datetime.fromisoformat(str(job.completed_at)) for job in completed_jobs)
    elapsed = max(0.0, (end - start).total_seconds())
    if elapsed < 60:
        return None
    return len(completed_jobs) / (elapsed / 3600.0)


def summarize_run(
    jobs: Sequence[TrainingJobRecord], controller: ControllerStatus
) -> RunProgress:
    counts = pd.Series([job.status for job in jobs], dtype="string").value_counts()
    eligible = sum(job.trainability == "eligible" for job in jobs)
    completed = int(counts.get(COMPLETED, 0))
    training = int(counts.get(TRAINING, 0))
    validating = int(counts.get(VALIDATING, 0))
    failed = int(counts.get(FAILED, 0))
    interrupted = int(counts.get(INTERRUPTED, 0))
    stale = int(counts.get(STALE, 0))
    aggregate = aggregate_training_progress(jobs)
    _, elapsed = _elapsed_bounds(jobs, controller)
    system_status = normalize_run_status(jobs, controller)
    rate = _historical_agents_per_hour(jobs)
    eta = None
    if (
        system_status == RUNNING_STATUS
        and rate is not None
        and rate > 0
        and completed < eligible
    ):
        eta = (eligible - completed) / rate * 3600.0
    return RunProgress(
        system_status=system_status,
        eligible=eligible,
        completed=completed,
        active=training + validating,
        training=training,
        validating=validating,
        queued=int(counts.get(QUEUED, 0)),
        failed=failed,
        interrupted=interrupted,
        stale=stale,
        ineligible=int(counts.get(INELIGIBLE, 0)),
        completed_training_timesteps=aggregate.completed_timesteps,
        requested_training_timesteps=aggregate.requested_timesteps,
        progress_percent=aggregate.percent,
        elapsed_seconds=elapsed,
        agents_per_hour=rate,
        estimated_remaining_seconds=eta,
    )


def default_run_selection(
    catalog: Sequence[RunCatalogEntry], explicit_run_id: str | None
) -> str | None:
    """Preserve explicit choice, else prefer active then newest persisted run."""

    entries = tuple(catalog)
    by_id = {entry.run_id: entry for entry in entries}
    if explicit_run_id in by_id:
        return explicit_run_id
    active = [
        entry
        for entry in entries
        if entry.status in {RUNNING_STATUS, STOPPING_AFTER_CURRENT_STATUS}
    ]
    if active:
        return max(active, key=lambda item: (item.created_at, item.run_id)).run_id
    dated = [entry for entry in entries if str(entry.created_at).strip()]
    if dated:
        return max(dated, key=lambda item: (item.created_at, item.run_id)).run_id
    production = [
        entry for entry in entries if entry.run_kind == PRODUCTION_RUN_KIND
    ]
    if production:
        return sorted(production, key=lambda item: item.run_id)[0].run_id
    return entries[0].run_id if entries else None


def _identity_lookup() -> pd.DataFrame:
    return load_frozen_research_identity().loc[
        :, ["symbol", "company_name", "sector"]
    ]


def build_job_table(store: TrainingRunStore) -> pd.DataFrame:
    """Return all run identities with download-like state and no TEST fields."""

    jobs = store.list_jobs()
    active = store.read_active_workers()
    rows: list[dict[str, object]] = []
    for job in jobs:
        worker = active.get(job.symbol, {})
        progress = (
            job_training_progress(job) if job.trainability == "eligible" else None
        )
        rows.append(
            {
                "symbol": job.symbol,
                "eligibility": job.trainability,
                "state": job.status,
                "exclusion_reason": (
                    job.trainability_reason if job.trainability == "ineligible" else ""
                ),
                "requested_timesteps": job.requested_timesteps,
                "actual_timesteps": job.completed_timesteps,
                "progress_percent": progress.percent if progress else None,
                "validation_status": job.validation_status,
                "runtime_seconds": job.wall_clock_duration_seconds,
                "started_at": job.started_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
                "attempts": job.retry_count + (1 if job.started_at is not None else 0),
                "last_error": job.failure_error_message,
                "error_type": (
                    str(job.failure_error_message).split(":", 1)[0]
                    if job.failure_error_message
                    else ""
                ),
                "model_path": job.model_path,
                "model_artifact_status": (
                    "available"
                    if job.model_path
                    and store.resolve_artifact(job.model_path).is_file()
                    else "missing"
                    if job.model_path
                    else "not_created"
                ),
                "effective_device": job.effective_device,
                "worker_pid": worker.get("worker_pid"),
                "worker_slot": None,
                "cpu_threads": (
                    PRODUCTION_CPU_THREADS_PER_WORKER
                    if job.status in {TRAINING, VALIDATING}
                    else None
                ),
            }
        )
    table = pd.DataFrame(rows)
    slots = {
        symbol: index + 1
        for index, symbol in enumerate(
            sorted(table.loc[table["state"].isin({TRAINING, VALIDATING}), "symbol"])
        )
    }
    table["worker_slot"] = table["symbol"].map(slots)
    table = table.merge(_identity_lookup(), on="symbol", how="left", validate="one_to_one")
    return table.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def load_run_snapshot(
    run_directory: Path, *, recover_dead: bool = True
) -> RunSnapshot:
    store = TrainingRunStore(Path(run_directory))
    manifest = store.read_manifest()
    kind = classify_run(manifest, store.run_directory)
    controller = controller_status(store.run_directory)
    if recover_dead and controller.state == "DEAD_CONTROLLER":
        recover_dead_controller(store)
        controller = controller_status(store.run_directory)
    jobs = store.list_jobs()
    return RunSnapshot(
        store=store,
        manifest=manifest,
        run_kind=kind,
        controller=controller,
        progress=summarize_run(jobs, controller),
        jobs=build_job_table(store),
    )


def list_run_catalog(*, runs_root: Path = TRAINING_RUNS_DIR) -> tuple[RunCatalogEntry, ...]:
    root = Path(runs_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        return ()
    entries: list[RunCatalogEntry] = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        try:
            store = TrainingRunStore(manifest_path.parent)
            manifest = store.read_manifest()
            progress = summarize_run(store.list_jobs(), controller_status(store.run_directory))
        except (
            OSError,
            ValueError,
            ProductionControlError,
            RecurrentOrchestratorError,
        ):
            continue
        kind = classify_run(manifest, store.run_directory)
        selected_count = None
        selected_symbol_hash = None
        if kind == "SELECTED":
            from .selective_training import load_selected_run_metadata

            selected = load_selected_run_metadata(store.run_directory)
            selected_count = len(selected.selected_symbols)
            selected_symbol_hash = selected.selected_symbol_hash
        entries.append(
            RunCatalogEntry(
                run_id=manifest.run_id,
                run_directory=store.run_directory,
                run_kind=kind,
                created_at=manifest.created_at,
                status=progress.system_status,
                identity_count=manifest.identity_count,
                eligible_count=manifest.eligible_count,
                universe_hash=manifest.universe_hash,
                selected_count=selected_count,
                selected_symbol_hash=selected_symbol_hash,
            )
        )
    return tuple(sorted(entries, key=lambda item: (item.created_at, item.run_id), reverse=True))


def _controller_command(store: TrainingRunStore, python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "reinforcement_learning.training.production_control",
        "--execute-run",
        "--run-directory",
        str(store.run_directory),
    ]


def launch_production_controller(
    store: TrainingRunStore,
    *,
    popen: Callable[..., object] = subprocess.Popen,
    python_executable: str = sys.executable,
) -> ControllerStatus:
    """Launch one qualified full or selected controller, fail-closed."""

    manifest = store.read_manifest()
    kind = classify_run(manifest, store.run_directory)
    if kind == PRODUCTION_RUN_KIND:
        _validate_production_manifest(manifest)
    elif kind == "SELECTED":
        from .selective_training import SelectiveTrainingError, validate_selected_run

        try:
            validate_selected_run(store, executable=True)
        except SelectiveTrainingError as exc:
            raise ProductionControlError(str(exc)) from exc
    else:
        raise ProductionControlError(
            "only FULL_PRODUCTION or SELECTED runs may launch from this controller"
        )
    jobs = store.list_jobs()
    if not any(job.status == QUEUED for job in jobs):
        raise ProductionControlError("training run has no queued jobs to start")
    state_path, lock_path, stop_path, log_path = _controller_paths(store.run_directory)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            current = controller_status(store.run_directory)
            if current.alive:
                raise ProductionControlError(
                    f"training controller is already running with PID {current.pid}"
                )
            stop_path.unlink(missing_ok=True)
            _write_controller_state(
                store,
                state="LAUNCHING",
                pid=None,
                message=f"Preparing detached {kind} controller launch.",
                started_at=utc_now(),
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
            try:
                process = popen(
                    _controller_command(store, python_executable),
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log_handle.close()
            pid = getattr(process, "pid", None)
            if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
                raise ProductionControlError("detached controller did not return a PID")
            atomic_write_json(
                {
                    "schema_version": PRODUCTION_CONTROL_VERSION,
                    "run_id": manifest.run_id,
                    "state": "STARTING",
                    "pid": pid,
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                    "completed_at": None,
                    "message": f"Detached {kind} controller launched.",
                    "command": _controller_command(store, python_executable),
                    "log_path": str(log_path.relative_to(store.run_directory)),
                    "test_partition_loaded": False,
                },
                state_path,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return controller_status(store.run_directory)


def request_stop_after_current(store: TrainingRunStore) -> ControllerStatus:
    status = controller_status(store.run_directory)
    if not status.alive:
        raise ProductionControlError("no live production controller can be paused")
    _, _, stop_path, _ = _controller_paths(store.run_directory)
    atomic_write_json(
        {"requested_at": utc_now(), "mode": "stop_after_current_jobs"}, stop_path
    )
    _write_controller_state(
        store,
        state="STOP_AFTER_CURRENT_REQUESTED",
        pid=status.pid,
        message="No new jobs will launch; current jobs may finish.",
        started_at=status.started_at,
    )
    return controller_status(store.run_directory)


def request_interrupt(
    store: TrainingRunStore,
    *,
    kill: Callable[[int, int], None] = os.kill,
) -> ControllerStatus:
    status = controller_status(store.run_directory)
    if not status.alive or status.pid is None:
        raise ProductionControlError("no live production controller can be interrupted")
    kill(status.pid, signal.SIGINT)
    _write_controller_state(
        store,
        state="INTERRUPT_REQUESTED",
        pid=status.pid,
        message="SIGINT sent; active jobs will be marked INTERRUPTED safely.",
        started_at=status.started_at,
    )
    return controller_status(store.run_directory)


def requeue_jobs(
    store: TrainingRunStore,
    *,
    statuses: frozenset[str],
    symbols: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if controller_status(store.run_directory).alive:
        raise ProductionControlError("cannot requeue jobs while controller is active")
    allowed = frozenset({FAILED, INTERRUPTED})
    if not statuses or not statuses.issubset(allowed):
        raise ProductionControlError("only FAILED or INTERRUPTED jobs may be requeued")
    selected = None if symbols is None else {str(value) for value in symbols}
    jobs = [job for job in store.list_jobs() if job.status in statuses]
    if selected is not None:
        unknown = selected.difference(job.symbol for job in jobs)
        if unknown:
            raise ProductionControlError(
                "selected retry symbols are not in the requested failure state: "
                + ", ".join(sorted(unknown))
            )
        jobs = [job for job in jobs if job.symbol in selected]
    restarted = [explicitly_requeue_job(store, job.symbol).symbol for job in jobs]
    return tuple(sorted(restarted))


def latest_job_diagnostics(store: TrainingRunStore, symbol: str) -> dict[str, object]:
    job = store.read_job(symbol)
    path = (
        store.run_directory
        / "logs"
        / symbol
        / f"attempt_{job.retry_count:03d}.json"
    )
    payload = _read_json(path)
    diagnostics = payload.get("training_diagnostics")
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def load_validation_metrics(store: TrainingRunStore, symbol: str) -> dict[str, object]:
    job = store.read_job(symbol)
    if not job.validation_metrics_reference:
        return {}
    payload = _read_json(store.resolve_artifact(job.validation_metrics_reference))
    if payload.get("test_evaluated") is True:
        raise ProductionControlError("validation artifact reports TEST access")
    return payload


def symbol_contract_summary(symbol: str) -> dict[str, object]:
    """Load contract metadata only; no TRAIN/VALIDATION/TEST dataframe is opened."""

    try:
        metadata = load_training_recurrent_contract_metadata(
            symbol, splits_dir=PROCESSED_SPLITS_DIR
        )
    except (OSError, ValueError, RuntimeError):
        return {}
    return {
        "train_rows": metadata.train.rows,
        "train_start": metadata.train.start,
        "train_end": metadata.train.end,
        "recurrent_contract_version": metadata.recurrent_contract_version,
        "feature_version": metadata.feature_version,
        "environment_version": metadata.environment_version,
        "observation_count": len(metadata.observation_features),
        "contract_path": str(metadata.contract_path),
    }


def registry_view(path: Path = MODEL_REGISTRY_PATH) -> pd.DataFrame:
    """Return a read-only recurrent/legacy registry view with integrity flags."""

    try:
        registry = load_model_registry(Path(path))
    except ModelRegistryError as exc:
        raise ProductionControlError(f"model registry is invalid: {exc}") from exc
    columns = (
        "model_id",
        "model_family",
        "symbol",
        "run_id",
        "algorithm",
        "policy",
        "training_status",
        "validation_status",
        "created_at",
        "model_artifact_status",
        "metadata_integrity",
    )
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        algorithm = str(row.get("algorithm") or "")
        model_path = Path(str(row.get("model_path") or ""))
        if model_path and not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        manifest_path = Path(str(row.get("manifest_path") or ""))
        if manifest_path and not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        expected_hash = str(row.get("manifest_sha256") or "")
        manifest_ok = bool(
            manifest_path
            and manifest_path.is_file()
            and _sha256(expected_hash)
            and sha256_file(manifest_path) == expected_hash
        )
        recurrent = "recurrent" in algorithm.lower()
        rows.append(
            {
                "model_id": row.get("model_id"),
                "model_family": "RECURRENT" if recurrent else "LEGACY",
                "symbol": row.get("symbol"),
                "run_id": "",
                "algorithm": algorithm,
                "policy": "MlpLstmPolicy" if recurrent else "MlpPolicy/legacy",
                "training_status": row.get("training_status"),
                "validation_status": row.get("validation_status"),
                "created_at": row.get("created_at"),
                "model_artifact_status": (
                    "available" if model_path and model_path.is_file() else "missing"
                ),
                "metadata_integrity": "verified" if manifest_ok else "unverified",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def recent_orchestration_events(
    store: TrainingRunStore, *, limit: int = 30
) -> pd.DataFrame:
    """Return a bounded, newest-first state-event view for monitoring."""

    if limit < 1:
        raise ValueError("event limit must be positive")
    active = store.read_active_workers()
    rows: list[dict[str, object]] = []
    for job in store.list_jobs():
        for event in job.state_history:
            rows.append(
                {
                    "timestamp": event.get("timestamp"),
                    "symbol": job.symbol,
                    "event": event.get("to"),
                    "message": event.get("message"),
                    "worker_pid": active.get(job.symbol, {}).get("worker_pid"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "event", "message", "worker_pid"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["timestamp", "symbol"], ascending=[False, True], kind="mergesort")
        .head(limit)
        .reset_index(drop=True)
    )


def bounded_log_tail(path: Path, *, maximum_bytes: int = 24_000) -> str:
    if maximum_bytes < 1:
        raise ValueError("maximum log bytes must be positive")
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read(maximum_bytes).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _execute_controller(store: TrainingRunStore) -> int:
    manifest = store.read_manifest()
    kind = classify_run(manifest, store.run_directory)
    if kind == PRODUCTION_RUN_KIND:
        _validate_production_manifest(manifest)
    elif kind == "SELECTED":
        from .selective_training import SelectiveTrainingError, validate_selected_run

        try:
            validate_selected_run(store, executable=True)
        except SelectiveTrainingError as exc:
            raise ProductionControlError(str(exc)) from exc
    else:
        raise ProductionControlError("run is not authorized for detached execution")
    # The parent persists the child PID immediately after ``Popen`` returns.
    # Waiting for that hand-off prevents a fast child from writing RUNNING and
    # then being overwritten by the parent's STARTING record.
    handoff_deadline = time.monotonic() + 5.0
    while time.monotonic() < handoff_deadline:
        handoff = controller_status(store.run_directory)
        if handoff.pid == os.getpid() and handoff.state == "STARTING":
            break
        time.sleep(0.02)
    else:
        raise ProductionControlError(
            "detached controller PID hand-off did not complete safely"
        )
    status = controller_status(store.run_directory)
    _write_controller_state(
        store,
        state="RUNNING",
        pid=os.getpid(),
        message=f"{kind} recurrent orchestration is running.",
        started_at=status.started_at or utc_now(),
    )
    _, _, stop_path, _ = _controller_paths(store.run_directory)
    exit_code = 0
    try:
        execute_queued_jobs(
            store,
            config=production_config(),
            max_jobs=manifest.eligible_count,
            workers=PRODUCTION_WORKERS,
            cpu_threads_per_worker=PRODUCTION_CPU_THREADS_PER_WORKER,
            fail_fast=False,
            stop_after_current_requested=stop_path.is_file,
        )
        progress = summarize_run(store.list_jobs(), ControllerStatus(
            state="FINISHING", pid=os.getpid(), alive=True,
            started_at=status.started_at, updated_at=utc_now(), completed_at=None,
            message="", log_path=None,
        ))
        final_state = (
            "COMPLETED"
            if progress.completed == progress.eligible
            else "PARTIAL_FAILURE"
            if progress.failed
            else "PAUSED_INTERRUPTED"
            if progress.interrupted
            else "STOPPED_AFTER_CURRENT"
        )
        if progress.failed or progress.interrupted:
            exit_code = 1
        _write_controller_state(
            store,
            state=final_state,
            pid=os.getpid(),
            message=f"Detached {kind} controller finished its current invocation.",
            started_at=status.started_at,
            completed_at=utc_now(),
        )
    except BaseException as exc:
        exit_code = 1
        _write_controller_state(
            store,
            state="FAILED",
            pid=os.getpid(),
            message=f"{type(exc).__name__}: {exc}",
            started_at=status.started_at,
            completed_at=utc_now(),
        )
    finally:
        stop_path.unlink(missing_ok=True)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached recurrent production control")
    parser.add_argument("--execute-run", action="store_true")
    parser.add_argument("--run-directory", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execute_run or args.run_directory is None:
        print("--execute-run and --run-directory are required")
        return 2
    return _execute_controller(TrainingRunStore(args.run_directory))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AggregateTrainingProgress",
    "BENCHMARK_RUN_KIND",
    "ControllerStatus",
    "LEGACY_RUN_KIND",
    "PRODUCTION_CONTROL_VERSION",
    "PRODUCTION_RUN_KIND",
    "ProductionControlError",
    "ProductionPlan",
    "RunCatalogEntry",
    "RunProgress",
    "RunSnapshot",
    "SMOKE_RUN_KIND",
    "TrainingProgressValue",
    "aggregate_training_progress",
    "bounded_log_tail",
    "build_job_table",
    "calculate_training_progress",
    "classify_run",
    "controller_status",
    "default_run_selection",
    "latest_job_diagnostics",
    "job_training_progress",
    "launch_production_controller",
    "list_run_catalog",
    "load_run_snapshot",
    "load_validation_metrics",
    "normalize_run_status",
    "prepare_production_run",
    "production_config",
    "production_plan",
    "recover_dead_controller",
    "recent_orchestration_events",
    "registry_view",
    "request_interrupt",
    "request_stop_after_current",
    "requeue_jobs",
    "summarize_run",
    "symbol_contract_summary",
]
