"""Selective recurrent training contracts and global model coverage.

The frozen full-production run remains a separate immutable object.  A selected
run contains only its explicitly confirmed eligible members and records its
relationship to the frozen research universe in an atomic sidecar.  Coverage is
derived from persisted jobs and verified artifacts across run history; registry
rows are deliberately not evidence of completed training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from data_pipeline.src.config import TRAINING_RUNS_DIR
from feature_engineering.storage import atomic_write_json
from reinforcement_learning.integrity import sha256_file

from .job_state import (
    COMPLETED,
    FAILED,
    INTERRUPTED,
    QUEUED,
    TRAINING,
    VALIDATING,
    TrainingJobRecord,
    TrainingRunManifest,
    canonical_hash,
)
from .recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    RecurrentUniverseDiscovery,
    TrainingRunStore,
    create_training_run,
)
from .recurrent_trainer import RECURRENT_TRAINER_VERSION


SELECTED_RUN_KIND = "SELECTED"
SELECTED_RUN_SCHEMA_VERSION = "recurrent_selected_run_v1"
SELECTED_UNIVERSE_VERSION = "selected_frozen_equity_membership_v1"
SELECTED_IDENTITY_POLICY = "SELECTED_MEMBERS_OF_FROZEN_RESEARCH_UNIVERSE"
SELECTED_EXECUTION_POLICY_VERSION = "selected_symbol_training_v1"
SELECTED_RUN_METADATA_FILENAME = "selected_run.json"
SELECTED_DEVICE_POLICY_VERSION = "selected_recurrent_device_policy_v1"

UNTRAINED = "UNTRAINED"
TRAINED = "TRAINED"
GLOBAL_COVERAGE_STATUSES = (
    UNTRAINED,
    QUEUED,
    TRAINING,
    VALIDATING,
    TRAINED,
    FAILED,
    INTERRUPTED,
)


class SelectiveTrainingError(RuntimeError):
    """Raised when selective membership or coverage cannot be trusted."""


@dataclass(frozen=True)
class SelectedRunMetadata:
    """Immutable selected-run identity and execution policy sidecar."""

    schema_version: str
    run_kind: str
    run_id: str
    frozen_identity_count: int
    frozen_universe_version: str
    frozen_universe_hash: str
    frozen_trainable_symbol_hash: str
    requested_symbols: tuple[str, ...]
    requested_symbol_hash: str
    selected_symbols: tuple[str, ...]
    selected_symbol_hash: str
    skipped_trained_symbols: tuple[str, ...]
    retrain_existing: bool
    attempt_version: int
    execution_training_policy: str
    requested_timesteps: int
    seed: int
    requested_device: str
    device_policy_version: str
    worker_count: int
    cpu_threads_per_worker: int
    cuda_execution_authorized: bool
    validation_enabled: bool
    test_partition_loaded: bool
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != SELECTED_RUN_SCHEMA_VERSION:
            raise SelectiveTrainingError("selected-run schema is incompatible")
        if self.run_kind != SELECTED_RUN_KIND:
            raise SelectiveTrainingError("selected run kind is invalid")
        if not self.run_id:
            raise SelectiveTrainingError("selected run id is required")
        for label, values in (
            ("requested", self.requested_symbols),
            ("selected", self.selected_symbols),
            ("skipped", self.skipped_trained_symbols),
        ):
            if tuple(sorted(set(values))) != values:
                raise SelectiveTrainingError(
                    f"{label} selected-run symbols must be sorted and unique"
                )
        if not self.requested_symbols or not self.selected_symbols:
            raise SelectiveTrainingError("selected run membership cannot be empty")
        if not set(self.selected_symbols).issubset(self.requested_symbols):
            raise SelectiveTrainingError("selected membership escaped the request")
        if not set(self.skipped_trained_symbols).issubset(self.requested_symbols):
            raise SelectiveTrainingError("skipped membership escaped the request")
        if set(self.selected_symbols).intersection(self.skipped_trained_symbols):
            raise SelectiveTrainingError("a symbol cannot be selected and skipped")
        if self.requested_symbol_hash != selected_membership_hash(
            self.requested_symbols
        ):
            raise SelectiveTrainingError("requested symbol hash does not reconcile")
        if self.selected_symbol_hash != selected_membership_hash(
            self.selected_symbols
        ):
            raise SelectiveTrainingError("selected symbol hash does not reconcile")
        if self.attempt_version < 0:
            raise SelectiveTrainingError("selected attempt version cannot be negative")
        if self.requested_timesteps < 1 or self.worker_count < 1:
            raise SelectiveTrainingError("selected execution counters must be positive")
        if self.cpu_threads_per_worker < 1:
            raise SelectiveTrainingError("selected CPU thread count must be positive")
        if self.test_partition_loaded:
            raise SelectiveTrainingError("TEST cannot enter selected-run metadata")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requested_symbols"] = list(self.requested_symbols)
        payload["selected_symbols"] = list(self.selected_symbols)
        payload["skipped_trained_symbols"] = list(self.skipped_trained_symbols)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelectedRunMetadata":
        values = dict(payload)
        for field in (
            "requested_symbols",
            "selected_symbols",
            "skipped_trained_symbols",
        ):
            raw = values.get(field)
            if not isinstance(raw, (list, tuple)):
                raise SelectiveTrainingError(f"selected-run {field} is malformed")
            values[field] = tuple(str(value) for value in raw)
        try:
            return cls(**values)
        except TypeError as exc:
            raise SelectiveTrainingError("selected-run metadata is malformed") from exc


@dataclass(frozen=True)
class CoverageSummary:
    eligible: int
    trained: int
    untrained: int
    training: int
    validating: int
    failed: int
    interrupted: int


def selected_membership_hash(symbols: Sequence[str]) -> str:
    """Return the deterministic membership identity for canonical symbols."""

    canonical = tuple(sorted({str(symbol).strip().upper() for symbol in symbols}))
    if not canonical or any(not symbol for symbol in canonical):
        raise SelectiveTrainingError("selected membership cannot be empty")
    return canonical_hash({"symbols": list(canonical)})


def selected_metadata_path(run_directory: Path) -> Path:
    return Path(run_directory) / SELECTED_RUN_METADATA_FILENAME


def load_selected_run_metadata(run_directory: Path) -> SelectedRunMetadata:
    path = selected_metadata_path(run_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SelectiveTrainingError("selected-run metadata is missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectiveTrainingError("selected-run metadata is unreadable") from exc
    if not isinstance(payload, dict):
        raise SelectiveTrainingError("selected-run metadata is malformed")
    return SelectedRunMetadata.from_dict(payload)


def validate_selected_run(
    store: TrainingRunStore, *, executable: bool = False
) -> SelectedRunMetadata:
    """Fail closed unless a selected sidecar, manifest, and jobs reconcile."""

    from .production_control import production_plan

    metadata = load_selected_run_metadata(store.run_directory)
    manifest = store.read_manifest()
    plan = production_plan()
    selected_hash = selected_membership_hash(metadata.selected_symbols)
    expected_universe_hash = canonical_hash(
        {
            "version": SELECTED_UNIVERSE_VERSION,
            "frozen_universe_hash": plan.universe_hash,
            "selected_symbol_hash": selected_hash,
        }
    )
    expected_execution_policy = (
        f"{SELECTED_EXECUTION_POLICY_VERSION}:"
        f"{metadata.attempt_version}:{selected_hash}"
    )
    checks = {
        "run id": manifest.run_id == metadata.run_id,
        "run kind": metadata.run_kind == SELECTED_RUN_KIND,
        "frozen identity count": metadata.frozen_identity_count == plan.identity_count,
        "frozen universe version": (
            metadata.frozen_universe_version == plan.frozen_universe_version
        ),
        "frozen universe hash": metadata.frozen_universe_hash == plan.universe_hash,
        "frozen trainable hash": (
            metadata.frozen_trainable_symbol_hash == plan.trainable_symbol_hash
        ),
        "identity policy": manifest.identity_policy == SELECTED_IDENTITY_POLICY,
        "identity snapshot": manifest.identity_snapshot == plan.identity_snapshot,
        "universe version": manifest.universe_version == SELECTED_UNIVERSE_VERSION,
        "selected universe hash": manifest.universe_hash == expected_universe_hash,
        "selected count": manifest.identity_count == len(metadata.selected_symbols),
        "eligible count": manifest.eligible_count == len(metadata.selected_symbols),
        "no ineligible jobs": manifest.ineligible_count == 0,
        "selected hash": manifest.trainable_symbol_hash == selected_hash,
        "execution policy": (
            manifest.execution_training_policy
            == metadata.execution_training_policy
            == expected_execution_policy
        ),
        "timesteps": manifest.requested_timesteps == metadata.requested_timesteps,
        "seed": manifest.seed == metadata.seed,
        "device": manifest.requested_device == metadata.requested_device,
        "trainer": manifest.agent_version == RECURRENT_TRAINER_VERSION,
        "validation": manifest.validation_enabled and metadata.validation_enabled,
        "TEST seal": (
            not manifest.test_partition_loaded and not metadata.test_partition_loaded
        ),
    }
    failed = [label for label, valid in checks.items() if not valid]
    if failed:
        raise SelectiveTrainingError(
            "selected-run contract is incompatible: " + ", ".join(failed)
        )
    jobs = store.list_jobs()
    if tuple(job.symbol for job in jobs) != metadata.selected_symbols:
        raise SelectiveTrainingError("selected membership no longer matches jobs")
    if any(job.trainability != "eligible" for job in jobs):
        raise SelectiveTrainingError("selected run contains an ineligible job")
    if executable:
        execution_checks = {
            "qualified timesteps": metadata.requested_timesteps
            == plan.requested_timesteps,
            "qualified seed": metadata.seed == plan.seed,
            "qualified CPU device": metadata.requested_device == plan.requested_device,
            "qualified worker count": metadata.worker_count == plan.worker_count,
            "qualified CPU threads": (
                metadata.cpu_threads_per_worker == plan.cpu_threads_per_worker
            ),
            "CUDA not authorized": not metadata.cuda_execution_authorized,
        }
        execution_failed = [
            label for label, valid in execution_checks.items() if not valid
        ]
        if execution_failed:
            raise SelectiveTrainingError(
                "selected run is not authorized for execution: "
                + ", ".join(execution_failed)
            )
    return metadata


def _selected_attempt(
    requested_hash: str, *, runs_root: Path, retrain_existing: bool
) -> int:
    attempts: list[int] = []
    for path in sorted(Path(runs_root).glob(f"*/{SELECTED_RUN_METADATA_FILENAME}")):
        try:
            metadata = load_selected_run_metadata(path.parent)
        except SelectiveTrainingError:
            continue
        if (
            metadata.requested_symbol_hash == requested_hash
            and metadata.retrain_existing == retrain_existing
        ):
            attempts.append(metadata.attempt_version)
    return max(attempts, default=-1) + 1


def _compatible_existing_selected_run(
    *,
    runs_root: Path,
    requested_hash: str,
    selected_symbols: tuple[str, ...],
) -> tuple[TrainingRunStore, SelectedRunMetadata] | None:
    """Return an existing default selected run instead of duplicating its queue."""

    for path in sorted(
        Path(runs_root).glob(f"*/{SELECTED_RUN_METADATA_FILENAME}"), reverse=True
    ):
        try:
            metadata = load_selected_run_metadata(path.parent)
            if (
                metadata.retrain_existing
                or metadata.requested_symbol_hash != requested_hash
                or metadata.selected_symbols != selected_symbols
            ):
                continue
            store = TrainingRunStore(path.parent)
            validate_selected_run(store)
        except (OSError, ValueError, RuntimeError):
            continue
        return store, metadata
    return None


def _selected_discovery(
    frozen: RecurrentUniverseDiscovery,
    symbols: tuple[str, ...],
    *,
    attempt_version: int,
) -> RecurrentUniverseDiscovery:
    records = frozen.records.loc[frozen.records["symbol"].isin(symbols)].copy(deep=True)
    records = records.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    if tuple(records["symbol"].astype(str)) != symbols:
        raise SelectiveTrainingError("selected discovery lost requested membership")
    if not records["category"].eq(ELIGIBLE_TRAINABLE).all():
        raise SelectiveTrainingError("selected discovery contains an ineligible symbol")
    membership_hash = selected_membership_hash(symbols)
    universe_hash = canonical_hash(
        {
            "version": SELECTED_UNIVERSE_VERSION,
            "frozen_universe_hash": frozen.universe_hash,
            "selected_symbol_hash": membership_hash,
        }
    )
    source_inventory_hash = canonical_hash(
        {
            "selected_universe_hash": universe_hash,
            "contracts": [
                {
                    "symbol": str(row.symbol),
                    "source_data_hash": str(row.source_data_hash),
                }
                for row in records.itertuples(index=False)
            ],
        }
    )
    return RecurrentUniverseDiscovery(
        records=records,
        universe_version=SELECTED_UNIVERSE_VERSION,
        universe_hash=universe_hash,
        identity_count=len(records),
        category_counts={ELIGIBLE_TRAINABLE: len(records)},
        source_inventory_hash=source_inventory_hash,
        identity_policy=SELECTED_IDENTITY_POLICY,
        identity_snapshot=frozen.identity_snapshot,
        execution_training_policy=(
            f"{SELECTED_EXECUTION_POLICY_VERSION}:{attempt_version}:{membership_hash}"
        ),
    )


def prepare_selected_run(
    symbols: Sequence[str],
    *,
    retrain_trained: bool = False,
    runs_root: Path = TRAINING_RUNS_DIR,
    coverage: pd.DataFrame | None = None,
    frozen_discovery: RecurrentUniverseDiscovery | None = None,
    created_at: str | None = None,
) -> tuple[TrainingRunStore, SelectedRunMetadata, bool]:
    """Create an immutable selected run without starting its controller."""

    from .production_control import _frozen_discovery, production_config, production_plan

    requested = tuple(sorted({str(symbol).strip().upper() for symbol in symbols}))
    if not requested or any(not symbol for symbol in requested):
        raise SelectiveTrainingError("select at least one eligible symbol")
    frozen = frozen_discovery or _frozen_discovery()
    eligible = set(
        frozen.records.loc[
            frozen.records["category"].eq(ELIGIBLE_TRAINABLE), "symbol"
        ].astype(str)
    )
    invalid = sorted(set(requested).difference(eligible))
    if invalid:
        raise SelectiveTrainingError(
            "selected symbols are not frozen eligible identities: " + ", ".join(invalid)
        )
    if coverage is None:
        coverage, _ = build_global_model_coverage(
            runs_root=Path(runs_root), frozen_discovery=frozen
        )
    required_coverage = {"symbol", "trained"}
    if not required_coverage.issubset(coverage.columns):
        raise SelectiveTrainingError("global coverage table is incomplete")
    trained = set(
        coverage.loc[coverage["trained"].astype(bool), "symbol"].astype(str)
    )
    skipped = () if retrain_trained else tuple(sorted(set(requested).intersection(trained)))
    selected = tuple(sorted(set(requested).difference(skipped)))
    if not selected:
        raise SelectiveTrainingError(
            "all selected symbols are already trained; use explicit retraining"
        )
    root = Path(runs_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    requested_hash = selected_membership_hash(requested)
    if not retrain_trained:
        existing = _compatible_existing_selected_run(
            runs_root=root,
            requested_hash=requested_hash,
            selected_symbols=selected,
        )
        if existing is not None:
            store, metadata = existing
            return store, metadata, False
    attempt = _selected_attempt(
        requested_hash, runs_root=root, retrain_existing=retrain_trained
    )
    discovery = _selected_discovery(frozen, selected, attempt_version=attempt)
    config = production_config()
    store = create_training_run(
        discovery,
        config=config,
        runs_root=root,
        validation_enabled=True,
        created_at=created_at,
    )
    manifest = store.read_manifest()
    plan = production_plan()
    metadata = SelectedRunMetadata(
        schema_version=SELECTED_RUN_SCHEMA_VERSION,
        run_kind=SELECTED_RUN_KIND,
        run_id=manifest.run_id,
        frozen_identity_count=plan.identity_count,
        frozen_universe_version=plan.frozen_universe_version,
        frozen_universe_hash=plan.universe_hash,
        frozen_trainable_symbol_hash=plan.trainable_symbol_hash,
        requested_symbols=requested,
        requested_symbol_hash=requested_hash,
        selected_symbols=selected,
        selected_symbol_hash=selected_membership_hash(selected),
        skipped_trained_symbols=skipped,
        retrain_existing=retrain_trained,
        attempt_version=attempt,
        execution_training_policy=discovery.execution_training_policy,
        requested_timesteps=config.total_timesteps,
        seed=config.seed,
        requested_device=config.device,
        device_policy_version=SELECTED_DEVICE_POLICY_VERSION,
        worker_count=plan.worker_count,
        cpu_threads_per_worker=plan.cpu_threads_per_worker,
        cuda_execution_authorized=False,
        validation_enabled=True,
        test_partition_loaded=False,
        created_at=manifest.created_at,
    )
    atomic_write_json(metadata.to_dict(), selected_metadata_path(store.run_directory))
    validate_selected_run(store)
    return store, metadata, True


def _read_validation_artifact(
    store: TrainingRunStore, job: TrainingJobRecord
) -> dict[str, object] | None:
    if not job.validation_metrics_reference:
        return None
    try:
        path = store.resolve_artifact(job.validation_metrics_reference)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def completed_job_is_trained(
    store: TrainingRunStore, job: TrainingJobRecord
) -> tuple[bool, str]:
    """Verify the persisted full-budget model and VALIDATION artifact."""

    from .production_control import production_plan

    plan = production_plan()
    if job.status != COMPLETED:
        return False, "training_not_completed"
    if job.agent_version != plan.trainer_version:
        return False, "trainer_metadata_incompatible"
    if job.requested_timesteps != plan.requested_timesteps:
        return False, "training_budget_incompatible"
    if job.completed_timesteps < job.requested_timesteps:
        return False, "training_budget_incomplete"
    if not job.model_path or not job.model_sha256:
        return False, "model_metadata_missing"
    try:
        model_path = store.resolve_artifact(job.model_path)
    except (OSError, ValueError, RuntimeError):
        return False, "model_path_invalid"
    if not model_path.is_file():
        return False, "model_artifact_missing"
    if sha256_file(model_path) != job.model_sha256:
        return False, "model_integrity_failed"
    if job.validation_status != "completed":
        return False, "validation_not_completed"
    validation = _read_validation_artifact(store, job)
    if validation is None:
        return False, "validation_artifact_missing"
    if validation.get("evaluation_partition") != "validation":
        return False, "validation_partition_invalid"
    if validation.get("symbol") != job.symbol:
        return False, "validation_symbol_mismatch"
    if validation.get("test_evaluated") is True:
        return False, "test_access_reported"
    compatibility = {
        "environment_version": job.environment_version,
        "feature_version": job.feature_version,
        "recurrent_contract_version": job.data_contract_version,
    }
    if any(validation.get(key) != value for key, value in compatibility.items()):
        return False, "validation_metadata_incompatible"
    if validation.get("model_parameters_unchanged") is not True:
        return False, "validation_integrity_unverified"
    return True, "verified"


def _valid_run_stores(runs_root: Path) -> tuple[TrainingRunStore, ...]:
    stores: list[TrainingRunStore] = []
    for manifest_path in sorted(Path(runs_root).glob("*/run_manifest.json")):
        try:
            store = TrainingRunStore(manifest_path.parent)
            manifest = store.read_manifest()
            store.list_jobs()
            if manifest.test_partition_loaded:
                continue
            selected_path = selected_metadata_path(store.run_directory)
            if selected_path.exists():
                validate_selected_run(store)
        except (OSError, ValueError, RuntimeError):
            continue
        stores.append(store)
    return tuple(stores)


def build_global_model_coverage(
    *,
    runs_root: Path = TRAINING_RUNS_DIR,
    frozen_discovery: RecurrentUniverseDiscovery | None = None,
) -> tuple[pd.DataFrame, CoverageSummary]:
    """Derive frozen-eligible coverage from valid persisted recurrent history."""

    from .production_control import _frozen_discovery, classify_run

    frozen = frozen_discovery or _frozen_discovery()
    identity = frozen.records.loc[
        frozen.records["category"].eq(ELIGIBLE_TRAINABLE),
        ["symbol", "company_name", "sector"],
    ].copy(deep=True)
    identity["symbol"] = identity["symbol"].astype(str)
    histories: dict[str, list[dict[str, object]]] = {
        symbol: [] for symbol in identity["symbol"]
    }
    for store in _valid_run_stores(Path(runs_root)):
        manifest = store.read_manifest()
        run_kind = classify_run(manifest, store.run_directory)
        for job in store.list_jobs():
            if job.symbol not in histories or job.trainability != "eligible":
                continue
            verified, integrity = completed_job_is_trained(store, job)
            histories[job.symbol].append(
                {
                    "job": job,
                    "run_id": manifest.run_id,
                    "run_kind": run_kind,
                    "verified": verified,
                    "integrity": integrity,
                }
            )
    rows: list[dict[str, object]] = []
    metadata = identity.set_index("symbol").to_dict(orient="index")
    for symbol in sorted(histories):
        history = sorted(
            histories[symbol],
            key=lambda item: (
                str(item["job"].updated_at),
                str(item["run_id"]),
            ),
        )
        latest = history[-1] if history else None
        verified_runs = [item for item in history if bool(item["verified"])]
        latest_success = verified_runs[-1] if verified_runs else None
        latest_job = latest["job"] if latest else None
        active_training = any(item["job"].status == TRAINING for item in history)
        active_validating = any(item["job"].status == VALIDATING for item in history)
        trained = latest_success is not None
        if active_validating:
            status = VALIDATING
        elif active_training:
            status = TRAINING
        elif trained:
            status = TRAINED
        elif latest_job is not None and latest_job.status in {
            QUEUED,
            FAILED,
            INTERRUPTED,
        }:
            status = latest_job.status
        else:
            status = UNTRAINED
        progress = latest_job.progress_percent if latest_job is not None else 0.0
        success_job = latest_success["job"] if latest_success else None
        display_job = success_job or latest_job
        row_metadata = metadata[symbol]
        rows.append(
            {
                "symbol": symbol,
                "company_name": row_metadata.get("company_name"),
                "sector": row_metadata.get("sector"),
                "coverage_status": status,
                "trained": trained,
                "latest_progress_percent": float(progress),
                "actual_timesteps": (
                    latest_job.completed_timesteps if latest_job is not None else 0
                ),
                "requested_timesteps": (
                    latest_job.requested_timesteps if latest_job is not None else 0
                ),
                "model_status": (
                    "verified"
                    if trained
                    else str(latest["integrity"])
                    if latest is not None
                    else "not_created"
                ),
                "validation_status": (
                    display_job.validation_status if display_job is not None else "not_requested"
                ),
                "latest_run_id": str(latest["run_id"]) if latest else "",
                "latest_run_kind": str(latest["run_kind"]) if latest else "",
                "latest_attempt": (
                    latest_job.retry_count if latest_job is not None else 0
                ),
                "latest_successful_run": (
                    str(latest_success["run_id"]) if latest_success else ""
                ),
                "model_path": success_job.model_path if success_job is not None else None,
                "currently_training": active_training,
                "currently_validating": active_validating,
            }
        )
    table = pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(
        drop=True
    )
    trained_count = int(table["trained"].sum())
    summary = CoverageSummary(
        eligible=len(table),
        trained=trained_count,
        untrained=len(table) - trained_count,
        training=int(table["currently_training"].sum()),
        validating=int(table["currently_validating"].sum()),
        failed=int(table["coverage_status"].eq(FAILED).sum()),
        interrupted=int(table["coverage_status"].eq(INTERRUPTED).sum()),
    )
    return table, summary


def filter_symbol_coverage(
    coverage: pd.DataFrame,
    *,
    statuses: Sequence[str] = (),
    sectors: Sequence[str] = (),
    search: str = "",
) -> pd.DataFrame:
    """Apply deterministic selector filters without changing coverage state."""

    required = {"symbol", "company_name", "sector", "coverage_status", "trained"}
    if not required.issubset(coverage.columns):
        raise SelectiveTrainingError("coverage table is missing selector columns")
    filtered = coverage.copy(deep=True)
    selected_statuses = {str(value) for value in statuses}
    if selected_statuses:
        masks: list[pd.Series] = []
        for status in selected_statuses:
            if status == TRAINED:
                masks.append(filtered["trained"].astype(bool))
            elif status == UNTRAINED:
                masks.append(~filtered["trained"].astype(bool))
            else:
                masks.append(filtered["coverage_status"].eq(status))
        status_mask = masks[0]
        for mask in masks[1:]:
            status_mask = status_mask | mask
        filtered = filtered.loc[status_mask]
    if sectors:
        filtered = filtered.loc[filtered["sector"].isin(tuple(sectors))]
    query = search.strip()
    if query:
        matched = filtered["symbol"].astype("string").str.contains(
            query, case=False, regex=False, na=False
        ) | filtered["company_name"].astype("string").str.contains(
            query, case=False, regex=False, na=False
        )
        filtered = filtered.loc[matched]
    return filtered.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def canonical_symbol_selection(
    symbols: Sequence[str], *, eligible_symbols: Sequence[str]
) -> tuple[str, ...]:
    """Normalize transient UI selection and discard stale noneligible members."""

    eligible = {str(symbol).strip().upper() for symbol in eligible_symbols}
    if any(not symbol for symbol in eligible):
        raise SelectiveTrainingError("eligible symbol membership is malformed")
    requested = {str(symbol).strip().upper() for symbol in symbols}
    return tuple(sorted(symbol for symbol in requested if symbol and symbol in eligible))


def reconcile_visible_symbol_selection(
    selected_symbols: Sequence[str],
    *,
    visible_symbols: Sequence[str],
    checked_visible_symbols: Sequence[str],
    eligible_symbols: Sequence[str],
) -> tuple[str, ...]:
    """Replace only the visible selection slice while preserving hidden rows."""

    eligible = set(
        canonical_symbol_selection(eligible_symbols, eligible_symbols=eligible_symbols)
    )
    current = set(
        canonical_symbol_selection(selected_symbols, eligible_symbols=eligible)
    )
    visible = {str(symbol).strip().upper() for symbol in visible_symbols}
    checked = {str(symbol).strip().upper() for symbol in checked_visible_symbols}
    if not visible.issubset(eligible):
        raise SelectiveTrainingError("visible selector rows escaped eligible membership")
    if not checked.issubset(visible):
        raise SelectiveTrainingError("checked symbols escaped the visible selector rows")
    return tuple(sorted(current.difference(visible).union(checked)))


def select_visible_symbols(
    selected_symbols: Sequence[str],
    *,
    visible_symbols: Sequence[str],
    eligible_symbols: Sequence[str],
) -> tuple[str, ...]:
    """Add all currently visible eligible rows to canonical selection."""

    return reconcile_visible_symbol_selection(
        selected_symbols,
        visible_symbols=visible_symbols,
        checked_visible_symbols=visible_symbols,
        eligible_symbols=eligible_symbols,
    )


def clear_visible_symbols(
    selected_symbols: Sequence[str],
    *,
    visible_symbols: Sequence[str],
    eligible_symbols: Sequence[str],
) -> tuple[str, ...]:
    """Remove only currently visible rows from canonical selection."""

    return reconcile_visible_symbol_selection(
        selected_symbols,
        visible_symbols=visible_symbols,
        checked_visible_symbols=(),
        eligible_symbols=eligible_symbols,
    )


__all__ = [
    "CoverageSummary",
    "GLOBAL_COVERAGE_STATUSES",
    "SELECTED_RUN_KIND",
    "SELECTED_RUN_METADATA_FILENAME",
    "SELECTED_RUN_SCHEMA_VERSION",
    "SelectedRunMetadata",
    "SelectiveTrainingError",
    "TRAINED",
    "UNTRAINED",
    "build_global_model_coverage",
    "canonical_symbol_selection",
    "clear_visible_symbols",
    "completed_job_is_trained",
    "filter_symbol_coverage",
    "load_selected_run_metadata",
    "prepare_selected_run",
    "reconcile_visible_symbol_selection",
    "select_visible_symbols",
    "selected_membership_hash",
    "validate_selected_run",
]
