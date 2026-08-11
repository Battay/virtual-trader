"""Pure safety and presentation helpers for the Streamlit PPO workflow.

This module deliberately performs no training, validation, or persistence at
import time.  Its action wrappers have fixed TRAIN/VALIDATION boundaries and
are called only from explicit Streamlit button branches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.presentation import (
    MISSING_VALUE,
    format_date,
    format_datetime,
    format_decimal,
    format_integer,
    format_percentage,
    format_price,
    safe_display_value,
    status_label,
)
from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    PROCESSED_SPLITS_DIR,
    SAVED_MODELS_DIR,
)
from feature_engineering.storage import safe_path_component
from reinforcement_learning.data_contract import (
    RLContractMetadata,
    RL_OBSERVATION_SCALER_FILENAME,
    load_rl_contract_metadata,
)
from reinforcement_learning.history_policy import (
    HistoryClass,
    classify_usable_history,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.training.config import PPO_CONFIG_VERSION, PPOConfig


PILOT_SYMBOLS = (
    "OGDC",
    "UBL",
    "FFC",
    "PPL",
    "MEBL",
    "LUCK",
    "HUBC",
    "PSO",
    "MLCF",
    "TRG",
)
TIMESTEP_PRESETS = (10_000, 25_000, 50_000, 100_000)
DEVICE_OPTIONS = ("cpu", "mps", "auto")
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42
CPU_MPS_SPEED_MULTIPLIER = 6.81
DEVICE_RECOMMENDATION = (
    "CPU — recommended for the current single-environment PPO workload"
)
PPO_WORKFLOW_SESSION_KEY = "ppo_single_symbol_workflow_v1"
JOB_PHASES = frozenset({"idle", "training", "validating", "persisting"})


@dataclass(frozen=True)
class SelectedSymbolSummary:
    """Compact, metadata-only summary of one canonical RL artifact set."""

    symbol: str
    train_rows: int
    train_start: str
    train_end: str
    validation_rows: int
    validation_start: str
    validation_end: str
    test_rows: int
    test_start: str
    test_end: str
    feature_version: str
    observation_shape: tuple[int, ...]
    rl_contract_version: str
    environment_version: str
    contract_path: Path
    contract_sha256: str
    observation_scaler_sha256: str
    observation_scaler_metadata_sha256: str
    train_validation_artifact_fingerprint: str


@dataclass(frozen=True)
class ReadySymbolCatalog:
    """Ready symbols, compact summaries, and explicit rejected reasons."""

    ready_symbols: tuple[str, ...]
    summaries: Mapping[str, SelectedSymbolSummary]
    rejected_reasons: Mapping[str, str]
    failure_categories: Mapping[str, str]


@dataclass(frozen=True)
class PPOReadinessReconciliation:
    """Auditable feature-readiness and compatible-contract intersection."""

    eligible_symbols: int
    compatible_rl_symbols: int
    intersection: int
    missing_contracts: int
    stale_contracts: int
    incompatible_feature_versions: int
    incompatible_contract_versions: int
    incompatible_environment_versions: int
    other_failures: int


@dataclass(frozen=True)
class SelectedSymbolTrainingProfile:
    """Read-only current readiness and future history route for one symbol."""

    symbol: str
    company_name: str
    sector: str
    usable_observations: int
    history_class: HistoryClass
    history_class_label: str
    current_mlp_ppo_ready: bool
    future_training_route: str


@dataclass(frozen=True)
class PPOWorkflowIdentity:
    """Every input/provenance value that identifies one in-memory candidate."""

    symbol: str
    requested_timesteps: int
    seed: int
    requested_device: str
    ppo_config_version: str
    contract_sha256: str
    observation_scaler_sha256: str
    observation_scaler_metadata_sha256: str
    train_validation_artifact_fingerprint: str

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("workflow symbol is required")
        if (
            isinstance(self.requested_timesteps, bool)
            or not isinstance(self.requested_timesteps, int)
            or self.requested_timesteps < 1
        ):
            raise ValueError("workflow timesteps must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("workflow seed must be a non-negative integer")
        if self.requested_device not in DEVICE_OPTIONS:
            raise ValueError("workflow device must be cpu, mps, or auto")
        if self.ppo_config_version != PPO_CONFIG_VERSION:
            raise ValueError("workflow PPO config version is incompatible")
        if len(self.contract_sha256) != 64:
            raise ValueError("workflow contract SHA-256 is invalid")
        if len(self.observation_scaler_sha256) != 64:
            raise ValueError("workflow observation scaler SHA-256 is invalid")
        if len(self.observation_scaler_metadata_sha256) != 64:
            raise ValueError(
                "workflow observation scaler metadata SHA-256 is invalid"
            )
        if len(self.train_validation_artifact_fingerprint) != 64:
            raise ValueError("workflow TRAIN/VALIDATION artifact fingerprint is invalid")


@dataclass(frozen=True)
class ActionAvailability:
    """Whether an explicit action is currently safe, with a readable reason."""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class CandidateVersionPreview:
    """Read-only expected identity; persistence performs authoritative allocation."""

    model_id: str
    model_version: int


def _as_ready(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def _contract_failure_category(error: Exception) -> str:
    """Return a stable audit category without weakening the original error."""
    detail = str(error).casefold()
    if "missing" in detail or "no such file" in detail:
        return "missing_contract"
    if "feature version" in detail:
        return "incompatible_feature_version"
    if "schema version" in detail or "contract version" in detail:
        return "incompatible_contract_version"
    if "environment version" in detail:
        return "incompatible_environment_version"
    if (
        "stale" in detail
        or "row counts differ" in detail
        or "date bounds differ" in detail
    ):
        return "stale_contract"
    return "other_failure"


def _summary(metadata: RLContractMetadata) -> SelectedSymbolSummary:
    scaler_path = metadata.contract_path.parent / RL_OBSERVATION_SCALER_FILENAME
    scaler_metadata_path = scaler_path.with_suffix(".json")
    artifact_digest = hashlib.sha256()
    for name in ("train.csv", "train_rl.csv", "validation.csv", "validation_rl.csv"):
        path = metadata.contract_path.parent / name
        details = path.stat()
        artifact_digest.update(name.encode("utf-8"))
        artifact_digest.update(str(details.st_mtime_ns).encode("ascii"))
        artifact_digest.update(str(details.st_size).encode("ascii"))
    return SelectedSymbolSummary(
        symbol=metadata.symbol,
        train_rows=metadata.train.rows,
        train_start=metadata.train.start,
        train_end=metadata.train.end,
        validation_rows=metadata.validation.rows,
        validation_start=metadata.validation.start,
        validation_end=metadata.validation.end,
        test_rows=metadata.test.rows,
        test_start=metadata.test.start,
        test_end=metadata.test.end,
        feature_version=metadata.feature_version,
        observation_shape=metadata.observation_shape,
        rl_contract_version=metadata.rl_contract_version,
        environment_version=metadata.environment_version,
        contract_path=metadata.contract_path,
        contract_sha256=sha256_file(metadata.contract_path),
        observation_scaler_sha256=sha256_file(scaler_path),
        observation_scaler_metadata_sha256=sha256_file(scaler_metadata_path),
        train_validation_artifact_fingerprint=artifact_digest.hexdigest(),
    )


def build_ready_symbol_catalog(
    status_table: pd.DataFrame,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    metadata_loader: Callable[..., RLContractMetadata] | None = None,
) -> ReadySymbolCatalog:
    """Intersect feature readiness with validated metadata-only RL artifacts."""
    required = {
        "symbol",
        "eligible",
        "readiness_status",
        "train_rows",
        "validation_rows",
        "test_rows",
        "processed_first_date",
        "processed_last_date",
    }
    missing = sorted(required.difference(status_table.columns))
    if missing:
        raise ValueError(f"readiness table is missing: {', '.join(missing)}")
    source = status_table.loc[:, list(required)].copy(deep=True)
    source["symbol"] = source["symbol"].astype("string").str.strip()
    if source["symbol"].isna().any() or source["symbol"].eq("").any():
        raise ValueError("readiness table contains an empty symbol")
    if source["symbol"].duplicated().any():
        raise ValueError("readiness table contains duplicate symbols")

    loader = metadata_loader or load_rl_contract_metadata
    summaries: dict[str, SelectedSymbolSummary] = {}
    rejected: dict[str, str] = {}
    failure_categories: dict[str, str] = {}
    for row in source.sort_values("symbol", kind="stable").itertuples(index=False):
        symbol = str(row.symbol)
        if not _as_ready(row.eligible) or str(row.readiness_status) != "Ready":
            rejected[symbol] = str(row.readiness_status) or "Not Ready"
            failure_categories[symbol] = "ineligible"
            continue
        try:
            metadata = loader(symbol, splits_dir=Path(splits_dir))
            if metadata.symbol != symbol:
                raise ValueError("RL metadata symbol does not match readiness symbol")
            summary = _summary(metadata)
            expected_counts = (
                int(row.train_rows),
                int(row.validation_rows),
                int(row.test_rows),
            )
            actual_counts = (
                summary.train_rows,
                summary.validation_rows,
                summary.test_rows,
            )
            if expected_counts != actual_counts:
                raise ValueError(
                    "RL partition row counts differ from current processed readiness"
                )
            first_date = pd.Timestamp(row.processed_first_date)
            last_date = pd.Timestamp(row.processed_last_date)
            if pd.isna(first_date) or pd.isna(last_date):
                raise ValueError("Current processed dataset bounds are unavailable")
            if (
                first_date.date().isoformat() != summary.train_start
                or last_date.date().isoformat() != summary.test_end
            ):
                raise ValueError(
                    "RL partition date bounds differ from current processed dataset"
                )
            summaries[symbol] = summary
        except Exception as exc:
            rejected[symbol] = f"RL contract unavailable: {exc}"
            failure_categories[symbol] = _contract_failure_category(exc)
    symbols = tuple(sorted(summaries, key=str.casefold))
    return ReadySymbolCatalog(
        symbols,
        dict(summaries),
        dict(rejected),
        dict(failure_categories),
    )


def readiness_reconciliation(
    status_table: pd.DataFrame,
    catalog: ReadySymbolCatalog,
) -> PPOReadinessReconciliation:
    """Summarize the exact eligible/compatible intersection and failures."""
    required = {"symbol", "eligible", "readiness_status"}
    missing = sorted(required.difference(status_table.columns))
    if missing:
        raise ValueError(f"readiness table is missing: {', '.join(missing)}")
    source = status_table.loc[:, list(required)].copy(deep=True)
    source["symbol"] = source["symbol"].astype("string").str.strip()
    eligible = {
        str(row.symbol)
        for row in source.itertuples(index=False)
        if _as_ready(row.eligible) and str(row.readiness_status) == "Ready"
    }
    compatible = eligible.intersection(catalog.ready_symbols)
    failures = {
        symbol: catalog.failure_categories.get(symbol, "other_failure")
        for symbol in eligible.difference(compatible)
    }
    counts = {
        category: sum(value == category for value in failures.values())
        for category in (
            "missing_contract",
            "stale_contract",
            "incompatible_feature_version",
            "incompatible_contract_version",
            "incompatible_environment_version",
            "other_failure",
        )
    }
    return PPOReadinessReconciliation(
        eligible_symbols=len(eligible),
        compatible_rl_symbols=len(compatible),
        intersection=len(compatible),
        missing_contracts=counts["missing_contract"],
        stale_contracts=counts["stale_contract"],
        incompatible_feature_versions=counts["incompatible_feature_version"],
        incompatible_contract_versions=counts["incompatible_contract_version"],
        incompatible_environment_versions=counts["incompatible_environment_version"],
        other_failures=counts["other_failure"],
    )


def selected_symbol_summary(
    catalog: ReadySymbolCatalog,
    symbol: str,
) -> SelectedSymbolSummary:
    """Return one ready symbol's summary or fail with the known readiness reason."""
    symbol_text = str(symbol).strip()
    summary = catalog.summaries.get(symbol_text)
    if summary is not None:
        return summary
    reason = catalog.rejected_reasons.get(symbol_text, "Symbol is not RL-ready")
    raise ValueError(f"{symbol_text or 'Selected symbol'} is not ready: {reason}")


def selected_symbol_training_profile(
    status_table: pd.DataFrame,
    catalog: ReadySymbolCatalog,
    symbol: str,
) -> SelectedSymbolTrainingProfile:
    """Combine canonical MLP readiness with the separate future history policy."""
    required = {"symbol", "company_name", "sector", "usable_rows"}
    missing = sorted(required.difference(status_table.columns))
    if missing:
        raise ValueError(f"readiness table is missing: {', '.join(missing)}")
    source = status_table.loc[:, list(required)].copy(deep=True)
    source["symbol"] = source["symbol"].astype("string").str.strip()
    symbol_text = str(symbol).strip()
    matches = source.loc[source["symbol"].eq(symbol_text)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one readiness record for {symbol_text!r}, found {len(matches)}"
        )
    row = matches.iloc[0]
    classification = classify_usable_history(row["usable_rows"])
    company_name = row["company_name"]
    sector = row["sector"]
    return SelectedSymbolTrainingProfile(
        symbol=symbol_text,
        company_name=("" if pd.isna(company_name) else str(company_name).strip()),
        sector=("" if pd.isna(sector) else str(sector).strip()),
        usable_observations=classification.usable_observations,
        history_class=classification.history_class,
        history_class_label=classification.label,
        current_mlp_ppo_ready=symbol_text in catalog.ready_symbols,
        future_training_route=classification.future_training_route,
    )


def future_history_class_counts(status_table: pd.DataFrame) -> dict[HistoryClass, int]:
    """Count future classes for active ordinary equities without changing readiness."""
    required = {"security_type", "usable_rows"}
    missing = sorted(required.difference(status_table.columns))
    if missing:
        raise ValueError(f"readiness table is missing: {', '.join(missing)}")
    source = status_table.loc[:, list(required)].copy(deep=True)
    ordinary = source.loc[source["security_type"].eq("ordinary_equity")]
    counts = {history_class: 0 for history_class in HistoryClass}
    for value in ordinary["usable_rows"]:
        classification = classify_usable_history(value)
        counts[classification.history_class] += 1
    return counts


def build_workflow_identity(
    summary: SelectedSymbolSummary,
    requested_timesteps: int,
    seed: int,
    requested_device: str,
) -> PPOWorkflowIdentity:
    """Build a normalized identity from requested—not resolved—configuration."""
    return PPOWorkflowIdentity(
        symbol=summary.symbol.strip(),
        requested_timesteps=requested_timesteps,
        seed=seed,
        requested_device=str(requested_device).strip().lower(),
        ppo_config_version=PPO_CONFIG_VERSION,
        contract_sha256=summary.contract_sha256,
        observation_scaler_sha256=summary.observation_scaler_sha256,
        observation_scaler_metadata_sha256=(
            summary.observation_scaler_metadata_sha256
        ),
        train_validation_artifact_fingerprint=(
            summary.train_validation_artifact_fingerprint
        ),
    )


def initialize_workflow_session(
    session: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Initialize session state without invoking any expensive action."""
    state = session.get(PPO_WORKFLOW_SESSION_KEY)
    if not isinstance(state, MutableMapping):
        state = {}
        session[PPO_WORKFLOW_SESSION_KEY] = state
    defaults: dict[str, object] = {
        "identity": None,
        "job_phase": "idle",
        "training_result": None,
        "validation_result": None,
        "persisted_bundle": None,
        "persisted_candidate_key": None,
        "training_error": None,
        "validation_error": None,
        "persistence_error": None,
        "progress": None,
    }
    for key, value in defaults.items():
        state.setdefault(key, value)
    if state["job_phase"] not in JOB_PHASES:
        state["job_phase"] = "idle"
    return state


def sync_workflow_identity(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
) -> bool:
    """Invalidate stale candidate state when any identity input changes."""
    state = initialize_workflow_session(session)
    previous = state.get("identity")
    if previous == identity:
        return False
    if state.get("job_phase") != "idle":
        raise RuntimeError("Cannot change PPO configuration while a job is running")
    for key in (
        "training_result",
        "validation_result",
        "persisted_bundle",
        "persisted_candidate_key",
        "training_error",
        "validation_error",
        "persistence_error",
        "progress",
    ):
        state[key] = None
    state["identity"] = identity
    return previous is not None


def _training_matches(result: object, identity: PPOWorkflowIdentity) -> bool:
    return bool(
        result is not None
        and getattr(result, "status", None) == "completed"
        and getattr(result, "model", None) is not None
        and getattr(result, "symbol", None) == identity.symbol
        and getattr(result, "seed", None) == identity.seed
        and getattr(result, "requested_timesteps", None)
        == identity.requested_timesteps
        and getattr(result, "requested_device", None) == identity.requested_device
        and getattr(result, "ppo_config_version", None) == identity.ppo_config_version
        and getattr(result, "source_rl_contract_sha256", None)
        == identity.contract_sha256
        and getattr(result, "source_observation_scaler_sha256", None)
        == identity.observation_scaler_sha256
        and getattr(result, "source_observation_scaler_metadata_sha256", None)
        == identity.observation_scaler_metadata_sha256
    )


def _candidate_key(
    identity: PPOWorkflowIdentity,
    validation_result: object,
) -> tuple[PPOWorkflowIdentity, str]:
    return identity, str(getattr(validation_result, "ppo_parameter_hash_after", ""))


def training_availability(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
) -> ActionAvailability:
    state = initialize_workflow_session(session)
    if state.get("identity") != identity:
        return ActionAvailability(False, "Configuration state is stale.")
    if state.get("job_phase") != "idle":
        return ActionAvailability(False, "Another PPO action is already running.")
    if _training_matches(state.get("training_result"), identity):
        return ActionAvailability(
            False,
            "This exact configuration already has an in-memory candidate.",
        )
    return ActionAvailability(True, "Ready for an explicit TRAIN-only run.")


def validation_availability(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
) -> ActionAvailability:
    state = initialize_workflow_session(session)
    if state.get("identity") != identity:
        return ActionAvailability(False, "Configuration state is stale.")
    if state.get("job_phase") != "idle":
        return ActionAvailability(False, "Another PPO action is already running.")
    if not _training_matches(state.get("training_result"), identity):
        return ActionAvailability(False, "Complete matching PPO training first.")
    if state.get("validation_result") is not None:
        return ActionAvailability(False, "This candidate was already validated.")
    return ActionAvailability(True, "Ready for explicit VALIDATION-only evaluation.")


def persistence_availability(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
) -> ActionAvailability:
    state = initialize_workflow_session(session)
    if state.get("identity") != identity:
        return ActionAvailability(False, "Configuration state is stale.")
    if state.get("job_phase") != "idle":
        return ActionAvailability(False, "Another PPO action is already running.")
    if not _training_matches(state.get("training_result"), identity):
        return ActionAvailability(False, "A matching trained candidate is required.")
    comparison = state.get("validation_result")
    if comparison is None:
        return ActionAvailability(False, "Complete validation first.")
    decision = getattr(comparison, "candidate_decision", None)
    if getattr(decision, "status", None) != "validation_pass":
        return ActionAvailability(
            False,
            "Only validation_pass can be saved as a production candidate.",
        )
    key = _candidate_key(identity, comparison)
    if state.get("persisted_candidate_key") == key:
        return ActionAvailability(False, "This candidate was already persisted.")
    return ActionAvailability(True, "Eligible for explicit candidate persistence.")


def claim_workflow_job(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
    phase: str,
) -> ActionAvailability:
    """Atomically claim one synchronous action within a Streamlit session."""
    state = initialize_workflow_session(session)
    if phase not in JOB_PHASES.difference({"idle"}):
        raise ValueError("job phase must be training, validating, or persisting")
    if state.get("identity") != identity:
        return ActionAvailability(False, "Configuration state is stale.")
    if state.get("job_phase") != "idle":
        return ActionAvailability(False, "Another PPO action is already running.")
    phase_gate = {
        "training": training_availability,
        "validating": validation_availability,
        "persisting": persistence_availability,
    }[phase](session, identity)
    if not phase_gate.allowed:
        return phase_gate
    state["job_phase"] = phase
    return ActionAvailability(True, f"{phase.capitalize()} job claimed.")


def release_workflow_job(session: MutableMapping[str, object]) -> None:
    """Always return the synchronous action guard to idle."""
    initialize_workflow_session(session)["job_phase"] = "idle"


def reset_workflow_results(session: MutableMapping[str, object]) -> None:
    """Discard only the current in-memory candidate and its derived results."""
    state = initialize_workflow_session(session)
    if state.get("job_phase") != "idle":
        raise RuntimeError("Cannot clear a PPO candidate while a job is running")
    for key in (
        "training_result",
        "validation_result",
        "persisted_bundle",
        "persisted_candidate_key",
        "training_error",
        "validation_error",
        "persistence_error",
        "progress",
    ):
        state[key] = None


def mark_candidate_persisted(
    session: MutableMapping[str, object],
    identity: PPOWorkflowIdentity,
    validation_result: object,
    bundle: object,
) -> None:
    state = initialize_workflow_session(session)
    if state.get("identity") != identity:
        raise ValueError("Cannot persist a stale workflow identity")
    state["persisted_candidate_key"] = _candidate_key(identity, validation_result)
    state["persisted_bundle"] = bundle


def run_training_action(
    identity: PPOWorkflowIdentity,
    *,
    progress_callback: Callable[[object], bool | None] | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    trainer: Callable[..., object] | None = None,
) -> object:
    """Run the production trainer with its fixed canonical TRAIN boundary."""
    if trainer is None:
        from reinforcement_learning.training.ppo_trainer import train_single_symbol

        trainer = train_single_symbol
    config = PPOConfig(
        seed=identity.seed,
        total_timesteps=identity.requested_timesteps,
        device=identity.requested_device,
    )
    return trainer(
        identity.symbol,
        config=config,
        output_dir=None,
        progress_callback=progress_callback,
        splits_dir=Path(splits_dir),
        smoke_test=False,
    )


def run_validation_action(
    training_result: object,
    identity: PPOWorkflowIdentity,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    evaluator: Callable[..., object] | None = None,
) -> object:
    """Run the fixed production VALIDATION comparison; TEST is not selectable."""
    if not _training_matches(training_result, identity):
        raise ValueError("Matching completed training is required for validation")
    if evaluator is None:
        from reinforcement_learning.evaluation.comparison import (
            compare_candidate_on_validation,
        )

        evaluator = compare_candidate_on_validation
    return evaluator(
        training_result.model,
        identity.symbol,
        trainer_result=training_result,
        deterministic_seed=identity.seed,
        random_seed=DEFAULT_SEED,
        splits_dir=Path(splits_dir),
    )


def run_persistence_action(
    training_result: object,
    validation_result: object,
    identity: PPOWorkflowIdentity,
    *,
    notes: str = "Saved explicitly from the 5B-5 Streamlit research workflow.",
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    persister: Callable[..., object] | None = None,
) -> object:
    """Persist only a validation-pass candidate; never promote it."""
    if not _training_matches(training_result, identity):
        raise ValueError("Matching completed training is required for persistence")
    decision = getattr(validation_result, "candidate_decision", None)
    if getattr(decision, "status", None) != "validation_pass":
        raise ValueError("Production candidate persistence requires validation_pass")
    if getattr(validation_result, "symbol", None) != identity.symbol:
        raise ValueError("Validation result symbol does not match the candidate")
    if persister is None:
        from reinforcement_learning.model_management.persistence import (
            persist_ppo_candidate,
        )

        persister = persist_ppo_candidate
    return persister(
        training_result,
        validation_result,
        symbol=identity.symbol,
        notes=notes,
        registry_path=Path(registry_path),
        saved_models_dir=Path(saved_models_dir),
        splits_dir=Path(splits_dir),
    )


def preview_candidate_version(
    registry: pd.DataFrame,
    symbol: str,
    *,
    saved_models_dir: Path = SAVED_MODELS_DIR,
) -> CandidateVersionPreview:
    """Inspect the likely next identity without acquiring or creating a lock."""
    from reinforcement_learning.model_management.persistence import (
        audit_registry_filesystem_consistency,
    )

    symbol_text = str(symbol).strip()
    audit = audit_registry_filesystem_consistency(
        registry=registry,
        model_scope="symbol",
        symbol=symbol_text,
        saved_models_dir=Path(saved_models_dir),
    )
    if not audit.consistent:
        raise ValueError(
            "Registry/filesystem versions disagree; reconcile before saving."
        )
    version = max((*audit.registry_versions, *audit.filesystem_versions), default=0) + 1
    identity = safe_path_component(symbol_text)
    return CandidateVersionPreview(
        model_id=f"ppo-symbol-{identity}-v{version:04d}",
        model_version=version,
    )


def pilot_readiness_table(
    ready_symbols: Iterable[str],
    *,
    rejected_reasons: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return the fixed future pilot in requested order with current readiness."""
    ready = {str(value).strip() for value in ready_symbols}
    reasons = rejected_reasons or {}
    records = []
    for symbol in PILOT_SYMBOLS:
        is_ready = symbol in ready
        records.append(
            {
                "Symbol": symbol,
                "Status": "Ready" if is_ready else "Not Ready",
                "Reason": (
                    "Validated RL contract"
                    if is_ready
                    else reasons.get(symbol, "Not in the current ready universe")
                ),
            }
        )
    return pd.DataFrame.from_records(records)


MODEL_HISTORY_COLUMNS = (
    "Model ID",
    "Symbol",
    "Version",
    "Algorithm",
    "Validation Status",
    "Promotion Status",
    "Environment Version",
    "Feature Version",
    "Created",
    "Last Trained",
    "Training Start",
    "Training End",
    "Validation Start",
    "Validation End",
    "Seed",
    "Model Status",
)


def registry_history_table(registry: pd.DataFrame) -> pd.DataFrame:
    """Format every registry version; never collapse history to latest only."""
    if registry.empty:
        return pd.DataFrame(columns=MODEL_HISTORY_COLUMNS)
    source = registry.copy(deep=True)
    required = {
        "model_id",
        "symbol",
        "model_version",
        "algorithm",
        "validation_status",
        "promotion_status",
        "environment_version",
        "feature_version",
        "created_at",
        "last_trained_at",
        "training_data_start",
        "training_data_end",
        "validation_data_start",
        "validation_data_end",
        "random_seed",
        "model_status",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"model registry is missing: {', '.join(missing)}")
    source = source.sort_values(
        ["created_at", "model_version"], ascending=False, kind="stable"
    )
    return pd.DataFrame(
        {
            "Model ID": source["model_id"].map(safe_display_value),
            "Symbol": source["symbol"].map(safe_display_value),
            "Version": source["model_version"].map(format_integer),
            "Algorithm": source["algorithm"].map(safe_display_value),
            "Validation Status": source["validation_status"].map(status_label),
            "Promotion Status": source["promotion_status"].map(status_label),
            "Environment Version": source["environment_version"].map(
                safe_display_value
            ),
            "Feature Version": source["feature_version"].map(safe_display_value),
            "Created": source["created_at"].map(format_datetime),
            "Last Trained": source["last_trained_at"].map(format_datetime),
            "Training Start": source["training_data_start"].map(format_date),
            "Training End": source["training_data_end"].map(format_date),
            "Validation Start": source["validation_data_start"].map(format_date),
            "Validation End": source["validation_data_end"].map(format_date),
            "Seed": source["random_seed"].map(format_integer),
            "Model Status": source["model_status"].map(status_label),
        }
    ).reset_index(drop=True)


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _format_metric(key: str, value: object) -> str:
    numeric = _finite(value)
    if numeric is None:
        return MISSING_VALUE
    if key in {
        "final_portfolio_value",
        "total_transaction_costs",
        "realized_profit_loss",
        "final_unrealized_profit_loss",
    }:
        return format_price(numeric)
    if key in {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    }:
        return format_percentage(numeric * 100.0, show_sign=False)
    if key == "exposure_percentage":
        return format_percentage(numeric, show_sign=False)
    if key == "number_of_trades":
        return format_integer(numeric)
    return format_decimal(numeric, precision=3)


VALIDATION_METRICS = (
    ("Final Portfolio Value", "final_portfolio_value"),
    ("Total Return", "total_return"),
    ("Annualized Return", "annualized_return"),
    ("Annualized Volatility", "annualized_volatility"),
    ("Sharpe", "sharpe_ratio"),
    ("Sortino", "sortino_ratio"),
    ("Maximum Drawdown", "maximum_drawdown"),
    ("Trades", "number_of_trades"),
    ("Transaction Costs", "total_transaction_costs"),
    ("Exposure", "exposure_percentage"),
    ("Realized P&L", "realized_profit_loss"),
    ("Unrealized P&L", "final_unrealized_profit_loss"),
)


def _strategies(result: object) -> tuple[tuple[str, object], ...]:
    return (
        ("PPO", getattr(result, "ppo")),
        ("Buy & Hold", getattr(result, "buy_and_hold")),
        ("Always Hold", getattr(result, "always_hold")),
        ("Random (seed 42)", getattr(result, "random")),
    )


def validation_metrics_table(result: object) -> pd.DataFrame:
    """Format the canonical comparison metrics without inventing values."""
    strategies = _strategies(result)
    records: list[dict[str, str]] = []
    for label, key in VALIDATION_METRICS:
        row = {"Metric": label}
        for strategy_name, strategy in strategies:
            metrics = getattr(strategy, "metrics", {})
            row[strategy_name] = _format_metric(key, metrics.get(key))
        records.append(row)
    return pd.DataFrame.from_records(records)


def validation_chart_frames(result: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return copied long-form portfolio/drawdown histories for native charts."""
    portfolio_frames: list[pd.DataFrame] = []
    drawdown_frames: list[pd.DataFrame] = []
    for strategy_name, strategy in _strategies(result):
        history = getattr(strategy, "history", pd.DataFrame()).copy(deep=True)
        required = {"execution_date", "portfolio_value", "drawdown"}
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(
                f"{strategy_name} validation history is missing: {', '.join(missing)}"
            )
        dates = pd.to_datetime(history["execution_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{strategy_name} validation history has invalid dates")
        portfolio_frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Portfolio Value": pd.to_numeric(
                        history["portfolio_value"], errors="coerce"
                    ),
                    "Strategy": strategy_name,
                }
            )
        )
        drawdown_frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Drawdown": pd.to_numeric(history["drawdown"], errors="coerce"),
                    "Strategy": strategy_name,
                }
            )
        )
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    drawdown = pd.concat(drawdown_frames, ignore_index=True)
    if not all(
        math.isfinite(value)
        for value in portfolio["Portfolio Value"].to_numpy(dtype=float)
    ):
        raise ValueError("validation portfolio history contains non-finite values")
    if not all(
        math.isfinite(value) for value in drawdown["Drawdown"].to_numpy(dtype=float)
    ):
        raise ValueError("validation drawdown history contains non-finite values")
    return portfolio, drawdown
