"""Versioned recurrent-RL metadata over canonical ``rl_partition_v1`` data.

The first recurrent baseline deliberately models each complete single-symbol
TRAIN or VALIDATION partition as one episode.  The module does not import a
recurrent algorithm and never exposes the sealed TEST frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from feature_engineering.schemas import FEATURE_VERSION
from feature_engineering.storage import atomic_write_json, safe_path_component

from .data_contract import (
    EXECUTION_ACCOUNTING_COLUMNS,
    IDENTITY_TIME_COLUMNS,
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
    RL_PARTITION_SCHEMA_VERSION,
    RLDataContractError,
    RLPartitionMetadata,
    load_rl_contract_metadata,
    load_rl_partition,
)
from .environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from .history_policy import HistoryClass, classify_usable_history
from .integrity import sha256_file


RL_RECURRENT_PARTITION_SCHEMA_VERSION = "rl_recurrent_partition_v1"
RECURRENT_EPISODE_SCHEMA_VERSION = "rl_recurrent_episode_boundaries_v1"
RECURRENT_DIRECTORY_NAME = "recurrent"
RECURRENT_CONTRACT_FILENAME = "recurrent_contract.json"
RECURRENT_EPISODE_BOUNDARIES_FILENAME = "episode_boundaries.json"
RECURRENT_LOADABLE_PARTITIONS = ("train", "validation")
RECURRENT_TRAINING_SCOPES = ("symbol", "sector", "generalized")
RECURRENT_NORMALIZATION_SCOPES = ("symbol", "sector", "global")
MINIMUM_SEQUENCE_ROWS = 2


class RecurrentDataContractError(ValueError):
    """Raised for an unavailable, stale, or incompatible recurrent artifact."""


@dataclass(frozen=True)
class RecurrentEligibility:
    """History-policy interpretation for future recurrent training."""

    history_class: HistoryClass
    history_class_label: str
    usable_observations: int
    recurrent_artifact_eligible: bool
    independent_recurrent_ready: bool
    transfer_fine_tune_eligible: bool
    reason: str


@dataclass(frozen=True)
class RecurrentEpisodeBoundary:
    """Inclusive row/date bounds for one recurrent episode."""

    episode_id: str
    symbol: str
    partition: str
    start_row: int
    end_row: int
    rows: int
    start: str
    end: str


@dataclass(frozen=True)
class RecurrentPartitionMetadata:
    """Metadata for a recurrent partition; TEST remains metadata-only."""

    name: str
    role: str
    rows: int
    start: str
    end: str
    episode_count: int
    sealed: bool


@dataclass(frozen=True)
class RecurrentContractMetadata:
    """Validated recurrent metadata without loading any partition frame."""

    symbol: str
    company: str
    sector: str | None
    sector_verified: bool
    contract_path: Path
    boundaries_path: Path
    recurrent_contract_version: str
    source_rl_contract_version: str
    feature_version: str
    environment_version: str
    observation_features: tuple[str, ...]
    dynamic_portfolio_features: tuple[str, ...]
    observation_shape: tuple[int, ...]
    execution_columns: tuple[str, ...]
    scaler_fit_partition: str
    normalization_scope: str
    training_scope: str
    universe_id: str
    universe_hash: str
    constituent_symbols: tuple[str, ...]
    cohort_cutoff: str
    sequence_length: int | None
    burn_in_length: int | None
    episode_length: int | None
    minimum_sequence_rows: int
    episode_strategy: str
    history: RecurrentEligibility
    train: RecurrentPartitionMetadata
    validation: RecurrentPartitionMetadata
    test: RecurrentPartitionMetadata


@dataclass(frozen=True)
class LoadedRecurrentPartition:
    """Canonical recurrent frame and its future episode-start representation."""

    symbol: str
    partition: str
    data: pd.DataFrame
    episode_start: np.ndarray
    episode_boundaries: tuple[RecurrentEpisodeBoundary, ...]
    metadata: RecurrentContractMetadata
    source_artifact_path: Path

    @property
    def reset_mask(self) -> np.ndarray:
        """Return an isolated copy of the Stable-Baselines-style reset mask."""

        return self.episode_start.copy()


@dataclass(frozen=True)
class RecurrentArtifactResult:
    """Files written for one single-symbol recurrent contract."""

    symbol: str
    recurrent_directory: Path
    contract_path: Path
    boundaries_path: Path
    contract: Mapping[str, object]


@dataclass(frozen=True)
class RecurrentArtifactBuildRecord:
    """One symbol's outcome during a recurrent-artifact migration."""

    symbol: str
    usable_observations: int
    history_class: HistoryClass
    generated: bool
    message: str
    contract_path: Path | None = None


@dataclass(frozen=True)
class RecurrentArtifactBuildSummary:
    """Auditable counts from a local recurrent-artifact migration."""

    mature_symbols_inspected: int
    recurrent_compatible_symbols_generated: int
    cold_start_symbols: int
    insufficient_symbols: int
    failures: int
    artifact_files_written: int
    records: tuple[RecurrentArtifactBuildRecord, ...]


def recurrent_eligibility(usable_observations: object) -> RecurrentEligibility:
    """Map the approved history policy to future recurrent eligibility."""

    classification = classify_usable_history(usable_observations)
    history_class = classification.history_class
    if history_class is HistoryClass.MATURE:
        return RecurrentEligibility(
            history_class=history_class,
            history_class_label=classification.label,
            usable_observations=classification.usable_observations,
            recurrent_artifact_eligible=True,
            independent_recurrent_ready=True,
            transfer_fine_tune_eligible=True,
            reason=(
                "Mature history is eligible for the single-symbol recurrent "
                "contract when all source artifacts validate."
            ),
        )
    if history_class is HistoryClass.COLD_START:
        return RecurrentEligibility(
            history_class=history_class,
            history_class_label=classification.label,
            usable_observations=classification.usable_observations,
            recurrent_artifact_eligible=False,
            independent_recurrent_ready=False,
            transfer_fine_tune_eligible=True,
            reason=(
                "Cold Start history is transfer/fine-tune eligible only; "
                "independent recurrent fitting is not approved."
            ),
        )
    return RecurrentEligibility(
        history_class=history_class,
        history_class_label=classification.label,
        usable_observations=classification.usable_observations,
        recurrent_artifact_eligible=False,
        independent_recurrent_ready=False,
        transfer_fine_tune_eligible=False,
        reason=(
            "Insufficient history is not eligible for symbol-specific recurrent "
            "fitting or fine-tuning."
        ),
    )


def recurrent_episode_start_mask(
    symbols: Sequence[object],
    partitions: Sequence[object],
    *,
    explicit_window_starts: Sequence[object] | None = None,
) -> np.ndarray:
    """Return deterministic reset flags for sequential recurrent observations.

    The first row always resets.  Every symbol or partition transition resets,
    and callers may additionally declare explicit window starts.  Feature values
    are intentionally not accepted, so future observations cannot affect masks.
    """

    if len(symbols) != len(partitions):
        raise RecurrentDataContractError(
            "symbols and partitions must have the same number of rows"
        )
    row_count = len(symbols)
    if row_count < 1:
        raise RecurrentDataContractError("an episode-start mask cannot be empty")
    if explicit_window_starts is None:
        window_starts = np.zeros(row_count, dtype=bool)
    else:
        if len(explicit_window_starts) != row_count:
            raise RecurrentDataContractError(
                "explicit window starts must match the sequential row count"
            )
        window_starts = np.asarray(explicit_window_starts, dtype=bool)

    symbol_values = tuple(str(value).strip() for value in symbols)
    partition_values = tuple(str(value).strip() for value in partitions)
    if any(not value for value in (*symbol_values, *partition_values)):
        raise RecurrentDataContractError(
            "symbols and partitions cannot contain empty values"
        )
    mask = window_starts.copy()
    mask[0] = True
    for index in range(1, row_count):
        if (
            symbol_values[index] != symbol_values[index - 1]
            or partition_values[index] != partition_values[index - 1]
        ):
            mask[index] = True
    return mask


def _canonical_json_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecurrentDataContractError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentDataContractError(
            f"{label} is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RecurrentDataContractError(f"{label} must contain a JSON object")
    return value


def _iso_date(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RecurrentDataContractError(f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise RecurrentDataContractError(f"{label} is invalid: {value!r}") from exc
    if parsed.isoformat() != value:
        raise RecurrentDataContractError(f"{label} must use YYYY-MM-DD")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecurrentDataContractError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentDataContractError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RecurrentDataContractError(f"{label} must be a lowercase SHA-256")
    return text


def _symbol_directory(symbol: str, splits_dir: Path) -> Path:
    return Path(splits_dir) / "symbols" / safe_path_component(symbol)


def _recurrent_paths(symbol: str, splits_dir: Path) -> tuple[Path, Path, Path]:
    directory = _symbol_directory(symbol, splits_dir) / RECURRENT_DIRECTORY_NAME
    return (
        directory,
        directory / RECURRENT_CONTRACT_FILENAME,
        directory / RECURRENT_EPISODE_BOUNDARIES_FILENAME,
    )


def _partition_payload(
    metadata: RLPartitionMetadata,
    *,
    role: str,
    sealed: bool,
    source_directory: Path,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rows": metadata.rows,
        "start": metadata.start,
        "end": metadata.end,
        "role": role,
        "episode_count": 1,
        "sealed": sealed,
    }
    if not sealed:
        raw_path = source_directory / f"{metadata.name}.csv"
        rl_path = source_directory / f"{metadata.name}_rl.csv"
        payload.update(
            {
                "source_raw_path": f"../{raw_path.name}",
                "source_raw_sha256": sha256_file(raw_path),
                "source_rl_path": f"../{rl_path.name}",
                "source_rl_sha256": sha256_file(rl_path),
            }
        )
    else:
        payload["frame_access"] = "sealed_metadata_only"
    return payload


def _episode_boundary(
    symbol: str,
    partition: RLPartitionMetadata,
) -> dict[str, object]:
    return {
        "episode_id": f"{symbol}:{partition.name}:0001",
        "symbol": symbol,
        "partition": partition.name,
        "start_row": 0,
        "end_row": partition.rows - 1,
        "rows": partition.rows,
        "start": partition.start,
        "end": partition.end,
    }


def persist_recurrent_contract(
    symbol: str,
    *,
    company: str,
    sector: str | None,
    sector_verified: bool,
    usable_observations: object,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    source_snapshot: Mapping[str, object] | None = None,
) -> RecurrentArtifactResult:
    """Validate canonical sources and write a single-symbol recurrent contract.

    Only TRAIN and VALIDATION frames are loaded.  TEST contributes bounds and
    row counts through metadata, and no TEST data file is opened by this path.
    """

    symbol_text = str(symbol).strip()
    company_text = str(company).strip()
    sector_text = str(sector).strip() if sector is not None else ""
    if not symbol_text:
        raise RecurrentDataContractError("symbol is required")
    if not company_text:
        raise RecurrentDataContractError("company is required")
    if not isinstance(sector_verified, bool):
        raise RecurrentDataContractError("sector_verified must be boolean")
    if sector_verified and not sector_text:
        raise RecurrentDataContractError(
            "a verified sector must include a non-empty sector value"
        )
    eligibility = recurrent_eligibility(usable_observations)
    if not eligibility.recurrent_artifact_eligible:
        raise RecurrentDataContractError(eligibility.reason)

    source_directory = _symbol_directory(symbol_text, Path(splits_dir))
    try:
        source_metadata = load_rl_contract_metadata(
            symbol_text,
            splits_dir=Path(splits_dir),
        )
        train_loaded = load_rl_partition(
            symbol_text,
            "train",
            splits_dir=Path(splits_dir),
        )
        validation_loaded = load_rl_partition(
            symbol_text,
            "validation",
            splits_dir=Path(splits_dir),
        )
    except RLDataContractError as exc:
        raise RecurrentDataContractError(
            f"Canonical RL source is incompatible for {symbol_text}: {exc}"
        ) from exc
    for name, loaded, expected in (
        ("train", train_loaded, source_metadata.train),
        ("validation", validation_loaded, source_metadata.validation),
    ):
        if len(loaded.data) != expected.rows or len(loaded.data) < MINIMUM_SEQUENCE_ROWS:
            raise RecurrentDataContractError(
                f"{name} does not satisfy recurrent row requirements"
            )
        dates = pd.to_datetime(loaded.data["date"], errors="coerce")
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise RecurrentDataContractError(
                f"{name} recurrent source is not strictly chronological"
            )

    constituent_symbols = (symbol_text,)
    cohort_cutoff = source_metadata.train.end
    universe_identity: dict[str, object] = {
        "training_scope": "symbol",
        "constituent_symbols": list(constituent_symbols),
        "sector": sector_text if sector_verified else None,
        "cohort_cutoff": cohort_cutoff,
    }
    universe_hash = _canonical_json_hash(universe_identity)
    universe_id = f"symbol:{symbol_text}:{cohort_cutoff}"

    _, contract_path, boundaries_path = _recurrent_paths(
        symbol_text, Path(splits_dir)
    )
    boundaries: dict[str, object] = {
        "artifact_schema_version": RECURRENT_EPISODE_SCHEMA_VERSION,
        "recurrent_contract_version": RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        "symbol": symbol_text,
        "episode_strategy": "full_partition",
        "partitions": {
            "train": [_episode_boundary(symbol_text, source_metadata.train)],
            "validation": [
                _episode_boundary(symbol_text, source_metadata.validation)
            ],
        },
    }
    atomic_write_json(boundaries, boundaries_path)

    scaler_path = source_directory / RL_OBSERVATION_SCALER_FILENAME
    scaler_metadata_path = scaler_path.with_suffix(".json")
    source_contract_path = source_directory / RL_CONTRACT_FILENAME
    snapshot = dict(source_snapshot or {})
    contract: dict[str, object] = {
        "artifact_schema_version": RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        "source_rl_contract_version": RL_PARTITION_SCHEMA_VERSION,
        "symbol": symbol_text,
        "company": company_text,
        "sector": sector_text if sector_verified else None,
        "sector_verified": sector_verified,
        "feature_version": source_metadata.feature_version,
        "environment_version": source_metadata.environment_version,
        "identity_time_columns": list(IDENTITY_TIME_COLUMNS),
        "execution_provenance": {
            "columns": list(EXECUTION_ACCOUNTING_COLUMNS),
            "values_unscaled": True,
            "source": "canonical raw partition companions from rl_partition_v1",
            "observation_at": "row_t",
            "execution_at": "row_t_plus_1_open",
            "mark_to_market_at": "row_t_plus_1_close",
            "lookahead_observations": 0,
        },
        "observation": {
            "market_features": list(source_metadata.observation_features),
            "dynamic_portfolio_features": list(
                source_metadata.dynamic_portfolio_features
            ),
            "shape": list(source_metadata.observation_shape),
            "dtype": "float32",
            "ordering": "market_features_then_dynamic_portfolio_features",
        },
        "normalization": {
            "normalization_scope": "symbol",
            "supported_future_scopes": list(RECURRENT_NORMALIZATION_SCOPES),
            "fit_partition": "train",
            "source_scaler_path": f"../{scaler_path.name}",
            "source_scaler_sha256": sha256_file(scaler_path),
            "source_scaler_metadata_path": f"../{scaler_metadata_path.name}",
            "source_scaler_metadata_sha256": sha256_file(scaler_metadata_path),
            "future_pooled_scaler_status": "not_implemented",
        },
        "sequence": {
            "episode_strategy": "full_partition",
            "sequence_length": None,
            "sequence_length_status": "future_configurable_not_applied",
            "burn_in_length": None,
            "burn_in_status": "not_used_in_v1",
            "episode_length": None,
            "episode_length_status": "complete_partition",
            "minimum_sequence_rows": MINIMUM_SEQUENCE_ROWS,
            "fixed_windows_enabled": False,
        },
        "reset_semantics": {
            "first_step_episode_start": True,
            "environment_reset_at_episode_start": True,
            "hidden_state_reset_at_episode_start": True,
            "reset_on_explicit_window_boundary": True,
            "reset_on_symbol_change": True,
            "reset_on_partition_change": True,
            "internal_reset_without_boundary": False,
        },
        "universe": {
            **universe_identity,
            "universe_id": universe_id,
            "universe_hash": universe_hash,
            "supported_training_scopes": list(RECURRENT_TRAINING_SCOPES),
            "historical_membership_fabricated": False,
        },
        "history_policy": {
            "history_class": eligibility.history_class.value,
            "history_class_label": eligibility.history_class_label,
            "usable_observations": eligibility.usable_observations,
            "recurrent_artifact_eligible": eligibility.recurrent_artifact_eligible,
            "independent_recurrent_ready": eligibility.independent_recurrent_ready,
            "transfer_fine_tune_eligible": eligibility.transfer_fine_tune_eligible,
            "reason": eligibility.reason,
        },
        "source_snapshot": {
            "rl_contract_path": f"../{source_contract_path.name}",
            "rl_contract_sha256": sha256_file(source_contract_path),
            "cohort_source": snapshot,
        },
        "episode_boundaries": {
            "path": RECURRENT_EPISODE_BOUNDARIES_FILENAME,
            "sha256": sha256_file(boundaries_path),
            "schema_version": RECURRENT_EPISODE_SCHEMA_VERSION,
        },
        "partitions": {
            "train": _partition_payload(
                source_metadata.train,
                role="future_recurrent_learning_only",
                sealed=False,
                source_directory=source_directory,
            ),
            "validation": _partition_payload(
                source_metadata.validation,
                role="future_recurrent_evaluation_only",
                sealed=False,
                source_directory=source_directory,
            ),
            "test": _partition_payload(
                source_metadata.test,
                role="sealed_final_evaluation",
                sealed=True,
                source_directory=source_directory,
            ),
        },
        "test_sealing": {
            "sealed": True,
            "metadata_only": True,
            "frame_loaded_during_build": False,
            "routine_loader_access": "prohibited",
            "evaluation_performed": False,
        },
    }
    atomic_write_json(contract, contract_path)
    return RecurrentArtifactResult(
        symbol=symbol_text,
        recurrent_directory=contract_path.parent,
        contract_path=contract_path,
        boundaries_path=boundaries_path,
        contract=contract,
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentDataContractError(f"{label} must be a JSON object")
    return value


def _sequence_optional_integer(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, label=label)


def _validate_recurrent_partition_metadata(
    name: str,
    payload: Mapping[str, object],
    source: RLPartitionMetadata,
) -> RecurrentPartitionMetadata:
    rows = _positive_integer(payload.get("rows"), label=f"{name} rows")
    start = _iso_date(payload.get("start"), label=f"{name} start")
    end = _iso_date(payload.get("end"), label=f"{name} end")
    if (rows, start, end) != (source.rows, source.start, source.end):
        raise RecurrentDataContractError(
            f"recurrent {name} metadata differs from rl_partition_v1"
        )
    role = str(payload.get("role", ""))
    episode_count = _positive_integer(
        payload.get("episode_count"), label=f"{name} episode count"
    )
    sealed = payload.get("sealed")
    if not isinstance(sealed, bool):
        raise RecurrentDataContractError(f"{name} sealed flag must be boolean")
    if name == "test":
        if not sealed or payload.get("frame_access") != "sealed_metadata_only":
            raise RecurrentDataContractError("TEST must remain sealed metadata-only")
        forbidden = {"source_raw_path", "source_rl_path"}.intersection(payload)
        if forbidden:
            raise RecurrentDataContractError(
                "sealed TEST metadata cannot expose recurrent frame paths"
            )
    elif sealed:
        raise RecurrentDataContractError(f"{name} cannot be sealed")
    return RecurrentPartitionMetadata(
        name=name,
        role=role,
        rows=rows,
        start=start,
        end=end,
        episode_count=episode_count,
        sealed=sealed,
    )


def _validate_source_file(
    directory: Path,
    payload: Mapping[str, object],
    *,
    path_field: str,
    hash_field: str,
    expected_name: str,
    label: str,
) -> Path:
    relative = payload.get(path_field)
    if relative != f"../{expected_name}":
        raise RecurrentDataContractError(f"{label} source path is incompatible")
    path = directory.parent / expected_name
    if not path.is_file():
        raise RecurrentDataContractError(f"{label} source is missing: {path}")
    expected_hash = _sha256(payload.get(hash_field), label=f"{label} hash")
    if sha256_file(path) != expected_hash:
        raise RecurrentDataContractError(f"{label} source hash is stale")
    return path


def _load_boundaries(
    path: Path,
    *,
    symbol: str,
    recurrent_contract_version: str,
) -> dict[str, object]:
    payload = _load_json(path, label="recurrent episode boundaries")
    if payload.get("artifact_schema_version") != RECURRENT_EPISODE_SCHEMA_VERSION:
        raise RecurrentDataContractError(
            "incompatible recurrent episode-boundary schema version"
        )
    if payload.get("recurrent_contract_version") != recurrent_contract_version:
        raise RecurrentDataContractError(
            "episode boundaries and recurrent contract versions differ"
        )
    if payload.get("symbol") != symbol:
        raise RecurrentDataContractError(
            "episode-boundary symbol differs from recurrent contract"
        )
    if payload.get("episode_strategy") != "full_partition":
        raise RecurrentDataContractError("unsupported recurrent episode strategy")
    return payload


def _validated_episode_boundaries(
    payload: Mapping[str, object],
    *,
    symbol: str,
    partition: RecurrentPartitionMetadata,
) -> tuple[RecurrentEpisodeBoundary, ...]:
    partitions = _mapping(payload.get("partitions"), label="episode partitions")
    values = partitions.get(partition.name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RecurrentDataContractError(
            f"episode boundaries have no {partition.name!r} sequence"
        )
    boundaries: list[RecurrentEpisodeBoundary] = []
    for index, raw_value in enumerate(values):
        value = _mapping(raw_value, label=f"{partition.name} episode {index + 1}")
        boundary = RecurrentEpisodeBoundary(
            episode_id=str(value.get("episode_id", "")),
            symbol=str(value.get("symbol", "")),
            partition=str(value.get("partition", "")),
            start_row=_non_negative_integer(
                value.get("start_row"), label="episode start row"
            ),
            end_row=_non_negative_integer(
                value.get("end_row"), label="episode end row"
            ),
            rows=_positive_integer(value.get("rows"), label="episode rows"),
            start=_iso_date(value.get("start"), label="episode start"),
            end=_iso_date(value.get("end"), label="episode end"),
        )
        boundaries.append(boundary)
    if len(boundaries) != partition.episode_count:
        raise RecurrentDataContractError("episode count differs from partition metadata")
    expected = RecurrentEpisodeBoundary(
        episode_id=f"{symbol}:{partition.name}:0001",
        symbol=symbol,
        partition=partition.name,
        start_row=0,
        end_row=partition.rows - 1,
        rows=partition.rows,
        start=partition.start,
        end=partition.end,
    )
    if tuple(boundaries) != (expected,):
        raise RecurrentDataContractError(
            f"{partition.name} must be exactly one complete partition episode"
        )
    return tuple(boundaries)


def _load_validated_recurrent_metadata(
    symbol: str,
    *,
    splits_dir: Path,
) -> tuple[RecurrentContractMetadata, dict[str, object], dict[str, object]]:
    symbol_text = str(symbol).strip()
    if not symbol_text:
        raise RecurrentDataContractError("symbol is required")
    directory, contract_path, boundaries_path = _recurrent_paths(
        symbol_text, Path(splits_dir)
    )
    contract = _load_json(contract_path, label="recurrent RL contract")
    if contract.get("artifact_schema_version") != RL_RECURRENT_PARTITION_SCHEMA_VERSION:
        raise RecurrentDataContractError(
            "incompatible recurrent RL contract version: expected "
            f"{RL_RECURRENT_PARTITION_SCHEMA_VERSION!r}"
        )
    if contract.get("source_rl_contract_version") != RL_PARTITION_SCHEMA_VERSION:
        raise RecurrentDataContractError(
            "recurrent contract references an incompatible rl_partition version"
        )
    if contract.get("symbol") != symbol_text:
        raise RecurrentDataContractError("recurrent contract symbol is incompatible")
    if contract.get("feature_version") != FEATURE_VERSION:
        raise RecurrentDataContractError("recurrent contract feature version is stale")
    if contract.get("environment_version") != ENVIRONMENT_VERSION:
        raise RecurrentDataContractError(
            "recurrent contract environment version is stale"
        )
    if tuple(contract.get("identity_time_columns", ())) != IDENTITY_TIME_COLUMNS:
        raise RecurrentDataContractError("recurrent identity/time schema is incompatible")

    observation = _mapping(contract.get("observation"), label="observation metadata")
    if tuple(observation.get("market_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise RecurrentDataContractError(
            "recurrent observation feature order is incompatible"
        )
    if tuple(observation.get("dynamic_portfolio_features", ())) != DYNAMIC_PORTFOLIO_FEATURES:
        raise RecurrentDataContractError(
            "recurrent dynamic feature order is incompatible"
        )
    expected_shape = (
        len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
    )
    if tuple(observation.get("shape", ())) != expected_shape:
        raise RecurrentDataContractError("recurrent observation shape is incompatible")

    execution = _mapping(
        contract.get("execution_provenance"), label="execution provenance"
    )
    if tuple(execution.get("columns", ())) != EXECUTION_ACCOUNTING_COLUMNS:
        raise RecurrentDataContractError("recurrent execution columns are incompatible")
    if execution.get("values_unscaled") is not True or execution.get("lookahead_observations") != 0:
        raise RecurrentDataContractError(
            "recurrent execution/look-ahead provenance is incompatible"
        )

    normalization = _mapping(
        contract.get("normalization"), label="normalization metadata"
    )
    if normalization.get("normalization_scope") != "symbol":
        raise RecurrentDataContractError(
            "recurrent v1 normalization scope must be 'symbol'"
        )
    if normalization.get("fit_partition") != "train":
        raise RecurrentDataContractError("recurrent scaler must be TRAIN-fitted")
    _validate_source_file(
        directory,
        normalization,
        path_field="source_scaler_path",
        hash_field="source_scaler_sha256",
        expected_name=RL_OBSERVATION_SCALER_FILENAME,
        label="observation scaler",
    )
    _validate_source_file(
        directory,
        normalization,
        path_field="source_scaler_metadata_path",
        hash_field="source_scaler_metadata_sha256",
        expected_name=RL_OBSERVATION_SCALER_FILENAME.replace(".joblib", ".json"),
        label="observation scaler metadata",
    )

    sequence = _mapping(contract.get("sequence"), label="sequence metadata")
    if sequence.get("episode_strategy") != "full_partition":
        raise RecurrentDataContractError("recurrent v1 requires full partitions")
    sequence_length = _sequence_optional_integer(
        sequence.get("sequence_length"), label="sequence length"
    )
    burn_in_length = _sequence_optional_integer(
        sequence.get("burn_in_length"), label="burn-in length"
    )
    episode_length = _sequence_optional_integer(
        sequence.get("episode_length"), label="episode length"
    )
    if any(value is not None for value in (sequence_length, burn_in_length, episode_length)):
        raise RecurrentDataContractError(
            "recurrent v1 does not configure fixed sequences, burn-in, or windows"
        )
    minimum_sequence_rows = _positive_integer(
        sequence.get("minimum_sequence_rows"), label="minimum sequence rows"
    )
    if minimum_sequence_rows != MINIMUM_SEQUENCE_ROWS:
        raise RecurrentDataContractError("minimum sequence rows are incompatible")

    resets = _mapping(contract.get("reset_semantics"), label="reset semantics")
    required_true = (
        "first_step_episode_start",
        "environment_reset_at_episode_start",
        "hidden_state_reset_at_episode_start",
        "reset_on_explicit_window_boundary",
        "reset_on_symbol_change",
        "reset_on_partition_change",
    )
    if any(resets.get(field) is not True for field in required_true) or resets.get(
        "internal_reset_without_boundary"
    ) is not False:
        raise RecurrentDataContractError("recurrent reset semantics are incompatible")

    source_metadata = load_rl_contract_metadata(symbol_text, splits_dir=Path(splits_dir))
    snapshot = _mapping(contract.get("source_snapshot"), label="source snapshot")
    _validate_source_file(
        directory,
        snapshot,
        path_field="rl_contract_path",
        hash_field="rl_contract_sha256",
        expected_name=RL_CONTRACT_FILENAME,
        label="rl_partition_v1 contract",
    )
    if source_metadata.feature_version != contract.get("feature_version"):
        raise RecurrentDataContractError("source and recurrent feature versions differ")

    partitions = _mapping(contract.get("partitions"), label="partition metadata")
    train = _validate_recurrent_partition_metadata(
        "train",
        _mapping(partitions.get("train"), label="train partition"),
        source_metadata.train,
    )
    validation = _validate_recurrent_partition_metadata(
        "validation",
        _mapping(partitions.get("validation"), label="validation partition"),
        source_metadata.validation,
    )
    test = _validate_recurrent_partition_metadata(
        "test",
        _mapping(partitions.get("test"), label="TEST partition"),
        source_metadata.test,
    )
    if not (date.fromisoformat(train.end) < date.fromisoformat(validation.start)):
        raise RecurrentDataContractError("TRAIN and VALIDATION boundaries overlap")
    if not (date.fromisoformat(validation.end) < date.fromisoformat(test.start)):
        raise RecurrentDataContractError("VALIDATION and TEST boundaries overlap")
    for name in RECURRENT_LOADABLE_PARTITIONS:
        partition_payload = _mapping(partitions[name], label=f"{name} partition")
        _validate_source_file(
            directory,
            partition_payload,
            path_field="source_raw_path",
            hash_field="source_raw_sha256",
            expected_name=f"{name}.csv",
            label=f"{name} raw partition",
        )
        _validate_source_file(
            directory,
            partition_payload,
            path_field="source_rl_path",
            hash_field="source_rl_sha256",
            expected_name=f"{name}_rl.csv",
            label=f"{name} RL partition",
        )

    boundaries_reference = _mapping(
        contract.get("episode_boundaries"), label="episode-boundary reference"
    )
    if boundaries_reference.get("path") != RECURRENT_EPISODE_BOUNDARIES_FILENAME:
        raise RecurrentDataContractError("episode-boundary path is incompatible")
    if boundaries_reference.get("schema_version") != RECURRENT_EPISODE_SCHEMA_VERSION:
        raise RecurrentDataContractError("episode-boundary version is incompatible")
    expected_boundaries_hash = _sha256(
        boundaries_reference.get("sha256"), label="episode-boundary hash"
    )
    if sha256_file(boundaries_path) != expected_boundaries_hash:
        raise RecurrentDataContractError("episode-boundary artifact hash is stale")
    boundaries = _load_boundaries(
        boundaries_path,
        symbol=symbol_text,
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    )
    _validated_episode_boundaries(
        boundaries, symbol=symbol_text, partition=train
    )
    _validated_episode_boundaries(
        boundaries, symbol=symbol_text, partition=validation
    )

    history = _mapping(contract.get("history_policy"), label="history policy")
    eligibility = recurrent_eligibility(history.get("usable_observations"))
    if (
        history.get("history_class") != eligibility.history_class.value
        or history.get("recurrent_artifact_eligible")
        != eligibility.recurrent_artifact_eligible
        or history.get("independent_recurrent_ready")
        != eligibility.independent_recurrent_ready
        or history.get("transfer_fine_tune_eligible")
        != eligibility.transfer_fine_tune_eligible
    ):
        raise RecurrentDataContractError("recurrent history-policy metadata is stale")
    if eligibility.history_class is not HistoryClass.MATURE:
        raise RecurrentDataContractError("recurrent artifact is not Mature-eligible")

    universe = _mapping(contract.get("universe"), label="universe metadata")
    if universe.get("training_scope") != "symbol":
        raise RecurrentDataContractError("single-symbol recurrent scope is incompatible")
    constituents = tuple(str(value) for value in universe.get("constituent_symbols", ()))
    if constituents != (symbol_text,):
        raise RecurrentDataContractError("recurrent constituent symbols are incompatible")
    universe_identity: dict[str, object] = {
        "training_scope": "symbol",
        "constituent_symbols": [symbol_text],
        "sector": contract.get("sector") if contract.get("sector_verified") is True else None,
        "cohort_cutoff": train.end,
    }
    expected_universe_hash = _canonical_json_hash(universe_identity)
    if universe.get("universe_hash") != expected_universe_hash:
        raise RecurrentDataContractError("recurrent universe hash is incompatible")
    expected_universe_id = f"symbol:{symbol_text}:{train.end}"
    if universe.get("universe_id") != expected_universe_id:
        raise RecurrentDataContractError("recurrent universe identifier is incompatible")
    if universe.get("cohort_cutoff") != train.end:
        raise RecurrentDataContractError("recurrent cohort cutoff is incompatible")
    sector_verified = contract.get("sector_verified")
    if not isinstance(sector_verified, bool):
        raise RecurrentDataContractError("sector verification metadata is invalid")
    if sector_verified and not str(contract.get("sector", "")).strip():
        raise RecurrentDataContractError("verified recurrent sector is missing")
    if not sector_verified and contract.get("sector") is not None:
        raise RecurrentDataContractError("unverified recurrent sector must be omitted")

    metadata = RecurrentContractMetadata(
        symbol=symbol_text,
        company=str(contract.get("company", "")),
        sector=(str(contract["sector"]) if contract.get("sector") is not None else None),
        sector_verified=sector_verified,
        contract_path=contract_path.resolve(),
        boundaries_path=boundaries_path.resolve(),
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        source_rl_contract_version=RL_PARTITION_SCHEMA_VERSION,
        feature_version=str(contract["feature_version"]),
        environment_version=str(contract["environment_version"]),
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        dynamic_portfolio_features=DYNAMIC_PORTFOLIO_FEATURES,
        observation_shape=expected_shape,
        execution_columns=EXECUTION_ACCOUNTING_COLUMNS,
        scaler_fit_partition="train",
        normalization_scope="symbol",
        training_scope="symbol",
        universe_id=expected_universe_id,
        universe_hash=expected_universe_hash,
        constituent_symbols=constituents,
        cohort_cutoff=_iso_date(universe.get("cohort_cutoff"), label="cohort cutoff"),
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        episode_length=episode_length,
        minimum_sequence_rows=minimum_sequence_rows,
        episode_strategy="full_partition",
        history=eligibility,
        train=train,
        validation=validation,
        test=test,
    )
    if not metadata.company.strip():
        raise RecurrentDataContractError("recurrent company metadata is missing")
    return metadata, contract, boundaries


def load_recurrent_contract_metadata(
    symbol: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> RecurrentContractMetadata:
    """Load recurrent metadata while keeping every partition frame unopened."""

    metadata, _, _ = _load_validated_recurrent_metadata(
        symbol, splits_dir=Path(splits_dir)
    )
    return metadata


def load_recurrent_partition(
    symbol: str,
    partition: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> LoadedRecurrentPartition:
    """Load canonical TRAIN or VALIDATION data and deterministic reset flags.

    TEST is deliberately absent from the loader API.  Its row/date metadata can
    be inspected through :func:`load_recurrent_contract_metadata` only.
    """

    if partition not in RECURRENT_LOADABLE_PARTITIONS:
        raise RecurrentDataContractError(
            "recurrent partition must be 'train' or 'validation'; TEST is sealed"
        )
    metadata, contract, boundaries_payload = _load_validated_recurrent_metadata(
        symbol, splits_dir=Path(splits_dir)
    )
    partition_metadata = getattr(metadata, partition)
    boundaries = _validated_episode_boundaries(
        boundaries_payload,
        symbol=metadata.symbol,
        partition=partition_metadata,
    )
    try:
        source = load_rl_partition(
            metadata.symbol,
            partition,
            splits_dir=Path(splits_dir),
        )
    except RLDataContractError as exc:
        raise RecurrentDataContractError(
            f"Could not load canonical {partition} recurrent source: {exc}"
        ) from exc
    data = source.data.copy(deep=True)
    if len(data) != partition_metadata.rows:
        raise RecurrentDataContractError(
            f"{partition} recurrent row count differs from contract"
        )
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise RecurrentDataContractError(
            f"{partition} recurrent rows are not chronological"
        )
    if (
        dates.iloc[0].date().isoformat(),
        dates.iloc[-1].date().isoformat(),
    ) != (partition_metadata.start, partition_metadata.end):
        raise RecurrentDataContractError(
            f"{partition} recurrent date range differs from contract"
        )
    mask = recurrent_episode_start_mask(
        data["symbol"].astype("string").tolist(),
        [partition] * len(data),
    )
    expected_starts = {boundary.start_row for boundary in boundaries}
    if set(np.flatnonzero(mask)) != expected_starts:
        raise RecurrentDataContractError(
            f"{partition} reset mask differs from episode boundaries"
        )
    partition_payload = _mapping(
        _mapping(contract.get("partitions"), label="partitions").get(partition),
        label=f"{partition} partition",
    )
    source_name = Path(str(partition_payload["source_rl_path"])).name
    return LoadedRecurrentPartition(
        symbol=metadata.symbol,
        partition=partition,
        data=data,
        episode_start=mask,
        episode_boundaries=boundaries,
        metadata=metadata,
        source_artifact_path=(metadata.contract_path.parent.parent / source_name),
    )


def build_recurrent_artifacts(
    status_table: pd.DataFrame,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    sector_source_path: Path | None = None,
) -> RecurrentArtifactBuildSummary:
    """Build recurrent metadata for Mature active ordinary equities only."""

    required = {
        "symbol",
        "company_name",
        "sector",
        "security_type",
        "is_active",
        "usable_rows",
    }
    missing = sorted(required.difference(status_table.columns))
    if missing:
        raise RecurrentDataContractError(
            f"status table is missing recurrent columns: {', '.join(missing)}"
        )
    source = status_table.loc[:, list(required)].copy(deep=True)
    source["symbol"] = source["symbol"].astype("string").str.strip()
    active = source["is_active"].map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true"}
        )
    )
    source = source.loc[
        source["security_type"].eq("ordinary_equity") & active
    ]
    if source["symbol"].isna().any() or source["symbol"].eq("").any():
        raise RecurrentDataContractError("status table contains an empty symbol")
    if source["symbol"].duplicated().any():
        raise RecurrentDataContractError("status table contains duplicate symbols")

    snapshot: dict[str, object] = {}
    source_path = Path(sector_source_path) if sector_source_path is not None else None
    if source_path is not None:
        if not source_path.is_file():
            raise RecurrentDataContractError(
                f"sector source snapshot is missing: {source_path}"
            )
        snapshot = {
            "company_registry_source": str(source_path.resolve()),
            "company_registry_sha256": sha256_file(source_path),
        }

    records: list[RecurrentArtifactBuildRecord] = []
    mature_count = cold_count = insufficient_count = generated_count = failure_count = 0
    for row in source.sort_values("symbol", kind="stable").itertuples(index=False):
        symbol = str(row.symbol)
        eligibility = recurrent_eligibility(row.usable_rows)
        if eligibility.history_class is HistoryClass.COLD_START:
            cold_count += 1
            records.append(
                RecurrentArtifactBuildRecord(
                    symbol,
                    eligibility.usable_observations,
                    eligibility.history_class,
                    False,
                    eligibility.reason,
                )
            )
            continue
        if eligibility.history_class is HistoryClass.INSUFFICIENT:
            insufficient_count += 1
            records.append(
                RecurrentArtifactBuildRecord(
                    symbol,
                    eligibility.usable_observations,
                    eligibility.history_class,
                    False,
                    eligibility.reason,
                )
            )
            continue
        mature_count += 1
        sector = "" if pd.isna(row.sector) else str(row.sector).strip()
        try:
            result = persist_recurrent_contract(
                symbol,
                company=str(row.company_name).strip(),
                sector=sector or None,
                sector_verified=bool(sector and source_path is not None),
                usable_observations=eligibility.usable_observations,
                splits_dir=Path(splits_dir),
                source_snapshot=snapshot,
            )
        except (RecurrentDataContractError, OSError, ValueError) as exc:
            failure_count += 1
            records.append(
                RecurrentArtifactBuildRecord(
                    symbol,
                    eligibility.usable_observations,
                    eligibility.history_class,
                    False,
                    str(exc),
                )
            )
        else:
            generated_count += 1
            records.append(
                RecurrentArtifactBuildRecord(
                    symbol,
                    eligibility.usable_observations,
                    eligibility.history_class,
                    True,
                    "Generated and validated recurrent contract.",
                    result.contract_path,
                )
            )
    return RecurrentArtifactBuildSummary(
        mature_symbols_inspected=mature_count,
        recurrent_compatible_symbols_generated=generated_count,
        cold_start_symbols=cold_count,
        insufficient_symbols=insufficient_count,
        failures=failure_count,
        artifact_files_written=generated_count * 2,
        records=tuple(records),
    )
