"""Canonical Parquet-backed, TRAIN-only recurrent artifacts.

This successor path is intentionally isolated from ``rl_partition_v1`` and
``rl_recurrent_partition_v1``.  It uses complete symbol/date metadata only to
freeze a chronological boundary, predicate-loads market values through that
boundary, and persists no VALIDATION or TEST frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_pipeline.src.config import (
    CANONICAL_RECURRENT_TRAIN_V2_DIR,
    PROJECT_ROOT,
    PROCESSED_SPLITS_DIR,
)
from data_pipeline.src.parquet_market_data import (
    load_market_data,
    load_symbol_market_date_inventory,
    resolve_market_parquet_path,
)
from feature_engineering.indicators import calculate_features
from feature_engineering.preprocessing import filter_ai_quality_rows
from feature_engineering.schemas import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_WARMUP_ROWS
from feature_engineering.storage import (
    atomic_dump_joblib,
    atomic_write_dataframe,
    atomic_write_json,
    safe_path_component,
)
from reinforcement_learning.data_contract import (
    EXECUTION_ACCOUNTING_COLUMNS,
    IDENTITY_TIME_COLUMNS,
    scaled_observation_column,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.history_policy import (
    HistoryClass,
    MATURE_MINIMUM_USABLE_OBSERVATIONS,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    LoadedRecurrentPartition,
    RecurrentContractMetadata,
    RecurrentDataContractError,
    RecurrentEligibility,
    RecurrentEpisodeBoundary,
    RecurrentPartitionMetadata,
    load_recurrent_contract_metadata,
    load_recurrent_partition,
    recurrent_eligibility,
    recurrent_episode_start_mask,
)


CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION = "canonical_symbol_features_v2"
RL_TRAIN_PARTITION_SCHEMA_VERSION = "rl_train_partition_v2"
RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION = "rl_recurrent_train_partition_v2"
CANONICAL_TRAIN_HISTORY_POLICY_VERSION = "canonical_recurrent_train_history_v2"
CANONICAL_EPISODE_BOUNDARY_VERSION = "rl_recurrent_train_episode_boundaries_v2"
CANONICAL_TRAIN_BOUNDARY_POLICY = "symbol_raw_market_dates_first_70_percent_v1"
CANONICAL_NORMALIZATION_SCOPE = "symbol_train_only"

CANONICAL_TRAIN_MINIMUM_USABLE_ROWS = MATURE_MINIMUM_USABLE_OBSERVATIONS
MECHANICAL_MINIMUM_USABLE_ROWS = 2
MINIMUM_RETAINED_ROWS_FOR_MATURE_TRAIN = (
    FEATURE_WARMUP_ROWS + CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
)
SUPPORTED_COMMON_EQUITY_SECURITY_TYPES = frozenset(
    {"ordinary_equity", "gem_equity"}
)

FEATURE_FILENAME = "train_features.csv"
RL_FILENAME = "train_rl.csv"
SCALER_FILENAME = "observation_scaler.joblib"
SCALER_METADATA_FILENAME = "observation_scaler.json"
BOUNDARIES_FILENAME = "episode_boundaries.json"
CONTRACT_FILENAME = "recurrent_train_contract.json"
RECOVERY_AUDIT_FILENAME = "recovery_audit.json"


class CanonicalRecurrentArtifactError(RecurrentDataContractError):
    """Raised when a TRAIN-only v2 artifact would violate its contract."""


@dataclass(frozen=True)
class CanonicalTrainEvidence:
    symbol: str
    company_name: str
    security_type: str
    train_cutoff: str
    canonical_train_rows: int
    quality_usable_ohlcv_rows: int
    quality_rows_removed: int
    feature_warmup_loss: int
    missing_feature_rows: int
    final_usable_feature_rows: int
    first_train_feature_date: str
    last_train_feature_date: str
    recurrent_episode_feasible: bool
    scaler_fit_feasible: bool
    recovery_valid: bool
    blocker: str


@dataclass(frozen=True)
class PreparedCanonicalTrain:
    evidence: CanonicalTrainEvidence
    features: pd.DataFrame


@dataclass(frozen=True)
class CanonicalArtifactBuildResult:
    symbol: str
    artifact_directory: Path
    contract_path: Path
    train_features_path: Path
    train_rl_path: Path
    scaler_path: Path
    scaler_metadata_path: Path
    boundaries_path: Path
    contract: Mapping[str, object]


@dataclass(frozen=True)
class CanonicalRecoverySummary:
    evidence: tuple[CanonicalTrainEvidence, ...]
    generated: tuple[CanonicalArtifactBuildResult, ...]

    @property
    def recovered_symbols(self) -> tuple[str, ...]:
        return tuple(result.symbol for result in self.generated)


def write_canonical_recovery_manifest(
    summary: CanonicalRecoverySummary,
    *,
    artifacts_dir: Path,
    identity_universe_hash: str,
    source_parquet_sha256: str,
) -> Path:
    """Persist deterministic eligibility evidence for discovery and audit."""

    payload = {
        "artifact_schema_version": "canonical_recurrent_recovery_audit_v1",
        "feature_version": CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
        "recurrent_contract_version": RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
        "history_policy_version": CANONICAL_TRAIN_HISTORY_POLICY_VERSION,
        "minimum_usable_train_rows": CANONICAL_TRAIN_MINIMUM_USABLE_ROWS,
        "identity_universe_hash": identity_universe_hash,
        "source_parquet_sha256": source_parquet_sha256,
        "recovered_symbols": list(summary.recovered_symbols),
        "evidence": [asdict(record) for record in summary.evidence],
    }
    path = Path(artifacts_dir) / RECOVERY_AUDIT_FILENAME
    atomic_write_json(payload, path)
    return path


def load_canonical_recovery_evidence(
    *,
    artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> dict[str, Mapping[str, object]]:
    """Load deterministic non-contract recovery evidence when available."""

    path = Path(artifacts_dir) / RECOVERY_AUDIT_FILENAME
    if not path.is_file():
        return {}
    payload = _load_json(path, label="canonical recurrent recovery audit")
    if (
        payload.get("artifact_schema_version")
        != "canonical_recurrent_recovery_audit_v1"
        or payload.get("history_policy_version")
        != CANONICAL_TRAIN_HISTORY_POLICY_VERSION
        or int(payload.get("minimum_usable_train_rows", 0))
        != CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
    ):
        raise CanonicalRecurrentArtifactError(
            "canonical recurrent recovery audit is incompatible"
        )
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise CanonicalRecurrentArtifactError(
            "canonical recurrent recovery evidence must be a list"
        )
    records: dict[str, Mapping[str, object]] = {}
    for raw in evidence:
        if not isinstance(raw, Mapping):
            raise CanonicalRecurrentArtifactError(
                "canonical recurrent recovery evidence row is invalid"
            )
        symbol = str(raw.get("symbol", "")).strip()
        if not symbol or symbol in records:
            raise CanonicalRecurrentArtifactError(
                "canonical recurrent recovery evidence symbols are invalid"
            )
        records[symbol] = raw
    return records


@dataclass(frozen=True)
class CanonicalRecurrentTrainMetadata:
    symbol: str
    company: str
    security_type: str
    contract_path: Path
    boundaries_path: Path
    recurrent_contract_version: str
    source_rl_contract_version: str
    feature_version: str
    feature_implementation_version: str
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
    validation_available: bool = False
    test_sealed: bool = True


@dataclass(frozen=True)
class LoadedCanonicalRecurrentTrain:
    symbol: str
    partition: str
    data: pd.DataFrame
    episode_start: np.ndarray
    episode_boundaries: tuple[RecurrentEpisodeBoundary, ...]
    metadata: CanonicalRecurrentTrainMetadata
    source_artifact_path: Path


def _portable_path(path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()


def _canonical_json_hash(payload: Mapping[str, object]) -> str:
    import hashlib

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_prefix(symbol_dates: pd.Series) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(
        pd.to_datetime(symbol_dates, errors="coerce").dropna().unique()
    ).sort_values()
    if len(dates) < 3:
        return dates[:0]
    count = max(1, int(len(dates) * 0.70))
    validation_count = max(1, int(len(dates) * 0.15))
    if count + validation_count >= len(dates):
        count = len(dates) - 2
    return dates[:count]


def _identity_by_symbol(identity: pd.DataFrame) -> dict[str, Mapping[str, object]]:
    required = {"symbol", "company_name", "security_type"}
    missing = sorted(required.difference(identity.columns))
    if missing:
        raise CanonicalRecurrentArtifactError(
            "identity is missing canonical fields: " + ", ".join(missing)
        )
    values = identity.copy(deep=True)
    values["symbol"] = values["symbol"].astype("string").str.strip()
    if values["symbol"].eq("").any() or values["symbol"].duplicated().any():
        raise CanonicalRecurrentArtifactError(
            "identity symbols must be unique and nonempty"
        )
    return {str(row["symbol"]): row for row in values.to_dict(orient="records")}


def prepare_canonical_train(
    symbol: str,
    *,
    identity_record: Mapping[str, object],
    inventory: pd.DataFrame,
    parquet_path: Path,
    market_loader: Callable[..., pd.DataFrame] = load_market_data,
) -> PreparedCanonicalTrain:
    """Build backward-looking features from predicate-pushed TRAIN values only."""

    symbol_text = str(symbol).strip()
    security_type = str(identity_record.get("security_type", "")).strip()
    company_name = str(identity_record.get("company_name", "")).strip()
    dates = inventory.loc[inventory["symbol"] == symbol_text, "market_date"]
    train_dates = _training_prefix(dates)
    if len(train_dates):
        cutoff = train_dates[-1].date().isoformat()
        market = market_loader(
            parquet_path,
            end_date=train_dates[-1].date(),
            symbols=[symbol_text],
        )
    else:
        cutoff = ""
        market = pd.DataFrame()
    if not market.empty:
        loaded_dates = pd.to_datetime(market["market_date"], errors="coerce")
        if loaded_dates.isna().any() or (loaded_dates > train_dates[-1]).any():
            raise CanonicalRecurrentArtifactError(
                f"{symbol_text}: market values crossed the TRAIN cutoff"
            )
        raw = market.rename(columns={"market_date": "date"})[
            ["symbol", "date", "open", "high", "low", "close", "volume"]
        ]
        quality = filter_ai_quality_rows(raw)
        quality_rows = len(quality.data)
        removed = len(raw) - quality_rows
        featured = calculate_features(quality.data)
        warmup = featured["is_warmup"].astype(bool)
        missing = ~warmup & featured.loc[:, FEATURE_COLUMNS].isna().any(axis=1)
        usable = featured.loc[~warmup & ~missing].copy(deep=True)
        warmup_loss = int(warmup.sum())
        missing_rows = int(missing.sum())
    else:
        quality_rows = removed = warmup_loss = missing_rows = 0
        usable = pd.DataFrame()

    if not usable.empty:
        columns = ["symbol", "date", *FEATURE_COLUMNS]
        features = usable.loc[:, columns].copy(deep=True)
        features["symbol"] = features["symbol"].astype("string")
        features["date"] = pd.to_datetime(features["date"], errors="raise")
        features["feature_version"] = CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION
        features = features.sort_values(["date", "symbol"], kind="mergesort").reset_index(
            drop=True
        )
        numeric = features.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise CanonicalRecurrentArtifactError(
                f"{symbol_text}: usable TRAIN features are not finite"
            )
        first = features["date"].iloc[0].date().isoformat()
        last = features["date"].iloc[-1].date().isoformat()
    else:
        features = pd.DataFrame(
            columns=["symbol", "date", *FEATURE_COLUMNS, "feature_version"]
        )
        first = last = ""

    episode_feasible = len(features) >= MECHANICAL_MINIMUM_USABLE_ROWS
    scaler_feasible = len(features) >= MECHANICAL_MINIMUM_USABLE_ROWS
    allowed_type = security_type in SUPPORTED_COMMON_EQUITY_SECURITY_TYPES
    recovery_valid = (
        allowed_type
        and episode_feasible
        and scaler_feasible
        and len(features) >= CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
    )
    if not allowed_type:
        blocker = f"unsupported authoritative security_type={security_type or '<missing>'}"
    elif len(features) < CANONICAL_TRAIN_MINIMUM_USABLE_ROWS:
        blocker = (
            f"{len(features)} final TRAIN feature rows is below the versioned "
            f"minimum of {CANONICAL_TRAIN_MINIMUM_USABLE_ROWS}"
        )
    else:
        blocker = ""
    evidence = CanonicalTrainEvidence(
        symbol=symbol_text,
        company_name=company_name,
        security_type=security_type,
        train_cutoff=cutoff,
        canonical_train_rows=len(train_dates),
        quality_usable_ohlcv_rows=quality_rows,
        quality_rows_removed=removed,
        feature_warmup_loss=warmup_loss,
        missing_feature_rows=missing_rows,
        final_usable_feature_rows=len(features),
        first_train_feature_date=first,
        last_train_feature_date=last,
        recurrent_episode_feasible=episode_feasible,
        scaler_fit_feasible=scaler_feasible,
        recovery_valid=recovery_valid,
        blocker=blocker,
    )
    return PreparedCanonicalTrain(evidence=evidence, features=features)


def _scaled_train_frame(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    if len(features) < CANONICAL_TRAIN_MINIMUM_USABLE_ROWS:
        raise CanonicalRecurrentArtifactError(
            "TRAIN features do not meet the canonical Mature minimum"
        )
    observations = features.loc[:, DEFAULT_OBSERVATION_FEATURES].astype("float64")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(observations)
    artifact = features.copy(deep=True)
    for index, feature in enumerate(DEFAULT_OBSERVATION_FEATURES):
        artifact[scaled_observation_column(feature)] = transformed[:, index]
    return artifact, scaler


def _write_artifact_directory(
    prepared: PreparedCanonicalTrain,
    *,
    destination: Path,
    source_parquet_path: Path,
    source_parquet_sha256: str,
    identity_universe_hash: str,
) -> CanonicalArtifactBuildResult:
    evidence = prepared.evidence
    if not evidence.recovery_valid:
        raise CanonicalRecurrentArtifactError(
            f"{evidence.symbol}: cannot generate: {evidence.blocker}"
        )
    if destination.exists():
        raise FileExistsError(f"canonical recurrent artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        feature_path = temporary / FEATURE_FILENAME
        rl_path = temporary / RL_FILENAME
        scaler_path = temporary / SCALER_FILENAME
        scaler_metadata_path = temporary / SCALER_METADATA_FILENAME
        boundaries_path = temporary / BOUNDARIES_FILENAME
        contract_path = temporary / CONTRACT_FILENAME

        rl_frame, scaler = _scaled_train_frame(prepared.features)
        atomic_write_dataframe(prepared.features, feature_path)
        atomic_write_dataframe(rl_frame, rl_path)
        atomic_dump_joblib(scaler, scaler_path)
        scaler_metadata = {
            "artifact_schema_version": RL_TRAIN_PARTITION_SCHEMA_VERSION,
            "fit_partition": "train",
            "scaled_features": list(DEFAULT_OBSERVATION_FEATURES),
            "training_rows": len(prepared.features),
            "training_mean": scaler.mean_.tolist(),
            "training_scale": scaler.scale_.tolist(),
        }
        atomic_write_json(scaler_metadata, scaler_metadata_path)

        start = evidence.first_train_feature_date
        end = evidence.last_train_feature_date
        boundaries = {
            "artifact_schema_version": CANONICAL_EPISODE_BOUNDARY_VERSION,
            "recurrent_contract_version": RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
            "episode_strategy": "full_train_partition",
            "symbol": evidence.symbol,
            "partitions": {
                "train": [
                    {
                        "episode_id": f"{evidence.symbol}:train:0001",
                        "symbol": evidence.symbol,
                        "partition": "train",
                        "start_row": 0,
                        "end_row": len(prepared.features) - 1,
                        "rows": len(prepared.features),
                        "start": start,
                        "end": end,
                    }
                ]
            },
        }
        atomic_write_json(boundaries, boundaries_path)

        contract = {
            "artifact_schema_version": RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
            "source_rl_contract_version": RL_TRAIN_PARTITION_SCHEMA_VERSION,
            "feature_version": CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
            "feature_implementation_version": FEATURE_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "symbol": evidence.symbol,
            "company": evidence.company_name,
            "security_type": evidence.security_type,
            "authoritative_instrument_category": "COMMON_EQUITY",
            "identity_time_columns": list(IDENTITY_TIME_COLUMNS),
            "execution_accounting_columns": list(EXECUTION_ACCOUNTING_COLUMNS),
            "observation_features": list(DEFAULT_OBSERVATION_FEATURES),
            "scaled_observation_columns": {
                feature: scaled_observation_column(feature)
                for feature in DEFAULT_OBSERVATION_FEATURES
            },
            "dynamic_portfolio_features": list(DYNAMIC_PORTFOLIO_FEATURES),
            "observation_shape": [
                len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES)
            ],
            "normalization": {
                "scope": CANONICAL_NORMALIZATION_SCOPE,
                "fit_partition": "train",
                "scaler_path": SCALER_FILENAME,
                "scaler_sha256": sha256_file(scaler_path),
                "scaler_metadata_path": SCALER_METADATA_FILENAME,
                "scaler_metadata_sha256": sha256_file(scaler_metadata_path),
            },
            "history_policy": {
                "version": CANONICAL_TRAIN_HISTORY_POLICY_VERSION,
                "minimum_usable_train_rows": CANONICAL_TRAIN_MINIMUM_USABLE_ROWS,
                "mechanical_minimum_episode_rows": MECHANICAL_MINIMUM_USABLE_ROWS,
                "feature_warmup_rows": FEATURE_WARMUP_ROWS,
                "minimum_retained_rows_for_mature_train": MINIMUM_RETAINED_ROWS_FOR_MATURE_TRAIN,
                "usable_train_rows": len(prepared.features),
                "history_class": HistoryClass.MATURE.value,
                "independent_recurrent_ready": True,
            },
            "train_boundary": {
                "policy": CANONICAL_TRAIN_BOUNDARY_POLICY,
                "cutoff": evidence.train_cutoff,
                "raw_market_dates": evidence.canonical_train_rows,
                "quality_usable_ohlcv_rows": evidence.quality_usable_ohlcv_rows,
                "quality_rows_removed": evidence.quality_rows_removed,
                "feature_warmup_loss": evidence.feature_warmup_loss,
                "missing_feature_rows": evidence.missing_feature_rows,
            },
            "partitions": {
                "train": {
                    "role": "recurrent_learning_only",
                    "rows": len(prepared.features),
                    "start": start,
                    "end": end,
                    "feature_path": FEATURE_FILENAME,
                    "feature_sha256": sha256_file(feature_path),
                    "rl_path": RL_FILENAME,
                    "rl_sha256": sha256_file(rl_path),
                    "episode_count": 1,
                },
                "validation": {
                    "available": False,
                    "frame_access": "not_built_not_loaded",
                },
                "test": {
                    "available": False,
                    "sealed": True,
                    "frame_access": "sealed_not_built_not_loaded",
                },
            },
            "episode_boundaries": {
                "path": BOUNDARIES_FILENAME,
                "sha256": sha256_file(boundaries_path),
                "schema_version": CANONICAL_EPISODE_BOUNDARY_VERSION,
            },
            "universe": {
                "universe_hash": identity_universe_hash,
                "universe_id": f"current-common-equity:{identity_universe_hash[:16]}",
                "training_scope": "symbol",
                "constituent_symbols": [evidence.symbol],
                "cohort_cutoff": evidence.train_cutoff,
            },
            "source_snapshot": {
                "parquet_reference": _portable_path(source_parquet_path),
                "parquet_sha256": source_parquet_sha256,
                "value_load_max_date": evidence.train_cutoff,
                "post_train_value_rows_loaded": 0,
            },
            "sequence": {
                "episode_strategy": "full_train_partition",
                "minimum_sequence_rows": MECHANICAL_MINIMUM_USABLE_ROWS,
                "sequence_length": None,
                "burn_in_length": None,
                "episode_length": None,
            },
            "test_sealing": {
                "sealed": True,
                "frame_loaded_during_build": False,
                "artifact_created": False,
            },
        }
        contract["deterministic_contract_identity"] = _canonical_json_hash(contract)
        atomic_write_json(contract, contract_path)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_canonical_artifact_result(evidence.symbol, artifacts_dir=destination.parent.parent)


def build_canonical_recurrent_train_artifacts(
    symbols: Sequence[str],
    *,
    identity: pd.DataFrame,
    identity_universe_hash: str,
    parquet_path: str | Path | None = None,
    artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
    market_loader: Callable[..., pd.DataFrame] = load_market_data,
    inventory_loader: Callable[..., pd.DataFrame] = load_symbol_market_date_inventory,
) -> CanonicalRecoverySummary:
    """Audit candidates and atomically build only those meeting v2 policy."""

    selected = tuple(sorted({str(symbol).strip() for symbol in symbols}))
    if not selected or any(not symbol for symbol in selected):
        raise CanonicalRecurrentArtifactError("at least one nonempty symbol is required")
    identity_map = _identity_by_symbol(identity)
    unknown = sorted(set(selected).difference(identity_map))
    if unknown:
        raise CanonicalRecurrentArtifactError(
            "symbols are absent from the authoritative identity: " + ", ".join(unknown)
        )
    resolved = resolve_market_parquet_path(parquet_path)
    inventory = inventory_loader(resolved, symbols=selected)
    inventory = inventory.copy(deep=True)
    inventory["symbol"] = inventory["symbol"].astype("string").str.strip()
    inventory["market_date"] = pd.to_datetime(inventory["market_date"], errors="coerce")
    if inventory["market_date"].isna().any():
        raise CanonicalRecurrentArtifactError("symbol/date inventory contains invalid dates")
    source_hash = sha256_file(resolved)
    evidence: list[CanonicalTrainEvidence] = []
    generated: list[CanonicalArtifactBuildResult] = []
    for symbol in selected:
        prepared = prepare_canonical_train(
            symbol,
            identity_record=identity_map[symbol],
            inventory=inventory,
            parquet_path=resolved,
            market_loader=market_loader,
        )
        evidence.append(prepared.evidence)
        if not prepared.evidence.recovery_valid:
            continue
        destination = Path(artifacts_dir) / "symbols" / safe_path_component(symbol)
        generated.append(
            _write_artifact_directory(
                prepared,
                destination=destination,
                source_parquet_path=resolved,
                source_parquet_sha256=source_hash,
                identity_universe_hash=identity_universe_hash,
            )
        )
    summary = CanonicalRecoverySummary(tuple(evidence), tuple(generated))
    write_canonical_recovery_manifest(
        summary,
        artifacts_dir=Path(artifacts_dir),
        identity_universe_hash=identity_universe_hash,
        source_parquet_sha256=source_hash,
    )
    return summary


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalRecurrentArtifactError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalRecurrentArtifactError(f"{label} must be a JSON object")
    return value


def _artifact_directory(symbol: str, artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "symbols" / safe_path_component(symbol)


def canonical_contract_path(symbol: str, *, artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR) -> Path:
    return _artifact_directory(symbol, artifacts_dir) / CONTRACT_FILENAME


def _require_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise CanonicalRecurrentArtifactError(f"{label} is missing: {path}")
    if str(expected) != sha256_file(path):
        raise CanonicalRecurrentArtifactError(f"{label} hash is stale")


def load_canonical_recurrent_train_metadata(
    symbol: str,
    *,
    artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> CanonicalRecurrentTrainMetadata:
    symbol_text = str(symbol).strip()
    directory = _artifact_directory(symbol_text, artifacts_dir)
    contract_path = directory / CONTRACT_FILENAME
    contract = _load_json(contract_path, label="canonical recurrent TRAIN contract")
    if contract.get("artifact_schema_version") != RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION:
        raise CanonicalRecurrentArtifactError("canonical recurrent TRAIN version is incompatible")
    if contract.get("source_rl_contract_version") != RL_TRAIN_PARTITION_SCHEMA_VERSION:
        raise CanonicalRecurrentArtifactError("canonical source TRAIN partition version is incompatible")
    if contract.get("feature_version") != CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION:
        raise CanonicalRecurrentArtifactError("canonical feature version is incompatible")
    if contract.get("feature_implementation_version") != FEATURE_VERSION:
        raise CanonicalRecurrentArtifactError("feature implementation version is stale")
    if contract.get("environment_version") != ENVIRONMENT_VERSION:
        raise CanonicalRecurrentArtifactError("environment version is incompatible")
    if contract.get("symbol") != symbol_text:
        raise CanonicalRecurrentArtifactError("contract symbol is incompatible")
    security_type = str(contract.get("security_type", ""))
    if security_type not in SUPPORTED_COMMON_EQUITY_SECURITY_TYPES:
        raise CanonicalRecurrentArtifactError("contract security type is unsupported")
    if contract.get("authoritative_instrument_category") != "COMMON_EQUITY":
        raise CanonicalRecurrentArtifactError("contract identity is not common equity")
    if tuple(contract.get("observation_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise CanonicalRecurrentArtifactError("observation feature ordering is incompatible")
    if tuple(contract.get("dynamic_portfolio_features", ())) != DYNAMIC_PORTFOLIO_FEATURES:
        raise CanonicalRecurrentArtifactError("dynamic feature ordering is incompatible")
    expected_shape = (len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),)
    if tuple(contract.get("observation_shape", ())) != expected_shape:
        raise CanonicalRecurrentArtifactError("observation shape is incompatible")
    if tuple(contract.get("execution_accounting_columns", ())) != EXECUTION_ACCOUNTING_COLUMNS:
        raise CanonicalRecurrentArtifactError("execution columns are incompatible")
    if tuple(contract.get("identity_time_columns", ())) != IDENTITY_TIME_COLUMNS:
        raise CanonicalRecurrentArtifactError("identity/time columns are incompatible")
    identity = contract.pop("deterministic_contract_identity", None)
    if identity != _canonical_json_hash(contract):
        raise CanonicalRecurrentArtifactError("deterministic contract identity is stale")

    history = contract.get("history_policy")
    if not isinstance(history, Mapping):
        raise CanonicalRecurrentArtifactError("history policy is missing")
    rows = int(history.get("usable_train_rows", 0))
    if (
        history.get("version") != CANONICAL_TRAIN_HISTORY_POLICY_VERSION
        or int(history.get("minimum_usable_train_rows", 0))
        != CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
        or rows < CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
        or history.get("history_class") != HistoryClass.MATURE.value
        or history.get("independent_recurrent_ready") is not True
    ):
        raise CanonicalRecurrentArtifactError("canonical TRAIN history policy is incompatible")

    partitions = contract.get("partitions")
    if not isinstance(partitions, Mapping) or not isinstance(partitions.get("train"), Mapping):
        raise CanonicalRecurrentArtifactError("TRAIN partition metadata is missing")
    train = partitions["train"]
    if int(train.get("rows", 0)) != rows:
        raise CanonicalRecurrentArtifactError("TRAIN row count differs from history policy")
    if partitions.get("validation") != {
        "available": False,
        "frame_access": "not_built_not_loaded",
    }:
        raise CanonicalRecurrentArtifactError("VALIDATION sealing metadata is incompatible")
    if partitions.get("test") != {
        "available": False,
        "sealed": True,
        "frame_access": "sealed_not_built_not_loaded",
    }:
        raise CanonicalRecurrentArtifactError("TEST sealing metadata is incompatible")
    start = str(train.get("start", ""))
    end = str(train.get("end", ""))
    try:
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise CanonicalRecurrentArtifactError("TRAIN date range is invalid") from exc
    feature_path = directory / FEATURE_FILENAME
    rl_path = directory / RL_FILENAME
    _require_hash(feature_path, train.get("feature_sha256"), label="TRAIN features")
    _require_hash(rl_path, train.get("rl_sha256"), label="TRAIN RL artifact")

    normalization = contract.get("normalization")
    if not isinstance(normalization, Mapping):
        raise CanonicalRecurrentArtifactError("normalization metadata is missing")
    if (
        normalization.get("scope") != CANONICAL_NORMALIZATION_SCOPE
        or normalization.get("fit_partition") != "train"
    ):
        raise CanonicalRecurrentArtifactError("normalization policy is incompatible")
    scaler_path = directory / SCALER_FILENAME
    scaler_metadata_path = directory / SCALER_METADATA_FILENAME
    _require_hash(scaler_path, normalization.get("scaler_sha256"), label="TRAIN scaler")
    _require_hash(
        scaler_metadata_path,
        normalization.get("scaler_metadata_sha256"),
        label="TRAIN scaler metadata",
    )
    scaler_metadata = _load_json(scaler_metadata_path, label="TRAIN scaler metadata")
    if (
        scaler_metadata.get("fit_partition") != "train"
        or tuple(scaler_metadata.get("scaled_features", ())) != DEFAULT_OBSERVATION_FEATURES
        or int(scaler_metadata.get("training_rows", 0)) != rows
    ):
        raise CanonicalRecurrentArtifactError("TRAIN scaler metadata is incompatible")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        raise CanonicalRecurrentArtifactError(f"TRAIN scaler is unreadable: {exc}") from exc
    if not isinstance(scaler, StandardScaler) or int(getattr(scaler, "n_features_in_", -1)) != len(DEFAULT_OBSERVATION_FEATURES):
        raise CanonicalRecurrentArtifactError("TRAIN scaler type/width is incompatible")
    if tuple(str(value) for value in getattr(scaler, "feature_names_in_", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise CanonicalRecurrentArtifactError("TRAIN scaler feature order is incompatible")

    boundary_meta = contract.get("episode_boundaries")
    if not isinstance(boundary_meta, Mapping):
        raise CanonicalRecurrentArtifactError("episode-boundary metadata is missing")
    boundaries_path = directory / BOUNDARIES_FILENAME
    _require_hash(boundaries_path, boundary_meta.get("sha256"), label="episode boundaries")
    universe = contract.get("universe")
    sequence = contract.get("sequence")
    if not isinstance(universe, Mapping) or not isinstance(sequence, Mapping):
        raise CanonicalRecurrentArtifactError("universe/sequence metadata is missing")
    return CanonicalRecurrentTrainMetadata(
        symbol=symbol_text,
        company=str(contract.get("company", "")),
        security_type=security_type,
        contract_path=contract_path.resolve(),
        boundaries_path=boundaries_path.resolve(),
        recurrent_contract_version=RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
        source_rl_contract_version=RL_TRAIN_PARTITION_SCHEMA_VERSION,
        feature_version=CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
        feature_implementation_version=FEATURE_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        dynamic_portfolio_features=DYNAMIC_PORTFOLIO_FEATURES,
        observation_shape=expected_shape,
        execution_columns=EXECUTION_ACCOUNTING_COLUMNS,
        scaler_fit_partition="train",
        normalization_scope=CANONICAL_NORMALIZATION_SCOPE,
        training_scope=str(universe.get("training_scope", "")),
        universe_id=str(universe.get("universe_id", "")),
        universe_hash=str(universe.get("universe_hash", "")),
        constituent_symbols=tuple(str(value) for value in universe.get("constituent_symbols", ())),
        cohort_cutoff=str(universe.get("cohort_cutoff", "")),
        sequence_length=sequence.get("sequence_length"),
        burn_in_length=sequence.get("burn_in_length"),
        episode_length=sequence.get("episode_length"),
        minimum_sequence_rows=int(sequence.get("minimum_sequence_rows", 0)),
        episode_strategy=str(sequence.get("episode_strategy", "")),
        history=recurrent_eligibility(rows),
        train=RecurrentPartitionMetadata(
            name="train",
            role=str(train.get("role", "")),
            rows=rows,
            start=start,
            end=end,
            episode_count=int(train.get("episode_count", 0)),
            sealed=False,
        ),
    )


def load_canonical_recurrent_train_partition(
    symbol: str,
    partition: str,
    *,
    artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> LoadedCanonicalRecurrentTrain:
    if partition != "train":
        raise CanonicalRecurrentArtifactError(
            "canonical recurrent v2 exposes TRAIN only; VALIDATION is unavailable and TEST is sealed"
        )
    metadata = load_canonical_recurrent_train_metadata(symbol, artifacts_dir=artifacts_dir)
    directory = metadata.contract_path.parent
    features = pd.read_csv(directory / FEATURE_FILENAME, dtype={"symbol": "string"})
    artifact = pd.read_csv(directory / RL_FILENAME, dtype={"symbol": "string"})
    for frame in (features, artifact):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if len(features) != metadata.train.rows or len(artifact) != metadata.train.rows:
        raise CanonicalRecurrentArtifactError("TRAIN artifact row count is stale")
    if not features.loc[:, list(IDENTITY_TIME_COLUMNS)].equals(
        artifact.loc[:, list(IDENTITY_TIME_COLUMNS)]
    ):
        raise CanonicalRecurrentArtifactError("TRAIN feature/RL identities differ")
    for column in EXECUTION_ACCOUNTING_COLUMNS:
        if not np.array_equal(
            features[column].to_numpy(), artifact[column].to_numpy(), equal_nan=True
        ):
            raise CanonicalRecurrentArtifactError(
                f"unscaled execution field {column!r} differs"
            )
    environment_data = artifact.copy(deep=True)
    for feature in DEFAULT_OBSERVATION_FEATURES:
        column = scaled_observation_column(feature)
        if column not in artifact:
            raise CanonicalRecurrentArtifactError(f"scaled feature is missing: {column}")
        environment_data[feature] = pd.to_numeric(artifact[column], errors="coerce")
    numeric = environment_data.loc[
        :, [*EXECUTION_ACCOUNTING_COLUMNS, *DEFAULT_OBSERVATION_FEATURES]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise CanonicalRecurrentArtifactError("environment TRAIN values are not finite")
    boundaries_payload = _load_json(metadata.boundaries_path, label="episode boundaries")
    raw_boundaries = boundaries_payload.get("partitions", {}).get("train", [])
    if not isinstance(raw_boundaries, list) or len(raw_boundaries) != 1:
        raise CanonicalRecurrentArtifactError("TRAIN must contain exactly one episode")
    raw = raw_boundaries[0]
    boundary = RecurrentEpisodeBoundary(
        episode_id=str(raw.get("episode_id", "")),
        symbol=str(raw.get("symbol", "")),
        partition=str(raw.get("partition", "")),
        start_row=int(raw.get("start_row", -1)),
        end_row=int(raw.get("end_row", -1)),
        rows=int(raw.get("rows", 0)),
        start=str(raw.get("start", "")),
        end=str(raw.get("end", "")),
    )
    expected = RecurrentEpisodeBoundary(
        episode_id=f"{metadata.symbol}:train:0001",
        symbol=metadata.symbol,
        partition="train",
        start_row=0,
        end_row=metadata.train.rows - 1,
        rows=metadata.train.rows,
        start=metadata.train.start,
        end=metadata.train.end,
    )
    if boundary != expected:
        raise CanonicalRecurrentArtifactError("TRAIN episode boundary is incompatible")
    mask = recurrent_episode_start_mask(
        environment_data["symbol"].astype("string").tolist(),
        ["train"] * len(environment_data),
    )
    return LoadedCanonicalRecurrentTrain(
        symbol=metadata.symbol,
        partition="train",
        data=environment_data,
        episode_start=mask,
        episode_boundaries=(boundary,),
        metadata=metadata,
        source_artifact_path=directory / RL_FILENAME,
    )


def load_canonical_artifact_result(
    symbol: str,
    *,
    artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> CanonicalArtifactBuildResult:
    metadata = load_canonical_recurrent_train_metadata(symbol, artifacts_dir=artifacts_dir)
    directory = metadata.contract_path.parent
    contract = _load_json(metadata.contract_path, label="canonical recurrent TRAIN contract")
    return CanonicalArtifactBuildResult(
        symbol=metadata.symbol,
        artifact_directory=directory,
        contract_path=metadata.contract_path,
        train_features_path=directory / FEATURE_FILENAME,
        train_rl_path=directory / RL_FILENAME,
        scaler_path=directory / SCALER_FILENAME,
        scaler_metadata_path=directory / SCALER_METADATA_FILENAME,
        boundaries_path=metadata.boundaries_path,
        contract=contract,
    )


def load_training_recurrent_contract_metadata(
    symbol: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    canonical_artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> CanonicalRecurrentTrainMetadata | RecurrentContractMetadata:
    """Prefer v2 when present; invalid v2 fails closed without v1 fallback."""

    if canonical_contract_path(symbol, artifacts_dir=canonical_artifacts_dir).is_file():
        return load_canonical_recurrent_train_metadata(
            symbol, artifacts_dir=canonical_artifacts_dir
        )
    return load_recurrent_contract_metadata(symbol, splits_dir=splits_dir)


def load_training_recurrent_partition(
    symbol: str,
    partition: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    canonical_artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
) -> LoadedCanonicalRecurrentTrain | LoadedRecurrentPartition:
    """Load v2 TRAIN when present, otherwise preserve exact v1 behavior."""

    if canonical_contract_path(symbol, artifacts_dir=canonical_artifacts_dir).is_file():
        return load_canonical_recurrent_train_partition(
            symbol, partition, artifacts_dir=canonical_artifacts_dir
        )
    return load_recurrent_partition(symbol, partition, splits_dir=splits_dir)


SUPPORTED_RECURRENT_TRAIN_CONTRACT_VERSIONS = frozenset(
    {
        RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
    }
)
