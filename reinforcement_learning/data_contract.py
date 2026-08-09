"""Versioned, leakage-safe data contract for RL environment partitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from feature_engineering.preprocessing import fit_training_scaler, save_scaler_artifact
from feature_engineering.storage import atomic_write_dataframe, atomic_write_json, safe_path_component

from .environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)


RL_PARTITION_SCHEMA_VERSION = "rl_partition_v1"
RL_CONTRACT_FILENAME = "rl_contract.json"
RL_OBSERVATION_SCALER_FILENAME = "rl_observation_scaler.joblib"
IDENTITY_TIME_COLUMNS = ("symbol", "date")
EXECUTION_ACCOUNTING_COLUMNS = ("open", "high", "low", "close", "volume")
PARTITION_NAMES = ("train", "validation", "test")
SCALED_OBSERVATION_PREFIX = "observation_scaled__"


class RLDataContractError(ValueError):
    """Raised when RL partition artifacts are missing, stale, or misaligned."""


@dataclass(frozen=True)
class RLPartitionArtifactResult:
    """Paths and metadata written for one split set's RL contract."""

    contract_path: Path
    scaler_path: Path
    partition_paths: Mapping[str, Path]
    contract: Mapping[str, object]


@dataclass(frozen=True)
class LoadedRLPartition:
    """Environment-ready data and its validated artifact contract."""

    symbol: str
    partition: str
    data: pd.DataFrame
    artifact_path: Path
    contract: Mapping[str, object]


def scaled_observation_column(feature: str) -> str:
    """Return the explicit artifact column for one scaled observation feature."""
    return f"{SCALED_OBSERVATION_PREFIX}{feature}"


def _validate_partition_identity(data: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = set((*IDENTITY_TIME_COLUMNS, *EXECUTION_ACCOUNTING_COLUMNS, *DEFAULT_OBSERVATION_FEATURES))
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RLDataContractError(
            f"{label} partition is missing RL contract columns: {', '.join(missing)}"
        )
    checked = data.copy(deep=True)
    checked["symbol"] = checked["symbol"].astype("string").str.strip()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce")
    if checked.empty:
        raise RLDataContractError(f"{label} partition cannot be empty")
    if checked["symbol"].isna().any() or checked["symbol"].eq("").any():
        raise RLDataContractError(f"{label} partition contains an empty symbol")
    if checked["date"].isna().any():
        raise RLDataContractError(f"{label} partition contains an invalid date")
    if checked.duplicated(list(IDENTITY_TIME_COLUMNS)).any():
        raise RLDataContractError(
            f"{label} partition contains duplicate (symbol, date) keys"
        )
    if not checked.sort_values(["date", "symbol"], kind="stable").index.equals(
        checked.index
    ):
        raise RLDataContractError(f"{label} partition is not chronological")
    numeric = checked.loc[
        :, [*EXECUTION_ACCOUNTING_COLUMNS, *DEFAULT_OBSERVATION_FEATURES]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RLDataContractError(f"{label} partition contains non-finite RL values")
    checked.loc[:, numeric.columns] = numeric
    return checked


def _rl_artifact_frame(raw: pd.DataFrame, scaled: pd.DataFrame, *, label: str) -> pd.DataFrame:
    source = _validate_partition_identity(raw, label=label)
    if len(source) != len(scaled):
        raise RLDataContractError(
            f"{label} raw/scaled row counts differ: {len(source)} != {len(scaled)}"
        )
    scaled_keys = scaled.loc[:, list(IDENTITY_TIME_COLUMNS)].copy()
    scaled_keys["symbol"] = scaled_keys["symbol"].astype("string").str.strip()
    scaled_keys["date"] = pd.to_datetime(scaled_keys["date"], errors="coerce")
    if not source.loc[:, list(IDENTITY_TIME_COLUMNS)].equals(scaled_keys):
        raise RLDataContractError(f"{label} raw/scaled (symbol, date) alignment differs")

    artifact = source.copy(deep=True)
    for feature in DEFAULT_OBSERVATION_FEATURES:
        values = pd.to_numeric(scaled[feature], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RLDataContractError(
                f"{label} scaled observation feature {feature!r} is not finite"
            )
        artifact[scaled_observation_column(feature)] = values.to_numpy(dtype=float)
    return artifact


def persist_rl_partition_artifacts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    *,
    feature_version: str,
) -> RLPartitionArtifactResult:
    """Persist aligned raw-price/scaled-observation RL partitions atomically."""
    raw_partitions = {
        "train": _validate_partition_identity(train, label="train"),
        "validation": _validate_partition_identity(validation, label="validation"),
        "test": _validate_partition_identity(test, label="test"),
    }
    scaling = fit_training_scaler(
        raw_partitions["train"],
        raw_partitions["validation"],
        raw_partitions["test"],
        feature_columns=DEFAULT_OBSERVATION_FEATURES,
    )
    scaled_partitions = {
        "train": scaling.train,
        "validation": scaling.validation,
        "test": scaling.test,
    }
    directory = Path(output_dir)
    partition_paths: dict[str, Path] = {}
    partition_contracts: dict[str, dict[str, object]] = {}
    for name in PARTITION_NAMES:
        artifact = _rl_artifact_frame(
            raw_partitions[name], scaled_partitions[name], label=name
        )
        path = directory / f"{name}_rl.csv"
        atomic_write_dataframe(artifact, path)
        partition_paths[name] = path
        dates = artifact["date"]
        partition_contracts[name] = {
            "path": str(path),
            "rows": len(artifact),
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
        }

    scaler_path, scaler_metadata_path = save_scaler_artifact(
        scaling, directory / RL_OBSERVATION_SCALER_FILENAME
    )
    contract: dict[str, object] = {
        "artifact_schema_version": RL_PARTITION_SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "feature_version": feature_version,
        "identity_time_columns": list(IDENTITY_TIME_COLUMNS),
        "execution_accounting_columns": list(EXECUTION_ACCOUNTING_COLUMNS),
        "observation_features": list(DEFAULT_OBSERVATION_FEATURES),
        "scaled_observation_columns": {
            feature: scaled_observation_column(feature)
            for feature in DEFAULT_OBSERVATION_FEATURES
        },
        "dynamic_portfolio_features": list(DYNAMIC_PORTFOLIO_FEATURES),
        "scaler_fit_partition": "train",
        "scaler_path": str(scaler_path),
        "scaler_metadata_path": str(scaler_metadata_path),
        "partitions": partition_contracts,
    }
    contract_path = directory / RL_CONTRACT_FILENAME
    atomic_write_json(contract, contract_path)
    return RLPartitionArtifactResult(
        contract_path=contract_path,
        scaler_path=scaler_path,
        partition_paths=partition_paths,
        contract=contract,
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RLDataContractError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RLDataContractError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RLDataContractError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_contract(path: Path) -> dict[str, object]:
    payload = _load_json_object(path, label="RL contract")
    version = payload.get("artifact_schema_version")
    if version != RL_PARTITION_SCHEMA_VERSION:
        raise RLDataContractError(
            "Incompatible RL artifact schema version: "
            f"expected {RL_PARTITION_SCHEMA_VERSION!r}, found {version!r}. "
            "Rebuild chronological split/scaler artifacts."
        )
    if payload.get("environment_version") != ENVIRONMENT_VERSION:
        raise RLDataContractError(
            "RL artifact environment version does not match "
            f"{ENVIRONMENT_VERSION!r}"
        )
    if tuple(payload.get("observation_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise RLDataContractError("RL artifact observation feature order is incompatible")
    if payload.get("scaler_fit_partition") != "train":
        raise RLDataContractError(
            "RL observation scaler must be fitted on the train partition"
        )
    return payload


def load_rl_partition(
    symbol: str,
    partition: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> LoadedRLPartition:
    """Load a strict hybrid frame for environment execution and observations."""
    if partition not in PARTITION_NAMES:
        raise RLDataContractError(
            f"partition must be one of {', '.join(PARTITION_NAMES)}"
        )
    symbol_text = str(symbol).strip()
    directory = Path(splits_dir) / "symbols" / safe_path_component(symbol_text)
    contract = _load_contract(directory / RL_CONTRACT_FILENAME)
    split_metadata = _load_json_object(
        directory / "metadata.json", label="split metadata"
    )
    if split_metadata.get("rl_artifact_schema_version") != RL_PARTITION_SCHEMA_VERSION:
        raise RLDataContractError(
            "Split metadata has a stale or missing RL artifact schema version. "
            "Rebuild chronological split/scaler artifacts."
        )
    if split_metadata.get("feature_version") != contract.get("feature_version"):
        raise RLDataContractError("RL contract and split feature versions differ")
    scaler_metadata = _load_json_object(
        directory / RL_OBSERVATION_SCALER_FILENAME.replace(".joblib", ".json"),
        label="RL observation scaler metadata",
    )
    if tuple(scaler_metadata.get("scaled_features", ())) != DEFAULT_OBSERVATION_FEATURES:
        raise RLDataContractError("RL observation scaler feature order is incompatible")
    if not (directory / RL_OBSERVATION_SCALER_FILENAME).is_file():
        raise RLDataContractError("RL observation scaler artifact is missing")
    contract_partitions = dict(contract.get("partitions", {}))
    partition_metadata = contract_partitions.get(partition)
    if not isinstance(partition_metadata, dict):
        raise RLDataContractError(f"RL contract has no {partition!r} partition")
    train_metadata = contract_partitions.get("train")
    if not isinstance(train_metadata, dict):
        raise RLDataContractError("RL contract has no 'train' partition")
    try:
        scaler_training_rows = int(scaler_metadata.get("training_rows", -1))
        contract_training_rows = int(train_metadata.get("rows", -2))
    except (TypeError, ValueError) as exc:
        raise RLDataContractError(
            "RL observation scaler training row metadata is invalid"
        ) from exc
    if scaler_training_rows != contract_training_rows:
        raise RLDataContractError("RL observation scaler training row count is stale")
    artifact_path = directory / f"{partition}_rl.csv"
    try:
        artifact = pd.read_csv(artifact_path, dtype={"symbol": "string"})
        raw = pd.read_csv(directory / f"{partition}.csv", dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise RLDataContractError(
            f"Could not load {partition!r} RL partition for {symbol_text}: {exc}"
        ) from exc
    artifact = _validate_partition_identity(artifact, label=f"{partition} RL")
    raw = _validate_partition_identity(raw, label=f"{partition} raw")
    if len(artifact) != int(partition_metadata.get("rows", -1)):
        raise RLDataContractError(f"{partition} RL row count does not match contract")
    expected_start = str(partition_metadata.get("start"))
    expected_end = str(partition_metadata.get("end"))
    actual_start = artifact["date"].min().date().isoformat()
    actual_end = artifact["date"].max().date().isoformat()
    if (actual_start, actual_end) != (expected_start, expected_end):
        raise RLDataContractError(f"{partition} RL date range does not match contract")
    if not artifact.loc[:, list(IDENTITY_TIME_COLUMNS)].equals(
        raw.loc[:, list(IDENTITY_TIME_COLUMNS)]
    ):
        raise RLDataContractError(f"{partition} RL/raw identity alignment differs")
    symbols = tuple(artifact["symbol"].unique())
    if symbols != (symbol_text,):
        raise RLDataContractError(
            f"{partition} RL artifact contains {symbols!r}, expected {symbol_text!r}"
        )
    for column in EXECUTION_ACCOUNTING_COLUMNS:
        if not np.array_equal(
            artifact[column].to_numpy(), raw[column].to_numpy(), equal_nan=True
        ):
            raise RLDataContractError(
                f"{partition} RL execution field {column!r} differs from raw partition"
            )
    feature_version = contract.get("feature_version")
    for label, frame in (("RL", artifact), ("raw", raw)):
        if "feature_version" not in frame:
            continue
        versions = set(frame["feature_version"].dropna().astype(str))
        if versions != {str(feature_version)}:
            raise RLDataContractError(
                f"{partition} {label} dataframe feature version is incompatible"
            )

    environment_data = artifact.copy(deep=True)
    scaled_mapping = dict(contract.get("scaled_observation_columns", {}))
    for feature in DEFAULT_OBSERVATION_FEATURES:
        scaled_column = scaled_mapping.get(feature)
        if scaled_column != scaled_observation_column(feature):
            raise RLDataContractError(
                f"RL contract scaled column for {feature!r} is incompatible"
            )
        if scaled_column not in artifact:
            raise RLDataContractError(
                f"{partition} RL artifact is missing {scaled_column!r}"
            )
        environment_data[feature] = pd.to_numeric(
            artifact[scaled_column], errors="coerce"
        )
    if not np.isfinite(
        environment_data.loc[:, list(DEFAULT_OBSERVATION_FEATURES)].to_numpy(
            dtype=float
        )
    ).all():
        raise RLDataContractError(f"{partition} RL observations are not finite")
    return LoadedRLPartition(
        symbol=symbol_text,
        partition=partition,
        data=environment_data,
        artifact_path=artifact_path,
        contract=contract,
    )
