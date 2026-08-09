"""Atomic, version-preserving model registry for future PPO training."""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import tempfile
from typing import Sequence

import pandas as pd

from data_pipeline.src.config import MODEL_REGISTRY_PATH
from feature_engineering.storage import safe_path_component
from reinforcement_learning.environments.config import ENVIRONMENT_VERSION

from .paths import master_model_paths, ppo_bundle_paths, symbol_model_paths


MODEL_REGISTRY_SCHEMA_VERSION = "model_registry_v2"


MODEL_REGISTRY_COLUMNS = (
    "model_id",
    "model_scope",
    "symbol",
    "algorithm",
    "model_version",
    "model_status",
    "training_status",
    "feature_version",
    "environment_version",
    "created_at",
    "last_trained_at",
    "complete_available_history_start",
    "complete_available_history_end",
    "training_data_start",
    "training_data_end",
    "validation_data_start",
    "validation_data_end",
    "test_data_start",
    "test_data_end",
    "training_rows",
    "validation_rows",
    "test_rows",
    "dataset_latest_date",
    "new_data_days",
    "needs_retraining",
    "model_path",
    "scaler_path",
    "metrics_path",
    "training_duration_seconds",
    "random_seed",
    "notes",
    "registry_schema_version",
    "artifact_schema_version",
    "rl_contract_version",
    "ppo_config_version",
    "validation_status",
    "promotion_status",
    "rl_contract_path",
    "scaler_metadata_path",
    "metadata_path",
    "config_path",
    "validation_metrics_path",
    "baseline_metrics_path",
    "registry_record_path",
    "manifest_path",
    "manifest_sha256",
    "observation_shape",
    "observation_features",
    "source_git_commit",
    "source_worktree_dirty",
)
MODEL_SCOPES = frozenset({"symbol", "master"})
MODEL_STATUSES = frozenset(
    {
        "not_trained",
        "training",
        "trained",
        "failed",
        "archived",
        "candidate",
        "experiment",
        "production",
        "superseded",
    }
)
VALIDATION_STATUSES = frozenset(
    {
        "not_evaluated",
        "validation_pass",
        "validation_fail",
        "insufficient_validation_data",
        "evaluation_error",
    }
)
PROMOTION_STATUSES = frozenset(
    {
        "not_promoted",
        "candidate",
        "experiment",
        "production",
        "superseded",
        "not_eligible",
    }
)
IDENTITY_CRITICAL_FIELDS = frozenset(
    {
        "model_id",
        "model_scope",
        "symbol",
        "algorithm",
        "model_version",
        "feature_version",
        "environment_version",
        "created_at",
        "registry_schema_version",
    }
)


class ModelRegistryError(ValueError):
    """Raised when model-registry structure or values are unsafe."""


def empty_model_registry() -> pd.DataFrame:
    """Return an empty registry with its stable machine-friendly schema."""
    return pd.DataFrame(columns=MODEL_REGISTRY_COLUMNS)


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _validate_registry_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized registry after enforcing identity invariants."""
    missing = sorted(set(MODEL_REGISTRY_COLUMNS).difference(data.columns))
    unknown = sorted(set(data.columns).difference(MODEL_REGISTRY_COLUMNS))
    if missing:
        raise ModelRegistryError(
            f"Model registry is missing columns: {', '.join(missing)}"
        )
    if unknown:
        raise ModelRegistryError(
            f"Model registry contains unknown columns: {', '.join(unknown)}"
        )
    normalized = data.loc[:, MODEL_REGISTRY_COLUMNS].copy()
    if normalized.empty:
        return normalized

    scopes = normalized["model_scope"].map(_text)
    invalid_scopes = set(scopes).difference(MODEL_SCOPES)
    if invalid_scopes:
        raise ModelRegistryError(
            f"Model registry contains invalid scopes: {', '.join(sorted(invalid_scopes))}"
        )
    statuses = normalized["model_status"].map(_text)
    invalid_statuses = set(statuses).difference(MODEL_STATUSES)
    if invalid_statuses:
        raise ModelRegistryError(
            "Model registry contains invalid statuses: "
            f"{', '.join(sorted(invalid_statuses))}"
        )
    validation_statuses = normalized["validation_status"].map(_text)
    invalid_validation = set(validation_statuses).difference(VALIDATION_STATUSES)
    if invalid_validation:
        raise ModelRegistryError(
            "Model registry contains invalid validation statuses: "
            f"{', '.join(sorted(invalid_validation))}"
        )
    promotion_statuses = normalized["promotion_status"].map(_text)
    invalid_promotion = set(promotion_statuses).difference(PROMOTION_STATUSES)
    if invalid_promotion:
        raise ModelRegistryError(
            "Model registry contains invalid promotion statuses: "
            f"{', '.join(sorted(invalid_promotion))}"
        )
    schema_versions = normalized["registry_schema_version"].map(_text)
    if not schema_versions.eq(MODEL_REGISTRY_SCHEMA_VERSION).all():
        raise ModelRegistryError(
            "Model registry contains an incompatible registry schema version"
        )

    numeric_versions = pd.to_numeric(
        normalized["model_version"], errors="coerce"
    )
    valid_versions = numeric_versions.map(
        lambda value: bool(
            pd.notna(value)
            and float(value) > 0
            and float(value).is_integer()
        )
    )
    if not valid_versions.all():
        raise ModelRegistryError(
            "Model registry versions must be positive integers"
        )
    normalized["model_version"] = numeric_versions.astype("int64")
    normalized["model_scope"] = scopes
    normalized["model_status"] = statuses
    normalized["validation_status"] = validation_statuses
    normalized["promotion_status"] = promotion_statuses

    symbols = normalized["symbol"].map(_text)
    normalized["symbol"] = pd.Series(symbols, dtype="string")
    identities = normalized.assign(_symbol_key=symbols).duplicated(
        ["model_scope", "_symbol_key", "model_version"]
    )
    if identities.any():
        raise ModelRegistryError(
            "Model registry contains duplicate scope/symbol/version identities"
        )
    model_ids = normalized["model_id"].map(_text)
    if model_ids.eq("").any() or model_ids.duplicated().any():
        raise ModelRegistryError(
            "Model registry contains empty or duplicate model IDs"
        )
    normalized["model_id"] = pd.Series(model_ids, dtype="string")
    for _, row in normalized.iterrows():
        scope = str(row["model_scope"])
        symbol = _text(row["symbol"])
        version = int(row["model_version"])
        if scope == "symbol" and not symbol:
            raise ModelRegistryError("symbol scope requires a symbol")
        if scope == "master" and symbol:
            raise ModelRegistryError("master scope cannot specify a symbol")
        if _text(row["algorithm"]) != "PPO":
            raise ModelRegistryError("Model registry algorithm must be PPO")
        try:
            identity = safe_path_component(symbol) if symbol else "master"
        except ValueError as exc:
            raise ModelRegistryError(f"Invalid registry symbol: {exc}") from exc
        expected_id = f"ppo-{scope}-{identity}-v{version:04d}"
        if _text(row["model_id"]) != expected_id:
            raise ModelRegistryError(
                f"Model ID is inconsistent with its identity: expected {expected_id}"
            )

    model_paths = normalized["model_path"].map(_text)
    nonempty_paths = model_paths.loc[model_paths.ne("")]
    if nonempty_paths.duplicated().any():
        raise ModelRegistryError("Model registry contains duplicate model paths")

    retraining = (
        normalized["needs_retraining"].astype("string").str.strip().str.lower()
    )
    invalid_retraining = retraining.loc[
        ~retraining.isin({"true", "false", "1", "0"})
    ]
    if not invalid_retraining.empty:
        raise ModelRegistryError("Model registry contains invalid retraining flags")
    normalized["needs_retraining"] = retraining.isin({"true", "1"})
    return normalized


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> pd.DataFrame:
    """Load the model registry or return a schema-correct empty dataframe."""
    registry_path = Path(path)
    if not registry_path.exists():
        return empty_model_registry()
    try:
        data = pd.read_csv(
            registry_path,
            dtype={"symbol": "string", "model_id": "string"},
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ModelRegistryError(f"Could not load model registry: {exc}") from exc
    return _validate_registry_dataframe(data)


def validate_model_record(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize exactly one complete registry record."""
    missing = sorted(set(MODEL_REGISTRY_COLUMNS).difference(record))
    unknown = sorted(set(record).difference(MODEL_REGISTRY_COLUMNS))
    if missing:
        raise ModelRegistryError(f"Model record is missing: {', '.join(missing)}")
    if unknown:
        raise ModelRegistryError(
            f"Model record contains unknown fields: {', '.join(unknown)}"
        )
    ordered = {column: record[column] for column in MODEL_REGISTRY_COLUMNS}
    validated = _validate_registry_dataframe(pd.DataFrame([ordered]))
    return validated.iloc[0].to_dict()


@contextmanager
def model_registry_lock(path: Path = MODEL_REGISTRY_PATH):
    """Hold the advisory registry transaction lock for allocation/publication."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_registry(data: pd.DataFrame, path: Path) -> None:
    """Validate, fsync, and atomically replace one complete registry CSV."""
    destination = Path(path)
    validated = _validate_registry_dataframe(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            validated.to_csv(handle, index=False, date_format="%Y-%m-%d")
            handle.flush()
            os.fsync(handle.fileno())
        load_model_registry(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_model_registry(path: Path = MODEL_REGISTRY_PATH) -> Path:
    """Create an empty registry atomically without replacing an existing one."""
    destination = Path(path)
    with model_registry_lock(destination):
        if destination.exists():
            load_model_registry(destination)
            return destination
        _atomic_write_registry(empty_model_registry(), destination)
    return destination


def next_model_version(
    registry: pd.DataFrame,
    *,
    model_scope: str,
    symbol: str = "",
) -> int:
    """Return the next immutable version number for a model identity."""
    if model_scope not in MODEL_SCOPES:
        raise ModelRegistryError(f"Unsupported model scope: {model_scope}")
    symbol_text = str(symbol).strip() if model_scope == "symbol" else ""
    if model_scope == "symbol" and not symbol_text:
        raise ModelRegistryError("symbol scope requires a symbol")
    validated = _validate_registry_dataframe(registry)
    if validated.empty:
        return 1
    matches = validated.loc[validated["model_scope"].astype(str) == model_scope]
    if model_scope == "symbol":
        matches = matches.loc[matches["symbol"].astype("string") == symbol_text]
    versions = pd.to_numeric(matches["model_version"], errors="coerce").dropna()
    return int(versions.max()) + 1 if not versions.empty else 1


def create_model_record(
    *,
    registry: pd.DataFrame,
    model_scope: str,
    symbol: str = "",
    feature_version: str,
    values: Mapping[str, object] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Create deterministic metadata for a new, untrained model version."""
    version = next_model_version(
        registry,
        model_scope=model_scope,
        symbol=symbol,
    )
    symbol_text = str(symbol).strip() if model_scope == "symbol" else ""
    identity = safe_path_component(symbol_text) if symbol_text else "master"
    model_id = f"ppo-{model_scope}-{identity}-v{version:04d}"
    paths = (
        symbol_model_paths(symbol_text, version)
        if model_scope == "symbol"
        else master_model_paths(version)
    )
    bundle_paths = ppo_bundle_paths(model_scope, symbol_text, version)
    timestamp = created_at or datetime.now(timezone.utc)
    record: dict[str, object] = {column: "" for column in MODEL_REGISTRY_COLUMNS}
    record.update(
        {
            "model_id": model_id,
            "model_scope": model_scope,
            "symbol": symbol_text,
            "algorithm": "PPO",
            "model_version": version,
            "model_status": "not_trained",
            "training_status": "never_trained",
            "feature_version": feature_version,
            "environment_version": ENVIRONMENT_VERSION,
            "created_at": timestamp.isoformat(),
            "new_data_days": 0,
            "needs_retraining": False,
            "model_path": str(paths.model),
            "scaler_path": str(paths.scaler),
            "metrics_path": str(paths.metrics),
            "registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
            "validation_status": "not_evaluated",
            "promotion_status": "not_promoted",
            "rl_contract_path": str(bundle_paths.rl_contract),
            "scaler_metadata_path": str(bundle_paths.scaler_metadata),
            "metadata_path": str(bundle_paths.metadata),
            "config_path": str(bundle_paths.ppo_config),
            "validation_metrics_path": str(bundle_paths.validation_metrics),
            "baseline_metrics_path": str(bundle_paths.baseline_metrics),
            "registry_record_path": str(bundle_paths.registry_record),
            "manifest_path": str(bundle_paths.manifest),
        }
    )
    if values:
        unknown = sorted(set(values).difference(MODEL_REGISTRY_COLUMNS))
        if unknown:
            raise ModelRegistryError(
                f"Unknown model registry fields: {', '.join(unknown)}"
            )
        forbidden = sorted(set(values).intersection(IDENTITY_CRITICAL_FIELDS))
        if forbidden:
            raise ModelRegistryError(
                "Identity-critical model fields cannot be overridden: "
                + ", ".join(forbidden)
            )
        record.update(values)
    return validate_model_record(record)


def _append_model_version_unlocked(
    record: Mapping[str, object],
    path: Path = MODEL_REGISTRY_PATH,
) -> pd.DataFrame:
    """Append while the caller holds :func:`model_registry_lock`."""
    normalized_record = validate_model_record(record)
    registry = load_model_registry(path)
    model_id = str(normalized_record["model_id"])
    if model_id in set(registry["model_id"].astype(str)):
        raise ModelRegistryError(f"Model ID already exists: {model_id}")
    updated = pd.concat(
        [registry, pd.DataFrame([normalized_record])],
        ignore_index=True,
    )
    validated = _validate_registry_dataframe(updated)
    _atomic_write_registry(validated, Path(path))
    return validated


def append_model_version(
    record: Mapping[str, object],
    path: Path = MODEL_REGISTRY_PATH,
) -> pd.DataFrame:
    """Append an immutable version atomically without overwriting history."""
    with model_registry_lock(path):
        return _append_model_version_unlocked(record, path)


def latest_model_versions(registry: pd.DataFrame) -> pd.DataFrame:
    """Return the newest preserved version for each scope/symbol identity."""
    data = _validate_registry_dataframe(registry)
    if data.empty:
        return data
    data = data.sort_values("model_version", kind="stable")
    return data.drop_duplicates(["model_scope", "symbol"], keep="last").reset_index(
        drop=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize or inspect the persistent model registry."""
    parser = argparse.ArgumentParser(description="Manage the PSX model registry")
    parser.add_argument("command", choices=("init", "show"))
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            path = initialize_model_registry()
            print(f"Model registry initialized: {path}")
        else:
            registry = load_model_registry()
            print(registry.to_string(index=False) if not registry.empty else "No model versions")
    except ModelRegistryError as exc:
        print(f"Model registry operation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
