"""Atomic, version-preserving model registry for future PPO training."""

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_pipeline.src.config import MODEL_REGISTRY_PATH
from feature_engineering.storage import atomic_write_dataframe, safe_path_component

from .paths import master_model_paths, symbol_model_paths


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
)
MODEL_SCOPES = frozenset({"symbol", "master"})
MODEL_STATUSES = frozenset({"not_trained", "training", "trained", "failed", "archived"})


class ModelRegistryError(ValueError):
    """Raised when model-registry structure or values are unsafe."""


def empty_model_registry() -> pd.DataFrame:
    """Return an empty registry with its stable machine-friendly schema."""
    return pd.DataFrame(columns=MODEL_REGISTRY_COLUMNS)


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> pd.DataFrame:
    """Load the model registry or return a schema-correct empty dataframe."""
    registry_path = Path(path)
    if not registry_path.exists():
        return empty_model_registry()
    try:
        data = pd.read_csv(registry_path, dtype={"symbol": "string", "model_id": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ModelRegistryError(f"Could not load model registry: {exc}") from exc
    missing = sorted(set(MODEL_REGISTRY_COLUMNS).difference(data.columns))
    if missing:
        raise ModelRegistryError(
            f"Model registry is missing columns: {', '.join(missing)}"
        )
    invalid_scopes = set(data["model_scope"].dropna().astype(str)).difference(MODEL_SCOPES)
    invalid_statuses = set(data["model_status"].dropna().astype(str)).difference(
        MODEL_STATUSES
    )
    if invalid_scopes:
        raise ModelRegistryError(
            f"Model registry contains invalid scopes: {', '.join(sorted(invalid_scopes))}"
        )
    if invalid_statuses:
        raise ModelRegistryError(
            "Model registry contains invalid statuses: "
            f"{', '.join(sorted(invalid_statuses))}"
        )
    if data["model_id"].duplicated().any():
        raise ModelRegistryError("Model registry contains duplicate model IDs")
    data["model_version"] = pd.to_numeric(data["model_version"], errors="coerce")
    if len(data) and data["model_version"].isna().any():
        raise ModelRegistryError("Model registry contains invalid model versions")
    retraining = data["needs_retraining"].astype("string").str.strip().str.lower()
    invalid_retraining = retraining.loc[
        ~retraining.isin({"true", "false", "1", "0"})
    ]
    if not invalid_retraining.empty:
        raise ModelRegistryError("Model registry contains invalid retraining flags")
    data["needs_retraining"] = retraining.isin({"true", "1"})
    return data.loc[:, MODEL_REGISTRY_COLUMNS].copy()


def initialize_model_registry(path: Path = MODEL_REGISTRY_PATH) -> Path:
    """Create an empty registry atomically without replacing an existing one."""
    destination = Path(path)
    if destination.exists():
        load_model_registry(destination)
        return destination
    atomic_write_dataframe(empty_model_registry(), destination)
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
    if registry.empty:
        return 1
    matches = registry.loc[registry["model_scope"].astype(str) == model_scope]
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
            "environment_version": "pending_3b",
            "created_at": timestamp.isoformat(),
            "new_data_days": 0,
            "needs_retraining": False,
            "model_path": str(paths.model),
            "scaler_path": str(paths.scaler),
            "metrics_path": str(paths.metrics),
        }
    )
    if values:
        unknown = sorted(set(values).difference(MODEL_REGISTRY_COLUMNS))
        if unknown:
            raise ModelRegistryError(
                f"Unknown model registry fields: {', '.join(unknown)}"
            )
        record.update(values)
    if record["model_status"] not in MODEL_STATUSES:
        raise ModelRegistryError(f"Unsupported model status: {record['model_status']}")
    return record


def append_model_version(
    record: Mapping[str, object],
    path: Path = MODEL_REGISTRY_PATH,
) -> pd.DataFrame:
    """Append an immutable version atomically without overwriting history."""
    registry = load_model_registry(path)
    missing = sorted(set(MODEL_REGISTRY_COLUMNS).difference(record))
    if missing:
        raise ModelRegistryError(f"Model record is missing: {', '.join(missing)}")
    model_id = str(record["model_id"])
    if model_id in set(registry["model_id"].astype(str)):
        raise ModelRegistryError(f"Model ID already exists: {model_id}")
    updated = pd.concat(
        [registry, pd.DataFrame([{column: record[column] for column in MODEL_REGISTRY_COLUMNS}])],
        ignore_index=True,
    )
    atomic_write_dataframe(updated, Path(path))
    return updated


def latest_model_versions(registry: pd.DataFrame) -> pd.DataFrame:
    """Return the newest preserved version for each scope/symbol identity."""
    if registry.empty:
        return registry.copy()
    data = registry.copy()
    data["model_version"] = pd.to_numeric(data["model_version"], errors="coerce")
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
