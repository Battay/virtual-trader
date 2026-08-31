"""Read-only global recurrent-model inventory and partition-boundary audit.

This module deliberately reads persisted JSON metadata, model hashes, and the
date column of the canonical per-symbol market CSV.  It never opens a TRAIN,
VALIDATION, or TEST partition dataframe and never loads a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from data_pipeline.src.config import (
    NATIVE_MARKET_SYMBOLS_DIR,
    PROCESSED_SPLITS_DIR,
    TRAINING_RUNS_DIR,
)
from feature_engineering.storage import safe_path_component
from reinforcement_learning.integrity import sha256_file

from .job_state import COMPLETED, TrainingJobRecord
from .recurrent_config import RECURRENT_PPO_CONFIG_VERSION
from .recurrent_orchestrator import TrainingRunStore
from .recurrent_trainer import RECURRENT_TRAINER_VERSION
from .selective_training import (
    _valid_run_stores,
    build_global_model_coverage,
    completed_job_is_trained,
)


SINGLE_SYMBOL_RL_PARTITION_CONTRACT = "rl_partition_v1"
RESEARCH_PARTITION_POLICY_VERSION = (
    "per_symbol_chronological_distinct_dates_70_15_15_v1"
)
SINGLE_SYMBOL_RL_PARTITION_PROTOCOL = {
    "name": "Single-symbol RL partition protocol",
    "contract": SINGLE_SYMBOL_RL_PARTITION_CONTRACT,
    "version": RESEARCH_PARTITION_POLICY_VERSION,
    "scope": "symbol",
    "train": (
        "first floor(70%) of each symbol's usable chronological observations"
    ),
    "validation": "next floor(15%) of that symbol's usable observations",
    "test": "remaining usable observations (approximately 15%); SEALED",
    "rule": (
        "symbol-specific chronological 70% TRAIN / 15% VALIDATION / "
        "remaining TEST allocation"
    ),
    "boundary_rule": "one market date cannot cross a partition",
    "feature_timing": "causal features are built before the split; warm-up rows are removed",
    "normalization": "scaler fit on TRAIN only; VALIDATION/TEST only transformed",
    "clustering_note": (
        "Clustering / relationship studies use a different common frozen "
        "temporal protocol. Those fixed research cutoffs do not define "
        "rl_partition_v1."
    ),
}
# Backward-compatible name for callers written before the UI terminology was
# clarified. The persisted split version and split algorithm are unchanged.
RESEARCH_PARTITION_POLICY = SINGLE_SYMBOL_RL_PARTITION_PROTOCOL


class ModelDetailsAuditError(RuntimeError):
    """Raised when persisted model or partition metadata fails closed."""


@dataclass(frozen=True)
class PartitionBoundary:
    """One persisted metadata-only partition boundary."""

    start: str
    end: str
    rows: int


@dataclass(frozen=True)
class SymbolPartitionManifest:
    """Validated v1 boundary metadata without opening partition observations."""

    symbol: str
    raw_available_start: str
    raw_available_end: str
    raw_available_rows: int
    usable_feature_start: str
    usable_feature_end: str
    usable_feature_rows: int
    train: PartitionBoundary
    validation: PartitionBoundary
    test: PartitionBoundary
    partition_contract_version: str
    recurrent_contract_version: str
    feature_version: str
    environment_version: str
    scaler_fit_partition: str
    split_policy_version: str
    test_sealed: bool
    test_metadata_only: bool
    contract_path: str
    source_contract_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in ("train", "validation", "test"):
            boundary = payload.pop(name)
            payload.update(
                {
                    f"{name}_start": boundary["start"],
                    f"{name}_end": boundary["end"],
                    f"{name}_rows": boundary["rows"],
                }
            )
        return payload


def single_symbol_rl_partition_protocol() -> dict[str, str]:
    """Return the exact persisted v1 single-symbol partition description."""

    return dict(SINGLE_SYMBOL_RL_PARTITION_PROTOCOL)


def research_partition_policy() -> dict[str, str]:
    """Compatibility alias for :func:`single_symbol_rl_partition_protocol`."""

    return single_symbol_rl_partition_protocol()


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelDetailsAuditError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelDetailsAuditError(f"{label} must contain a JSON object: {path}")
    return payload


def _boundary(value: object, *, label: str) -> PartitionBoundary:
    if not isinstance(value, Mapping):
        raise ModelDetailsAuditError(f"{label} boundary metadata is missing")
    try:
        start = str(value["start"])
        end = str(value["end"])
        rows = int(value["rows"])
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelDetailsAuditError(f"{label} boundary metadata is invalid") from exc
    if rows < 1 or start_date > end_date:
        raise ModelDetailsAuditError(f"{label} boundary metadata is invalid")
    return PartitionBoundary(start=start, end=end, rows=rows)


def _raw_date_bounds(
    symbol: str,
    *,
    market_symbols_dir: Path,
) -> tuple[str, str, int]:
    """Read only current raw market-date availability, never OHLCV/returns."""

    path = Path(market_symbols_dir) / f"{safe_path_component(symbol)}.csv"
    try:
        frame = pd.read_csv(path, usecols=["market_date"])
    except (OSError, ValueError) as exc:
        raise ModelDetailsAuditError(
            f"canonical market-date metadata is unavailable for {symbol}: {exc}"
        ) from exc
    dates = pd.to_datetime(frame["market_date"], errors="coerce")
    if dates.empty or dates.isna().any():
        raise ModelDetailsAuditError(
            f"canonical market-date metadata is invalid for {symbol}"
        )
    return (
        dates.min().date().isoformat(),
        dates.max().date().isoformat(),
        int(dates.nunique()),
    )


def read_symbol_partition_manifest(
    symbol: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    market_symbols_dir: Path = NATIVE_MARKET_SYMBOLS_DIR,
) -> SymbolPartitionManifest:
    """Validate boundary JSON only; partition CSV observations stay unopened."""

    symbol_text = str(symbol).strip()
    directory = Path(splits_dir) / "symbols" / safe_path_component(symbol_text)
    metadata_path = directory / "metadata.json"
    recurrent_path = directory / "recurrent" / "recurrent_contract.json"
    boundaries_path = directory / "recurrent" / "episode_boundaries.json"
    scaler_metadata_path = directory / "rl_observation_scaler.json"
    split = _json_object(metadata_path, label=f"{symbol_text} split metadata")
    recurrent = _json_object(
        recurrent_path, label=f"{symbol_text} recurrent contract metadata"
    )

    proportions = split.get("proportions")
    expected_proportions = {"training": 0.70, "validation": 0.15, "testing": 0.15}
    if split.get("scope") != "symbol" or proportions != expected_proportions:
        raise ModelDetailsAuditError(
            f"{symbol_text} does not use the authoritative per-symbol 70/15/15 split"
        )
    train = _boundary(split.get("training"), label=f"{symbol_text} TRAIN")
    validation = _boundary(
        split.get("validation"), label=f"{symbol_text} VALIDATION"
    )
    test = _boundary(split.get("testing"), label=f"{symbol_text} TEST")
    if not date.fromisoformat(train.end) < date.fromisoformat(validation.start):
        raise ModelDetailsAuditError(f"{symbol_text} TRAIN/VALIDATION overlap")
    if not date.fromisoformat(validation.end) < date.fromisoformat(test.start):
        raise ModelDetailsAuditError(f"{symbol_text} VALIDATION/TEST overlap")

    total_rows = train.rows + validation.rows + test.rows
    expected_train = max(1, int(total_rows * 0.70))
    expected_validation = max(1, int(total_rows * 0.15))
    if expected_train + expected_validation >= total_rows:
        expected_train, expected_validation = total_rows - 2, 1
    if (train.rows, validation.rows) != (expected_train, expected_validation):
        raise ModelDetailsAuditError(
            f"{symbol_text} persisted row counts do not reproduce the split policy"
        )

    if recurrent.get("artifact_schema_version") != "rl_recurrent_partition_v1":
        raise ModelDetailsAuditError(
            f"{symbol_text} recurrent partition contract is incompatible"
        )
    if recurrent.get("source_rl_contract_version") != "rl_partition_v1":
        raise ModelDetailsAuditError(f"{symbol_text} RL partition contract is incompatible")
    if recurrent.get("feature_version") != split.get("feature_version"):
        raise ModelDetailsAuditError(f"{symbol_text} feature versions differ")
    normalization = recurrent.get("normalization")
    if not isinstance(normalization, Mapping) or normalization.get("fit_partition") != "train":
        raise ModelDetailsAuditError(f"{symbol_text} scaler is not TRAIN-fitted")
    recurrent_partitions = recurrent.get("partitions")
    if not isinstance(recurrent_partitions, Mapping):
        raise ModelDetailsAuditError(f"{symbol_text} recurrent boundaries are missing")
    for name, expected in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        actual = _boundary(
            recurrent_partitions.get(name), label=f"{symbol_text} recurrent {name}"
        )
        if actual != expected:
            raise ModelDetailsAuditError(
                f"{symbol_text} recurrent {name} boundary differs from split metadata"
            )
    recurrent_test = recurrent_partitions.get("test")
    test_sealing = recurrent.get("test_sealing")
    test_metadata_only = bool(
        isinstance(recurrent_test, Mapping)
        and recurrent_test.get("sealed") is True
        and recurrent_test.get("frame_access") == "sealed_metadata_only"
        and isinstance(test_sealing, Mapping)
        and test_sealing.get("sealed") is True
        and test_sealing.get("metadata_only") is True
        and test_sealing.get("evaluation_performed") is False
        and test_sealing.get("frame_loaded_during_build") is False
    )
    if not test_metadata_only:
        raise ModelDetailsAuditError(f"{symbol_text} TEST sealing metadata is incompatible")
    sequence = recurrent.get("sequence")
    if not isinstance(sequence, Mapping) or (
        sequence.get("episode_strategy") != "full_partition"
        or sequence.get("fixed_windows_enabled") is not False
    ):
        raise ModelDetailsAuditError(
            f"{symbol_text} recurrent episode boundaries are incompatible"
        )

    boundaries = _json_object(
        boundaries_path, label=f"{symbol_text} recurrent episode boundaries"
    )
    if (
        boundaries.get("artifact_schema_version")
        != "rl_recurrent_episode_boundaries_v1"
        or boundaries.get("recurrent_contract_version")
        != "rl_recurrent_partition_v1"
        or boundaries.get("symbol") != symbol_text
        or boundaries.get("episode_strategy") != "full_partition"
    ):
        raise ModelDetailsAuditError(
            f"{symbol_text} episode-boundary metadata is incompatible"
        )
    boundary_partitions = boundaries.get("partitions")
    if not isinstance(boundary_partitions, Mapping):
        raise ModelDetailsAuditError(f"{symbol_text} episode partitions are missing")
    for name, expected in (("train", train), ("validation", validation)):
        episodes = boundary_partitions.get(name)
        if not isinstance(episodes, list) or len(episodes) != 1:
            raise ModelDetailsAuditError(
                f"{symbol_text} {name} must be one complete recurrent episode"
            )
        episode = episodes[0]
        if not isinstance(episode, Mapping) or (
            episode.get("symbol") != symbol_text
            or episode.get("partition") != name
            or int(episode.get("start_row", -1)) != 0
            or int(episode.get("end_row", -1)) != expected.rows - 1
            or int(episode.get("rows", 0)) != expected.rows
            or episode.get("start") != expected.start
            or episode.get("end") != expected.end
        ):
            raise ModelDetailsAuditError(
                f"{symbol_text} {name} recurrent episode crosses its partition"
            )
    scaler_metadata = _json_object(
        scaler_metadata_path, label=f"{symbol_text} scaler metadata"
    )
    if int(scaler_metadata.get("training_rows", 0)) != train.rows:
        raise ModelDetailsAuditError(
            f"{symbol_text} scaler metadata is not tied to the TRAIN rows"
        )

    raw_start, raw_end, raw_rows = _raw_date_bounds(
        symbol_text, market_symbols_dir=Path(market_symbols_dir)
    )
    return SymbolPartitionManifest(
        symbol=symbol_text,
        raw_available_start=raw_start,
        raw_available_end=raw_end,
        raw_available_rows=raw_rows,
        usable_feature_start=train.start,
        usable_feature_end=test.end,
        usable_feature_rows=total_rows,
        train=train,
        validation=validation,
        test=test,
        partition_contract_version="rl_partition_v1",
        recurrent_contract_version="rl_recurrent_partition_v1",
        feature_version=str(split.get("feature_version", "")),
        environment_version=str(recurrent.get("environment_version", "")),
        scaler_fit_partition="train",
        split_policy_version=RESEARCH_PARTITION_POLICY_VERSION,
        test_sealed=True,
        test_metadata_only=True,
        contract_path=str(recurrent_path.resolve()),
        source_contract_sha256=sha256_file(recurrent_path),
    )


def _validation_metadata(
    store: TrainingRunStore, job: TrainingJobRecord
) -> dict[str, object]:
    if not job.validation_metrics_reference:
        raise ModelDetailsAuditError(f"{job.symbol} has no validation metadata")
    payload = _json_object(
        store.resolve_artifact(job.validation_metrics_reference),
        label=f"{job.symbol} validation metadata",
    )
    if (
        payload.get("evaluation_partition") != "validation"
        or payload.get("model_parameters_unchanged") is not True
        or payload.get("test_evaluated") is True
    ):
        raise ModelDetailsAuditError(
            f"{job.symbol} validation provenance is incompatible"
        )
    return payload


def _training_log_metadata(
    store: TrainingRunStore, job: TrainingJobRecord
) -> dict[str, object]:
    path = (
        store.run_directory
        / "logs"
        / safe_path_component(job.symbol)
        / f"attempt_{job.retry_count:03d}.json"
    )
    payload = _json_object(path, label=f"{job.symbol} training log metadata")
    if (
        payload.get("status") != "completed"
        or payload.get("test_partition_loaded") is not False
    ):
        raise ModelDetailsAuditError(f"{job.symbol} training provenance is incompatible")
    return payload


def build_global_verified_model_inventory(
    *,
    coverage: pd.DataFrame | None = None,
    runs_root: Path = TRAINING_RUNS_DIR,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    market_symbols_dir: Path = NATIVE_MARKET_SYMBOLS_DIR,
    partition_loader: Callable[..., SymbolPartitionManifest] = read_symbol_partition_manifest,
) -> pd.DataFrame:
    """Return one latest verified recurrent model per frozen-eligible symbol."""

    from .production_control import classify_run

    coverage_table = (
        build_global_model_coverage(runs_root=Path(runs_root))[0]
        if coverage is None
        else coverage.copy(deep=True)
    )
    required = {"symbol", "company_name", "sector", "trained"}
    if not required.issubset(coverage_table.columns):
        raise ModelDetailsAuditError("global coverage is missing model identity columns")
    trained_identity = coverage_table.loc[
        coverage_table["trained"].astype(bool),
        ["symbol", "company_name", "sector"],
    ].copy()
    identity = trained_identity.set_index("symbol").to_dict(orient="index")
    candidates: dict[str, list[dict[str, object]]] = {
        str(symbol): [] for symbol in trained_identity["symbol"]
    }
    for store in _valid_run_stores(Path(runs_root)):
        manifest = store.read_manifest()
        run_type = classify_run(manifest, store.run_directory)
        for job in store.list_jobs():
            if job.symbol not in candidates:
                continue
            verified, verification = completed_job_is_trained(store, job)
            if not verified:
                continue
            candidates[job.symbol].append(
                {
                    "store": store,
                    "manifest": manifest,
                    "job": job,
                    "run_type": run_type,
                    "verification": verification,
                }
            )

    rows: list[dict[str, object]] = []
    for symbol in sorted(candidates):
        history = sorted(
            candidates[symbol],
            key=lambda item: (
                str(item["job"].updated_at),
                str(item["manifest"].run_id),
            ),
        )
        if not history:
            # Coverage may be stale or injected. Never fabricate availability.
            continue
        selected = history[-1]
        store = selected["store"]
        manifest = selected["manifest"]
        job = selected["job"]
        if not isinstance(store, TrainingRunStore) or not isinstance(job, TrainingJobRecord):
            raise ModelDetailsAuditError("verified run history contains invalid objects")
        partition = partition_loader(
            symbol,
            splits_dir=Path(splits_dir),
            market_symbols_dir=Path(market_symbols_dir),
        )
        if (
            job.status != COMPLETED
            or job.source_data_hash != partition.source_contract_sha256
            or job.data_contract_version != partition.recurrent_contract_version
            or job.feature_version != partition.feature_version
            or job.environment_version != partition.environment_version
            or job.agent_version != RECURRENT_TRAINER_VERSION
        ):
            raise ModelDetailsAuditError(
                f"{symbol} model and partition contract metadata differ"
            )
        validation = _validation_metadata(store, job)
        training_log = _training_log_metadata(store, job)
        if (
            validation.get("recurrent_contract_version")
            != partition.recurrent_contract_version
            or validation.get("feature_version") != partition.feature_version
            or validation.get("environment_version") != partition.environment_version
            or training_log.get("training_start") != partition.train.start
            or training_log.get("training_end") != partition.train.end
            or int(training_log.get("training_rows", -1)) != partition.train.rows
        ):
            raise ModelDetailsAuditError(
                f"{symbol} training/validation metadata differs from its partition"
            )
        state_order = [str(item.get("to", "")) for item in job.state_history]
        if "TRAINING" not in state_order or "VALIDATING" not in state_order:
            raise ModelDetailsAuditError(
                f"{symbol} does not prove validation occurred after training"
            )
        if state_order.index("VALIDATING") <= state_order.index("TRAINING"):
            raise ModelDetailsAuditError(
                f"{symbol} validation transition precedes training"
            )
        if manifest.test_partition_loaded:
            raise ModelDetailsAuditError(f"{symbol} run reports TEST access")

        row = {
            "symbol": symbol,
            "company_name": identity[symbol].get("company_name"),
            "sector": identity[symbol].get("sector"),
            "run_type": str(selected["run_type"]),
            "run_id": manifest.run_id,
            "attempt": job.retry_count + 1,
            "training_status": job.status,
            "validation_status": job.validation_status,
            "artifact_verification": str(selected["verification"]),
            "algorithm": "RecurrentPPO",
            "policy": "MlpLstmPolicy",
            "trainer_version": job.agent_version,
            "recurrent_config_version": RECURRENT_PPO_CONFIG_VERSION,
            "requested_timesteps": job.requested_timesteps,
            "actual_timesteps": job.completed_timesteps,
            "seed": job.seed,
            "hyperparameters_hash": job.hyperparameters_hash,
            "requested_device": job.requested_device,
            "effective_device": job.effective_device,
            "runtime_seconds": job.wall_clock_duration_seconds,
            "validation_partition": str(validation.get("evaluation_partition")),
            "validation_after_training": True,
            "validation_parameters_unchanged": True,
            "test_partition_loaded": False,
            "run_directory": str(store.run_directory.resolve()),
            "model_path": job.model_path,
            "model_sha256": job.model_sha256,
            "validation_metrics_reference": job.validation_metrics_reference,
            "diagnostics_available": bool(training_log.get("training_diagnostics")),
        }
        row.update(partition.to_dict())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(
        drop=True
    )


def audit_model_contract_compatibility(inventory: pd.DataFrame) -> dict[str, object]:
    """Fail closed on methodology drift while permitting symbol/date differences."""

    if inventory.empty:
        raise ModelDetailsAuditError("verified model inventory is empty")
    fields = (
        "partition_contract_version",
        "recurrent_contract_version",
        "feature_version",
        "environment_version",
        "algorithm",
        "policy",
        "trainer_version",
        "recurrent_config_version",
        "requested_timesteps",
        "seed",
        "hyperparameters_hash",
        "split_policy_version",
        "scaler_fit_partition",
    )
    differences = {
        field: sorted({str(value) for value in inventory[field].dropna()})
        for field in fields
        if inventory[field].nunique(dropna=False) != 1
    }
    return {
        "compatible": not differences,
        "differences": differences,
        "model_count": len(inventory),
        "run_types": sorted(inventory["run_type"].unique()),
        "test_partition_loaded": bool(inventory["test_partition_loaded"].any()),
        "partition_overlap": bool(
            (
                pd.to_datetime(inventory["train_end"])
                >= pd.to_datetime(inventory["validation_start"])
            ).any()
            or (
                pd.to_datetime(inventory["validation_end"])
                >= pd.to_datetime(inventory["test_start"])
            ).any()
        ),
    }


__all__ = [
    "ModelDetailsAuditError",
    "PartitionBoundary",
    "RESEARCH_PARTITION_POLICY",
    "RESEARCH_PARTITION_POLICY_VERSION",
    "SINGLE_SYMBOL_RL_PARTITION_CONTRACT",
    "SINGLE_SYMBOL_RL_PARTITION_PROTOCOL",
    "SymbolPartitionManifest",
    "audit_model_contract_compatibility",
    "build_global_verified_model_inventory",
    "read_symbol_partition_manifest",
    "research_partition_policy",
    "single_symbol_rl_partition_protocol",
]
