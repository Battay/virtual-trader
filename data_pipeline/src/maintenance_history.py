"""Atomic persistent history for market-data maintenance operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence
from uuid import uuid4

from .config import DATA_MAINTENANCE_HISTORY_PATH


MAINTENANCE_HISTORY_VERSION = "data_maintenance_history_v1"
MAINTENANCE_OPERATION_TYPES = frozenset(
    {
        "FETCH_SELECTED",
        "PARQUET_REPAIR_SELECTED",
        "MASTER_RECONCILE",
        "FULL_REBUILD",
        "AUTOMATION_UPDATE",
    }
)


class MaintenanceHistoryError(RuntimeError):
    """Raised when durable maintenance history cannot be trusted or written."""


@dataclass(frozen=True)
class MaintenanceDateResult:
    """One date's result within a maintenance operation."""

    trading_date: str
    status: str
    message: str = ""
    input_classification: str = ""
    attempts: int = 0
    output_path: str | None = None


@dataclass(frozen=True)
class MaintenanceOperation:
    """Auditable record of one finalized market-data maintenance operation."""

    operation_id: str
    operation_type: str
    timestamp: str
    requested_dates: tuple[str, ...] = ()
    executed_dates: tuple[str, ...] = ()
    skipped_dates: tuple[str, ...] = ()
    per_date_results: tuple[MaintenanceDateResult, ...] = ()
    artifact_status: Mapping[str, object] | None = None
    errors: tuple[str, ...] = ()
    master_latest_date: str | None = None
    source_set_hash: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifact_status"] = dict(self.artifact_status or {})
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "MaintenanceOperation":
        if not isinstance(payload, dict):
            raise ValueError("maintenance operation must be an object")
        operation_type = str(payload.get("operation_type") or "")
        if operation_type not in MAINTENANCE_OPERATION_TYPES:
            raise ValueError(f"unsupported maintenance operation: {operation_type}")
        raw_results = payload.get("per_date_results", [])
        if not isinstance(raw_results, list):
            raise ValueError("per_date_results must be a list")
        if any(not isinstance(item, dict) for item in raw_results):
            raise ValueError("per_date_results contains a non-object entry")
        results = tuple(
            MaintenanceDateResult(
                trading_date=str(item["trading_date"]),
                status=str(item["status"]),
                message=str(item.get("message") or ""),
                input_classification=str(item.get("input_classification") or ""),
                attempts=int(item.get("attempts", 0)),
                output_path=(
                    str(item["output_path"]) if item.get("output_path") else None
                ),
            )
            for item in raw_results
        )

        def strings(name: str) -> tuple[str, ...]:
            value = payload.get(name, [])
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            return tuple(str(item) for item in value)

        artifact_status = payload.get("artifact_status", {})
        if not isinstance(artifact_status, dict):
            raise ValueError("artifact_status must be an object")
        return cls(
            operation_id=str(payload["operation_id"]),
            operation_type=operation_type,
            timestamp=str(payload["timestamp"]),
            requested_dates=strings("requested_dates"),
            executed_dates=strings("executed_dates"),
            skipped_dates=strings("skipped_dates"),
            per_date_results=results,
            artifact_status=dict(artifact_status),
            errors=strings("errors"),
            master_latest_date=(
                str(payload["master_latest_date"])
                if payload.get("master_latest_date")
                else None
            ),
            source_set_hash=(
                str(payload["source_set_hash"])
                if payload.get("source_set_hash")
                else None
            ),
            content_hash=(
                str(payload["content_hash"])
                if payload.get("content_hash")
                else None
            ),
        )


@dataclass(frozen=True)
class MaintenanceHistory:
    """Loaded operation ledger plus a non-destructive parse diagnostic."""

    entries: tuple[MaintenanceOperation, ...] = ()
    error: str | None = None
    version: str = MAINTENANCE_HISTORY_VERSION


def new_operation(
    operation_type: str,
    *,
    requested_dates: Sequence[str] = (),
    executed_dates: Sequence[str] = (),
    skipped_dates: Sequence[str] = (),
    per_date_results: Sequence[MaintenanceDateResult] = (),
    artifact_status: Mapping[str, object] | None = None,
    errors: Sequence[str] = (),
    master_latest_date: str | None = None,
    source_set_hash: str | None = None,
    content_hash: str | None = None,
    operation_id: str | None = None,
    timestamp: str | None = None,
) -> MaintenanceOperation:
    """Create a normalized finalized operation without writing it."""

    if operation_type not in MAINTENANCE_OPERATION_TYPES:
        raise ValueError(f"unsupported maintenance operation: {operation_type}")
    return MaintenanceOperation(
        operation_id=operation_id or uuid4().hex,
        operation_type=operation_type,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        requested_dates=tuple(sorted({str(value) for value in requested_dates})),
        executed_dates=tuple(sorted({str(value) for value in executed_dates})),
        skipped_dates=tuple(sorted({str(value) for value in skipped_dates})),
        per_date_results=tuple(
            sorted(per_date_results, key=lambda item: item.trading_date)
        ),
        artifact_status=dict(artifact_status or {}),
        errors=tuple(str(value) for value in errors),
        master_latest_date=master_latest_date,
        source_set_hash=source_set_hash,
        content_hash=content_hash,
    )


def load_maintenance_history(
    path: Path = DATA_MAINTENANCE_HISTORY_PATH,
) -> MaintenanceHistory:
    """Load history safely; malformed history is reported and never discarded."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return MaintenanceHistory()
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("maintenance history must be an object")
        version = str(payload.get("version") or "")
        if version != MAINTENANCE_HISTORY_VERSION:
            raise ValueError(f"unsupported maintenance history version: {version}")
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("maintenance history entries must be a list")
        entries = tuple(MaintenanceOperation.from_dict(item) for item in raw_entries)
        return MaintenanceHistory(entries=entries, version=version)
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return MaintenanceHistory(error=f"Could not load maintenance history: {exc}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_maintenance_operation(
    operation: MaintenanceOperation,
    path: Path = DATA_MAINTENANCE_HISTORY_PATH,
) -> Path:
    """Atomically append one operation, failing closed on malformed history."""

    existing = load_maintenance_history(path)
    if existing.error is not None:
        raise MaintenanceHistoryError(existing.error)
    if any(item.operation_id == operation.operation_id for item in existing.entries):
        raise MaintenanceHistoryError(
            f"Duplicate maintenance operation id: {operation.operation_id}"
        )
    payload = {
        "version": MAINTENANCE_HISTORY_VERSION,
        "entries": [
            item.to_dict() for item in (*existing.entries, operation)
        ],
    }
    destination = Path(path)
    _atomic_write(
        destination,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return destination
