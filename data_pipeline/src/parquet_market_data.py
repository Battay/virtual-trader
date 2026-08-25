"""Read-only access and quality auditing for consolidated PSX market Parquet.

PSX Data Sync owns and produces the Parquet file.  This module deliberately
contains no write, conversion, cleaning, download, or training integration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    DEFAULT_PSX_MARKET_PARQUET_PATH,
    PROJECT_ROOT,
    PSX_MARKET_PARQUET_ENV_VAR,
)


REQUIRED_MARKET_COLUMNS = (
    "market_date",
    "symbol",
    "ldcp",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
)
FLOAT_MARKET_COLUMNS = (
    "ldcp",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
)
OHLC_RELATIONSHIP_KEYS = (
    "high_below_low",
    "high_below_open",
    "high_below_close",
    "low_above_open",
    "low_above_close",
    "invalid_ohlc_rows",
)


class MarketParquetError(RuntimeError):
    """Base error for the read-only consolidated-market boundary."""


class MarketParquetNotFoundError(MarketParquetError):
    """Raised when the configured consolidated Parquet file is absent."""


class MarketParquetSchemaError(MarketParquetError):
    """Raised when a data load encounters an incompatible schema."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(str(error) for error in errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class MarketParquetSchemaValidation:
    """Broad Arrow-type compatibility result for the required contract."""

    valid: bool
    actual_types: Mapping[str, str]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "actual_types": dict(self.actual_types),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MarketParquetAudit:
    """Observation-only consolidated dataset audit result."""

    file_path: str
    file_size_bytes: int
    row_count: int
    unique_symbol_count: int | None
    unique_market_date_count: int | None
    earliest_market_date: str | None
    latest_market_date: str | None
    duplicate_market_date_symbol_count: int | None
    null_counts: Mapping[str, int | None]
    negative_volume_count: int | None
    non_positive_close_count: int | None
    zero_open_rows: int | None
    zero_high_rows: int | None
    zero_low_rows: int | None
    rows_with_any_zero_ohl: int | None
    invalid_ohlc_relationship_counts: Mapping[str, int | None]
    schema_valid: bool
    schema_errors: tuple[str, ...]
    schema_types: Mapping[str, str]

    @property
    def total_required_null_count(self) -> int:
        return sum(value for value in self.null_counts.values() if value is not None)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["null_counts"] = dict(self.null_counts)
        payload["invalid_ohlc_relationship_counts"] = dict(
            self.invalid_ohlc_relationship_counts
        )
        payload["schema_errors"] = list(self.schema_errors)
        payload["schema_types"] = dict(self.schema_types)
        payload["total_required_null_count"] = self.total_required_null_count
        return payload


def resolve_market_parquet_path(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve explicit, environment, or sibling-project development path.

    Relative explicit paths follow the caller's current working directory.
    Relative environment paths are anchored to ``virtual-trader`` so service
    launch directories cannot silently change their meaning.
    """

    if path is not None:
        candidate = Path(path).expanduser()
    else:
        environment = os.environ if environ is None else environ
        configured = str(environment.get(PSX_MARKET_PARQUET_ENV_VAR, "")).strip()
        if configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
        else:
            candidate = DEFAULT_PSX_MARKET_PARQUET_PATH
    return candidate.resolve(strict=False)


def _require_existing_file(path: str | os.PathLike[str] | None) -> Path:
    resolved = resolve_market_parquet_path(path)
    if not resolved.is_file():
        raise MarketParquetNotFoundError(
            f"Consolidated PSX market Parquet file does not exist: {resolved}"
        )
    return resolved


def _market_column_type_compatibility(
    column: str, arrow_type: pa.DataType
) -> tuple[bool, str]:
    if column == "market_date":
        return pa.types.is_date(arrow_type), "an Arrow date type"
    if column == "symbol":
        return (
            pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type),
            "an Arrow string type",
        )
    if column in FLOAT_MARKET_COLUMNS:
        return pa.types.is_floating(arrow_type), "an Arrow floating-point type"
    if column == "volume":
        return pa.types.is_integer(arrow_type), "an Arrow integer type"
    raise KeyError(f"No required market type contract for {column!r}")


def validate_market_parquet_schema(schema: pa.Schema) -> MarketParquetSchemaValidation:
    """Validate required fields using broad compatible Arrow data families."""

    actual_types = {field.name: str(field.type) for field in schema}
    errors: list[str] = []
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in schema.names]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    for column in REQUIRED_MARKET_COLUMNS:
        if column not in schema.names:
            continue
        arrow_type = schema.field(column).type
        compatible, expected = _market_column_type_compatibility(column, arrow_type)
        if not compatible:
            errors.append(
                f"Column {column!r} must be {expected}; found {arrow_type}"
            )
    return MarketParquetSchemaValidation(
        valid=not errors,
        actual_types=actual_types,
        errors=tuple(errors),
    )


def inspect_market_parquet_schema(
    path: str | os.PathLike[str] | None = None,
) -> MarketParquetSchemaValidation:
    """Inspect only Parquet metadata and return schema compatibility."""

    resolved = _require_existing_file(path)
    try:
        schema = pq.ParquetFile(resolved).schema_arrow
    except (OSError, pa.ArrowException) as exc:
        raise MarketParquetError(f"Could not inspect Parquet schema: {exc}") from exc
    return validate_market_parquet_schema(schema)


def _validated_file(path: str | os.PathLike[str] | None) -> Path:
    resolved = _require_existing_file(path)
    validation = inspect_market_parquet_schema(resolved)
    if not validation.valid:
        raise MarketParquetSchemaError(validation.errors)
    return resolved


def _parse_date(value: date | datetime | str | None, *, label: str) -> date | None:
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date or date-like value") from exc
    if pd.isna(parsed):
        raise ValueError(f"{label} cannot be missing")
    return parsed.date()


def _canonical_symbols(symbols: Sequence[str] | None) -> tuple[str, ...] | None:
    if symbols is None:
        return None
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a sequence, not a single string")
    canonical = tuple(sorted({str(symbol).strip() for symbol in symbols}))
    if not canonical or any(not symbol for symbol in canonical):
        raise ValueError("symbols cannot be empty")
    return canonical


def load_market_data(
    path: str | os.PathLike[str] | None = None,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load explicitly requested rows using Parquet predicate pushdown.

    Both date bounds are inclusive.  Returned rows are always stably ordered
    by ``market_date, symbol`` with a fresh zero-based index.
    """

    resolved = _validated_file(path)
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date cannot be after end_date")
    selected_symbols = _canonical_symbols(symbols)
    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("market_date", ">=", start))
    if end is not None:
        filters.append(("market_date", "<=", end))
    if selected_symbols is not None:
        filters.append(("symbol", "in", list(selected_symbols)))
    try:
        table = pq.read_table(resolved, filters=filters or None)
    except (OSError, pa.ArrowException) as exc:
        raise MarketParquetError(f"Could not read consolidated Parquet data: {exc}") from exc
    frame = table.to_pandas()
    return frame.sort_values(
        ["market_date", "symbol"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def load_market_calendar(
    path: str | os.PathLike[str] | None = None,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> pd.DatetimeIndex:
    """Load only the distinct market-date calendar in deterministic order.

    This deliberately reads no symbol, price, return, or volume values.  It is
    useful for defining temporal boundaries before a predicate-pushed TRAIN
    load, so sealed later partitions never need to enter memory.
    """

    resolved = _validated_file(path)
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date cannot be after end_date")
    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("market_date", ">=", start))
    if end is not None:
        filters.append(("market_date", "<=", end))
    try:
        table = pq.read_table(
            resolved,
            columns=["market_date"],
            filters=filters or None,
        )
    except (OSError, pa.ArrowException) as exc:
        raise MarketParquetError(
            f"Could not read consolidated market calendar: {exc}"
        ) from exc
    dates = pd.DatetimeIndex(
        pd.to_datetime(table.column("market_date").to_pandas(), errors="coerce")
    )
    if dates.isna().any():
        raise MarketParquetError("Consolidated market calendar contains null dates")
    return dates.unique().sort_values()


def load_market_date_range(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    *,
    path: str | os.PathLike[str] | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convenience API for an inclusive, predicate-pushed date range."""

    return load_market_data(
        path,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
    )


def _iso_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def audit_market_parquet(
    path: str | os.PathLike[str] | None = None,
) -> MarketParquetAudit:
    """Audit the source without cleaning, normalizing, or writing any data."""

    resolved = _require_existing_file(path)
    try:
        parquet_file = pq.ParquetFile(resolved)
    except (OSError, pa.ArrowException) as exc:
        raise MarketParquetError(f"Could not open consolidated Parquet data: {exc}") from exc
    validation = validate_market_parquet_schema(parquet_file.schema_arrow)
    compatible_columns = {
        column: (
            column in parquet_file.schema_arrow.names
            and _market_column_type_compatibility(
                column, parquet_file.schema_arrow.field(column).type
            )[0]
        )
        for column in REQUIRED_MARKET_COLUMNS
    }
    available = [
        column for column in REQUIRED_MARKET_COLUMNS if column in parquet_file.schema_arrow.names
    ]
    try:
        frame = pq.read_table(resolved, columns=available).to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise MarketParquetError(f"Could not audit consolidated Parquet data: {exc}") from exc

    null_counts: dict[str, int | None] = {
        column: int(frame[column].isna().sum()) if column in frame else None
        for column in REQUIRED_MARKET_COLUMNS
    }
    unique_symbols = (
        int(frame["symbol"].nunique(dropna=True))
        if compatible_columns["symbol"]
        else None
    )
    unique_dates = (
        int(frame["market_date"].nunique(dropna=True))
        if compatible_columns["market_date"]
        else None
    )
    if compatible_columns["market_date"]:
        valid_dates = frame["market_date"].dropna()
        earliest = _iso_or_none(valid_dates.min()) if not valid_dates.empty else None
        latest = _iso_or_none(valid_dates.max()) if not valid_dates.empty else None
    else:
        earliest = latest = None
    duplicate_count = (
        int(frame.duplicated(["market_date", "symbol"]).sum())
        if compatible_columns["market_date"] and compatible_columns["symbol"]
        else None
    )
    negative_volume = (
        int((frame["volume"].notna() & (frame["volume"] < 0)).sum())
        if compatible_columns["volume"]
        else None
    )
    non_positive_close = (
        int((frame["close"].notna() & (frame["close"] <= 0)).sum())
        if compatible_columns["close"]
        else None
    )

    zero_ohl_masks: dict[str, pd.Series] = {}
    zero_ohl_counts: dict[str, int | None] = {}
    for column in ("open", "high", "low"):
        if compatible_columns[column]:
            mask = frame[column].notna() & (frame[column] == 0)
            zero_ohl_masks[column] = mask
            zero_ohl_counts[column] = int(mask.sum())
        else:
            zero_ohl_counts[column] = None
    if len(zero_ohl_masks) == 3:
        any_zero_ohl = pd.Series(False, index=frame.index)
        for mask in zero_ohl_masks.values():
            any_zero_ohl |= mask
        rows_with_any_zero_ohl = int(any_zero_ohl.sum())
    else:
        rows_with_any_zero_ohl = None

    relationship_masks: dict[str, pd.Series] = {}
    relation_specs = {
        "high_below_low": ("high", "low", lambda left, right: left < right),
        "high_below_open": ("high", "open", lambda left, right: left < right),
        "high_below_close": ("high", "close", lambda left, right: left < right),
        "low_above_open": ("low", "open", lambda left, right: left > right),
        "low_above_close": ("low", "close", lambda left, right: left > right),
    }
    invalid_relationships: dict[str, int | None] = {}
    for name, (left_name, right_name, comparator) in relation_specs.items():
        if compatible_columns[left_name] and compatible_columns[right_name]:
            left = frame[left_name]
            right = frame[right_name]
            # PSX source-quality policy uses zero O/H/L as an allowed
            # unavailable-value sentinel.  A relationship is meaningful only
            # when both participating prices are strictly positive.
            comparable = left.notna() & right.notna() & (left > 0) & (right > 0)
            mask = comparable & comparator(left, right)
            relationship_masks[name] = mask
            invalid_relationships[name] = int(mask.sum())
        else:
            invalid_relationships[name] = None
    if len(relationship_masks) == len(relation_specs):
        combined = pd.Series(False, index=frame.index)
        for mask in relationship_masks.values():
            combined |= mask
        invalid_relationships["invalid_ohlc_rows"] = int(combined.sum())
    else:
        invalid_relationships["invalid_ohlc_rows"] = None

    return MarketParquetAudit(
        file_path=str(resolved),
        file_size_bytes=resolved.stat().st_size,
        row_count=int(parquet_file.metadata.num_rows),
        unique_symbol_count=unique_symbols,
        unique_market_date_count=unique_dates,
        earliest_market_date=earliest,
        latest_market_date=latest,
        duplicate_market_date_symbol_count=duplicate_count,
        null_counts=null_counts,
        negative_volume_count=negative_volume,
        non_positive_close_count=non_positive_close,
        zero_open_rows=zero_ohl_counts["open"],
        zero_high_rows=zero_ohl_counts["high"],
        zero_low_rows=zero_ohl_counts["low"],
        rows_with_any_zero_ohl=rows_with_any_zero_ohl,
        invalid_ohlc_relationship_counts={
            key: invalid_relationships.get(key) for key in OHLC_RELATIONSHIP_KEYS
        },
        schema_valid=validation.valid,
        schema_errors=validation.errors,
        schema_types=validation.actual_types,
    )


def _print_audit(audit: MarketParquetAudit) -> None:
    print(f"Path: {audit.file_path}")
    print(f"File size: {audit.file_size_bytes:,} bytes")
    print(f"Schema valid: {audit.schema_valid}")
    if audit.schema_errors:
        print("Schema errors: " + " | ".join(audit.schema_errors))
    print(f"Rows: {audit.row_count:,}")
    print(f"Symbols: {audit.unique_symbol_count}")
    print(f"Market dates: {audit.unique_market_date_count}")
    print(f"Date range: {audit.earliest_market_date} to {audit.latest_market_date}")
    print(
        "Duplicate (market_date, symbol) rows beyond first: "
        f"{audit.duplicate_market_date_symbol_count}"
    )
    print(f"Required-column nulls: {audit.total_required_null_count}")
    print("Null counts: " + json.dumps(dict(audit.null_counts), sort_keys=True))
    print(f"Negative volume rows: {audit.negative_volume_count}")
    print(f"Non-positive close rows: {audit.non_positive_close_count}")
    print("Source-policy-allowed unavailable zero O/H/L values:")
    print(f"  Zero open rows: {audit.zero_open_rows}")
    print(f"  Zero high rows: {audit.zero_high_rows}")
    print(f"  Zero low rows: {audit.zero_low_rows}")
    print(f"  Rows with any zero O/H/L: {audit.rows_with_any_zero_ohl}")
    print(
        "Positive comparable-value OHLC inconsistencies: "
        + json.dumps(dict(audit.invalid_ohlc_relationship_counts), sort_keys=True)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only inspection of PSX Data Sync consolidated Parquet."
    )
    parser.add_argument("--path", help=f"Override {PSX_MARKET_PARQUET_ENV_VAR}.")
    parser.add_argument("--audit", action="store_true", help="Run the full quality audit.")
    parser.add_argument(
        "--start", "--start-date", dest="start_date", help="Inclusive ISO start date."
    )
    parser.add_argument(
        "--end", "--end-date", dest="end_date", help="Inclusive ISO end date."
    )
    parser.add_argument(
        "--symbol", action="append", dest="symbols", help="Filter a symbol; repeatable."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.audit:
            audit = audit_market_parquet(args.path)
            _print_audit(audit)
            return 0 if audit.schema_valid else 2
        if not any((args.start_date, args.end_date, args.symbols)):
            parser.error("use --audit or provide a date/symbol filter")
        frame = load_market_data(
            args.path,
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols,
        )
        print(f"Rows: {len(frame):,}")
        print(f"Symbols: {frame['symbol'].nunique(dropna=True):,}")
        print(f"Market dates: {frame['market_date'].nunique(dropna=True):,}")
        if not frame.empty:
            print(
                "Date range: "
                f"{_iso_or_none(frame['market_date'].min())} to "
                f"{_iso_or_none(frame['market_date'].max())}"
            )
        return 0
    except (MarketParquetError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI subprocesses
    raise SystemExit(main())


__all__ = (
    "MarketParquetAudit",
    "MarketParquetError",
    "MarketParquetNotFoundError",
    "MarketParquetSchemaError",
    "MarketParquetSchemaValidation",
    "REQUIRED_MARKET_COLUMNS",
    "audit_market_parquet",
    "inspect_market_parquet_schema",
    "load_market_calendar",
    "load_market_data",
    "load_market_date_range",
    "main",
    "resolve_market_parquet_path",
    "validate_market_parquet_schema",
)
