"""Native, deterministic CSV -> sector-enriched CSV/Parquet market pipeline.

Daily CSV files are immutable ingestion evidence.  Every generated artifact is
derived from one normalized in-memory record contract, preventing the master
CSV, per-symbol CSVs, daily Parquets, and consolidated Parquet from acquiring
different transformation semantics.

Sector fields describe the *current* authoritative listing snapshot.  They are
identity/context annotations, not historical point-in-time sector membership.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    BOOTSTRAP_PSX_MARKET_PARQUET_PATH,
    CURRENT_LISTINGS_PATH,
    DAILY_MARKET_PARQUET_DIR,
    LOCAL_PSX_MARKET_PARQUET_PATH,
    NATIVE_MARKET_MASTER_PATH,
    NATIVE_MARKET_PIPELINE_STATE_PATH,
    NATIVE_MARKET_SYMBOLS_DIR,
    PROJECT_ROOT,
    RAW_CSV_DIR,
)
from .official_listings import load_listing_snapshot


NATIVE_MARKET_SCHEMA_VERSION = "native_market_record_v1"
NATIVE_MARKET_PIPELINE_VERSION = "native_csv_parquet_pipeline_v1"
SECTOR_CONTEXT = "current_listing_context_not_historical_membership"
ALLOWED_SECTOR_SECURITY_TYPES = frozenset({"ordinary_equity", "gem_equity"})

CORE_MARKET_COLUMNS = (
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
SECTOR_PROVENANCE_COLUMNS = (
    "sector_current",
    "sector_source",
    "sector_snapshot_date",
)
CANONICAL_MARKET_COLUMNS = CORE_MARKET_COLUMNS + SECTOR_PROVENANCE_COLUMNS
FLOAT_COLUMNS = ("ldcp", "open", "high", "low", "close", "change", "change_percent")
BUSINESS_KEY = ("market_date", "symbol")

CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        *(pa.field(column, pa.float64(), nullable=False) for column in FLOAT_COLUMNS),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("sector_current", pa.string()),
        pa.field("sector_source", pa.string()),
        pa.field("sector_snapshot_date", pa.string()),
    ]
)


class NativeMarketPipelineError(RuntimeError):
    """Raised when native-market generation cannot complete safely."""


class DuplicateMarketRecordError(NativeMarketPipelineError):
    """Raised when source records collide on the canonical business key."""


class MigrationComparisonError(NativeMarketPipelineError):
    """Raised when the native artifact differs from the bootstrap Parquet."""


@dataclass(frozen=True)
class NativeMarketPaths:
    """All generated destinations; injectable so tests never touch real data."""

    master_csv: Path = NATIVE_MARKET_MASTER_PATH
    symbol_csv_dir: Path = NATIVE_MARKET_SYMBOLS_DIR
    daily_parquet_dir: Path = DAILY_MARKET_PARQUET_DIR
    consolidated_parquet: Path = LOCAL_PSX_MARKET_PARQUET_PATH
    state_json: Path = NATIVE_MARKET_PIPELINE_STATE_PATH


@dataclass(frozen=True)
class SourceEvidence:
    """Portable identity for one immutable daily CSV input."""

    name: str
    sha256: str
    size_bytes: int
    origin: str = "unspecified"


@dataclass(frozen=True)
class MigrationComparison:
    """Exact overlapping-core comparison against the bootstrap Parquet."""

    passed: bool
    native_rows: int
    bootstrap_rows: int
    native_symbols: int
    bootstrap_symbols: int
    native_dates: int
    bootstrap_dates: int
    native_duplicates: int
    bootstrap_duplicates: int
    native_core_hash: str
    bootstrap_core_hash: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeMarketBuildResult:
    """Structured result for a successfully promoted native-market build."""

    operation: str
    source_files: int
    source_dates: tuple[str, ...]
    rows_read: int
    rows_accepted: int
    rows_rejected: int
    duplicate_count: int
    master_rows: int
    consolidated_rows: int
    symbol_count: int
    earliest_date: str | None
    latest_date: str | None
    sector_matched_symbols: int
    schema_version: str
    content_hash: str
    source_set_hash: str
    consolidated_sha256: str
    status: str
    paths: NativeMarketPaths = field(repr=False, compare=False)
    rows_added: int = 0
    rows_replaced: int = 0
    daily_parquets_written: int = 0
    symbol_csvs_written: int = 0
    migration_comparison: MigrationComparison | None = None
    idempotent_noop: bool = False


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 without changing source metadata or content."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_name(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Fixtures and bootstrap sources stay portable by using their basename.
        return Path(path).name


def _source_origin(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(RAW_CSV_DIR.resolve())
        return "virtual_trader_raw_csv"
    except ValueError:
        pass
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
        return "virtual_trader_project_source"
    except ValueError:
        return "external_validated_csv"


def source_evidence(path: Path) -> SourceEvidence:
    resolved = Path(path)
    if not resolved.is_file():
        raise NativeMarketPipelineError(f"Source CSV does not exist: {resolved}")
    return SourceEvidence(
        name=_portable_source_name(resolved),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
        origin=_source_origin(resolved),
    )


def _canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_set_hash(sources: Sequence[SourceEvidence]) -> str:
    payload = [asdict(item) for item in sorted(sources, key=lambda item: item.name)]
    return _canonical_json_hash(payload)


def canonical_content_hash(records: pd.DataFrame, *, core_only: bool = False) -> str:
    """Hash normalized values independently of Parquet/CSV physical encoding."""

    columns = CORE_MARKET_COLUMNS if core_only else CANONICAL_MARKET_COLUMNS
    ordered = records.loc[:, list(columns)].copy()
    ordered["market_date"] = pd.to_datetime(ordered["market_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    ordered["symbol"] = ordered["symbol"].astype("string")
    # CSV type inference may represent an all-integral price column as int64,
    # while the canonical Arrow schema stores every market price as float64.
    # Normalize logical types before hashing so equivalent values have one
    # identity regardless of their physical CSV/Parquet representation.
    for column in FLOAT_COLUMNS:
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(
                ordered[column], errors="raise"
            ).astype("float64")
    if "volume" in ordered.columns:
        ordered["volume"] = pd.to_numeric(
            ordered["volume"], errors="raise"
        ).astype("int64")
    for column in set(columns).intersection(SECTOR_PROVENANCE_COLUMNS):
        ordered[column] = ordered[column].astype("string").fillna("<NULL>")
    row_hashes = pd.util.hash_pandas_object(
        ordered, index=False, categorize=False
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update(("core" if core_only else NATIVE_MARKET_SCHEMA_VERSION).encode())
    digest.update("|".join(columns).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _source_csv_paths(source_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(Path(source_dir).glob("market_*.csv")))
    if not paths:
        raise NativeMarketPipelineError(
            f"No daily market CSV files found in {Path(source_dir)}"
        )
    return paths


def _read_source_csv(path: Path) -> pd.DataFrame:
    try:
        source = pd.read_csv(path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise NativeMarketPipelineError(f"Could not read source CSV {path}: {exc}") from exc
    if "market_date" not in source and "date" in source:
        source = source.rename(columns={"date": "market_date"})
    if "market_date" not in source:
        name = Path(path).stem
        encoded_date = name.removeprefix("market_")
        try:
            parsed_name_date = pd.Timestamp(encoded_date).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise NativeMarketPipelineError(
                f"Source CSV {path} has no date column and no market_YYYY-MM-DD filename"
            ) from exc
        source["market_date"] = parsed_name_date
    missing = [column for column in CORE_MARKET_COLUMNS if column not in source]
    if missing:
        raise NativeMarketPipelineError(
            f"Source CSV {path} is missing required columns: {', '.join(missing)}"
        )
    return source.loc[:, list(CORE_MARKET_COLUMNS)].copy()


def normalize_market_records(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate source records without cleaning or imputation."""

    source = records.copy()
    if "market_date" not in source and "date" in source:
        source = source.rename(columns={"date": "market_date"})
    missing = [column for column in CORE_MARKET_COLUMNS if column not in source]
    if missing:
        raise NativeMarketPipelineError(
            "Canonical input is missing required columns: " + ", ".join(missing)
        )
    normalized = source.loc[:, list(CORE_MARKET_COLUMNS)].copy()
    normalized["market_date"] = pd.to_datetime(
        normalized["market_date"], errors="coerce"
    ).dt.normalize()
    normalized["symbol"] = (
        normalized["symbol"].astype("string").str.strip().str.upper()
    )
    for column in FLOAT_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(
            "float64"
        )
    numeric_volume = pd.to_numeric(normalized["volume"], errors="coerce")
    if numeric_volume.isna().any() or not np.isfinite(numeric_volume).all():
        raise NativeMarketPipelineError("volume contains missing or non-finite values")
    if not np.equal(numeric_volume, np.floor(numeric_volume)).all():
        raise NativeMarketPipelineError("volume contains non-integral values")
    normalized["volume"] = numeric_volume.astype("int64")

    if normalized["market_date"].isna().any():
        raise NativeMarketPipelineError("market_date contains missing or invalid values")
    if normalized["symbol"].isna().any() or (normalized["symbol"] == "").any():
        raise NativeMarketPipelineError("symbol contains missing or blank values")
    numeric_values = normalized.loc[:, list(FLOAT_COLUMNS)].to_numpy(dtype="float64")
    if not np.isfinite(numeric_values).all():
        raise NativeMarketPipelineError("market prices contain missing or non-finite values")
    if (normalized["volume"] < 0).any():
        raise NativeMarketPipelineError("volume contains negative values")
    if (normalized["close"] <= 0).any():
        raise NativeMarketPipelineError("close must remain positive")

    duplicate_mask = normalized.duplicated(list(BUSINESS_KEY), keep=False)
    if duplicate_mask.any():
        examples = (
            normalized.loc[duplicate_mask, list(BUSINESS_KEY)]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .agg("/".join, axis=1)
            .tolist()
        )
        raise DuplicateMarketRecordError(
            "Duplicate (market_date, symbol) source records: " + ", ".join(examples)
        )
    return normalized.sort_values(
        list(BUSINESS_KEY), kind="mergesort"
    ).reset_index(drop=True)


def load_and_normalize_sources(
    paths: Sequence[Path],
) -> tuple[pd.DataFrame, tuple[SourceEvidence, ...], int]:
    """Read a source set once and reject cross-file key collisions."""

    ordered_paths = tuple(sorted({Path(path) for path in paths}, key=lambda path: str(path)))
    if not ordered_paths:
        raise NativeMarketPipelineError("At least one source CSV is required")
    frames = [_read_source_csv(path) for path in ordered_paths]
    rows_read = sum(len(frame) for frame in frames)
    combined = pd.concat(frames, ignore_index=True)
    normalized = normalize_market_records(combined)
    evidence = tuple(source_evidence(path) for path in ordered_paths)
    return normalized, evidence, rows_read


def _clean_optional_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.eq(""))


def enrich_current_sector(
    normalized: pd.DataFrame,
    listings: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Attach current authoritative sector context to common equities only."""

    required = {"symbol", "security_type", "sector", "source", "snapshot_date"}
    missing = sorted(required.difference(listings.columns))
    if missing:
        raise NativeMarketPipelineError(
            "Listing snapshot is missing sector provenance columns: " + ", ".join(missing)
        )
    evidence = listings.copy()
    evidence["symbol"] = evidence["symbol"].astype("string").str.strip().str.upper()
    evidence["security_type"] = (
        evidence["security_type"].astype("string").str.strip().str.lower()
    )
    normalized_sector = evidence["sector"].astype("string").str.strip().str.upper()
    excluded_sector = (
        normalized_sector.str.contains("FUND", na=False)
        | normalized_sector.str.contains("REAL ESTATE INVESTMENT TRUST", na=False)
        | normalized_sector.str.contains("REIT", na=False)
        | normalized_sector.eq("MODARABAS")
    )
    evidence = evidence.loc[
        evidence["security_type"].isin(ALLOWED_SECTOR_SECURITY_TYPES)
        & ~excluded_sector,
        ["symbol", "sector", "source", "snapshot_date"],
    ].copy()
    if evidence["symbol"].duplicated().any():
        raise NativeMarketPipelineError(
            "Authoritative common-equity listing evidence contains duplicate symbols"
        )
    evidence = evidence.rename(
        columns={
            "sector": "sector_current",
            "source": "sector_source",
            "snapshot_date": "sector_snapshot_date",
        }
    )
    for column in SECTOR_PROVENANCE_COLUMNS:
        evidence[column] = _clean_optional_text(evidence[column])
    enriched = normalized.merge(
        evidence,
        on="symbol",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    enriched = enriched.loc[:, list(CANONICAL_MARKET_COLUMNS)].sort_values(
        list(BUSINESS_KEY), kind="mergesort"
    ).reset_index(drop=True)
    for column in SECTOR_PROVENANCE_COLUMNS:
        enriched[column] = enriched[column].astype("string")
    matched = int(
        enriched.loc[enriched["sector_current"].notna(), "symbol"].nunique()
    )
    return enriched, matched


def _arrow_metadata(*, source_hash: str, content_hash: str) -> dict[bytes, bytes]:
    values = {
        "schema_version": NATIVE_MARKET_SCHEMA_VERSION,
        "pipeline_version": NATIVE_MARKET_PIPELINE_VERSION,
        "sector_context": SECTOR_CONTEXT,
        "source_set_hash": source_hash,
        "canonical_content_hash": content_hash,
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def _to_arrow_table(
    records: pd.DataFrame, *, source_hash: str, content_hash: str
) -> pa.Table:
    serializable = records.loc[:, list(CANONICAL_MARKET_COLUMNS)].copy()
    serializable["market_date"] = pd.to_datetime(serializable["market_date"]).dt.date
    for column in SECTOR_PROVENANCE_COLUMNS:
        serializable[column] = serializable[column].astype(object).where(
            serializable[column].notna(), None
        )
    table = pa.Table.from_pandas(
        serializable,
        schema=CANONICAL_ARROW_SCHEMA,
        preserve_index=False,
        safe=True,
    )
    return table.replace_schema_metadata(
        _arrow_metadata(source_hash=source_hash, content_hash=content_hash)
    )


def _write_csv(records: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = records.loc[:, list(CANONICAL_MARKET_COLUMNS)].copy()
    output["market_date"] = pd.to_datetime(output["market_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    # Seventeen significant digits preserve an IEEE-754 float64 value across a
    # CSV write/read round trip, keeping the canonical CSV value-equivalent to
    # its Parquet representation.
    output.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_parquet(
    records: pd.DataFrame,
    path: Path,
    *,
    source_hash: str,
    content_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _to_arrow_table(
        records, source_hash=source_hash, content_hash=content_hash
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=["symbol", *SECTOR_PROVENANCE_COLUMNS],
        row_group_size=100_000,
    )


def _symbol_file_name(symbol: str) -> str:
    return quote(str(symbol), safe="-_.") + ".csv"


NativeProgressCallback = Callable[[str, str], None]


def _emit_native_progress(
    callback: NativeProgressCallback | None, stage: str, message: str
) -> None:
    if callback is not None:
        callback(stage, message)


def _clone_generated_directory(source: Path, destination: Path) -> None:
    """Stage unchanged files efficiently without mutating their hard links."""

    if not source.is_dir():
        raise NativeMarketPipelineError(
            f"Existing generated artifact directory is missing: {source}"
        )

    def link_or_copy(source_name: str, destination_name: str) -> str:
        try:
            os.link(source_name, destination_name)
        except OSError:
            shutil.copy2(source_name, destination_name)
        return destination_name

    shutil.copytree(source, destination, copy_function=link_or_copy)


def _unlink_staged_file(path: Path) -> None:
    # A staged unchanged file may be a hard link to the accepted artifact.
    # Unlink before rewriting so the existing bundle's inode remains untouched.
    if path.exists() or path.is_symlink():
        path.unlink()


def _write_generated_bundle(
    records: pd.DataFrame,
    staging_root: Path,
    *,
    sources: Sequence[SourceEvidence],
    operation: str,
    rows_read: int,
    operation_rows_accepted: int,
    duplicate_count: int,
    sector_matched_symbols: int,
    rows_added: int,
    rows_replaced: int = 0,
    base_paths: NativeMarketPaths | None = None,
    affected_symbols: Sequence[str] | None = None,
    affected_dates: Sequence[date] | None = None,
    progress_callback: NativeProgressCallback | None = None,
) -> tuple[NativeMarketPaths, dict[str, object]]:
    stage_paths = NativeMarketPaths(
        master_csv=staging_root / "master" / "psx_master.csv",
        symbol_csv_dir=staging_root / "market_symbols",
        daily_parquet_dir=staging_root / "parquet" / "daily",
        consolidated_parquet=staging_root / "parquet" / "market.parquet",
        state_json=staging_root / "state" / "native_market_pipeline_state.json",
    )
    source_hash = source_set_hash(sources)
    content_hash = canonical_content_hash(records)
    _emit_native_progress(
        progress_callback,
        "native_market_csv",
        f"Writing native master CSV with {len(records):,} canonical rows",
    )
    _write_csv(records, stage_paths.master_csv)

    selected_symbols = (
        tuple(sorted(set(str(value) for value in affected_symbols)))
        if affected_symbols is not None
        else tuple(sorted(records["symbol"].astype(str).unique()))
    )
    selected_dates = (
        tuple(sorted(set(affected_dates)))
        if affected_dates is not None
        else tuple(sorted(pd.to_datetime(records["market_date"]).dt.date.unique()))
    )
    if base_paths is not None:
        _clone_generated_directory(
            base_paths.symbol_csv_dir, stage_paths.symbol_csv_dir
        )
        _clone_generated_directory(
            base_paths.daily_parquet_dir, stage_paths.daily_parquet_dir
        )
    _emit_native_progress(
        progress_callback,
        "native_symbol_csvs",
        f"Updating {len(selected_symbols):,} affected per-symbol CSVs",
    )
    symbol_groups = records.loc[
        records["symbol"].astype(str).isin(selected_symbols)
    ].groupby("symbol", sort=True, observed=True)
    for symbol, group in symbol_groups:
        output_path = stage_paths.symbol_csv_dir / _symbol_file_name(str(symbol))
        _unlink_staged_file(output_path)
        _write_csv(group.reset_index(drop=True), output_path)
    _emit_native_progress(
        progress_callback,
        "native_daily_parquets",
        f"Writing {len(selected_dates):,} affected daily Parquet files",
    )
    date_mask = pd.to_datetime(records["market_date"]).dt.date.isin(selected_dates)
    date_groups = records.loc[date_mask].groupby(
        "market_date", sort=True, observed=True
    )
    for market_date, group in date_groups:
        date_text = pd.Timestamp(market_date).date().isoformat()
        daily_content_hash = canonical_content_hash(group.reset_index(drop=True))
        output_path = stage_paths.daily_parquet_dir / f"market_{date_text}.parquet"
        _unlink_staged_file(output_path)
        _write_parquet(
            group.reset_index(drop=True),
            output_path,
            source_hash=source_hash,
            content_hash=daily_content_hash,
        )
    _emit_native_progress(
        progress_callback,
        "native_consolidated_parquet",
        "Atomically staging the consolidated market Parquet",
    )
    _write_parquet(
        records,
        stage_paths.consolidated_parquet,
        source_hash=source_hash,
        content_hash=content_hash,
    )
    dates = pd.to_datetime(records["market_date"])
    state: dict[str, object] = {
        "pipeline_version": NATIVE_MARKET_PIPELINE_VERSION,
        "schema_version": NATIVE_MARKET_SCHEMA_VERSION,
        "sector_context": SECTOR_CONTEXT,
        "operation": operation,
        "source_dates": sorted(dates.dt.date.astype(str).unique().tolist()),
        "source_files": [
            asdict(item) for item in sorted(sources, key=lambda item: item.name)
        ],
        "source_set_hash": source_hash,
        "rows_read": int(rows_read),
        "rows_accepted": int(operation_rows_accepted),
        "rows_rejected": 0,
        "duplicate_count": int(duplicate_count),
        "rows_added": int(rows_added),
        "rows_replaced": int(rows_replaced),
        "daily_parquets_written": len(selected_dates),
        "symbol_csvs_written": len(selected_symbols),
        "sector_snapshot_dates": sorted(
            records["sector_snapshot_date"].dropna().astype(str).unique().tolist()
        ),
        "sector_matched_symbols": int(sector_matched_symbols),
        "master_csv_row_count": int(len(records)),
        "consolidated_parquet_row_count": int(len(records)),
        "earliest_date": dates.min().date().isoformat() if not dates.empty else None,
        "latest_date": dates.max().date().isoformat() if not dates.empty else None,
        "symbol_count": int(records["symbol"].nunique()),
        "canonical_content_hash": content_hash,
        "consolidated_sha256": sha256_file(stage_paths.consolidated_parquet),
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failure_reason": None,
    }
    stage_paths.state_json.parent.mkdir(parents=True, exist_ok=True)
    stage_paths.state_json.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _validate_generated_bundle(stage_paths, expected=records)
    return stage_paths, state


def _validate_generated_bundle(
    paths: NativeMarketPaths, *, expected: pd.DataFrame
) -> None:
    """Fail before promotion if any staged artifact violates the contract."""

    parquet_file = pq.ParquetFile(paths.consolidated_parquet)
    if parquet_file.schema_arrow.remove_metadata() != CANONICAL_ARROW_SCHEMA:
        raise NativeMarketPipelineError("Staged consolidated Parquet schema is invalid")
    actual = pq.read_table(paths.consolidated_parquet).to_pandas()
    actual["market_date"] = pd.to_datetime(actual["market_date"])
    for column in SECTOR_PROVENANCE_COLUMNS:
        actual[column] = actual[column].astype("string")
    actual = actual.loc[:, list(CANONICAL_MARKET_COLUMNS)]
    if len(actual) != len(expected):
        raise NativeMarketPipelineError("Staged consolidated Parquet row count changed")
    if actual.duplicated(list(BUSINESS_KEY)).any():
        raise NativeMarketPipelineError("Staged consolidated Parquet contains duplicate keys")
    ordered = actual.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)
    if not actual.reset_index(drop=True).equals(ordered):
        raise NativeMarketPipelineError("Staged consolidated Parquet is not deterministically sorted")
    if canonical_content_hash(actual) != canonical_content_hash(expected):
        raise NativeMarketPipelineError("Staged consolidated Parquet values changed")
    master = pd.read_csv(paths.master_csv, dtype={"symbol": "string"})
    if len(master) != len(expected) or tuple(master.columns) != CANONICAL_MARKET_COLUMNS:
        raise NativeMarketPipelineError("Staged master CSV failed schema/row validation")
    if len(tuple(paths.symbol_csv_dir.glob("*.csv"))) != expected["symbol"].nunique():
        raise NativeMarketPipelineError("Per-symbol CSV generation silently lost symbols")
    if len(tuple(paths.daily_parquet_dir.glob("market_*.parquet"))) != expected[
        "market_date"
    ].nunique():
        raise NativeMarketPipelineError("Daily Parquet generation silently lost dates")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _promote_bundle(
    staged: NativeMarketPaths,
    destinations: NativeMarketPaths,
    *,
    before_promote: Callable[[Path, Path], None] | None = None,
) -> None:
    """Promote all outputs transactionally, restoring old outputs on failure."""

    pairs = (
        (staged.master_csv, destinations.master_csv),
        (staged.symbol_csv_dir, destinations.symbol_csv_dir),
        (staged.daily_parquet_dir, destinations.daily_parquet_dir),
        (staged.consolidated_parquet, destinations.consolidated_parquet),
        (staged.state_json, destinations.state_json),
    )
    transaction_parent = destinations.consolidated_parquet.parent.parent
    transaction_parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(prefix=".native-market-backup-", dir=transaction_parent)
    )
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for index, (source, target) in enumerate(pairs):
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = backup_root / str(index)
            if target.exists() or target.is_symlink():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backups[target] = backup
            if before_promote is not None:
                before_promote(source, target)
            os.replace(source, target)
            promoted.append(target)
    except Exception:
        for target in reversed(promoted):
            _remove_path(target)
        for target, backup in reversed(tuple(backups.items())):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def compare_parquet_core(
    native_path: Path,
    bootstrap_path: Path,
) -> MigrationComparison:
    """Compare exact normalized core values without touching either source."""

    errors: list[str] = []
    frames: list[pd.DataFrame] = []
    for label, path in (("native", native_path), ("bootstrap", bootstrap_path)):
        if not Path(path).is_file():
            raise NativeMarketPipelineError(f"{label} Parquet does not exist: {path}")
        table = pq.read_table(path, columns=list(CORE_MARKET_COLUMNS))
        frame = table.to_pandas().loc[:, list(CORE_MARKET_COLUMNS)]
        frame["market_date"] = pd.to_datetime(frame["market_date"])
        frame["symbol"] = frame["symbol"].astype("string")
        frame = frame.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)
        frames.append(frame)
    native, bootstrap = frames
    native_duplicates = int(native.duplicated(list(BUSINESS_KEY)).sum())
    bootstrap_duplicates = int(bootstrap.duplicated(list(BUSINESS_KEY)).sum())
    if len(native) != len(bootstrap):
        errors.append(f"row count differs: native={len(native)}, bootstrap={len(bootstrap)}")
    if native_duplicates or bootstrap_duplicates:
        errors.append(
            f"duplicate keys differ/are present: native={native_duplicates}, bootstrap={bootstrap_duplicates}"
        )
    native_hash = canonical_content_hash(native, core_only=True)
    bootstrap_hash = canonical_content_hash(bootstrap, core_only=True)
    if native_hash != bootstrap_hash:
        errors.append("canonical core value hash differs")
    return MigrationComparison(
        passed=not errors,
        native_rows=len(native),
        bootstrap_rows=len(bootstrap),
        native_symbols=int(native["symbol"].nunique()),
        bootstrap_symbols=int(bootstrap["symbol"].nunique()),
        native_dates=int(native["market_date"].nunique()),
        bootstrap_dates=int(bootstrap["market_date"].nunique()),
        native_duplicates=native_duplicates,
        bootstrap_duplicates=bootstrap_duplicates,
        native_core_hash=native_hash,
        bootstrap_core_hash=bootstrap_hash,
        errors=tuple(errors),
    )


def _result_from_state(
    state: Mapping[str, object],
    *,
    paths: NativeMarketPaths,
    comparison: MigrationComparison | None,
    idempotent_noop: bool = False,
) -> NativeMarketBuildResult:
    return NativeMarketBuildResult(
        operation=str(state["operation"]),
        source_files=len(state.get("source_files", [])),
        source_dates=tuple(str(value) for value in state.get("source_dates", [])),
        rows_read=int(state["rows_read"]),
        rows_accepted=int(state["rows_accepted"]),
        rows_rejected=int(state["rows_rejected"]),
        duplicate_count=int(state["duplicate_count"]),
        master_rows=int(state["master_csv_row_count"]),
        consolidated_rows=int(state["consolidated_parquet_row_count"]),
        symbol_count=int(state["symbol_count"]),
        earliest_date=state.get("earliest_date") and str(state["earliest_date"]),
        latest_date=state.get("latest_date") and str(state["latest_date"]),
        sector_matched_symbols=int(state["sector_matched_symbols"]),
        schema_version=str(state["schema_version"]),
        content_hash=str(state["canonical_content_hash"]),
        source_set_hash=str(state["source_set_hash"]),
        consolidated_sha256=str(state["consolidated_sha256"]),
        status=str(state["status"]),
        paths=paths,
        rows_added=int(state.get("rows_added", state.get("rows_accepted", 0))),
        rows_replaced=int(state.get("rows_replaced", 0)),
        daily_parquets_written=int(state.get("daily_parquets_written", 0)),
        symbol_csvs_written=int(state.get("symbol_csvs_written", 0)),
        migration_comparison=comparison,
        idempotent_noop=idempotent_noop,
    )


def _full_rebuild_impl(
    *,
    source_csv_dir: Path = RAW_CSV_DIR,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    compare_bootstrap_path: Path | None = None,
    before_promote: Callable[[Path, Path], None] | None = None,
) -> NativeMarketBuildResult:
    """Build a complete staged bundle and atomically promote only after validation."""

    source_paths = _source_csv_paths(source_csv_dir)
    normalized, sources, rows_read = load_and_normalize_sources(source_paths)
    if paths.consolidated_parquet.is_file():
        _assert_full_source_does_not_regress(
            _existing_canonical_records(paths.consolidated_parquet), normalized
        )
    listings = load_listing_snapshot(Path(listings_path))
    enriched, matched = enrich_current_sector(normalized, listings)
    staging_parent = paths.consolidated_parquet.parent.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".native-market-stage-", dir=staging_parent)
    )
    comparison: MigrationComparison | None = None
    try:
        staged, state = _write_generated_bundle(
            enriched,
            staging_root,
            sources=sources,
            operation="full_rebuild",
            rows_read=rows_read,
            operation_rows_accepted=len(enriched),
            duplicate_count=0,
            sector_matched_symbols=matched,
            rows_added=len(enriched),
        )
        if compare_bootstrap_path is not None:
            comparison = compare_parquet_core(
                staged.consolidated_parquet, Path(compare_bootstrap_path)
            )
            if not comparison.passed:
                raise MigrationComparisonError("; ".join(comparison.errors))
        _promote_bundle(staged, paths, before_promote=before_promote)
        return _result_from_state(state, paths=paths, comparison=comparison)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _write_failure_audit(
    *,
    paths: NativeMarketPaths,
    operation: str,
    source_paths: Sequence[Path],
    error: BaseException,
) -> None:
    """Record failure separately without changing the last valid bundle state."""

    evidence: list[dict[str, object]] = []
    for path in sorted((Path(value) for value in source_paths), key=lambda value: str(value)):
        try:
            evidence.append(asdict(source_evidence(path)))
        except NativeMarketPipelineError:
            evidence.append({"name": _portable_source_name(path), "sha256": None, "size_bytes": None})
    payload = {
        "pipeline_version": NATIVE_MARKET_PIPELINE_VERSION,
        "schema_version": NATIVE_MARKET_SCHEMA_VERSION,
        "operation": operation,
        "source_files": evidence,
        "status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failure_reason": f"{type(error).__name__}: {error}",
    }
    failure_path = paths.state_json.with_name("native_market_pipeline_last_failure.json")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=failure_path.parent,
        prefix=f".{failure_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, failure_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def full_rebuild(
    *,
    source_csv_dir: Path = RAW_CSV_DIR,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    compare_bootstrap_path: Path | None = None,
    before_promote: Callable[[Path, Path], None] | None = None,
) -> NativeMarketBuildResult:
    """Run a full rebuild and retain a separate atomic failure audit."""

    try:
        return _full_rebuild_impl(
            source_csv_dir=source_csv_dir,
            listings_path=listings_path,
            paths=paths,
            compare_bootstrap_path=compare_bootstrap_path,
            before_promote=before_promote,
        )
    except Exception as exc:
        _write_failure_audit(
            paths=paths,
            operation="full_rebuild",
            source_paths=tuple(sorted(Path(source_csv_dir).glob("market_*.csv"))),
            error=exc,
        )
        raise


def _load_existing_state(path: Path) -> dict[str, object]:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeMarketPipelineError(f"Could not load native pipeline state: {exc}") from exc
    if state.get("schema_version") != NATIVE_MARKET_SCHEMA_VERSION:
        raise NativeMarketPipelineError("Existing native pipeline state has an incompatible schema version")
    return state


def _state_source_evidence(state: Mapping[str, object]) -> tuple[SourceEvidence, ...]:
    """Load source provenance while migrating pre-origin v1 state in memory."""

    evidence: list[SourceEvidence] = []
    for item in state.get("source_files", []):
        if not isinstance(item, Mapping):
            raise NativeMarketPipelineError("Native source manifest entry is invalid")
        name = str(item["name"])
        evidence.append(
            SourceEvidence(
                name=name,
                sha256=str(item["sha256"]),
                size_bytes=int(item["size_bytes"]),
                origin=str(
                    item.get(
                        "origin",
                        (
                            "virtual_trader_raw_csv"
                            if name.startswith("data/raw/csv/")
                            else "external_validated_csv"
                        ),
                    )
                ),
            )
        )
    return tuple(sorted(evidence, key=lambda value: value.name))


def _existing_canonical_records(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(CANONICAL_MARKET_COLUMNS))
    records = table.to_pandas()
    records["market_date"] = pd.to_datetime(records["market_date"])
    records["symbol"] = records["symbol"].astype("string")
    for column in FLOAT_COLUMNS:
        records[column] = records[column].astype("float64")
    records["volume"] = records["volume"].astype("int64")
    for column in SECTOR_PROVENANCE_COLUMNS:
        records[column] = records[column].astype("string")
    return records.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)


def _rebuild_generated_artifacts_impl(
    *,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    before_promote: Callable[[Path, Path], None] | None = None,
    progress_callback: NativeProgressCallback | None = None,
) -> NativeMarketBuildResult:
    """Regenerate native outputs from the validated canonical Parquet/state pair.

    This is an explicit repair operation.  It preserves the source manifest and
    never substitutes a local CSV for an external/bootstrap source implicitly.
    """

    if not paths.consolidated_parquet.is_file() or not paths.state_json.is_file():
        raise NativeMarketPipelineError(
            "Artifact rebuild requires an existing native Parquet and state"
        )
    state = _load_existing_state(paths.state_json)
    metadata = pq.ParquetFile(paths.consolidated_parquet).schema_arrow.metadata or {}
    parquet_source_hash = metadata.get(b"source_set_hash", b"").decode()
    parquet_content_hash = metadata.get(b"canonical_content_hash", b"").decode()
    if (
        parquet_source_hash != str(state.get("source_set_hash") or "")
        or parquet_content_hash != str(state.get("canonical_content_hash") or "")
    ):
        raise NativeMarketPipelineError(
            "Native state and consolidated Parquet provenance are not aligned"
        )
    existing = _existing_canonical_records(paths.consolidated_parquet)
    if canonical_content_hash(existing) != parquet_content_hash:
        raise NativeMarketPipelineError(
            "Consolidated Parquet content does not match its recorded hash"
        )
    listings = load_listing_snapshot(Path(listings_path))
    core = existing.loc[:, list(CORE_MARKET_COLUMNS)]
    enriched, matched = enrich_current_sector(core, listings)
    sources = _state_source_evidence(state)
    staging_parent = paths.consolidated_parquet.parent.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".native-market-stage-", dir=staging_parent)
    )
    try:
        staged, rebuilt_state = _write_generated_bundle(
            enriched,
            staging_root,
            sources=sources,
            operation="artifact_rebuild",
            rows_read=0,
            operation_rows_accepted=0,
            duplicate_count=0,
            sector_matched_symbols=matched,
            rows_added=0,
            progress_callback=progress_callback,
        )
        _promote_bundle(staged, paths, before_promote=before_promote)
        return _result_from_state(
            rebuilt_state, paths=paths, comparison=None
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def rebuild_generated_artifacts(
    *,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    before_promote: Callable[[Path, Path], None] | None = None,
    progress_callback: NativeProgressCallback | None = None,
) -> NativeMarketBuildResult:
    """Run a source-preserving canonical artifact rebuild atomically."""

    try:
        return _rebuild_generated_artifacts_impl(
            listings_path=listings_path,
            paths=paths,
            before_promote=before_promote,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        _write_failure_audit(
            paths=paths,
            operation="artifact_rebuild",
            source_paths=(),
            error=exc,
        )
        raise


def _assert_full_source_does_not_regress(
    existing: pd.DataFrame, rebuilt_core: pd.DataFrame
) -> None:
    """Refuse an implicit deletion/correction policy during a full rebuild."""

    old = existing.loc[:, list(CORE_MARKET_COLUMNS)].set_index(list(BUSINESS_KEY))
    new = rebuilt_core.loc[:, list(CORE_MARKET_COLUMNS)].set_index(list(BUSINESS_KEY))
    missing = old.index.difference(new.index)
    if len(missing):
        examples = [str(value) for value in missing[:5].tolist()]
        raise NativeMarketPipelineError(
            "Full-rebuild source set would remove existing canonical records; "
            "no source-regression policy is configured. Missing keys: "
            + ", ".join(examples)
        )
    comparable_columns = [
        column for column in CORE_MARKET_COLUMNS if column not in BUSINESS_KEY
    ]
    old_values = old.loc[:, comparable_columns].sort_index()
    new_values = new.loc[old_values.index, comparable_columns].sort_index()
    if not old_values.equals(new_values):
        raise NativeMarketPipelineError(
            "Full-rebuild source values conflict with existing canonical records; "
            "no replacement policy is configured"
        )


def _merge_incremental_core(
    existing: pd.DataFrame, incoming: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    existing_core = existing.loc[:, list(CORE_MARKET_COLUMNS)].copy()
    left = existing_core.set_index(list(BUSINESS_KEY), drop=False)
    right = incoming.set_index(list(BUSINESS_KEY), drop=False)
    overlap = left.index.intersection(right.index)
    idempotent_rows = 0
    for key in overlap:
        old = left.loc[key, list(CORE_MARKET_COLUMNS)]
        new = right.loc[key, list(CORE_MARKET_COLUMNS)]
        if isinstance(old, pd.DataFrame) or isinstance(new, pd.DataFrame):
            raise DuplicateMarketRecordError(f"Non-unique existing/incoming key: {key}")
        if not old.equals(new):
            raise DuplicateMarketRecordError(
                f"Incremental source conflicts with existing record: {key}"
            )
        idempotent_rows += 1
    additions = incoming.loc[~incoming.set_index(list(BUSINESS_KEY)).index.isin(overlap)]
    combined = pd.concat([existing_core, additions], ignore_index=True)
    return normalize_market_records(combined), idempotent_rows


def _incremental_update_impl(
    source_csvs: Sequence[Path],
    *,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    before_promote: Callable[[Path, Path], None] | None = None,
    progress_callback: NativeProgressCallback | None = None,
) -> NativeMarketBuildResult:
    """Merge new daily evidence and atomically rewrite the generated bundle.

    Rewriting the 54-MB consolidated Parquet is intentionally preferred over a
    fake append.  Daily partitions remain independently inspectable, while a
    full atomic consolidated rewrite keeps ordering and uniqueness simple.
    """

    if not paths.consolidated_parquet.is_file() or not paths.state_json.is_file():
        raise NativeMarketPipelineError(
            "Incremental update requires an existing validated native bundle"
        )
    incoming, incoming_sources, rows_read = load_and_normalize_sources(source_csvs)
    old_state = _load_existing_state(paths.state_json)
    existing = _existing_canonical_records(paths.consolidated_parquet)
    combined_core, idempotent_rows = _merge_incremental_core(existing, incoming)
    previous_sources = _state_source_evidence(old_state)
    by_name = {item.name: item for item in previous_sources}
    for item in incoming_sources:
        prior = by_name.get(item.name)
        if prior is not None and prior.sha256 != item.sha256:
            raise DuplicateMarketRecordError(
                f"Source replacement is not allowed without an explicit policy: {item.name}"
            )
        by_name[item.name] = item
    listings = load_listing_snapshot(Path(listings_path))
    enriched, matched = enrich_current_sector(combined_core, listings)
    current_snapshot_dates = sorted(
        enriched["sector_snapshot_date"].dropna().astype(str).unique().tolist()
    )
    merged_sources = tuple(sorted(by_name.values(), key=lambda item: item.name))
    merged_source_hash = source_set_hash(merged_sources)
    if (
        idempotent_rows == len(incoming)
        and canonical_content_hash(enriched) == str(old_state["canonical_content_hash"])
        and sorted(old_state.get("sector_snapshot_dates", [])) == current_snapshot_dates
        and merged_source_hash == str(old_state["source_set_hash"])
    ):
        return _result_from_state(
            old_state,
            paths=paths,
            comparison=None,
            idempotent_noop=True,
        )
    staging_parent = paths.consolidated_parquet.parent.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".native-market-stage-", dir=staging_parent)
    )
    try:
        staged, state = _write_generated_bundle(
            enriched,
            staging_root,
            sources=merged_sources,
            operation="incremental",
            rows_read=rows_read,
            operation_rows_accepted=len(incoming) - idempotent_rows,
            duplicate_count=idempotent_rows,
            sector_matched_symbols=matched,
            rows_added=len(incoming) - idempotent_rows,
            base_paths=paths,
            affected_symbols=tuple(incoming["symbol"].astype(str).unique()),
            affected_dates=tuple(pd.to_datetime(incoming["market_date"]).dt.date.unique()),
            progress_callback=progress_callback,
        )
        _promote_bundle(staged, paths, before_promote=before_promote)
        return _result_from_state(state, paths=paths, comparison=None)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def incremental_update(
    source_csvs: Sequence[Path],
    *,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    paths: NativeMarketPaths = NativeMarketPaths(),
    before_promote: Callable[[Path, Path], None] | None = None,
    progress_callback: NativeProgressCallback | None = None,
) -> NativeMarketBuildResult:
    """Run an incremental update and retain a separate atomic failure audit."""

    try:
        return _incremental_update_impl(
            source_csvs,
            listings_path=listings_path,
            paths=paths,
            before_promote=before_promote,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        _write_failure_audit(
            paths=paths,
            operation="incremental",
            source_paths=source_csvs,
            error=exc,
        )
        raise


def _print_result(result: NativeMarketBuildResult) -> None:
    print(f"Operation: {result.operation}")
    print(f"Status: {result.status}")
    print(f"Rows: {result.consolidated_rows:,}")
    print(f"Symbols: {result.symbol_count:,}")
    print(f"Date range: {result.earliest_date} through {result.latest_date}")
    print(f"Sector-matched symbols: {result.sector_matched_symbols:,}")
    print(f"Canonical content hash: {result.content_hash}")
    print(f"Source-set hash: {result.source_set_hash}")
    print(f"Consolidated Parquet: {result.paths.consolidated_parquet}")
    if result.idempotent_noop:
        print("Incremental result: idempotent no-op")
    if result.migration_comparison is not None:
        comparison = result.migration_comparison
        print(f"Bootstrap comparison passed: {comparison.passed}")
        print(f"Bootstrap core hash: {comparison.bootstrap_core_hash}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build virtual-trader's native sector-enriched market artifacts"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-rebuild", action="store_true")
    mode.add_argument("--incremental", nargs="+", metavar="CSV")
    parser.add_argument("--source-dir", type=Path, default=RAW_CSV_DIR)
    parser.add_argument("--listings", type=Path, default=CURRENT_LISTINGS_PATH)
    parser.add_argument(
        "--compare-bootstrap",
        action="store_true",
        help="Require an exact core comparison with the sibling bootstrap Parquet",
    )
    args = parser.parse_args(argv)
    try:
        if args.full_rebuild:
            result = full_rebuild(
                source_csv_dir=args.source_dir,
                listings_path=args.listings,
                compare_bootstrap_path=(
                    BOOTSTRAP_PSX_MARKET_PARQUET_PATH
                    if args.compare_bootstrap
                    else None
                ),
            )
        else:
            result = incremental_update(
                tuple(Path(value) for value in args.incremental),
                listings_path=args.listings,
            )
    except Exception as exc:
        print(f"Native market pipeline failed: {exc}")
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
