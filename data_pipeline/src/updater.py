"""Incremental discovery and collection for missing PSX daily data."""

from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
from pathlib import Path
import re
from typing import Callable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .client import PsxClient
from .config import (
    LOCAL_PSX_MARKET_PARQUET_PATH,
    NATIVE_MARKET_PIPELINE_STATE_PATH,
    RAW_CSV_DIR,
)
from .main import (
    CollectionResult,
    DateProcessor,
    MarketClient,
    OUTPUT_FIELDS,
    collect_single_date,
    process_date,
)


LOGGER = logging.getLogger(__name__)
DAILY_CSV_PATTERN = re.compile(r"market_(\d{4}-\d{2}-\d{2})\.csv")
UpdateProgressCallback = Callable[[str, str], None]


class BootstrapStartDateRequired(ValueError):
    """Raised when an empty local store has no explicit bootstrap date."""


@dataclass(frozen=True)
class IncrementalUpdateResult:
    """Structured outcome of one incremental update attempt."""

    requested_end_date: date
    available_dates_before: tuple[date, ...]
    latest_stored_date: date | None
    missing_dates: tuple[date, ...]
    collection: CollectionResult
    local_source_dates_before: tuple[date, ...] = ()
    external_source_dates_before: tuple[date, ...] = ()
    excluded_non_request_dates: tuple[date, ...] = ()
    source_evidence_inconsistencies: tuple[str, ...] = ()

    @property
    def successful_dates(self) -> tuple[date, ...]:
        return self.collection.successful_dates

    @property
    def skipped_dates(self) -> tuple[date, ...]:
        return self.collection.skipped_dates

    @property
    def failed_dates(self) -> tuple[tuple[date, str], ...]:
        return self.collection.failed_dates

    @property
    def output_csv_paths(self) -> tuple[Path, ...]:
        return self.collection.output_csv_paths

    @property
    def total_processed(self) -> int:
        return self.collection.total_processed


@dataclass(frozen=True)
class SourceEvidenceInventory:
    """Date-level source provenance independent from analytical Parquet rows."""

    local_csv_dates: tuple[date, ...]
    native_manifest_dates: tuple[date, ...]
    external_manifest_dates: tuple[date, ...]
    parquet_dates: tuple[date, ...]
    parquet_only_dates: tuple[date, ...]
    inconsistencies: tuple[str, ...]
    local_weekend_dates: tuple[date, ...] = ()

    @property
    def accepted_source_dates(self) -> tuple[date, ...]:
        # A legacy automation bug could create a date-stamped CSV for a weekend
        # when the endpoint returned a non-empty table.  A filename plus injected
        # date is not sufficient evidence that a weekend was a PSX trading date.
        return tuple(
            sorted(
                value
                for value in set(self.local_csv_dates).union(
                    self.native_manifest_dates
                )
                if value.weekday() < 5
            )
        )


def _date_from_daily_filename(path: Path) -> date | None:
    match = DAILY_CSV_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def is_valid_daily_csv(path: Path, expected_date: date) -> bool:
    """Return whether a daily CSV is non-empty, complete, and date-matching."""
    return valid_daily_csv_row_count(path, expected_date) is not None


def valid_daily_csv_row_count(path: Path, expected_date: date) -> int | None:
    """Return a valid daily CSV's equity row count, or ``None`` if invalid."""
    try:
        frame = pd.read_csv(path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return None

    if frame.empty or not set(OUTPUT_FIELDS).issubset(frame.columns):
        return None
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = bool(
        parsed_dates.notna().all()
        and (parsed_dates.dt.date == expected_date).all()
    )
    return len(frame) if valid else None


def discover_available_raw_dates(csv_dir: Path = RAW_CSV_DIR) -> tuple[date, ...]:
    """Discover dates represented by valid generated daily CSV files."""
    directory = Path(csv_dir)
    if not directory.is_dir():
        return ()

    available: list[date] = []
    for path in sorted(directory.glob("market_*.csv")):
        trading_date = _date_from_daily_filename(path)
        if trading_date is None:
            LOGGER.warning("Ignoring unrecognized daily CSV filename: %s", path)
            continue
        if not is_valid_daily_csv(path, trading_date):
            LOGGER.warning("Ignoring invalid daily CSV: %s", path)
            continue
        available.append(trading_date)
    return tuple(sorted(set(available)))


def _date_from_source_name(value: object) -> date | None:
    return _date_from_daily_filename(Path(str(value)))


def discover_source_evidence(
    *,
    csv_dir: Path = RAW_CSV_DIR,
    native_state_path: Path = NATIVE_MARKET_PIPELINE_STATE_PATH,
    parquet_path: Path = LOCAL_PSX_MARKET_PARQUET_PATH,
) -> SourceEvidenceInventory:
    """Inventory CSV evidence while keeping Parquet-only dates diagnostic.

    A native manifest date counts as evidence because it records the originating
    CSV name/hash. A date found only in Parquet does not count as source evidence.
    """

    local = discover_available_raw_dates(csv_dir)
    manifest_dates: set[date] = set()
    external_dates: set[date] = set()
    state_source_hash: str | None = None
    state_content_hash: str | None = None
    state_path = Path(native_state_path)
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_source_hash = str(state.get("source_set_hash") or "") or None
            state_content_hash = str(state.get("canonical_content_hash") or "") or None
            for item in state.get("source_files", []):
                if not isinstance(item, dict):
                    continue
                source_date = _date_from_source_name(item.get("name"))
                if source_date is None:
                    continue
                manifest_dates.add(source_date)
                origin = str(item.get("origin") or "")
                name = str(item.get("name") or "")
                if origin == "external_validated_csv" or (
                    not origin and not name.startswith("data/raw/csv/")
                ):
                    external_dates.add(source_date)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid native source manifest %s: %s", state_path, exc)

    parquet_dates: set[date] = set()
    parquet_trusted = False
    parquet_file_path = Path(parquet_path)
    if parquet_file_path.is_file():
        try:
            parquet_file = pq.ParquetFile(parquet_file_path)
            metadata = parquet_file.schema_arrow.metadata or {}
            parquet_source_hash = metadata.get(b"source_set_hash", b"").decode()
            parquet_content_hash = metadata.get(b"canonical_content_hash", b"").decode()
            parquet_trusted = bool(
                state_source_hash
                and state_content_hash
                and parquet_source_hash == state_source_hash
                and parquet_content_hash == state_content_hash
            )
            date_values = pq.read_table(
                parquet_file_path, columns=["market_date"]
            ).column("market_date").unique().to_pylist()
            parquet_dates = {value for value in date_values if isinstance(value, date)}
        except (OSError, ValueError, pa.ArrowException) as exc:
            LOGGER.warning("Could not inspect native Parquet source dates: %s", exc)

    inconsistencies: list[str] = []
    local_weekends = {value for value in local if value.weekday() >= 5}
    if local_weekends:
        inconsistencies.append(
            f"{len(local_weekends)} local CSV date(s) fall on weekends and are not eligible for native ingestion"
        )
    if manifest_dates and not parquet_trusted:
        inconsistencies.append(
            "Native source manifest and consolidated Parquet hashes are not aligned"
        )
        # An untrusted manifest cannot suppress a source request.
        manifest_dates.clear()
        external_dates.clear()
    parquet_only = parquet_dates.difference(manifest_dates).difference(local)
    if parquet_only:
        inconsistencies.append(
            f"{len(parquet_only)} Parquet date(s) have no recorded CSV source evidence"
        )
    external_only = external_dates.difference(local)
    if external_only:
        inconsistencies.append(
            f"{len(external_only)} native date(s) use external/bootstrap CSV evidence rather than the local raw store"
        )
    elif external_dates:
        inconsistencies.append(
            f"{len(external_dates)} native date(s) retain external/bootstrap CSV provenance; local counterpart files also exist"
        )
    local_not_native = set(local).difference(parquet_dates)
    pending_weekdays = local_not_native.difference(local_weekends)
    if pending_weekdays:
        inconsistencies.append(
            f"{len(pending_weekdays)} eligible local weekday CSV date(s) are not yet represented in native Parquet"
        )
    return SourceEvidenceInventory(
        local_csv_dates=local,
        native_manifest_dates=tuple(sorted(manifest_dates)),
        external_manifest_dates=tuple(sorted(external_dates)),
        parquet_dates=tuple(sorted(parquet_dates)),
        parquet_only_dates=tuple(sorted(parquet_only)),
        inconsistencies=tuple(inconsistencies),
        local_weekend_dates=tuple(sorted(local_weekends)),
    )


def determine_missing_dates(
    available_dates: tuple[date, ...] | set[date],
    end_date: date,
    bootstrap_start_date: date | None = None,
    *,
    excluded_dates: Sequence[date] = (),
    include_weekends: bool = False,
) -> tuple[date, ...]:
    """Return unresolved request dates through ``end_date`` inclusively."""
    available = set(available_dates)
    excluded = set(excluded_dates)
    if bootstrap_start_date is None:
        eligible_available = [value for value in available if value <= end_date]
        if not eligible_available:
            raise BootstrapStartDateRequired(
                "No valid local CSV exists; provide a bootstrap start date"
            )
        start_date = min(eligible_available)
    else:
        start_date = bootstrap_start_date

    if end_date < start_date:
        raise ValueError("end date cannot be earlier than bootstrap start date")

    missing: list[date] = []
    current_date = start_date
    while current_date <= end_date:
        if (
            current_date not in available
            and current_date not in excluded
            and (include_weekends or current_date.weekday() < 5)
        ):
            missing.append(current_date)
        current_date += timedelta(days=1)
    return tuple(missing)


def run_incremental_update(
    end_date: date,
    *,
    bootstrap_start_date: date | None = None,
    csv_dir: Path = RAW_CSV_DIR,
    client: MarketClient | None = None,
    date_processor: DateProcessor = process_date,
    available_source_dates: Sequence[date] | None = None,
    local_source_dates: Sequence[date] | None = None,
    external_source_dates: Sequence[date] = (),
    excluded_dates: Sequence[date] = (),
    source_evidence_inconsistencies: Sequence[str] = (),
    progress_callback: UpdateProgressCallback | None = None,
) -> IncrementalUpdateResult:
    """Fetch only locally missing dates and continue after skips or failures."""
    discovered_local = (
        tuple(local_source_dates)
        if local_source_dates is not None
        else discover_available_raw_dates(csv_dir)
    )
    available_dates = (
        tuple(sorted(set(available_source_dates)))
        if available_source_dates is not None
        else discovered_local
    )
    missing_dates = determine_missing_dates(
        available_dates,
        end_date,
        bootstrap_start_date,
        excluded_dates=excluded_dates,
    )
    if progress_callback is not None:
        progress_callback(
            "fetching",
            f"Fetching {len(missing_dates):,} unresolved source date(s)",
        )
    psx_client = (client if client is not None else PsxClient()) if missing_dates else None

    successful_dates: list[date] = []
    skipped_dates: list[date] = []
    failed_dates: list[tuple[date, str]] = []
    output_paths: list[Path] = []

    for missing_date in missing_dates:
        result = collect_single_date(
            missing_date,
            client=psx_client,
            date_processor=date_processor,
        )
        successful_dates.extend(result.successful_dates)
        skipped_dates.extend(result.skipped_dates)
        failed_dates.extend(result.failed_dates)
        output_paths.extend(result.output_csv_paths)

    collection_start = missing_dates[0] if missing_dates else end_date
    collection_end = missing_dates[-1] if missing_dates else end_date
    collection = CollectionResult(
        start_date=collection_start,
        end_date=collection_end,
        total_processed=len(missing_dates),
        successful_dates=tuple(successful_dates),
        skipped_dates=tuple(skipped_dates),
        failed_dates=tuple(failed_dates),
        output_csv_paths=tuple(output_paths),
    )
    stored_after = set(available_dates).union(successful_dates)
    return IncrementalUpdateResult(
        requested_end_date=end_date,
        available_dates_before=available_dates,
        latest_stored_date=max(stored_after) if stored_after else None,
        missing_dates=missing_dates,
        collection=collection,
        local_source_dates_before=tuple(discovered_local),
        external_source_dates_before=tuple(sorted(set(external_source_dates))),
        excluded_non_request_dates=tuple(sorted(set(excluded_dates))),
        source_evidence_inconsistencies=tuple(source_evidence_inconsistencies),
    )
