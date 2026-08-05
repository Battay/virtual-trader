"""Incremental discovery and collection for missing PSX daily data."""

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
import re

import pandas as pd

from .client import PsxClient
from .config import RAW_CSV_DIR
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


def determine_missing_dates(
    available_dates: tuple[date, ...] | set[date],
    end_date: date,
    bootstrap_start_date: date | None = None,
) -> tuple[date, ...]:
    """Return missing calendar dates through ``end_date`` inclusively."""
    available = set(available_dates)
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
        if current_date not in available:
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
) -> IncrementalUpdateResult:
    """Fetch only locally missing dates and continue after skips or failures."""
    available_dates = discover_available_raw_dates(csv_dir)
    missing_dates = determine_missing_dates(
        available_dates,
        end_date,
        bootstrap_start_date,
    )
    psx_client = client if client is not None else PsxClient()

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
    )
