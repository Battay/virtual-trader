"""Sequential, resumable historical backfill for PSX daily market data."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Literal

from . import main as main_pipeline
from .client import PsxClient
from .config import BACKFILL_STATE_PATH, RAW_CSV_DIR
from .main import (
    CollectionResult,
    DateProcessor,
    MarketClient,
    collect_single_date,
    iter_calendar_dates,
    process_date,
)
from .updater import (
    discover_available_raw_dates,
    is_valid_daily_csv,
    valid_daily_csv_row_count,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_EMPTY_RETRY_DELAYS = (3.0, 8.0, 20.0)
DEFAULT_EMPTY_MAX_ATTEMPTS = len(DEFAULT_EMPTY_RETRY_DELAYS) + 1
DEFAULT_CIRCUIT_WINDOW = 10
DEFAULT_CIRCUIT_EMPTY_RATIO = 0.70
CIRCUIT_BREAKER_MESSAGE = (
    "Backfill paused because the PSX source is returning unusually many empty "
    "weekday responses. Existing successful files are safe."
)

BackfillOutcome = Literal[
    "successful",
    "non_trading",
    "temporary_unavailable",
    "failed",
    "already_downloaded",
]


class BackfillStateError(ValueError):
    """Raised when saved state cannot be used for the requested backfill."""


@dataclass(frozen=True)
class BackfillSuccessRecord:
    """Auditable details for a date confirmed to contain valid equity rows."""

    trading_date: date
    valid_rows: int
    output_path: str
    parsed_rows: int | None = None
    rejected_rows: int | None = None
    message: str = ""
    reconciled: bool = False
    attempt_count: int = 0
    response_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class BackfillAttemptRecord:
    """Compact audit record for one HTTP/processing attempt."""

    attempt_number: int
    response_bytes: int | None
    parsed_rows: int | None
    valid_rows: int | None
    raw_html_path: str | None
    result: str


@dataclass(frozen=True)
class BackfillState:
    """Durable progress for one requested historical range."""

    requested_start_date: date
    requested_end_date: date
    last_attempted_date: date | None = None
    last_successful_date: date | None = None
    successful_dates: tuple[date, ...] = ()
    non_trading_dates: tuple[date, ...] = ()
    temporary_skips: tuple[tuple[date, str], ...] = ()
    failed_dates: tuple[tuple[date, str], ...] = ()
    already_downloaded_dates: tuple[date, ...] = ()
    success_records: tuple[BackfillSuccessRecord, ...] = ()
    attempt_records: tuple[tuple[date, tuple[BackfillAttemptRecord, ...]], ...] = ()
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    status: str = "not_started"
    last_message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "requested_start_date": self.requested_start_date.isoformat(),
            "requested_end_date": self.requested_end_date.isoformat(),
            "last_attempted_date": (
                self.last_attempted_date.isoformat()
                if self.last_attempted_date is not None
                else None
            ),
            "last_successful_date": (
                self.last_successful_date.isoformat()
                if self.last_successful_date is not None
                else None
            ),
            "successful_dates": [value.isoformat() for value in self.successful_dates],
            "non_trading_dates": [
                value.isoformat() for value in self.non_trading_dates
            ],
            "temporary_skips": {
                value.isoformat(): reason for value, reason in self.temporary_skips
            },
            "failed_dates": {
                value.isoformat(): reason for value, reason in self.failed_dates
            },
            "already_downloaded_dates": [
                value.isoformat() for value in self.already_downloaded_dates
            ],
            "success_records": {
                record.trading_date.isoformat(): {
                    "valid_rows": record.valid_rows,
                    "output_path": record.output_path,
                    "parsed_rows": record.parsed_rows,
                    "rejected_rows": record.rejected_rows,
                    "message": record.message,
                    "reconciled": record.reconciled,
                    "attempt_count": record.attempt_count,
                    "response_sizes": list(record.response_sizes),
                }
                for record in self.success_records
            },
            "attempt_records": {
                trading_date.isoformat(): [
                    {
                        "attempt_number": attempt.attempt_number,
                        "response_bytes": attempt.response_bytes,
                        "parsed_rows": attempt.parsed_rows,
                        "valid_rows": attempt.valid_rows,
                        "raw_html_path": attempt.raw_html_path,
                        "result": attempt.result,
                    }
                    for attempt in attempts
                ]
                for trading_date, attempts in self.attempt_records
            },
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at or None,
            "status": self.status,
            "last_message": self.last_message,
        }

    @classmethod
    def from_dict(cls, values: object) -> BackfillState:
        """Validate and deserialize saved progress."""
        if not isinstance(values, dict):
            raise BackfillStateError("backfill state must be a JSON object")

        def parsed_date(name: str, *, optional: bool = False) -> date | None:
            raw = values.get(name)
            if optional and raw in {None, ""}:
                return None
            if not isinstance(raw, str):
                raise BackfillStateError(f"backfill state field {name} is invalid")
            try:
                return date.fromisoformat(raw)
            except ValueError as exc:
                raise BackfillStateError(
                    f"backfill state field {name} is invalid"
                ) from exc

        def parsed_dates(name: str) -> tuple[date, ...]:
            raw = values.get(name, [])
            if not isinstance(raw, list):
                raise BackfillStateError(f"backfill state field {name} is invalid")
            try:
                return tuple(sorted({date.fromisoformat(str(value)) for value in raw}))
            except ValueError as exc:
                raise BackfillStateError(
                    f"backfill state field {name} is invalid"
                ) from exc

        def parsed_reasons(name: str) -> tuple[tuple[date, str], ...]:
            raw = values.get(name, {})
            if not isinstance(raw, dict):
                raise BackfillStateError(f"backfill state field {name} is invalid")
            try:
                return tuple(
                    sorted(
                        (date.fromisoformat(str(key)), str(reason))
                        for key, reason in raw.items()
                    )
                )
            except ValueError as exc:
                raise BackfillStateError(
                    f"backfill state field {name} is invalid"
                ) from exc

        def parsed_success_records() -> tuple[BackfillSuccessRecord, ...]:
            raw = values.get("success_records", {})
            if not isinstance(raw, dict):
                raise BackfillStateError(
                    "backfill state field success_records is invalid"
                )
            records: list[BackfillSuccessRecord] = []
            try:
                for key, detail in raw.items():
                    if not isinstance(detail, dict):
                        raise TypeError
                    parsed_rows = detail.get("parsed_rows")
                    rejected_rows = detail.get("rejected_rows")
                    records.append(
                        BackfillSuccessRecord(
                            trading_date=date.fromisoformat(str(key)),
                            valid_rows=int(detail["valid_rows"]),
                            output_path=str(detail["output_path"]),
                            parsed_rows=(
                                int(parsed_rows) if parsed_rows is not None else None
                            ),
                            rejected_rows=(
                                int(rejected_rows)
                                if rejected_rows is not None
                                else None
                            ),
                            message=str(detail.get("message") or ""),
                            reconciled=bool(detail.get("reconciled", False)),
                            attempt_count=int(detail.get("attempt_count", 0)),
                            response_sizes=tuple(
                                int(value)
                                for value in detail.get("response_sizes", [])
                            ),
                        )
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise BackfillStateError(
                    "backfill state field success_records is invalid"
                ) from exc
            return tuple(sorted(records, key=lambda record: record.trading_date))

        def parsed_attempt_records() -> tuple[
            tuple[date, tuple[BackfillAttemptRecord, ...]], ...
        ]:
            raw = values.get("attempt_records", {})
            if not isinstance(raw, dict):
                raise BackfillStateError(
                    "backfill state field attempt_records is invalid"
                )
            records: list[tuple[date, tuple[BackfillAttemptRecord, ...]]] = []
            try:
                for key, attempts in raw.items():
                    if not isinstance(attempts, list):
                        raise TypeError
                    parsed_attempts = tuple(
                        BackfillAttemptRecord(
                            attempt_number=int(attempt["attempt_number"]),
                            response_bytes=(
                                int(attempt["response_bytes"])
                                if attempt.get("response_bytes") is not None
                                else None
                            ),
                            parsed_rows=(
                                int(attempt["parsed_rows"])
                                if attempt.get("parsed_rows") is not None
                                else None
                            ),
                            valid_rows=(
                                int(attempt["valid_rows"])
                                if attempt.get("valid_rows") is not None
                                else None
                            ),
                            raw_html_path=(
                                str(attempt["raw_html_path"])
                                if attempt.get("raw_html_path") is not None
                                else None
                            ),
                            result=str(attempt["result"]),
                        )
                        for attempt in attempts
                        if isinstance(attempt, dict)
                    )
                    records.append((date.fromisoformat(str(key)), parsed_attempts))
            except (KeyError, TypeError, ValueError) as exc:
                raise BackfillStateError(
                    "backfill state field attempt_records is invalid"
                ) from exc
            return tuple(sorted(records))

        start = parsed_date("requested_start_date")
        end = parsed_date("requested_end_date")
        if not isinstance(start, date) or not isinstance(end, date) or end < start:
            raise BackfillStateError("backfill state contains an invalid date range")
        return cls(
            requested_start_date=start,
            requested_end_date=end,
            last_attempted_date=parsed_date("last_attempted_date", optional=True),
            last_successful_date=parsed_date("last_successful_date", optional=True),
            successful_dates=parsed_dates("successful_dates"),
            non_trading_dates=parsed_dates("non_trading_dates"),
            temporary_skips=parsed_reasons("temporary_skips"),
            failed_dates=parsed_reasons("failed_dates"),
            already_downloaded_dates=parsed_dates("already_downloaded_dates"),
            success_records=parsed_success_records(),
            attempt_records=parsed_attempt_records(),
            started_at=str(values.get("started_at") or ""),
            updated_at=str(values.get("updated_at") or ""),
            completed_at=str(values.get("completed_at") or ""),
            status=str(values.get("status") or "not_started"),
            last_message=str(values.get("last_message") or ""),
        )


@dataclass(frozen=True)
class BackfillPlan:
    """Read-only plan for a requested historical range."""

    start_date: date
    end_date: date
    total_calendar_dates: int
    existing_successful_dates: tuple[date, ...]
    dates_requiring_requests: tuple[date, ...]
    weekend_dates: tuple[date, ...]
    unresolved_skipped_dates: tuple[date, ...]
    failed_dates_eligible_for_retry: tuple[date, ...]
    estimated_request_count: int
    estimated_minimum_duration_seconds: float


@dataclass(frozen=True)
class BackfillDateResult:
    """Classified outcome for one attempted or locally resolved date."""

    trading_date: date
    status: BackfillOutcome
    reason: str
    output_path: Path | None = None
    parsed_rows: int | None = None
    valid_rows: int | None = None
    rejected_rows: int | None = None
    reconciled: bool = False
    attempts: tuple[BackfillAttemptRecord, ...] = ()

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def response_sizes(self) -> tuple[int, ...]:
        return tuple(
            attempt.response_bytes
            for attempt in self.attempts
            if attempt.response_bytes is not None
        )


@dataclass(frozen=True)
class BackfillProgress:
    """Progress event emitted after a requested date is durably recorded."""

    completed_requests: int
    scheduled_requests: int
    outcome: BackfillDateResult
    state: BackfillState


@dataclass(frozen=True)
class BackfillRunResult:
    """Structured result returned by programmatic and CLI backfill runs."""

    plan: BackfillPlan
    attempted_dates: tuple[date, ...]
    outcomes: tuple[BackfillDateResult, ...]
    state: BackfillState | None
    dry_run: bool
    interrupted: bool
    circuit_breaker_triggered: bool = False
    pause_reason: str = ""

    def count(self, status: BackfillOutcome) -> int:
        """Count this run's outcomes for one status."""
        return sum(outcome.status == status for outcome in self.outcomes)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def load_backfill_state(path: Path = BACKFILL_STATE_PATH) -> BackfillState | None:
    """Load saved state, returning a safe empty result when it is malformed."""
    state_path = Path(path)
    if not state_path.is_file():
        return None
    try:
        values = json.loads(state_path.read_text(encoding="utf-8"))
        return BackfillState.from_dict(values)
    except (OSError, UnicodeError, json.JSONDecodeError, BackfillStateError) as exc:
        LOGGER.warning("Ignoring malformed backfill state %s: %s", state_path, exc)
        return None


def write_backfill_state(
    state: BackfillState,
    path: Path = BACKFILL_STATE_PATH,
) -> Path:
    """Persist backfill progress atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _state_dates(values: tuple[tuple[date, str], ...]) -> set[date]:
    return {value for value, _ in values}


def create_backfill_plan(
    start_date: date,
    end_date: date,
    *,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    state: BackfillState | None = None,
    csv_dir: Path = RAW_CSV_DIR,
    retry_failed: bool = False,
    retry_temporary_only: bool = False,
    today: date | None = None,
    available_dates: Sequence[date] | None = None,
) -> BackfillPlan:
    """Plan required sequential requests without changing local state."""
    if end_date < start_date:
        raise ValueError("end date cannot be earlier than start date")
    if delay_seconds < 0:
        raise ValueError("delay seconds cannot be negative")
    current_date = today or date.today()
    calendar_dates = tuple(iter_calendar_dates(start_date, end_date))
    discovered = (
        tuple(available_dates)
        if available_dates is not None
        else discover_available_raw_dates(Path(csv_dir))
    )
    existing = set(discovered).intersection(calendar_dates)
    weekends = {value for value in calendar_dates if value.weekday() >= 5}
    known_non_trading = (
        {
            value
            for value in state.non_trading_dates
            if value.weekday() >= 5
        }
        if state is not None
        else set()
    )
    temporary = _state_dates(state.temporary_skips) if state is not None else set()
    failed = _state_dates(state.failed_dates) if state is not None else set()
    unresolved = {
        value
        for value in temporary
        if start_date <= value <= end_date and value not in existing
    }
    request_dates: list[date] = []

    for trading_date in calendar_dates:
        if trading_date in existing or trading_date in weekends:
            continue
        if trading_date >= current_date:
            unresolved.add(trading_date)
            continue
        if retry_temporary_only and trading_date not in temporary:
            continue
        if trading_date in known_non_trading:
            continue
        if trading_date in failed and not retry_failed:
            continue
        request_dates.append(trading_date)

    estimated_requests = len(request_dates)
    return BackfillPlan(
        start_date=start_date,
        end_date=end_date,
        total_calendar_dates=len(calendar_dates),
        existing_successful_dates=tuple(sorted(existing)),
        dates_requiring_requests=tuple(request_dates),
        weekend_dates=tuple(sorted(weekends)),
        unresolved_skipped_dates=tuple(sorted(unresolved)),
        failed_dates_eligible_for_retry=tuple(
            sorted(value for value in failed if start_date <= value <= end_date)
        ),
        estimated_request_count=estimated_requests,
        estimated_minimum_duration_seconds=(
            max(0, estimated_requests - 1) * delay_seconds
        ),
    )


def _new_state(
    start_date: date,
    end_date: date,
    clock: Callable[[], datetime],
) -> BackfillState:
    timestamp = _timestamp(clock)
    return BackfillState(
        requested_start_date=start_date,
        requested_end_date=end_date,
        started_at=timestamp,
        updated_at=timestamp,
        status="running",
        last_message="Backfill started",
    )


def _record_outcome(
    state: BackfillState,
    outcome: BackfillDateResult,
    clock: Callable[[], datetime],
    *,
    attempted: bool = True,
) -> BackfillState:
    trading_date = outcome.trading_date
    successful = set(state.successful_dates)
    non_trading = set(state.non_trading_dates)
    temporary = dict(state.temporary_skips)
    failed = dict(state.failed_dates)
    downloaded = set(state.already_downloaded_dates)
    success_records = {
        record.trading_date: record for record in state.success_records
    }
    attempt_records = dict(state.attempt_records)
    successful.discard(trading_date)
    non_trading.discard(trading_date)
    temporary.pop(trading_date, None)
    failed.pop(trading_date, None)
    downloaded.discard(trading_date)
    success_records.pop(trading_date, None)

    if outcome.status == "successful":
        successful.add(trading_date)
        if outcome.valid_rows is None or outcome.output_path is None:
            raise ValueError("successful outcome requires row count and output path")
        success_records[trading_date] = BackfillSuccessRecord(
            trading_date=trading_date,
            valid_rows=outcome.valid_rows,
            output_path=str(outcome.output_path),
            parsed_rows=outcome.parsed_rows,
            rejected_rows=outcome.rejected_rows,
            message=outcome.reason,
            reconciled=outcome.reconciled,
            attempt_count=outcome.attempt_count,
            response_sizes=outcome.response_sizes,
        )
    elif outcome.status == "non_trading":
        non_trading.add(trading_date)
    elif outcome.status == "temporary_unavailable":
        temporary[trading_date] = outcome.reason
    elif outcome.status == "failed":
        failed[trading_date] = outcome.reason
    else:
        downloaded.add(trading_date)
    if outcome.attempts:
        attempt_records[trading_date] = outcome.attempts

    return replace(
        state,
        last_attempted_date=(
            trading_date if attempted else state.last_attempted_date
        ),
        last_successful_date=max(successful) if successful else None,
        successful_dates=tuple(sorted(successful)),
        non_trading_dates=tuple(sorted(non_trading)),
        temporary_skips=tuple(sorted(temporary.items())),
        failed_dates=tuple(sorted(failed.items())),
        already_downloaded_dates=tuple(sorted(downloaded)),
        success_records=tuple(
            success_records[value] for value in sorted(success_records)
        ),
        attempt_records=tuple(sorted(attempt_records.items())),
        updated_at=_timestamp(clock),
        status="running",
        last_message=f"{trading_date.isoformat()}: {outcome.reason}",
    )


def _classify_empty_response(trading_date: date, today: date) -> BackfillDateResult:
    if trading_date.weekday() >= 5:
        return BackfillDateResult(
            trading_date,
            "non_trading",
            "Weekend date",
        )
    if trading_date >= today:
        reason = "Today is not final" if trading_date == today else "Future date"
        return BackfillDateResult(trading_date, "temporary_unavailable", reason)
    return BackfillDateResult(
        trading_date,
        "temporary_unavailable",
        "Weekday response contained no equity rows; eligible for retry",
    )


def _outcome_from_collection(
    trading_date: date,
    collection: CollectionResult,
    *,
    csv_validator: Callable[[Path, date], bool],
    today: date,
) -> BackfillDateResult:
    """Classify one requested date from its matching processing result."""
    matching_results = tuple(
        result
        for result in collection.date_results
        if result.trading_date == trading_date
    )
    if len(matching_results) > 1:
        raise RuntimeError("single-date collection returned duplicate date results")
    if matching_results:
        result = matching_results[0]
        if result.status == "skipped":
            return _classify_empty_response(trading_date, today)
        if result.parsed_rows <= 0 or result.valid_rows <= 0:
            raise ValueError("successful processing result contains no valid equity rows")
        if result.output_path is None:
            raise OSError("successful processing result has no output CSV path")
        output_path = Path(result.output_path)
        expected_name = f"market_{trading_date.isoformat()}.csv"
        if output_path.name != expected_name:
            raise OSError("processing result output CSV does not match requested date")
        if not csv_validator(output_path, trading_date):
            raise OSError("collection completed without a valid daily CSV")
        return BackfillDateResult(
            trading_date,
            "successful",
            f"Saved {result.valid_rows} valid equity rows",
            output_path,
            parsed_rows=result.parsed_rows,
            valid_rows=result.valid_rows,
            rejected_rows=result.rejected_rows,
        )

    matching_failures = tuple(
        reason for failed_date, reason in collection.failed_dates
        if failed_date == trading_date
    )
    if matching_failures:
        return BackfillDateResult(trading_date, "failed", matching_failures[0])
    if trading_date in collection.skipped_dates:
        return _classify_empty_response(trading_date, today)
    raise RuntimeError("single-date collection returned no result for requested date")


class _AttemptAuditClient:
    """Capture one response while delegating transport to a fresh client."""

    def __init__(self, client: MarketClient, attempt_number: int, raw_dir: Path) -> None:
        self.client = client
        self.attempt_number = attempt_number
        self.raw_dir = raw_dir
        self.response_bytes: int | None = None
        self.raw_html_path: Path | None = None

    def fetch_market_by_date(self, trading_date: date) -> str:
        html = self.client.fetch_market_by_date(trading_date)
        encoded = html.encode("utf-8")
        self.response_bytes = len(encoded)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_name = tempfile.mkstemp(
            dir=self.raw_dir,
            prefix=(
                f"market_{trading_date.isoformat()}_"
                f"attempt_{self.attempt_number:02d}_"
            ),
            suffix=".html",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.raw_html_path = Path(raw_name)
        return html


def _collect_with_empty_retries(
    trading_date: date,
    *,
    client: MarketClient | None,
    client_factory: Callable[[], MarketClient] | None,
    date_processor: DateProcessor,
    csv_validator: Callable[[Path, date], bool],
    today: date,
    max_attempts: int,
    retry_delays: Sequence[float],
    retry_jitter_seconds: float,
    jitter_fn: Callable[[float, float], float],
    sleep_fn: Callable[[float], None],
    attempt_html_dir: Path,
) -> tuple[BackfillDateResult, bool]:
    """Retry an empty weekday with a fresh production client per attempt."""
    if max_attempts < 1:
        raise ValueError("empty response max attempts must be at least 1")
    if any(delay < 0 for delay in retry_delays):
        raise ValueError("empty response retry delays cannot be negative")
    if retry_jitter_seconds < 0:
        raise ValueError("retry jitter seconds cannot be negative")

    attempts: list[BackfillAttemptRecord] = []
    first_attempt_empty = False
    final_outcome: BackfillDateResult | None = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_client = (
            client_factory()
            if client_factory is not None
            else client if client is not None else PsxClient()
        )
        audit_client = _AttemptAuditClient(
            attempt_client,
            attempt_number,
            attempt_html_dir,
        )
        collection = collect_single_date(
            trading_date,
            client=audit_client,
            date_processor=date_processor,
        )
        outcome = _outcome_from_collection(
            trading_date,
            collection,
            csv_validator=csv_validator,
            today=today,
        )
        matching = next(
            (
                result
                for result in collection.date_results
                if result.trading_date == trading_date
            ),
            None,
        )
        attempts.append(
            BackfillAttemptRecord(
                attempt_number=attempt_number,
                response_bytes=audit_client.response_bytes,
                parsed_rows=matching.parsed_rows if matching is not None else None,
                valid_rows=matching.valid_rows if matching is not None else None,
                raw_html_path=(
                    str(audit_client.raw_html_path)
                    if audit_client.raw_html_path is not None
                    else None
                ),
                result=outcome.status,
            )
        )
        final_outcome = replace(outcome, attempts=tuple(attempts))
        empty_weekday = (
            outcome.status == "temporary_unavailable"
            and trading_date.weekday() < 5
            and matching is not None
            and matching.status == "skipped"
        )
        if attempt_number == 1:
            first_attempt_empty = empty_weekday
        if not empty_weekday or attempt_number >= max_attempts:
            break
        delay_index = min(attempt_number - 1, len(retry_delays) - 1)
        base_delay = retry_delays[delay_index] if retry_delays else 0.0
        jitter = (
            jitter_fn(0.0, retry_jitter_seconds)
            if retry_jitter_seconds > 0
            else 0.0
        )
        sleep_fn(base_delay + jitter)

    if final_outcome is None:
        raise RuntimeError("backfill retry loop produced no outcome")
    if final_outcome.status == "temporary_unavailable":
        final_outcome = replace(
            final_outcome,
            reason=(
                f"Temporary unavailable after {final_outcome.attempt_count} attempts; "
                "weekday responses contained no equity rows; eligible for retry"
            ),
        )
    return final_outcome, first_attempt_empty


def _short_reason(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _matching_state(
    saved: BackfillState | None,
    start_date: date,
    end_date: date,
    clock: Callable[[], datetime],
) -> BackfillState:
    if saved is None:
        return _new_state(start_date, end_date, clock)
    if (
        saved.requested_start_date != start_date
        or saved.requested_end_date != end_date
    ):
        raise BackfillStateError(
            "saved backfill range does not match the requested start and end dates"
        )
    return replace(
        saved,
        status="running",
        completed_at="",
        updated_at=_timestamp(clock),
        last_message="Backfill resumed",
    )


def run_backfill(
    start_date: date,
    end_date: date,
    *,
    resume: bool = False,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    max_dates: int | None = None,
    retry_failed: bool = False,
    retry_temporary_only: bool = False,
    dry_run: bool = False,
    state_path: Path = BACKFILL_STATE_PATH,
    csv_dir: Path = RAW_CSV_DIR,
    client: MarketClient | None = None,
    client_factory: Callable[[], MarketClient] | None = None,
    date_processor: DateProcessor = process_date,
    daily_csv_validator: Callable[[Path, date], bool] = is_valid_daily_csv,
    sleep_fn: Callable[[float], None] = time.sleep,
    empty_max_attempts: int = DEFAULT_EMPTY_MAX_ATTEMPTS,
    empty_retry_delays: Sequence[float] = DEFAULT_EMPTY_RETRY_DELAYS,
    retry_jitter_seconds: float = 0.75,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    circuit_window: int = DEFAULT_CIRCUIT_WINDOW,
    circuit_empty_ratio: float = DEFAULT_CIRCUIT_EMPTY_RATIO,
    attempt_html_dir: Path | None = None,
    progress_callback: Callable[[BackfillProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    today: date | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BackfillRunResult:
    """Run a polite sequential backfill while persisting every outcome."""
    if max_dates is not None and max_dates < 1:
        raise ValueError("max dates must be at least 1")
    if circuit_window < 1:
        raise ValueError("circuit window must be at least 1")
    if not 0 < circuit_empty_ratio <= 1:
        raise ValueError("circuit empty ratio must be within (0, 1]")
    current_date = today or date.today()
    saved = (
        load_backfill_state(state_path)
        if (resume or retry_failed or retry_temporary_only)
        else None
    )
    state = _matching_state(saved, start_date, end_date, clock)
    plan = create_backfill_plan(
        start_date,
        end_date,
        delay_seconds=delay_seconds,
        state=state,
        csv_dir=csv_dir,
        retry_failed=retry_failed,
        retry_temporary_only=retry_temporary_only,
        today=current_date,
    )
    if dry_run:
        return BackfillRunResult(plan, (), (), None, True, False)

    outcomes: list[BackfillDateResult] = []
    for existing_date in plan.existing_successful_dates:
        if existing_date in set(state.successful_dates):
            continue
        output_path = Path(csv_dir) / f"market_{existing_date.isoformat()}.csv"
        needs_reconciliation = (
            existing_date in set(state.non_trading_dates)
            or existing_date in _state_dates(state.temporary_skips)
        )
        if needs_reconciliation:
            valid_rows = valid_daily_csv_row_count(output_path, existing_date)
            if valid_rows is None:
                continue
            outcome = BackfillDateResult(
                existing_date,
                "successful",
                f"Reconciled existing CSV with {valid_rows} valid equity rows",
                output_path,
                valid_rows=valid_rows,
                reconciled=True,
            )
            outcomes.append(outcome)
        else:
            outcome = BackfillDateResult(
                existing_date,
                "already_downloaded",
                "Valid daily CSV already exists",
                output_path,
            )
        state = _record_outcome(
            state,
            outcome,
            clock,
            attempted=False,
        )
    for weekend_date in plan.weekend_dates:
        if weekend_date not in set(plan.existing_successful_dates):
            state = _record_outcome(
                state,
                BackfillDateResult(weekend_date, "non_trading", "Weekend date"),
                clock,
                attempted=False,
            )
    for unresolved_date in plan.unresolved_skipped_dates:
        if unresolved_date >= current_date:
            state = _record_outcome(
                state,
                _classify_empty_response(unresolved_date, current_date),
                clock,
                attempted=False,
            )
    write_backfill_state(state, state_path)

    scheduled = plan.dates_requiring_requests
    if max_dates is not None:
        scheduled = scheduled[:max_dates]
    attempted: list[date] = []
    interrupted = False
    stopped = False
    circuit_breaker_triggered = False
    first_attempt_empty_window: list[bool] = []
    raw_attempt_directory = Path(attempt_html_dir or main_pipeline.RAW_HTML_DIR)

    for index, trading_date in enumerate(scheduled):
        if should_stop is not None and should_stop():
            stopped = True
            break
        attempted.append(trading_date)
        first_attempt_empty = False
        try:
            expected_path = Path(csv_dir) / f"market_{trading_date.isoformat()}.csv"
            if daily_csv_validator(expected_path, trading_date):
                valid_rows = valid_daily_csv_row_count(expected_path, trading_date)
                was_misclassified = (
                    trading_date in set(state.non_trading_dates)
                    or trading_date in _state_dates(state.temporary_skips)
                )
                if was_misclassified and valid_rows is not None:
                    outcome = BackfillDateResult(
                        trading_date,
                        "successful",
                        (
                            "Reconciled existing CSV with "
                            f"{valid_rows} valid equity rows"
                        ),
                        expected_path,
                        valid_rows=valid_rows,
                        reconciled=True,
                    )
                else:
                    outcome = BackfillDateResult(
                        trading_date,
                        "already_downloaded",
                        (
                            f"Valid daily CSV with {valid_rows} equity rows appeared "
                            "before request"
                            if valid_rows is not None
                            else "Valid daily CSV appeared before request"
                        ),
                        expected_path,
                        valid_rows=valid_rows,
                    )
            else:
                outcome, first_attempt_empty = _collect_with_empty_retries(
                    trading_date,
                    client=client,
                    client_factory=client_factory,
                    date_processor=date_processor,
                    csv_validator=daily_csv_validator,
                    today=current_date,
                    max_attempts=empty_max_attempts,
                    retry_delays=empty_retry_delays,
                    retry_jitter_seconds=retry_jitter_seconds,
                    jitter_fn=jitter_fn,
                    sleep_fn=sleep_fn,
                    attempt_html_dir=raw_attempt_directory,
                )
        except KeyboardInterrupt:
            interrupted = True
            state = replace(
                state,
                last_attempted_date=trading_date,
                updated_at=_timestamp(clock),
                status="interrupted",
                last_message=f"Interrupted while processing {trading_date.isoformat()}",
            )
            write_backfill_state(state, state_path)
            break
        except Exception as exc:
            outcome = BackfillDateResult(
                trading_date,
                "failed",
                _short_reason(exc),
            )

        outcomes.append(outcome)
        state = _record_outcome(state, outcome, clock)
        write_backfill_state(state, state_path)
        LOGGER.info(
            "Backfill %s: %s (%s)",
            trading_date.isoformat(),
            outcome.status,
            outcome.reason,
        )
        if progress_callback is not None:
            try:
                progress_callback(
                    BackfillProgress(index + 1, len(scheduled), outcome, state)
                )
            except KeyboardInterrupt:
                interrupted = True
                state = replace(
                    state,
                    updated_at=_timestamp(clock),
                    status="interrupted",
                    last_message="Interrupted after saving the current request",
                )
                write_backfill_state(state, state_path)
                break
        if trading_date.weekday() < 5:
            first_attempt_empty_window.append(first_attempt_empty)
            first_attempt_empty_window = first_attempt_empty_window[-circuit_window:]
            if (
                len(first_attempt_empty_window) == circuit_window
                and sum(first_attempt_empty_window) / circuit_window
                >= circuit_empty_ratio
            ):
                circuit_breaker_triggered = True
                break
        if should_stop is not None and should_stop():
            stopped = True
            break
        if index < len(scheduled) - 1 and delay_seconds > 0:
            try:
                sleep_fn(delay_seconds)
            except KeyboardInterrupt:
                interrupted = True
                state = replace(
                    state,
                    updated_at=_timestamp(clock),
                    status="interrupted",
                    last_message="Interrupted during the inter-request delay",
                )
                write_backfill_state(state, state_path)
                break

    if not interrupted:
        refreshed_plan = create_backfill_plan(
            start_date,
            end_date,
            delay_seconds=delay_seconds,
            state=state,
            csv_dir=csv_dir,
            retry_failed=retry_failed,
            retry_temporary_only=retry_temporary_only,
            today=current_date,
        )
        unresolved = bool(
            refreshed_plan.dates_requiring_requests
            or refreshed_plan.unresolved_skipped_dates
            or state.failed_dates
        )
        if circuit_breaker_triggered:
            status = "paused"
            message = CIRCUIT_BREAKER_MESSAGE
        elif stopped:
            status = "paused"
            message = "Stopped after the current request"
        elif unresolved:
            status = "paused"
            message = "Backfill batch finished with remaining or unresolved dates"
        else:
            status = "completed"
            message = "Backfill range completed"
        state = replace(
            state,
            status=status,
            updated_at=_timestamp(clock),
            completed_at=_timestamp(clock) if status == "completed" else "",
            last_message=message,
        )
        write_backfill_state(state, state_path)

    return BackfillRunResult(
        plan=plan,
        attempted_dates=tuple(attempted),
        outcomes=tuple(outcomes),
        state=state,
        dry_run=False,
        interrupted=interrupted,
        circuit_breaker_triggered=circuit_breaker_triggered,
        pause_reason=CIRCUIT_BREAKER_MESSAGE if circuit_breaker_triggered else "",
    )


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def _print_plan(plan: BackfillPlan) -> None:
    print(f"Requested start date: {plan.start_date.isoformat()}")
    print(f"Requested end date: {plan.end_date.isoformat()}")
    print(f"Total calendar dates: {plan.total_calendar_dates}")
    print(f"Existing successful dates: {len(plan.existing_successful_dates)}")
    print(f"Dates requiring requests: {len(plan.dates_requiring_requests)}")
    print(f"Weekend dates: {len(plan.weekend_dates)}")
    print(f"Unresolved skipped dates: {len(plan.unresolved_skipped_dates)}")
    print(
        "Failed dates eligible for retry: "
        f"{len(plan.failed_dates_eligible_for_retry)}"
    )
    print(f"Estimated request count: {plan.estimated_request_count}")
    print(
        "Estimated minimum duration: "
        f"{plan.estimated_minimum_duration_seconds:.1f} seconds"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone historical backfill command."""
    parser = argparse.ArgumentParser(description="Backfill historical PSX daily data")
    parser.add_argument("--start-date", type=_iso_date, required=True)
    parser.add_argument("--end-date", type=_iso_date, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
    )
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.end_date < args.start_date:
        parser.error("--end-date cannot be earlier than --start-date")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds cannot be negative")
    if args.max_dates is not None and args.max_dates < 1:
        parser.error("--max-dates must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_backfill(
            args.start_date,
            args.end_date,
            resume=args.resume,
            delay_seconds=args.delay_seconds,
            max_dates=args.max_dates,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    except (BackfillStateError, OSError, ValueError) as exc:
        LOGGER.error("Backfill failed: %s", exc)
        return 1
    _print_plan(result.plan)
    if result.dry_run:
        print("Status: dry run; no requests or state changes were made")
        return 0
    print(f"Attempted requests: {len(result.attempted_dates)}")
    for status in (
        "successful",
        "non_trading",
        "temporary_unavailable",
        "failed",
        "already_downloaded",
    ):
        print(f"{status.replace('_', ' ').title()}: {result.count(status)}")
    print(f"Status: {result.state.status if result.state else 'unknown'}")
    print(f"State file: {BACKFILL_STATE_PATH}")
    if result.interrupted:
        return 130
    return 1 if result.count("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
