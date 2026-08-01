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
import tempfile
import time
from typing import Literal

from .client import PsxClient
from .config import BACKFILL_STATE_PATH, RAW_CSV_DIR
from .main import (
    DateProcessingResult,
    DateProcessor,
    MarketClient,
    iter_calendar_dates,
    process_date,
)
from .updater import discover_available_raw_dates, is_valid_daily_csv


LOGGER = logging.getLogger(__name__)
DEFAULT_DELAY_SECONDS = 1.0
RECENT_AMBIGUOUS_DAYS = 2

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
    known_non_trading = set(state.non_trading_dates) if state is not None else set()
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
    successful.discard(trading_date)
    non_trading.discard(trading_date)
    temporary.pop(trading_date, None)
    failed.pop(trading_date, None)
    downloaded.discard(trading_date)

    last_successful = state.last_successful_date
    if outcome.status == "successful":
        successful.add(trading_date)
        last_successful = trading_date
    elif outcome.status == "non_trading":
        non_trading.add(trading_date)
    elif outcome.status == "temporary_unavailable":
        temporary[trading_date] = outcome.reason
    elif outcome.status == "failed":
        failed[trading_date] = outcome.reason
    else:
        downloaded.add(trading_date)

    return replace(
        state,
        last_attempted_date=(
            trading_date if attempted else state.last_attempted_date
        ),
        last_successful_date=last_successful,
        successful_dates=tuple(sorted(successful)),
        non_trading_dates=tuple(sorted(non_trading)),
        temporary_skips=tuple(sorted(temporary.items())),
        failed_dates=tuple(sorted(failed.items())),
        already_downloaded_dates=tuple(sorted(downloaded)),
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
    age_days = (today - trading_date).days
    if trading_date >= today:
        reason = "Today is not final" if trading_date == today else "Future date"
        return BackfillDateResult(trading_date, "temporary_unavailable", reason)
    if age_days <= RECENT_AMBIGUOUS_DAYS:
        return BackfillDateResult(
            trading_date,
            "temporary_unavailable",
            "Recent empty response may not be published yet",
        )
    return BackfillDateResult(
        trading_date,
        "non_trading",
        "Historical endpoint returned no equity rows",
    )


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
    dry_run: bool = False,
    state_path: Path = BACKFILL_STATE_PATH,
    csv_dir: Path = RAW_CSV_DIR,
    client: MarketClient | None = None,
    date_processor: DateProcessor = process_date,
    daily_csv_validator: Callable[[Path, date], bool] = is_valid_daily_csv,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress_callback: Callable[[BackfillProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    today: date | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BackfillRunResult:
    """Run a polite sequential backfill while persisting every outcome."""
    if max_dates is not None and max_dates < 1:
        raise ValueError("max dates must be at least 1")
    current_date = today or date.today()
    saved = load_backfill_state(state_path) if (resume or retry_failed) else None
    state = _matching_state(saved, start_date, end_date, clock)
    plan = create_backfill_plan(
        start_date,
        end_date,
        delay_seconds=delay_seconds,
        state=state,
        csv_dir=csv_dir,
        retry_failed=retry_failed,
        today=current_date,
    )
    if dry_run:
        return BackfillRunResult(plan, (), (), None, True, False)

    for existing_date in plan.existing_successful_dates:
        if existing_date in set(state.successful_dates):
            continue
        state = _record_outcome(
            state,
            BackfillDateResult(
                existing_date,
                "already_downloaded",
                "Valid daily CSV already exists",
                Path(csv_dir) / f"market_{existing_date.isoformat()}.csv",
            ),
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
    psx_client = client if client is not None else PsxClient()
    outcomes: list[BackfillDateResult] = []
    attempted: list[date] = []
    interrupted = False
    stopped = False

    for index, trading_date in enumerate(scheduled):
        if should_stop is not None and should_stop():
            stopped = True
            break
        attempted.append(trading_date)
        try:
            expected_path = Path(csv_dir) / f"market_{trading_date.isoformat()}.csv"
            if daily_csv_validator(expected_path, trading_date):
                outcome = BackfillDateResult(
                    trading_date,
                    "already_downloaded",
                    "Valid daily CSV appeared before request",
                    expected_path,
                )
            else:
                result: DateProcessingResult = date_processor(trading_date, psx_client)
                if result.status == "skipped":
                    outcome = _classify_empty_response(trading_date, current_date)
                else:
                    output_path = result.output_path or expected_path
                    if not daily_csv_validator(Path(output_path), trading_date):
                        raise OSError(
                            "collection completed without a valid daily CSV"
                        )
                    outcome = BackfillDateResult(
                        trading_date,
                        "successful",
                        f"Saved {result.valid_rows} valid equity rows",
                        Path(output_path),
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
            today=current_date,
        )
        unresolved = bool(
            refreshed_plan.dates_requiring_requests
            or refreshed_plan.unresolved_skipped_dates
            or state.failed_dates
        )
        if stopped:
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
