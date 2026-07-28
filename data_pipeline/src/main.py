"""Command-line entry point for the one-date PSX pipeline."""

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol, Sequence

from .client import PsxClient
from .config import RAW_DATA_DIR, REJECTED_DATA_DIR
from .parser import parse_market_html
from .validator import validate_records


LOGGER = logging.getLogger(__name__)
OUTPUT_FIELDS = (
    "symbol",
    "date",
    "ldcp",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
)

DateStatus = Literal["successful", "skipped"]


@dataclass(frozen=True)
class DateProcessingResult:
    """Outcome and row counts for one requested calendar date."""

    trading_date: date
    status: DateStatus
    parsed_rows: int
    valid_rows: int
    rejected_rows: int
    output_path: Path | None


@dataclass(frozen=True)
class RangeProcessingResult:
    """Aggregate outcome for an inclusive date-range request."""

    start_date: date
    end_date: date
    total_dates: int
    successful_dates: int
    skipped_dates: int
    failed_dates: tuple[tuple[date, str], ...]


class MarketClient(Protocol):
    """Client interface needed by the date-processing workflow."""

    def fetch_market_by_date(self, trading_date: date) -> str:
        """Fetch market HTML for one date."""
        ...


DateProcessor = Callable[[date, MarketClient], DateProcessingResult]


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def iter_calendar_dates(start_date: date, end_date: date) -> Iterator[date]:
    """Yield every calendar date in an inclusive, valid date range."""
    if end_date < start_date:
        raise ValueError("end date cannot be earlier than start date")

    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OUTPUT_FIELDS)
    for record in records:
        for field in record:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def process_date(
    trading_date: date, client: MarketClient | None = None
) -> DateProcessingResult:
    """Fetch, preserve, parse, validate, and export one calendar date."""
    date_text = trading_date.isoformat()
    psx_client = client if client is not None else PsxClient()
    html = psx_client.fetch_market_by_date(trading_date)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    html_path = RAW_DATA_DIR / f"market_{date_text}.html"
    html_path.write_text(html, encoding="utf-8")

    parsed, parse_rejections = parse_market_html(html, trading_date)
    if not parsed and not parse_rejections:
        LOGGER.info("Skipping %s: response contains no equity rows", date_text)
        return DateProcessingResult(
            trading_date=trading_date,
            status="skipped",
            parsed_rows=0,
            valid_rows=0,
            rejected_rows=0,
            output_path=None,
        )

    valid, validation_rejections = validate_records(parsed)
    valid.sort(key=lambda record: str(record["symbol"]))

    csv_path = RAW_DATA_DIR / f"market_{date_text}.csv"
    _write_csv(csv_path, valid)

    rejected = [*parse_rejections, *validation_rejections]
    if rejected:
        rejected_path = REJECTED_DATA_DIR / f"market_{date_text}_rejected.csv"
        _write_csv(rejected_path, rejected)

    return DateProcessingResult(
        trading_date=trading_date,
        status="successful",
        parsed_rows=len(parsed),
        valid_rows=len(valid),
        rejected_rows=len(rejected),
        output_path=csv_path,
    )


def _print_single_date_summary(result: DateProcessingResult) -> None:
    """Print the existing concise one-date summary."""
    date_text = result.trading_date.isoformat()
    print(f"Requested date: {date_text}")
    print(f"Parsed rows: {result.parsed_rows}")
    print(f"Valid rows: {result.valid_rows}")
    print(f"Rejected rows: {result.rejected_rows}")
    if result.status == "skipped":
        print("Status: skipped (response contains no equity rows)")
        print("Output file: not created")
    else:
        print(f"Output file: {result.output_path}")


def run(trading_date: date) -> DateProcessingResult:
    """Run and report the original one-date pipeline mode."""
    result = process_date(trading_date)
    _print_single_date_summary(result)
    return result


def _short_failure_reason(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def run_date_range(
    start_date: date,
    end_date: date,
    *,
    client: MarketClient | None = None,
    date_processor: DateProcessor = process_date,
) -> RangeProcessingResult:
    """Process an inclusive range, continuing after individual date failures."""
    requested_dates = iter_calendar_dates(start_date, end_date)
    psx_client = client if client is not None else PsxClient()
    total_dates = 0
    successful_dates = 0
    skipped_dates = 0
    failures: list[tuple[date, str]] = []

    for trading_date in requested_dates:
        total_dates += 1
        try:
            result = date_processor(trading_date, psx_client)
        except Exception as exc:
            reason = _short_failure_reason(exc)
            failures.append((trading_date, reason))
            LOGGER.error("Failed %s: %s", trading_date.isoformat(), reason)
            continue

        if result.status == "skipped":
            skipped_dates += 1
        else:
            successful_dates += 1

    summary = RangeProcessingResult(
        start_date=start_date,
        end_date=end_date,
        total_dates=total_dates,
        successful_dates=successful_dates,
        skipped_dates=skipped_dates,
        failed_dates=tuple(failures),
    )
    _print_range_summary(summary)
    return summary


def _print_range_summary(result: RangeProcessingResult) -> None:
    """Print aggregate range counts and any per-date failures."""
    print(f"Requested start date: {result.start_date.isoformat()}")
    print(f"Requested end date: {result.end_date.isoformat()}")
    print(f"Total calendar dates processed: {result.total_dates}")
    print(f"Successful dates: {result.successful_dates}")
    print(f"Skipped dates: {result.skipped_dates}")
    print(f"Failed dates: {len(result.failed_dates)}")
    if result.failed_dates:
        print("Failed date details:")
        for failed_date, reason in result.failed_dates:
            print(f"- {failed_date.isoformat()}: {reason}")
    else:
        print("Failed date details: none")


def _validate_cli_dates(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Enforce either one date or one complete, ordered date range."""
    has_range_value = args.start_date is not None or args.end_date is not None
    if args.trading_date is not None and has_range_value:
        parser.error("--date cannot be combined with --start-date or --end-date")
    if args.trading_date is None and not has_range_value:
        parser.error("provide either --date or both --start-date and --end-date")
    if args.trading_date is None and (
        args.start_date is None or args.end_date is None
    ):
        parser.error("--start-date and --end-date must be provided together")
    if (
        args.start_date is not None
        and args.end_date is not None
        and args.end_date < args.start_date
    ):
        parser.error("--end-date cannot be earlier than --start-date")


def main(argv: Sequence[str] | None = None) -> int:
    """Run either the one-date or inclusive date-range pipeline."""
    parser = argparse.ArgumentParser(description="Fetch historical PSX market data")
    parser.add_argument("--date", type=_iso_date, dest="trading_date")
    parser.add_argument("--start-date", type=_iso_date)
    parser.add_argument("--end-date", type=_iso_date)
    args = parser.parse_args(argv)
    _validate_cli_dates(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.start_date is not None and args.end_date is not None:
        summary = run_date_range(args.start_date, args.end_date)
        return 1 if summary.failed_dates else 0

    try:
        run(args.trading_date)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
