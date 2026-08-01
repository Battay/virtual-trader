"""Standalone command for manual and launchd-triggered PSX updates."""

import argparse
from datetime import date
import logging
from typing import Sequence

from .automation import (
    configure_auto_update_logging,
    run_manual_update,
    run_scheduled_update,
)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run a manual incremental update or the configured scheduled job."""
    parser = argparse.ArgumentParser(description="Incrementally update PSX data")
    subparsers = parser.add_subparsers(dest="command")
    manual_parser = subparsers.add_parser("update", help="Fetch missing data now")
    manual_parser.add_argument("--end-date", type=_iso_date)
    manual_parser.add_argument("--bootstrap-start-date", type=_iso_date)
    subparsers.add_parser("scheduled", help="Run only when automation is enabled")
    args = parser.parse_args(argv)

    configure_auto_update_logging()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "scheduled":
        result = run_scheduled_update()
    else:
        result = run_manual_update(
            end_date=getattr(args, "end_date", None),
            bootstrap_start_date=getattr(args, "bootstrap_start_date", None),
        )

    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    if result.update_result is not None:
        print(f"Dates processed: {result.update_result.total_processed}")
        print(f"Successful: {len(result.update_result.successful_dates)}")
        print(f"Skipped: {len(result.update_result.skipped_dates)}")
        print(f"Failed: {len(result.update_result.failed_dates)}")
    if result.master_result is not None:
        print(f"Master file: {result.master_result.output_path}")
    print(
        "Stages: "
        f"market={'ok' if result.market_update_succeeded else 'not completed'}, "
        f"master={'ok' if result.master_rebuild_succeeded else 'not completed'}, "
        f"listings={'ok' if result.listing_refresh_succeeded else 'not completed'}, "
        f"registry={'ok' if result.registry_rebuild_succeeded else 'not completed'}"
    )
    print(f"Cached listings used: {'yes' if result.cached_listings_used else 'no'}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
