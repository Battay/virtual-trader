"""Refresh official PSX index series independently and resiliently."""

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from data_pipeline.src.config import PROJECT_TIMEZONE

from .index_client import PsxIndexClient
from .index_config import (
    COMBINED_INDEX_MASTER_PATH, REFRESH_METADATA_PATH, SUPPORTED_INDEX_CODES,
    require_supported_index,
)
from .index_parser import parse_index_series
from .index_store import build_combined_master, update_index_csv, write_raw_snapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexRefreshResult:
    requested_indices: tuple[str, ...]
    successful_indices: tuple[str, ...]
    failed_indices: tuple[tuple[str, str], ...]
    observations_parsed: int
    earliest_date: date | None
    latest_date: date | None
    output_paths: tuple[Path, ...]
    combined_master_path: Path
    cached_data_used: bool

    @property
    def has_usable_data(self) -> bool:
        return bool(self.successful_indices) or self.cached_data_used

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["earliest_date"] = self.earliest_date.isoformat() if self.earliest_date else None
        values["latest_date"] = self.latest_date.isoformat() if self.latest_date else None
        values["output_paths"] = [str(path) for path in self.output_paths]
        values["combined_master_path"] = str(self.combined_master_path)
        return values


def refresh_indices(
    index_codes: Sequence[str] | None = None,
    *,
    client: PsxIndexClient | None = None,
    save_raw: bool = True,
    metadata_path: Path = REFRESH_METADATA_PATH,
    combined_path: Path = COMBINED_INDEX_MASTER_PATH,
) -> IndexRefreshResult:
    requested = tuple(index_codes or SUPPORTED_INDEX_CODES)
    definitions = tuple(require_supported_index(code) for code in requested)
    psx_client = client or PsxIndexClient()
    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    output_paths: list[Path] = []
    dates: list[date] = []
    observations = 0
    cached = False
    for definition in definitions:
        try:
            payload = psx_client.fetch_index_series(definition.code)
            parsed = parse_index_series(payload, definition.code)
            if parsed.data.empty:
                raise ValueError(f"No valid observations for {definition.code}")
            if save_raw:
                write_raw_snapshot(payload, definition.raw_path)
            output_paths.append(update_index_csv(definition.code, parsed.data))
            successful.append(definition.code)
            observations += len(parsed.data)
            parsed_dates = parsed.data["date"].map(date.fromisoformat)
            dates.extend(parsed_dates.tolist())
            if parsed.rejected:
                LOGGER.warning("Rejected %s malformed %s observations", len(parsed.rejected), definition.code)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failed.append((definition.code, reason))
            cached = cached or definition.master_path.exists()
            LOGGER.error("Index refresh failed for %s: %s", definition.code, reason)
    build_combined_master(output_path=combined_path)
    result = IndexRefreshResult(
        requested_indices=tuple(definition.code for definition in definitions),
        successful_indices=tuple(successful), failed_indices=tuple(failed),
        observations_parsed=observations,
        earliest_date=min(dates) if dates else None, latest_date=max(dates) if dates else None,
        output_paths=tuple(output_paths), combined_master_path=Path(combined_path),
        cached_data_used=cached,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = result.to_dict() | {"refreshed_at": datetime.now(ZoneInfo(PROJECT_TIMEZONE)).isoformat(timespec="seconds")}
    write_raw_snapshot(metadata, metadata_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh official PSX index series")
    parser.add_argument("--index", action="append", choices=SUPPORTED_INDEX_CODES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-raw-snapshot", action="store_true")
    args = parser.parse_args(argv)
    codes = SUPPORTED_INDEX_CODES if args.all or not args.index else tuple(args.index)
    client = PsxIndexClient(timeout=args.timeout) if args.timeout else PsxIndexClient()
    result = refresh_indices(codes, client=client, save_raw=not args.no_raw_snapshot)
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.failed_indices and not result.has_usable_data else 0


if __name__ == "__main__":
    raise SystemExit(main())
