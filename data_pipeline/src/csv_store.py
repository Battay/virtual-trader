"""Deterministic persistent master-dataset storage for PSX daily CSV files."""

import argparse
from dataclasses import dataclass
from datetime import date
import logging
import os
from pathlib import Path
import tempfile
from typing import Sequence

import pandas as pd

from .config import LEGACY_MARKET_COMPAT_PATH, RAW_CSV_DIR
from .main import OUTPUT_FIELDS


LOGGER = logging.getLogger(__name__)
BUSINESS_KEY = ("symbol", "date")


@dataclass(frozen=True)
class MasterBuildResult:
    """Metrics and warnings produced by a master-dataset rebuild."""

    output_path: Path
    total_rows: int
    unique_symbols: int
    earliest_date: date | None
    latest_date: date | None
    duplicate_count: int
    source_files: int
    errors: tuple[str, ...]


def _read_raw_files(
    raw_csv_dir: Path,
) -> tuple[pd.DataFrame, tuple[Path, ...], tuple[str, ...]]:
    frames: list[pd.DataFrame] = []
    loaded_paths: list[Path] = []
    errors: list[str] = []

    paths = tuple(sorted(Path(raw_csv_dir).glob("market_*.csv")))
    for source_order, path in enumerate(paths):
        try:
            frame = pd.read_csv(path, dtype={"symbol": "string"})
        except (
            OSError,
            UnicodeError,
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            errors.append(f"Could not read {path}: {exc}")
            continue

        missing_columns = [field for field in OUTPUT_FIELDS if field not in frame]
        if frame.empty or missing_columns:
            detail = (
                "contains no rows"
                if frame.empty
                else f"is missing columns: {', '.join(missing_columns)}"
            )
            errors.append(f"Ignoring {path}: {detail}")
            continue

        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        if parsed_dates.isna().any():
            errors.append(f"Ignoring {path}: contains invalid dates")
            continue

        selected = frame.loc[:, list(OUTPUT_FIELDS)].copy()
        selected["symbol"] = selected["symbol"].astype("string").str.strip()
        selected["date"] = parsed_dates
        selected["_source_order"] = source_order
        try:
            selected["_source_modified_ns"] = path.stat().st_mtime_ns
        except OSError as exc:
            errors.append(f"Could not inspect {path}: {exc}")
            continue
        selected["_row_order"] = range(len(selected))
        selected["_source_path"] = str(path)
        frames.append(selected)
        loaded_paths.append(path)

    if not frames:
        empty = pd.DataFrame(
            columns=[
                *OUTPUT_FIELDS,
                "_source_order",
                "_source_modified_ns",
                "_row_order",
            ]
        )
        return empty, tuple(loaded_paths), tuple(errors)
    return pd.concat(frames, ignore_index=True), tuple(loaded_paths), tuple(errors)


def _warn_about_conflicts(data: pd.DataFrame) -> None:
    duplicate_rows = data.loc[data.duplicated(list(BUSINESS_KEY), keep=False)]
    if duplicate_rows.empty:
        return

    for (symbol, trading_date), group in duplicate_rows.groupby(
        list(BUSINESS_KEY),
        sort=True,
        dropna=False,
    ):
        compared = group.loc[:, list(OUTPUT_FIELDS)].drop_duplicates()
        if len(compared) <= 1:
            continue
        newest = group.sort_values(
            ["_source_modified_ns", "_source_order", "_row_order"],
            kind="stable",
        ).iloc[-1]
        LOGGER.warning(
            "Conflicting rows for %s on %s; keeping newest raw-file version from %s",
            symbol,
            pd.Timestamp(trading_date).date().isoformat(),
            newest["_source_path"],
        )


def _atomic_write_csv(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        data.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_master_dataset(
    *,
    raw_csv_dir: Path = RAW_CSV_DIR,
    output_path: Path = LEGACY_MARKET_COMPAT_PATH,
) -> MasterBuildResult:
    """Build the explicitly named legacy compatibility CSV.

    The canonical market master is owned by ``native_market_pipeline``.  This
    legacy utility must never overwrite it.
    """
    raw_data, loaded_paths, errors = _read_raw_files(Path(raw_csv_dir))
    duplicate_count = int(
        raw_data.duplicated(list(BUSINESS_KEY), keep="last").sum()
    )
    _warn_about_conflicts(raw_data)

    if raw_data.empty:
        master = pd.DataFrame(columns=OUTPUT_FIELDS)
        earliest_date = None
        latest_date = None
        unique_symbols = 0
    else:
        master = (
            raw_data.sort_values(
                ["_source_modified_ns", "_source_order", "_row_order"],
                kind="stable",
            )
            .drop_duplicates(list(BUSINESS_KEY), keep="last")
            .sort_values(["date", "symbol"], kind="stable")
            .loc[:, list(OUTPUT_FIELDS)]
            .reset_index(drop=True)
        )
        earliest_date = master["date"].min().date()
        latest_date = master["date"].max().date()
        unique_symbols = int(master["symbol"].nunique())
        master["date"] = master["date"].dt.strftime("%Y-%m-%d")

    _atomic_write_csv(master, Path(output_path))
    return MasterBuildResult(
        output_path=Path(output_path),
        total_rows=len(master),
        unique_symbols=unique_symbols,
        earliest_date=earliest_date,
        latest_date=latest_date,
        duplicate_count=duplicate_count,
        source_files=len(loaded_paths),
        errors=errors,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the legacy compatibility dataset from daily raw CSVs."""
    parser = argparse.ArgumentParser(
        description="Rebuild the legacy PSX market compatibility CSV"
    )
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = build_master_dataset()
    except Exception as exc:
        LOGGER.error("Master rebuild failed: %s", exc)
        return 1

    print(f"Master file: {result.output_path}")
    print(f"Source files: {result.source_files}")
    print(f"Total rows: {result.total_rows}")
    print(f"Unique symbols: {result.unique_symbols}")
    print(f"Earliest date: {result.earliest_date or 'N/A'}")
    print(f"Latest date: {result.latest_date or 'N/A'}")
    print(f"Duplicates handled: {result.duplicate_count}")
    for error in result.errors:
        LOGGER.warning(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
