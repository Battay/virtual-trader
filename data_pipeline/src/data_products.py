"""Ordered post-backfill rebuild of master, registry, AI, and readiness data."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering.dataset_builder import (
    build_master_ai_dataset,
    build_symbol_datasets,
)
from feature_engineering.readiness import build_training_readiness_report
from feature_engineering.splitting import create_master_split, create_symbol_split

from .company_registry import build_company_registry
from .config import (
    AI_MINIMUM_USABLE_ROWS,
    COMPANY_REGISTRY_PATH,
    MASTER_CSV_PATH,
    PROCESSED_MASTER_PATH,
    PROCESSED_SYMBOLS_DIR,
)
from .csv_store import build_master_dataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitRefreshResult:
    """Chronological split refresh counts and non-fatal errors."""

    split_sets: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DataProductsRebuildResult:
    """Structured summary of the ordered post-backfill rebuild."""

    raw_dates_added: int
    master_rows: int
    master_symbols: int
    processed_rows: int
    processed_symbols: int
    symbols_ready_for_training: int
    insufficient_history_symbols: int
    split_sets: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible summary."""
        values = asdict(self)
        values["errors"] = list(self.errors)
        return values


def refresh_chronological_splits(
    *,
    processed_master_path: Path = PROCESSED_MASTER_PATH,
    processed_symbols_dir: Path = PROCESSED_SYMBOLS_DIR,
) -> SplitRefreshResult:
    """Refresh split/scaler artifacts for every non-empty processed dataset."""
    created = 0
    errors: list[str] = []
    master_path = Path(processed_master_path)
    if master_path.is_file():
        try:
            master = pd.read_csv(master_path, dtype={"symbol": "string"})
            if not master.empty:
                create_master_split(processed_master_path=master_path)
                created += 1
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"Master split refresh failed: {exc}")

    for path in sorted(Path(processed_symbols_dir).glob("*.csv")):
        try:
            data = pd.read_csv(path, dtype={"symbol": "string"})
            if data.empty:
                continue
            create_symbol_split(
                path.stem,
                processed_symbols_dir=Path(processed_symbols_dir),
            )
            created += 1
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"Symbol split refresh failed for {path.stem}: {exc}")
    return SplitRefreshResult(created, tuple(errors))


def _default_readiness_builder(
    minimum_usable_rows: int,
) -> pd.DataFrame:
    market = pd.read_csv(MASTER_CSV_PATH, dtype={"symbol": "string"})
    registry = pd.read_csv(COMPANY_REGISTRY_PATH, dtype={"symbol": "string"})
    return build_training_readiness_report(
        market,
        registry,
        minimum_usable_rows=minimum_usable_rows,
    )


def rebuild_data_products(
    *,
    raw_dates_added: int = 0,
    minimum_usable_rows: int = AI_MINIMUM_USABLE_ROWS,
    master_builder: Callable[[], Any] = build_master_dataset,
    registry_builder: Callable[[], Any] = build_company_registry,
    master_ai_builder: Callable[[], Any] = build_master_ai_dataset,
    symbol_ai_builder: Callable[..., Any] = build_symbol_datasets,
    split_refresher: Callable[[], SplitRefreshResult] = refresh_chronological_splits,
    readiness_builder: Callable[[int], pd.DataFrame] = _default_readiness_builder,
) -> DataProductsRebuildResult:
    """Run all post-backfill products in dependency order without training."""
    if raw_dates_added < 0:
        raise ValueError("raw dates added cannot be negative")
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")

    errors: list[str] = []
    master_rows = master_symbols = processed_rows = processed_symbols = 0
    ready_symbols = insufficient_symbols = split_sets = 0

    try:
        master_result = master_builder()
        master_rows = int(getattr(master_result, "total_rows", 0))
        master_symbols = int(getattr(master_result, "unique_symbols", 0))
        errors.extend(str(value) for value in getattr(master_result, "errors", ()))
    except Exception as exc:
        errors.append(f"Master dataset rebuild failed: {exc}")

    try:
        registry_builder()
    except Exception as exc:
        errors.append(f"Company registry rebuild failed: {exc}")

    try:
        master_metrics = master_ai_builder()
        processed_rows = int(getattr(master_metrics, "output_rows", 0))
    except Exception as exc:
        errors.append(f"Master AI dataset build failed: {exc}")

    try:
        symbol_metrics = symbol_ai_builder(
            minimum_usable_rows=minimum_usable_rows
        )
        processed_symbols = int(getattr(symbol_metrics, "unique_symbols", 0))
    except Exception as exc:
        errors.append(f"Symbol AI dataset build failed: {exc}")

    try:
        split_result = split_refresher()
        split_sets = split_result.split_sets
        errors.extend(split_result.errors)
    except Exception as exc:
        errors.append(f"Chronological split refresh failed: {exc}")

    try:
        readiness = readiness_builder(minimum_usable_rows)
        ready_symbols = int(readiness["readiness_status"].eq("Ready").sum())
        insufficient_symbols = int(
            readiness["readiness_status"].eq("Insufficient History").sum()
        )
    except Exception as exc:
        errors.append(f"Training-readiness refresh failed: {exc}")

    return DataProductsRebuildResult(
        raw_dates_added=raw_dates_added,
        master_rows=master_rows,
        master_symbols=master_symbols,
        processed_rows=processed_rows,
        processed_symbols=processed_symbols,
        symbols_ready_for_training=ready_symbols,
        insufficient_history_symbols=insufficient_symbols,
        split_sets=split_sets,
        errors=tuple(errors),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit post-backfill rebuild command."""
    parser = argparse.ArgumentParser(description="Rebuild PSX data products")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Rebuild master, registry, AI datasets, splits, and readiness",
    )
    rebuild_parser.add_argument("--raw-dates-added", type=int, default=0)
    rebuild_parser.add_argument(
        "--minimum-history",
        type=int,
        default=AI_MINIMUM_USABLE_ROWS,
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = rebuild_data_products(
            raw_dates_added=args.raw_dates_added,
            minimum_usable_rows=args.minimum_history,
        )
    except ValueError as exc:
        LOGGER.error("Data-product rebuild failed: %s", exc)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
