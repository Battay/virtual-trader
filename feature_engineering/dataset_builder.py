"""Build atomic per-symbol and combined PSX AI feature datasets."""

import argparse
from collections.abc import Collection, Sequence
from datetime import date
import json
import logging
from pathlib import Path

import pandas as pd

from data_pipeline.src.config import (
    AI_MINIMUM_USABLE_ROWS,
    COMPANY_REGISTRY_PATH,
    MASTER_CSV_PATH,
    PROCESSED_MASTER_PATH,
    PROCESSED_SYMBOLS_DIR,
    INDICES_MASTER_PATH,
)
from market_intelligence.feature_joiner import join_market_context

from .indicators import calculate_features
from .preprocessing import (
    DataQualityError,
    attach_registry_metadata,
    fatal_quality_errors_by_symbol,
    validate_required_market_columns,
)
from .schemas import (
    AI_DATASET_COLUMNS,
    DEFAULT_MASTER_SECURITY_TYPES,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    DatasetBuildMetrics,
    DatasetValidationResult,
)
from .storage import atomic_write_dataframe, safe_path_component


LOGGER = logging.getLogger(__name__)


def load_ai_sources(
    *,
    master_path: Path = MASTER_CSV_PATH,
    registry_path: Path = COMPANY_REGISTRY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load machine-friendly source data while preserving symbol strings."""
    master = pd.read_csv(Path(master_path), dtype={"symbol": "string"})
    registry = pd.read_csv(Path(registry_path), dtype={"symbol": "string"})
    return master, registry


def _prepare_feature_rows(
    market_data: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    index_data: pd.DataFrame | None = None,
    max_market_forward_fill_days: int = 0,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    validate_required_market_columns(market_data)
    quality_errors = fatal_quality_errors_by_symbol(market_data)
    fatal_symbols = set(quality_errors).difference({"<missing>"})
    clean = market_data.copy()
    clean["symbol"] = clean["symbol"].astype("string").str.strip()
    clean = clean.loc[
        clean["symbol"].notna()
        & (clean["symbol"] != "")
        & ~clean["symbol"].isin(fatal_symbols)
    ]
    featured = attach_registry_metadata(calculate_features(clean), registry)
    context_in = index_data if index_data is not None else pd.DataFrame()
    return join_market_context(
        featured,
        context_in,
        max_forward_fill_days=max_market_forward_fill_days,
    ), quality_errors


def _usable_rows(featured: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if featured.empty:
        return featured.reindex(columns=AI_DATASET_COLUMNS), 0, 0
    warmup = featured["is_warmup"].astype(bool)
    feature_missing = featured.loc[:, FEATURE_COLUMNS].isna().any(axis=1)
    warmup_rows = int(warmup.sum())
    missing_rows = int((~warmup & feature_missing).sum())
    usable = featured.loc[~warmup & ~feature_missing, AI_DATASET_COLUMNS].copy()
    usable["symbol"] = usable["symbol"].astype("string")
    usable["date"] = pd.to_datetime(usable["date"], errors="coerce")
    return usable.reset_index(drop=True), warmup_rows, missing_rows


def _metrics(
    *,
    input_rows: int,
    output: pd.DataFrame,
    skipped: Collection[str],
    warmup_rows: int,
    missing_rows: int,
    output_paths: Sequence[Path],
    market_context_included: bool = False,
) -> DatasetBuildMetrics:
    dates = pd.to_datetime(output.get("date"), errors="coerce").dropna()
    return DatasetBuildMetrics(
        input_rows=input_rows,
        output_rows=len(output),
        unique_symbols=int(output["symbol"].nunique()) if "symbol" in output else 0,
        symbols_skipped=tuple(sorted(set(str(symbol) for symbol in skipped))),
        warmup_rows_removed=warmup_rows,
        missing_rows=missing_rows,
        earliest_date=dates.min().date() if not dates.empty else None,
        latest_date=dates.max().date() if not dates.empty else None,
        feature_version=FEATURE_VERSION,
        output_paths=tuple(Path(path) for path in output_paths),
        market_context_included=market_context_included,
    )


def build_symbol_datasets(
    *,
    market_data: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    symbols: Collection[str] | None = None,
    minimum_usable_rows: int = AI_MINIMUM_USABLE_ROWS,
    output_dir: Path = PROCESSED_SYMBOLS_DIR,
    index_data: pd.DataFrame | None = None,
    include_market_context: bool = True,
    max_market_forward_fill_days: int = 0,
) -> DatasetBuildMetrics:
    """Build eligible active ordinary-equity datasets from one implementation."""
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")
    if market_data is None or registry is None:
        loaded_market, loaded_registry = load_ai_sources()
        market_data = loaded_market if market_data is None else market_data
        registry = loaded_registry if registry is None else registry

    requested = (
        {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
        if symbols is not None
        else None
    )
    source = market_data.copy()
    source["symbol"] = source["symbol"].astype("string").str.strip()
    if requested is not None:
        source = source.loc[source["symbol"].isin(requested)]

    if index_data is None and include_market_context and INDICES_MASTER_PATH.exists():
        index_data = pd.read_csv(INDICES_MASTER_PATH)
    context_included = bool(index_data is not None and not index_data.empty)
    featured, quality_errors = _prepare_feature_rows(
        source, registry, index_data=index_data if include_market_context else None,
        max_market_forward_fill_days=max_market_forward_fill_days,
    )
    available_symbols = set(featured["symbol"].astype("string"))
    candidates = requested if requested is not None else available_symbols
    skipped = set(quality_errors).difference({"<missing>"})
    skipped.update(candidates.difference(available_symbols))
    output_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    warmup_rows = 0
    missing_rows = 0

    for symbol in sorted(available_symbols):
        group = featured.loc[featured["symbol"] == symbol].copy()
        usable, symbol_warmup, symbol_missing = _usable_rows(group)
        warmup_rows += symbol_warmup
        missing_rows += symbol_missing
        is_active = bool(group["is_active"].iloc[-1])
        security_type = str(group["security_type"].iloc[-1])
        if (
            not is_active
            or security_type != "ordinary_equity"
            or len(usable) < minimum_usable_rows
        ):
            skipped.add(str(symbol))
            continue
        path = Path(output_dir) / f"{safe_path_component(str(symbol))}.csv"
        atomic_write_dataframe(usable, path)
        output_paths.append(path)
        output_frames.append(usable)

    output = (
        pd.concat(output_frames, ignore_index=True)
        if output_frames
        else pd.DataFrame(columns=AI_DATASET_COLUMNS)
    )
    return _metrics(
        input_rows=len(source),
        output=output,
        skipped=skipped,
        warmup_rows=warmup_rows,
        missing_rows=missing_rows,
        output_paths=output_paths,
        market_context_included=context_included,
    )


def build_master_ai_dataset(
    *,
    market_data: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    supported_security_types: Collection[str] = DEFAULT_MASTER_SECURITY_TYPES,
    output_path: Path = PROCESSED_MASTER_PATH,
    index_data: pd.DataFrame | None = None,
    include_market_context: bool = True,
    max_market_forward_fill_days: int = 0,
) -> DatasetBuildMetrics:
    """Build an all-lifecycle master feature dataset with symbol identity."""
    if market_data is None or registry is None:
        loaded_market, loaded_registry = load_ai_sources()
        market_data = loaded_market if market_data is None else market_data
        registry = loaded_registry if registry is None else registry

    if index_data is None and include_market_context and INDICES_MASTER_PATH.exists():
        index_data = pd.read_csv(INDICES_MASTER_PATH)
    context_included = bool(index_data is not None and not index_data.empty)
    featured, quality_errors = _prepare_feature_rows(
        market_data, registry, index_data=index_data if include_market_context else None,
        max_market_forward_fill_days=max_market_forward_fill_days,
    )
    all_symbols = set(market_data["symbol"].astype("string").str.strip().dropna())
    supported = set(str(value) for value in supported_security_types)
    supported_rows = featured.loc[featured["security_type"].isin(supported)].copy()
    unsupported_symbols = set(featured.loc[
        ~featured["security_type"].isin(supported), "symbol"
    ].astype("string"))
    usable, warmup_rows, missing_rows = _usable_rows(supported_rows)
    output_symbols = set(usable["symbol"].astype("string"))
    skipped = set(quality_errors).difference({"<missing>"})
    skipped.update(unsupported_symbols)
    skipped.update(all_symbols.difference(output_symbols))
    output = usable.sort_values(["date", "symbol"], kind="stable").reset_index(
        drop=True
    )
    atomic_write_dataframe(output, Path(output_path))
    return _metrics(
        input_rows=len(market_data),
        output=output,
        skipped=skipped,
        warmup_rows=warmup_rows,
        missing_rows=missing_rows,
        output_paths=(Path(output_path),),
        market_context_included=context_included,
    )


def validate_ai_dataset(path: Path) -> DatasetValidationResult:
    """Validate a generated AI dataset without modifying it."""
    errors: list[str] = []
    try:
        data = pd.read_csv(Path(path), dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return DatasetValidationResult(False, 0, 0, None, None, (str(exc),))
    missing = sorted(set(AI_DATASET_COLUMNS).difference(data.columns))
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    dates = (
        pd.to_datetime(data["date"], errors="coerce")
        if "date" in data
        else pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    )
    if len(data) and dates.isna().any():
        errors.append("Dataset contains invalid dates")
    if "symbol" in data and data["symbol"].astype("string").str.strip().eq("").any():
        errors.append("Dataset contains empty symbols")
    return DatasetValidationResult(
        valid=not errors,
        rows=len(data),
        unique_symbols=int(data["symbol"].nunique()) if "symbol" in data else 0,
        earliest_date=dates.min().date() if dates.notna().any() else None,
        latest_date=dates.max().date() if dates.notna().any() else None,
        errors=tuple(errors),
    )


def _print_metrics(label: str, metrics: DatasetBuildMetrics) -> None:
    print(f"{label}: {json.dumps(metrics.to_dict(), indent=2)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate generated AI datasets from the command line."""
    parser = argparse.ArgumentParser(description="Build PSX AI-ready datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    symbol_parser = subparsers.add_parser("symbols", help="Build symbol datasets")
    symbol_parser.add_argument("--symbols", nargs="*")
    symbol_parser.add_argument(
        "--minimum-history",
        type=int,
        default=AI_MINIMUM_USABLE_ROWS,
    )
    subparsers.add_parser("master", help="Build the combined master AI dataset")
    all_parser = subparsers.add_parser("all", help="Build symbol and master datasets")
    all_parser.add_argument(
        "--minimum-history",
        type=int,
        default=AI_MINIMUM_USABLE_ROWS,
    )
    validate_parser = subparsers.add_parser("validate", help="Validate AI CSV files")
    validate_parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "symbols":
            _print_metrics(
                "Symbol datasets",
                build_symbol_datasets(
                    symbols=args.symbols,
                    minimum_usable_rows=args.minimum_history,
                ),
            )
        elif args.command == "master":
            _print_metrics("Master dataset", build_master_ai_dataset())
        elif args.command == "all":
            _print_metrics(
                "Symbol datasets",
                build_symbol_datasets(minimum_usable_rows=args.minimum_history),
            )
            _print_metrics("Master dataset", build_master_ai_dataset())
        else:
            invalid = False
            for path in args.paths:
                result = validate_ai_dataset(path)
                print(f"{path}: {'valid' if result.valid else '; '.join(result.errors)}")
                invalid = invalid or not result.valid
            return 1 if invalid else 0
    except (DataQualityError, OSError, ValueError) as exc:
        LOGGER.error("AI dataset operation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
