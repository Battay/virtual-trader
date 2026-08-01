"""Leakage-safe chronological splitting and training-only scaling."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from data_pipeline.src.config import (
    PROCESSED_MASTER_PATH,
    PROCESSED_SPLITS_DIR,
    PROCESSED_SYMBOLS_DIR,
)

from .preprocessing import DataQualityError, fit_training_scaler, save_scaler_artifact
from .schemas import FEATURE_VERSION
from .storage import atomic_write_dataframe, atomic_write_json, safe_path_component


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological train, validation, and test partitions with metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class SplitArtifactResult:
    """Paths and metadata produced when a split is persisted."""

    output_dir: Path
    metadata_path: Path
    scaler_path: Path
    metadata: dict[str, object]


def _validate_proportions(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) <= 0:
        raise ValueError("split proportions must all be positive")
    if abs((train + validation + test) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")


def _date_partition_counts(
    total_dates: int,
    train_proportion: float,
    validation_proportion: float,
) -> tuple[int, int, int]:
    if total_dates < 3:
        raise DataQualityError("at least three distinct dates are required to split")
    train_count = max(1, int(total_dates * train_proportion))
    validation_count = max(1, int(total_dates * validation_proportion))
    if train_count + validation_count >= total_dates:
        train_count = total_dates - 2
        validation_count = 1
    test_count = total_dates - train_count - validation_count
    return train_count, validation_count, test_count


def _partition_metadata(data: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    return {
        "start": dates.min().date().isoformat() if not dates.empty else None,
        "end": dates.max().date().isoformat() if not dates.empty else None,
        "rows": len(data),
        "symbols": sorted(set(data["symbol"].astype("string"))),
    }


def chronological_split(
    data: pd.DataFrame,
    *,
    scope: str,
    train_proportion: float = 0.70,
    validation_proportion: float = 0.15,
    test_proportion: float = 0.15,
) -> ChronologicalSplit:
    """Split by date boundaries so no trading date crosses a partition."""
    _validate_proportions(
        train_proportion,
        validation_proportion,
        test_proportion,
    )
    if scope not in {"symbol", "master"}:
        raise ValueError("scope must be 'symbol' or 'master'")
    required = {"symbol", "date"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataQualityError(
            f"AI dataset is missing split columns: {', '.join(missing)}"
        )
    if data.empty:
        raise DataQualityError("AI dataset is empty and cannot be split")

    sorted_data = data.copy()
    sorted_data["symbol"] = sorted_data["symbol"].astype("string")
    sorted_data["date"] = pd.to_datetime(sorted_data["date"], errors="coerce")
    if sorted_data["date"].isna().any():
        raise DataQualityError("AI dataset contains invalid dates")
    if scope == "symbol" and sorted_data["symbol"].nunique() != 1:
        raise DataQualityError("symbol split requires exactly one symbol")
    sorted_data = sorted_data.sort_values(
        ["date", "symbol"],
        kind="stable",
    ).reset_index(drop=True)
    dates = sorted_data["date"].drop_duplicates().sort_values().tolist()
    train_count, validation_count, _ = _date_partition_counts(
        len(dates),
        train_proportion,
        validation_proportion,
    )
    train_dates = set(dates[:train_count])
    validation_dates = set(dates[train_count : train_count + validation_count])
    test_dates = set(dates[train_count + validation_count :])
    train = sorted_data.loc[sorted_data["date"].isin(train_dates)].reset_index(drop=True)
    validation = sorted_data.loc[
        sorted_data["date"].isin(validation_dates)
    ].reset_index(drop=True)
    test = sorted_data.loc[sorted_data["date"].isin(test_dates)].reset_index(drop=True)
    metadata = {
        "scope": scope,
        "feature_version": FEATURE_VERSION,
        "proportions": {
            "training": train_proportion,
            "validation": validation_proportion,
            "testing": test_proportion,
        },
        "training": _partition_metadata(train),
        "validation": _partition_metadata(validation),
        "testing": _partition_metadata(test),
    }
    return ChronologicalSplit(train, validation, test, metadata)


def persist_split_artifacts(
    split: ChronologicalSplit,
    output_dir: Path,
) -> SplitArtifactResult:
    """Persist raw/scaled partitions, scaler, and metadata atomically."""
    directory = Path(output_dir)
    atomic_write_dataframe(split.train, directory / "train.csv")
    atomic_write_dataframe(split.validation, directory / "validation.csv")
    atomic_write_dataframe(split.test, directory / "test.csv")
    scaling = fit_training_scaler(split.train, split.validation, split.test)
    atomic_write_dataframe(scaling.train, directory / "train_scaled.csv")
    atomic_write_dataframe(scaling.validation, directory / "validation_scaled.csv")
    atomic_write_dataframe(scaling.test, directory / "test_scaled.csv")
    scaler_path, scaler_metadata_path = save_scaler_artifact(
        scaling,
        directory / "standard_scaler.joblib",
    )
    metadata = {
        **split.metadata,
        "scaled_features": list(scaling.scaled_features),
        "scaler_path": str(scaler_path),
        "scaler_metadata_path": str(scaler_metadata_path),
    }
    metadata_path = directory / "metadata.json"
    atomic_write_json(metadata, metadata_path)
    return SplitArtifactResult(directory, metadata_path, scaler_path, metadata)


def create_symbol_split(
    symbol: str,
    *,
    processed_symbols_dir: Path = PROCESSED_SYMBOLS_DIR,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> SplitArtifactResult:
    """Create split/scaler artifacts for one complete processed symbol history."""
    component = safe_path_component(symbol)
    path = Path(processed_symbols_dir) / f"{component}.csv"
    data = pd.read_csv(path, dtype={"symbol": "string"})
    split = chronological_split(data, scope="symbol")
    return persist_split_artifacts(split, Path(splits_dir) / "symbols" / component)


def create_master_split(
    *,
    processed_master_path: Path = PROCESSED_MASTER_PATH,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> SplitArtifactResult:
    """Create a global-date split and scaler for the complete master AI data."""
    data = pd.read_csv(Path(processed_master_path), dtype={"symbol": "string"})
    split = chronological_split(data, scope="master")
    return persist_split_artifacts(split, Path(splits_dir) / "master")


def main(argv: Sequence[str] | None = None) -> int:
    """Create chronological AI dataset splits from the command line."""
    parser = argparse.ArgumentParser(description="Split PSX AI-ready datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    symbol_parser = subparsers.add_parser("symbols", help="Split symbol datasets")
    symbol_parser.add_argument("--symbols", nargs="*")
    subparsers.add_parser("master", help="Split the master AI dataset")
    subparsers.add_parser("all", help="Split every processed dataset")
    args = parser.parse_args(argv)
    try:
        if args.command in {"symbols", "all"}:
            symbols = args.symbols if args.command == "symbols" else None
            paths = tuple(sorted(Path(PROCESSED_SYMBOLS_DIR).glob("*.csv")))
            selected = (
                {safe_path_component(symbol) for symbol in symbols}
                if symbols
                else None
            )
            for path in paths:
                if selected is not None and path.stem not in selected:
                    continue
                result = create_symbol_split(path.stem)
                print(json.dumps(result.metadata, indent=2))
        if args.command in {"master", "all"}:
            result = create_master_split()
            print(json.dumps(result.metadata, indent=2))
    except (DataQualityError, OSError, ValueError) as exc:
        print(f"Split operation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
