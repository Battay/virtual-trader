"""Load and filter generated PSX CSV files without creating derived files."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from data_pipeline.src.config import RAW_CSV_DIR


MARKET_COLUMNS = (
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


@dataclass(frozen=True)
class DatasetLoadResult:
    """Combined local dataset, discovered files, and non-fatal read errors."""

    data: pd.DataFrame
    csv_paths: tuple[Path, ...]
    errors: tuple[str, ...]

    @property
    def file_count(self) -> int:
        """Return the number of generated CSV files discovered."""
        return len(self.csv_paths)


@dataclass(frozen=True)
class DatasetSummary:
    """Summary metrics for the combined local PSX dataset."""

    csv_files: int
    trading_dates: int
    unique_symbols: int
    total_rows: int
    earliest_date: date | None
    latest_date: date | None


def empty_market_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with the expected market-data columns."""
    data: dict[str, pd.Series] = {
        "symbol": pd.Series(dtype="string"),
        "date": pd.Series(dtype="datetime64[ns]"),
    }
    for column in MARKET_COLUMNS[2:-1]:
        data[column] = pd.Series(dtype="float64")
    data["volume"] = pd.Series(dtype="Int64")
    return pd.DataFrame(data)


def sort_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological copy without removing duplicate rows."""
    sorted_data = data.copy()
    if "date" in sorted_data.columns:
        sorted_data["date"] = pd.to_datetime(sorted_data["date"], errors="coerce")

    sort_columns = [
        column for column in ("date", "symbol") if column in sorted_data.columns
    ]
    if sort_columns:
        sorted_data = sorted_data.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )
    return sorted_data.reset_index(drop=True)


def _combine_csv_paths(
    csv_paths: Sequence[Path | str],
) -> tuple[pd.DataFrame | None, tuple[str, ...]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for returned_path in csv_paths:
        path = Path(returned_path)
        if not path.is_file():
            errors.append(f"Returned CSV path does not exist: {path}")
            continue
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

        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frames.append(frame)

    if not frames:
        return None, tuple(errors)
    return sort_market_data(pd.concat(frames, ignore_index=True)), tuple(errors)


def load_csv_preview(
    csv_paths: Sequence[Path | str],
) -> tuple[pd.DataFrame | None, tuple[str, ...]]:
    """Load only the CSV paths returned by a pipeline collection result."""
    return _combine_csv_paths(csv_paths)


def load_market_dataset(csv_dir: Path = RAW_CSV_DIR) -> DatasetLoadResult:
    """Combine all generated daily CSV files from ``csv_dir`` in memory."""
    directory = Path(csv_dir)
    csv_paths = (
        tuple(sorted(directory.glob("market_*.csv")))
        if directory.is_dir()
        else ()
    )
    combined, errors = _combine_csv_paths(csv_paths)
    return DatasetLoadResult(
        data=combined if combined is not None else empty_market_dataframe(),
        csv_paths=csv_paths,
        errors=errors,
    )


def filter_market_data(
    data: pd.DataFrame,
    *,
    symbol: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Filter market rows by symbol and inclusive dates, then sort by date."""
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end date cannot be earlier than start date")

    filtered = data.copy()
    if symbol is not None:
        if "symbol" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered.loc[filtered["symbol"].astype("string") == str(symbol)]

    if start_date is not None or end_date is not None:
        if "date" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        parsed_dates = pd.to_datetime(filtered["date"], errors="coerce")
        filtered = filtered.assign(date=parsed_dates)
        if start_date is not None:
            filtered = filtered.loc[filtered["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            filtered = filtered.loc[filtered["date"] <= pd.Timestamp(end_date)]

    return sort_market_data(filtered)


def summarize_dataset(result: DatasetLoadResult) -> DatasetSummary:
    """Compute dashboard metrics for a loaded local dataset."""
    data = result.data
    dates = (
        pd.to_datetime(data["date"], errors="coerce").dropna()
        if "date" in data.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    symbols = (
        data["symbol"].astype("string").dropna()
        if "symbol" in data.columns
        else pd.Series(dtype="string")
    )
    symbols = symbols.loc[symbols.str.strip() != ""]

    return DatasetSummary(
        csv_files=result.file_count,
        trading_dates=int(dates.nunique()),
        unique_symbols=int(symbols.nunique()),
        total_rows=len(data),
        earliest_date=dates.min().date() if not dates.empty else None,
        latest_date=dates.max().date() if not dates.empty else None,
    )
