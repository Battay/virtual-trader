"""Load and filter generated PSX CSV files without creating derived files."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from data_pipeline.src.config import MASTER_CSV_PATH, RAW_CSV_DIR


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
COMPANY_SUMMARY_COLUMNS = (
    "symbol",
    "company_name",
    "security_type",
    "sector",
    "official_status",
    "activity_status",
    "lifecycle_status",
    "first_seen_date",
    "last_seen_date",
    "trading_days",
    "latest_close",
    "officially_listed",
    "board",
    "listing_segment",
)
_COMPANY_METADATA_COLUMNS = (
    "company_name",
    "security_type",
    "sector",
    "official_status",
    "activity_status",
    "lifecycle_status",
    "officially_listed",
    "board",
    "listing_segment",
)
STOCK_HISTORY_PERIODS = ("1M", "3M", "6M", "1Y", "All")
_STOCK_HISTORY_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


@dataclass(frozen=True)
class DatasetLoadResult:
    """Combined local dataset, discovered files, and non-fatal read errors."""

    data: pd.DataFrame
    csv_paths: tuple[Path, ...]
    errors: tuple[str, ...]
    source: Literal["master", "raw"] = "raw"
    message: str = ""

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


@dataclass(frozen=True)
class PaginationState:
    """Resolved pagination metadata for a dataframe-backed table."""

    page_number: int
    rows_per_page: int
    total_rows: int
    total_pages: int
    start_row: int
    end_row: int


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
        source="raw",
        message="Using combined daily raw CSV files.",
    )


def load_dashboard_dataset(
    *,
    master_csv_path: Path = MASTER_CSV_PATH,
    raw_csv_dir: Path = RAW_CSV_DIR,
) -> DatasetLoadResult:
    """Load the master CSV when available, otherwise combine daily raw CSVs."""
    raw_directory = Path(raw_csv_dir)
    raw_paths = (
        tuple(sorted(raw_directory.glob("market_*.csv")))
        if raw_directory.is_dir()
        else ()
    )
    master_path = Path(master_csv_path)
    if master_path.is_file():
        master_data, master_errors = _combine_csv_paths((master_path,))
        if master_data is not None:
            missing_columns = [
                column for column in MARKET_COLUMNS if column not in master_data.columns
            ]
            if not missing_columns:
                return DatasetLoadResult(
                    data=master_data,
                    csv_paths=raw_paths,
                    errors=master_errors,
                    source="master",
                    message=f"Using persistent master dataset: {master_path}",
                )
            master_errors = (
                *master_errors,
                f"Master dataset is missing columns: {', '.join(missing_columns)}",
            )

        raw_result = load_market_dataset(raw_directory)
        return DatasetLoadResult(
            data=raw_result.data,
            csv_paths=raw_result.csv_paths,
            errors=(*master_errors, *raw_result.errors),
            source="raw",
            message=(
                "Master dataset could not be loaded; using combined daily raw CSVs."
            ),
        )

    raw_result = load_market_dataset(raw_directory)
    return DatasetLoadResult(
        data=raw_result.data,
        csv_paths=raw_result.csv_paths,
        errors=raw_result.errors,
        source="raw",
        message="Master dataset has not been built; using combined daily raw CSVs.",
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


def build_company_summary(
    market_data: pd.DataFrame,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate market history to one row per symbol and attach metadata."""
    if "symbol" not in market_data.columns:
        return pd.DataFrame(columns=COMPANY_SUMMARY_COLUMNS)

    market = market_data.copy()
    market["symbol"] = market["symbol"].astype("string").str.strip()
    market = market.loc[market["symbol"].notna() & (market["symbol"] != "")]
    if market.empty:
        return pd.DataFrame(columns=COMPANY_SUMMARY_COLUMNS)

    market["_summary_date"] = (
        pd.to_datetime(market["date"], errors="coerce")
        if "date" in market.columns
        else pd.NaT
    )
    market["_summary_close"] = (
        pd.to_numeric(market["close"], errors="coerce")
        if "close" in market.columns
        else float("nan")
    )
    company_summary = (
        market.groupby("symbol", sort=False, as_index=False)
        .agg(
            first_seen_date=("_summary_date", "min"),
            last_seen_date=("_summary_date", "max"),
            trading_days=("_summary_date", "nunique"),
        )
    )

    dated_rows = market.dropna(subset=["_summary_date"]).sort_values(
        ["symbol", "_summary_date"],
        kind="stable",
    )
    latest_closes = (
        dated_rows.groupby("symbol", sort=False, as_index=False)
        .tail(1)[["symbol", "_summary_close"]]
        .rename(columns={"_summary_close": "latest_close"})
    )
    company_summary = company_summary.merge(
        latest_closes,
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    if registry is not None and not registry.empty and "symbol" in registry:
        metadata_columns = [
            column for column in _COMPANY_METADATA_COLUMNS if column in registry
        ]
        metadata = registry[["symbol", *metadata_columns]].copy()
        metadata["symbol"] = metadata["symbol"].astype("string").str.strip()
        metadata = metadata.drop_duplicates("symbol", keep="last")
        company_summary = company_summary.merge(
            metadata,
            on="symbol",
            how="left",
            validate="one_to_one",
        )

    for column in _COMPANY_METADATA_COLUMNS:
        if column not in company_summary:
            company_summary[column] = pd.NA

    company_summary["trading_days"] = pd.to_numeric(
        company_summary["trading_days"],
        errors="coerce",
    ).astype("Int64")
    company_summary["latest_close"] = pd.to_numeric(
        company_summary["latest_close"],
        errors="coerce",
    )
    return company_summary.loc[:, COMPANY_SUMMARY_COLUMNS].sort_values(
        "symbol",
        kind="stable",
    ).reset_index(drop=True)


def filter_security_history(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return only one security's rows as a chronological, independent copy."""
    normalized_symbol = str(symbol).strip()
    if not normalized_symbol:
        raise ValueError("symbol cannot be empty")
    return filter_market_data(data, symbol=normalized_symbol)


def filter_history_period(
    data: pd.DataFrame,
    period: str,
    reference_date: date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return chronological history within ``period`` up to a reference date."""
    if period not in STOCK_HISTORY_PERIODS:
        raise ValueError(
            f"Unsupported stock history period {period!r}; "
            f"expected one of {', '.join(STOCK_HISTORY_PERIODS)}"
        )
    sorted_data = sort_market_data(data)
    if sorted_data.empty or "date" not in sorted_data:
        return sorted_data

    dates = pd.to_datetime(sorted_data["date"], errors="coerce")
    if not dates.notna().any():
        return sorted_data.iloc[0:0].copy()

    if reference_date is None:
        resolved_reference = dates.max()
    else:
        try:
            resolved_reference = pd.Timestamp(reference_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference date must be a valid date") from exc
        if pd.isna(resolved_reference):
            raise ValueError("reference date must be a valid date")

    within_reference = dates <= resolved_reference
    if period == "All":
        return sort_market_data(sorted_data.loc[within_reference])

    cutoff = resolved_reference - _STOCK_HISTORY_OFFSETS[period]
    return sort_market_data(
        sorted_data.loc[within_reference & (dates >= cutoff)]
    )


def filter_stock_history_period(
    data: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """Compatibility wrapper for filtering relative to the latest trading date."""
    return filter_history_period(data, period)


def sort_history_newest_first(data: pd.DataFrame) -> pd.DataFrame:
    """Return a stable newest-first copy of security history."""
    newest_first = data.copy()
    if "date" in newest_first.columns:
        newest_first["date"] = pd.to_datetime(
            newest_first["date"],
            errors="coerce",
        )
        newest_first = newest_first.sort_values(
            "date",
            ascending=False,
            kind="stable",
            na_position="last",
        )
    return newest_first.reset_index(drop=True)


def resolve_pagination(
    total_rows: int,
    page_number: int,
    rows_per_page: int,
) -> PaginationState:
    """Clamp a requested page and return display bounds for the table."""
    if isinstance(rows_per_page, bool) or not isinstance(rows_per_page, int):
        raise ValueError("rows per page must be a positive integer")
    if rows_per_page <= 0:
        raise ValueError("rows per page must be a positive integer")
    if total_rows < 0:
        raise ValueError("total rows cannot be negative")

    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    requested_page = (
        page_number
        if isinstance(page_number, int) and not isinstance(page_number, bool)
        else 1
    )
    resolved_page = min(max(requested_page, 1), total_pages)
    start_index = (resolved_page - 1) * rows_per_page
    end_index = min(start_index + rows_per_page, total_rows)
    start_row = start_index + 1 if total_rows else 0

    return PaginationState(
        page_number=resolved_page,
        rows_per_page=rows_per_page,
        total_rows=total_rows,
        total_pages=total_pages,
        start_row=start_row,
        end_row=end_index,
    )


def paginate_dataframe(
    data: pd.DataFrame,
    page_number: int,
    rows_per_page: int,
) -> pd.DataFrame:
    """Return a copied page, safely clamping the requested page number."""
    pagination = resolve_pagination(len(data), page_number, rows_per_page)
    start_index = (pagination.page_number - 1) * pagination.rows_per_page
    return data.iloc[start_index : pagination.end_row].copy().reset_index(drop=True)


def history_csv_bytes(data: pd.DataFrame) -> bytes:
    """Serialize every supplied history row without changing the dataframe."""
    export_data = data.copy()
    if "date" in export_data.columns:
        export_data["date"] = pd.to_datetime(
            export_data["date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    return export_data.to_csv(index=False).encode("utf-8")


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
