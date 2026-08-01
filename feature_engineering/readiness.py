"""Reusable AI dataset-readiness calculations for symbols and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from data_pipeline.src.config import AI_MINIMUM_USABLE_ROWS, PROCESSED_SYMBOLS_DIR

from .dataset_builder import validate_ai_dataset
from .indicators import calculate_features
from .preprocessing import fatal_quality_errors_by_symbol, validate_required_market_columns
from .schemas import FEATURE_COLUMNS, DatasetBuildMetrics
from .splitting import chronological_split
from .storage import safe_path_component


READINESS_STATUSES = (
    "Ready",
    "Insufficient History",
    "Data Quality Issue",
    "Unsupported Security Type",
    "Missing Processed Dataset",
)
READINESS_COLUMNS = (
    "symbol",
    "company_name",
    "raw_trading_rows",
    "earliest_raw_date",
    "latest_raw_date",
    "warmup_rows_removed",
    "usable_feature_rows",
    "minimum_usable_rows",
    "additional_rows_required",
    "train_rows",
    "validation_rows",
    "test_rows",
    "readiness_status",
)


@dataclass(frozen=True)
class SymbolBuildReadiness:
    """Notebook-friendly readiness summary from one symbol build."""

    symbol: str
    raw_history_rows: int
    warmup_rows_removed: int
    usable_rows: int
    minimum_usable_rows: int
    additional_usable_rows_required: int
    earliest_available_date: date | None
    latest_available_date: date | None
    processed_path: Path
    is_training_ready: bool

    def to_display_frame(self) -> pd.DataFrame:
        """Return human-readable readiness measurements."""
        return pd.DataFrame(
            [
                ("Raw history rows", self.raw_history_rows),
                ("Warm-up rows removed", self.warmup_rows_removed),
                ("Usable rows", self.usable_rows),
                ("Configured minimum usable rows", self.minimum_usable_rows),
                (
                    "Additional usable rows required",
                    self.additional_usable_rows_required,
                ),
                (
                    "Earliest available date",
                    self.earliest_available_date.isoformat()
                    if self.earliest_available_date
                    else "—",
                ),
                (
                    "Latest available date",
                    self.latest_available_date.isoformat()
                    if self.latest_available_date
                    else "—",
                ),
            ],
            columns=["Readiness measure", "Value"],
        )


def additional_required_rows(usable_rows: int, minimum_usable_rows: int) -> int:
    """Return the remaining usable-row gap without allowing negative values."""
    if usable_rows < 0:
        raise ValueError("usable rows cannot be negative")
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")
    return max(0, minimum_usable_rows - usable_rows)


def summarize_symbol_build_readiness(
    *,
    symbol: str,
    raw_history: pd.DataFrame,
    metrics: DatasetBuildMetrics,
    minimum_usable_rows: int,
    processed_path: Path,
) -> SymbolBuildReadiness:
    """Interpret symbol-builder metrics without treating a skip as an error."""
    dates = (
        pd.to_datetime(raw_history["date"], errors="coerce").dropna()
        if "date" in raw_history
        else pd.Series(dtype="datetime64[ns]")
    )
    usable_rows = max(
        0,
        metrics.input_rows
        - metrics.warmup_rows_removed
        - metrics.missing_rows,
    )
    output_paths = {Path(path) for path in metrics.output_paths}
    path = Path(processed_path)
    ready = (
        symbol not in metrics.symbols_skipped
        and path in output_paths
        and path.is_file()
        and usable_rows >= minimum_usable_rows
    )
    return SymbolBuildReadiness(
        symbol=symbol,
        raw_history_rows=metrics.input_rows,
        warmup_rows_removed=metrics.warmup_rows_removed,
        usable_rows=usable_rows,
        minimum_usable_rows=minimum_usable_rows,
        additional_usable_rows_required=additional_required_rows(
            usable_rows,
            minimum_usable_rows,
        ),
        earliest_available_date=dates.min().date() if not dates.empty else None,
        latest_available_date=dates.max().date() if not dates.empty else None,
        processed_path=path,
        is_training_ready=ready,
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="string"),
            "company_name": pd.Series(dtype="string"),
            "raw_trading_rows": pd.Series(dtype="int64"),
            "earliest_raw_date": pd.Series(dtype="object"),
            "latest_raw_date": pd.Series(dtype="object"),
            "warmup_rows_removed": pd.Series(dtype="int64"),
            "usable_feature_rows": pd.Series(dtype="int64"),
            "minimum_usable_rows": pd.Series(dtype="int64"),
            "additional_rows_required": pd.Series(dtype="int64"),
            "train_rows": pd.Series(dtype="int64"),
            "validation_rows": pd.Series(dtype="int64"),
            "test_rows": pd.Series(dtype="int64"),
            "readiness_status": pd.Series(dtype="string"),
        }
    )


def build_training_readiness_report(
    market_data: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    minimum_usable_rows: int = AI_MINIMUM_USABLE_ROWS,
    processed_symbols_dir: Path = PROCESSED_SYMBOLS_DIR,
) -> pd.DataFrame:
    """Return exact feature and split readiness for every active symbol."""
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")
    validate_required_market_columns(market_data)
    required_registry = {
        "symbol",
        "company_name",
        "officially_listed",
        "activity_status",
        "security_type",
    }
    missing_registry = sorted(required_registry.difference(registry.columns))
    if missing_registry:
        raise ValueError(
            f"Registry is missing readiness columns: {', '.join(missing_registry)}"
        )

    market = market_data.copy()
    market["symbol"] = market["symbol"].astype("string").str.strip()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    fatal_errors = fatal_quality_errors_by_symbol(market)
    fatal_symbols = set(fatal_errors).difference({"<missing>"})
    feature_source = market.loc[
        market["symbol"].notna()
        & (market["symbol"] != "")
        & ~market["symbol"].isin(fatal_symbols)
    ]
    featured = calculate_features(feature_source)
    featured_by_symbol = {
        str(symbol): group
        for symbol, group in featured.groupby("symbol", sort=False)
    }
    records: list[dict[str, object]] = []

    for registry_row in registry.to_dict(orient="records"):
        symbol = str(registry_row.get("symbol", "")).strip()
        active = _as_bool(registry_row.get("officially_listed")) and str(
            registry_row.get("activity_status", "")
        ) == "recently_traded"
        if not symbol or not active:
            continue

        raw = market.loc[market["symbol"] == symbol].sort_values(
            "date",
            kind="stable",
        )
        raw_dates = raw["date"].dropna()
        symbol_features = featured_by_symbol.get(symbol, pd.DataFrame())
        if symbol_features.empty:
            warmup_rows = 0
            usable_rows = 0
        else:
            warmup = symbol_features["is_warmup"].astype(bool)
            missing_features = symbol_features.loc[:, FEATURE_COLUMNS].isna().any(axis=1)
            warmup_rows = int(warmup.sum())
            usable_rows = int((~warmup & ~missing_features).sum())

        security_type = str(registry_row.get("security_type", "unknown"))
        processed_path = Path(processed_symbols_dir) / (
            f"{safe_path_component(symbol)}.csv"
        )
        train_rows = validation_rows = test_rows = 0
        if symbol in fatal_symbols:
            status = "Data Quality Issue"
        elif security_type != "ordinary_equity":
            status = "Unsupported Security Type"
        elif usable_rows < minimum_usable_rows:
            status = "Insufficient History"
        elif not processed_path.is_file():
            status = "Missing Processed Dataset"
        else:
            validation = validate_ai_dataset(processed_path)
            try:
                processed = pd.read_csv(
                    processed_path,
                    dtype={"symbol": "string"},
                )
                valid_symbol = (
                    not processed.empty
                    and set(processed["symbol"].astype("string")) == {symbol}
                )
                if not validation.valid or not valid_symbol:
                    raise ValueError("processed symbol dataset is inconsistent")
                split = chronological_split(processed, scope="symbol")
                train_rows = len(split.train)
                validation_rows = len(split.validation)
                test_rows = len(split.test)
                status = "Ready"
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                status = "Data Quality Issue"

        records.append(
            {
                "symbol": symbol,
                "company_name": str(registry_row.get("company_name", "")),
                "raw_trading_rows": len(raw),
                "earliest_raw_date": (
                    raw_dates.min().date() if not raw_dates.empty else None
                ),
                "latest_raw_date": (
                    raw_dates.max().date() if not raw_dates.empty else None
                ),
                "warmup_rows_removed": warmup_rows,
                "usable_feature_rows": usable_rows,
                "minimum_usable_rows": minimum_usable_rows,
                "additional_rows_required": additional_required_rows(
                    usable_rows,
                    minimum_usable_rows,
                ),
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
                "readiness_status": status,
            }
        )

    if not records:
        return _empty_report()
    return (
        pd.DataFrame.from_records(records, columns=READINESS_COLUMNS)
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )
