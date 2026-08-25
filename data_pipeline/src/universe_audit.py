"""Descriptive Phase 1 universe audit over consolidated PSX Parquet data.

This module constructs no eligibility decision.  It reads market history via
the Milestone 7A boundary, optionally tags rows from the current local company
registry, and reports per-symbol evidence for later research design.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .config import COMPANY_REGISTRY_PATH
from .parquet_market_data import (
    MarketParquetError,
    load_market_data,
    resolve_market_parquet_path,
)


UNIVERSE_AUDIT_COLUMNS = (
    "symbol",
    "company_name",
    "sector",
    "metadata_matched",
    "first_market_date",
    "last_market_date",
    "observation_count",
    "calendar_span_days",
    "unique_market_dates",
    "coverage_ratio",
    "average_volume",
    "median_volume",
    "zero_volume_count",
    "zero_volume_ratio",
    "zero_open_count",
    "zero_high_count",
    "zero_low_count",
    "rows_with_any_zero_ohl",
    "zero_ohl_ratio",
    "positive_close_count",
    "positive_close_ratio",
)
QUANTILE_PROBABILITIES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
QUANTILE_LABELS = ("min", "p10", "p25", "p50", "p75", "p90", "max")
HISTORY_SPAN_THRESHOLDS_DAYS = {
    "at_least_1_year": 365,
    "at_least_2_years": 2 * 365,
    "at_least_3_years": 3 * 365,
    "at_least_5_years": 5 * 365,
}
NONTRIVIAL_MINIMUM_OBSERVATIONS = 2


class UniverseAuditError(RuntimeError):
    """Raised when a deterministic descriptive universe cannot be built."""


@dataclass(frozen=True)
class UniverseAuditSummary:
    """Dataset-level descriptive statistics with no eligibility thresholds."""

    total_historical_symbols: int
    global_unique_market_dates: int
    global_first_market_date: str | None
    global_last_market_date: str | None
    history_length_counts: Mapping[str, int]
    coverage_ratio_quantiles: Mapping[str, float | None]
    observation_count_quantiles: Mapping[str, float | None]
    average_volume_quantiles: Mapping[str, float | None]
    median_volume_quantiles: Mapping[str, float | None]
    zero_ohl_ratio_quantiles: Mapping[str, float | None]
    symbols_with_sector_metadata: int
    symbols_without_sector_metadata: int
    symbols_matched_to_registry: int
    symbols_unmatched_to_registry: int
    top_symbols_by_observation_count: tuple[Mapping[str, object], ...] = field(
        default_factory=tuple
    )
    bottom_nontrivial_symbols_by_observation_count: tuple[
        Mapping[str, object], ...
    ] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "history_length_counts",
            "coverage_ratio_quantiles",
            "observation_count_quantiles",
            "average_volume_quantiles",
            "median_volume_quantiles",
            "zero_ohl_ratio_quantiles",
        ):
            payload[name] = dict(getattr(self, name))
        payload["top_symbols_by_observation_count"] = [
            dict(row) for row in self.top_symbols_by_observation_count
        ]
        payload["bottom_nontrivial_symbols_by_observation_count"] = [
            dict(row) for row in self.bottom_nontrivial_symbols_by_observation_count
        ]
        return payload


@dataclass(frozen=True)
class UniverseAuditResult:
    """Per-symbol table plus its reconciled descriptive summary."""

    symbols: pd.DataFrame = field(repr=False, compare=False)
    summary: UniverseAuditSummary
    parquet_path: Path
    registry_path: Path | None


def _clean_optional_text(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    return cleaned.mask(cleaned == "", pd.NA)


def load_company_metadata(
    path: str | os.PathLike[str] | None = COMPANY_REGISTRY_PATH,
) -> pd.DataFrame | None:
    """Load only one-to-one symbol/company/sector tags when locally available."""

    if path is None:
        return None
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        return None
    try:
        registry = pd.read_csv(resolved, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise UniverseAuditError(f"Could not read company registry {resolved}: {exc}") from exc
    missing = [column for column in ("symbol", "company_name", "sector") if column not in registry]
    if missing:
        raise UniverseAuditError(
            "Company registry is missing metadata columns: " + ", ".join(missing)
        )
    metadata = registry.loc[:, ["symbol", "company_name", "sector"]].copy()
    metadata["symbol"] = _clean_optional_text(metadata["symbol"])
    if metadata["symbol"].isna().any():
        raise UniverseAuditError("Company registry contains a blank symbol")
    duplicates = metadata.loc[metadata["symbol"].duplicated(keep=False), "symbol"]
    if not duplicates.empty:
        raise UniverseAuditError(
            "Company registry contains duplicate symbols: "
            + ", ".join(sorted(duplicates.astype(str).unique()))
        )
    metadata["company_name"] = _clean_optional_text(metadata["company_name"])
    metadata["sector"] = _clean_optional_text(metadata["sector"])
    return metadata.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _validate_market_frame(market: pd.DataFrame) -> pd.DataFrame:
    required = ("market_date", "symbol", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in market]
    if missing:
        raise UniverseAuditError(
            "Market data is missing universe-audit columns: " + ", ".join(missing)
        )
    frame = market.loc[:, required].copy(deep=True)
    frame["symbol"] = _clean_optional_text(frame["symbol"])
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    if frame["symbol"].isna().any():
        raise UniverseAuditError("Market data contains a blank symbol")
    if frame["market_date"].isna().any():
        raise UniverseAuditError("Market data contains an invalid market_date")
    if frame.empty:
        raise UniverseAuditError("Market data is empty")
    return frame


def build_symbol_universe_audit(
    market: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute one descriptive row per symbol without excluding any symbol."""

    frame = _validate_market_frame(market)
    global_market_date_count = int(frame["market_date"].nunique())
    if global_market_date_count < 1:  # pragma: no cover - guarded above
        raise UniverseAuditError("Global market date set is empty")

    frame["zero_volume"] = frame["volume"].notna() & (frame["volume"] == 0)
    frame["zero_open"] = frame["open"].notna() & (frame["open"] == 0)
    frame["zero_high"] = frame["high"].notna() & (frame["high"] == 0)
    frame["zero_low"] = frame["low"].notna() & (frame["low"] == 0)
    frame["any_zero_ohl"] = frame[["zero_open", "zero_high", "zero_low"]].any(
        axis=1
    )
    frame["positive_close"] = frame["close"].notna() & (frame["close"] > 0)

    grouped = (
        frame.groupby("symbol", sort=True, observed=True, dropna=False)
        .agg(
            first_market_date=("market_date", "min"),
            last_market_date=("market_date", "max"),
            observation_count=("market_date", "size"),
            unique_market_dates=("market_date", "nunique"),
            average_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            zero_volume_count=("zero_volume", "sum"),
            zero_open_count=("zero_open", "sum"),
            zero_high_count=("zero_high", "sum"),
            zero_low_count=("zero_low", "sum"),
            rows_with_any_zero_ohl=("any_zero_ohl", "sum"),
            positive_close_count=("positive_close", "sum"),
        )
        .reset_index()
    )
    integer_columns = (
        "observation_count",
        "unique_market_dates",
        "zero_volume_count",
        "zero_open_count",
        "zero_high_count",
        "zero_low_count",
        "rows_with_any_zero_ohl",
        "positive_close_count",
    )
    for column in integer_columns:
        grouped[column] = grouped[column].astype("int64")
    grouped["calendar_span_days"] = (
        grouped["last_market_date"] - grouped["first_market_date"]
    ).dt.days.astype("int64")
    grouped["coverage_ratio"] = (
        grouped["unique_market_dates"] / global_market_date_count
    )
    grouped["zero_volume_ratio"] = (
        grouped["zero_volume_count"] / grouped["observation_count"]
    )
    grouped["zero_ohl_ratio"] = (
        grouped["rows_with_any_zero_ohl"] / grouped["observation_count"]
    )
    grouped["positive_close_ratio"] = (
        grouped["positive_close_count"] / grouped["observation_count"]
    )
    grouped["first_market_date"] = grouped["first_market_date"].dt.date
    grouped["last_market_date"] = grouped["last_market_date"].dt.date

    if metadata is None:
        grouped.insert(1, "company_name", pd.Series(pd.NA, index=grouped.index, dtype="string"))
        grouped.insert(2, "sector", pd.Series(pd.NA, index=grouped.index, dtype="string"))
        grouped.insert(3, "metadata_matched", False)
    else:
        required_metadata = ("symbol", "company_name", "sector")
        missing = [column for column in required_metadata if column not in metadata]
        if missing:
            raise UniverseAuditError(
                "Metadata is missing columns: " + ", ".join(missing)
            )
        tags = metadata.loc[:, required_metadata].copy(deep=True)
        tags["symbol"] = _clean_optional_text(tags["symbol"])
        if tags["symbol"].isna().any() or tags["symbol"].duplicated().any():
            raise UniverseAuditError(
                "Metadata symbols must be non-blank and unique for a many-to-one join"
            )
        tags["company_name"] = _clean_optional_text(tags["company_name"])
        tags["sector"] = _clean_optional_text(tags["sector"])
        tags["metadata_matched"] = True
        grouped = grouped.merge(
            tags,
            on="symbol",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        grouped["metadata_matched"] = grouped["metadata_matched"].fillna(False).astype(bool)

    return grouped.loc[:, UNIVERSE_AUDIT_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {label: None for label in QUANTILE_LABELS}
    measured = numeric.quantile(QUANTILE_PROBABILITIES, interpolation="linear")
    return {
        label: float(measured.iloc[index])
        for index, label in enumerate(QUANTILE_LABELS)
    }


def _symbol_rows(frame: pd.DataFrame) -> tuple[Mapping[str, object], ...]:
    columns = ("symbol", "observation_count", "first_market_date", "last_market_date")
    return tuple(frame.loc[:, columns].to_dict(orient="records"))


def summarize_symbol_universe(universe: pd.DataFrame) -> UniverseAuditSummary:
    """Summarize the complete descriptive table without selecting members."""

    missing = [column for column in UNIVERSE_AUDIT_COLUMNS if column not in universe]
    if missing:
        raise UniverseAuditError(
            "Universe table is missing summary columns: " + ", ".join(missing)
        )
    ordered = universe.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    first_dates = pd.to_datetime(ordered["first_market_date"], errors="coerce")
    last_dates = pd.to_datetime(ordered["last_market_date"], errors="coerce")
    top = ordered.sort_values(
        ["observation_count", "symbol"], ascending=[False, True], kind="mergesort"
    ).head(20)
    bottom = ordered.loc[
        ordered["observation_count"] >= NONTRIVIAL_MINIMUM_OBSERVATIONS
    ].sort_values(
        ["observation_count", "symbol"], ascending=[True, True], kind="mergesort"
    ).head(20)
    sector_available = _clean_optional_text(ordered["sector"]).notna()
    metadata_matched = ordered["metadata_matched"].fillna(False).astype(bool)
    history_counts = {
        label: int((ordered["calendar_span_days"] >= days).sum())
        for label, days in HISTORY_SPAN_THRESHOLDS_DAYS.items()
    }
    return UniverseAuditSummary(
        total_historical_symbols=len(ordered),
        global_unique_market_dates=int(
            round(
                (ordered["unique_market_dates"] / ordered["coverage_ratio"])
                .replace([float("inf"), -float("inf")], pd.NA)
                .dropna()
                .median()
            )
        ),
        global_first_market_date=(
            first_dates.min().date().isoformat() if not first_dates.empty else None
        ),
        global_last_market_date=(
            last_dates.max().date().isoformat() if not last_dates.empty else None
        ),
        history_length_counts=history_counts,
        coverage_ratio_quantiles=_quantiles(ordered["coverage_ratio"]),
        observation_count_quantiles=_quantiles(ordered["observation_count"]),
        average_volume_quantiles=_quantiles(ordered["average_volume"]),
        median_volume_quantiles=_quantiles(ordered["median_volume"]),
        zero_ohl_ratio_quantiles=_quantiles(ordered["zero_ohl_ratio"]),
        symbols_with_sector_metadata=int(sector_available.sum()),
        symbols_without_sector_metadata=int((~sector_available).sum()),
        symbols_matched_to_registry=int(metadata_matched.sum()),
        symbols_unmatched_to_registry=int((~metadata_matched).sum()),
        top_symbols_by_observation_count=_symbol_rows(top),
        bottom_nontrivial_symbols_by_observation_count=_symbol_rows(bottom),
    )


def run_universe_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] | None = COMPANY_REGISTRY_PATH,
) -> UniverseAuditResult:
    """Explicitly load the consolidated history and build its descriptive audit."""

    resolved_parquet = resolve_market_parquet_path(parquet_path)
    market = load_market_data(resolved_parquet)
    metadata = load_company_metadata(registry_path)
    symbols = build_symbol_universe_audit(market, metadata=metadata)
    return UniverseAuditResult(
        symbols=symbols,
        summary=summarize_symbol_universe(symbols),
        parquet_path=resolved_parquet,
        registry_path=(
            Path(registry_path).expanduser().resolve(strict=False)
            if registry_path is not None and Path(registry_path).expanduser().is_file()
            else None
        ),
    )


def write_universe_audit_csv(
    universe: pd.DataFrame,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    source_parquet_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write deterministic CSV only when explicitly requested by the caller."""

    output = Path(output_path).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".csv":
        raise UniverseAuditError("Universe audit output must use a .csv suffix")
    if source_parquet_path is not None and output == Path(source_parquet_path).resolve(
        strict=False
    ):
        raise UniverseAuditError("Audit output cannot target the source Parquet file")
    if not output.parent.is_dir():
        raise UniverseAuditError(f"Output directory does not exist: {output.parent}")
    if output.exists() and not overwrite:
        raise UniverseAuditError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )
    ordered = universe.loc[:, UNIVERSE_AUDIT_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    )
    mode = "w" if overwrite else "x"
    ordered.to_csv(
        output,
        index=False,
        mode=mode,
        lineterminator="\n",
        na_rep="",
        float_format="%.12g",
        date_format="%Y-%m-%d",
    )
    return output


def _print_quantiles(label: str, values: Mapping[str, float | None]) -> None:
    print(f"{label}: {json.dumps(dict(values), sort_keys=False)}")


def _print_symbol_list(label: str, rows: Sequence[Mapping[str, object]]) -> None:
    print(label + ":")
    for row in rows:
        print(
            f"  {row['symbol']}: {int(row['observation_count']):,} observations "
            f"({row['first_market_date']} to {row['last_market_date']})"
        )


def _print_summary(result: UniverseAuditResult) -> None:
    summary = result.summary
    print(f"Parquet: {result.parquet_path}")
    print(f"Company registry: {result.registry_path or 'not available'}")
    print(f"Total historical symbols: {summary.total_historical_symbols:,}")
    print(f"Global market dates: {summary.global_unique_market_dates:,}")
    print(
        f"Global date range: {summary.global_first_market_date} to "
        f"{summary.global_last_market_date}"
    )
    print("History-length counts: " + json.dumps(dict(summary.history_length_counts)))
    _print_quantiles("Observation-count quantiles", summary.observation_count_quantiles)
    _print_quantiles("Coverage-ratio quantiles", summary.coverage_ratio_quantiles)
    _print_quantiles("Average-volume quantiles", summary.average_volume_quantiles)
    _print_quantiles("Median-volume quantiles", summary.median_volume_quantiles)
    _print_quantiles("Zero-OHL-ratio quantiles", summary.zero_ohl_ratio_quantiles)
    print(f"Symbols with sector metadata: {summary.symbols_with_sector_metadata:,}")
    print(f"Symbols without sector metadata: {summary.symbols_without_sector_metadata:,}")
    print(f"Symbols matched to registry: {summary.symbols_matched_to_registry:,}")
    print(f"Symbols unmatched to registry: {summary.symbols_unmatched_to_registry:,}")
    _print_symbol_list(
        "Top 20 symbols by observation count", summary.top_symbols_by_observation_count
    )
    _print_symbol_list(
        f"Bottom 20 symbols with at least {NONTRIVIAL_MINIMUM_OBSERVATIONS} observations",
        summary.bottom_nontrivial_symbols_by_observation_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a descriptive, non-filtering PSX clustering-universe audit."
    )
    parser.add_argument("--path", help="Override the consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry",
        default=str(COMPANY_REGISTRY_PATH),
        help="Optional local company registry CSV; missing paths are treated as unavailable.",
    )
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--output-csv", help="Explicit deterministic CSV output path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an explicitly requested audit CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.overwrite and not args.output_csv:
        parser.error("--overwrite requires --output-csv")
    try:
        result = run_universe_audit(
            parquet_path=args.path,
            registry_path=None if args.no_metadata else args.company_registry,
        )
        _print_summary(result)
        if args.output_csv:
            output = write_universe_audit_csv(
                result.symbols,
                args.output_csv,
                overwrite=args.overwrite,
                source_parquet_path=result.parquet_path,
            )
            print(f"Audit CSV: {output}")
        return 0
    except (MarketParquetError, UniverseAuditError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())


__all__ = (
    "HISTORY_SPAN_THRESHOLDS_DAYS",
    "NONTRIVIAL_MINIMUM_OBSERVATIONS",
    "QUANTILE_LABELS",
    "UNIVERSE_AUDIT_COLUMNS",
    "UniverseAuditError",
    "UniverseAuditResult",
    "UniverseAuditSummary",
    "build_symbol_universe_audit",
    "load_company_metadata",
    "main",
    "run_universe_audit",
    "summarize_symbol_universe",
    "write_universe_audit_csv",
)
