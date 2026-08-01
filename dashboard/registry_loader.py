"""Load and filter the persisted company registry for dashboard pages."""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from dashboard.presentation import status_label
from data_pipeline.src.company_registry import REGISTRY_COLUMNS
from data_pipeline.src.config import COMPANY_REGISTRY_PATH


@dataclass(frozen=True)
class RegistryLoadResult:
    """Registry dataframe plus non-fatal loading information."""

    data: pd.DataFrame
    path: Path
    errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Return whether a valid, non-empty registry is available."""
        return not self.data.empty and not self.errors


@dataclass(frozen=True)
class RegistryDisplayMetrics:
    """Unambiguous registry counts for dashboard presentation."""

    total_securities: int
    currently_listed: int
    listed_and_recently_traded: int
    historical_only: int
    listed_not_recently_traded: int
    new_listings: int
    suspended: int
    non_compliant: int
    officially_delisted: int
    unknown: int


def empty_registry_dataframe() -> pd.DataFrame:
    """Return an empty registry with stable expected columns."""
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _parse_boolean(values: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.loc[~normalized.isin({"true", "false", "1", "0"})]
    if not invalid.empty:
        raise ValueError(f"Registry contains invalid {field} values")
    return normalized.isin({"true", "1"})


def load_company_registry(
    path: Path = COMPANY_REGISTRY_PATH,
) -> RegistryLoadResult:
    """Load the local company registry without making network requests."""
    registry_path = Path(path)
    if not registry_path.is_file():
        return RegistryLoadResult(
            data=empty_registry_dataframe(),
            path=registry_path,
            errors=(),
        )
    try:
        data = pd.read_csv(registry_path, dtype={"symbol": "string"})
        missing = [column for column in REGISTRY_COLUMNS if column not in data]
        if data.empty or missing:
            detail = "contains no rows" if data.empty else f"missing: {', '.join(missing)}"
            raise ValueError(f"Registry is invalid: {detail}")
        data["symbol"] = data["symbol"].astype("string").str.strip()
        for column in ("first_seen_date", "last_seen_date"):
            data[column] = pd.to_datetime(data[column], errors="coerce")
        for column in ("officially_listed", "is_new_listing", "cached_listings_used"):
            data[column] = _parse_boolean(data[column], column)
        for column in ("trading_days", "days_since_last_seen", "shares", "free_float"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.sort_values("symbol", kind="stable").reset_index(drop=True)
        return RegistryLoadResult(data=data, path=registry_path, errors=())
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return RegistryLoadResult(
            data=empty_registry_dataframe(),
            path=registry_path,
            errors=(f"Could not load company registry {registry_path}: {exc}",),
        )


def _apply_values_filter(
    data: pd.DataFrame,
    column: str,
    selected: Collection[str] | None,
) -> pd.DataFrame:
    if not selected:
        return data
    return data.loc[data[column].astype("string").isin(set(selected))]


def filter_company_registry(
    registry: pd.DataFrame,
    *,
    lifecycle_statuses: Collection[str] | None = None,
    official_statuses: Collection[str] | None = None,
    activity_statuses: Collection[str] | None = None,
    security_types: Collection[str] | None = None,
    sectors: Collection[str] | None = None,
    boards: Collection[str] | None = None,
    listing_segments: Collection[str] | None = None,
    search: str = "",
    officially_listed_only: bool = False,
    ordinary_equities_only: bool = False,
    first_seen_start: date | None = None,
    first_seen_end: date | None = None,
    last_seen_start: date | None = None,
    last_seen_end: date | None = None,
    minimum_trading_days: int | None = None,
) -> pd.DataFrame:
    """Apply inclusive registry filters and return symbol-sorted rows."""
    if (
        first_seen_start is not None
        and first_seen_end is not None
        and first_seen_end < first_seen_start
    ):
        raise ValueError("first-seen end date cannot be earlier than start date")
    if (
        last_seen_start is not None
        and last_seen_end is not None
        and last_seen_end < last_seen_start
    ):
        raise ValueError("last-seen end date cannot be earlier than start date")
    if minimum_trading_days is not None and minimum_trading_days < 0:
        raise ValueError("minimum trading days cannot be negative")

    filtered = registry.copy()
    for column, selected in (
        ("lifecycle_status", lifecycle_statuses),
        ("official_status", official_statuses),
        ("activity_status", activity_statuses),
        ("security_type", security_types),
        ("sector", sectors),
        ("board", boards),
        ("listing_segment", listing_segments),
    ):
        filtered = _apply_values_filter(filtered, column, selected)
    if officially_listed_only:
        listed = filtered["officially_listed"]
        if not pd.api.types.is_bool_dtype(listed):
            listed = listed.astype("string").str.lower().isin({"true", "1"})
        filtered = filtered.loc[listed]
    if ordinary_equities_only:
        filtered = filtered.loc[
            filtered["security_type"].astype("string") == "ordinary_equity"
        ]
    normalized_search = search.strip()
    if normalized_search:
        symbol_matches = filtered["symbol"].astype("string").str.contains(
            normalized_search,
            case=False,
            regex=False,
            na=False,
        )
        name_matches = filtered["company_name"].astype("string").str.contains(
            normalized_search,
            case=False,
            regex=False,
            na=False,
        )
        filtered = filtered.loc[symbol_matches | name_matches]

    for column, start, end in (
        ("first_seen_date", first_seen_start, first_seen_end),
        ("last_seen_date", last_seen_start, last_seen_end),
    ):
        if start is None and end is None:
            continue
        parsed = pd.to_datetime(filtered[column], errors="coerce")
        if start is not None:
            filtered = filtered.loc[parsed >= pd.Timestamp(start)]
            parsed = parsed.loc[filtered.index]
        if end is not None:
            filtered = filtered.loc[parsed <= pd.Timestamp(end)]

    if minimum_trading_days is not None:
        trading_days = pd.to_numeric(filtered["trading_days"], errors="coerce")
        filtered = filtered.loc[trading_days >= minimum_trading_days]

    return filtered.sort_values("symbol", kind="stable").reset_index(drop=True)


def restrict_market_data_by_registry(
    market_data: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    officially_listed_only: bool = False,
    ordinary_equities_only: bool = False,
    lifecycle_statuses: Collection[str] | None = None,
    security_types: Collection[str] | None = None,
) -> pd.DataFrame:
    """Restrict market rows to registry filters without changing price data."""
    if not any(
        (
            officially_listed_only,
            ordinary_equities_only,
            bool(lifecycle_statuses),
            bool(security_types),
        )
    ):
        return market_data.copy()
    eligible = filter_company_registry(
        registry,
        officially_listed_only=officially_listed_only,
        ordinary_equities_only=ordinary_equities_only,
        lifecycle_statuses=lifecycle_statuses,
        security_types=security_types,
    )
    symbols = set(eligible["symbol"].astype("string"))
    if "symbol" not in market_data:
        return market_data.iloc[0:0].copy()
    return market_data.loc[
        market_data["symbol"].astype("string").isin(symbols)
    ].copy()


def lifecycle_status_label(value: str) -> str:
    """Return a concise user-facing lifecycle badge label."""
    return status_label(value)


def summarize_registry_for_display(
    registry: pd.DataFrame,
) -> RegistryDisplayMetrics:
    """Calculate dashboard metrics without changing persisted registry meanings."""
    listed = registry["officially_listed"]
    if not pd.api.types.is_bool_dtype(listed):
        listed = listed.astype("string").str.lower().isin({"true", "1"})
    activity = registry["activity_status"].astype("string")
    lifecycle = registry["lifecycle_status"].astype("string")
    official = registry["official_status"].astype("string")
    is_new = registry["is_new_listing"]
    if not pd.api.types.is_bool_dtype(is_new):
        is_new = is_new.astype("string").str.lower().isin({"true", "1"})

    return RegistryDisplayMetrics(
        total_securities=len(registry),
        currently_listed=int(listed.sum()),
        listed_and_recently_traded=int((listed & (activity == "recently_traded")).sum()),
        historical_only=int((lifecycle == "historical_only").sum()),
        listed_not_recently_traded=int(
            (listed & (activity != "recently_traded")).sum()
        ),
        new_listings=int(is_new.sum()),
        suspended=int((official == "suspended").sum()),
        non_compliant=int((official == "non_compliant").sum()),
        officially_delisted=int((official == "delisted").sum()),
        unknown=int((lifecycle == "unknown").sum()),
    )
