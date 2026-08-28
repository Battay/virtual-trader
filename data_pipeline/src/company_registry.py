"""Build and maintain the PSX company/security lifecycle registry."""

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import logging
from pathlib import Path
import re
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .config import (
    COMPANY_OVERRIDES_PATH,
    COMPANY_REGISTRY_PATH,
    CURRENT_LISTINGS_PATH,
    MASTER_CSV_PATH,
    NEW_LISTING_WINDOW_DAYS,
    PROJECT_TIMEZONE,
    RECENT_TRADING_WINDOW_DAYS,
)
from .official_listings import (
    LISTING_SNAPSHOT_COLUMNS,
    ListingTableFetcher,
    ListingsRefreshResult,
    load_listing_snapshot,
    refresh_official_listings,
    write_dataframe_atomically,
)
from .market_schema import MarketSchemaError, with_legacy_date_alias


LOGGER = logging.getLogger(__name__)
OFFICIAL_STATUSES = {
    "listed",
    "suspended",
    "non_compliant",
    "delisted",
    "historical",
    "unknown",
}
SECURITY_TYPES = {
    "ordinary_equity",
    "etf",
    "right",
    "preference_share",
    "gem_equity",
    "other",
    "unknown",
}
OVERRIDE_COLUMNS = (
    "symbol",
    "company_name_override",
    "official_status_override",
    "previous_symbol",
    "successor_symbol",
    "corporate_action_type",
    "notes",
)
REGISTRY_COLUMNS = (
    "symbol",
    "company_name",
    "security_type",
    "sector",
    "board",
    "listing_segment",
    "clearing_type",
    "listed_in",
    "shares",
    "free_float",
    "officially_listed",
    "official_status",
    "first_seen_date",
    "last_seen_date",
    "trading_days",
    "days_since_last_seen",
    "activity_status",
    "lifecycle_status",
    "is_new_listing",
    "source",
    "listing_refreshed_at",
    "cached_listings_used",
    "registry_updated_at",
    "previous_symbol",
    "successor_symbol",
    "corporate_action_type",
    "notes",
)
FUTURE_CONTRACT_PATTERN = re.compile(
    r"-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]?$"
)


class RegistryError(RuntimeError):
    """Raised when registry inputs are invalid or unavailable."""


class OverrideValidationError(RegistryError):
    """Raised when the optional curated override file is unsafe to apply."""


@dataclass(frozen=True)
class RegistryBuildResult:
    """Metrics and paths produced by one deterministic registry build."""

    output_path: Path
    total_registry_symbols: int
    currently_listed: int
    recently_traded: int
    listed_not_recently_traded: int
    new_listings: int
    historical_only: int
    suspended: int
    non_compliant: int
    delisted: int
    unknown: int
    registry_updated_at: str
    listing_refreshed_at: str
    cached_listings_used: bool
    overrides_applied: int


@dataclass(frozen=True)
class RegistryRefreshResult:
    """Combined official-listing refresh and registry-build result."""

    listings: ListingsRefreshResult
    registry: RegistryBuildResult


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_symbol(value: object) -> str:
    return _normalize_text(value).upper()


def _read_master_history(path: Path) -> pd.DataFrame:
    master_path = Path(path)
    try:
        data = pd.read_csv(master_path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise RegistryError(f"Could not read master dataset {master_path}: {exc}") from exc

    try:
        data = with_legacy_date_alias(data)
    except MarketSchemaError as exc:
        raise RegistryError(f"Master dataset {master_path} is invalid: {exc}") from exc
    missing = [column for column in ("symbol", "date") if column not in data]
    if missing:
        raise RegistryError(
            f"Master dataset {master_path} is missing columns: {', '.join(missing)}"
        )
    if data.empty:
        return pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "first_seen_date": pd.Series(dtype="datetime64[ns]"),
                "last_seen_date": pd.Series(dtype="datetime64[ns]"),
                "trading_days": pd.Series(dtype="int64"),
            }
        )

    data = data.loc[:, ["symbol", "date"]].copy()
    data["symbol"] = data["symbol"].map(_normalize_symbol).astype("string")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if (data["symbol"] == "").any() or data["date"].isna().any():
        raise RegistryError(
            f"Master dataset {master_path} contains blank symbols or invalid dates"
        )
    return (
        data.groupby("symbol", as_index=False, sort=True)
        .agg(
            first_seen_date=("date", "min"),
            last_seen_date=("date", "max"),
            trading_days=("date", "nunique"),
        )
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )


def _validate_listing_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in LISTING_SNAPSHOT_COLUMNS if column not in data]
    if data.empty or missing:
        detail = "contains no rows" if data.empty else f"missing: {', '.join(missing)}"
        raise RegistryError(f"Official listing data is invalid: {detail}")
    listings = data.loc[:, list(LISTING_SNAPSHOT_COLUMNS)].copy()
    listings["symbol"] = listings["symbol"].map(_normalize_symbol).astype("string")
    if (listings["symbol"] == "").any():
        raise RegistryError("Official listing data contains a blank symbol")
    if listings["symbol"].duplicated().any():
        duplicates = sorted(
            listings.loc[listings["symbol"].duplicated(False), "symbol"].unique()
        )
        raise RegistryError(
            f"Official listing data contains duplicate symbols: {', '.join(duplicates)}"
        )
    for column in (
        "company_name",
        "security_type",
        "sector",
        "board",
        "listing_segment",
        "clearing_type",
        "listed_in",
        "official_status",
        "source",
        "non_compliance_reason",
    ):
        listings[column] = listings[column].map(_normalize_text)
    return listings.sort_values("symbol", kind="stable").reset_index(drop=True)


def _read_overrides(path: Path) -> pd.DataFrame:
    override_path = Path(path)
    if not override_path.is_file():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    try:
        data = pd.read_csv(override_path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise OverrideValidationError(
            f"Could not read company overrides {override_path}: {exc}"
        ) from exc

    if "symbol" not in data:
        raise OverrideValidationError(
            f"Company overrides {override_path} must contain a symbol column"
        )
    unexpected = sorted(set(data.columns).difference(OVERRIDE_COLUMNS))
    if unexpected:
        raise OverrideValidationError(
            f"Company overrides {override_path} has unsupported columns: "
            f"{', '.join(unexpected)}"
        )
    for column in OVERRIDE_COLUMNS:
        if column not in data:
            data[column] = ""
    data = data.loc[:, list(OVERRIDE_COLUMNS)].fillna("")
    data["symbol"] = data["symbol"].map(_normalize_symbol).astype("string")
    if data.empty:
        return data
    if (data["symbol"] == "").any():
        raise OverrideValidationError("Company overrides contains a blank symbol")
    if data["symbol"].duplicated().any():
        duplicates = sorted(
            data.loc[data["symbol"].duplicated(False), "symbol"].unique()
        )
        raise OverrideValidationError(
            f"Company overrides contains duplicate symbols: {', '.join(duplicates)}"
        )
    statuses = data["official_status_override"].map(_normalize_text).str.lower()
    invalid_statuses = sorted(
        set(statuses.loc[(statuses != "") & ~statuses.isin(OFFICIAL_STATUSES)])
    )
    if invalid_statuses:
        raise OverrideValidationError(
            "Company overrides contains invalid official statuses: "
            + ", ".join(invalid_statuses)
        )
    data["official_status_override"] = statuses
    for column in OVERRIDE_COLUMNS[1:]:
        if column != "official_status_override":
            data[column] = data[column].map(_normalize_text)
    for column in ("previous_symbol", "successor_symbol"):
        data[column] = data[column].str.upper()
    return data.sort_values("symbol", kind="stable").reset_index(drop=True)


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.loc[~normalized.isin({"true", "false", "1", "0"})]
    if not invalid.empty:
        raise RegistryError(
            "Official listing data contains invalid officially_listed values"
        )
    return normalized.isin({"true", "1"})


def _historical_security_type(symbol: str) -> str:
    return "other" if FUTURE_CONTRACT_PATTERN.search(symbol) else "unknown"


def _classify_lifecycle(record: pd.Series) -> str:
    official_status = str(record["official_status"])
    if official_status == "delisted":
        return "officially_delisted"
    if official_status == "suspended":
        return "suspended"
    if official_status == "non_compliant":
        return "non_compliant"
    if bool(record["is_new_listing"]) and bool(record["officially_listed"]):
        return "newly_listed"
    if bool(record["officially_listed"]):
        if record["activity_status"] == "recently_traded":
            return "listed_recently_traded"
        return "listed_not_recently_traded"
    if official_status == "historical":
        return "historical_only"
    return "unknown"


def _append_note(existing: object, added: object) -> str:
    current = _normalize_text(existing)
    addition = _normalize_text(added)
    if not addition:
        return current
    return f"{current}; {addition}" if current else addition


def _apply_overrides(registry: pd.DataFrame, overrides: pd.DataFrame) -> int:
    if overrides.empty:
        return 0
    registry_symbols = set(registry["symbol"].astype(str))
    unknown_symbols = sorted(set(overrides["symbol"].astype(str)) - registry_symbols)
    if unknown_symbols:
        raise OverrideValidationError(
            "Company overrides references symbols absent from official listings and "
            f"market history: {', '.join(unknown_symbols)}"
        )

    for override in overrides.to_dict(orient="records"):
        symbol = str(override["symbol"])
        index = registry.index[registry["symbol"] == symbol][0]
        if override["company_name_override"]:
            registry.at[index, "company_name"] = override["company_name_override"]
        if override["official_status_override"]:
            registry.at[index, "official_status"] = override[
                "official_status_override"
            ]
        for field in (
            "previous_symbol",
            "successor_symbol",
            "corporate_action_type",
        ):
            if override[field]:
                registry.at[index, field] = override[field]
        registry.at[index, "notes"] = _append_note(
            registry.at[index, "notes"],
            override["notes"],
        )
        registry.at[index, "source"] = _append_note(
            registry.at[index, "source"],
            "manual_override",
        )
        LOGGER.info("Applied evidence-based company override for %s", symbol)
    return len(overrides)


def build_company_registry(
    *,
    listing_data: pd.DataFrame | None = None,
    listing_snapshot_path: Path = CURRENT_LISTINGS_PATH,
    master_path: Path = MASTER_CSV_PATH,
    output_path: Path = COMPANY_REGISTRY_PATH,
    overrides_path: Path = COMPANY_OVERRIDES_PATH,
    reference_date: date | None = None,
    registry_updated_at: datetime | None = None,
    recent_trading_window_days: int = RECENT_TRADING_WINDOW_DAYS,
    new_listing_window_days: int = NEW_LISTING_WINDOW_DAYS,
    cached_listings_used: bool = False,
) -> RegistryBuildResult:
    """Outer-merge official listings and history into a deterministic registry."""
    if recent_trading_window_days < 0 or new_listing_window_days < 0:
        raise ValueError("Registry classification windows cannot be negative")
    timestamp = registry_updated_at or datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    if timestamp.tzinfo is None:
        raise ValueError("registry_updated_at must be timezone-aware")
    timestamp = timestamp.astimezone(ZoneInfo(PROJECT_TIMEZONE))
    effective_reference_date = reference_date or timestamp.date()

    listings = _validate_listing_data(
        listing_data.copy()
        if listing_data is not None
        else load_listing_snapshot(Path(listing_snapshot_path))
    )
    history = _read_master_history(Path(master_path))
    overrides = _read_overrides(Path(overrides_path))

    registry = listings.merge(history, on="symbol", how="outer", sort=True)
    registry["officially_listed"] = _boolean_series(
        registry["officially_listed"].fillna(False)
    )
    historical_mask = ~registry["officially_listed"]
    for column in (
        "company_name",
        "sector",
        "board",
        "listing_segment",
        "clearing_type",
        "listed_in",
        "source",
        "listing_refreshed_at",
        "non_compliance_reason",
    ):
        registry[column] = registry[column].map(_normalize_text)
    registry.loc[historical_mask, "official_status"] = "historical"
    registry.loc[historical_mask, "source"] = "master_market_history"
    registry.loc[historical_mask, "security_type"] = registry.loc[
        historical_mask,
        "symbol",
    ].map(_historical_security_type)
    registry["official_status"] = registry["official_status"].map(
        lambda value: _normalize_text(value).lower() or "unknown"
    )
    registry["security_type"] = registry["security_type"].map(
        lambda value: _normalize_text(value).lower() or "unknown"
    )
    if not set(registry["official_status"]).issubset(OFFICIAL_STATUSES):
        raise RegistryError("Official listing data contains unsupported statuses")
    if not set(registry["security_type"]).issubset(SECURITY_TYPES):
        raise RegistryError("Official listing data contains unsupported security types")

    registry["trading_days"] = registry["trading_days"].fillna(0).astype(int)
    registry["days_since_last_seen"] = registry["last_seen_date"].map(
        lambda value: (
            max(0, (effective_reference_date - pd.Timestamp(value).date()).days)
            if not pd.isna(value)
            else pd.NA
        )
    )
    registry["activity_status"] = registry["last_seen_date"].map(
        lambda value: (
            "never_seen_in_market_history"
            if pd.isna(value)
            else (
                "recently_traded"
                if max(
                    0,
                    (effective_reference_date - pd.Timestamp(value).date()).days,
                )
                <= recent_trading_window_days
                else "not_recently_traded"
            )
        )
    )
    registry["is_new_listing"] = registry["first_seen_date"].map(
        lambda value: (
            False
            if pd.isna(value)
            else max(
                0,
                (effective_reference_date - pd.Timestamp(value).date()).days,
            )
            <= new_listing_window_days
        )
    ) & registry["officially_listed"]
    registry["previous_symbol"] = ""
    registry["successor_symbol"] = ""
    registry["corporate_action_type"] = ""
    registry["notes"] = registry["non_compliance_reason"].map(_normalize_text)
    overrides_applied = _apply_overrides(registry, overrides)
    registry["lifecycle_status"] = registry.apply(_classify_lifecycle, axis=1)
    registry["registry_updated_at"] = timestamp.isoformat(timespec="seconds")
    registry["cached_listings_used"] = bool(cached_listings_used)

    for column in ("first_seen_date", "last_seen_date"):
        registry[column] = pd.to_datetime(registry[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    registry = (
        registry.loc[:, list(REGISTRY_COLUMNS)]
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )
    write_dataframe_atomically(registry, Path(output_path))

    result = RegistryBuildResult(
        output_path=Path(output_path),
        total_registry_symbols=len(registry),
        currently_listed=int(registry["officially_listed"].sum()),
        recently_traded=int(
            (registry["activity_status"] == "recently_traded").sum()
        ),
        listed_not_recently_traded=int(
            (
                registry["officially_listed"]
                & (registry["activity_status"] != "recently_traded")
            ).sum()
        ),
        new_listings=int(registry["is_new_listing"].sum()),
        historical_only=int(
            (registry["lifecycle_status"] == "historical_only").sum()
        ),
        suspended=int((registry["official_status"] == "suspended").sum()),
        non_compliant=int(
            (registry["official_status"] == "non_compliant").sum()
        ),
        delisted=int((registry["official_status"] == "delisted").sum()),
        unknown=int((registry["lifecycle_status"] == "unknown").sum()),
        registry_updated_at=timestamp.isoformat(timespec="seconds"),
        listing_refreshed_at=_normalize_text(
            listings["listing_refreshed_at"].iloc[0]
        ),
        cached_listings_used=bool(cached_listings_used),
        overrides_applied=overrides_applied,
    )
    LOGGER.info(
        "Registry built: total=%s listed=%s recent=%s historical_only=%s "
        "non_compliant=%s",
        result.total_registry_symbols,
        result.currently_listed,
        result.recently_traded,
        result.historical_only,
        result.non_compliant,
    )
    return result


def refresh_and_build_registry(
    *,
    client: ListingTableFetcher | None = None,
    current_snapshot_path: Path = CURRENT_LISTINGS_PATH,
    snapshots_dir: Path | None = None,
    master_path: Path = MASTER_CSV_PATH,
    output_path: Path = COMPANY_REGISTRY_PATH,
    overrides_path: Path = COMPANY_OVERRIDES_PATH,
    reference_date: date | None = None,
    refreshed_at: datetime | None = None,
) -> RegistryRefreshResult:
    """Refresh official listings and rebuild the registry from trusted inputs."""
    refresh_kwargs: dict[str, object] = {
        "client": client,
        "current_snapshot_path": current_snapshot_path,
        "refreshed_at": refreshed_at,
    }
    if snapshots_dir is not None:
        refresh_kwargs["snapshots_dir"] = snapshots_dir
    listings_result = refresh_official_listings(**refresh_kwargs)
    registry_result = build_company_registry(
        listing_data=listings_result.data,
        listing_snapshot_path=listings_result.current_snapshot_path,
        master_path=master_path,
        output_path=output_path,
        overrides_path=overrides_path,
        reference_date=reference_date,
        registry_updated_at=refreshed_at,
        cached_listings_used=listings_result.used_cache,
    )
    return RegistryRefreshResult(
        listings=listings_result,
        registry=registry_result,
    )


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def _print_registry_metrics(result: RegistryBuildResult) -> None:
    print(f"Registry file: {result.output_path}")
    print(f"Total securities: {result.total_registry_symbols}")
    print(f"Currently listed: {result.currently_listed}")
    print(f"Recently traded: {result.recently_traded}")
    print(f"Listed but not recently traded: {result.listed_not_recently_traded}")
    print(f"New listings: {result.new_listings}")
    print(f"Historical only: {result.historical_only}")
    print(f"Suspended: {result.suspended}")
    print(f"Non-compliant: {result.non_compliant}")
    print(f"Delisted: {result.delisted}")
    print(f"Unknown: {result.unknown}")
    print(f"Registry updated at: {result.registry_updated_at}")


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh official listings or rebuild the company registry."""
    parser = argparse.ArgumentParser(description="Manage the PSX company registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "refresh-listings",
        help="Refresh the official PSX listing snapshot",
    )
    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Rebuild the registry from cached listings and market history",
    )
    rebuild_parser.add_argument("--reference-date", type=_iso_date)
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Refresh official listings and rebuild the registry",
    )
    refresh_parser.add_argument("--reference-date", type=_iso_date)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if args.command == "refresh-listings":
            listings_result = refresh_official_listings()
            print(f"Listing snapshot: {listings_result.current_snapshot_path}")
            print(f"Official securities: {listings_result.row_count}")
            print(f"Cached data used: {'yes' if listings_result.used_cache else 'no'}")
            print(listings_result.message)
            return 0
        if args.command == "rebuild":
            registry_result = build_company_registry(
                reference_date=args.reference_date,
            )
        else:
            combined = refresh_and_build_registry(
                reference_date=args.reference_date,
            )
            registry_result = combined.registry
            print(
                f"Cached listings used: "
                f"{'yes' if combined.listings.used_cache else 'no'}"
            )
        _print_registry_metrics(registry_result)
        return 0
    except Exception as exc:
        LOGGER.error("Company registry command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
