"""Fetch, parse, validate, and cache the official PSX listing snapshot."""

from dataclasses import dataclass
from datetime import date, datetime
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests

from .config import (
    CURRENT_LISTINGS_PATH,
    LISTINGS_METADATA_DIR,
    PROJECT_TIMEZONE,
    PSX_LISTINGS_TABLE_URL_TEMPLATE,
    REQUEST_TIMEOUT_SECONDS,
)


LOGGER = logging.getLogger(__name__)
LISTING_ENDPOINTS = (
    ("main", "nc", "normal_counter"),
    ("main", "dc", "non_compliant_segment"),
    ("gem", "nc", "normal_counter"),
    ("gem", "dc", "non_compliant_segment"),
)
LISTING_SNAPSHOT_COLUMNS = (
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
    "non_compliance_reason",
    "source",
    "listing_refreshed_at",
    "snapshot_date",
)
CORE_HEADER_FIELDS = {
    "symbol",
    "company_name",
    "sector",
    "clearing_type",
    "shares",
    "free_float",
    "listed_in",
}
HEADER_ALIASES = {
    "symbol": "symbol",
    "name": "company_name",
    "company_name": "company_name",
    "security_name": "company_name",
    "sector": "sector",
    "clearing_type": "clearing_type",
    "shares": "shares",
    "free_float": "free_float",
    "listed_in": "listed_in",
    "non_compliance_of_psx_regulations": "non_compliance_reason",
    "non_compliance": "non_compliance_reason",
}


class OfficialListingsError(RuntimeError):
    """Base error for official listing acquisition or validation."""


class ListingsRequestError(OfficialListingsError):
    """Raised when an official PSX listing request fails."""


class ListingsParseError(OfficialListingsError):
    """Raised when official listing HTML no longer matches the known schema."""


class ListingsUnavailableError(OfficialListingsError):
    """Raised when neither live nor cached official listings are usable."""


class ListingTableFetcher(Protocol):
    """HTTP behavior required by the listing refresh workflow."""

    def fetch_listing_table(self, board: str, segment: str) -> str:
        """Return one official listing-table HTML fragment."""
        ...


@dataclass(frozen=True)
class ListingsRefreshResult:
    """Structured result of a live or cached official-listing refresh."""

    data: pd.DataFrame
    current_snapshot_path: Path
    dated_snapshot_path: Path | None
    row_count: int
    duplicate_count: int
    used_cache: bool
    listing_refreshed_at: str
    message: str
    live_error: str | None = None


class PsxListingsClient:
    """Small requests client for the official PSX listing fragments."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.session = session if session is not None else requests.Session()
        self.timeout_seconds = timeout_seconds
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 PSXVirtualTrader/1.0"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    def fetch_listing_table(self, board: str, segment: str) -> str:
        """Fetch one current official listing table without browser cookies."""
        if (board, segment) not in {
            (endpoint_board, endpoint_segment)
            for endpoint_board, endpoint_segment, _ in LISTING_ENDPOINTS
        }:
            raise ValueError(f"Unsupported listing endpoint: {board}/{segment}")

        url = PSX_LISTINGS_TABLE_URL_TEMPLATE.format(
            board=board,
            segment=segment,
        )
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ListingsRequestError(
                f"PSX listing request timed out for {board}/{segment}"
            ) from exc
        except requests.RequestException as exc:
            raise ListingsRequestError(
                f"PSX listing request failed for {board}/{segment}: {exc}"
            ) from exc

        if not response.text.strip():
            raise ListingsRequestError(
                f"PSX returned an empty listing response for {board}/{segment}"
            )
        content_type = response.headers.get("Content-Type", "")
        if content_type and "html" not in content_type.lower():
            raise ListingsRequestError(
                "PSX returned an unexpected listing response type for "
                f"{board}/{segment}: {content_type}"
            )
        return response.text


def _normalize_header(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_integer(value: str, field: str, symbol: str) -> int | None:
    normalized = value.replace(",", "").strip()
    if not normalized or normalized in {"-", "N/A"}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise ListingsParseError(
            f"Invalid {field} value for {symbol}: {value!r}"
        ) from exc


def infer_security_type(
    symbol: str,
    company_name: str,
    sector: str,
    board: str,
) -> str:
    """Infer a conservative instrument type from reliable listing metadata."""
    normalized_symbol = symbol.upper()
    description = f"{company_name} {sector}".upper()
    if re.search(r"\bRIGHTS?\b", description):
        return "right"
    if (
        "ETF" in normalized_symbol
        or re.search(r"\bETF\b", description)
        or "EXCHANGE TRADED FUND" in description
    ):
        return "etf"
    if "PREF" in description or "PREFERENCE" in description:
        return "preference_share"
    if board == "gem":
        return "gem_equity"
    if "REAL ESTATE INVESTMENT TRUST" in description or re.search(
        r"\bREIT\b",
        description,
    ):
        return "other"
    return "ordinary_equity"


def parse_listing_html(
    html: str,
    *,
    board: str,
    segment: str,
    source_url: str,
    refreshed_at: datetime,
    allow_empty: bool = False,
) -> list[dict[str, object]]:
    """Parse a PSX listing fragment using normalized header names."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ListingsParseError(f"No listing table found for {board}/{segment}")

    header_cells = table.select("thead th")
    raw_headers = [_normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells]
    mapped_headers = [HEADER_ALIASES.get(header, header) for header in raw_headers]
    missing_headers = sorted(CORE_HEADER_FIELDS.difference(mapped_headers))
    if missing_headers:
        LOGGER.error(
            "Official listing columns changed for %s/%s; missing: %s",
            board,
            segment,
            ", ".join(missing_headers),
        )
        raise ListingsParseError(
            f"Listing table {board}/{segment} is missing required columns: "
            f"{', '.join(missing_headers)}"
        )
    if segment == "dc" and "non_compliance_reason" not in mapped_headers:
        raise ListingsParseError(
            f"Listing table {board}/{segment} is missing the non-compliance column"
        )

    records: list[dict[str, object]] = []
    for row_number, row in enumerate(table.select("tbody tr"), start=1):
        cells = row.select("td")
        if not cells:
            continue
        if len(cells) != len(mapped_headers):
            raise ListingsParseError(
                f"Listing row {row_number} for {board}/{segment} has "
                f"{len(cells)} cells; expected {len(mapped_headers)}"
            )

        values: dict[str, str] = {}
        for field, cell in zip(mapped_headers, cells):
            if field in {"symbol", "listed_in"} and cell.has_attr("data-search"):
                value = cell.get("data-search", "")
            else:
                value = cell.get_text(" ", strip=True)
            values[field] = _normalize_text(value)

        symbol = values["symbol"].upper()
        if not symbol:
            raise ListingsParseError(
                f"Listing row {row_number} for {board}/{segment} has no symbol"
            )
        company_name = values.get("company_name", "")
        sector = values.get("sector", "")
        records.append(
            {
                "symbol": symbol,
                "company_name": company_name,
                "security_type": infer_security_type(
                    symbol,
                    company_name,
                    sector,
                    board,
                ),
                "sector": sector,
                "board": "GEM" if board == "gem" else "Main",
                "listing_segment": (
                    "non_compliant_segment"
                    if segment == "dc"
                    else "normal_counter"
                ),
                "clearing_type": values.get("clearing_type", ""),
                "listed_in": values.get("listed_in", ""),
                "shares": _parse_integer(values.get("shares", ""), "shares", symbol),
                "free_float": _parse_integer(
                    values.get("free_float", ""),
                    "free_float",
                    symbol,
                ),
                "officially_listed": True,
                "official_status": (
                    "non_compliant" if segment == "dc" else "listed"
                ),
                "non_compliance_reason": values.get(
                    "non_compliance_reason",
                    "",
                ),
                "source": source_url,
                "listing_refreshed_at": refreshed_at.isoformat(timespec="seconds"),
                "snapshot_date": refreshed_at.date().isoformat(),
            }
        )

    if not records and not allow_empty:
        raise ListingsParseError(
            f"Official listing parser produced zero rows for {board}/{segment}"
        )
    return records


def _deduplicate_listings(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_mask = data.duplicated("symbol", keep=False)
    duplicate_count = int(data.duplicated("symbol", keep="last").sum())
    if duplicate_count:
        duplicates = sorted(data.loc[duplicate_mask, "symbol"].astype(str).unique())
        LOGGER.warning(
            "Official listing source contains %s duplicate symbol rows: %s",
            duplicate_count,
            ", ".join(duplicates),
        )

    status_priority = data["official_status"].map(
        {"listed": 0, "non_compliant": 1}
    ).fillna(-1)
    deduplicated = (
        data.assign(_status_priority=status_priority)
        .sort_values(["symbol", "_status_priority"], kind="stable")
        .drop_duplicates("symbol", keep="last")
        .drop(columns="_status_priority")
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )
    return deduplicated, duplicate_count


def fetch_current_listings(
    client: ListingTableFetcher | None = None,
    *,
    refreshed_at: datetime | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch and combine all official main/GEM listing segments."""
    timestamp = refreshed_at or datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    if timestamp.tzinfo is None:
        raise ValueError("refreshed_at must be timezone-aware")
    timestamp = timestamp.astimezone(ZoneInfo(PROJECT_TIMEZONE))
    listing_client = client if client is not None else PsxListingsClient()
    records: list[dict[str, object]] = []

    LOGGER.info("Refreshing official PSX listing tables")
    for board, segment, _ in LISTING_ENDPOINTS:
        url = PSX_LISTINGS_TABLE_URL_TEMPLATE.format(
            board=board,
            segment=segment,
        )
        html = listing_client.fetch_listing_table(board, segment)
        parsed = parse_listing_html(
            html,
            board=board,
            segment=segment,
            source_url=url,
            refreshed_at=timestamp,
            allow_empty=True,
        )
        LOGGER.info(
            "Parsed %s official listing rows from %s/%s",
            len(parsed),
            board,
            segment,
        )
        records.extend(parsed)

    if not records:
        raise ListingsParseError(
            "Official PSX listing refresh produced zero rows across all segments"
        )
    data = pd.DataFrame(records, columns=LISTING_SNAPSHOT_COLUMNS)
    data, duplicate_count = _deduplicate_listings(data)
    LOGGER.info("Official PSX listing refresh completed with %s rows", len(data))
    return data, duplicate_count


def write_dataframe_atomically(data: pd.DataFrame, path: Path) -> None:
    """Write a dataframe using a same-directory temporary file and replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        data.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_listing_snapshot(path: Path) -> pd.DataFrame:
    """Load and validate a non-empty official listing snapshot."""
    snapshot_path = Path(path)
    try:
        data = pd.read_csv(snapshot_path, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ListingsUnavailableError(
            f"Could not read listing snapshot {snapshot_path}: {exc}"
        ) from exc
    missing = [column for column in LISTING_SNAPSHOT_COLUMNS if column not in data]
    if data.empty or missing:
        detail = "contains no rows" if data.empty else f"missing: {', '.join(missing)}"
        raise ListingsUnavailableError(
            f"Invalid listing snapshot {snapshot_path}: {detail}"
        )
    data["symbol"] = data["symbol"].astype("string").str.strip().str.upper()
    if data["symbol"].isna().any() or (data["symbol"] == "").any():
        raise ListingsUnavailableError(
            f"Invalid listing snapshot {snapshot_path}: blank symbols"
        )
    if data["symbol"].duplicated().any():
        raise ListingsUnavailableError(
            f"Invalid listing snapshot {snapshot_path}: duplicate symbols"
        )
    return data.sort_values("symbol", kind="stable").reset_index(drop=True)


def _load_most_recent_cached_snapshot(
    current_snapshot_path: Path,
    snapshots_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    candidates: list[Path] = []
    current_path = Path(current_snapshot_path)
    if current_path.is_file():
        candidates.append(current_path)
    directory = Path(snapshots_dir)
    if directory.is_dir():
        candidates.extend(
            path
            for path in sorted(directory.glob("listings_*.csv"), reverse=True)
            if path != current_path
        )

    errors: list[str] = []
    for candidate in candidates:
        try:
            return load_listing_snapshot(candidate), candidate
        except ListingsUnavailableError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no cached snapshots exist"
    raise ListingsUnavailableError(f"No valid cached PSX listing snapshot: {detail}")


def refresh_official_listings(
    *,
    client: ListingTableFetcher | None = None,
    current_snapshot_path: Path = CURRENT_LISTINGS_PATH,
    snapshots_dir: Path = LISTINGS_METADATA_DIR,
    refreshed_at: datetime | None = None,
    retain_dated_snapshot: bool = True,
) -> ListingsRefreshResult:
    """Refresh official listings, falling back to the newest valid cache."""
    timestamp = refreshed_at or datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    if timestamp.tzinfo is None:
        raise ValueError("refreshed_at must be timezone-aware")
    timestamp = timestamp.astimezone(ZoneInfo(PROJECT_TIMEZONE))
    current_path = Path(current_snapshot_path)
    snapshot_directory = Path(snapshots_dir)

    try:
        data, duplicate_count = fetch_current_listings(
            client,
            refreshed_at=timestamp,
        )
        write_dataframe_atomically(data, current_path)
        dated_path: Path | None = None
        if retain_dated_snapshot:
            dated_path = snapshot_directory / f"listings_{timestamp.date().isoformat()}.csv"
            write_dataframe_atomically(data, dated_path)
        message = f"Refreshed {len(data)} official PSX listings from the live source"
        LOGGER.info(message)
        return ListingsRefreshResult(
            data=data,
            current_snapshot_path=current_path,
            dated_snapshot_path=dated_path,
            row_count=len(data),
            duplicate_count=duplicate_count,
            used_cache=False,
            listing_refreshed_at=timestamp.isoformat(timespec="seconds"),
            message=message,
        )
    except Exception as exc:
        live_error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Live official listing refresh failed: %s", live_error)
        cached_data, cached_path = _load_most_recent_cached_snapshot(
            current_path,
            snapshot_directory,
        )
        refreshed_value = str(cached_data["listing_refreshed_at"].iloc[0])
        message = (
            f"Using cached official PSX listings from {cached_path}; "
            f"live refresh failed"
        )
        LOGGER.warning(message)
        return ListingsRefreshResult(
            data=cached_data,
            current_snapshot_path=cached_path,
            dated_snapshot_path=None,
            row_count=len(cached_data),
            duplicate_count=0,
            used_cache=True,
            listing_refreshed_at=refreshed_value,
            message=message,
            live_error=live_error,
        )
