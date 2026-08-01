"""Offline tests for official PSX listing parsing and snapshot fallback."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import pytest

from data_pipeline.src.official_listings import (
    ListingsParseError,
    ListingsRequestError,
    ListingsUnavailableError,
    PsxListingsClient,
    fetch_current_listings,
    infer_security_type,
    parse_listing_html,
    refresh_official_listings,
)


FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 7, 30, 17, 15, tzinfo=ZoneInfo("Asia/Karachi"))
EMPTY_LISTINGS_HTML = """
<table>
  <thead><tr>
    <th>Symbol</th><th>Name</th><th>Sector</th><th>Clearing Type</th>
    <th>Shares</th><th>Free Float</th><th>Listed In</th>
  </tr></thead>
  <tbody></tbody>
</table>
"""
EMPTY_NON_COMPLIANT_HTML = EMPTY_LISTINGS_HTML.replace(
    "<th>Clearing Type</th>",
    "<th>Non-Compliance of PSX Regulations</th><th>Clearing Type</th>",
)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FixtureClient:
    """Return saved fixture HTML for all four official endpoints."""

    def fetch_listing_table(self, board: str, segment: str) -> str:
        if board == "main" and segment == "nc":
            return _fixture("listings_main_nc.html")
        if board == "main" and segment == "dc":
            return _fixture("listings_main_dc.html")
        return EMPTY_NON_COMPLIANT_HTML if segment == "dc" else EMPTY_LISTINGS_HTML


class FailingClient:
    def fetch_listing_table(self, board: str, segment: str) -> str:
        raise ListingsRequestError("offline fixture failure")


class EmptyClient:
    def fetch_listing_table(self, board: str, segment: str) -> str:
        return EMPTY_NON_COMPLIANT_HTML if segment == "dc" else EMPTY_LISTINGS_HTML


def test_http_client_rejects_a_non_html_response() -> None:
    class Response:
        text = '{"unexpected": "json"}'
        headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, url: str, *, timeout: int) -> Response:
            return Response()

    client = PsxListingsClient(session=Session())

    with pytest.raises(ListingsRequestError, match="unexpected listing response type"):
        client.fetch_listing_table("main", "nc")


def test_parses_official_fixture_and_preserves_numeric_symbol_as_text() -> None:
    records = parse_listing_html(
        _fixture("listings_main_nc.html"),
        board="main",
        segment="nc",
        source_url="https://dps.psx.com.pk/listings-table/main/nc",
        refreshed_at=NOW,
    )

    assert records[0]["symbol"] == "786"
    assert isinstance(records[0]["symbol"], str)
    assert records[0]["company_name"] == "786 Investments Limited"
    assert records[0]["shares"] == 19_964_501
    assert records[0]["official_status"] == "listed"


def test_parser_uses_normalized_headers_when_column_order_changes() -> None:
    soup = BeautifulSoup(_fixture("listings_main_nc.html"), "html.parser")
    order = [1, 0, 6, 2, 5, 3, 4]
    header = soup.select_one("thead tr")
    assert header is not None
    header_cells = header.find_all("th", recursive=False)
    header.clear()
    for index in order:
        header.append(header_cells[index])
    for row in soup.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        row.clear()
        for index in order:
            row.append(cells[index])

    records = parse_listing_html(
        str(soup),
        board="main",
        segment="nc",
        source_url="fixture",
        refreshed_at=NOW,
    )

    assert records[0]["symbol"] == "786"
    assert records[0]["company_name"] == "786 Investments Limited"
    assert records[0]["free_float"] == 6_987_575


def test_parser_fails_clearly_when_required_column_is_missing() -> None:
    html = _fixture("listings_main_nc.html").replace("<th>Symbol</th>", "")

    with pytest.raises(ListingsParseError, match="missing required columns: symbol"):
        parse_listing_html(
            html,
            board="main",
            segment="nc",
            source_url="fixture",
            refreshed_at=NOW,
        )


def test_missing_company_name_is_preserved_without_fabrication() -> None:
    html = _fixture("listings_main_nc.html").replace(
        "<td>786 Investments Limited</td>",
        "<td></td>",
        1,
    )

    records = parse_listing_html(
        html,
        board="main",
        segment="nc",
        source_url="fixture",
        refreshed_at=NOW,
    )

    assert records[0]["company_name"] == ""


@pytest.mark.parametrize(
    ("symbol", "name", "sector", "board", "expected"),
    [
        ("ACIETF", "Alfalah Consumer Index ETF", "ETF", "main", "etf"),
        ("786R", "786 Investment (Right)", "Investment", "main", "right"),
        ("ASLPS", "Aisha Steel Convertible Pref", "Steel", "main", "preference_share"),
        ("GEMNETS", "Nets International", "Technology", "gem", "gem_equity"),
        ("DCR", "Dolmen City REIT", "Real Estate Investment Trust", "main", "other"),
        ("OGDC", "Oil and Gas Development", "Oil & Gas", "main", "ordinary_equity"),
    ],
)
def test_security_type_classification(
    symbol: str,
    name: str,
    sector: str,
    board: str,
    expected: str,
) -> None:
    assert infer_security_type(symbol, name, sector, board) == expected


def test_duplicate_official_symbol_prefers_non_compliant_status() -> None:
    class DuplicateClient(FixtureClient):
        def fetch_listing_table(self, board: str, segment: str) -> str:
            html = super().fetch_listing_table(board, segment)
            if board == "main" and segment == "dc":
                return html.replace("AAL", "786")
            return html

    data, duplicate_count = fetch_current_listings(
        DuplicateClient(),
        refreshed_at=NOW,
    )

    symbol = data.loc[data["symbol"] == "786"].iloc[0]
    assert duplicate_count == 1
    assert symbol["official_status"] == "non_compliant"
    assert data["symbol"].is_unique


def test_current_snapshot_is_atomic_and_loads_as_strings(tmp_path: Path) -> None:
    current = tmp_path / "listings" / "current_listings.csv"
    result = refresh_official_listings(
        client=FixtureClient(),
        current_snapshot_path=current,
        snapshots_dir=current.parent,
        refreshed_at=NOW,
    )

    saved = pd.read_csv(current, dtype={"symbol": "string"})
    assert result.used_cache is False
    assert result.row_count == 4
    assert "786" in saved["symbol"].tolist()
    assert list(current.parent.glob("*.tmp")) == []


def test_live_failure_falls_back_to_valid_cached_snapshot(tmp_path: Path) -> None:
    current = tmp_path / "listings" / "current_listings.csv"
    live = refresh_official_listings(
        client=FixtureClient(),
        current_snapshot_path=current,
        snapshots_dir=current.parent,
        refreshed_at=NOW,
    )
    original_bytes = current.read_bytes()

    cached = refresh_official_listings(
        client=FailingClient(),
        current_snapshot_path=current,
        snapshots_dir=current.parent,
        refreshed_at=NOW,
    )

    assert live.used_cache is False
    assert cached.used_cache is True
    assert cached.row_count == live.row_count
    assert current.read_bytes() == original_bytes
    assert "live refresh failed" in cached.message


def test_zero_live_rows_do_not_replace_cache_or_create_empty_snapshot(
    tmp_path: Path,
) -> None:
    current = tmp_path / "listings" / "current_listings.csv"
    refresh_official_listings(
        client=FixtureClient(),
        current_snapshot_path=current,
        snapshots_dir=current.parent,
        refreshed_at=NOW,
    )
    original_bytes = current.read_bytes()

    result = refresh_official_listings(
        client=EmptyClient(),
        current_snapshot_path=current,
        snapshots_dir=current.parent,
        refreshed_at=NOW,
    )

    assert result.used_cache is True
    assert current.read_bytes() == original_bytes


def test_live_and_cache_failure_raises_without_creating_snapshot(
    tmp_path: Path,
) -> None:
    current = tmp_path / "listings" / "current_listings.csv"

    with pytest.raises(ListingsUnavailableError, match="No valid cached"):
        refresh_official_listings(
            client=FailingClient(),
            current_snapshot_path=current,
            snapshots_dir=current.parent,
            refreshed_at=NOW,
        )

    assert not current.exists()
