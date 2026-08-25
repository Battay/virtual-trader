"""Offline tests for the frozen current common-equity identity universe."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.equity_universe import (
    EQUITY_UNIVERSE_COLUMNS,
    EquityUniverseError,
    build_current_common_equity_universe,
    build_identity_payload,
    deterministic_universe_identity,
    run_equity_universe,
    write_equity_universe_artifacts,
)
from data_pipeline.src.official_listings import LISTING_SNAPSHOT_COLUMNS


MARKET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32()),
        pa.field("symbol", pa.string()),
        pa.field("ldcp", pa.float64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("change_percent", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
OFFICIAL_SOURCE = "https://dps.psx.com.pk/listings-table/main/nc"
SNAPSHOT_DATE = "2026-08-02"
REFRESHED_AT = "2026-08-02T02:54:25+05:00"
REGISTRY_COLUMNS = (
    "symbol",
    "company_name",
    "security_type",
    "sector",
    "officially_listed",
    "official_status",
    "lifecycle_status",
    "source",
    "previous_symbol",
    "successor_symbol",
    "corporate_action_type",
)


def _instrument_specs() -> tuple[tuple[str, str, str, bool], ...]:
    return (
        ("CURR", "ordinary_equity", "ENGINEERING", True),
        ("GEMX", "gem_equity", "TECHNOLOGY & COMMUNICATION", True),
        ("ETFONE", "etf", "EXCHANGE TRADED FUNDS", True),
        ("RIGHT1", "right", "ENGINEERING", True),
        ("PREF", "preference_share", "CHEMICAL", True),
        ("MODX", "ordinary_equity", "MODARABAS", True),
        ("REITX", "other", "REAL ESTATE INVESTMENT TRUST", True),
        ("UNKNOWN", "unknown", "", False),
        ("P03PIB050824", "unknown", "", False),
        ("BAFLTFC7", "unknown", "", False),
    )


def _market_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (symbol, _, _, _) in enumerate(reversed(_instrument_specs())):
        rows.append(
            {
                "market_date": date(2024, 1, 1 + (index % 2)),
                "symbol": symbol,
                "ldcp": 10.0,
                # CURR deliberately has source-policy-allowed unavailable OHL
                # and zero volume; identity membership must not filter it.
                "open": 0.0 if symbol == "CURR" else 10.0,
                "high": 0.0 if symbol == "CURR" else 11.0,
                "low": 0.0 if symbol == "CURR" else 9.0,
                "close": 10.5,
                "change": 0.5,
                "change_percent": 5.0,
                "volume": 0 if symbol == "CURR" else 100 + index,
            }
        )
    return pd.DataFrame(rows)


def _registry_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, security_type, sector, listed in _instrument_specs():
        rows.append(
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited" if listed else pd.NA,
                "security_type": security_type,
                "sector": sector if listed else pd.NA,
                "officially_listed": listed,
                "official_status": "listed" if listed else "historical",
                "lifecycle_status": (
                    "listed_recently_traded" if listed else "historical_only"
                ),
                "source": OFFICIAL_SOURCE if listed else "master_market_history",
                "previous_symbol": pd.NA,
                "successor_symbol": pd.NA,
                "corporate_action_type": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def _listing_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, security_type, sector, listed in _instrument_specs():
        if not listed:
            continue
        row = {column: "" for column in LISTING_SNAPSHOT_COLUMNS}
        row.update(
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "security_type": security_type,
                "sector": sector,
                "board": "GEM" if security_type == "gem_equity" else "Main",
                "listing_segment": "normal_counter",
                "clearing_type": "NC",
                "listed_in": "ALLSHR",
                "shares": 1_000_000,
                "free_float": 100_000,
                "officially_listed": True,
                "official_status": "listed",
                "source": OFFICIAL_SOURCE,
                "listing_refreshed_at": REFRESHED_AT,
                "snapshot_date": SNAPSHOT_DATE,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=LISTING_SNAPSHOT_COLUMNS)


def _build_records(
    *,
    market: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    listings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_current_common_equity_universe(
        _market_frame() if market is None else market,
        registry=_registry_frame() if registry is None else registry,
        listings=_listing_frame() if listings is None else listings,
    )


def test_only_authoritative_common_equities_are_included_without_quality_filter() -> None:
    records = _build_records().set_index("symbol")

    assert list(records.index) == ["CURR", "GEMX"]
    assert records.loc["CURR", "observation_count"] == 1
    assert records.loc["CURR", "median_volume"] == 0
    assert records.loc["CURR", "zero_volume_ratio"] == pytest.approx(1.0)
    assert records.loc["CURR", "zero_ohl_ratio"] == pytest.approx(1.0)
    assert set(records["instrument_category"]) == {"COMMON_EQUITY"}


def test_unknown_fund_right_preference_modaraba_reit_debt_and_government_excluded() -> None:
    records = _build_records()

    excluded = {
        "UNKNOWN",
        "ETFONE",
        "RIGHT1",
        "PREF",
        "MODX",
        "REITX",
        "P03PIB050824",
        "BAFLTFC7",
    }
    assert excluded.isdisjoint(set(records["symbol"]))


def test_ordering_and_universe_identity_are_deterministic_and_quality_independent() -> None:
    first = _build_records()
    shuffled_market = _market_frame().sample(frac=1.0, random_state=11)
    shuffled_registry = _registry_frame().sample(frac=1.0, random_state=12)
    shuffled_listings = _listing_frame().sample(frac=1.0, random_state=13)
    second = _build_records(
        market=shuffled_market,
        registry=shuffled_registry,
        listings=shuffled_listings,
    )
    changed_quality = _market_frame()
    changed_quality.loc[changed_quality["symbol"] == "CURR", "volume"] = 999_999
    third = _build_records(market=changed_quality)

    assert tuple(first.columns) == EQUITY_UNIVERSE_COLUMNS
    assert first["symbol"].tolist() == ["CURR", "GEMX"]
    pd.testing.assert_frame_equal(first, second)
    first_hash = deterministic_universe_identity(build_identity_payload(first))
    assert first_hash == deterministic_universe_identity(build_identity_payload(second))
    assert first_hash == deterministic_universe_identity(build_identity_payload(third))

    changed_identity = first.copy(deep=True)
    changed_identity.loc[changed_identity["symbol"] == "CURR", "sector"] = "CEMENT"
    assert first_hash != deterministic_universe_identity(
        build_identity_payload(changed_identity)
    )


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    parquet = tmp_path / "market.parquet"
    registry = tmp_path / "company_registry.csv"
    listings = tmp_path / "current_listings.csv"
    pq.write_table(
        pa.Table.from_pandas(
            _market_frame(), schema=MARKET_SCHEMA, preserve_index=False
        ),
        parquet,
    )
    _registry_frame().to_csv(registry, index=False)
    _listing_frame().to_csv(listings, index=False)
    return parquet, registry, listings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_run_preserves_source_parquet_and_returns_provenance(
    tmp_path: Path,
) -> None:
    parquet, registry, listings = _write_sources(tmp_path)
    before = _sha256(parquet)

    result = run_equity_universe(
        parquet_path=parquet,
        registry_path=registry,
        listing_snapshot_path=listings,
    )

    assert result.summary.universe_count == 2
    assert result.provenance.listing_snapshot_date == SNAPSHOT_DATE
    assert result.provenance.universe_hash == result.summary.universe_hash
    assert result.provenance.research_limitation.startswith(
        "This universe uses current authoritative"
    )
    assert _sha256(parquet) == before


def test_artifacts_are_deterministic_and_refuse_implicit_overwrite(
    tmp_path: Path,
) -> None:
    parquet, registry, listings = _write_sources(tmp_path)
    result = run_equity_universe(
        parquet_path=parquet,
        registry_path=registry,
        listing_snapshot_path=listings,
    )
    output = tmp_path / "artifacts" / "current_common_equity.csv"

    csv_path, json_path = write_equity_universe_artifacts(result, output)
    csv_bytes = csv_path.read_bytes()
    json_bytes = json_path.read_bytes()
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))

    assert sidecar["universe_hash"] == result.summary.universe_hash
    with pytest.raises(EquityUniverseError, match="--overwrite"):
        write_equity_universe_artifacts(result, output)
    write_equity_universe_artifacts(result, output, overwrite=True)
    assert csv_path.read_bytes() == csv_bytes
    assert json_path.read_bytes() == json_bytes


def test_listing_classification_inconsistency_fails_closed() -> None:
    listings = _listing_frame().loc[lambda frame: frame["symbol"] != "CURR"]

    with pytest.raises(EquityUniverseError, match="missing from the listing snapshot"):
        _build_records(listings=listings)
