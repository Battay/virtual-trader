"""Offline tests for registry merging, lifecycle rules, and overrides."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data_pipeline.src.company_registry import (
    OverrideValidationError,
    build_company_registry,
    refresh_and_build_registry,
)
from data_pipeline.src.official_listings import (
    LISTING_SNAPSHOT_COLUMNS,
    ListingsRequestError,
    ListingsUnavailableError,
)


NOW = datetime(2026, 7, 31, 17, 15, tzinfo=ZoneInfo("Asia/Karachi"))


def _official_row(
    symbol: str,
    *,
    company_name: str = "Example Limited",
    official_status: str = "listed",
    security_type: str = "ordinary_equity",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "company_name": company_name,
        "security_type": security_type,
        "sector": "Technology",
        "board": "Main",
        "listing_segment": (
            "non_compliant_segment"
            if official_status == "non_compliant"
            else "normal_counter"
        ),
        "clearing_type": "NC",
        "listed_in": "ALLSHR",
        "shares": 1_000_000,
        "free_float": 100_000,
        "officially_listed": True,
        "official_status": official_status,
        "non_compliance_reason": (
            "5.11.1(a)" if official_status == "non_compliant" else ""
        ),
        "source": "https://dps.psx.com.pk/listings-table/main/nc",
        "listing_refreshed_at": NOW.isoformat(timespec="seconds"),
        "snapshot_date": NOW.date().isoformat(),
    }


def _listing_data(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=LISTING_SNAPSHOT_COLUMNS)


def _write_master(path: Path, rows: list[dict[str, object]]) -> None:
    data = pd.DataFrame(rows, columns=("symbol", "date", "close"))
    data.to_csv(path, index=False)


def test_outer_merge_keeps_official_and_historical_symbols(tmp_path: Path) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    _write_master(
        master_path,
        [
            {"symbol": "ACTIVE", "date": "2026-07-30", "close": 10.0},
            {"symbol": "ACTIVE", "date": "2026-07-29", "close": 9.0},
            {"symbol": "FUT-AUG", "date": "2026-07-30", "close": 11.0},
        ],
    )

    result = build_company_registry(
        listing_data=_listing_data(
            _official_row("ACTIVE"),
            _official_row("NOHISTORY"),
        ),
        master_path=master_path,
        output_path=output_path,
        overrides_path=tmp_path / "missing-overrides.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )

    registry = pd.read_csv(output_path, dtype={"symbol": "string"})
    no_history = registry.loc[registry["symbol"] == "NOHISTORY"].iloc[0]
    historical = registry.loc[registry["symbol"] == "FUT-AUG"].iloc[0]
    assert registry["symbol"].tolist() == ["ACTIVE", "FUT-AUG", "NOHISTORY"]
    assert bool(no_history["officially_listed"]) is True
    assert no_history["activity_status"] == "never_seen_in_market_history"
    assert no_history["lifecycle_status"] == "listed_not_recently_traded"
    assert historical["official_status"] == "historical"
    assert historical["lifecycle_status"] == "historical_only"
    assert historical["security_type"] == "other"
    assert result.total_registry_symbols == 3
    assert result.currently_listed == 2
    assert result.historical_only == 1


def test_non_compliant_listing_remains_officially_listed_without_recent_trade(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    _write_master(
        master_path,
        [{"symbol": "NCL", "date": "2026-05-01", "close": 10.0}],
    )

    result = build_company_registry(
        listing_data=_listing_data(
            _official_row("NCL", official_status="non_compliant")
        ),
        master_path=master_path,
        output_path=output_path,
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )

    row = pd.read_csv(output_path).iloc[0]
    assert bool(row["officially_listed"]) is True
    assert row["activity_status"] == "not_recently_traded"
    assert row["lifecycle_status"] == "non_compliant"
    assert result.listed_not_recently_traded == 1
    assert result.non_compliant == 1


def test_new_and_recent_classification_respects_configurable_thresholds(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.csv"
    _write_master(
        master_path,
        [{"symbol": "BOUNDARY", "date": "2026-07-01", "close": 10.0}],
    )
    listing_data = _listing_data(_official_row("BOUNDARY"))

    build_company_registry(
        listing_data=listing_data,
        master_path=master_path,
        output_path=tmp_path / "window30.csv",
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
        recent_trading_window_days=30,
        new_listing_window_days=30,
    )
    build_company_registry(
        listing_data=listing_data,
        master_path=master_path,
        output_path=tmp_path / "window29.csv",
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
        recent_trading_window_days=29,
        new_listing_window_days=29,
    )

    within = pd.read_csv(tmp_path / "window30.csv").iloc[0]
    outside = pd.read_csv(tmp_path / "window29.csv").iloc[0]
    assert within["activity_status"] == "recently_traded"
    assert bool(within["is_new_listing"]) is True
    assert within["lifecycle_status"] == "newly_listed"
    assert outside["activity_status"] == "not_recently_traded"
    assert bool(outside["is_new_listing"]) is False
    assert outside["lifecycle_status"] == "listed_not_recently_traded"


def test_registry_build_is_deterministic_with_fixed_reference_time(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    _write_master(
        master_path,
        [
            {"symbol": "B", "date": "2026-07-29", "close": 2.0},
            {"symbol": "A", "date": "2026-07-28", "close": 1.0},
        ],
    )
    listings = _listing_data(_official_row("B"), _official_row("A"))

    first = build_company_registry(
        listing_data=listings,
        master_path=master_path,
        output_path=output_path,
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )
    first_bytes = output_path.read_bytes()
    second = build_company_registry(
        listing_data=listings,
        master_path=master_path,
        output_path=output_path,
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert pd.read_csv(output_path)["symbol"].tolist() == ["A", "B"]


def test_optional_evidence_based_override_is_applied(tmp_path: Path) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    override_path = tmp_path / "company_overrides.csv"
    _write_master(
        master_path,
        [{"symbol": "RENAMED", "date": "2026-07-30", "close": 10.0}],
    )
    pd.DataFrame(
        [
            {
                "symbol": "RENAMED",
                "company_name_override": "Renamed Company Limited",
                "official_status_override": "suspended",
                "previous_symbol": "OLD",
                "successor_symbol": "",
                "corporate_action_type": "rename",
                "notes": "Supported by PSX notice reference",
            }
        ]
    ).to_csv(override_path, index=False)

    result = build_company_registry(
        listing_data=_listing_data(_official_row("RENAMED")),
        master_path=master_path,
        output_path=output_path,
        overrides_path=override_path,
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )

    row = pd.read_csv(output_path).iloc[0]
    assert row["company_name"] == "Renamed Company Limited"
    assert row["official_status"] == "suspended"
    assert row["lifecycle_status"] == "suspended"
    assert row["previous_symbol"] == "OLD"
    assert row["corporate_action_type"] == "rename"
    assert "Supported by PSX notice reference" in row["notes"]
    assert result.overrides_applied == 1


def test_invalid_override_is_rejected_before_registry_replacement(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    output_path.write_text("existing registry", encoding="utf-8")
    override_path = tmp_path / "company_overrides.csv"
    _write_master(
        master_path,
        [{"symbol": "A", "date": "2026-07-30", "close": 10.0}],
    )
    pd.DataFrame(
        [{"symbol": "A", "official_status_override": "probably_delisted"}]
    ).to_csv(override_path, index=False)

    with pytest.raises(OverrideValidationError, match="invalid official statuses"):
        build_company_registry(
            listing_data=_listing_data(_official_row("A")),
            master_path=master_path,
            output_path=output_path,
            overrides_path=override_path,
            reference_date=date(2026, 7, 31),
            registry_updated_at=NOW,
        )

    assert output_path.read_text(encoding="utf-8") == "existing registry"


def test_missing_company_name_remains_empty_in_registry(tmp_path: Path) -> None:
    master_path = tmp_path / "master.csv"
    output_path = tmp_path / "registry.csv"
    _write_master(
        master_path,
        [{"symbol": "NONAME", "date": "2026-07-30", "close": 10.0}],
    )

    build_company_registry(
        listing_data=_listing_data(_official_row("NONAME", company_name="")),
        master_path=master_path,
        output_path=output_path,
        overrides_path=tmp_path / "missing.csv",
        reference_date=date(2026, 7, 31),
        registry_updated_at=NOW,
    )

    row = pd.read_csv(output_path, keep_default_na=False).iloc[0]
    assert row["company_name"] == ""


def test_existing_registry_is_preserved_when_live_and_cache_both_fail(
    tmp_path: Path,
) -> None:
    class FailingClient:
        def fetch_listing_table(self, board: str, segment: str) -> str:
            raise ListingsRequestError("offline")

    output_path = tmp_path / "company_registry.csv"
    output_path.write_text("existing valid registry", encoding="utf-8")
    master_path = tmp_path / "master.csv"
    _write_master(master_path, [])

    with pytest.raises(ListingsUnavailableError, match="No valid cached"):
        refresh_and_build_registry(
            client=FailingClient(),
            current_snapshot_path=tmp_path / "listings" / "current.csv",
            snapshots_dir=tmp_path / "listings",
            master_path=master_path,
            output_path=output_path,
            overrides_path=tmp_path / "missing.csv",
            reference_date=date(2026, 7, 31),
            refreshed_at=NOW,
        )

    assert output_path.read_text(encoding="utf-8") == "existing valid registry"
