"""Offline tests for persisted scheduling state and standalone orchestration."""

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data_pipeline.src.automation import (
    AutomationConfig,
    UpdateAlreadyRunning,
    UpdateLock,
    karachi_today,
    load_automation_config,
    run_scheduled_update,
    save_automation_config,
)
from data_pipeline.src.company_registry import RegistryBuildResult
from data_pipeline.src.csv_store import MasterBuildResult
from data_pipeline.src.main import CollectionResult
from data_pipeline.src.official_listings import ListingsRefreshResult
from data_pipeline.src.native_market_pipeline import (
    NativeMarketBuildResult,
    NativeMarketPaths,
)
from data_pipeline.src.updater import IncrementalUpdateResult, SourceEvidenceInventory
from market_intelligence.refresh_indices import IndexRefreshResult


def _update_result(end_date: date) -> IncrementalUpdateResult:
    return IncrementalUpdateResult(
        requested_end_date=end_date,
        available_dates_before=(),
        latest_stored_date=end_date,
        missing_dates=(end_date,),
        collection=CollectionResult(
            start_date=end_date,
            end_date=end_date,
            total_processed=1,
            successful_dates=(end_date,),
            skipped_dates=(),
            failed_dates=(),
            output_csv_paths=(Path(f"market_{end_date.isoformat()}.csv"),),
        ),
    )


def _master_result(path: Path) -> MasterBuildResult:
    return MasterBuildResult(
        output_path=path,
        total_rows=10,
        unique_symbols=2,
        earliest_date=date(2026, 7, 1),
        latest_date=date(2026, 7, 28),
        duplicate_count=0,
        source_files=2,
        errors=(),
    )


def _listings_result(path: Path, *, used_cache: bool = False) -> ListingsRefreshResult:
    return ListingsRefreshResult(
        data=pd.DataFrame([{"symbol": "OGDC"}]),
        current_snapshot_path=path,
        dated_snapshot_path=None,
        row_count=1,
        duplicate_count=0,
        used_cache=used_cache,
        listing_refreshed_at="2026-07-28T01:30:00+05:00",
        message="fixture listings",
        live_error="offline" if used_cache else None,
    )


def _registry_result(path: Path, *, cached: bool = False) -> RegistryBuildResult:
    return RegistryBuildResult(
        output_path=path,
        total_registry_symbols=3,
        currently_listed=2,
        recently_traded=2,
        listed_not_recently_traded=0,
        new_listings=1,
        historical_only=1,
        suspended=0,
        non_compliant=0,
        delisted=0,
        unknown=0,
        registry_updated_at="2026-07-28T01:30:00+05:00",
        listing_refreshed_at="2026-07-28T01:30:00+05:00",
        cached_listings_used=cached,
        overrides_applied=0,
    )


def _index_result(path: Path) -> IndexRefreshResult:
    return IndexRefreshResult(
        requested_indices=("KSE100",), successful_indices=("KSE100",),
        failed_indices=(), observations_parsed=2,
        earliest_date=date(2026, 7, 27), latest_date=date(2026, 7, 28),
        output_paths=(path,), combined_master_path=path, cached_data_used=False,
    )


def _evidence() -> SourceEvidenceInventory:
    return SourceEvidenceInventory((), (), (), (), (), ())


def _native_result(path: Path, *, latest: date = date(2026, 7, 28)) -> NativeMarketBuildResult:
    paths = NativeMarketPaths(
        master_csv=path / "master.csv",
        symbol_csv_dir=path / "symbols",
        daily_parquet_dir=path / "daily",
        consolidated_parquet=path / "market.parquet",
        state_json=path / "state.json",
    )
    return NativeMarketBuildResult(
        operation="incremental",
        source_files=1,
        source_dates=(latest.isoformat(),),
        rows_read=2,
        rows_accepted=2,
        rows_rejected=0,
        duplicate_count=0,
        master_rows=10,
        consolidated_rows=10,
        symbol_count=2,
        earliest_date="2026-07-01",
        latest_date=latest.isoformat(),
        sector_matched_symbols=2,
        schema_version="native_market_record_v1",
        content_hash="content",
        source_set_hash="sources",
        consolidated_sha256="sha",
        status="completed",
        paths=paths,
        rows_added=2,
        daily_parquets_written=1,
        symbol_csvs_written=2,
    )


def test_metadata_save_and_load_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "metadata" / "automation.json"
    config = AutomationConfig(
        enabled=True,
        bootstrap_start_date=date(2026, 7, 1),
        last_status="success",
        last_message="done",
    )

    save_automation_config(config, path)

    assert load_automation_config(path) == config
    assert list(path.parent.glob("*.tmp")) == []


def test_malformed_metadata_uses_safe_disabled_defaults(tmp_path: Path) -> None:
    path = tmp_path / "automation.json"
    path.write_text("{not valid json", encoding="utf-8")

    config = load_automation_config(path)

    assert config.enabled is False
    assert config.last_status == "configuration_error"
    assert "Could not load automation configuration" in config.last_message


def test_disabled_scheduled_runner_does_nothing(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    save_automation_config(AutomationConfig(enabled=False), config_path)
    calls: list[str] = []

    def updater(**kwargs: Any) -> IncrementalUpdateResult:
        calls.append("updater")
        return _update_result(kwargs["end_date"])

    def builder() -> MasterBuildResult:
        calls.append("builder")
        return _master_result(tmp_path / "master.csv")

    def listing_refresher(**kwargs: Any) -> ListingsRefreshResult:
        calls.append("listings")
        return _listings_result(tmp_path / "listings.csv")

    def registry_builder(**kwargs: Any) -> RegistryBuildResult:
        calls.append("registry")
        return _registry_result(tmp_path / "registry.csv")

    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=updater,
        index_refresher=lambda: _index_result(tmp_path / "indices.csv"),
        master_builder=builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
    )

    assert result.status == "disabled"
    assert result.exit_code == 0
    assert calls == []


def test_enabled_runner_uses_native_orchestration_without_optional_ai(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    save_automation_config(
        AutomationConfig(
            enabled=True,
            bootstrap_start_date=date(2026, 7, 1),
        ),
        config_path,
    )
    calls: list[tuple[str, object]] = []

    def updater(**kwargs: Any) -> IncrementalUpdateResult:
        calls.append(("updater", kwargs))
        return _update_result(kwargs["end_date"])

    def builder() -> MasterBuildResult:
        calls.append(("builder", None))
        return _master_result(tmp_path / "master.csv")

    def listing_refresher(**kwargs: Any) -> ListingsRefreshResult:
        calls.append(("listings", kwargs))
        return _listings_result(tmp_path / "listings.csv")

    def registry_builder(**kwargs: Any) -> RegistryBuildResult:
        calls.append(("registry", kwargs))
        return _registry_result(tmp_path / "registry.csv")

    def native_updater(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        calls.append(("native", kwargs))
        return _native_result(tmp_path / "native")

    now = datetime(2026, 7, 27, 20, 30, tzinfo=timezone.utc)
    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=updater,
        index_refresher=lambda: _index_result(tmp_path / "indices.csv"),
        master_builder=builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
        evidence_discoverer=lambda **kwargs: _evidence(),
        native_updater=native_updater,
        now=now,
    )

    assert [name for name, _ in calls] == [
        "updater",
        "listings",
        "native",
        "registry",
    ]
    updater_arguments = calls[0][1]
    assert isinstance(updater_arguments, dict)
    assert updater_arguments["end_date"] == date(2026, 7, 28)
    assert updater_arguments["bootstrap_start_date"] == date(2026, 7, 1)
    assert result.status == "success"
    assert result.exit_code == 0
    assert result.market_update_succeeded is True
    assert result.master_rebuild_succeeded is False
    assert result.native_update_succeeded is True
    assert result.listing_refresh_succeeded is True
    assert result.registry_rebuild_succeeded is True
    assert result.cached_listings_used is False
    saved = load_automation_config(config_path)
    assert saved.last_status == "success"
    assert saved.last_success_at is not None


def test_scheduled_runner_treats_skipped_dates_as_success(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    save_automation_config(
        AutomationConfig(
            enabled=True,
            bootstrap_start_date=date(2026, 7, 26),
        ),
        config_path,
    )

    def updater(**kwargs: Any) -> IncrementalUpdateResult:
        end_date = kwargs["end_date"]
        return IncrementalUpdateResult(
            requested_end_date=end_date,
            available_dates_before=(),
            latest_stored_date=None,
            missing_dates=(end_date,),
            collection=CollectionResult(
                start_date=end_date,
                end_date=end_date,
                total_processed=1,
                successful_dates=(),
                skipped_dates=(end_date,),
                failed_dates=(),
                output_csv_paths=(),
            ),
        )

    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=updater,
        index_refresher=lambda: _index_result(tmp_path / "indices.csv"),
        master_builder=lambda: _master_result(tmp_path / "master.csv"),
        listing_refresher=lambda **kwargs: _listings_result(
            tmp_path / "listings.csv"
        ),
        registry_builder=lambda **kwargs: _registry_result(
            tmp_path / "registry.csv"
        ),
        evidence_discoverer=lambda **kwargs: _evidence(),
        native_updater=lambda paths, **kwargs: _native_result(tmp_path / "native"),
        now=datetime(2026, 7, 26, 12, 15, tzinfo=timezone.utc),
    )

    assert result.status == "partial_success"
    assert result.exit_code == 0
    assert result.update_result is not None
    assert result.update_result.skipped_dates == (date(2026, 7, 26),)


def test_cached_listing_fallback_is_recorded_as_success(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    save_automation_config(
        AutomationConfig(enabled=True, bootstrap_start_date=date(2026, 7, 1)),
        config_path,
    )

    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=lambda **kwargs: _update_result(kwargs["end_date"]),
        index_refresher=lambda: _index_result(tmp_path / "indices.csv"),
        master_builder=lambda: _master_result(tmp_path / "master.csv"),
        listing_refresher=lambda **kwargs: _listings_result(
            tmp_path / "listings.csv",
            used_cache=True,
        ),
        registry_builder=lambda **kwargs: _registry_result(
            tmp_path / "registry.csv",
            cached=True,
        ),
        evidence_discoverer=lambda **kwargs: _evidence(),
        native_updater=lambda paths, **kwargs: _native_result(tmp_path / "native"),
        now=datetime(2026, 7, 27, 20, 30, tzinfo=timezone.utc),
    )

    assert result.status == "success"
    assert result.exit_code == 0
    assert result.cached_listings_used is True


def test_overlapping_lock_is_prevented(tmp_path: Path) -> None:
    path = tmp_path / "auto_update.lock"
    first = UpdateLock(path)
    first.acquire()
    try:
        with pytest.raises(UpdateAlreadyRunning, match="Another automation run"):
            UpdateLock(path).acquire()
    finally:
        first.release()


def test_stale_lock_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "auto_update.lock"
    path.write_text("stale", encoding="utf-8")
    old_timestamp = (datetime.now().timestamp() - timedelta(hours=7).total_seconds())
    os.utime(path, (old_timestamp, old_timestamp))

    lock = UpdateLock(path, stale_after=timedelta(hours=6))
    lock.acquire()
    try:
        assert lock.acquired is True
        assert "created_at" in path.read_text(encoding="utf-8")
    finally:
        lock.release()


def test_lock_release_does_not_remove_a_replacement_lock(tmp_path: Path) -> None:
    path = tmp_path / "auto_update.lock"
    lock = UpdateLock(path)
    lock.acquire()
    path.write_text("replacement-owner", encoding="utf-8")

    lock.release()

    assert path.read_text(encoding="utf-8") == "replacement-owner"


def test_karachi_date_uses_zoneinfo_at_utc_day_boundary() -> None:
    utc_time = datetime(2026, 7, 27, 20, 30, tzinfo=timezone.utc)

    assert karachi_today(utc_time) == date(2026, 7, 28)
