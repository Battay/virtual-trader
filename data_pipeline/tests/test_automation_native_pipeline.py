"""Offline acceptance tests for unified automation/native orchestration."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src import automation
from data_pipeline.src.automation import (
    AutomationConfig,
    AutomationRunResult,
    SourceDateDisposition,
    load_automation_config,
    reconcile_native_source_csvs,
    recover_stale_automation_state,
    run_manual_update,
    run_scheduled_update,
    save_automation_config,
)
from data_pipeline.src.company_registry import RegistryBuildResult
from data_pipeline.src.csv_store import MasterBuildResult
from data_pipeline.src.main import CollectionResult
from data_pipeline.src.native_market_pipeline import (
    NativeMarketBuildResult,
    NativeMarketPaths,
)
from data_pipeline.src.official_listings import ListingsRefreshResult
from data_pipeline.src.updater import (
    IncrementalUpdateResult,
    SourceEvidenceInventory,
    discover_source_evidence,
)
from market_intelligence.refresh_indices import IndexRefreshResult


TARGET = date(2026, 8, 27)


def _collection_result(
    *,
    successful: tuple[date, ...] = (),
    skipped: tuple[date, ...] = (),
    failed: tuple[tuple[date, str], ...] = (),
    missing: tuple[date, ...] | None = None,
    root: Path,
) -> IncrementalUpdateResult:
    requested = missing if missing is not None else tuple(
        [*successful, *skipped, *(value for value, _ in failed)]
    )
    outputs = tuple(root / f"market_{value.isoformat()}.csv" for value in successful)
    return IncrementalUpdateResult(
        requested_end_date=TARGET,
        available_dates_before=(),
        latest_stored_date=max(successful) if successful else None,
        missing_dates=requested,
        collection=CollectionResult(
            start_date=requested[0] if requested else TARGET,
            end_date=requested[-1] if requested else TARGET,
            total_processed=len(requested),
            successful_dates=successful,
            skipped_dates=skipped,
            failed_dates=failed,
            output_csv_paths=outputs,
        ),
    )


def _evidence(
    *,
    local: tuple[date, ...] = (),
    manifest: tuple[date, ...] = (),
    external: tuple[date, ...] = (),
    inconsistencies: tuple[str, ...] = (),
) -> SourceEvidenceInventory:
    return SourceEvidenceInventory(
        local_csv_dates=local,
        native_manifest_dates=manifest,
        external_manifest_dates=external,
        parquet_dates=manifest,
        parquet_only_dates=(),
        inconsistencies=inconsistencies,
    )


def _native_result(root: Path, *, rows_added: int = 2, daily: int = 1) -> NativeMarketBuildResult:
    paths = NativeMarketPaths(
        master_csv=root / "master.csv",
        symbol_csv_dir=root / "symbols",
        daily_parquet_dir=root / "daily",
        consolidated_parquet=root / "market.parquet",
        state_json=root / "state.json",
    )
    return NativeMarketBuildResult(
        operation="incremental",
        source_files=1,
        source_dates=(TARGET.isoformat(),),
        rows_read=rows_added,
        rows_accepted=rows_added,
        rows_rejected=0,
        duplicate_count=0,
        master_rows=100,
        consolidated_rows=100,
        symbol_count=2,
        earliest_date="2020-01-01",
        latest_date=TARGET.isoformat(),
        sector_matched_symbols=2,
        schema_version="native_market_record_v1",
        content_hash="content",
        source_set_hash="source",
        consolidated_sha256="sha",
        status="completed",
        paths=paths,
        rows_added=rows_added,
        rows_replaced=0,
        daily_parquets_written=daily,
        symbol_csvs_written=2,
    )


def _listing_result(root: Path) -> ListingsRefreshResult:
    return ListingsRefreshResult(
        data=pd.DataFrame([{"symbol": "AAA"}]),
        current_snapshot_path=root / "listings.csv",
        dated_snapshot_path=None,
        row_count=1,
        duplicate_count=0,
        used_cache=False,
        listing_refreshed_at="2026-08-27T17:00:00+05:00",
        message="fixture",
    )


def _registry_result(root: Path) -> RegistryBuildResult:
    return RegistryBuildResult(
        output_path=root / "registry.csv",
        total_registry_symbols=1,
        currently_listed=1,
        recently_traded=1,
        listed_not_recently_traded=0,
        new_listings=0,
        historical_only=0,
        suspended=0,
        non_compliant=0,
        delisted=0,
        unknown=0,
        registry_updated_at="2026-08-27T17:00:00+05:00",
        listing_refreshed_at="2026-08-27T17:00:00+05:00",
        cached_listings_used=False,
        overrides_applied=0,
    )


def _master_result(root: Path) -> MasterBuildResult:
    return MasterBuildResult(
        output_path=root / "legacy.csv",
        total_rows=10,
        unique_symbols=1,
        earliest_date=date(2026, 1, 1),
        latest_date=TARGET,
        duplicate_count=0,
        source_files=1,
        errors=(),
    )


def _index_result(root: Path) -> IndexRefreshResult:
    return IndexRefreshResult(
        requested_indices=("KSE100",),
        successful_indices=("KSE100",),
        failed_indices=(),
        observations_parsed=1,
        earliest_date=TARGET,
        latest_date=TARGET,
        output_paths=(root / "index.csv",),
        combined_master_path=root / "index.csv",
        cached_data_used=False,
    )


def _run(
    tmp_path: Path,
    update: IncrementalUpdateResult,
    *,
    config: AutomationConfig | None = None,
    native_updater: Any = None,
    calls: list[str] | None = None,
    evidence: SourceEvidenceInventory | None = None,
) -> AutomationRunResult:
    observed = calls if calls is not None else []
    config_path = tmp_path / "automation.json"
    save_automation_config(config or AutomationConfig(enabled=True), config_path)

    def native(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        observed.append("native")
        if native_updater is not None:
            return native_updater(paths, **kwargs)
        return _native_result(tmp_path / "native")

    return run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=lambda **kwargs: update,
        evidence_discoverer=lambda **kwargs: evidence or _evidence(),
        native_updater=native,
        listing_refresher=lambda **kwargs: (
            observed.append("listings") or _listing_result(tmp_path)
        ),
        registry_builder=lambda **kwargs: (
            observed.append("registry") or _registry_result(tmp_path)
        ),
        index_refresher=lambda: (
            observed.append("indices") or _index_result(tmp_path)
        ),
        master_builder=lambda: (
            observed.append("legacy_master") or _master_result(tmp_path)
        ),
        symbol_ai_builder=lambda: observed.append("symbol_ai"),
        master_ai_builder=lambda: observed.append("master_ai"),
        backfill_state_path=tmp_path / "backfill.json",
        now=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
    )


def test_no_update_needed_is_fast_and_does_not_refresh_or_rebuild(tmp_path: Path) -> None:
    calls: list[str] = []
    update = _collection_result(root=tmp_path, missing=())

    result = _run(tmp_path, update, calls=calls)

    assert result.status == "no_update_needed"
    assert result.native_update_succeeded is True
    assert calls == []
    saved = load_automation_config(tmp_path / "automation.json")
    assert saved.last_status == "no_update_needed"
    assert saved.last_run is not None and saved.last_run.finished_at is not None


@pytest.mark.parametrize(
    "successful",
    [
        (date(2026, 8, 27),),
        (date(2024, 5, 15),),
        (date(2026, 8, 26), date(2026, 8, 27)),
    ],
)
def test_latest_old_and_multiple_downloads_use_the_same_incremental_native_path(
    tmp_path: Path, successful: tuple[date, ...]
) -> None:
    update = _collection_result(root=tmp_path, successful=successful)
    captured: list[tuple[Path, ...]] = []

    def native(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        captured.append(tuple(Path(value) for value in paths))
        return _native_result(tmp_path / "native", rows_added=len(successful), daily=len(successful))

    result = _run(tmp_path, update, native_updater=native)

    assert result.status == "success"
    assert [path.name for path in captured[0]] == [
        f"market_{value.isoformat()}.csv" for value in successful
    ]
    assert result.native_result is not None
    assert result.native_result.daily_parquets_written == len(successful)


def test_failed_network_date_is_finalized_without_native_false_success(tmp_path: Path) -> None:
    failed_date = date(2026, 8, 27)
    update = _collection_result(
        root=tmp_path, failed=((failed_date, "RequestError: offline"),)
    )

    result = _run(tmp_path, update)

    assert result.status == "failed"
    assert result.native_update_succeeded is True  # no source artifact existed to apply
    saved = load_automation_config(tmp_path / "automation.json")
    assert saved.last_status == "failed"
    assert saved.last_run is not None
    assert saved.last_run.dates_failed == ((failed_date.isoformat(), "RequestError: offline"),)


def test_empty_weekday_is_deferred_and_status_is_finalized(tmp_path: Path) -> None:
    skipped = date(2026, 8, 27)
    update = _collection_result(root=tmp_path, skipped=(skipped,))

    result = _run(tmp_path, update)

    assert result.status == "partial_success"
    saved = load_automation_config(tmp_path / "automation.json")
    assert saved.last_status == "partial_success"
    assert skipped in saved.deferred_empty_dates
    assert saved.last_run is not None and saved.last_run.finished_at is not None


def test_ai_rebuild_runs_only_after_native_success(tmp_path: Path) -> None:
    calls: list[str] = []
    update = _collection_result(root=tmp_path, successful=(TARGET,))

    result = _run(
        tmp_path,
        update,
        config=AutomationConfig(enabled=True, rebuild_ai_datasets=True),
        calls=calls,
    )

    assert result.status == "success"
    assert calls == [
        "listings",
        "native",
        "registry",
        "indices",
        "legacy_master",
        "symbol_ai",
        "master_ai",
    ]
    assert result.ai_rebuild_succeeded is True


def test_native_failure_blocks_ai_and_never_leaves_running_state(tmp_path: Path) -> None:
    calls: list[str] = []
    update = _collection_result(root=tmp_path, successful=(TARGET,))

    def fail_native(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        raise RuntimeError("native validation failed")

    result = _run(
        tmp_path,
        update,
        config=AutomationConfig(enabled=True, rebuild_ai_datasets=True),
        native_updater=fail_native,
        calls=calls,
    )

    assert result.status == "failed"
    assert result.native_update_succeeded is False
    assert "indices" not in calls and "symbol_ai" not in calls
    saved = load_automation_config(tmp_path / "automation.json")
    assert saved.last_status == "failed"
    assert saved.last_run is not None
    assert saved.last_run.native_update_status == "failed"


def test_dead_lock_owner_and_running_state_recover_immediately(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    lock_path = tmp_path / "lock"
    save_automation_config(
        AutomationConfig(last_status="running", last_message="stuck"), config_path
    )
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "created_at": "2026-08-28T00:00:00+05:00"}),
        encoding="utf-8",
    )

    recovered = recover_stale_automation_state(
        config_path=config_path,
        lock_path=lock_path,
        now=datetime(2026, 8, 27, 22, tzinfo=timezone.utc),
    )

    assert recovered.last_status == "failed"
    assert "abandoned" in recovered.last_message
    assert not lock_path.exists()


def test_recovered_stale_run_is_carried_into_next_run_audit(tmp_path: Path) -> None:
    config_path = tmp_path / "automation.json"
    lock_path = tmp_path / "lock"
    save_automation_config(
        AutomationConfig(
            enabled=True,
            last_status="running",
            last_message="stuck",
            last_run=automation.AutomationRunAudit(
                started_at="2026-08-27T17:00:00+05:00",
                finished_at=None,
                target_end_date=TARGET.isoformat(),
            ),
        ),
        config_path,
    )
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "created_at": "2026-08-27T17:00:00+05:00"}),
        encoding="utf-8",
    )

    result = run_manual_update(
        end_date=TARGET,
        config_path=config_path,
        lock_path=lock_path,
        updater=lambda **kwargs: _collection_result(root=tmp_path, missing=()),
        evidence_discoverer=lambda **kwargs: _evidence(),
        backfill_state_path=tmp_path / "backfill.json",
        now=datetime(2026, 8, 27, 22, tzinfo=timezone.utc),
    )

    assert result.status == "no_update_needed"
    assert result.audit is not None and result.audit.stale_run_recovered is True


def test_manual_and_scheduled_wrappers_delegate_to_same_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    result = AutomationRunResult("success", "ok", 0)

    def shared(config: AutomationConfig, **kwargs: Any) -> AutomationRunResult:
        calls.append(kwargs["end_date"].isoformat())
        return result

    monkeypatch.setattr(automation, "run_update_orchestration", shared)
    config_path = tmp_path / "automation.json"
    save_automation_config(AutomationConfig(enabled=True), config_path)

    manual = run_manual_update(
        end_date=TARGET,
        config_path=config_path,
        lock_path=tmp_path / "manual.lock",
    )
    scheduled = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "scheduled.lock",
        now=datetime(2026, 8, 27, 20, tzinfo=timezone.utc),
    )

    assert manual is result and scheduled is result
    assert calls == [TARGET.isoformat(), date(2026, 8, 28).isoformat()]


def test_source_inventory_does_not_treat_parquet_only_date_as_csv_evidence(
    tmp_path: Path,
) -> None:
    local_date = date(2026, 8, 19)
    external_date = date(2026, 8, 20)
    parquet_only = date(2026, 8, 21)
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": local_date.isoformat(),
                "ldcp": 9.0,
                "open": 9.5,
                "high": 10.5,
                "low": 9.0,
                "close": 10.0,
                "change": 1.0,
                "change_percent": 11.11,
                "volume": 10,
            }
        ]
    ).to_csv(raw / f"market_{local_date.isoformat()}.csv", index=False)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "source_set_hash": "sources",
                "canonical_content_hash": "content",
                "source_files": [
                    {
                        "name": f"market_{external_date.isoformat()}.csv",
                        "sha256": "abc",
                        "size_bytes": 1,
                        "origin": "external_validated_csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    table = pa.table(
        {"market_date": pa.array([local_date, external_date, parquet_only], type=pa.date32())}
    ).replace_schema_metadata(
        {b"source_set_hash": b"sources", b"canonical_content_hash": b"content"}
    )
    parquet_path = tmp_path / "market.parquet"
    pq.write_table(table, parquet_path)

    inventory = discover_source_evidence(
        csv_dir=raw,
        native_state_path=state_path,
        parquet_path=parquet_path,
    )

    assert inventory.accepted_source_dates == (local_date, external_date)
    assert inventory.external_manifest_dates == (external_date,)
    assert inventory.parquet_only_dates == (parquet_only,)
    assert any("Parquet date" in value for value in inventory.inconsistencies)


def test_weekend_csv_is_reported_but_never_accepted_for_native_ingestion(
    tmp_path: Path,
) -> None:
    weekend = date(2026, 8, 23)
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": weekend.isoformat(),
                "ldcp": 9.0,
                "open": 9.5,
                "high": 10.5,
                "low": 9.0,
                "close": 10.0,
                "change": 1.0,
                "change_percent": 11.11,
                "volume": 10,
            }
        ]
    ).to_csv(raw / f"market_{weekend.isoformat()}.csv", index=False)

    inventory = discover_source_evidence(
        csv_dir=raw,
        native_state_path=tmp_path / "state.json",
        parquet_path=tmp_path / "market.parquet",
    )

    assert inventory.local_weekend_dates == (weekend,)
    assert weekend not in inventory.accepted_source_dates
    assert any("weekend" in value for value in inventory.inconsistencies)
    assert not any("not yet represented" in value for value in inventory.inconsistencies)


def test_pending_weekend_csv_is_not_passed_to_native_updater(tmp_path: Path) -> None:
    weekday = date(2026, 8, 24)
    weekend = date(2026, 8, 23)
    captured: list[Path] = []

    def native(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        captured.extend(Path(value) for value in paths)
        return _native_result(tmp_path / "native")

    result = _run(
        tmp_path,
        _collection_result(root=tmp_path, missing=()),
        native_updater=native,
        evidence=_evidence(local=(weekend, weekday)),
    )

    assert result.status == "success"
    assert [path.name for path in captured] == [f"market_{weekday.isoformat()}.csv"]


def test_orchestrator_never_receives_test_partition_inputs(tmp_path: Path) -> None:
    update = _collection_result(root=tmp_path, successful=(TARGET,))
    observed: list[Path] = []

    def native(paths: object, **kwargs: Any) -> NativeMarketBuildResult:
        observed.extend(Path(value) for value in paths)
        return _native_result(tmp_path / "native")

    result = _run(tmp_path, update, native_updater=native)

    assert result.status == "success"
    assert all("test" not in path.name.lower() for path in observed)
    assert all(path.name.startswith("market_") for path in observed)


def test_source_dispositions_round_trip_and_suppress_only_automatic_retries(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "automation.json"
    blocked = SourceDateDisposition(
        date(2026, 8, 14),
        "confirmed_non_trading",
        "holiday",
        "official evidence",
        False,
        "2026-08-28T03:00:00+05:00",
    )
    retryable = SourceDateDisposition(
        date(2026, 8, 28),
        "not_final",
        "today",
        "early check",
        True,
        "2026-08-28T03:00:00+05:00",
    )
    save_automation_config(
        AutomationConfig(source_date_dispositions=(blocked, retryable)), config_path
    )

    loaded = load_automation_config(config_path)
    excluded = automation._known_non_request_dates(
        backfill_state_path=tmp_path / "backfill.json",
        deferred_dates=(),
        source_dispositions=loaded.source_date_dispositions,
    )

    assert loaded.source_date_dispositions == (blocked, retryable)
    assert blocked.trading_date in excluded
    assert retryable.trading_date not in excluded


def test_lock_protected_source_reconciliation_reuses_native_stage_order(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source = tmp_path / f"market_{TARGET.isoformat()}.csv"
    source.write_text("fixture", encoding="utf-8")

    result = reconcile_native_source_csvs(
        (source,),
        reference_date=TARGET,
        lock_path=tmp_path / "lock",
        listing_refresher=lambda **kwargs: (
            calls.append("listings") or _listing_result(tmp_path)
        ),
        native_updater=lambda paths, **kwargs: (
            calls.append("native") or _native_result(tmp_path / "native")
        ),
        registry_builder=lambda **kwargs: (
            calls.append("registry") or _registry_result(tmp_path)
        ),
        now=datetime(2026, 8, 27, 22, tzinfo=timezone.utc),
    )

    assert calls == ["listings", "native", "registry"]
    assert result.native.rows_added == 2
    assert not (tmp_path / "lock").exists()
