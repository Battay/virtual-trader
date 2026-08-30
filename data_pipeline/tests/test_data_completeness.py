"""Offline coverage for unified completeness and selective repair controls."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.backfill import BackfillDateResult
from data_pipeline.src.data_completeness import (
    CsvDateRecord,
    DataCompletenessError,
    DateClassification,
    ParquetDateRecord,
    ParquetDateState,
    build_data_completeness_inventory,
    clear_visible_selection,
    fetch_selected_dates,
    load_trading_date_evidence,
    reconcile_pending_source_dates,
    repair_selected_parquet_dates,
    select_visible_actionable,
    update_visible_selection,
    validate_fetch_selection,
)
from data_pipeline.src.maintenance_history import (
    MaintenanceHistoryError,
    append_maintenance_operation,
    load_maintenance_history,
    new_operation,
)
from data_pipeline.src.native_market_pipeline import (
    CANONICAL_MARKET_COLUMNS,
    NativeMarketPaths,
    NativeMarketPipelineError,
    canonical_content_hash,
    canonical_content_hashes_by_date,
    full_rebuild,
    repair_daily_parquet_partitions,
)


def _row(day: str, symbol: str = "AAA", close: float = 10.0) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": day,
        "ldcp": close - 1.0,
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "change": 1.0,
        "change_percent": 1.0,
        "volume": 100,
    }


def _write_source(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_listings(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "company_name": "Alpha Limited",
                "security_type": "ordinary_equity",
                "sector": "COMMERCIAL BANKS",
                "board": "Main",
                "listing_segment": "normal_counter",
                "clearing_type": "NC",
                "listed_in": "ALLSHR",
                "shares": 1,
                "free_float": 1,
                "officially_listed": True,
                "official_status": "listed",
                "non_compliance_reason": "",
                "source": "fixture",
                "listing_refreshed_at": "2026-08-02T00:00:00+05:00",
                "snapshot_date": "2026-08-02",
            }
        ]
    ).to_csv(path, index=False)
    return path


def _paths(root: Path) -> NativeMarketPaths:
    return NativeMarketPaths(
        master_csv=root / "master/psx_master.csv",
        symbol_csv_dir=root / "processed/market_symbols",
        daily_parquet_dir=root / "parquet/daily",
        consolidated_parquet=root / "parquet/market.parquet",
        state_json=root / "metadata/native_market_pipeline_state.json",
    )


def _write_evidence(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "trading_date_evidence_v1",
                "source_snapshot": {"path": "fixture"},
                "confirmed_non_trading_dates": ["2024-01-03"],
                "likely_non_trading_dates": ["2024-01-04"],
                "source_anomaly_dates": ["2024-01-05"],
                "not_final_dates": ["2024-01-08"],
                "failed_retryable_dates": ["2024-01-09"],
                "date_details": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _build(
    tmp_path: Path,
    source_dates: tuple[str, ...] = ("2024-01-02",),
) -> tuple[NativeMarketPaths, Path]:
    sources = tmp_path / "raw/csv"
    for index, value in enumerate(source_dates):
        _write_source(
            sources / f"market_{value}.csv",
            [_row(value, close=10.0 + index)],
        )
    paths = _paths(tmp_path)
    full_rebuild(
        source_csv_dir=sources,
        listings_path=_write_listings(tmp_path / "listings.csv"),
        paths=paths,
    )
    return paths, sources


def _inventory(tmp_path: Path):
    paths, sources = _build(tmp_path)
    return build_data_completeness_inventory(
        raw_csv_dir=sources,
        raw_html_dir=tmp_path / "raw/html",
        daily_parquet_dir=paths.daily_parquet_dir,
        master_csv_path=paths.master_csv,
        consolidated_path=paths.consolidated_parquet,
        native_state_path=paths.state_json,
        backfill_state_path=tmp_path / "metadata/backfill.json",
        automation_config_path=tmp_path / "metadata/automation.json",
        evidence_path=_write_evidence(tmp_path / "metadata/evidence.json"),
        history_path=tmp_path / "metadata/history.json",
        end_date=date(2024, 1, 10),
        now=datetime(2024, 1, 11, tzinfo=timezone.utc),
    )


def _csv_record(day: date, classification: DateClassification, actionable: bool) -> CsvDateRecord:
    return CsvDateRecord(
        day,
        classification,
        "MISSING",
        "No retained HTML",
        0,
        "not_attempted",
        None,
        actionable,
        "fixture",
    )


def test_date_classification_keeps_weekends_and_evidence_distinct(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    by_date = {item.trading_date: item for item in inventory.csv_records}

    assert by_date[date(2024, 1, 2)].classification == DateClassification.CURRENT
    assert by_date[date(2024, 1, 3)].classification == DateClassification.CONFIRMED_NON_TRADING
    assert by_date[date(2024, 1, 4)].classification == DateClassification.LIKELY_NON_TRADING
    assert by_date[date(2024, 1, 5)].classification == DateClassification.SOURCE_ANOMALY
    assert by_date[date(2024, 1, 6)].classification == DateClassification.WEEKEND
    assert by_date[date(2024, 1, 7)].classification == DateClassification.WEEKEND
    assert by_date[date(2024, 1, 8)].classification == DateClassification.NOT_FINAL
    assert by_date[date(2024, 1, 9)].classification == DateClassification.FAILED_RETRYABLE
    assert by_date[date(2024, 1, 10)].classification == DateClassification.MISSING
    assert not by_date[date(2024, 1, 3)].actionable
    assert not by_date[date(2024, 1, 5)].actionable
    assert by_date[date(2024, 1, 8)].actionable
    assert inventory.csv_summary.actionable_missing_dates == 3


def test_inventory_reports_master_parity_and_current_daily_partition(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)

    assert inventory.parquet_summary.current == 1
    assert inventory.parquet_summary.missing == 0
    assert inventory.master_csv.integrity_status == "PASS"
    assert inventory.master_parquet.integrity_status == "PASS"
    assert inventory.master_parity.key_parity
    assert inventory.master_parity.logical_content_parity
    assert inventory.pending_source_dates == ()


def test_vectorized_daily_hashes_match_individual_canonical_slices(
    tmp_path: Path,
) -> None:
    paths, _ = _build(tmp_path, ("2024-01-02", "2024-01-03"))
    canonical = pq.read_table(paths.consolidated_parquet).to_pandas()

    hashes = canonical_content_hashes_by_date(canonical)

    assert hashes == {
        trading_date: canonical_content_hash(
            canonical.loc[
                pd.to_datetime(canonical["market_date"]).dt.date == trading_date
            ].reset_index(drop=True)
        )
        for trading_date in (date(2024, 1, 2), date(2024, 1, 3))
    }


def test_invalid_csv_is_visible_but_not_fetchable(tmp_path: Path) -> None:
    paths, sources = _build(tmp_path)
    bad = sources / "market_2024-01-10.csv"
    bad.write_text("symbol,date\nAAA,wrong\n", encoding="utf-8")
    inventory = build_data_completeness_inventory(
        raw_csv_dir=sources,
        raw_html_dir=tmp_path / "html",
        daily_parquet_dir=paths.daily_parquet_dir,
        master_csv_path=paths.master_csv,
        consolidated_path=paths.consolidated_parquet,
        native_state_path=paths.state_json,
        backfill_state_path=tmp_path / "backfill.json",
        automation_config_path=tmp_path / "automation.json",
        evidence_path=_write_evidence(tmp_path / "evidence.json"),
        history_path=tmp_path / "history.json",
        end_date=date(2024, 1, 10),
        now=datetime(2024, 1, 11, tzinfo=timezone.utc),
    )
    record = next(item for item in inventory.csv_records if item.trading_date == date(2024, 1, 10))
    assert record.classification == DateClassification.INVALID_SOURCE
    assert not record.actionable
    with pytest.raises(DataCompletenessError, match="not currently actionable"):
        validate_fetch_selection((record.trading_date,), inventory.csv_records)


def test_selection_helpers_preserve_hidden_membership() -> None:
    first, second, third = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    records = (
        _csv_record(first, DateClassification.MISSING, True),
        _csv_record(second, DateClassification.SOURCE_ANOMALY, False),
    )

    assert update_visible_selection((third,), (first, second), (first,)) == (first, third)
    assert select_visible_actionable((third,), records) == (first, third)
    assert clear_visible_selection((first, third), (first, second)) == (third,)


def test_selected_fetch_uses_exact_membership_and_persists_result(tmp_path: Path) -> None:
    first, second = date(2024, 1, 8), date(2024, 1, 9)
    records = (
        _csv_record(first, DateClassification.MISSING, True),
        _csv_record(second, DateClassification.MISSING, True),
    )
    calls: list[date] = []
    reconciled: list[tuple[Path, ...]] = []

    def collector(value: date) -> BackfillDateResult:
        calls.append(value)
        return BackfillDateResult(value, "successful", "ok", tmp_path / f"market_{value}.csv", valid_rows=1)

    native = SimpleNamespace(
        rows_added=2,
        daily_parquets_written=1,
        symbol_csvs_written=1,
        latest_date="2024-01-09",
        source_set_hash="source",
        content_hash="content",
    )

    def reconciler(paths, **kwargs):
        reconciled.append(tuple(paths))
        return SimpleNamespace(native=native)

    result = fetch_selected_dates(
        (second,),
        records,
        collector=collector,
        reconciler=reconciler,
        history_path=tmp_path / "history.json",
    )

    assert calls == [second]
    assert len(reconciled) == 1
    assert result.status == "COMPLETED"
    history = load_maintenance_history(tmp_path / "history.json")
    assert history.entries[0].requested_dates == (second.isoformat(),)
    assert history.entries[0].per_date_results[0].status == "successful"


def test_selected_fetch_is_idempotent_when_source_is_already_downloaded(
    tmp_path: Path,
) -> None:
    trading_date = date(2024, 1, 8)
    record = _csv_record(trading_date, DateClassification.MISSING, True)
    source = tmp_path / f"market_{trading_date.isoformat()}.csv"
    reconciled: list[tuple[Path, ...]] = []
    native = SimpleNamespace(
        rows_added=0,
        daily_parquets_written=0,
        symbol_csvs_written=0,
        latest_date="2024-01-08",
        source_set_hash="source",
        content_hash="content",
    )

    result = fetch_selected_dates(
        (trading_date,),
        (record,),
        collector=lambda value: BackfillDateResult(
            value,
            "already_downloaded",
            "Valid daily CSV already exists",
            source,
            valid_rows=1,
        ),
        reconciler=lambda paths, **kwargs: (
            reconciled.append(tuple(paths)) or SimpleNamespace(native=native)
        ),
        history_path=tmp_path / "history.json",
    )

    assert result.status == "COMPLETED"
    assert reconciled == [(source,)]
    operation = load_maintenance_history(tmp_path / "history.json").entries[0]
    assert operation.artifact_status == {
        "canonical_master_csv": "CURRENT",
        "consolidated_parquet": "CURRENT",
        "daily_partitions_affected": 0,
        "symbol_artifacts_affected": 0,
        "logical_parity": "PASS",
    }


def test_failed_selected_fetch_never_reconciles(tmp_path: Path) -> None:
    trading_date = date(2024, 1, 8)
    record = _csv_record(trading_date, DateClassification.MISSING, True)
    reconciled = False

    def reconciler(*args, **kwargs):
        nonlocal reconciled
        reconciled = True
        raise AssertionError("must not reconcile")

    result = fetch_selected_dates(
        (trading_date,),
        (record,),
        collector=lambda value: BackfillDateResult(value, "failed", "invalid"),
        reconciler=reconciler,
        history_path=tmp_path / "history.json",
    )

    assert result.status == "FAILED"
    assert not reconciled


def test_parquet_inventory_detects_missing_stale_corrupt_and_orphan(tmp_path: Path) -> None:
    paths, sources = _build(
        tmp_path,
        ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
    )
    (paths.daily_parquet_dir / "market_2024-01-02.parquet").unlink()
    stale_path = paths.daily_parquet_dir / "market_2024-01-03.parquet"
    stale = pq.read_table(stale_path).to_pandas()
    stale.loc[0, "close"] = 999.0
    pq.write_table(
        pa.Table.from_pandas(
            stale,
            schema=pq.ParquetFile(stale_path).schema_arrow.remove_metadata(),
            preserve_index=False,
        ),
        stale_path,
    )
    (paths.daily_parquet_dir / "market_2024-01-04.parquet").write_bytes(b"not parquet")
    orphan = pq.read_table(paths.daily_parquet_dir / "market_2024-01-05.parquet")
    pq.write_table(orphan, paths.daily_parquet_dir / "market_2024-01-06.parquet")
    inventory = build_data_completeness_inventory(
        raw_csv_dir=sources,
        raw_html_dir=tmp_path / "html",
        daily_parquet_dir=paths.daily_parquet_dir,
        master_csv_path=paths.master_csv,
        consolidated_path=paths.consolidated_parquet,
        native_state_path=paths.state_json,
        backfill_state_path=tmp_path / "backfill.json",
        automation_config_path=tmp_path / "automation.json",
        evidence_path=_write_evidence(tmp_path / "evidence.json"),
        history_path=tmp_path / "history.json",
        end_date=date(2024, 1, 10),
        now=datetime(2024, 1, 11, tzinfo=timezone.utc),
    )
    by_date = {item.trading_date: item.state for item in inventory.parquet_records}
    assert by_date[date(2024, 1, 2)] == ParquetDateState.MISSING
    assert by_date[date(2024, 1, 3)] == ParquetDateState.STALE
    assert by_date[date(2024, 1, 4)] == ParquetDateState.CORRUPT
    assert by_date[date(2024, 1, 5)] == ParquetDateState.CURRENT
    assert by_date[date(2024, 1, 6)] == ParquetDateState.ORPHAN


def test_selective_daily_repair_touches_only_selected_partition(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path, ("2024-01-02", "2024-01-03"))
    selected_path = paths.daily_parquet_dir / "market_2024-01-02.parquet"
    untouched_path = paths.daily_parquet_dir / "market_2024-01-03.parquet"
    master_before = hashlib.sha256(paths.master_csv.read_bytes()).hexdigest()
    consolidated_before = hashlib.sha256(paths.consolidated_parquet.read_bytes()).hexdigest()
    untouched_before = hashlib.sha256(untouched_path.read_bytes()).hexdigest()
    selected_path.unlink()

    result = repair_daily_parquet_partitions((date(2024, 1, 2),), paths=paths)

    assert result.repaired_dates == ("2024-01-02",)
    assert hashlib.sha256(untouched_path.read_bytes()).hexdigest() == untouched_before
    assert hashlib.sha256(paths.master_csv.read_bytes()).hexdigest() == master_before
    assert hashlib.sha256(paths.consolidated_parquet.read_bytes()).hexdigest() == consolidated_before
    repaired = pq.read_table(selected_path).to_pandas()
    canonical = pq.read_table(
        paths.consolidated_parquet,
        filters=[("market_date", "=", date(2024, 1, 2))],
    ).to_pandas()
    assert canonical_content_hash(repaired) == canonical_content_hash(canonical)
    noop = repair_daily_parquet_partitions((date(2024, 1, 2),), paths=paths)
    assert noop.idempotent_noop


def test_selective_daily_repair_rolls_back_on_promotion_failure(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path, ("2024-01-02", "2024-01-03"))
    targets = [
        paths.daily_parquet_dir / "market_2024-01-02.parquet",
        paths.daily_parquet_dir / "market_2024-01-03.parquet",
        paths.state_json,
    ]
    before = {path: path.read_bytes() for path in targets}
    targets[0].write_bytes(b"bad")
    targets[1].write_bytes(b"bad")
    damaged = {path: path.read_bytes() for path in targets}
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated")

    with pytest.raises(RuntimeError, match="simulated"):
        repair_daily_parquet_partitions(
            (date(2024, 1, 2), date(2024, 1, 3)),
            paths=paths,
            before_promote=fail_second,
        )
    assert {path: path.read_bytes() for path in targets} == damaged
    assert before[paths.state_json] == paths.state_json.read_bytes()


def test_repair_wrapper_rejects_orphan_and_records_valid_repair(tmp_path: Path) -> None:
    selected = date(2024, 1, 2)
    records = (
        ParquetDateRecord(selected, ParquetDateState.MISSING, None, None, 1, "a", None, "missing"),
        ParquetDateRecord(date(2024, 1, 3), ParquetDateState.ORPHAN, None, 1, None, None, "b", "orphan"),
    )
    result = SimpleNamespace(
        requested_dates=(selected.isoformat(),),
        repaired_dates=(selected.isoformat(),),
        already_current_dates=(),
        daily_parquets_written=1,
        master_csv_status="CURRENT",
        consolidated_parquet_status="CURRENT",
        logical_parity=True,
        source_set_hash="source",
        content_hash="content",
        latest_date=selected.isoformat(),
        idempotent_noop=False,
    )
    output = repair_selected_parquet_dates(
        (selected,),
        records,
        repairer=lambda dates, paths: result,
        paths=_paths(tmp_path),
        lock_path=tmp_path / "lock",
        history_path=tmp_path / "history.json",
    )
    assert output is result
    assert load_maintenance_history(tmp_path / "history.json").entries[0].operation_type == "PARQUET_REPAIR_SELECTED"
    with pytest.raises(DataCompletenessError, match="not repairable"):
        repair_selected_parquet_dates(
            (date(2024, 1, 3),), records, repairer=lambda dates, paths: result,
            paths=_paths(tmp_path), lock_path=tmp_path / "lock2",
            history_path=tmp_path / "history.json",
        )


def test_pending_source_reconciliation_validates_without_http(tmp_path: Path) -> None:
    trading_date = date(2024, 1, 2)
    source = _write_source(tmp_path / "raw/market_2024-01-02.csv", [_row("2024-01-02")])
    seen: list[Path] = []
    native = SimpleNamespace(
        rows_added=1, daily_parquets_written=1, symbol_csvs_written=1,
        latest_date="2024-01-02", source_set_hash="source", content_hash="content",
    )

    def reconciler(paths, **kwargs):
        seen.extend(paths)
        return SimpleNamespace(native=native)

    reconcile_pending_source_dates(
        (trading_date,),
        raw_csv_dir=source.parent,
        reconciler=reconciler,
        history_path=tmp_path / "history.json",
    )
    assert seen == [source]
    assert load_maintenance_history(tmp_path / "history.json").entries[0].operation_type == "MASTER_RECONCILE"


def test_history_survives_reload_and_malformed_history_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    operation = new_operation(
        "FETCH_SELECTED",
        requested_dates=("2024-01-02",),
        operation_id="fixed",
        timestamp="2024-01-02T00:00:00+00:00",
    )
    append_maintenance_operation(operation, path)
    assert load_maintenance_history(path).entries == (operation,)
    path.write_text("{bad", encoding="utf-8")
    assert load_maintenance_history(path).error
    with pytest.raises(MaintenanceHistoryError):
        append_maintenance_operation(operation, path)


def test_evidence_ledger_is_deterministic_and_rejects_duplicate_classes(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path / "evidence.json")
    first = load_trading_date_evidence(path)
    second = load_trading_date_evidence(path)
    assert first == second
    payload = json.loads(path.read_text())
    payload["likely_non_trading_dates"].append("2024-01-03")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "duplicate" in (load_trading_date_evidence(path).error or "")


def test_repair_rejects_noncanonical_date(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path)
    with pytest.raises(NativeMarketPipelineError, match="absent"):
        repair_daily_parquet_partitions((date(2024, 1, 9),), paths=paths)
