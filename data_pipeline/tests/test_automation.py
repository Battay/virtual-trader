"""Offline tests for persisted scheduling state and standalone orchestration."""

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

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
from data_pipeline.src.csv_store import MasterBuildResult
from data_pipeline.src.main import CollectionResult
from data_pipeline.src.updater import IncrementalUpdateResult


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

    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=updater,
        master_builder=builder,
    )

    assert result.status == "disabled"
    assert result.exit_code == 0
    assert calls == []


def test_enabled_runner_calls_updater_then_master_builder(tmp_path: Path) -> None:
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

    now = datetime(2026, 7, 27, 20, 30, tzinfo=timezone.utc)
    result = run_scheduled_update(
        config_path=config_path,
        lock_path=tmp_path / "lock",
        updater=updater,
        master_builder=builder,
        now=now,
    )

    assert [name for name, _ in calls] == ["updater", "builder"]
    updater_arguments = calls[0][1]
    assert isinstance(updater_arguments, dict)
    assert updater_arguments["end_date"] == date(2026, 7, 28)
    assert updater_arguments["bootstrap_start_date"] == date(2026, 7, 1)
    assert result.status == "success"
    assert result.exit_code == 0
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
        master_builder=lambda: _master_result(tmp_path / "master.csv"),
        now=datetime(2026, 7, 26, 12, 15, tzinfo=timezone.utc),
    )

    assert result.status == "success"
    assert result.exit_code == 0
    assert result.update_result is not None
    assert result.update_result.skipped_dates == (date(2026, 7, 26),)


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
