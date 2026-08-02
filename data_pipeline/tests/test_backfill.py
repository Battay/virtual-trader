"""Offline tests for resumable, sequential PSX historical backfill."""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dashboard.backfill_preview import (
    PREVIEW_INPUTS_KEY,
    PREVIEW_PLAN_KEY,
    build_preview_inputs,
    create_preview_safely,
    preview_is_stale,
    preview_status_message,
    resume_is_eligible,
    state_for_preview_range,
    store_backfill_preview,
)
from data_pipeline.src import backfill as backfill_module
from data_pipeline.src.backfill import (
    BackfillDateResult,
    BackfillPlan,
    BackfillState,
    create_backfill_plan,
    load_backfill_state,
    run_backfill,
    write_backfill_state,
)
from data_pipeline.src.main import DateProcessingResult, OUTPUT_FIELDS


FIXED_CLOCK = lambda: datetime(2026, 8, 1, 12, tzinfo=timezone.utc)


def _preview_plan(*request_dates: date) -> BackfillPlan:
    return BackfillPlan(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        total_calendar_dates=3,
        existing_successful_dates=(),
        dates_requiring_requests=request_dates,
        weekend_dates=(),
        unresolved_skipped_dates=(),
        failed_dates_eligible_for_retry=(),
        estimated_request_count=len(request_dates),
        estimated_minimum_duration_seconds=max(0, len(request_dates) - 1),
    )


def _write_daily_csv(directory: Path, trading_date: date) -> Path:
    path = directory / f"market_{trading_date.isoformat()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "symbol": "MCB",
        "date": trading_date.isoformat(),
        "ldcp": 100.0,
        "open": 101.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "change": 2.0,
        "change_percent": 2.0,
        "volume": 1_000,
    }
    pd.DataFrame([row], columns=OUTPUT_FIELDS).to_csv(path, index=False)
    return path


def _success_processor(csv_dir: Path, calls: list[date]):
    def processor(
        trading_date: date,
        client: Any,
    ) -> DateProcessingResult:
        calls.append(trading_date)
        path = _write_daily_csv(csv_dir, trading_date)
        return DateProcessingResult(
            trading_date,
            "successful",
            1,
            1,
            0,
            path,
        )

    return processor


def _skipped_result(trading_date: date, client: Any) -> DateProcessingResult:
    return DateProcessingResult(trading_date, "skipped", 0, 0, 0, None)


def test_backfill_plan_excludes_existing_dates_and_reports_weekends(
    tmp_path: Path,
) -> None:
    csv_dir = tmp_path / "csv"
    _write_daily_csv(csv_dir, date(2026, 7, 6))

    plan = create_backfill_plan(
        date(2026, 7, 4),
        date(2026, 7, 8),
        delay_seconds=2.0,
        csv_dir=csv_dir,
        today=date(2026, 8, 1),
    )

    assert plan.total_calendar_dates == 5
    assert plan.existing_successful_dates == (date(2026, 7, 6),)
    assert plan.weekend_dates == (date(2026, 7, 4), date(2026, 7, 5))
    assert plan.dates_requiring_requests == (
        date(2026, 7, 7),
        date(2026, 7, 8),
    )
    assert plan.estimated_request_count == 2
    assert plan.estimated_minimum_duration_seconds == 2.0


def test_preview_result_and_inputs_are_stored() -> None:
    state: dict[str, object] = {}
    plan = _preview_plan(date(2026, 7, 1))
    inputs = build_preview_inputs(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        delay_seconds=1,
        max_dates=10,
        retry_failed=False,
    )

    store_backfill_preview(state, plan, inputs)

    assert state[PREVIEW_PLAN_KEY] is plan
    assert state[PREVIEW_INPUTS_KEY] == inputs


def test_zero_request_preview_has_explicit_status_and_cannot_resume() -> None:
    plan = _preview_plan()

    assert preview_status_message(plan) == (
        "No requests are required for this range. Existing files and "
        "non-trading dates already cover it."
    )
    assert not resume_is_eligible(plan, stale=False)


def test_non_zero_preview_exposes_request_dates_and_can_resume() -> None:
    request_dates = (date(2026, 7, 1), date(2026, 7, 2))
    plan = _preview_plan(*request_dates)

    assert plan.dates_requiring_requests == request_dates
    assert "2 request date(s)" in preview_status_message(plan)
    assert resume_is_eligible(plan, stale=False)


def test_changed_preview_inputs_are_stale_and_disable_resume() -> None:
    plan = _preview_plan(date(2026, 7, 1))
    stored = build_preview_inputs(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        delay_seconds=1,
        max_dates=10,
        retry_failed=False,
    )
    changed = {**stored, "max_dates": 25}

    assert preview_is_stale(stored, changed)
    assert not resume_is_eligible(plan, stale=True)


def test_completed_state_for_previous_range_does_not_affect_new_preview() -> None:
    old_state = BackfillState(
        requested_start_date=date(2026, 7, 1),
        requested_end_date=date(2026, 7, 10),
        status="completed",
    )

    assert state_for_preview_range(
        old_state,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 10),
    ) is None
    assert state_for_preview_range(
        old_state,
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 10),
    ) is old_state


def test_planner_errors_become_safe_preview_messages() -> None:
    def broken_planner() -> BackfillPlan:
        raise RuntimeError("planner unavailable")

    plan, error = create_preview_safely(broken_planner)

    assert plan is None
    assert error == "Could not preview the backfill plan: planner unavailable"


def test_preview_calls_only_planner_and_never_live_processor() -> None:
    calls: list[str] = []
    expected = _preview_plan(date(2026, 7, 1))

    def planner() -> BackfillPlan:
        calls.append("planner")
        return expected

    def live_processor() -> None:
        calls.append("live")

    plan, error = create_preview_safely(planner)

    assert live_processor is not None
    assert plan is expected and error is None
    assert calls == ["planner"]


def test_backfill_requests_are_chronological_and_delay_is_mockable(
    tmp_path: Path,
) -> None:
    calls: list[date] = []
    delays: list[float] = []
    csv_dir = tmp_path / "csv"

    result = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 8),
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        delay_seconds=1.5,
        sleep_fn=delays.append,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert calls == [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)]
    assert result.attempted_dates == tuple(calls)
    assert delays == [1.5, 1.5]
    assert result.state is not None and result.state.status == "completed"


def test_resume_uses_saved_progress_and_does_not_repeat_success(tmp_path: Path) -> None:
    calls: list[date] = []
    csv_dir = tmp_path / "csv"
    state_path = tmp_path / "state.json"
    processor = _success_processor(csv_dir, calls)
    first = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 8),
        max_dates=1,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=processor,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    second = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 8),
        resume=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=processor,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert first.attempted_dates == (date(2026, 7, 6),)
    assert second.attempted_dates == (date(2026, 7, 7), date(2026, 7, 8))
    assert calls == [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)]
    assert second.state is not None
    assert second.state.successful_dates == tuple(calls)


def test_atomic_state_write_preserves_existing_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "backfill_state.json"
    path.write_text("original\n", encoding="utf-8")
    state = BackfillState(date(2026, 7, 1), date(2026, 7, 2))

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(backfill_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_backfill_state(state, path)

    assert path.read_text(encoding="utf-8") == "original\n"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_malformed_state_recovers_safely(tmp_path: Path) -> None:
    path = tmp_path / "backfill_state.json"
    path.write_text("{broken", encoding="utf-8")

    assert load_backfill_state(path) is None
    assert path.read_text(encoding="utf-8") == "{broken"


def test_interrupted_run_persists_completed_dates_and_interruption(
    tmp_path: Path,
) -> None:
    calls: list[date] = []
    csv_dir = tmp_path / "csv"
    state_path = tmp_path / "state.json"

    def interrupting_processor(
        trading_date: date,
        client: Any,
    ) -> DateProcessingResult:
        if calls:
            raise KeyboardInterrupt
        calls.append(trading_date)
        path = _write_daily_csv(csv_dir, trading_date)
        return DateProcessingResult(trading_date, "successful", 1, 1, 0, path)

    result = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 8),
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=interrupting_processor,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    saved = load_backfill_state(state_path)

    assert result.interrupted
    assert saved is not None
    assert saved.successful_dates == (date(2026, 7, 6),)
    assert saved.last_attempted_date == date(2026, 7, 7)
    assert saved.status == "interrupted"


def test_max_dates_and_dry_run_do_not_overreach(tmp_path: Path) -> None:
    calls: list[date] = []
    csv_dir = tmp_path / "csv"
    state_path = tmp_path / "state.json"
    dry = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 10),
        dry_run=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    assert dry.dry_run and dry.attempted_dates == ()
    assert not state_path.exists()

    limited = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 10),
        max_dates=2,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert limited.state is not None
    assert limited.attempted_dates == (date(2026, 7, 6), date(2026, 7, 7))
    assert calls == [date(2026, 7, 6), date(2026, 7, 7)]


def test_retry_failed_controls_whether_saved_failure_is_requested(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    failed_date = date(2026, 7, 6)
    state = BackfillState(
        failed_date,
        failed_date,
        failed_dates=((failed_date, "network unavailable"),),
    )
    write_backfill_state(state, state_path)
    calls: list[date] = []
    csv_dir = tmp_path / "csv"
    without_retry = run_backfill(
        failed_date,
        failed_date,
        resume=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    with_retry = run_backfill(
        failed_date,
        failed_date,
        resume=True,
        retry_failed=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert without_retry.attempted_dates == ()
    assert with_retry.attempted_dates == (failed_date,)
    assert calls == [failed_date]


def test_ambiguous_empty_date_remains_retryable_but_old_empty_is_complete(
    tmp_path: Path,
) -> None:
    recent = date(2026, 7, 30)
    old = date(2026, 7, 27)
    recent_state = tmp_path / "recent.json"
    old_state = tmp_path / "old.json"

    first_recent = run_backfill(
        recent,
        recent,
        state_path=recent_state,
        csv_dir=tmp_path / "recent_csv",
        client=object(),
        date_processor=_skipped_result,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    second_recent = run_backfill(
        recent,
        recent,
        resume=True,
        state_path=recent_state,
        csv_dir=tmp_path / "recent_csv",
        client=object(),
        date_processor=_skipped_result,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    first_old = run_backfill(
        old,
        old,
        state_path=old_state,
        csv_dir=tmp_path / "old_csv",
        client=object(),
        date_processor=_skipped_result,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    second_old = run_backfill(
        old,
        old,
        resume=True,
        state_path=old_state,
        csv_dir=tmp_path / "old_csv",
        client=object(),
        date_processor=_skipped_result,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert first_recent.count("temporary_unavailable") == 1
    assert second_recent.attempted_dates == (recent,)
    assert first_old.count("non_trading") == 1
    assert second_old.attempted_dates == ()


def test_today_and_future_dates_are_temporary_without_live_requests(
    tmp_path: Path,
) -> None:
    calls: list[date] = []
    today = date(2026, 8, 3)
    result = run_backfill(
        today,
        date(2026, 8, 4),
        state_path=tmp_path / "state.json",
        csv_dir=tmp_path / "csv",
        client=object(),
        date_processor=_success_processor(tmp_path / "csv", calls),
        delay_seconds=0,
        today=today,
        clock=FIXED_CLOCK,
    )

    assert calls == []
    assert result.attempted_dates == ()
    assert result.state is not None
    assert tuple(value for value, _ in result.state.temporary_skips) == (
        today,
        date(2026, 8, 4),
    )


def test_one_failed_date_does_not_stop_later_dates(tmp_path: Path) -> None:
    calls: list[date] = []
    csv_dir = tmp_path / "csv"

    def processor(trading_date: date, client: Any) -> DateProcessingResult:
        calls.append(trading_date)
        if trading_date == date(2026, 7, 7):
            raise OSError("network unavailable")
        path = _write_daily_csv(csv_dir, trading_date)
        return DateProcessingResult(trading_date, "successful", 1, 1, 0, path)

    result = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 8),
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        client=object(),
        date_processor=processor,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert calls == [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)]
    assert result.count("successful") == 2
    assert result.count("failed") == 1
    assert result.outcomes[1] == BackfillDateResult(
        date(2026, 7, 7),
        "failed",
        "OSError: network unavailable",
    )
