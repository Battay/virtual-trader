"""Offline tests for resumable, sequential PSX historical backfill."""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dashboard.backfill_preview import (
    PREVIEW_INPUTS_KEY,
    PREVIEW_PLAN_KEY,
    backfill_date_bounds,
    build_preview_inputs,
    clamp_backfill_end_date,
    create_preview_safely,
    initial_backfill_dates,
    preview_is_stale,
    preview_status_message,
    resume_is_eligible,
    state_for_preview_range,
    store_backfill_preview,
    summarize_backfill_batch,
)
from data_pipeline.src import backfill as backfill_module
from data_pipeline.src import main as main_module
from data_pipeline.src.backfill import (
    BackfillDateResult,
    BackfillPlan,
    BackfillState,
    BackfillSuccessRecord,
    create_backfill_plan,
    load_backfill_state,
    run_backfill,
    write_backfill_state,
)
from data_pipeline.src.config import PSX_HISTORICAL_MIN_DATE
from data_pipeline.src.main import (
    CollectionResult,
    DateProcessingResult,
    OUTPUT_FIELDS,
    iter_calendar_dates,
)


FIXED_CLOCK = lambda: datetime(2026, 8, 1, 12, tzinfo=timezone.utc)

VALID_EQUITY_HTML = """
<table>
  <tr data-type="equity">
    <td data-value="OGDC"></td><td data-value="220.50"></td>
    <td data-value="221.00"></td><td data-value="225.25"></td>
    <td data-value="219.75"></td><td data-value="224.00"></td>
    <td data-value="3.50"></td><td data-value="1.59"></td>
    <td data-value="1,234"></td>
  </tr>
</table>
"""


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


def test_latest_batch_summary_separates_downloads_from_reconciliation() -> None:
    attempted = tuple(date(2026, 7, day) for day in range(1, 11))
    previous_success = date(2026, 6, 30)
    reconciled_dates = tuple(date(2026, 6, day) for day in (27, 28, 29))
    outcomes = (
        *(
            BackfillDateResult(value, "successful", "downloaded")
            for value in attempted[:6]
        ),
        *(
            BackfillDateResult(
                value,
                "successful",
                "reconciled",
                reconciled=True,
            )
            for value in reconciled_dates
        ),
        BackfillDateResult(attempted[6], "temporary_unavailable", "empty"),
        BackfillDateResult(attempted[7], "temporary_unavailable", "empty"),
        BackfillDateResult(attempted[8], "failed", "network"),
        BackfillDateResult(attempted[9], "non_trading", "known holiday"),
    )
    result = backfill_module.BackfillRunResult(
        plan=_preview_plan(*attempted),
        attempted_dates=attempted,
        outcomes=outcomes,
        state=BackfillState(
            requested_start_date=attempted[0],
            requested_end_date=attempted[-1],
            successful_dates=(previous_success,),
        ),
        dry_run=False,
        interrupted=False,
    )

    summary = summarize_backfill_batch(result)

    assert summary.requests_attempted == 10
    assert summary.downloads_successful == 6
    assert summary.downloads_successful <= summary.requests_attempted
    assert summary.existing_csv_reconciled == 3
    assert summary.non_trading_resolved == 1
    assert summary.temporarily_unavailable == 2
    assert summary.failed == 1
    assert summary.total_dates_resolved == 13
    assert previous_success not in {outcome.trading_date for outcome in outcomes}


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


def test_backfill_bounds_allow_more_than_ten_years_through_project_today() -> None:
    project_today = date(2026, 8, 5)

    assert PSX_HISTORICAL_MIN_DATE == date(2016, 7, 26)
    assert project_today > date(2026, 7, 26)
    assert clamp_backfill_end_date(
        PSX_HISTORICAL_MIN_DATE,
        project_today,
        latest_allowed_date=project_today,
    ) == project_today
    assert list(iter_calendar_dates(PSX_HISTORICAL_MIN_DATE, project_today))[-1] == (
        project_today
    )


def test_backfill_maximum_uses_asia_karachi_calendar_date() -> None:
    utc_before_karachi_midnight = datetime(
        2026,
        8,
        4,
        20,
        0,
        tzinfo=timezone.utc,
    )

    assert backfill_date_bounds(utc_before_karachi_midnight) == (
        PSX_HISTORICAL_MIN_DATE,
        date(2026, 8, 5),
    )


def test_changing_start_preserves_valid_end_and_clamps_only_invalid_end() -> None:
    latest_allowed = date(2026, 8, 5)

    assert clamp_backfill_end_date(
        date(2020, 1, 1),
        latest_allowed,
        latest_allowed_date=latest_allowed,
    ) == latest_allowed
    assert clamp_backfill_end_date(
        date(2026, 8, 1),
        date(2026, 7, 31),
        latest_allowed_date=latest_allowed,
    ) == date(2026, 8, 1)
    assert clamp_backfill_end_date(
        date(2026, 8, 1),
        date(2026, 8, 20),
        latest_allowed_date=latest_allowed,
    ) == latest_allowed


def test_saved_backfill_range_restores_without_shortening() -> None:
    saved = BackfillState(
        requested_start_date=PSX_HISTORICAL_MIN_DATE,
        requested_end_date=date(2026, 8, 5),
    )

    restored = initial_backfill_dates(
        saved_state=saved,
        default_start=date(2026, 7, 1),
        default_end=date(2026, 8, 1),
    )

    assert restored == (PSX_HISTORICAL_MIN_DATE, date(2026, 8, 5))


def test_end_before_start_remains_rejected() -> None:
    with pytest.raises(ValueError, match="end date cannot be earlier"):
        create_backfill_plan(
            date(2026, 8, 5),
            date(2026, 8, 4),
            today=date(2026, 8, 5),
        )


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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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


def test_every_empty_weekday_remains_retryable(
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
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
        empty_retry_delays=(0, 0),
        retry_jitter_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert first_recent.count("temporary_unavailable") == 1
    assert second_recent.attempted_dates == (recent,)
    assert first_old.count("temporary_unavailable") == 1
    assert second_old.attempted_dates == (old,)


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
    assert result.outcomes[1].trading_date == date(2026, 7, 7)
    assert result.outcomes[1].status == "failed"
    assert result.outcomes[1].reason == "OSError: network unavailable"
    assert result.outcomes[1].attempt_count == 1


def test_backfill_uses_fresh_cli_pipeline_for_each_requested_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CLI-downloadable date cannot become empty through client reuse."""
    instances: list[object] = []

    class SingleRequestClient:
        def __init__(self) -> None:
            self.requests = 0
            instances.append(self)

        def fetch_market_by_date(self, trading_date: date) -> str:
            self.requests += 1
            if self.requests > 1:
                return "<table></table>"
            return VALID_EQUITY_HTML

    html_dir = tmp_path / "raw" / "html"
    csv_dir = tmp_path / "raw" / "csv"
    monkeypatch.setattr(main_module, "PsxClient", SingleRequestClient)
    monkeypatch.setattr(
        backfill_module,
        "PsxClient",
        SingleRequestClient,
        raising=False,
    )
    monkeypatch.setattr(main_module, "RAW_HTML_DIR", html_dir)
    monkeypatch.setattr(main_module, "RAW_CSV_DIR", csv_dir)
    monkeypatch.setattr(main_module, "REJECTED_DATA_DIR", tmp_path / "rejected")

    result = run_backfill(
        date(2016, 7, 27),
        date(2016, 7, 28),
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert result.count("successful") == 2
    assert result.count("non_trading") == 0
    assert [outcome.reason for outcome in result.outcomes] == [
        "Saved 1 valid equity rows",
        "Saved 1 valid equity rows",
    ]
    assert len(instances) == 2
    assert all(instance.requests == 1 for instance in instances)
    assert (csv_dir / "market_2016-07-27.csv").is_file()
    assert (csv_dir / "market_2016-07-28.csv").is_file()


def test_matching_valid_date_result_overrides_contradictory_aggregate_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = date(2016, 8, 10)
    unrelated = date(2016, 8, 9)
    csv_dir = tmp_path / "csv"
    output_path = csv_dir / f"market_{requested.isoformat()}.csv"
    matching_result = DateProcessingResult(
        requested,
        "successful",
        503,
        503,
        0,
        output_path,
    )
    unrelated_result = DateProcessingResult(
        unrelated,
        "skipped",
        0,
        0,
        0,
        None,
    )

    def contradictory_collection(*args: object, **kwargs: object) -> CollectionResult:
        _write_daily_csv(csv_dir, requested)
        return CollectionResult(
            start_date=requested,
            end_date=requested,
            total_processed=1,
            successful_dates=(),
            skipped_dates=(requested,),
            failed_dates=(),
            output_csv_paths=(),
            date_results=(unrelated_result, matching_result),
        )

    monkeypatch.setattr(backfill_module, "collect_single_date", contradictory_collection)
    result = run_backfill(
        requested,
        requested,
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        client=object(),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert result.count("successful") == 1
    assert result.count("temporary_unavailable") == 0
    assert result.outcomes[0].reason == "Saved 503 valid equity rows"
    assert result.outcomes[0].output_path == output_path
    assert result.state is not None
    assert result.state.successful_dates == (requested,)
    assert result.state.last_successful_date == requested
    assert result.state.success_records == (
        BackfillSuccessRecord(
            trading_date=requested,
            valid_rows=503,
            output_path=str(output_path),
            parsed_rows=503,
                rejected_rows=0,
                message="Saved 503 valid equity rows",
                attempt_count=1,
            ),
    )


def test_weekend_is_permanent_non_trading_without_request(tmp_path: Path) -> None:
    weekend = date(2026, 7, 25)
    calls: list[date] = []
    result = run_backfill(
        weekend,
        weekend,
        csv_dir=tmp_path / "csv",
        state_path=tmp_path / "state.json",
        client=object(),
        date_processor=_success_processor(tmp_path / "csv", calls),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert calls == []
    assert result.attempted_dates == ()
    assert result.state is not None
    assert result.state.non_trading_dates == (weekend,)


def test_valid_csv_reconciles_incorrect_skip_state_and_persists_details(
    tmp_path: Path,
) -> None:
    first = date(2016, 8, 9)
    second = date(2016, 8, 10)
    unrelated_failed = date(2016, 8, 11)
    csv_dir = tmp_path / "csv"
    first_path = _write_daily_csv(csv_dir, first)
    second_path = _write_daily_csv(csv_dir, second)
    state_path = tmp_path / "state.json"
    original = BackfillState(
        requested_start_date=first,
        requested_end_date=unrelated_failed,
        non_trading_dates=(first,),
        temporary_skips=((second, "old empty response"),),
        failed_dates=((unrelated_failed, "network unavailable"),),
    )
    write_backfill_state(original, state_path)

    result = run_backfill(
        first,
        unrelated_failed,
        resume=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    persisted = load_backfill_state(state_path)

    assert result.attempted_dates == ()
    assert result.count("successful") == 2
    assert persisted is not None
    assert persisted.successful_dates == (first, second)
    assert persisted.last_successful_date == second
    assert persisted.non_trading_dates == ()
    assert persisted.temporary_skips == ()
    assert persisted.failed_dates == ((unrelated_failed, "network unavailable"),)
    persisted_details = [
        (record.valid_rows, record.output_path, record.reconciled)
        for record in persisted.success_records
    ]
    assert persisted_details == [
        (1, str(first_path), True),
        (1, str(second_path), True),
    ]


def test_empty_weekday_succeeds_on_second_fresh_client_with_backoff_and_audit(
    tmp_path: Path,
) -> None:
    trading_date = date(2016, 10, 12)
    responses = iter(["e" * 903, "s" * 344_000])
    clients: list[object] = []
    delays: list[float] = []
    csv_dir = tmp_path / "csv"

    class ResponseClient:
        def __init__(self, html: str) -> None:
            self.html = html
            self.calls = 0
            clients.append(self)

        def fetch_market_by_date(self, requested: date) -> str:
            self.calls += 1
            return self.html

    def factory() -> ResponseClient:
        return ResponseClient(next(responses))

    def processor(requested: date, client: Any) -> DateProcessingResult:
        html = client.fetch_market_by_date(requested)
        if html.startswith("e"):
            return DateProcessingResult(requested, "skipped", 0, 0, 0, None)
        output = _write_daily_csv(csv_dir, requested)
        return DateProcessingResult(requested, "successful", 504, 504, 0, output)

    result = run_backfill(
        trading_date,
        trading_date,
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        client_factory=factory,
        date_processor=processor,
        sleep_fn=delays.append,
        empty_max_attempts=3,
        empty_retry_delays=(3, 8, 20),
        retry_jitter_seconds=0.5,
        jitter_fn=lambda low, high: 0.25,
        attempt_html_dir=tmp_path / "html",
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    outcome = result.outcomes[0]
    assert outcome.status == "successful"
    assert outcome.attempt_count == 2
    assert outcome.response_sizes == (903, 344_000)
    assert delays == [3.25]
    assert len(clients) == 2
    assert all(client.calls == 1 for client in clients)
    assert all(
        attempt.raw_html_path is not None
        and Path(attempt.raw_html_path).is_file()
        for attempt in outcome.attempts
    )


def test_repeatedly_empty_weekday_is_temporary_with_recorded_attempts(
    tmp_path: Path,
) -> None:
    trading_date = date(2016, 10, 12)
    delays: list[float] = []

    class EmptyClient:
        def fetch_market_by_date(self, requested: date) -> str:
            return "e" * 903

    def processor(requested: date, client: Any) -> DateProcessingResult:
        client.fetch_market_by_date(requested)
        return DateProcessingResult(requested, "skipped", 0, 0, 0, None)

    result = run_backfill(
        trading_date,
        trading_date,
        csv_dir=tmp_path / "csv",
        state_path=tmp_path / "state.json",
        client_factory=EmptyClient,
        date_processor=processor,
        sleep_fn=delays.append,
        empty_max_attempts=3,
        empty_retry_delays=(3, 8, 20),
        retry_jitter_seconds=0,
        attempt_html_dir=tmp_path / "html",
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )
    persisted = load_backfill_state(tmp_path / "state.json")

    outcome = result.outcomes[0]
    assert outcome.status == "temporary_unavailable"
    assert outcome.attempt_count == 3
    assert outcome.response_sizes == (903, 903, 903)
    assert delays == [3, 8]
    assert persisted is not None
    assert dict(persisted.attempt_records)[trading_date] == outcome.attempts


def test_circuit_breaker_pauses_before_consuming_remaining_dates(
    tmp_path: Path,
) -> None:
    calls: list[date] = []
    csv_dir = tmp_path / "csv"
    successful = {
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
    }

    def unhealthy_processor(
        trading_date: date,
        client: Any,
    ) -> DateProcessingResult:
        calls.append(trading_date)
        if trading_date in successful:
            path = _write_daily_csv(csv_dir, trading_date)
            return DateProcessingResult(trading_date, "successful", 1, 1, 0, path)
        return DateProcessingResult(trading_date, "skipped", 0, 0, 0, None)

    result = run_backfill(
        date(2026, 7, 6),
        date(2026, 7, 21),
        csv_dir=csv_dir,
        state_path=tmp_path / "state.json",
        client=object(),
        date_processor=unhealthy_processor,
        empty_max_attempts=1,
        circuit_window=10,
        circuit_empty_ratio=0.70,
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert result.circuit_breaker_triggered
    assert result.state is not None and result.state.status == "paused"
    assert "unusually many empty weekday responses" in result.state.last_message
    assert result.attempted_dates[-1] == date(2026, 7, 17)
    assert date(2026, 7, 20) not in calls
    assert date(2026, 7, 21) not in calls
    assert result.state.successful_dates == tuple(sorted(successful))


def test_retry_temporary_only_processes_no_other_weekdays(tmp_path: Path) -> None:
    temporary_date = date(2026, 7, 6)
    untouched_date = date(2026, 7, 7)
    csv_dir = tmp_path / "csv"
    state_path = tmp_path / "state.json"
    write_backfill_state(
        BackfillState(
            requested_start_date=temporary_date,
            requested_end_date=untouched_date,
            temporary_skips=((temporary_date, "empty weekday"),),
        ),
        state_path,
    )
    calls: list[date] = []

    result = run_backfill(
        temporary_date,
        untouched_date,
        retry_temporary_only=True,
        csv_dir=csv_dir,
        state_path=state_path,
        client=object(),
        date_processor=_success_processor(csv_dir, calls),
        delay_seconds=0,
        today=date(2026, 8, 1),
        clock=FIXED_CLOCK,
    )

    assert calls == [temporary_date]
    assert result.attempted_dates == (temporary_date,)
    assert result.state is not None
    assert result.state.successful_dates == (temporary_date,)
