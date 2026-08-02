"""Plan and run bounded, resumable historical PSX data backfills."""

from datetime import timedelta

import pandas as pd
import streamlit as st

from dashboard.backfill_preview import (
    PREVIEW_ERROR_KEY,
    PREVIEW_INPUTS_KEY,
    PREVIEW_PLAN_KEY,
    build_preview_inputs,
    clear_backfill_preview,
    create_preview_safely,
    preview_is_stale,
    preview_status_message,
    resume_is_eligible,
    state_for_preview_range,
    store_backfill_preview,
)
from dashboard.presentation import format_date, format_integer, safe_display_value
from data_pipeline.src.automation import karachi_today
from data_pipeline.src.backfill import (
    BackfillPlan,
    BackfillProgress,
    BackfillStateError,
    create_backfill_plan,
    load_backfill_state,
    run_backfill,
)
from data_pipeline.src.config import BACKFILL_STATE_PATH, RAW_CSV_DIR
from data_pipeline.src.data_products import rebuild_data_products
from data_pipeline.src.updater import discover_available_raw_dates


def _plan_frame(plan: BackfillPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Requested start date", plan.start_date.isoformat()),
            ("Requested end date", plan.end_date.isoformat()),
            ("Total calendar dates", f"{plan.total_calendar_dates:,}"),
            (
                "Existing successful dates",
                f"{len(plan.existing_successful_dates):,}",
            ),
            (
                "Dates requiring requests",
                f"{len(plan.dates_requiring_requests):,}",
            ),
            ("Weekend dates", f"{len(plan.weekend_dates):,}"),
            (
                "Unresolved skipped dates",
                f"{len(plan.unresolved_skipped_dates):,}",
            ),
            (
                "Failed dates eligible for retry",
                f"{len(plan.failed_dates_eligible_for_retry):,}",
            ),
            ("Estimated request count", f"{plan.estimated_request_count:,}"),
            (
                "Estimated minimum duration",
                f"{plan.estimated_minimum_duration_seconds / 60:.1f} minutes",
            ),
        ],
        columns=["Plan measure", "Value"],
    )


def _reason_frame(values: tuple[tuple[object, str], ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": format_date(trading_date),
                "Reason": safe_display_value(reason),
            }
            for trading_date, reason in values
        ]
    )


st.title("Historical Data Backfill")
st.caption(
    "Collect older PSX daily files sequentially and resume safely. No model "
    "training is performed."
)
st.warning(
    "Large historical ranges can take hours. Preview first and run small batches "
    "such as 10 request dates before attempting a longer backfill."
)

today = karachi_today()
available_dates = discover_available_raw_dates(RAW_CSV_DIR)
saved_state = load_backfill_state(BACKFILL_STATE_PATH)
earliest_local = min(available_dates) if available_dates else None
latest_local = max(available_dates) if available_dates else None
default_end = earliest_local or (today - timedelta(days=1))
default_start = default_end - timedelta(days=30)

saved_temporary = len(saved_state.temporary_skips) if saved_state else 0
saved_failed = len(saved_state.failed_dates) if saved_state else 0
with st.container(horizontal=True):
    st.metric("Earliest locally stored", format_date(earliest_local), border=True)
    st.metric("Latest locally stored", format_date(latest_local), border=True)
    st.metric("Stored trading dates", format_integer(len(available_dates)), border=True)
    st.metric("Unresolved skipped", format_integer(saved_temporary), border=True)
    st.metric("Failed dates", format_integer(saved_failed), border=True)

if saved_state is None:
    st.info("No saved backfill state exists yet.")
else:
    total_saved_range = (
        saved_state.requested_end_date - saved_state.requested_start_date
    ).days + 1
    resolved_dates = set(saved_state.successful_dates).union(
        saved_state.non_trading_dates,
        saved_state.already_downloaded_dates,
    )
    progress_value = min(1.0, len(resolved_dates) / max(1, total_saved_range))
    st.progress(
        progress_value,
        text=(
            f"Saved progress: {len(resolved_dates):,} of "
            f"{total_saved_range:,} calendar dates resolved"
        ),
    )
    with st.container(horizontal=True):
        st.metric("Current status", safe_display_value(saved_state.status), border=True)
        st.metric(
            "Last attempted date",
            format_date(saved_state.last_attempted_date),
            border=True,
        )
        st.metric(
            "Last successful date",
            format_date(saved_state.last_successful_date),
            border=True,
        )
    st.caption(saved_state.last_message)

with st.container(border=True):
    with st.container(horizontal=True):
        selected_start = st.date_input(
            "Start date",
            value=default_start,
            format="YYYY-MM-DD",
            key="backfill_start_date",
        )
        selected_end = st.date_input(
            "End date",
            value=default_end,
            format="YYYY-MM-DD",
            key="backfill_end_date",
        )
        delay_seconds = st.number_input(
            "Delay between requests (seconds)",
            min_value=0.0,
            value=1.0,
            step=0.5,
            help="Requests are sequential; no parallel fetching is used.",
            key="backfill_delay_seconds",
        )
        maximum_dates = st.number_input(
            "Maximum request dates this run",
            min_value=1,
            value=10,
            step=1,
            help="This bounds each interactive batch and does not change the plan.",
            key="backfill_max_dates",
        )
    stop_after_current = st.toggle(
        "Stop after current date/request",
        help="Limits this submitted batch to one live request, then saves progress.",
        key="backfill_stop_after_current",
    )
    current_preview_inputs = build_preview_inputs(
        start_date=selected_start,
        end_date=selected_end,
        delay_seconds=float(delay_seconds),
        max_dates=int(maximum_dates),
        retry_failed=False,
    )
    stored_preview_plan = st.session_state.get(PREVIEW_PLAN_KEY)
    stored_preview_inputs = st.session_state.get(PREVIEW_INPUTS_KEY)
    preview_stale = preview_is_stale(
        stored_preview_inputs,
        current_preview_inputs,
    )
    resume_allowed = resume_is_eligible(
        stored_preview_plan,
        stale=preview_stale,
    )
    with st.container(horizontal=True):
        preview_clicked = st.button(
            "Preview Backfill Plan",
            icon=":material/preview:",
            key="backfill_preview_plan_button",
        )
        resume_clicked = st.button(
            "Resume Backfill",
            icon=":material/play_arrow:",
            type="primary",
            disabled=not resume_allowed,
            key="backfill_resume_button",
        )
        retry_clicked = st.button(
            "Retry Failed Dates",
            icon=":material/replay:",
            key="backfill_retry_failed_button",
        )

date_error = selected_end < selected_start
if date_error:
    st.error("End date cannot be earlier than start date.")

if preview_clicked:
    if date_error:
        clear_backfill_preview(st.session_state)
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
    else:
        matching_state = state_for_preview_range(
            saved_state,
            start_date=selected_start,
            end_date=selected_end,
        )
        preview_plan, preview_error = create_preview_safely(
            create_backfill_plan,
            selected_start,
            selected_end,
            delay_seconds=float(delay_seconds),
            state=matching_state,
            csv_dir=RAW_CSV_DIR,
            retry_failed=False,
            today=today,
            available_dates=available_dates,
        )
        if preview_error is not None:
            clear_backfill_preview(st.session_state)
            st.session_state[PREVIEW_ERROR_KEY] = preview_error
        elif preview_plan is not None:
            store_backfill_preview(
                st.session_state,
                preview_plan,
                current_preview_inputs,
            )
            st.rerun()

preview_error = st.session_state.get(PREVIEW_ERROR_KEY)
if preview_error:
    st.error(str(preview_error))

if resume_clicked and not resume_allowed:
    st.warning("Preview the current inputs before resuming the backfill.")

if (resume_clicked and resume_allowed) or (retry_clicked and not date_error):
    progress_bar = st.progress(0.0, text="Preparing backfill batch...")
    run_status = st.status("Backfill batch running", expanded=True)

    def show_progress(progress: BackfillProgress) -> None:
        fraction = progress.completed_requests / max(1, progress.scheduled_requests)
        progress_bar.progress(
            fraction,
            text=(
                f"Processed {progress.completed_requests} of "
                f"{progress.scheduled_requests} requests"
            ),
        )
        run_status.write(
            f"{progress.outcome.trading_date.isoformat()}: "
            f"{progress.outcome.status.replace('_', ' ')} — "
            f"{progress.outcome.reason}"
        )

    try:
        result = run_backfill(
            selected_start,
            selected_end,
            resume=True,
            delay_seconds=float(delay_seconds),
            max_dates=(1 if stop_after_current else int(maximum_dates)),
            retry_failed=retry_clicked,
            progress_callback=show_progress,
            today=today,
        )
        st.session_state["backfill_result"] = result
        clear_backfill_preview(st.session_state)
        final_state = result.state.status if result.state else "unknown"
        run_status.update(
            label=f"Backfill batch finished: {final_state}",
            state="complete",
            expanded=True,
        )
    except (BackfillStateError, OSError, ValueError) as exc:
        run_status.update(label="Backfill batch failed", state="error", expanded=True)
        st.error(str(exc))

displayed_plan = st.session_state.get(PREVIEW_PLAN_KEY)
displayed_inputs = st.session_state.get(PREVIEW_INPUTS_KEY)
if displayed_plan is not None:
    displayed_stale = preview_is_stale(displayed_inputs, current_preview_inputs)
    st.subheader("Backfill plan preview")
    if displayed_stale:
        st.warning("The controls changed after this preview. Preview again before resuming.")
    elif displayed_plan.estimated_request_count == 0:
        st.success(preview_status_message(displayed_plan))
    else:
        st.info(preview_status_message(displayed_plan))
    st.dataframe(_plan_frame(displayed_plan), hide_index=True, width="stretch")
    if displayed_plan.dates_requiring_requests:
        with st.expander(
            f"Dates requiring requests ({displayed_plan.estimated_request_count:,})",
            icon=":material/calendar_month:",
        ):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Request date": [
                            value.isoformat()
                            for value in displayed_plan.dates_requiring_requests
                        ]
                    }
                ),
                hide_index=True,
                width="stretch",
            )

latest_result = st.session_state.get("backfill_result")
if latest_result is not None:
    st.subheader("Latest batch summary")
    with st.container(horizontal=True):
        st.metric("Requests attempted", len(latest_result.attempted_dates), border=True)
        st.metric("Successful", latest_result.count("successful"), border=True)
        st.metric("Non-trading", latest_result.count("non_trading"), border=True)
        st.metric(
            "Temporarily unavailable",
            latest_result.count("temporary_unavailable"),
            border=True,
        )
        st.metric("Failed", latest_result.count("failed"), border=True)
    failures = tuple(
        (outcome.trading_date, outcome.reason)
        for outcome in latest_result.outcomes
        if outcome.status == "failed"
    )
    if failures:
        st.dataframe(_reason_frame(failures), hide_index=True, width="stretch")

if saved_state and (saved_state.temporary_skips or saved_state.failed_dates):
    details = st.expander(
        "Unresolved date details",
        icon=":material/error:",
        on_change="rerun",
    )
    if details.open:
        with details:
            if saved_state.temporary_skips:
                st.markdown("**Temporarily unavailable**")
                st.dataframe(
                    _reason_frame(saved_state.temporary_skips),
                    hide_index=True,
                    width="stretch",
                )
            if saved_state.failed_dates:
                st.markdown("**Failed**")
                st.dataframe(
                    _reason_frame(saved_state.failed_dates),
                    hide_index=True,
                    width="stretch",
                )

st.subheader("Rebuild data products")
st.caption(
    "Explicitly rebuilds the master dataset, company registry, AI datasets, "
    "chronological splits/scalers, and readiness report. It never starts training."
)
if st.button(
    "Rebuild Data Products",
    icon=":material/build:",
    type="primary",
):
    raw_dates_added = (
        latest_result.count("successful") if latest_result is not None else 0
    )
    with st.status("Rebuilding data products in dependency order", expanded=True) as status:
        rebuilt = rebuild_data_products(raw_dates_added=raw_dates_added)
        st.session_state["data_products_result"] = rebuilt
        if rebuilt.errors:
            status.update(
                label="Data products rebuilt with errors",
                state="error",
                expanded=True,
            )
        else:
            status.update(
                label="Data products rebuilt",
                state="complete",
                expanded=False,
            )

rebuilt = st.session_state.get("data_products_result")
if rebuilt is not None:
    with st.container(horizontal=True):
        st.metric("Raw dates added", rebuilt.raw_dates_added, border=True)
        st.metric("Master rows", rebuilt.master_rows, border=True)
        st.metric("Master symbols", rebuilt.master_symbols, border=True)
        st.metric("Processed rows", rebuilt.processed_rows, border=True)
        st.metric("Processed symbols", rebuilt.processed_symbols, border=True)
        st.metric("Ready for training", rebuilt.symbols_ready_for_training, border=True)
        st.metric(
            "Insufficient history",
            rebuilt.insufficient_history_symbols,
            border=True,
        )
    for error in rebuilt.errors:
        st.error(error)
