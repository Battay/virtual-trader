"""Unified PSX source completeness, selective repair, and manual fetch controls."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_csv_preview
from dashboard.presentation import column_label, format_date
from data_pipeline.src.data_completeness import (
    CsvDateRecord,
    DataCompletenessInventory,
    ParquetDateRecord,
    build_data_completeness_inventory,
    clear_visible_selection,
    fetch_selected_dates,
    reconcile_pending_source_dates,
    repair_selected_parquet_dates,
    select_visible_actionable,
    update_visible_selection,
)
from data_pipeline.src.main import CollectionResult, collect_date_range, collect_single_date


CSV_SELECTION_KEY = "fetch_control_csv_selected_dates"
CSV_TABLE_KEY = "fetch_control_csv_attention_table"
PARQUET_SELECTION_KEY = "fetch_control_parquet_selected_dates"
PARQUET_TABLE_KEY = "fetch_control_parquet_attention_table"


def _format_dates(dates: tuple[date, ...]) -> str:
    return ", ".join(format_date(value) for value in dates) if dates else "None"


def _skipped_reason(skipped_date: date, today: date) -> str:
    if skipped_date > today:
        return "future date; PSX data is not available yet"
    if skipped_date == today:
        return "today's PSX data may not be published or complete yet"
    return "PSX response contained no equity rows (non-trading or unavailable date)"


def _show_collection_details(result: CollectionResult, today: date) -> None:
    with st.container(horizontal=True):
        st.metric("Total dates", result.total_processed, border=True)
        st.metric("Successful", result.successful_count, border=True)
        st.metric("Skipped", result.skipped_count, border=True)
        st.metric("Failed", result.failed_count, border=True)
    st.subheader("Date details")
    st.write(f"Successful: {_format_dates(result.successful_dates)}")
    if result.skipped_dates:
        st.write("Skipped:")
        for skipped_date in result.skipped_dates:
            st.write(f"- {format_date(skipped_date)}: {_skipped_reason(skipped_date, today)}")
    else:
        st.write("Skipped: None")
    if result.failed_dates:
        st.write("Failed:")
        for failed_date, reason in result.failed_dates:
            st.write(f"- {format_date(failed_date)}: {reason}")
    else:
        st.write("Failed: None")
    st.subheader("Saved CSV files")
    if not result.output_csv_paths:
        st.info("No CSV was generated.")
        return
    for path in result.output_csv_paths:
        st.code(str(path), language=None)
    preview, preview_errors = load_csv_preview(result.output_csv_paths)
    for error in preview_errors:
        st.warning(error)
    if preview is not None:
        st.subheader("Generated data preview")
        st.dataframe(
            preview.rename(columns={column: column_label(column) for column in preview.columns}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("CSV paths were returned, but none could be loaded for preview.")


@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def _load_inventory() -> DataCompletenessInventory:
    return build_data_completeness_inventory()


def _clear_inventory_cache() -> None:
    _load_inventory.clear()


def _selected_dates(key: str) -> tuple[date, ...]:
    raw = st.session_state.get(key, ())
    values: list[date] = []
    for value in raw:
        if isinstance(value, date):
            values.append(value)
        else:
            try:
                values.append(date.fromisoformat(str(value)))
            except ValueError:
                continue
    return tuple(sorted(set(values)))


def _set_selection(key: str, values: tuple[date, ...], table_key: str) -> None:
    st.session_state[key] = values
    st.session_state.pop(table_key, None)


def _csv_attention_frame(records: tuple[CsvDateRecord, ...], selected: tuple[date, ...]) -> pd.DataFrame:
    selected_set = set(selected)
    return pd.DataFrame(
        [
            {
                "Selected": item.trading_date in selected_set,
                "Date": item.trading_date,
                "Classification": item.classification.value,
                "Existing CSV": item.csv_status,
                "Raw evidence": item.raw_evidence_status,
                "Attempts": item.attempts,
                "Last result": item.last_result,
                "Last checked": item.last_checked,
                "Actionability": "ACTIONABLE" if item.actionable else "REVIEW_ONLY",
                "Notes/reason": item.reason,
            }
            for item in records
        ]
    )


def _parquet_attention_frame(records: tuple[ParquetDateRecord, ...], selected: tuple[date, ...]) -> pd.DataFrame:
    selected_set = set(selected)
    return pd.DataFrame(
        [
            {
                "Selected": item.trading_date in selected_set,
                "Date": item.trading_date,
                "State": item.state.value,
                "Rows": item.row_count,
                "Expected rows": item.expected_rows,
                "Actionability": "REPAIRABLE" if item.repairable else "REVIEW_ONLY",
                "Notes/reason": item.reason,
            }
            for item in records
        ]
    )


def _render_summary(inventory: DataCompletenessInventory) -> None:
    st.subheader("Data completeness summary")
    raw = inventory.csv_summary
    daily = inventory.parquet_summary
    with st.container(border=True):
        st.markdown("**Raw / CSV**")
        with st.container(horizontal=True):
            st.metric("Earliest", format_date(raw.earliest_valid_date), border=True)
            st.metric("Latest", format_date(raw.latest_valid_date), border=True)
            st.metric("Valid weekday dates", f"{raw.valid_accepted_dates:,}", border=True)
            st.metric("Actionable missing", f"{raw.actionable_missing_dates:,}", border=True)
        with st.container(horizontal=True):
            st.metric("Likely non-trading", raw.likely_non_trading_candidates, border=True)
            st.metric("Confirmed non-trading", raw.confirmed_non_trading_dates, border=True)
            st.metric("Source anomalies", raw.source_anomalies, border=True)
            st.metric("Retryable / not final", raw.retryable_not_final_dates, border=True)
            st.metric("Invalid CSV", raw.invalid_corrupt_csv_count, border=True)
    with st.container(border=True):
        st.markdown("**Daily Parquet**")
        with st.container(horizontal=True):
            st.metric("Earliest", format_date(daily.earliest_date), border=True)
            st.metric("Latest", format_date(daily.latest_date), border=True)
            st.metric("Current", f"{daily.current:,}", border=True)
            st.metric("Missing", daily.missing, border=True)
            st.metric("Stale", daily.stale, border=True)
            st.metric("Corrupt", daily.corrupt, border=True)
            st.metric("Orphan", daily.orphan, border=True)


def _render_csv_controls(inventory: DataCompletenessInventory) -> None:
    st.subheader("Raw / CSV completeness")
    st.caption(
        "Confirmed non-trading dates and weekends are never fetchable. Likely "
        "non-trading dates and source anomalies remain visible for review."
    )
    attention = tuple(
        item for item in inventory.csv_records
        if item.classification.value not in {"CURRENT", "WEEKEND", "CONFIRMED_NON_TRADING"}
    )
    options = (
        "MISSING", "FAILED_RETRYABLE", "NOT_FINAL", "SOURCE_ANOMALY",
        "LIKELY_NON_TRADING", "INVALID_SOURCE",
    )
    filters = st.pills(
        "Classification", options, default=options, selection_mode="multi",
        key="fetch_control_csv_filters",
    )
    if attention:
        bounds = (attention[0].trading_date, attention[-1].trading_date)
        selected_range = st.date_input(
            "Attention date range", value=bounds, min_value=inventory.start_date,
            max_value=inventory.end_date, key="fetch_control_csv_date_range",
        )
        range_start, range_end = selected_range if isinstance(selected_range, tuple) else bounds
    else:
        range_start = range_end = inventory.end_date
    filtered = tuple(
        item for item in attention
        if item.classification.value in set(filters or ())
        and range_start <= item.trading_date <= range_end
    )
    selected = _selected_dates(CSV_SELECTION_KEY)
    visible_dates = tuple(item.trading_date for item in filtered)
    with st.container(horizontal=True):
        st.button(
            "Select visible actionable", icon=":material/select_all:",
            on_click=_set_selection,
            args=(CSV_SELECTION_KEY, select_visible_actionable(selected, filtered), CSV_TABLE_KEY),
            disabled=not any(item.actionable for item in filtered),
        )
        st.button(
            "Clear visible", icon=":material/deselect:", on_click=_set_selection,
            args=(CSV_SELECTION_KEY, clear_visible_selection(selected, visible_dates), CSV_TABLE_KEY),
            disabled=not set(selected).intersection(visible_dates),
        )
        st.button(
            "Clear all", icon=":material/clear_all:", on_click=_set_selection,
            args=(CSV_SELECTION_KEY, (), CSV_TABLE_KEY), disabled=not selected,
        )
    selected = _selected_dates(CSV_SELECTION_KEY)
    st.caption(
        f"Selected: {len(selected):,} total · Visible: {len(filtered):,} · "
        f"Selected in view: {len(set(selected).intersection(visible_dates)):,}"
    )
    frame = _csv_attention_frame(filtered, selected)
    if frame.empty:
        st.info("No source dates match the current attention filters.")
    else:
        default_rows = [
            index for index, item in enumerate(filtered)
            if item.trading_date in set(selected) and item.actionable
        ]
        event = st.dataframe(
            frame, key=CSV_TABLE_KEY, on_select="rerun", selection_mode="multi-row",
            selection_default={"selection": {"rows": default_rows}},
            hide_index=True, width="stretch", height=360,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            },
        )
        selected_visible = tuple(
            filtered[index].trading_date for index in event.selection.rows
            if 0 <= index < len(filtered) and filtered[index].actionable
        )
        updated = update_visible_selection(selected, visible_dates, selected_visible)
        if updated != selected:
            st.session_state[CSV_SELECTION_KEY] = updated
    selected = _selected_dates(CSV_SELECTION_KEY)
    if selected:
        st.write("Selected fetch dates: " + ", ".join(value.isoformat() for value in selected))
    confirmed = st.checkbox(
        "I confirm this will make bounded PSX network requests for exactly the selected dates.",
        key="fetch_control_network_confirmation", disabled=not selected,
    )
    if st.button(
        "Fetch selected dates", type="primary", icon=":material/download:",
        disabled=not selected or not confirmed,
    ):
        with st.status("Fetching selected dates", expanded=True) as status:
            try:
                result = fetch_selected_dates(selected, inventory.csv_records)
                st.session_state["fetch_control_last_selected_result"] = result
                status.update(
                    label=f"Selected fetch finished: {result.status.lower()}",
                    state="error" if result.status == "FAILED" else "complete",
                )
                _set_selection(CSV_SELECTION_KEY, (), CSV_TABLE_KEY)
                _clear_inventory_cache()
            except Exception as exc:
                status.update(label="Selected fetch failed", state="error")
                st.error(f"Selected fetch could not be completed: {exc}")


def _render_parquet_controls(inventory: DataCompletenessInventory) -> None:
    st.subheader("Daily Parquet health")
    attention = tuple(item for item in inventory.parquet_records if item.state.value != "CURRENT")
    state_filter = st.pills(
        "Parquet state", ("MISSING", "STALE", "CORRUPT", "ORPHAN"),
        default=("MISSING", "STALE", "CORRUPT", "ORPHAN"),
        selection_mode="multi", key="fetch_control_parquet_filters",
    )
    filtered = tuple(item for item in attention if item.state.value in set(state_filter or ()))
    selected = _selected_dates(PARQUET_SELECTION_KEY)
    visible_dates = tuple(item.trading_date for item in filtered)
    repairable_visible = tuple(item for item in filtered if item.repairable)
    with st.container(horizontal=True):
        st.button(
            "Select visible repairable", icon=":material/select_all:", on_click=_set_selection,
            args=(
                PARQUET_SELECTION_KEY,
                tuple(sorted(set(selected).union(item.trading_date for item in repairable_visible))),
                PARQUET_TABLE_KEY,
            ), disabled=not repairable_visible,
        )
        st.button(
            "Clear visible Parquet selection", icon=":material/deselect:", on_click=_set_selection,
            args=(PARQUET_SELECTION_KEY, clear_visible_selection(selected, visible_dates), PARQUET_TABLE_KEY),
            disabled=not set(selected).intersection(visible_dates),
        )
        st.button(
            "Clear all Parquet selection", icon=":material/clear_all:", on_click=_set_selection,
            args=(PARQUET_SELECTION_KEY, (), PARQUET_TABLE_KEY), disabled=not selected,
        )
    selected = _selected_dates(PARQUET_SELECTION_KEY)
    frame = _parquet_attention_frame(filtered, selected)
    if frame.empty:
        st.success("Every canonical daily Parquet partition is current.")
    else:
        default_rows = [
            index for index, item in enumerate(filtered)
            if item.trading_date in set(selected) and item.repairable
        ]
        event = st.dataframe(
            frame, key=PARQUET_TABLE_KEY, on_select="rerun", selection_mode="multi-row",
            selection_default={"selection": {"rows": default_rows}},
            hide_index=True, width="stretch", height=320,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            },
        )
        selected_visible = tuple(
            filtered[index].trading_date for index in event.selection.rows
            if 0 <= index < len(filtered) and filtered[index].repairable
        )
        updated = update_visible_selection(selected, visible_dates, selected_visible)
        if updated != selected:
            st.session_state[PARQUET_SELECTION_KEY] = updated
    selected = _selected_dates(PARQUET_SELECTION_KEY)
    if st.button(
        "Repair selected Parquet dates", type="primary", icon=":material/build:",
        disabled=not selected,
    ):
        with st.status("Repairing selected daily partitions", expanded=True) as status:
            try:
                result = repair_selected_parquet_dates(selected, inventory.parquet_records)
                st.session_state["fetch_control_last_parquet_result"] = result
                status.update(
                    label=("Selected partitions were already current" if result.idempotent_noop
                           else f"Repaired {result.daily_parquets_written:,} partition(s)"),
                    state="complete",
                )
                _set_selection(PARQUET_SELECTION_KEY, (), PARQUET_TABLE_KEY)
                _clear_inventory_cache()
            except Exception as exc:
                status.update(label="Daily Parquet repair failed", state="error")
                st.error(f"Selected repair could not be completed: {exc}")


def _render_master_status(inventory: DataCompletenessInventory) -> None:
    st.subheader("Master datasets")
    left, right = st.columns(2)
    for container, title, value in (
        (left, "Canonical master CSV", inventory.master_csv),
        (right, "Consolidated master Parquet", inventory.master_parquet),
    ):
        with container.container(border=True):
            st.markdown(f"**{title}**")
            st.code(str(value.path), language=None)
            st.write(f"Latest incorporated date: {format_date(value.latest_date)}")
            st.write(
                f"Rows: {value.row_count or 0:,} · Dates: {value.date_count or 0:,} · "
                f"Symbols: {value.symbol_count or 0:,}"
            )
            if value.integrity_status == "PASS":
                st.success("Integrity: PASS")
            else:
                st.error("Integrity: FAIL")
                for error in value.errors:
                    st.caption(error)
    parity = inventory.master_parity
    with st.container(border=True):
        st.markdown("**Master parity**")
        with st.container(horizontal=True):
            st.metric("Key parity", "PASS" if parity.key_parity else "FAIL", border=True)
            st.metric(
                "Logical content parity", "PASS" if parity.logical_content_parity else "FAIL",
                border=True,
            )
            st.metric("Source-set hash", parity.source_set_hash_status, border=True)
            st.metric("Content hash", parity.content_hash_status, border=True)
        if inventory.pending_source_dates:
            st.warning(
                f"{len(inventory.pending_source_dates):,} valid source date(s) await canonical "
                "incorporation: " + ", ".join(value.isoformat() for value in inventory.pending_source_dates)
            )
        if inventory.canonical_dates_with_noncurrent_daily:
            st.warning(
                f"{len(inventory.canonical_dates_with_noncurrent_daily):,} canonical date(s) "
                "have non-current daily Parquet partitions."
            )
        if inventory.pending_source_dates:
            if st.button(
                "Reconcile pending valid sources",
                icon=":material/sync:",
                type="primary",
            ):
                with st.status("Reconciling pending valid sources", expanded=True) as status:
                    try:
                        result = reconcile_pending_source_dates(
                            inventory.pending_source_dates
                        )
                        st.session_state["fetch_control_last_pending_result"] = result
                        status.update(
                            label=(
                                f"Reconciled {len(inventory.pending_source_dates):,} "
                                "pending source date(s)"
                            ),
                            state="complete",
                        )
                        _clear_inventory_cache()
                    except Exception as exc:
                        status.update(label="Pending reconciliation failed", state="error")
                        st.error(f"Pending sources could not be reconciled: {exc}")


def _render_latest_reconciliation() -> None:
    fetch_result = st.session_state.get("fetch_control_last_selected_result")
    parquet_result = st.session_state.get("fetch_control_last_parquet_result")
    if fetch_result is None and parquet_result is None:
        return
    st.subheader("Latest reconciliation result")
    if fetch_result is not None:
        native = (
            fetch_result.reconciliation.native
            if fetch_result.reconciliation is not None
            else None
        )
        with st.container(border=True):
            st.markdown("**Selected source fetch**")
            with st.container(horizontal=True):
                st.metric("Status", fetch_result.status, border=True)
                st.metric("Executed dates", len(fetch_result.executed_dates), border=True)
                st.metric(
                    "Canonical master CSV",
                    "UPDATED" if native and native.rows_added else "CURRENT",
                    border=True,
                )
                st.metric(
                    "Consolidated Parquet",
                    "UPDATED" if native and native.rows_added else "CURRENT",
                    border=True,
                )
                st.metric(
                    "Daily partitions affected",
                    native.daily_parquets_written if native else 0,
                    border=True,
                )
                st.metric(
                    "Symbol artifacts affected",
                    native.symbol_csvs_written if native else 0,
                    border=True,
                )
            if fetch_result.errors:
                for error in fetch_result.errors:
                    st.error(error)
    if parquet_result is not None:
        with st.container(border=True):
            st.markdown("**Selected daily Parquet repair**")
            with st.container(horizontal=True):
                st.metric("Canonical master CSV", parquet_result.master_csv_status, border=True)
                st.metric(
                    "Consolidated Parquet",
                    parquet_result.consolidated_parquet_status,
                    border=True,
                )
                st.metric("Daily partitions affected", parquet_result.daily_parquets_written, border=True)
                st.metric("Symbol artifacts affected", 0, border=True)
                st.metric("Logical parity", "PASS" if parquet_result.logical_parity else "FAIL", border=True)


def _render_history(inventory: DataCompletenessInventory) -> None:
    st.subheader("Operation history")
    if inventory.history.error:
        st.error(inventory.history.error)
        return
    if not inventory.history.entries:
        st.info("No unified maintenance operations have been recorded yet.")
        return
    rows = [
        {
            "Timestamp": item.timestamp,
            "Operation": item.operation_type,
            "Requested": len(item.requested_dates),
            "Executed": len(item.executed_dates),
            "Skipped": len(item.skipped_dates),
            "Errors": len(item.errors),
            "Latest master date": item.master_latest_date,
            "Operation ID": item.operation_id,
        }
        for item in reversed(inventory.history.entries[-50:])
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_manual_fetch() -> None:
    st.subheader("Manual date or range fetch")
    st.caption(
        "This legacy direct fetch remains available for deliberate inspection. "
        "Use the completeness controls above for missing-date maintenance."
    )
    today = date.today()
    mode = st.segmented_control(
        "Collection mode", ("Single date", "Date range"), default="Single date",
        required=True, key="fetch_mode",
    )
    if mode == "Single date":
        selected_date = st.date_input("Trading date", value=today, key="fetch_date")
        start_date = end_date = selected_date
    else:
        start_column, end_column = st.columns(2)
        start_date = start_column.date_input("Start date", value=today, key="fetch_start_date")
        end_date = end_column.date_input("End date", value=today, key="fetch_end_date")
    if start_date > today:
        st.warning("The selected date or range is in the future.")
    elif end_date > today:
        st.warning("The selected range includes future dates; those dates may be skipped.")
    elif end_date == today:
        st.warning("The selection includes today; today's data may not be complete yet.")
    if st.button("Fetch manual date/range", icon=":material/download:"):
        if end_date < start_date:
            st.session_state["collection_result"] = None
            st.error("End date cannot be earlier than start date.")
        else:
            try:
                with st.spinner("Fetching and processing PSX data..."):
                    result = (collect_single_date(start_date) if mode == "Single date"
                              else collect_date_range(start_date, end_date))
                st.session_state["collection_result"] = result
                _clear_inventory_cache()
            except Exception as exc:
                st.session_state["collection_result"] = None
                st.error(f"Data collection could not be completed: {exc}")
    stored_result = st.session_state.get("collection_result")
    if stored_result is not None:
        _show_collection_details(stored_result, today)


st.title("Fetch Data")
st.caption(
    "Audit source completeness, fetch exact missing dates, and repair isolated "
    "daily Parquet partitions without rebuilding unrelated artifacts."
)
st.session_state.setdefault("collection_result", None)
st.session_state.setdefault(CSV_SELECTION_KEY, ())
st.session_state.setdefault(PARQUET_SELECTION_KEY, ())

with st.container(horizontal=True, horizontal_alignment="right"):
    if st.button("Refresh inventory", icon=":material/refresh:"):
        _clear_inventory_cache()
        st.rerun()

inventory_slot = st.container()
try:
    with inventory_slot.skeleton(height=240):
        inventory = _load_inventory()
except Exception as exc:
    inventory_slot.error(f"Data completeness inventory could not be loaded: {exc}")
else:
    for warning in inventory.warnings:
        inventory_slot.warning(warning)
    _render_summary(inventory)
    _render_csv_controls(inventory)
    _render_parquet_controls(inventory)
    _render_master_status(inventory)
    _render_latest_reconciliation()
    _render_history(inventory)

manual = st.expander("Existing manual fetch controls", on_change="rerun")
if manual.open:
    with manual:
        _render_manual_fetch()
