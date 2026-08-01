"""Fetch PSX data for one date or an inclusive date range."""

from datetime import date

import streamlit as st

from dashboard.data_loader import load_csv_preview
from dashboard.presentation import column_label, format_date
from data_pipeline.src.main import (
    CollectionResult,
    collect_date_range,
    collect_single_date,
)


def _format_dates(dates: tuple[date, ...]) -> str:
    return ", ".join(format_date(value) for value in dates) if dates else "None"


def _skipped_reason(skipped_date: date, today: date) -> str:
    if skipped_date > today:
        return "future date; PSX data is not available yet"
    if skipped_date == today:
        return "today's PSX data may not be published or complete yet"
    return "PSX response contained no equity rows (non-trading or unavailable date)"


def _show_collection_details(result: CollectionResult, today: date) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total dates", result.total_processed, border=True)
    metric_columns[1].metric("Successful", result.successful_count, border=True)
    metric_columns[2].metric("Skipped", result.skipped_count, border=True)
    metric_columns[3].metric("Failed", result.failed_count, border=True)

    st.subheader("Date details")
    st.write(f"Successful: {_format_dates(result.successful_dates)}")
    if result.skipped_dates:
        st.write("Skipped:")
        for skipped_date in result.skipped_dates:
            st.write(
                f"- {format_date(skipped_date)}: "
                f"{_skipped_reason(skipped_date, today)}"
            )
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
        display_preview = preview.rename(
            columns={column: column_label(column) for column in preview.columns}
        )
        st.dataframe(display_preview, width="stretch", hide_index=True)
    else:
        st.warning("CSV paths were returned, but none could be loaded for preview.")


st.title("Fetch Data")
st.caption("Download PSX historical equity data without leaving the dashboard.")
st.session_state.setdefault("collection_result", None)

today = date.today()
mode = st.segmented_control(
    "Collection mode",
    ("Single date", "Date range"),
    default="Single date",
    required=True,
    key="fetch_mode",
)
if mode == "Single date":
    selected_date = st.date_input("Trading date", value=today, key="fetch_date")
    start_date = selected_date
    end_date = selected_date
else:
    start_column, end_column = st.columns(2)
    start_date = start_column.date_input(
        "Start date",
        value=today,
        key="fetch_start_date",
    )
    end_date = end_column.date_input(
        "End date",
        value=today,
        key="fetch_end_date",
    )

if start_date > today:
    st.warning("The selected date or range is in the future; PSX data is not available yet.")
elif end_date > today:
    st.warning("The selected range includes future dates; those dates may be skipped.")
elif end_date == today:
    st.warning("The selection includes today; today's PSX data may not be complete yet.")

if st.button("Fetch data", type="primary", icon=":material/download:"):
    if end_date < start_date:
        st.session_state["collection_result"] = None
        st.error("End date cannot be earlier than start date.")
    else:
        try:
            with st.spinner("Fetching and processing PSX data..."):
                if mode == "Single date":
                    collection_result = collect_single_date(start_date)
                else:
                    collection_result = collect_date_range(start_date, end_date)
            st.session_state["collection_result"] = collection_result
        except Exception as exc:
            st.session_state["collection_result"] = None
            st.error(f"Data collection could not be completed: {exc}")

stored_result = st.session_state.get("collection_result")
if stored_result is not None:
    _show_collection_details(stored_result, today)
