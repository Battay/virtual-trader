"""Streamlit frontend for collecting and previewing PSX historical data."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from data_pipeline.src.main import (
    CollectionResult,
    collect_date_range,
    collect_single_date,
)


def _format_dates(dates: tuple[date, ...]) -> str:
    """Format collected dates for compact display."""
    return ", ".join(value.isoformat() for value in dates) if dates else "None"


def load_csv_preview(
    csv_paths: Sequence[Path | str],
) -> tuple[pd.DataFrame | None, tuple[str, ...]]:
    """Load and combine only the CSV paths returned by the data pipeline."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for returned_path in csv_paths:
        path = Path(returned_path)
        if not path.is_file():
            errors.append(f"Returned CSV path does not exist: {path}")
            continue
        try:
            frames.append(pd.read_csv(path))
        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            errors.append(f"Could not preview {path}: {exc}")

    preview = pd.concat(frames, ignore_index=True) if frames else None
    return preview, tuple(errors)


def _show_collection_details(result: CollectionResult) -> None:
    """Render metrics, per-date outcomes, paths, and a combined CSV preview."""
    total_column, success_column, skipped_column, failed_column = st.columns(4)
    total_column.metric("Total dates", result.total_processed)
    success_column.metric("Successful", result.successful_count)
    skipped_column.metric("Skipped", result.skipped_count)
    failed_column.metric("Failed", result.failed_count)

    st.subheader("Date details")
    st.write(f"Successful: {_format_dates(result.successful_dates)}")
    st.write(f"Skipped: {_format_dates(result.skipped_dates)}")
    if result.failed_dates:
        st.write("Failed:")
        for failed_date, reason in result.failed_dates:
            st.write(f"- {failed_date.isoformat()}: {reason}")
    else:
        st.write("Failed: None")

    st.subheader("Saved CSV files")
    if result.output_csv_paths:
        for path in result.output_csv_paths:
            st.code(str(path), language=None)
    else:
        st.info("No CSV was generated.")
        return

    preview, preview_errors = load_csv_preview(result.output_csv_paths)
    for error in preview_errors:
        st.warning(error)

    if preview is not None:
        st.subheader("Generated data preview")
        st.dataframe(preview, width="stretch", hide_index=True)
    else:
        st.warning("CSV paths were returned, but none could be loaded for preview.")


def main() -> None:
    """Render the PSX data collection application."""
    st.set_page_config(page_title="PSX Data Collection", layout="wide")
    st.title("Pakistan Stock Exchange Data Collection System")
    st.caption("Download and preview historical PSX equity data.")
    st.session_state.setdefault("collection_result", None)

    mode = st.radio("Collection mode", ("Single date", "Date range"))
    if mode == "Single date":
        selected_date = st.date_input("Trading date", value=date.today())
        start_date = selected_date
        end_date = selected_date
    else:
        start_column, end_column = st.columns(2)
        start_date = start_column.date_input("Start date", value=date.today())
        end_date = end_column.date_input("End date", value=date.today())

    if st.button("Fetch Data", type="primary"):
        if end_date < start_date:
            st.session_state["collection_result"] = None
            st.error("End date cannot be earlier than start date.")
        else:
            try:
                with st.spinner("Fetching and processing PSX data..."):
                    if mode == "Single date":
                        result = collect_single_date(start_date)
                    else:
                        result = collect_date_range(start_date, end_date)
                st.session_state["collection_result"] = result
            except Exception as exc:
                st.session_state["collection_result"] = None
                st.error(f"Data collection could not be completed: {exc}")

    collection_result = st.session_state.get("collection_result")
    if collection_result is not None:
        _show_collection_details(collection_result)


if __name__ == "__main__":
    main()
