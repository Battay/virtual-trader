"""Explore all locally generated PSX daily CSV files in memory."""

import streamlit as st

from dashboard.data_loader import (
    filter_market_data,
    load_market_dataset,
    summarize_dataset,
)


st.title("Dataset explorer")
st.caption("Inspect and download filtered views of the generated daily CSV files.")

load_result = load_market_dataset()
for error in load_result.errors:
    st.warning(error)

if load_result.file_count == 0:
    st.info("No generated CSV files were found. Use the Fetch data page first.")
    st.stop()

summary = summarize_dataset(load_result)
first_metric_row = st.columns(3)
first_metric_row[0].metric("CSV files", summary.csv_files, border=True)
first_metric_row[1].metric("Trading dates", summary.trading_dates, border=True)
first_metric_row[2].metric("Unique symbols", summary.unique_symbols, border=True)
second_metric_row = st.columns(3)
second_metric_row[0].metric("Total rows", summary.total_rows, border=True)
second_metric_row[1].metric(
    "Earliest date",
    summary.earliest_date.isoformat() if summary.earliest_date else "N/A",
    border=True,
)
second_metric_row[2].metric(
    "Latest date",
    summary.latest_date.isoformat() if summary.latest_date else "N/A",
    border=True,
)

if load_result.data.empty:
    st.info("Generated CSV files exist, but they contain no rows to explore.")
    st.stop()

symbols = sorted(
    str(symbol)
    for symbol in load_result.data["symbol"].dropna().astype("string").unique()
)
selected_symbol = st.selectbox("Symbol", ["All symbols", *symbols])
symbol_filter = None if selected_symbol == "All symbols" else selected_symbol

start_date = None
end_date = None
if st.checkbox("Filter by date range"):
    date_columns = st.columns(2)
    start_date = date_columns[0].date_input(
        "From date",
        value=summary.earliest_date,
        key="dataset_start_date",
    )
    end_date = date_columns[1].date_input(
        "To date",
        value=summary.latest_date,
        key="dataset_end_date",
    )
    if end_date < start_date:
        st.error("End date cannot be earlier than start date.")
        st.stop()

filtered_data = filter_market_data(
    load_result.data,
    symbol=symbol_filter,
    start_date=start_date,
    end_date=end_date,
)

st.subheader("Filtered data")
st.caption(f"{len(filtered_data):,} rows")
st.dataframe(filtered_data, width="stretch", hide_index=True)
st.download_button(
    "Download filtered CSV",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="psx_filtered_data.csv",
    mime="text/csv",
    icon=":material/download:",
    on_click="ignore",
)
