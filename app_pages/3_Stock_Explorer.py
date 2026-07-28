"""Inspect the available local history for one PSX stock symbol."""

import pandas as pd
import streamlit as st

from dashboard.data_loader import filter_market_data, load_market_dataset


def _price(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):,.2f}"


st.title("Stock explorer")
st.caption("Review local historical rows and close-price movement for one symbol.")

load_result = load_market_dataset()
for error in load_result.errors:
    st.warning(error)

if load_result.file_count == 0 or load_result.data.empty:
    st.info("No local market data is available. Use the Fetch data page first.")
    st.stop()

symbols = sorted(
    str(symbol)
    for symbol in load_result.data["symbol"].dropna().astype("string").unique()
)
if not symbols:
    st.info("The local dataset does not contain any stock symbols.")
    st.stop()

selected_symbol = st.selectbox("Stock symbol", symbols)
stock_history = filter_market_data(load_result.data, symbol=selected_symbol)
if stock_history.empty:
    st.info(f"No rows are available for {selected_symbol}.")
    st.stop()

for numeric_column in ("open", "high", "low", "close", "volume"):
    stock_history[numeric_column] = pd.to_numeric(
        stock_history[numeric_column],
        errors="coerce",
    )

latest_row = stock_history.iloc[-1]
latest_date = latest_row["date"]
period_high = stock_history["high"].max()
period_low = stock_history["low"].min()
total_volume = stock_history["volume"].sum(min_count=1)

metric_columns = st.columns(6)
metric_columns[0].metric(
    "Latest date",
    latest_date.date().isoformat() if not pd.isna(latest_date) else "N/A",
    border=True,
)
metric_columns[1].metric("Latest close", _price(latest_row["close"]), border=True)
metric_columns[2].metric("Latest open", _price(latest_row["open"]), border=True)
metric_columns[3].metric("Period high", _price(period_high), border=True)
metric_columns[4].metric("Period low", _price(period_low), border=True)
metric_columns[5].metric(
    "Total volume",
    f"{int(total_volume):,}" if not pd.isna(total_volume) else "N/A",
    border=True,
)

chart_data = stock_history.dropna(subset=["date", "close"])
st.subheader("Close price history")
if chart_data.empty:
    st.info("No close-price values are available to chart.")
else:
    st.line_chart(
        chart_data,
        x="date",
        y="close",
        x_label="Date",
        y_label="Close price",
    )

st.subheader("Historical rows")
st.dataframe(stock_history, width="stretch", hide_index=True)
