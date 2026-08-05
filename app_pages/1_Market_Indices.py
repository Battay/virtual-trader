"""Explore and refresh official PSX index time series."""

import pandas as pd
import streamlit as st

from data_pipeline.src.config import INDICES_MASTER_PATH
from dashboard.presentation import (
    format_date,
    format_decimal,
    format_directional_percentage,
    format_integer,
    format_percentage,
)
from market_intelligence.index_config import SUPPORTED_INDICES
from market_intelligence.index_metrics import calculate_index_metrics
from market_intelligence.market_health import (
    INDEX_HEALTH_WEIGHTS,
    calculate_index_health,
)
from market_intelligence.refresh_indices import refresh_indices


@st.cache_data(ttl="5m", max_entries=10)
def _load() -> pd.DataFrame:
    try: return pd.read_csv(INDICES_MASTER_PATH)
    except (OSError, ValueError, pd.errors.ParserError): return pd.DataFrame()


st.title("Market indices")
st.caption("Official PSX end-of-day time series. Server-side date ranges are unavailable; ranges below are filtered locally.")
labels = {definition.display_name: code for code, definition in SUPPORTED_INDICES.items()}
selected_label = st.selectbox("Index", tuple(labels), key="indices_selected_index")
code = labels[selected_label]
with st.container(horizontal=True):
    if st.button("Refresh selected index", icon=":material/refresh:"):
        with st.spinner("Refreshing official index data..."):
            result = refresh_indices((code,))
        st.cache_data.clear()
        st.success(f"Refreshed {len(result.successful_indices)} index series.")
    if st.button("Refresh all indices", icon=":material/sync:"):
        with st.spinner("Refreshing all official index data..."):
            result = refresh_indices()
        st.cache_data.clear()
        st.success(f"Refreshed {len(result.successful_indices)} index series.")

data = _load()
if data.empty or code not in set(data.get("index_code", pd.Series(dtype=str)).astype(str)):
    st.info("No stored observations are available for this index. Use a refresh action above.")
    st.stop()
history = data.loc[data["index_code"].astype(str) == code].copy()
history["date"] = pd.to_datetime(history["date"], errors="coerce")
history = history.dropna(subset=["date"]).sort_values("date")
period = st.segmented_control("Range", ("1M", "3M", "6M", "1Y", "YTD", "Maximum"), default="6M")
latest_date = history["date"].max()
offsets = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}
if period in offsets: shown = history.loc[history["date"] >= latest_date - offsets[period]]
elif period == "YTD": shown = history.loc[history["date"] >= pd.Timestamp(latest_date.year, 1, 1)]
else: shown = history
metric = calculate_index_metrics(data, code)
health = calculate_index_health(metric)
st.subheader(f"{selected_label} health")
with st.container(border=True):
    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Index Health Score",
            f"{format_decimal(health.score, precision=1)} / 100"
            if health.score is not None
            else "—",
        )
        st.metric("Condition", health.label)
        st.metric("Reference Date", format_date(health.reference_date))
    st.caption(
        "Index Health is a descriptive analytical indicator, not investment advice."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Component": list(INDEX_HEALTH_WEIGHTS),
                "Weight": list(INDEX_HEALTH_WEIGHTS.values()),
                "Points": list(health.component_scores.values()),
            }
        ),
        hide_index=True,
    )
    for explanation in health.explanations:
        st.caption(explanation)

st.subheader(f"{selected_label} metrics")
with st.container(horizontal=True):
    st.metric(
        "Latest value",
        format_decimal(metric.latest_value),
        format_directional_percentage(metric.latest_daily_change_percent),
        delta_color="off",
        border=True,
    )
    st.metric("Previous value", format_decimal(metric.previous_value), border=True)
    st.metric(
        "Daily change",
        format_decimal(metric.latest_daily_change, show_sign=True),
        border=True,
    )
    st.metric("Latest volume", format_integer(metric.latest_volume), border=True)
    st.metric("Average volume (20-day)", format_decimal(metric.average_volume_20), border=True)
    st.metric("One-week return", format_percentage(metric.one_week_return), border=True)
    st.metric("One-month return", format_percentage(metric.one_month_return), border=True)
    st.metric("Three-month return", format_percentage(metric.three_month_return), border=True)
    st.metric("Six-month return", format_percentage(metric.six_month_return), border=True)
    st.metric("One-year return", format_percentage(metric.one_year_return), border=True)
    st.metric("Versus SMA-20", format_percentage(metric.versus_ma_20_percent), border=True)
    st.metric("Versus SMA-50", format_percentage(metric.versus_ma_50_percent), border=True)
    st.metric(
        "20-day annualized volatility",
        format_percentage(metric.rolling_volatility_20, show_sign=False),
        border=True,
    )
st.subheader(f"{selected_label} history")
st.line_chart(shown, x="date", y="value", x_label="Trading date", y_label="Index value")
show_volume = st.toggle("Show volume chart")
if show_volume: st.bar_chart(shown, x="date", y="volume", x_label="Trading date", y_label="Volume")
st.caption(f"Observed range: {format_date(history['date'].min())} to {format_date(history['date'].max())} · Source: Official PSX end-of-day time series")
display = shown.rename(columns={"date": "Trading date", "value": "Index value", "volume": "Volume", "open": "Open/reference", "daily_change": "Daily change", "daily_change_percent": "Daily change %"})
st.dataframe(display[["Trading date", "Index value", "Volume", "Open/reference", "Daily change", "Daily change %"]].sort_values("Trading date", ascending=False), hide_index=True)
st.download_button("Download selected history", shown.to_csv(index=False).encode(), file_name=f"{code}_history.csv", mime="text/csv", icon=":material/download:")
