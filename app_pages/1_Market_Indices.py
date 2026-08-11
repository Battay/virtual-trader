"""Explore and refresh official PSX index time series."""

import pandas as pd
import streamlit as st

from data_pipeline.src.config import INDICES_MASTER_PATH
from dashboard.index_period_presentation import (
    DEFAULT_INDEX_PERIOD,
    index_chart_frame,
    index_drawdown_chart,
    index_health_breakdown_frame,
    index_level_chart,
    index_period_contract_values,
    index_period_summary_values,
    index_rolling_volatility_chart,
    index_source_identity,
    resolve_index_period,
)
from dashboard.presentation import (
    format_date,
    format_decimal,
    format_directional_percentage,
    format_integer,
    format_percentage,
)
from market_intelligence.index_config import SUPPORTED_INDICES
from market_intelligence.index_periods import (
    INDEX_PERIOD_OPTIONS,
    analyze_index_period,
)
from market_intelligence.market_health import (
    calculate_period_index_health,
)
from market_intelligence.refresh_indices import refresh_indices


@st.cache_data(ttl="5m", max_entries=10)
def _load(
    path: str,
    modified_at_ns: int | None,
    size_bytes: int | None,
) -> pd.DataFrame:
    """Load the local master with file identity included in the cache key."""
    _ = (modified_at_ns, size_bytes)
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


st.title("Market indices")
st.caption(
    "Official PSX end-of-day time series. Server-side date ranges are unavailable; "
    "the selected period is filtered and analyzed locally."
)
labels = {
    definition.display_name: code
    for code, definition in SUPPORTED_INDICES.items()
}
selected_label = st.selectbox("Index", tuple(labels), key="indices_selected_index")
selected_label = selected_label or next(iter(labels))
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

data = _load(*index_source_identity(INDICES_MASTER_PATH))
available_codes = set(
    data.get("index_code", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
)
if data.empty or code not in available_codes:
    st.info(
        "No stored observations are available for this index. Use a refresh "
        "action above."
    )
    st.stop()

period_value = st.segmented_control(
    "Range",
    INDEX_PERIOD_OPTIONS,
    default=DEFAULT_INDEX_PERIOD,
    key="indices_selected_period",
)
selected_period = resolve_index_period(period_value)
analysis = analyze_index_period(data, code, selected_period)
if analysis.metadata.observations == 0:
    st.info(f"No {selected_label} observations are available for {selected_period}.")
    st.stop()
health = calculate_period_index_health(analysis)
contract = index_period_contract_values(analysis)
summary = index_period_summary_values(analysis, health)
chart_data = index_chart_frame(analysis, display_label=selected_label)
latest_row = chart_data.iloc[-1]

st.subheader(f"{selected_label} period contract")
with st.container(border=True):
    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Requested period",
            str(contract["Requested Period"]),
        )
        st.metric("Actual start", format_date(contract["Actual Start Date"]))
        st.metric("Actual end", format_date(contract["Actual End Date"]))
        st.metric(
            "Trading observations",
            format_integer(contract["Trading Observations"]),
        )
        st.metric("Start level", format_decimal(contract["Start Value"]))
        st.metric("End level", format_decimal(contract["End Value"]))
    st.caption(
        "All analytics below use exactly this visible period. Moving averages and "
        "rolling statistics do not use observations before the actual start date."
    )

st.subheader(f"{selected_label} health — {selected_period}")
with st.container(border=True):
    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Index health score",
            (
                f"{format_decimal(summary['Health Score'], precision=1)} / 100"
                if summary["Health Score"] is not None
                else "—"
            ),
        )
        st.metric("Condition", health.label)
        st.metric("Reference date", format_date(health.reference_date))
        st.metric(
            "Analytical coverage",
            f"{format_decimal(health.coverage_percentage, precision=1)}%",
        )
    st.caption(
        "Index health is a descriptive analytical indicator, not investment advice."
    )
    if health.coverage_percentage < 100:
        unavailable = ", ".join(health.unavailable_components) or "unspecified inputs"
        st.warning(
            f"{selected_period} has limited analytical coverage. Unavailable: "
            f"{unavailable}. The score uses {health.available_weight}% of configured "
            "weight and is normalized to 100."
        )
    breakdown = index_health_breakdown_frame(health)
    st.dataframe(
        breakdown,
        hide_index=True,
        column_config={
            "Input": st.column_config.NumberColumn(format="%.2f"),
            "Factor %": st.column_config.NumberColumn(format="%.2f%%"),
            "Raw Points": st.column_config.NumberColumn(format="%.2f"),
            "Normalized Contribution": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )
    normalized_total = breakdown["Normalized Contribution"].sum(min_count=1)
    st.caption(
        "Score formula: raw points ÷ available configured weight × 100. "
        f"Displayed normalized contributions total "
        f"{format_decimal(normalized_total, precision=2)} before the final "
        "one-decimal score rounding."
    )

st.subheader(f"{selected_label} selected-period summary")
with st.container(horizontal=True, gap="small"):
    st.metric(
        "Latest level",
        format_decimal(summary["Latest Level"]),
        format_directional_percentage(latest_row["Daily Change %"]),
        delta_color="off",
        border=True,
    )
    st.metric(
        "Selected-period return",
        format_directional_percentage(summary["Selected-Period Return"]),
        border=True,
    )
    st.metric(
        "Period high",
        format_decimal(summary["Period High"]),
        border=True,
    )
    st.metric(
        "Period low",
        format_decimal(summary["Period Low"]),
        border=True,
    )
    st.metric(
        "Selected-period volatility",
        format_percentage(
            summary["Selected-Period Volatility"], show_sign=False
        ),
        border=True,
    )
    st.metric(
        "Maximum drawdown",
        format_percentage(summary["Maximum Drawdown"]),
        border=True,
    )

st.subheader(f"{selected_label} analytical history — {selected_period}")
st.altair_chart(index_level_chart(chart_data), width="stretch")
st.caption(
    "MA20 and MA50 are causal within the selected visible period; unavailable "
    "warm-up values are left blank. Hover for date, index level, and daily change."
)

drawdown_column, volatility_column = st.columns(2)
with drawdown_column:
    with st.container(border=True):
        st.markdown("**Drawdown over time**")
        st.altair_chart(index_drawdown_chart(chart_data), width="stretch")
with volatility_column:
    with st.container(border=True):
        st.markdown("**Rolling volatility (20 observations)**")
        if chart_data["Rolling Volatility 20D %"].notna().any():
            st.altair_chart(
                index_rolling_volatility_chart(chart_data),
                width="stretch",
            )
        else:
            st.info("Insufficient selected-period history for rolling volatility.")

show_volume = st.toggle("Show volume chart", key="indices_show_volume")
if show_volume:
    st.bar_chart(
        chart_data,
        x="Trading Date",
        y="Volume",
        x_label="Trading date",
        y_label="Volume",
    )

st.caption(
    f"Selected range: {format_date(contract['Actual Start Date'])} to "
    f"{format_date(contract['Actual End Date'])} · Source: Official PSX "
    "end-of-day time series"
)
history_columns = [
    "Trading Date",
    "Index Level",
    "MA20",
    "MA50",
    "Daily Change",
    "Daily Change %",
    "Drawdown %",
    "Rolling Volatility 20D %",
    "Volume",
]
if "Open / Reference" in chart_data:
    history_columns.insert(2, "Open / Reference")
display = chart_data.loc[:, history_columns].sort_values(
    "Trading Date", ascending=False
)
st.dataframe(
    display,
    hide_index=True,
    column_config={
        "Trading Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Index Level": st.column_config.NumberColumn(format="%.2f"),
        "Daily Change": st.column_config.NumberColumn(format="%+.2f"),
        "Daily Change %": st.column_config.NumberColumn(format="%+.2f%%"),
        "Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
        "Rolling Volatility 20D %": st.column_config.NumberColumn(format="%.2f%%"),
        "Volume": st.column_config.NumberColumn(format="%d"),
    },
)
st.download_button(
    "Download selected analysis",
    chart_data.to_csv(index=False).encode(),
    file_name=f"{code}_{selected_period}_index_analysis.csv",
    mime="text/csv",
    icon=":material/download:",
)
