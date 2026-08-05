"""Descriptive PSX market-condition overview."""

import json

import pandas as pd
import streamlit as st

from dashboard.market_overview import (
    ALL_INDICES_LABEL,
    INDEX_RANGE_OPTIONS,
    INDEX_VIEW_CODES,
    INDEX_VIEW_OPTIONS,
    automation_status_label,
    filter_index_range,
    index_health_comparison,
    index_health_from_filtered_data,
    index_trend_summary,
    market_summary_values,
    normalized_index_performance_from_filtered,
    single_index_performance_from_filtered,
    single_index_period_summary,
)
from dashboard.presentation import (
    format_date,
    format_datetime,
    format_decimal,
    format_directional_percentage,
    format_integer,
    format_percentage,
)
from data_pipeline.src.automation import load_automation_config
from data_pipeline.src.config import (
    INDICES_MASTER_PATH,
    INDICES_REFRESH_METADATA_PATH,
    MASTER_CSV_PATH,
    MODEL_REGISTRY_PATH,
    PROCESSED_MASTER_PATH,
)
from market_intelligence.index_config import SUPPORTED_INDICES
from market_intelligence.index_metrics import calculate_index_metrics
from market_intelligence.market_breadth import calculate_market_breadth
from market_intelligence.market_health import (
    INDEX_HEALTH_WEIGHTS,
    MARKET_HEALTH_WEIGHTS,
    calculate_index_health_scores,
    calculate_market_health,
)


@st.cache_data(ttl="5m", max_entries=10)
def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


st.title("Market Overview")
st.caption("Official PSX indices and descriptive local market intelligence.")

indices = _read_csv(str(INDICES_MASTER_PATH))
equities = _read_csv(str(MASTER_CSV_PATH))
processed = _read_csv(str(PROCESSED_MASTER_PATH))
models = _read_csv(str(MODEL_REGISTRY_PATH))
metrics = (
    {
        code: calculate_index_metrics(indices, code)
        for code in SUPPORTED_INDICES
    }
    if not indices.empty
    else {}
)
breadth = (
    calculate_market_breadth(equities)
    if not equities.empty
    else calculate_market_breadth(pd.DataFrame())
)
health = calculate_market_health(metrics, breadth)
index_health = calculate_index_health_scores(metrics)
summary = market_summary_values(
    score=health.score,
    condition=health.label,
    latest_date=breadth.reference_date,
)

st.subheader("Overall Market Health")
with st.container(border=True):
    with st.container(horizontal=True, gap="small"):
        st.metric("Overall Condition", summary["Market Condition"])
        score = summary["Market Health Score"]
        score_text = (
            f"{format_decimal(score, precision=0)} / 100"
            if score is not None
            else "—"
        )
        st.metric("Overall Score", score_text)
        st.metric(
            "Latest Reference Date",
            format_date(summary["Latest Trading Date"]),
        )
    st.caption(
        "Overall Market Health combines index trends, equity-market breadth, "
        "moving-average position, volatility, and volume participation. It is "
        "calculated separately from the per-index health scores."
    )
    st.caption("This is a descriptive analytical indicator, not investment advice.")

with st.expander("Overall Market Health methodology"):
    st.write(
        "Available components are weighted using the fixed formula below; missing "
        "components are excluded and available weights are rescaled to 100."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Component": list(MARKET_HEALTH_WEIGHTS),
                "Weight": list(MARKET_HEALTH_WEIGHTS.values()),
                "Points": list(health.component_scores.values()),
            }
        ),
        hide_index=True,
    )
    for explanation in health.explanations:
        st.caption(explanation)

st.subheader("Index Health Overview")
st.caption("Index Health is a descriptive analytical indicator, not investment advice.")
if not metrics:
    st.info(
        "Index data is not available yet. Open Market Indices to refresh the "
        "official series."
    )
else:
    with st.container(horizontal=True, gap="small"):
        for code, definition in SUPPORTED_INDICES.items():
            metric = metrics[code]
            index_result = index_health[code]
            with st.container(border=True):
                st.markdown(f"**{definition.display_name}**")
                st.metric(
                    "Condition",
                    index_result.label,
                    f"{format_decimal(index_result.score, precision=1)} / 100"
                    if index_result.score is not None
                    else "—",
                    delta_color="off",
                )
                st.write(f"Latest value: {format_decimal(metric.latest_value)}")
                st.write(
                    "Daily change: "
                    f"{format_directional_percentage(metric.latest_daily_change_percent)}"
                )
                st.write(
                    "1-month return: "
                    f"{format_directional_percentage(metric.one_month_return)}"
                )
                st.write(f"Trend: {index_trend_summary(metric)}")
                st.write(
                    "Volatility: "
                    f"{format_percentage(metric.rolling_volatility_20, show_sign=False)} "
                    "annualized (20-day)"
                    if metric.rolling_volatility_20 is not None
                    else "Volatility: Not Available"
                )
                st.caption(f"Observation date: {format_date(metric.latest_date)}")

st.subheader("Index Performance")
selected_view = st.selectbox(
    "Index view",
    INDEX_VIEW_OPTIONS,
    index=0,
    key="market_overview_index_view",
)
period = st.segmented_control(
    "Visible period",
    INDEX_RANGE_OPTIONS,
    default="6M",
    key="market_overview_index_period",
)
labels = {
    code: definition.display_name
    for code, definition in SUPPORTED_INDICES.items()
}
selected_period = period or "6M"
selected_code = INDEX_VIEW_CODES[selected_view]
visible_indices = filter_index_range(indices, selected_period)
period_index_health = index_health_from_filtered_data(
    visible_indices,
    index_codes=tuple(SUPPORTED_INDICES),
)
st.markdown(f"**{selected_view}**")
if selected_code is None:
    performance = normalized_index_performance_from_filtered(
        visible_indices,
        display_labels=labels,
    )
    if performance.empty:
        st.info("Insufficient index history is available for this comparison.")
    else:
        st.line_chart(
            performance,
            x="Trading Date",
            y="Normalized Performance",
            color="Index",
            x_label="Trading Date",
            y_label="Performance (Starting Value = 100)",
        )
        st.caption(
            "Each available index starts at 100 on its first observation in the "
            "selected period."
        )
    st.dataframe(
        index_health_comparison(period_index_health, display_labels=labels),
        hide_index=True,
        column_config={
            "Health Score": st.column_config.NumberColumn(format="%.1f / 100"),
            "Observation Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )
else:
    performance = single_index_performance_from_filtered(
        visible_indices,
        index_code=selected_code,
        display_label=selected_view,
    )
    period_summary = single_index_period_summary(performance)
    metric = calculate_index_metrics(visible_indices, selected_code)
    selected_health = period_index_health.get(selected_code)
    with st.container(border=True):
        st.markdown(f"**{selected_view} Health — {selected_period}**")
        with st.container(horizontal=True, gap="small"):
            st.metric(
                "Index Health Score",
                (
                    f"{format_decimal(selected_health.score, precision=1)} / 100"
                    if selected_health and selected_health.score is not None
                    else "—"
                ),
            )
            st.metric(
                "Index Condition",
                selected_health.label if selected_health else "Unavailable",
            )
            st.metric(
                "Reference Date",
                format_date(selected_health.reference_date if selected_health else None),
            )
            st.metric(
                "Data Coverage",
                (
                    f"{format_integer(selected_health.observation_count)} trading observations"
                    if selected_health
                    else "—"
                ),
            )
        st.caption(
            "Index Health is a descriptive analytical indicator, not investment advice."
        )
        if selected_health:
            if selected_health.coverage_percentage < 100:
                unavailable = ", ".join(selected_health.unavailable_components)
                st.warning(
                    f"{selected_period} view has insufficient history for: "
                    f"{unavailable}. Score uses "
                    f"{format_decimal(selected_health.coverage_percentage, precision=1)}% "
                    "of configured components and is normalized to 100."
                )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Component": list(INDEX_HEALTH_WEIGHTS),
                        "Weight": list(INDEX_HEALTH_WEIGHTS.values()),
                        "Points": list(selected_health.component_scores.values()),
                    }
                ),
                hide_index=True,
            )
            for explanation in selected_health.explanations:
                st.caption(explanation)
    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Latest Value",
            format_decimal(metric.latest_value if metric else None),
            border=True,
        )
        st.metric(
            "Daily Change",
            format_decimal(
                metric.latest_daily_change if metric else None,
                show_sign=True,
            ),
            border=True,
        )
        st.metric(
            "Daily Change %",
            format_directional_percentage(
                metric.latest_daily_change_percent if metric else None
            ),
            border=True,
        )
        st.metric(
            "Period Return",
            format_directional_percentage(period_summary["Period Return"]),
            border=True,
        )
        st.metric(
            "Period High",
            format_decimal(period_summary["Period High"]),
            border=True,
        )
        st.metric(
            "Period Low",
            format_decimal(period_summary["Period Low"]),
            border=True,
        )
        st.metric(
            "Latest Observation Date",
            format_date(metric.latest_date if metric else None),
            border=True,
        )
    if performance.empty:
        st.info(f"No {selected_view} history is available for this period.")
    else:
        st.line_chart(
            performance,
            x="Trading Date",
            y="Index Value",
            x_label="Trading Date",
            y_label="Index Value",
        )

st.subheader("Market Breadth")
breadth_available = breadth.reference_date is not None
with st.container(horizontal=True, gap="small"):
    st.metric(
        "Advancers",
        format_integer(breadth.advancing if breadth_available else None),
        border=True,
    )
    st.metric(
        "Decliners",
        format_integer(breadth.declining if breadth_available else None),
        border=True,
    )
    st.metric(
        "Unchanged",
        format_integer(breadth.unchanged if breadth_available else None),
        border=True,
    )
    st.metric(
        "Advance/Decline Ratio",
        format_decimal(
            breadth.advance_decline_ratio if breadth_available else None
        ),
        border=True,
    )
    st.metric(
        "Total Volume",
        format_integer(
            breadth.total_traded_volume if breadth_available else None
        ),
        border=True,
    )
    st.metric(
        "Universe Size",
        format_integer(breadth.universe_size if breadth_available else None),
        border=True,
    )
st.caption(
    f"Reference date: {format_date(breadth.reference_date)}. Universe: all valid "
    "securities on the latest locally available equity date."
)

st.subheader("System Status")
config = load_automation_config()
ai_ready = (
    int(processed["symbol"].nunique())
    if not processed.empty and "symbol" in processed
    else None
)
needs_retraining = (
    int(models["retraining_status"].eq("retraining_recommended").sum())
    if not models.empty and "retraining_status" in models
    else None
)
latest_equity_date = (
    pd.to_datetime(equities["date"], errors="coerce").max()
    if not equities.empty and "date" in equities
    else None
)
index_refreshed_at = None
try:
    metadata = json.loads(INDICES_REFRESH_METADATA_PATH.read_text(encoding="utf-8"))
    index_refreshed_at = metadata.get("refreshed_at")
except (OSError, UnicodeError, ValueError, TypeError):
    pass

with st.container(horizontal=True, gap="small"):
    st.metric("Automation", automation_status_label(config.enabled), border=True)
    st.metric(
        "Latest Equity Data",
        format_date(latest_equity_date),
        border=True,
    )
    st.metric(
        "Latest Index Refresh",
        format_datetime(index_refreshed_at),
        border=True,
    )
    st.metric("AI-Ready Symbols", format_integer(ai_ready), border=True)
    st.metric(
        "Models Needing Retraining",
        format_integer(needs_retraining),
        border=True,
    )
