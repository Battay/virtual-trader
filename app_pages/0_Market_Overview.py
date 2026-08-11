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
    index_period_analyses,
    market_summary_values,
    normalized_index_performance_from_filtered,
)
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
    MARKET_HEALTH_WEIGHTS,
    calculate_market_health,
    calculate_period_index_health_scores,
)


@st.cache_data(ttl="5m", max_entries=10)
def _read_csv(
    path: str,
    modified_at_ns: int | None,
    size_bytes: int | None,
) -> pd.DataFrame:
    del modified_at_ns, size_bytes
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _load_csv(path) -> pd.DataFrame:
    """Load local data with file identity included in the Streamlit cache key."""
    return _read_csv(*index_source_identity(path))


st.title("Market Overview")
st.caption("Official PSX indices and descriptive local market intelligence.")

indices = _load_csv(INDICES_MASTER_PATH)
equities = _load_csv(MASTER_CSV_PATH)
processed = _load_csv(PROCESSED_MASTER_PATH)
models = _load_csv(MODEL_REGISTRY_PATH)
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

st.subheader("Index performance and period health")
st.caption(
    "The chart, metrics, health score, and component table below all use the "
    "same selected-period observations. Index Health is descriptive analysis, "
    "not investment advice."
)
selected_view = st.selectbox(
    "Index view",
    INDEX_VIEW_OPTIONS,
    index=0,
    key="market_overview_index_view",
)
period = st.segmented_control(
    "Visible period",
    INDEX_RANGE_OPTIONS,
    default=DEFAULT_INDEX_PERIOD,
    key="market_overview_index_period",
)
labels = {
    code: definition.display_name
    for code, definition in SUPPORTED_INDICES.items()
}
selected_period = resolve_index_period(period)
selected_code = INDEX_VIEW_CODES[selected_view]
period_analyses = index_period_analyses(
    indices,
    selected_period,
    index_codes=tuple(SUPPORTED_INDICES),
)
period_index_health = calculate_period_index_health_scores(period_analyses)
visible_frames = [
    analysis.causal_frame
    for analysis in period_analyses.values()
    if not analysis.causal_frame.empty
]
visible_indices = (
    pd.concat(visible_frames, ignore_index=True)
    if visible_frames
    else pd.DataFrame()
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
    comparison_rows: list[dict[str, object]] = []
    for code, label in labels.items():
        analysis = period_analyses[code]
        index_result = period_index_health[code]
        contract = index_period_contract_values(analysis)
        period_summary = index_period_summary_values(analysis, index_result)
        comparison_rows.append(
            {
                "Index": label,
                "Period": contract["Requested Period"],
                "Start": contract["Actual Start Date"],
                "End": contract["Actual End Date"],
                "Observations": contract["Trading Observations"],
                "Start Value": contract["Start Value"],
                "End Value": contract["End Value"],
                "Return %": period_summary["Selected-Period Return"],
                "Volatility %": period_summary["Selected-Period Volatility"],
                "Max Drawdown %": period_summary["Maximum Drawdown"],
                "Health Score": period_summary["Health Score"],
                "Condition": index_result.label,
                "Coverage %": index_result.coverage_percentage,
            }
        )
    st.dataframe(
        pd.DataFrame(comparison_rows),
        hide_index=True,
        column_config={
            "Health Score": st.column_config.NumberColumn(format="%.1f / 100"),
            "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Start Value": st.column_config.NumberColumn(format="%,.2f"),
            "End Value": st.column_config.NumberColumn(format="%,.2f"),
            "Return %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
            "Max Drawdown %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Coverage %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        width="stretch",
    )
    st.caption(
        f"Health methodology: {next(iter(period_index_health.values())).methodology_version} · "
        "each index is filtered independently from its own latest observation."
    )
else:
    analysis = period_analyses[selected_code]
    selected_health = period_index_health[selected_code]
    contract = index_period_contract_values(analysis)
    period_summary = index_period_summary_values(analysis, selected_health)
    chart_frame = index_chart_frame(analysis, display_label=selected_view)
    latest_row = chart_frame.iloc[-1] if not chart_frame.empty else None

    with st.container(border=True):
        st.markdown(f"**{selected_view} health — {selected_period}**")
        with st.container(horizontal=True, gap="small"):
            st.metric(
                "Index health score",
                (
                    f"{format_decimal(selected_health.score, precision=1)} / 100"
                    if selected_health.score is not None
                    else "—"
                ),
                border=True,
            )
            st.metric(
                "Index condition",
                selected_health.label,
                border=True,
            )
            st.metric(
                "Actual start",
                format_date(contract["Actual Start Date"]),
                border=True,
            )
            st.metric(
                "Actual end",
                format_date(contract["Actual End Date"]),
                border=True,
            )
            st.metric(
                "Trading observations",
                format_integer(contract["Trading Observations"]),
                border=True,
            )
            st.metric(
                "Component coverage",
                format_percentage(
                    selected_health.coverage_percentage,
                    show_sign=False,
                ),
                border=True,
            )
        if selected_health.coverage_percentage < 100:
            unavailable = ", ".join(selected_health.unavailable_components)
            st.warning(
                f"{selected_period} has insufficient selected-period history for: "
                f"{unavailable}. The score uses "
                f"{format_decimal(selected_health.coverage_percentage, precision=1)}% "
                "of configured components and normalizes available weight to 100."
            )
        st.caption(
            f"Methodology {selected_health.methodology_version}. The score equals "
            "the sum of the normalized contributions shown below and remains a "
            "descriptive analytical indicator, not investment advice."
        )
        breakdown = index_health_breakdown_frame(selected_health)
        with st.expander("Period health component breakdown"):
            st.dataframe(
                breakdown,
                hide_index=True,
                column_config={
                    "Input": st.column_config.NumberColumn(format="%.3f"),
                    "Configured Weight": st.column_config.NumberColumn(format="%d"),
                    "Factor %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Raw Points": st.column_config.NumberColumn(format="%.3f"),
                    "Normalized Contribution": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
                width="stretch",
            )
            for explanation in selected_health.explanations:
                st.caption(explanation)

    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Selected-period return",
            format_directional_percentage(period_summary["Selected-Period Return"]),
            border=True,
        )
        st.metric(
            "Period high",
            format_decimal(period_summary["Period High"]),
            border=True,
        )
        st.metric(
            "Period low",
            format_decimal(period_summary["Period Low"]),
            border=True,
        )
        st.metric(
            "Latest level",
            format_decimal(period_summary["Latest Level"]),
            border=True,
        )
        st.metric(
            "Selected-period volatility",
            format_percentage(
                period_summary["Selected-Period Volatility"],
                show_sign=False,
            ),
            border=True,
        )
        st.metric(
            "Maximum drawdown",
            format_directional_percentage(period_summary["Maximum Drawdown"]),
            border=True,
        )
        st.metric(
            "Start value",
            format_decimal(contract["Start Value"]),
            border=True,
        )
        st.metric(
            "End value",
            format_decimal(contract["End Value"]),
            border=True,
        )
        st.metric(
            "Latest daily change",
            format_decimal(
                latest_row["Daily Change"] if latest_row is not None else None,
                show_sign=True,
            ),
            border=True,
        )
        st.metric(
            "Latest daily change %",
            format_directional_percentage(
                latest_row["Daily Change %"] if latest_row is not None else None
            ),
            border=True,
        )

    if chart_frame.empty:
        st.info(f"No {selected_view} history is available for this period.")
    else:
        st.altair_chart(
            index_level_chart(chart_frame),
            width="stretch",
            key="market_overview_index_level_chart",
        )
        st.caption(
            "Index level with causal MA20 and MA50. Moving averages begin only "
            "after enough observations exist inside the selected period."
        )
        st.markdown("**Selected-period drawdown**")
        st.altair_chart(
            index_drawdown_chart(chart_frame),
            width="stretch",
            key="market_overview_drawdown_chart",
        )
        if chart_frame["Rolling Volatility 20D %"].notna().any():
            st.markdown("**Causal rolling volatility**")
            st.altair_chart(
                index_rolling_volatility_chart(chart_frame),
                width="stretch",
                key="market_overview_volatility_chart",
            )
        else:
            st.info(
                "Rolling 20-observation volatility is unavailable for this "
                "selected period."
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
