"""Pure presentation transforms for canonical PSX index-period analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import altair as alt
import pandas as pd

from market_intelligence.index_periods import (
    INDEX_PERIOD_OPTIONS,
    IndexPeriodAnalysis,
)
from market_intelligence.market_health import (
    PERIOD_INDEX_HEALTH_WEIGHTS,
    IndexHealth,
)


DEFAULT_INDEX_PERIOD = "6M"
_HEALTH_INPUT_UNITS: Mapping[str, str] = {
    "Selected-period return": "%",
    "Trend consistency": "%",
    "Period momentum": "%",
    "Selected-period volatility resilience": "% annualized volatility",
    "Maximum-drawdown resilience": "% drawdown",
    "Value relative to SMA-20": "%",
    "Value relative to SMA-50": "%",
    "Volume participation": "ratio",
}


def resolve_index_period(
    period: object,
    *,
    default: str = DEFAULT_INDEX_PERIOD,
) -> str:
    """Resolve a widget value to one supported canonical period."""
    if default not in INDEX_PERIOD_OPTIONS:
        raise ValueError(f"Unsupported default index period: {default!r}")
    return str(period) if period in INDEX_PERIOD_OPTIONS else default


def index_source_identity(path: Path) -> tuple[str, int | None, int | None]:
    """Return a cache-safe local file identity without reading file contents."""
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return str(source), None, None
    return str(source), int(stat.st_mtime_ns), int(stat.st_size)


def index_period_contract_values(
    analysis: IndexPeriodAnalysis,
) -> dict[str, object]:
    """Expose one period contract with stable, readable field names."""
    metadata = analysis.metadata
    return {
        "Requested Period": metadata.requested_period,
        "Actual Start Date": metadata.actual_start_date,
        "Actual End Date": metadata.actual_end_date,
        "Trading Observations": metadata.observations,
        "Start Value": metadata.start_value,
        "End Value": metadata.end_value,
    }


def index_period_summary_values(
    analysis: IndexPeriodAnalysis,
    health: IndexHealth,
) -> dict[str, object]:
    """Return selected-period summary values from one canonical analysis."""
    return {
        "Selected-Period Return": analysis.period_return_percent,
        "Period High": analysis.period_high,
        "Period Low": analysis.period_low,
        "Latest Level": analysis.latest_value,
        "Selected-Period Volatility": analysis.annualized_volatility_percent,
        "Maximum Drawdown": analysis.maximum_drawdown_percent,
        "Health Score": health.score,
    }


def index_health_breakdown_frame(health: IndexHealth) -> pd.DataFrame:
    """Build a reproducible component table in configured-weight order."""
    rows: list[dict[str, object]] = []
    for component, weight in PERIOD_INDEX_HEALTH_WEIGHTS.items():
        factor = health.component_factors.get(component)
        rows.append(
            {
                "Component": component,
                "Input": health.component_inputs.get(component),
                "Input Unit": _HEALTH_INPUT_UNITS.get(component, ""),
                "Configured Weight": weight,
                "Factor %": None if factor is None else float(factor) * 100,
                "Raw Points": health.component_scores.get(component),
                "Normalized Contribution": health.normalized_contributions.get(
                    component
                ),
            }
        )
    return pd.DataFrame(rows)


def index_chart_frame(
    analysis: IndexPeriodAnalysis,
    *,
    display_label: str,
) -> pd.DataFrame:
    """Return a readable chart/table copy of the exact causal period frame."""
    causal = analysis.causal_frame.copy(deep=True)
    columns: dict[str, pd.Series] = {}

    def values(name: str, *, numeric: bool = False) -> pd.Series:
        if name not in causal:
            return pd.Series(pd.NA, index=causal.index, dtype="object")
        series = causal[name].copy(deep=True)
        return pd.to_numeric(series, errors="coerce") if numeric else series

    columns["Trading Date"] = pd.to_datetime(values("date"), errors="coerce")
    columns["Index"] = pd.Series(display_label, index=causal.index, dtype="string")
    columns["Index Level"] = values("value", numeric=True)
    columns["MA20"] = values("ma_20", numeric=True)
    columns["MA50"] = values("ma_50", numeric=True)
    columns["Daily Change"] = values("daily_change", numeric=True)
    columns["Daily Change %"] = values("daily_change_percent", numeric=True)
    columns["Drawdown %"] = values("drawdown_percent", numeric=True)
    columns["Rolling Volatility 20D %"] = values(
        "rolling_volatility_20_percent", numeric=True
    )
    columns["Volume"] = values("volume", numeric=True)
    if "open" in causal:
        columns["Open / Reference"] = values("open", numeric=True)
    return pd.DataFrame(columns).reset_index(drop=True)


def index_level_chart(frame: pd.DataFrame) -> alt.Chart:
    """Build the selected index level plus causal MA20/MA50 chart."""
    tooltip = [
        alt.Tooltip("Trading Date:T", title="Date"),
        alt.Tooltip("Index Level:Q", title="Index level", format=",.2f"),
        alt.Tooltip("Daily Change:Q", title="Daily change", format="+,.2f"),
        alt.Tooltip("Daily Change %:Q", title="Daily change %", format="+.2f"),
        alt.Tooltip("MA20:Q", title="MA20", format=",.2f"),
        alt.Tooltip("MA50:Q", title="MA50", format=",.2f"),
    ]
    return (
        alt.Chart(frame)
        .transform_fold(
            ["Index Level", "MA20", "MA50"],
            as_=["Series", "Displayed Level"],
        )
        .mark_line()
        .encode(
            x=alt.X("Trading Date:T", title="Trading date"),
            y=alt.Y("Displayed Level:Q", title="Index level", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Index Level", "MA20", "MA50"],
                    range=["#1f77b4", "#ff7f0e", "#2ca02c"],
                ),
            ),
            strokeWidth=alt.condition(
                alt.datum.Series == "Index Level",
                alt.value(2.5),
                alt.value(1.5),
            ),
            tooltip=tooltip,
        )
        .properties(height=420)
        .interactive()
    )


def index_drawdown_chart(frame: pd.DataFrame) -> alt.Chart:
    """Build a compact selected-period drawdown chart on its own scale."""
    return (
        alt.Chart(frame)
        .mark_area(line=True, opacity=0.25, color="#d62728")
        .encode(
            x=alt.X("Trading Date:T", title="Trading date"),
            y=alt.Y("Drawdown %:Q", title="Drawdown %"),
            tooltip=[
                alt.Tooltip("Trading Date:T", title="Date"),
                alt.Tooltip("Drawdown %:Q", title="Drawdown %", format="+.2f"),
            ],
        )
        .properties(height=210)
        .interactive()
    )


def index_rolling_volatility_chart(frame: pd.DataFrame) -> alt.Chart:
    """Build a compact causal rolling-volatility chart on its own scale."""
    return (
        alt.Chart(frame)
        .mark_line(color="#9467bd", strokeWidth=2)
        .encode(
            x=alt.X("Trading Date:T", title="Trading date"),
            y=alt.Y(
                "Rolling Volatility 20D %:Q",
                title="Annualized volatility %",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Trading Date:T", title="Date"),
                alt.Tooltip(
                    "Rolling Volatility 20D %:Q",
                    title="20-day annualized volatility %",
                    format=".2f",
                ),
            ],
        )
        .properties(height=210)
        .interactive()
    )
