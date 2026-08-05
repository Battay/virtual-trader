"""Pure presentation helpers for the Market Overview dashboard page."""

from collections.abc import Mapping

import pandas as pd

from market_intelligence.index_metrics import IndexMetrics
from market_intelligence.index_metrics import calculate_index_metrics
from market_intelligence.market_health import (
    IndexHealth,
    calculate_index_health_scores,
)


INDEX_RANGE_OPTIONS = ("1M", "3M", "6M", "1Y", "Maximum")
ALL_INDICES_LABEL = "All Indices"
INDEX_VIEW_CODES = {
    ALL_INDICES_LABEL: None,
    "KSE-100 Index": "KSE100",
    "KSE-30 Index": "KSE30",
    "KMI-30 Index": "KMI30",
    "KSE All Share Index": "ALLSHR",
}
INDEX_VIEW_OPTIONS = tuple(INDEX_VIEW_CODES)
_RANGE_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


def filter_index_range(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """Return a chronological copy within a local display range."""
    if period not in INDEX_RANGE_OPTIONS:
        raise ValueError(
            f"Unsupported index range {period!r}; expected one of "
            f"{', '.join(INDEX_RANGE_OPTIONS)}"
        )
    filtered = data.copy(deep=True)
    if filtered.empty or "date" not in filtered:
        return filtered
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    sort_columns = ["date"]
    if "index_code" in filtered:
        sort_columns.append("index_code")
    filtered = filtered.dropna(subset=["date"]).sort_values(
        sort_columns, kind="stable"
    )
    if filtered.empty or period == "Maximum":
        return filtered.reset_index(drop=True)
    latest = filtered["date"].max()
    return filtered.loc[
        filtered["date"] >= latest - _RANGE_OFFSETS[period]
    ].reset_index(drop=True)


def normalized_index_performance(
    data: pd.DataFrame,
    period: str,
    *,
    display_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize each available index to 100 at its first visible observation."""
    filtered = filter_index_range(data, period)
    return normalized_index_performance_from_filtered(
        filtered, display_labels=display_labels
    )


def normalized_index_performance_from_filtered(
    filtered: pd.DataFrame,
    *,
    display_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize index series from one canonical pre-filtered frame."""
    filtered = filtered.copy(deep=True)
    required = {"index_code", "date", "value"}
    if filtered.empty or not required.issubset(filtered.columns):
        return pd.DataFrame(
            columns=("Trading Date", "Index", "Normalized Performance")
        )
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    filtered = filtered.dropna(subset=["index_code", "value"])
    frames: list[pd.DataFrame] = []
    for code, group in filtered.groupby("index_code", sort=False):
        group = group.sort_values("date", kind="stable").copy()
        if group.empty:
            continue
        base = float(group["value"].iloc[0])
        if base == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Trading Date": group["date"],
                    "Index": display_labels.get(str(code), str(code)),
                    "Normalized Performance": group["value"].div(base).mul(100),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=("Trading Date", "Index", "Normalized Performance")
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["Trading Date", "Index"], kind="stable"
    ).reset_index(drop=True)


def single_index_performance(
    data: pd.DataFrame,
    period: str,
    *,
    index_code: str,
    display_label: str,
) -> pd.DataFrame:
    """Return one index's chronological raw values for the selected period."""
    filtered = filter_index_range(data, period)
    return single_index_performance_from_filtered(
        filtered,
        index_code=index_code,
        display_label=display_label,
    )


def single_index_performance_from_filtered(
    filtered: pd.DataFrame,
    *,
    index_code: str,
    display_label: str,
) -> pd.DataFrame:
    """Return raw values from one canonical pre-filtered frame."""
    filtered = filtered.copy(deep=True)
    required = {"index_code", "date", "value"}
    if filtered.empty or not required.issubset(filtered.columns):
        return pd.DataFrame(columns=("Trading Date", "Index", "Index Value"))
    selected = filtered.loc[
        filtered["index_code"].astype(str).eq(index_code),
        ["date", "value"],
    ].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.dropna(subset=["date", "value"])
    if selected.empty:
        return pd.DataFrame(columns=("Trading Date", "Index", "Index Value"))
    return pd.DataFrame(
        {
            "Trading Date": selected["date"],
            "Index": display_label,
            "Index Value": selected["value"],
        }
    ).sort_values("Trading Date", kind="stable").reset_index(drop=True)


def index_health_from_filtered_data(
    filtered: pd.DataFrame,
    *,
    index_codes: tuple[str, ...],
) -> dict[str, IndexHealth]:
    """Calculate independent health using only one pre-filtered data window."""
    frame = filtered.copy(deep=True)
    required = {"index_code", "date", "value", "volume"}
    if not required.issubset(frame.columns):
        frame = pd.DataFrame(columns=tuple(required))
    metrics = {
        code: calculate_index_metrics(frame, code)
        for code in index_codes
    }
    counts = {
        code: int(frame["index_code"].astype(str).eq(code).sum())
        for code in index_codes
    }
    return calculate_index_health_scores(metrics, observation_counts=counts)


def single_index_period_summary(performance: pd.DataFrame) -> dict[str, object]:
    """Summarize an already-filtered raw index series for display."""
    if performance.empty or "Index Value" not in performance:
        return {
            "Period Return": None,
            "Period High": None,
            "Period Low": None,
        }
    values = pd.to_numeric(performance["Index Value"], errors="coerce").dropna()
    if values.empty:
        period_return = None
        period_high = None
        period_low = None
    else:
        first = float(values.iloc[0])
        latest = float(values.iloc[-1])
        period_return = (latest / first - 1) * 100 if first else None
        period_high = float(values.max())
        period_low = float(values.min())
    return {
        "Period Return": period_return,
        "Period High": period_high,
        "Period Low": period_low,
    }


def index_health_comparison(
    health_scores: Mapping[str, IndexHealth],
    *,
    display_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Build a readable comparison without replacing independent results."""
    rows = []
    for code, label in display_labels.items():
        health = health_scores.get(code)
        rows.append(
            {
                "Index": label,
                "Health Score": health.score if health else None,
                "Condition": health.label if health else "Unavailable",
                "Observation Date": health.reference_date if health else None,
                "Trading Observations": health.observation_count if health else 0,
                "Coverage %": health.coverage_percentage if health else 0.0,
            }
        )
    return pd.DataFrame(rows)


def index_trend_summary(metric: IndexMetrics) -> str:
    """Describe an index's own moving-average position in plain language."""
    positions = (
        ("SMA-20", metric.versus_ma_20_percent),
        ("SMA-50", metric.versus_ma_50_percent),
    )
    available = [(name, value) for name, value in positions if value is not None]
    if not available:
        return "Not Available"
    descriptions = [
        f"{'Above' if value > 0 else 'Below' if value < 0 else 'At'} {name}"
        for name, value in available
    ]
    return " · ".join(descriptions)


def market_summary_values(
    *, score: float | None, condition: str, latest_date: object
) -> dict[str, object]:
    """Expose only the three values allowed in the compact market summary."""
    return {
        "Market Health Score": score,
        "Market Condition": condition,
        "Latest Trading Date": latest_date,
    }


def automation_status_label(enabled: bool | None) -> str:
    """Return a readable operational status without exposing backend enums."""
    if enabled is None:
        return "—"
    return "Enabled" if enabled else "Disabled"
