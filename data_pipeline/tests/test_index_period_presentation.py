"""Offline tests for canonical index-period dashboard presentation helpers."""

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

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
from market_intelligence.index_periods import (
    INDEX_PERIOD_OPTIONS,
    analyze_index_period,
)
from market_intelligence.market_health import (
    PERIOD_INDEX_HEALTH_WEIGHTS,
    calculate_period_index_health,
)


def _indices(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range(end="2026-08-05", periods=rows, freq="D")
    frames = []
    for code, offset in (("KSE100", 0.0), ("KSE30", 50.0)):
        frames.append(
            pd.DataFrame(
                {
                    "index_code": code,
                    "date": dates,
                    "value": [100.0 + offset + step * 0.5 for step in range(rows)],
                    "volume": [1_000 + step for step in range(rows)],
                    "open": [99.0 + offset + step * 0.5 for step in range(rows)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_period_widget_resolution_uses_exact_contract_and_safe_fallback() -> None:
    assert DEFAULT_INDEX_PERIOD == "6M"
    assert tuple(resolve_index_period(period) for period in INDEX_PERIOD_OPTIONS) == (
        "1M",
        "3M",
        "6M",
        "1Y",
        "Maximum",
    )
    assert resolve_index_period(None) == "6M"
    assert resolve_index_period("YTD") == "6M"
    with pytest.raises(ValueError, match="Unsupported default"):
        resolve_index_period(None, default="YTD")


def test_source_cache_identity_includes_path_mtime_and_size(tmp_path: Path) -> None:
    source = tmp_path / "indices.csv"
    missing = index_source_identity(source)
    source.write_text("index_code,date,value\nKSE100,2026-08-05,100\n", encoding="utf-8")
    present = index_source_identity(source)

    assert missing == (str(source), None, None)
    assert present[0] == str(source)
    assert isinstance(present[1], int)
    assert present[2] == source.stat().st_size


def test_contract_and_summary_values_come_from_same_selected_analysis() -> None:
    source = _indices()
    analysis = analyze_index_period(source, "KSE30", "1M")
    health = calculate_period_index_health(analysis)

    contract = index_period_contract_values(analysis)
    summary = index_period_summary_values(analysis, health)

    assert contract == {
        "Requested Period": "1M",
        "Actual Start Date": analysis.metadata.actual_start_date,
        "Actual End Date": analysis.metadata.actual_end_date,
        "Trading Observations": analysis.metadata.observations,
        "Start Value": analysis.metadata.start_value,
        "End Value": analysis.metadata.end_value,
    }
    assert summary["Selected-Period Return"] == analysis.period_return_percent
    assert summary["Selected-Period Volatility"] == (
        analysis.annualized_volatility_percent
    )
    assert summary["Maximum Drawdown"] == analysis.maximum_drawdown_percent
    assert summary["Health Score"] == health.score


def test_health_breakdown_reproduces_score_and_keeps_missing_component_missing() -> None:
    analysis = analyze_index_period(_indices(), "KSE100", "1M")
    health = calculate_period_index_health(analysis)
    breakdown = index_health_breakdown_frame(health)

    assert breakdown["Component"].tolist() == list(PERIOD_INDEX_HEALTH_WEIGHTS)
    assert breakdown["Configured Weight"].sum() == 100
    assert breakdown["Normalized Contribution"].sum() == pytest.approx(
        health.score, abs=0.05
    )
    sma50 = breakdown.loc[
        breakdown["Component"].eq("Value relative to SMA-50")
    ].iloc[0]
    assert pd.isna(sma50["Input"])
    assert pd.isna(sma50["Raw Points"])
    assert pd.isna(sma50["Normalized Contribution"])


def test_chart_frame_is_exact_causal_period_and_does_not_mutate_source() -> None:
    source = _indices().sample(frac=1, random_state=17).reset_index(drop=True)
    original = source.copy(deep=True)
    analysis = analyze_index_period(source, "KSE30", "1M")

    chart = index_chart_frame(analysis, display_label="KSE-30 Index")

    pdt.assert_frame_equal(source, original)
    assert len(chart) == analysis.metadata.observations
    assert chart["Trading Date"].min().date() == analysis.metadata.actual_start_date
    assert chart["Trading Date"].max().date() == analysis.metadata.actual_end_date
    assert chart["Index"].unique().tolist() == ["KSE-30 Index"]
    assert pd.isna(chart.loc[0, "Daily Change"])
    assert pd.isna(chart.loc[0, "Daily Change %"])
    assert chart["MA20"].first_valid_index() == 19
    assert chart["MA50"].isna().all()
    assert chart["Rolling Volatility 20D %"].first_valid_index() == 20
    assert chart["Index Level"].iloc[0] == analysis.metadata.start_value
    assert chart["Index Level"].iloc[-1] == analysis.metadata.end_value


def test_chart_specs_expose_required_series_and_hover_fields() -> None:
    analysis = analyze_index_period(_indices(), "KSE100", "Maximum")
    frame = index_chart_frame(analysis, display_label="KSE-100 Index")

    level = index_level_chart(frame).to_dict()
    drawdown = index_drawdown_chart(frame).to_dict()
    volatility = index_rolling_volatility_chart(frame).to_dict()

    assert level["transform"][0]["fold"] == ["Index Level", "MA20", "MA50"]
    tooltip_fields = {
        item["field"] for item in level["encoding"]["tooltip"]
    }
    assert {"Trading Date", "Index Level", "Daily Change", "Daily Change %"} <= (
        tooltip_fields
    )
    assert drawdown["encoding"]["y"]["field"] == "Drawdown %"
    assert volatility["encoding"]["y"]["field"] == (
        "Rolling Volatility 20D %"
    )


def test_changing_index_and_period_changes_presentation_input() -> None:
    source = _indices()
    kse100_1m = index_chart_frame(
        analyze_index_period(source, "KSE100", "1M"),
        display_label="KSE-100 Index",
    )
    kse100_3m = index_chart_frame(
        analyze_index_period(source, "KSE100", "3M"),
        display_label="KSE-100 Index",
    )
    kse30_1m = index_chart_frame(
        analyze_index_period(source, "KSE30", "1M"),
        display_label="KSE-30 Index",
    )

    assert len(kse100_1m) < len(kse100_3m)
    assert kse100_1m["Trading Date"].min() > kse100_3m["Trading Date"].min()
    assert not kse100_1m["Index Level"].equals(kse30_1m["Index Level"])


def test_both_market_pages_are_wired_to_the_canonical_period_contract() -> None:
    project_root = Path(__file__).resolve().parents[2]
    overview = (project_root / "app_pages" / "0_Market_Overview.py").read_text(
        encoding="utf-8"
    )
    indices = (project_root / "app_pages" / "1_Market_Indices.py").read_text(
        encoding="utf-8"
    )

    assert "index_period_analyses(" in overview
    assert "calculate_period_index_health_scores(" in overview
    assert "analyze_index_period(" in indices
    assert "calculate_period_index_health(" in indices
    assert "pd.DateOffset" not in overview
    assert "pd.DateOffset" not in indices
    assert "calculate_index_metrics(data, code)" not in indices
