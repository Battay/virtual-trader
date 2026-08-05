"""Offline tests for Market Overview presentation transformations."""

import pandas as pd
import pandas.testing as pdt
import pytest

from dashboard.market_overview import (
    ALL_INDICES_LABEL,
    INDEX_VIEW_CODES,
    INDEX_VIEW_OPTIONS,
    automation_status_label,
    filter_index_range,
    market_summary_values,
    normalized_index_performance,
    single_index_performance,
    single_index_period_summary,
)


def _indices() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "index_code": code,
                    "date": dates,
                    "value": [base + step for step in range(100)],
                }
            )
            for code, base in (("KSE100", 100.0), ("KSE30", 200.0))
        ],
        ignore_index=True,
    )


def test_normalized_comparison_starts_each_available_series_at_100() -> None:
    normalized = normalized_index_performance(
        _indices(),
        "3M",
        display_labels={"KSE100": "KSE-100 Index", "KSE30": "KSE-30 Index"},
    )

    first_values = normalized.groupby("Index")["Normalized Performance"].first()
    assert first_values.to_dict() == {
        "KSE-100 Index": 100.0,
        "KSE-30 Index": 100.0,
    }


def test_index_selector_uses_readable_labels_and_defaults_to_all_indices() -> None:
    assert INDEX_VIEW_OPTIONS[0] == ALL_INDICES_LABEL == "All Indices"
    assert INDEX_VIEW_OPTIONS == (
        "All Indices",
        "KSE-100 Index",
        "KSE-30 Index",
        "KMI-30 Index",
        "KSE All Share Index",
    )
    assert not {"KSE100", "KSE30", "KMI30", "ALLSHR"}.intersection(
        INDEX_VIEW_OPTIONS
    )


def test_selector_mapping_preserves_official_internal_codes() -> None:
    assert INDEX_VIEW_CODES == {
        "All Indices": None,
        "KSE-100 Index": "KSE100",
        "KSE-30 Index": "KSE30",
        "KMI-30 Index": "KMI30",
        "KSE All Share Index": "ALLSHR",
    }


def test_single_index_mode_returns_only_raw_selected_series() -> None:
    performance = single_index_performance(
        _indices(),
        "Maximum",
        index_code="KSE30",
        display_label="KSE-30 Index",
    )

    assert performance["Index"].unique().tolist() == ["KSE-30 Index"]
    assert performance["Index Value"].iloc[0] == 200.0
    assert performance["Index Value"].iloc[-1] == 299.0
    assert "Normalized Performance" not in performance


def test_single_index_range_filter_does_not_mutate_source() -> None:
    source = _indices().sample(frac=1, random_state=11).reset_index(drop=True)
    original = source.copy(deep=True)

    performance = single_index_performance(
        source,
        "1M",
        index_code="KSE100",
        display_label="KSE-100 Index",
    )

    pdt.assert_frame_equal(source, original)
    dates = performance["Trading Date"]
    assert dates.is_monotonic_increasing
    assert dates.min() >= dates.max() - pd.DateOffset(months=1)


def test_missing_selected_index_is_safe() -> None:
    performance = single_index_performance(
        _indices(),
        "6M",
        index_code="ALLSHR",
        display_label="KSE All Share Index",
    )

    assert performance.empty
    assert single_index_period_summary(performance) == {
        "Period Return": None,
        "Period High": None,
        "Period Low": None,
    }


def test_single_index_period_summary_uses_filtered_raw_values() -> None:
    performance = single_index_performance(
        _indices(),
        "Maximum",
        index_code="KSE100",
        display_label="KSE-100 Index",
    )

    assert single_index_period_summary(performance) == {
        "Period Return": 99.0,
        "Period High": 199.0,
        "Period Low": 100.0,
    }


def test_range_filter_is_local_chronological_and_does_not_mutate_source() -> None:
    source = _indices().sample(frac=1, random_state=7).reset_index(drop=True)
    original = source.copy(deep=True)

    filtered = filter_index_range(source, "1M")

    pdt.assert_frame_equal(source, original)
    assert filtered["date"].is_monotonic_increasing
    assert filtered["date"].min() >= filtered["date"].max() - pd.DateOffset(months=1)


def test_missing_or_zero_base_index_series_are_handled_gracefully() -> None:
    missing = normalized_index_performance(
        pd.DataFrame(), "1M", display_labels={}
    )
    zero_base = normalized_index_performance(
        pd.DataFrame(
            {
                "index_code": ["KSE100", "KSE100"],
                "date": ["2025-01-01", "2025-01-02"],
                "value": [0, 1],
            }
        ),
        "Maximum",
        display_labels={"KSE100": "KSE-100 Index"},
    )

    assert missing.empty
    assert zero_base.empty


def test_market_summary_contains_no_index_or_breadth_duplicates() -> None:
    summary = market_summary_values(
        score=56,
        condition="Bullish",
        latest_date="2026-07-31",
    )

    assert tuple(summary) == (
        "Market Health Score",
        "Market Condition",
        "Latest Trading Date",
    )
    forbidden = {"KSE-100", "Advancers", "Decliners", "AI-Ready Symbols"}
    assert forbidden.isdisjoint(summary)


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, "Enabled"), (False, "Disabled"), (None, "—")],
)
def test_automation_status_is_readable(enabled: bool | None, expected: str) -> None:
    assert automation_status_label(enabled) == expected
