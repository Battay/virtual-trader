"""Deterministic offline tests for independent PSX index health scoring."""

from dataclasses import replace

import pandas as pd

from dashboard.market_overview import (
    filter_index_range,
    index_health_comparison,
    index_health_from_filtered_data,
    single_index_performance_from_filtered,
)
from market_intelligence.index_config import SUPPORTED_INDICES
from market_intelligence.index_metrics import calculate_index_metrics
from market_intelligence.market_breadth import calculate_market_breadth
from market_intelligence.market_health import (
    INDEX_HEALTH_WEIGHTS,
    IndexHealth,
    calculate_index_health,
    calculate_index_health_scores,
    calculate_market_health,
)


def _index_history() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    frames = []
    for code, start, end, volume in (
        ("KSE100", 100.0, 200.0, 1_500),
        ("KSE30", 200.0, 100.0, 800),
        ("KMI30", 100.0, 105.0, 1_000),
        ("ALLSHR", 100.0, 150.0, 1_200),
    ):
        values = [start + (end - start) * step / 79 for step in range(80)]
        frames.append(
            pd.DataFrame(
                {
                    "index_code": code,
                    "date": dates,
                    "value": values,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _metrics(data: pd.DataFrame):
    return {
        code: calculate_index_metrics(data, code)
        for code in SUPPORTED_INDICES
    }


def test_index_health_weights_total_100_and_scores_are_clamped() -> None:
    assert sum(INDEX_HEALTH_WEIGHTS.values()) == 100
    for health in calculate_index_health_scores(_metrics(_index_history())).values():
        assert health.score is not None
        assert 0 <= health.score <= 100


def test_each_index_receives_an_independent_score_and_condition() -> None:
    health = calculate_index_health_scores(_metrics(_index_history()))

    assert tuple(health) == tuple(SUPPORTED_INDICES)
    assert health["KSE100"].label == "Strongly Bullish"
    assert health["KSE30"].label == "Strongly Bearish"
    assert len({result.score for result in health.values()}) > 1
    assert len({result.label for result in health.values()}) > 1


def test_other_index_observations_cannot_change_selected_index_health() -> None:
    source = _index_history()
    before = calculate_index_health(calculate_index_metrics(source, "KSE100"))
    changed = source.copy(deep=True)
    changed.loc[changed["index_code"].eq("KSE30"), "value"] *= 100
    after = calculate_index_health(calculate_index_metrics(changed, "KSE100"))

    assert after == before


def test_insufficient_history_is_explicitly_unavailable() -> None:
    metric = calculate_index_metrics(_index_history(), "MISSING")
    health = calculate_index_health(metric)

    assert health.score is None
    assert health.label == "Unavailable"
    assert all(value is None for value in health.component_scores.values())
    assert "Insufficient history" in health.explanations[0]


def test_overall_market_health_remains_separate_from_index_health() -> None:
    metrics = _metrics(_index_history())
    index_health = calculate_index_health_scores(metrics)
    breadth = calculate_market_breadth(
        pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "date": ["2025-03-21"] * 3,
                "change": [1, -1, 0],
                "volume": [10, 20, 30],
            }
        )
    )
    overall = calculate_market_health(metrics, breadth)

    assert overall is not index_health
    assert "Advance/decline ratio" in overall.component_scores
    assert all(
        "Advance/decline ratio" not in result.component_scores
        for result in index_health.values()
    )


def test_all_indices_comparison_preserves_four_independent_results() -> None:
    health = calculate_index_health_scores(_metrics(_index_history()))
    labels = {
        code: definition.display_name
        for code, definition in SUPPORTED_INDICES.items()
    }
    comparison = index_health_comparison(health, display_labels=labels)

    assert comparison["Index"].tolist() == list(labels.values())
    assert comparison["Health Score"].nunique() > 1
    assert comparison["Condition"].nunique() > 1
    assert not set(SUPPORTED_INDICES).intersection(comparison["Index"])


def test_selected_index_uses_its_own_health_not_an_overall_condition() -> None:
    health = calculate_index_health_scores(_metrics(_index_history()))
    selected_code = "KSE30"
    selected = health[selected_code]

    assert selected.index_code == selected_code
    assert selected.score == health["KSE30"].score
    assert selected.label == "Strongly Bearish"
    assert selected.label != health["KSE100"].label


def test_comparison_does_not_copy_one_global_condition_to_every_card() -> None:
    base = calculate_index_health(calculate_index_metrics(_index_history(), "KSE100"))
    results: dict[str, IndexHealth] = {
        "KSE100": base,
        "KSE30": replace(base, index_code="KSE30", score=20.0, label="Strongly Bearish"),
        "KMI30": replace(base, index_code="KMI30", score=50.0, label="Neutral"),
        "ALLSHR": replace(base, index_code="ALLSHR", score=65.0, label="Bullish"),
    }
    labels = {
        code: definition.display_name
        for code, definition in SUPPORTED_INDICES.items()
    }

    comparison = index_health_comparison(results, display_labels=labels)

    assert comparison["Condition"].tolist() == [
        base.label,
        "Strongly Bearish",
        "Neutral",
        "Bullish",
    ]


def _period_sensitive_history() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    frames = []
    for position, code in enumerate(SUPPORTED_INDICES):
        values = []
        value = 100.0 + position * 20
        for step in range(400):
            if step < 310:
                value += 0.15
            elif step < 370:
                value -= 0.45
            else:
                value += 1.2
            values.append(value)
        frames.append(
            pd.DataFrame(
                {
                    "index_code": code,
                    "date": dates,
                    "value": values,
                    "volume": [1_000 + step % 20 for step in range(400)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_period_change_filters_different_input_before_health_scoring() -> None:
    source = _period_sensitive_history()
    one_month = filter_index_range(source, "1M")
    three_months = filter_index_range(source, "3M")
    health_1m = index_health_from_filtered_data(
        one_month, index_codes=("KSE30",)
    )["KSE30"]
    health_3m = index_health_from_filtered_data(
        three_months, index_codes=("KSE30",)
    )["KSE30"]

    assert len(one_month) < len(three_months)
    assert health_1m.observation_count < health_3m.observation_count
    assert health_1m.score != health_3m.score
    assert health_1m.component_scores != health_3m.component_scores


def test_maximum_health_uses_complete_available_series() -> None:
    source = _period_sensitive_history()
    maximum = filter_index_range(source, "Maximum")
    result = index_health_from_filtered_data(
        maximum, index_codes=("KSE100",)
    )["KSE100"]

    assert result.observation_count == 400
    assert maximum["index_code"].eq("KSE100").sum() == 400


def test_short_period_reports_unavailable_sma50_and_component_coverage() -> None:
    visible = filter_index_range(_period_sensitive_history(), "1M")
    result = index_health_from_filtered_data(
        visible, index_codes=("KSE30",)
    )["KSE30"]

    assert result.observation_count < 50
    assert result.component_scores["Value relative to SMA-50"] is None
    assert "Value relative to SMA-50" in result.unavailable_components
    assert result.available_weight < 100
    assert result.coverage_percentage == result.available_weight
    assert result.score is not None and 0 <= result.score <= 100


def test_chart_and_health_share_exact_period_boundaries_without_mutation() -> None:
    source = _period_sensitive_history()
    original = source.copy(deep=True)
    visible = filter_index_range(source, "3M")
    chart = single_index_performance_from_filtered(
        visible,
        index_code="KSE30",
        display_label="KSE-30 Index",
    )
    health = index_health_from_filtered_data(
        visible, index_codes=("KSE30",)
    )["KSE30"]

    pd.testing.assert_frame_equal(source, original)
    assert health.observation_count == len(chart)
    assert pd.Timestamp(health.reference_date) == chart["Trading Date"].max()
    assert chart["Trading Date"].min() == visible.loc[
        visible["index_code"].eq("KSE30"), "date"
    ].min()


def test_all_indices_health_is_period_specific_and_independent() -> None:
    source = _period_sensitive_history()
    visible = filter_index_range(source, "6M")
    results = index_health_from_filtered_data(
        visible, index_codes=tuple(SUPPORTED_INDICES)
    )

    assert tuple(results) == tuple(SUPPORTED_INDICES)
    assert all(result.observation_count == 185 for result in results.values())
    assert all(result.reference_date.isoformat() == "2026-02-04" for result in results.values())
