"""Offline tests for the canonical index-period analysis and health contract."""

from dataclasses import FrozenInstanceError
import math

import pandas as pd
import pandas.testing as pdt
import pytest

from dashboard.market_overview import filter_index_range, index_health_for_period
from market_intelligence.index_periods import (
    INDEX_PERIOD_OPTIONS,
    INDEX_PERIOD_VERSION,
    analyze_index_period,
    analyze_index_periods,
    combine_index_periods,
    filter_index_period,
)
from market_intelligence.market_health import (
    PERIOD_INDEX_HEALTH_VERSION,
    PERIOD_INDEX_HEALTH_WEIGHTS,
    calculate_period_index_health,
    calculate_period_index_health_scores,
)


def _series(
    code: str = "KSE100",
    *,
    rows: int = 400,
    end: str = "2026-08-05",
    values: list[float] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=rows, freq="D")
    resolved_values = values or [100.0 + step for step in range(rows)]
    return pd.DataFrame(
        {
            "index_code": code,
            "date": dates,
            "value": resolved_values,
            "volume": [1_000 + step for step in range(rows)],
        }
    )


def _regime_history() -> pd.DataFrame:
    frames = []
    for position, code in enumerate(("KSE100", "KSE30", "KMI30", "ALLSHR")):
        value = 100.0 + position * 25
        values = []
        for step in range(500):
            if step < 280:
                value += 0.18
            elif step < 390:
                value -= 0.55
            elif step < 465:
                value += 0.12
            else:
                value += 1.15 + position * 0.05
            values.append(value)
        frame = _series(code, rows=500, values=values)
        frame["volume"] = [1_000 + position * 100 + step % 23 for step in range(500)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_supported_period_contract_is_exact_and_rejects_ytd() -> None:
    assert INDEX_PERIOD_OPTIONS == ("1M", "3M", "6M", "1Y", "Maximum")
    with pytest.raises(ValueError, match="Unsupported index period"):
        filter_index_period(_series(), "KSE100", "YTD")


def test_filter_uses_selected_index_own_latest_date() -> None:
    current = _series("KSE100", rows=181, end="2026-08-05")
    lagging = _series("KSE30", rows=181, end="2026-06-05")
    source = pd.concat([current, lagging], ignore_index=True)

    selected = filter_index_period(source, "KSE30", "1M")
    combined = combine_index_periods(source, ("KSE100", "KSE30"), "1M")

    assert selected["date"].max() == pd.Timestamp("2026-06-05")
    assert selected["date"].min() == pd.Timestamp("2026-05-05")
    assert combined.loc[combined["index_code"].eq("KSE100"), "date"].min() == pd.Timestamp(
        "2026-07-05"
    )
    assert combined.loc[combined["index_code"].eq("KSE30"), "date"].min() == pd.Timestamp(
        "2026-05-05"
    )


def test_legacy_range_wrapper_delegates_to_per_index_contract() -> None:
    source = pd.concat(
        [
            _series("KSE100", rows=181, end="2026-08-05"),
            _series("KSE30", rows=181, end="2026-06-05"),
        ],
        ignore_index=True,
    )

    filtered = filter_index_range(source, "1M")

    starts = filtered.groupby("index_code")["date"].min().to_dict()
    assert starts == {
        "KSE100": pd.Timestamp("2026-07-05"),
        "KSE30": pd.Timestamp("2026-05-05"),
    }


def test_metadata_exposes_actual_contract_and_is_frozen() -> None:
    analysis = analyze_index_period(_series(rows=100), "KSE100", "3M")
    metadata = analysis.metadata

    assert metadata.contract_version == INDEX_PERIOD_VERSION
    assert metadata.index_code == "KSE100"
    assert metadata.requested_period == "3M"
    assert metadata.actual_start_date.isoformat() == "2026-05-05"
    assert metadata.actual_end_date.isoformat() == "2026-08-05"
    assert metadata.observations == 93
    assert metadata.start_value == 107.0
    assert metadata.end_value == 199.0
    with pytest.raises(FrozenInstanceError):
        metadata.observations = 5  # type: ignore[misc]


def test_maximum_uses_complete_selected_index_series() -> None:
    source = pd.concat([_series("KSE100", rows=400), _series("KSE30", rows=120)])

    analysis = analyze_index_period(source, "KSE100", "Maximum")

    assert analysis.metadata.observations == 400
    assert analysis.metadata.actual_start_date == source.iloc[0]["date"].date()
    assert len(analysis.causal_frame) == 400


def test_periods_produce_expected_nested_boundaries() -> None:
    source = _series(rows=500)
    analyses = {
        period: analyze_index_period(source, "KSE100", period)
        for period in INDEX_PERIOD_OPTIONS
    }

    counts = [analyses[period].metadata.observations for period in INDEX_PERIOD_OPTIONS]
    starts = [analyses[period].metadata.actual_start_date for period in INDEX_PERIOD_OPTIONS]
    assert counts == sorted(counts)
    assert len(set(counts)) == 5
    assert starts == sorted(starts, reverse=True)
    assert all(
        analysis.metadata.actual_end_date.isoformat() == "2026-08-05"
        for analysis in analyses.values()
    )


def test_selected_period_metrics_are_exact() -> None:
    values = [100.0, 110.0, 105.0, 120.0]
    analysis = analyze_index_period(
        _series(rows=4, end="2026-01-04", values=values),
        "KSE100",
        "Maximum",
    )
    returns = pd.Series(values).pct_change(fill_method=None).dropna()
    expected_volatility = returns.std(ddof=0) * math.sqrt(252) * 100

    assert analysis.period_return_percent == pytest.approx(20.0)
    assert analysis.period_high == 120.0
    assert analysis.period_low == 100.0
    assert analysis.latest_value == 120.0
    assert analysis.annualized_volatility_percent == pytest.approx(expected_volatility)
    assert analysis.maximum_drawdown_percent == pytest.approx((105 / 110 - 1) * 100)
    assert analysis.trend_consistency_percent == pytest.approx(2 / 3 * 100)


def test_causal_features_never_pull_pre_period_observations() -> None:
    source = _series(rows=100)
    source.loc[:60, "value"] = 1_000_000.0
    selected = filter_index_period(source, "KSE100", "1M")
    isolated = selected.copy(deep=True)

    from_full = analyze_index_period(source, "KSE100", "1M")
    from_isolated = analyze_index_period(isolated, "KSE100", "Maximum")

    pdt.assert_series_equal(
        from_full.causal_frame["ma_20"],
        from_isolated.causal_frame["ma_20"],
        check_names=False,
    )
    pdt.assert_series_equal(
        from_full.causal_frame["rolling_volatility_20_percent"],
        from_isolated.causal_frame["rolling_volatility_20_percent"],
        check_names=False,
    )
    assert pd.isna(from_full.causal_frame.loc[0, "daily_change"])
    assert pd.isna(from_full.causal_frame.loc[0, "daily_change_percent"])
    assert from_full.causal_frame["ma_20"].first_valid_index() == 19
    assert from_full.causal_frame["ma_50"].isna().all()


def test_ma20_ma50_and_rolling_volatility_are_causal() -> None:
    source = _series(rows=80)
    original_analysis = analyze_index_period(source, "KSE100", "Maximum")
    changed = source.copy(deep=True)
    changed.loc[60:, "value"] *= 3
    changed_analysis = analyze_index_period(changed, "KSE100", "Maximum")

    original = original_analysis.causal_frame
    revised = changed_analysis.causal_frame
    pdt.assert_series_equal(original.loc[:59, "ma_20"], revised.loc[:59, "ma_20"])
    pdt.assert_series_equal(original.loc[:59, "ma_50"], revised.loc[:59, "ma_50"])
    pdt.assert_series_equal(
        original.loc[:59, "rolling_volatility_20_percent"],
        revised.loc[:59, "rolling_volatility_20_percent"],
    )
    assert original["ma_20"].first_valid_index() == 19
    assert original["ma_50"].first_valid_index() == 49
    assert original["rolling_volatility_20_percent"].first_valid_index() == 20


def test_drawdown_path_is_causal_and_correct() -> None:
    analysis = analyze_index_period(
        _series(rows=5, values=[100.0, 120.0, 90.0, 108.0, 135.0]),
        "KSE100",
        "Maximum",
    )

    assert analysis.causal_frame["drawdown_percent"].tolist() == pytest.approx(
        [0.0, 0.0, -25.0, -10.0, 0.0]
    )
    assert analysis.maximum_drawdown_percent == -25.0


def test_source_dataframe_is_not_mutated() -> None:
    source = _series(rows=100).sample(frac=1, random_state=7).reset_index(drop=True)
    original = source.copy(deep=True)

    analyze_index_period(source, "KSE100", "1M")
    filter_index_period(source, "KSE100", "6M")

    pdt.assert_frame_equal(source, original)


def test_period_health_has_explicit_version_weights_inputs_and_contributions() -> None:
    analysis = analyze_index_period(_regime_history(), "KSE30", "6M")
    health = calculate_period_index_health(analysis)

    assert health.methodology_version == PERIOD_INDEX_HEALTH_VERSION
    assert health.requested_period == "6M"
    assert tuple(health.component_scores) == tuple(PERIOD_INDEX_HEALTH_WEIGHTS)
    assert sum(PERIOD_INDEX_HEALTH_WEIGHTS.values()) == 100
    assert health.component_inputs["Selected-period return"] == pytest.approx(
        analysis.period_return_percent
    )
    assert health.component_inputs[
        "Selected-period volatility resilience"
    ] == pytest.approx(analysis.annualized_volatility_percent)
    normalized_total = sum(
        value for value in health.normalized_contributions.values() if value is not None
    )
    assert normalized_total == pytest.approx(health.score, abs=0.05)
    assert 0 <= health.score <= 100


def test_health_recomputes_independently_for_every_period() -> None:
    source = _regime_history()
    health = {
        period: calculate_period_index_health(
            analyze_index_period(source, "KSE30", period)
        )
        for period in INDEX_PERIOD_OPTIONS
    }

    assert len({result.observation_count for result in health.values()}) == 5
    assert len({result.score for result in health.values()}) >= 4
    assert len(
        {
            result.component_inputs["Selected-period return"]
            for result in health.values()
        }
    ) == 5


def test_short_period_reweights_unavailable_sma50_without_zero_award() -> None:
    health = calculate_period_index_health(
        analyze_index_period(_series(rows=400), "KSE100", "1M")
    )

    assert health.component_inputs["Value relative to SMA-50"] is None
    assert health.component_scores["Value relative to SMA-50"] is None
    assert health.normalized_contributions["Value relative to SMA-50"] is None
    assert "Value relative to SMA-50" in health.unavailable_components
    assert health.available_weight == 90
    assert health.coverage_percentage == 90.0
    assert 0 <= health.score <= 100


def test_empty_period_health_is_unavailable_without_crashing() -> None:
    analysis = analyze_index_period(pd.DataFrame(), "KSE100", "1M")
    health = calculate_period_index_health(analysis)

    assert health.score is None
    assert health.label == "Unavailable"
    assert health.available_weight == 0
    assert health.coverage_percentage == 0.0
    assert set(health.unavailable_components) == set(PERIOD_INDEX_HEALTH_WEIGHTS)


def test_extreme_inputs_remain_bounded() -> None:
    rising = _series(rows=100, values=[100.0 * 1.05**step for step in range(100)])
    falling = _series(rows=100, values=[10_000.0 * 0.95**step for step in range(100)])

    for source in (rising, falling):
        score = calculate_period_index_health(
            analyze_index_period(source, "KSE100", "Maximum")
        ).score
        assert score is not None
        assert 0 <= score <= 100


def test_all_indices_receive_independent_period_health() -> None:
    source = _regime_history()
    codes = ("KSE100", "KSE30", "KMI30", "ALLSHR")
    analyses = analyze_index_periods(source, codes, "3M")
    direct = calculate_period_index_health_scores(analyses)
    wrapped = index_health_for_period(source, "3M", index_codes=codes)

    assert tuple(direct) == codes
    assert direct == wrapped
    assert all(result.requested_period == "3M" for result in direct.values())
    assert all(result.observation_count == 93 for result in direct.values())
    assert len({result.score for result in direct.values()}) > 1
