"""Offline tests for the non-final Phase 1 clustering methodology audit."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.clustering_methodology import (
    ClusteringMethodologyError,
    NonPositiveCloseError,
    build_eligibility_sensitivity,
    build_return_matrix,
    canonical_temporal_date_partitions,
    compare_correlation_structures,
    construct_close_returns,
    correlation_to_distance,
    deterministic_complete_pair_core,
    evaluate_cluster_counts,
    evaluate_temporal_stability,
    minimum_overlap_correlation,
    pairwise_overlap_counts,
    pairwise_overlap_table,
    training_market_view,
)
from data_pipeline.src.parquet_market_data import load_market_data


MARKET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32()),
        pa.field("symbol", pa.string()),
        pa.field("ldcp", pa.float64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("change_percent", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)


def _return_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            "symbol": ["B", "A", "A", "B", "B"],
            "close": [12.0, 20.0, 22.0, 11.0, 10.0],
        }
    )


def test_returns_are_within_symbol_sorted_and_missing_sessions_are_not_filled() -> None:
    market = _return_market()
    source = market.copy(deep=True)
    result = construct_close_returns(
        market,
        ["B", "A"],
        global_market_dates=pd.date_range("2024-01-01", periods=3),
    )

    # A has no 2 January observation, so its 1 -> 3 January price change is
    # rejected rather than treated as a daily return or filled synthetically.
    assert result.symbols_with_returns == 1
    assert result.first_observation_rows_rejected == 2
    assert result.gap_spanning_rows_rejected == 1
    assert result.returns["symbol"].tolist() == ["B", "B"]
    assert result.returns["market_date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    assert result.returns["simple_return"].tolist() == pytest.approx(
        [0.10, 1.0 / 11.0]
    )
    assert result.returns["log_return"].tolist() == pytest.approx(
        [np.log(1.1), np.log(12.0 / 11.0)]
    )
    pd.testing.assert_frame_equal(market, source)


def test_non_positive_close_is_rejected_before_return_construction() -> None:
    market = _return_market()
    market.loc[market.index[0], "close"] = 0.0

    with pytest.raises(NonPositiveCloseError, match="positive close"):
        construct_close_returns(
            market,
            ["A", "B"],
            global_market_dates=pd.date_range("2024-01-01", periods=3),
        )


def _return_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, np.nan],
            "B": [0.02, 0.04, 0.06, -0.08, 0.10],
            "C": [np.nan, np.nan, 0.05, np.nan, 0.07],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )


def test_pairwise_overlap_counts_ratios_and_minimum_overlap_correlations() -> None:
    matrix = _return_matrix()
    counts = pairwise_overlap_counts(matrix)
    pairs = pairwise_overlap_table(matrix, counts).set_index(
        ["symbol_a", "symbol_b"]
    )

    assert counts.loc["A", "A"] == 4
    assert counts.loc["A", "B"] == 4
    assert counts.loc["A", "C"] == 1
    assert counts.loc["B", "C"] == 2
    assert pairs.loc[("A", "B"), "overlap_ratio_min_history"] == 1.0
    assert pairs.loc[("B", "C"), "overlap_ratio_union"] == pytest.approx(0.4)

    pearson = minimum_overlap_correlation(
        matrix, method="pearson", minimum_overlap=3, overlap_counts=counts
    )
    spearman = minimum_overlap_correlation(
        matrix, method="spearman", minimum_overlap=3, overlap_counts=counts
    )
    expected_pearson = matrix["A"].corr(matrix["B"], method="pearson")
    expected_spearman = matrix["A"].corr(matrix["B"], method="spearman")
    assert pearson.loc["A", "B"] == pytest.approx(expected_pearson)
    assert spearman.loc["A", "B"] == pytest.approx(expected_spearman)
    assert pearson.loc["A", "B"] != pytest.approx(spearman.loc["A", "B"])
    assert np.isnan(pearson.loc["A", "C"])
    assert np.isnan(spearman.loc["B", "C"])

    comparison = compare_correlation_structures(pearson, spearman)
    assert comparison["shared_finite_pairs"] == 1
    assert comparison["mean_absolute_difference"] == pytest.approx(
        abs(expected_pearson - expected_spearman)
    )


def test_return_matrix_never_fills_missing_values() -> None:
    returns = pd.DataFrame(
        {
            "market_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["A", "B"],
            "simple_return": [0.1, 0.2],
            "log_return": [np.log(1.1), np.log(1.2)],
        }
    )

    matrix = build_return_matrix(
        returns, value_column="log_return", symbols=["B", "A", "C"]
    )

    assert list(matrix.columns) == ["A", "B", "C"]
    assert matrix.isna().sum().to_dict() == {"A": 1, "B": 1, "C": 2}


def test_correlation_distance_transforms_and_ward_guard() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    angular = correlation_to_distance(correlation, transform="angular_chord")
    one_minus = correlation_to_distance(correlation, transform="one_minus")

    assert angular.loc["A", "B"] == pytest.approx(np.sqrt(2.0))
    assert angular.loc["A", "C"] == pytest.approx(2.0)
    assert one_minus.loc["A", "B"] == pytest.approx(1.0)
    assert one_minus.loc["A", "C"] == pytest.approx(2.0)
    assert np.diag(angular).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ClusteringMethodologyError, match="Ward linkage"):
        evaluate_cluster_counts(
            correlation,
            angular,
            cluster_counts=(2,),
            linkages=("ward",),
        )


def test_canonical_temporal_split_keeps_validation_and_test_out_of_training() -> None:
    dates = pd.date_range("2024-01-01", periods=10)
    partitions = canonical_temporal_date_partitions(dates)
    market = pd.DataFrame(
        {
            "market_date": list(dates) * 2,
            "symbol": ["B"] * 10 + ["A"] * 10,
            "close": np.arange(20, dtype=float) + 10.0,
        }
    )
    source = market.copy(deep=True)

    training = training_market_view(market, partitions)

    assert len(partitions.training_dates) == 7
    assert len(partitions.validation_dates) == 1
    assert len(partitions.test_dates) == 2
    assert set(training["market_date"]) == set(dates[:7])
    assert set(training["market_date"]).isdisjoint(partitions.validation_dates)
    assert set(training["market_date"]).isdisjoint(partitions.test_dates)
    assert list(training["symbol"].head(2)) == ["A", "B"]
    pd.testing.assert_frame_equal(market, source)


def test_complete_pair_core_and_eligibility_sensitivity_are_deterministic() -> None:
    symbols = ["A", "B", "C", "D"]
    valid = pd.DataFrame(True, index=symbols, columns=symbols)
    valid.loc["C", "D"] = False
    valid.loc["D", "C"] = False
    first_core = deterministic_complete_pair_core(valid)
    second_core = deterministic_complete_pair_core(valid)
    assert first_core == second_core == ("A", "B", "C")

    diagnostics = pd.DataFrame(
        {
            "symbol": symbols,
            "training_return_observations": [300, 300, 200, 50],
            "active_span_coverage": [0.9, 0.8, 0.75, 1.0],
        }
    )
    overlaps = pd.DataFrame(
        [
            [300, 280, 180, 40],
            [280, 300, 190, 45],
            [180, 190, 200, 30],
            [40, 45, 30, 50],
        ],
        index=symbols,
        columns=symbols,
    )
    first = build_eligibility_sensitivity(
        diagnostics,
        overlaps,
        overlap_thresholds=(120,),
        active_coverage_thresholds=(0.7,),
        minimum_peers=(1, 2),
    )
    second = build_eligibility_sensitivity(
        diagnostics.sample(frac=1.0, random_state=7),
        overlaps,
        overlap_thresholds=(120,),
        active_coverage_thresholds=(0.7,),
        minimum_peers=(1, 2),
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["eligible_symbol_count"].tolist() == [3, 3]


def _diagnostic_correlation() -> pd.DataFrame:
    symbols = list("ABCDEF")
    factors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    correlation = np.corrcoef(factors)
    # np.corrcoef of two-dimensional row vectors yields +/-1; add a small
    # positive-definite identity component to keep a useful finite distance.
    correlation = 0.8 * correlation + 0.2 * np.eye(len(symbols))
    return pd.DataFrame(correlation, index=symbols, columns=symbols)


def test_cluster_count_and_temporal_stability_diagnostics_are_deterministic() -> None:
    correlation = _diagnostic_correlation()
    distance = correlation_to_distance(correlation)
    sectors = {
        symbol: ("LEFT" if symbol < "D" else "RIGHT")
        for symbol in correlation.index
    }

    first = evaluate_cluster_counts(
        correlation,
        distance,
        sector_by_symbol=sectors,
        cluster_counts=(2, 3),
    )
    second = evaluate_cluster_counts(
        correlation,
        distance,
        sector_by_symbol=sectors,
        cluster_counts=(2, 3),
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["status"]) == {"completed_non_final_diagnostic"}
    assert (first["sector_nmi_posthoc"] >= 0).all()

    stability = evaluate_temporal_stability(
        correlation, correlation.copy(deep=True), cluster_counts=(2, 3)
    )
    assert stability["adjusted_rand_index"].tolist() == pytest.approx([1, 1, 1, 1])
    assert stability["normalized_mutual_information"].tolist() == pytest.approx(
        [1, 1, 1, 1]
    )


def _write_market_parquet(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for offset in range(8):
        for symbol_index, symbol in enumerate(("A", "B", "C")):
            close = 10.0 + symbol_index + offset * (symbol_index + 1) / 10.0
            rows.append(
                {
                    "market_date": date(2024, 1, 1) + timedelta(days=offset),
                    "symbol": symbol,
                    "ldcp": close - 0.1,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "change": 0.1,
                    "change_percent": 1.0,
                    "volume": 100 + offset,
                }
            )
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(rows), schema=MARKET_SCHEMA, preserve_index=False
        ),
        path,
    )


def test_read_only_diagnostics_leave_source_parquet_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    _write_market_parquet(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    market = load_market_data(path)
    partitions = canonical_temporal_date_partitions(market["market_date"])
    training = training_market_view(market, partitions)
    result = construct_close_returns(
        training,
        ["A", "B", "C"],
        global_market_dates=partitions.training_dates,
    )
    matrix = build_return_matrix(
        result.returns,
        value_column="log_return",
        symbols=["A", "B", "C"],
    )
    pairwise_overlap_counts(matrix)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
