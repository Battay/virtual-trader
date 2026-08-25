"""Offline tests for Phase 1 final clustering-protocol selection."""

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
    build_return_matrix,
    canonical_temporal_date_partitions,
    construct_close_returns,
    deterministic_complete_pair_core,
    pairwise_overlap_counts,
    training_market_view,
)
from data_pipeline.src.clustering_protocol import (
    ClusteringProtocolError,
    build_positive_correlation_graph,
    build_unassigned_reason_table,
    cluster_quality_diagnostics,
    deterministic_joint_connected_core,
    deterministic_temporal_windows,
    eligible_symbols_for_overlap_floor,
    graph_geodesic_distance,
    hierarchical_labels,
    mark_pareto_nondominated,
    spectral_graph_labels,
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


def test_overlap_floor_retention_is_explicit_and_deterministic() -> None:
    symbols = list("ABCDE")
    diagnostics = pd.DataFrame(
        {
            "symbol": symbols,
            "training_return_observations": [600, 500, 300, 200, 0],
            "active_span_coverage": [0.9, 0.8, 0.7, 0.4, 0.0],
        }
    )
    overlaps = pd.DataFrame(
        [
            [600, 500, 290, 150, 0],
            [500, 500, 280, 140, 0],
            [290, 280, 300, 130, 0],
            [150, 140, 130, 200, 0],
            [0, 0, 0, 0, 0],
        ],
        index=symbols,
        columns=symbols,
    )

    at_120, peers_120 = eligible_symbols_for_overlap_floor(
        diagnostics,
        overlaps,
        overlap_floor=120,
        minimum_active_span_coverage=0.5,
        minimum_overlapping_peers=2,
    )
    at_252, _ = eligible_symbols_for_overlap_floor(
        diagnostics.sample(frac=1.0, random_state=9),
        overlaps,
        overlap_floor=252,
        minimum_active_span_coverage=0.5,
        minimum_overlapping_peers=2,
    )

    assert at_120 == ("A", "B", "C")
    assert at_252 == ("A", "B", "C")
    assert peers_120.to_dict() == {"A": 3, "B": 3, "C": 3, "D": 3, "E": 0}


def test_complete_core_is_deterministic_when_some_distances_are_missing() -> None:
    symbols = list("ABCDE")
    valid = pd.DataFrame(True, index=symbols, columns=symbols)
    valid.loc["D", "E"] = False
    valid.loc["E", "D"] = False

    first = deterministic_complete_pair_core(valid, candidates=symbols)
    second = deterministic_complete_pair_core(valid, candidates=reversed(symbols))

    assert first == second == ("A", "B", "C", "D")


def test_graph_preserves_missing_edge_semantics_and_refuses_disconnected_fit() -> None:
    symbols = ["A", "B", "C", "D"]
    correlation = pd.DataFrame(
        [
            [1.0, 0.8, np.nan, np.nan],
            [0.8, 1.0, np.nan, np.nan],
            [np.nan, np.nan, 1.0, 0.7],
            [np.nan, np.nan, 0.7, 1.0],
        ],
        index=symbols,
        columns=symbols,
    )
    overlaps = pd.DataFrame(
        [
            [300, 280, 20, 20],
            [280, 300, 20, 20],
            [20, 20, 300, 270],
            [20, 20, 270, 300],
        ],
        index=symbols,
        columns=symbols,
    )

    graph = build_positive_correlation_graph(
        correlation, overlaps, overlap_floor=120
    )

    assert graph.component_count == 2
    assert graph.component_sizes == (2, 2)
    assert graph.edge_count == 2
    assert not graph.edge_mask.loc["A", "C"]
    # Zero is adjacency notation for no edge, not an imputed rho value; the
    # source correlation remains missing and is retained separately.
    assert graph.affinity.loc["A", "C"] == 0.0
    assert np.isnan(correlation.loc["A", "C"])
    with pytest.raises(ClusteringProtocolError, match="disconnected"):
        spectral_graph_labels(graph, cluster_count=2)
    with pytest.raises(ClusteringProtocolError, match="disconnected"):
        graph_geodesic_distance(correlation, graph)


def test_connected_spectral_graph_fit_and_distance_are_deterministic() -> None:
    correlation = _structured_correlation()
    overlaps = pd.DataFrame(
        300,
        index=correlation.index,
        columns=correlation.columns,
        dtype="int64",
    )
    graph = build_positive_correlation_graph(
        correlation, overlaps, overlap_floor=120
    )

    distance = graph_geodesic_distance(correlation, graph)
    first = spectral_graph_labels(graph, cluster_count=3)
    second = spectral_graph_labels(graph, cluster_count=3)

    assert graph.component_count == 1
    assert np.isfinite(distance.to_numpy()).all()
    assert first.tolist() == second.tolist()


def test_joint_connected_core_is_deterministic_across_temporal_graphs() -> None:
    symbols = list("ABCDE")
    early = pd.DataFrame(False, index=symbols, columns=symbols)
    late = pd.DataFrame(False, index=symbols, columns=symbols)
    for left, right in zip(symbols[:-1], symbols[1:], strict=True):
        early.loc[left, right] = early.loc[right, left] = True
    for left, right in (("A", "B"), ("B", "C"), ("C", "D")):
        late.loc[left, right] = late.loc[right, left] = True

    first = deterministic_joint_connected_core(
        early, late, candidates=symbols
    )
    second = deterministic_joint_connected_core(
        early, late, candidates=reversed(symbols)
    )

    assert first == second == ("A", "B", "C", "D")


def test_temporal_windows_are_deterministic_proper_train_subsets() -> None:
    dates = pd.date_range("2020-01-01", periods=10)

    first = deterministic_temporal_windows(dates, fraction=0.7)
    second = deterministic_temporal_windows(reversed(dates), fraction=0.7)

    assert first == second
    assert first.early_dates == tuple(dates[:7])
    assert first.late_dates == tuple(dates[-7:])
    assert first.shared_date_count == 4
    assert set(first.early_dates).union(first.late_dates) == set(dates)


def test_dominant_cluster_and_balance_diagnostics_are_explicit() -> None:
    symbols = list("ABCDEF")
    correlation = pd.DataFrame(
        np.full((len(symbols), len(symbols)), 0.25),
        index=symbols,
        columns=symbols,
        dtype="float64",
    )
    for symbol in symbols:
        correlation.loc[symbol, symbol] = 1.0
    distance = pd.DataFrame(
        np.sqrt(2.0 * (1.0 - correlation)),
        index=symbols,
        columns=symbols,
    )
    for symbol in symbols:
        distance.loc[symbol, symbol] = 0.0
    labels = np.array([0, 0, 0, 0, 0, 1])

    quality = cluster_quality_diagnostics(correlation, distance, labels)

    assert quality["largest_cluster_fraction"] == pytest.approx(5 / 6)
    assert quality["minimum_cluster_size"] == 1
    assert quality["singleton_cluster_count"] == 1
    assert quality["tiny_cluster_count"] == 1


def test_unassigned_reasons_are_prioritized_and_cover_identity_universe() -> None:
    symbols = list("ABCDEFG")
    diagnostics = pd.DataFrame(
        {
            "symbol": symbols,
            "training_return_observations": [300, 300, 0, 100, 300, 300, 300],
            "active_span_coverage": [0.8, 0.8, 0.0, 0.9, 0.4, 0.8, 0.8],
        }
    )
    overlaps = pd.DataFrame(0, index=symbols, columns=symbols, dtype="int64")
    for symbol, value in zip(
        symbols, [300, 300, 0, 100, 300, 300, 300], strict=True
    ):
        overlaps.loc[symbol, symbol] = value
    overlaps.loc["A", "B"] = overlaps.loc["B", "A"] = 280
    overlaps.loc["A", "G"] = overlaps.loc["G", "A"] = 270

    statuses = build_unassigned_reason_table(
        symbols,
        diagnostics,
        overlaps,
        clustered_symbols=("A", "B"),
        overlap_floor=252,
        protocol="hierarchical_complete",
        minimum_overlapping_peers=1,
    ).set_index("symbol")

    assert statuses.loc["A", "reason"] == "clustered"
    assert statuses.loc["C", "reason"] == "no_usable_train_returns"
    assert statuses.loc["D", "reason"] == "insufficient_return_observations"
    assert statuses.loc["E", "reason"] == "insufficient_active_span_coverage"
    assert statuses.loc["F", "reason"] == "insufficient_pairwise_overlap"
    assert statuses.loc["G", "reason"] == "incomplete_pairwise_distance_matrix"
    assert len(statuses) == len(symbols)


def _structured_correlation() -> pd.DataFrame:
    symbols = list("ABCDEFGH")
    loadings = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.0],
            [0.8, 0.3, 0.0],
            [0.1, 1.0, 0.0],
            [0.2, 0.9, 0.0],
            [0.3, 0.8, 0.0],
            [0.0, 0.1, 1.0],
            [0.0, 0.2, 0.9],
        ]
    )
    covariance = loadings @ loadings.T + np.eye(len(symbols)) * 0.2
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    return pd.DataFrame(correlation, index=symbols, columns=symbols)


def test_protocol_labels_and_pareto_evidence_are_deterministic() -> None:
    correlation = _structured_correlation()
    distance_values = np.sqrt(2.0 * (1.0 - correlation.to_numpy()))
    distance = pd.DataFrame(
        distance_values, index=correlation.index, columns=correlation.columns
    )
    first = hierarchical_labels(distance, linkage="complete", cluster_count=3)
    second = hierarchical_labels(distance, linkage="complete", cluster_count=3)
    assert first.tolist() == second.tolist()

    decision = pd.DataFrame(
        {
            "clustered_symbol_count": [8, 8],
            "silhouette": [0.2, 0.1],
            "cohesion_separation_gap": [0.3, 0.2],
            "adjusted_rand_index": [0.5, 0.4],
            "normalized_mutual_information": [0.6, 0.5],
            "minimum_cluster_size": [2, 1],
            "largest_cluster_fraction": [0.5, 0.8],
            "cluster_size_coefficient_of_variation": [0.2, 0.8],
            "singleton_cluster_count": [0, 1],
            "tiny_cluster_count": [0, 1],
        }
    )
    marked = mark_pareto_nondominated(decision)
    assert marked["pareto_nondominated"].tolist() == [True, False]


def test_test_partition_is_not_present_in_protocol_training_view() -> None:
    dates = pd.date_range("2024-01-01", periods=20)
    partitions = canonical_temporal_date_partitions(dates)
    market = pd.DataFrame(
        {
            "market_date": list(dates) * 2,
            "symbol": ["A"] * 20 + ["B"] * 20,
            "close": np.arange(40, dtype=float) + 10.0,
        }
    )

    training = training_market_view(market, partitions)
    returns = construct_close_returns(
        training,
        ["A", "B"],
        global_market_dates=partitions.training_dates,
    )

    assert set(returns.returns["market_date"]).isdisjoint(
        partitions.validation_dates
    )
    assert set(returns.returns["market_date"]).isdisjoint(partitions.test_dates)
    assert returns.returns["market_date"].max() <= pd.Timestamp(
        partitions.training_end
    )


def _write_market_parquet(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for offset in range(12):
        for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
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
    table = pa.Table.from_pandas(
        pd.DataFrame(rows), schema=MARKET_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path)


def test_protocol_diagnostics_leave_source_parquet_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    _write_market_parquet(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    market = load_market_data(path)
    partitions = canonical_temporal_date_partitions(market["market_date"])
    training = training_market_view(market, partitions)
    returns = construct_close_returns(
        training,
        ["A", "B", "C", "D"],
        global_market_dates=partitions.training_dates,
    )
    matrix = build_return_matrix(
        returns.returns,
        value_column="log_return",
        symbols=["A", "B", "C", "D"],
    )
    pairwise_overlap_counts(matrix)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
