"""Final-protocol selection diagnostics for Phase 1 stock clustering.

The module compares clustering protocols but deliberately cannot persist final
cluster assignments.  All fitted correlations and clusters use the canonical
TRAIN partition; validation and TEST remain outside fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .clustering_methodology import (
    CANDIDATE_CLUSTER_COUNTS,
    ClusteringMethodologyError,
    TemporalDatePartitions,
    build_return_matrix,
    build_training_symbol_diagnostics,
    canonical_temporal_date_partitions,
    construct_close_returns,
    correlation_to_distance,
    deterministic_complete_pair_core,
    minimum_overlap_correlation,
    pairwise_overlap_counts,
    training_market_view,
)
from .config import COMPANY_REGISTRY_PATH, CURRENT_LISTINGS_PATH
from .equity_universe import (
    EquityUniverseError,
    EquityUniverseResult,
    run_equity_universe,
)
from .official_listings import ListingsUnavailableError
from .parquet_market_data import (
    MarketParquetError,
    load_market_data,
    resolve_market_parquet_path,
)
from .universe_audit import UniverseAuditError
from .universe_methodology import UniverseMethodologyError


PROTOCOL_SELECTION_VERSION = "phase1_clustering_protocol_selection_v1"
PROTOCOL_SELECTION_DECISION = "blocked_weak_cluster_structure"
CANDIDATE_OVERLAP_FLOORS = (120, 252, 504)
MINIMUM_ACTIVE_SPAN_COVERAGE = 0.50
MINIMUM_OVERLAPPING_PEERS = 20
GRAPH_CORRELATION_THRESHOLD = 0.0
GRAPH_WEAK_DEGREE_THRESHOLD = 5
TEMPORAL_WINDOW_FRACTION = 0.70
SPECTRAL_RANDOM_SEED = 42

# The reference is the strongest 7C.2 evidence-table candidate within the
# supervisor's preferred range.  It remains a robustness reference, not a
# selected or persisted final assignment: its weak silhouettes and sensitivity
# prevent the audit from claiming a final clustering solution.
ROBUSTNESS_REFERENCE_OVERLAP = 120
ROBUSTNESS_REFERENCE_PROTOCOL = "hierarchical_complete"
ROBUSTNESS_REFERENCE_K = 15

PROTOCOL_NAMES = (
    "hierarchical_average",
    "hierarchical_complete",
    "spectral_positive_correlation_graph",
)

RETENTION_COLUMNS = (
    "overlap_floor",
    "eligible_symbol_count",
    "eligible_identity_retention",
    "eligible_train_capable_retention",
    "complete_core_count",
    "complete_core_identity_retention",
    "complete_core_train_capable_retention",
    "graph_component_count",
    "graph_largest_component_count",
    "graph_identity_retention",
    "graph_train_capable_retention",
    "graph_isolated_symbol_count",
    "graph_weak_symbol_count",
    "graph_edge_count",
    "graph_density",
    "available_pair_count",
    "possible_pair_count",
    "available_pair_fraction",
    "overlap_minimum",
    "overlap_p10",
    "overlap_median",
    "overlap_p90",
    "overlap_maximum",
    "fisher_z_standard_error_at_floor",
    "approximate_95_percent_r_width_at_zero",
)

EVIDENCE_COLUMNS = (
    "overlap_floor",
    "protocol",
    "requested_clusters",
    "actual_clusters",
    "clustered_symbol_count",
    "identity_retention",
    "train_capable_retention",
    "silhouette",
    "mean_within_cluster_correlation",
    "mean_between_cluster_correlation",
    "cohesion_separation_gap",
    "minimum_cluster_size",
    "maximum_cluster_size",
    "largest_cluster_fraction",
    "cluster_size_coefficient_of_variation",
    "singleton_cluster_count",
    "tiny_cluster_count",
    "sector_nmi_posthoc",
    "finite_within_pair_count",
    "finite_between_pair_count",
    "status",
)

TEMPORAL_COLUMNS = (
    "overlap_floor",
    "protocol",
    "requested_clusters",
    "temporal_common_symbol_count",
    "temporal_symbol_retention",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "early_largest_cluster_fraction",
    "late_largest_cluster_fraction",
    "largest_cluster_fraction_change",
    "early_minimum_cluster_size",
    "late_minimum_cluster_size",
    "early_singleton_cluster_count",
    "late_singleton_cluster_count",
    "early_tiny_cluster_count",
    "late_tiny_cluster_count",
    "early_cluster_size_coefficient_of_variation",
    "late_cluster_size_coefficient_of_variation",
    "status",
)

ROBUSTNESS_COLUMNS = (
    "return_representation",
    "correlation_method",
    "protocol",
    "overlap_floor",
    "requested_clusters",
    "common_symbol_count",
    "adjusted_rand_index_vs_primary",
    "normalized_mutual_information_vs_primary",
    "status",
)

UNASSIGNED_COLUMNS = (
    "symbol",
    "training_return_observations",
    "active_span_coverage",
    "overlapping_peer_count",
    "clustered_for_reference_protocol",
    "reason",
)


class ClusteringProtocolError(RuntimeError):
    """Raised when a protocol comparison would violate its data contract."""


@dataclass(frozen=True)
class TemporalWindows:
    """Deterministic overlapping TRAIN-only windows used for stability."""

    early_dates: tuple[pd.Timestamp, ...] = field(repr=False)
    late_dates: tuple[pd.Timestamp, ...] = field(repr=False)
    early_start: str
    early_end: str
    late_start: str
    late_end: str
    window_date_count: int
    shared_date_count: int
    fraction: float


@dataclass(frozen=True)
class SimilarityGraph:
    """Explicit positive-correlation graph; absent edges are not correlations."""

    symbols: tuple[str, ...]
    edge_mask: pd.DataFrame = field(repr=False, compare=False)
    affinity: pd.DataFrame = field(repr=False, compare=False)
    component_labels: pd.Series = field(repr=False, compare=False)
    degrees: pd.Series = field(repr=False, compare=False)
    component_sizes: tuple[int, ...]
    largest_component_symbols: tuple[str, ...]
    edge_count: int
    density: float
    isolated_symbol_count: int
    weak_symbol_count: int

    @property
    def component_count(self) -> int:
        return len(self.component_sizes)


@dataclass(frozen=True)
class ProtocolSelectionSummary:
    protocol_selection_version: str
    identity_universe_count: int
    train_return_capable_count: int
    identity_universe_hash: str
    training_date_range: tuple[str, str]
    validation_date_range: tuple[str, str]
    test_date_range: tuple[str, str]
    candidate_overlap_floors: tuple[int, ...]
    active_span_coverage_floor: float
    minimum_overlapping_peers: int
    graph_correlation_threshold: float
    temporal_window_fraction: float
    temporal_window_ranges: tuple[tuple[str, str], tuple[str, str]]
    robustness_reference: Mapping[str, object]
    protocol_selection_decision: str
    final_protocol_selected: bool
    validation_returns_used_for_fitting: bool
    test_returns_used_for_clustering: bool
    final_assignments_written: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_overlap_floors"] = list(
            self.candidate_overlap_floors
        )
        return payload


@dataclass(frozen=True)
class ProtocolSelectionResult:
    summary: ProtocolSelectionSummary
    retention_comparison: pd.DataFrame = field(repr=False, compare=False)
    evidence_table: pd.DataFrame = field(repr=False, compare=False)
    temporal_stability: pd.DataFrame = field(repr=False, compare=False)
    decision_table: pd.DataFrame = field(repr=False, compare=False)
    robustness_comparison: pd.DataFrame = field(repr=False, compare=False)
    reference_unassigned_reasons: pd.DataFrame = field(
        repr=False, compare=False
    )
    parquet_path: Path


def deterministic_temporal_windows(
    training_dates: Sequence[object] | pd.Series | pd.Index,
    *,
    fraction: float = TEMPORAL_WINDOW_FRACTION,
) -> TemporalWindows:
    """Create deterministic early/late overlapping windows from TRAIN only."""

    if not 0.5 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0.5 and 1")
    dates = pd.DatetimeIndex(
        pd.to_datetime(list(training_dates), errors="coerce")
    )
    if dates.isna().any():
        raise ClusteringProtocolError("Temporal window dates contain invalid values")
    dates = dates.unique().sort_values()
    if len(dates) < 4:
        raise ClusteringProtocolError(
            "At least four TRAIN dates are required for temporal windows"
        )
    length = max(2, int(math.ceil(len(dates) * fraction)))
    if length >= len(dates):
        raise ClusteringProtocolError("Temporal windows must be proper TRAIN subsets")
    early = tuple(dates[:length])
    late = tuple(dates[-length:])
    shared = len(set(early).intersection(late))
    return TemporalWindows(
        early_dates=early,
        late_dates=late,
        early_start=early[0].date().isoformat(),
        early_end=early[-1].date().isoformat(),
        late_start=late[0].date().isoformat(),
        late_end=late[-1].date().isoformat(),
        window_date_count=length,
        shared_date_count=shared,
        fraction=float(fraction),
    )


def eligible_symbols_for_overlap_floor(
    symbol_diagnostics: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    *,
    overlap_floor: int,
    minimum_active_span_coverage: float = MINIMUM_ACTIVE_SPAN_COVERAGE,
    minimum_overlapping_peers: int = MINIMUM_OVERLAPPING_PEERS,
) -> tuple[tuple[str, ...], pd.Series]:
    """Apply a transparent TRAIN-only clustering-eligibility prerequisite."""

    if overlap_floor < 2:
        raise ValueError("overlap_floor must be at least 2")
    if not 0 <= minimum_active_span_coverage <= 1:
        raise ValueError("minimum_active_span_coverage must be between 0 and 1")
    if minimum_overlapping_peers < 1:
        raise ValueError("minimum_overlapping_peers must be positive")
    required = {
        "symbol",
        "training_return_observations",
        "active_span_coverage",
    }
    missing = sorted(required.difference(symbol_diagnostics.columns))
    if missing:
        raise ClusteringProtocolError(
            "Symbol diagnostics are missing eligibility fields: "
            + ", ".join(missing)
        )
    diagnostics = symbol_diagnostics.copy(deep=True)
    diagnostics["symbol"] = diagnostics["symbol"].astype("string")
    if diagnostics["symbol"].duplicated().any():
        raise ClusteringProtocolError("Symbol diagnostics contain duplicates")
    diagnostics = diagnostics.set_index("symbol").sort_index()
    if list(overlap_counts.index) != list(overlap_counts.columns):
        raise ClusteringProtocolError("Overlap matrix ordering is inconsistent")
    qualifies = overlap_counts.ge(overlap_floor)
    diagonal_qualifies = pd.Series(
        np.diag(qualifies.to_numpy(dtype=bool)),
        index=qualifies.index,
        dtype="int64",
    )
    peers = (
        qualifies.sum(axis=1).astype("int64") - diagonal_qualifies
    ).astype("int64")
    peers = peers.reindex(diagnostics.index).fillna(0).astype("int64")
    eligible = diagnostics.index[
        diagnostics["training_return_observations"].ge(overlap_floor)
        & diagnostics["active_span_coverage"].ge(
            minimum_active_span_coverage
        )
        & peers.ge(minimum_overlapping_peers)
    ]
    return tuple(sorted(map(str, eligible))), peers


def build_positive_correlation_graph(
    correlation: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    *,
    overlap_floor: int,
    correlation_threshold: float = GRAPH_CORRELATION_THRESHOLD,
    weak_degree_threshold: int = GRAPH_WEAK_DEGREE_THRESHOLD,
) -> SimilarityGraph:
    """Build an explicit graph without treating absent correlations as zero.

    An edge exists only when its pair has the required overlap and its finite
    Pearson/Spearman correlation is strictly above the declared similarity
    threshold.  A zero in the returned affinity matrix means *no graph edge*;
    ``edge_mask`` preserves that semantic distinction from rho == 0.
    """

    if overlap_floor < 2:
        raise ValueError("overlap_floor must be at least 2")
    if weak_degree_threshold < 1:
        raise ValueError("weak_degree_threshold must be positive")
    if correlation.shape[0] != correlation.shape[1]:
        raise ClusteringProtocolError("Correlation matrix must be square")
    symbols = tuple(map(str, correlation.index))
    if symbols != tuple(map(str, correlation.columns)):
        raise ClusteringProtocolError("Correlation matrix ordering is inconsistent")
    overlap = overlap_counts.reindex(
        index=correlation.index, columns=correlation.columns
    )
    if overlap.isna().any().any():
        raise ClusteringProtocolError("Overlap matrix is not aligned to correlations")
    values = correlation.to_numpy(dtype="float64", copy=True)
    overlap_values = overlap.to_numpy(dtype="int64", copy=True)
    edge_values = (
        np.isfinite(values)
        & (overlap_values >= overlap_floor)
        & (values > correlation_threshold)
    )
    edge_values &= edge_values.T
    np.fill_diagonal(edge_values, False)
    affinity_values = np.zeros_like(values, dtype="float64")
    symmetric_correlation = np.clip((values + values.T) / 2.0, -1.0, 1.0)
    affinity_values[edge_values] = symmetric_correlation[edge_values]
    np.fill_diagonal(affinity_values, 1.0)
    edge_mask = pd.DataFrame(
        edge_values, index=correlation.index.copy(), columns=correlation.columns.copy()
    )
    affinity = pd.DataFrame(
        affinity_values,
        index=correlation.index.copy(),
        columns=correlation.columns.copy(),
    )
    component_count, labels = connected_components(
        csr_matrix(edge_values.astype("int8")),
        directed=False,
        return_labels=True,
    )
    label_series = pd.Series(labels, index=correlation.index, dtype="int64")
    components: list[tuple[str, ...]] = []
    for label in range(component_count):
        components.append(
            tuple(sorted(map(str, label_series.index[label_series == label])))
        )
    components.sort(key=lambda item: (-len(item), item))
    sizes = tuple(len(component) for component in components)
    largest = components[0] if components else ()
    degrees = pd.Series(
        edge_values.sum(axis=1).astype("int64"),
        index=correlation.index.copy(),
        dtype="int64",
    )
    possible_edges = len(symbols) * (len(symbols) - 1) // 2
    edge_count = int(edge_values[np.triu_indices(len(symbols), k=1)].sum())
    return SimilarityGraph(
        symbols=symbols,
        edge_mask=edge_mask,
        affinity=affinity,
        component_labels=label_series,
        degrees=degrees,
        component_sizes=sizes,
        largest_component_symbols=largest,
        edge_count=edge_count,
        density=edge_count / possible_edges if possible_edges else 0.0,
        isolated_symbol_count=int((degrees == 0).sum()),
        weak_symbol_count=int((degrees < weak_degree_threshold).sum()),
    )


def graph_geodesic_distance(
    correlation: pd.DataFrame,
    graph: SimilarityGraph,
) -> pd.DataFrame:
    """Derive shortest-path angular distance for one connected graph."""

    if graph.component_count != 1:
        raise ClusteringProtocolError(
            "Graph-distance clustering refuses disconnected components"
        )
    correlation = correlation.reindex(
        index=graph.symbols, columns=graph.symbols
    )
    values = correlation.to_numpy(dtype="float64", copy=True)
    edges = graph.edge_mask.to_numpy(dtype=bool, copy=True)
    weights = np.zeros_like(values, dtype="float64")
    angular = np.sqrt(
        np.maximum(0.0, 2.0 * (1.0 - np.clip(values, -1.0, 1.0)))
    )
    weights[edges] = np.maximum(angular[edges], np.finfo("float64").eps)
    distances = shortest_path(
        csr_matrix(weights), directed=False, unweighted=False
    )
    if not np.isfinite(distances).all():
        raise ClusteringProtocolError("Graph geodesic distance is disconnected")
    np.fill_diagonal(distances, 0.0)
    return pd.DataFrame(
        distances,
        index=correlation.index.copy(),
        columns=correlation.columns.copy(),
    )


def hierarchical_labels(
    distance: pd.DataFrame,
    *,
    linkage: str,
    cluster_count: int,
) -> np.ndarray:
    """Fit deterministic agglomerative labels to a complete distance matrix."""

    if linkage not in {"average", "complete"}:
        raise ValueError("linkage must be average or complete")
    if cluster_count < 2 or cluster_count >= len(distance):
        raise ValueError("cluster_count must be between 2 and symbol_count - 1")
    model = AgglomerativeClustering(
        n_clusters=cluster_count,
        metric="precomputed",
        linkage=linkage,
    )
    return model.fit_predict(distance.to_numpy(dtype="float64"))


def spectral_graph_labels(
    graph: SimilarityGraph,
    *,
    cluster_count: int,
    seed: int = SPECTRAL_RANDOM_SEED,
) -> np.ndarray:
    """Fit deterministic spectral labels without forcing disconnected nodes."""

    if graph.component_count != 1:
        raise ClusteringProtocolError(
            "Spectral clustering refuses a disconnected similarity graph"
        )
    if cluster_count < 2 or cluster_count >= len(graph.symbols):
        raise ValueError("cluster_count must be between 2 and symbol_count - 1")
    model = SpectralClustering(
        n_clusters=cluster_count,
        affinity="precomputed",
        assign_labels="cluster_qr",
        eigen_solver="arpack",
        random_state=seed,
    )
    return model.fit_predict(graph.affinity.to_numpy(dtype="float64"))


def cluster_quality_diagnostics(
    correlation: pd.DataFrame,
    distance: pd.DataFrame,
    labels: Sequence[int] | np.ndarray,
    *,
    sector_by_symbol: Mapping[str, str] | None = None,
) -> dict[str, float | int]:
    """Compute balance, cohesion, separation, and post-hoc sector evidence."""

    labels_array = np.asarray(labels, dtype="int64")
    if len(labels_array) != len(correlation):
        raise ClusteringProtocolError("Labels do not align with correlations")
    if list(correlation.index) != list(distance.index):
        raise ClusteringProtocolError("Correlation/distance symbols differ")
    unique_labels = np.unique(labels_array)
    if len(unique_labels) < 2:
        raise ClusteringProtocolError("At least two clusters are required")
    values = correlation.to_numpy(dtype="float64")
    upper = np.triu_indices_from(values, k=1)
    finite = np.isfinite(values[upper])
    same = labels_array[upper[0]] == labels_array[upper[1]]
    within = values[upper][finite & same]
    between = values[upper][finite & ~same]
    sizes = pd.Series(labels_array).value_counts().sort_index().to_numpy()
    within_mean = float(within.mean()) if len(within) else float("nan")
    between_mean = float(between.mean()) if len(between) else float("nan")
    sector_nmi = float("nan")
    if sector_by_symbol is not None:
        sectors = [
            sector_by_symbol.get(str(symbol), "UNKNOWN")
            for symbol in correlation.index
        ]
        sector_nmi = float(
            normalized_mutual_info_score(sectors, labels_array)
        )
    return {
        "actual_clusters": int(len(unique_labels)),
        "silhouette": float(
            silhouette_score(
                distance.to_numpy(dtype="float64"),
                labels_array,
                metric="precomputed",
            )
        ),
        "mean_within_cluster_correlation": within_mean,
        "mean_between_cluster_correlation": between_mean,
        "cohesion_separation_gap": within_mean - between_mean,
        "minimum_cluster_size": int(sizes.min()),
        "maximum_cluster_size": int(sizes.max()),
        "largest_cluster_fraction": float(sizes.max() / sizes.sum()),
        "cluster_size_coefficient_of_variation": float(
            sizes.std(ddof=0) / sizes.mean()
        ),
        "singleton_cluster_count": int((sizes == 1).sum()),
        "tiny_cluster_count": int((sizes < 3).sum()),
        "sector_nmi_posthoc": sector_nmi,
        "finite_within_pair_count": int(len(within)),
        "finite_between_pair_count": int(len(between)),
    }


def _cluster_size_snapshot(labels: np.ndarray) -> dict[str, float | int]:
    sizes = pd.Series(labels).value_counts().sort_index().to_numpy()
    return {
        "largest_cluster_fraction": float(sizes.max() / sizes.sum()),
        "minimum_cluster_size": int(sizes.min()),
        "singleton_cluster_count": int((sizes == 1).sum()),
        "tiny_cluster_count": int((sizes < 3).sum()),
        "cluster_size_coefficient_of_variation": float(
            sizes.std(ddof=0) / sizes.mean()
        ),
    }


def _complete_core(
    correlation: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[str, ...]:
    valid = correlation.notna()
    return deterministic_complete_pair_core(valid, candidates=candidates)


def _subset_graph(
    correlation: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    symbols: Sequence[str],
    *,
    overlap_floor: int,
) -> SimilarityGraph:
    ordered = list(sorted(map(str, symbols)))
    return build_positive_correlation_graph(
        correlation.loc[ordered, ordered],
        overlap_counts.loc[ordered, ordered],
        overlap_floor=overlap_floor,
    )


def deterministic_joint_connected_core(
    early_edges: pd.DataFrame,
    late_edges: pd.DataFrame,
    *,
    candidates: Sequence[str],
) -> tuple[str, ...]:
    """Find a deterministic symbol subset connected in both temporal graphs."""

    if list(early_edges.index) != list(early_edges.columns):
        raise ClusteringProtocolError("Early graph edge ordering is inconsistent")
    if list(late_edges.index) != list(late_edges.columns):
        raise ClusteringProtocolError("Late graph edge ordering is inconsistent")
    if set(early_edges.index) != set(late_edges.index):
        raise ClusteringProtocolError("Temporal graph symbols differ")
    remaining = set(map(str, candidates))
    unknown = remaining.difference(map(str, early_edges.index))
    if unknown:
        raise ClusteringProtocolError(
            "Temporal graph candidates are missing: " + ", ".join(sorted(unknown))
        )
    while len(remaining) > 1:
        ordered = sorted(remaining)
        early = early_edges.loc[ordered, ordered].to_numpy(
            dtype=bool, copy=True
        )
        late = late_edges.loc[ordered, ordered].to_numpy(dtype=bool, copy=True)
        early_count = connected_components(
            csr_matrix(early.astype("int8")), directed=False, return_labels=False
        )
        late_count = connected_components(
            csr_matrix(late.astype("int8")), directed=False, return_labels=False
        )
        if early_count == 1 and late_count == 1:
            break
        combined_degree = early.sum(axis=1) + late.sum(axis=1)
        minimum = int(combined_degree.min())
        tied = [
            symbol
            for symbol, degree in zip(ordered, combined_degree, strict=True)
            if int(degree) == minimum
        ]
        remaining.remove(sorted(tied)[-1])
    return tuple(sorted(remaining))


def _pair_reliability(
    overlap_counts: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    overlap_floor: int,
) -> dict[str, float | int]:
    overlap_values = overlap_counts.to_numpy(dtype="int64")
    correlation_values = correlation.to_numpy(dtype="float64")
    upper = np.triu_indices_from(overlap_values, k=1)
    available = (
        (overlap_values[upper] >= overlap_floor)
        & np.isfinite(correlation_values[upper])
    )
    observed = overlap_values[upper][available]
    possible = len(upper[0])
    if len(observed):
        quantiles = np.quantile(observed, [0.0, 0.10, 0.50, 0.90, 1.0])
    else:
        quantiles = [np.nan] * 5
    fisher_se = 1.0 / math.sqrt(overlap_floor - 3)
    return {
        "available_pair_count": int(available.sum()),
        "possible_pair_count": int(possible),
        "available_pair_fraction": (
            float(available.sum() / possible) if possible else 0.0
        ),
        "overlap_minimum": float(quantiles[0]),
        "overlap_p10": float(quantiles[1]),
        "overlap_median": float(quantiles[2]),
        "overlap_p90": float(quantiles[3]),
        "overlap_maximum": float(quantiles[4]),
        "fisher_z_standard_error_at_floor": float(fisher_se),
        "approximate_95_percent_r_width_at_zero": float(1.96 * fisher_se),
    }


def _evaluate_protocol(
    correlation: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    clustered_symbols: Sequence[str],
    *,
    overlap_floor: int,
    protocol: str,
    cluster_counts: Sequence[int],
    identity_count: int,
    train_capable_count: int,
    sector_by_symbol: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    symbols = list(sorted(map(str, clustered_symbols)))
    selected_correlation = correlation.loc[symbols, symbols]
    labels_by_k: dict[int, np.ndarray] = {}
    if protocol.startswith("hierarchical_"):
        distance = correlation_to_distance(selected_correlation)
        linkage = protocol.removeprefix("hierarchical_")
        label_builder = lambda count: hierarchical_labels(  # noqa: E731
            distance, linkage=linkage, cluster_count=count
        )
    elif protocol == "spectral_positive_correlation_graph":
        graph = _subset_graph(
            correlation,
            overlap_counts,
            symbols,
            overlap_floor=overlap_floor,
        )
        if graph.component_count != 1:
            raise ClusteringProtocolError(
                "Spectral protocol received a disconnected graph"
            )
        distance = graph_geodesic_distance(selected_correlation, graph)
        label_builder = lambda count: spectral_graph_labels(  # noqa: E731
            graph, cluster_count=count
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    rows: list[dict[str, object]] = []
    for cluster_count in cluster_counts:
        if cluster_count < 2 or cluster_count >= len(symbols):
            rows.append(
                {
                    "overlap_floor": overlap_floor,
                    "protocol": protocol,
                    "requested_clusters": int(cluster_count),
                    "actual_clusters": 0,
                    "clustered_symbol_count": len(symbols),
                    "identity_retention": len(symbols) / identity_count,
                    "train_capable_retention": len(symbols) / train_capable_count,
                    **{
                        column: np.nan
                        for column in EVIDENCE_COLUMNS[7:-1]
                    },
                    "status": "insufficient_symbols",
                }
            )
            continue
        labels = label_builder(int(cluster_count))
        labels_by_k[int(cluster_count)] = labels
        quality = cluster_quality_diagnostics(
            selected_correlation,
            distance,
            labels,
            sector_by_symbol=sector_by_symbol,
        )
        rows.append(
            {
                "overlap_floor": overlap_floor,
                "protocol": protocol,
                "requested_clusters": int(cluster_count),
                "clustered_symbol_count": len(symbols),
                "identity_retention": len(symbols) / identity_count,
                "train_capable_retention": len(symbols) / train_capable_count,
                **quality,
                "status": "completed_non_final_protocol_diagnostic",
            }
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS), labels_by_k


def _window_return_matrix(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
    *,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_set = set(dates)
    subset = training_market.loc[
        training_market["market_date"].isin(date_set)
    ].copy(deep=True)
    built = construct_close_returns(
        subset,
        symbols,
        global_market_dates=dates,
    )
    matrix = build_return_matrix(
        built.returns, value_column=value_column, symbols=symbols
    )
    return subset, matrix


def _temporal_protocol_rows(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    windows: TemporalWindows,
    *,
    overlap_floor: int,
    cluster_counts: Sequence[int],
    full_protocol_sizes: Mapping[str, int],
) -> pd.DataFrame:
    early_market, early_matrix = _window_return_matrix(
        training_market,
        symbols,
        windows.early_dates,
        value_column="log_return",
    )
    late_market, late_matrix = _window_return_matrix(
        training_market,
        symbols,
        windows.late_dates,
        value_column="log_return",
    )
    early_overlap = pairwise_overlap_counts(early_matrix)
    late_overlap = pairwise_overlap_counts(late_matrix)
    early_diagnostics = build_training_symbol_diagnostics(
        early_market, symbols, early_matrix
    )
    late_diagnostics = build_training_symbol_diagnostics(
        late_market, symbols, late_matrix
    )
    early_eligible, _ = eligible_symbols_for_overlap_floor(
        early_diagnostics,
        early_overlap,
        overlap_floor=overlap_floor,
    )
    late_eligible, _ = eligible_symbols_for_overlap_floor(
        late_diagnostics,
        late_overlap,
        overlap_floor=overlap_floor,
    )
    common_eligible = tuple(
        sorted(set(early_eligible).intersection(late_eligible))
    )
    early_correlation = minimum_overlap_correlation(
        early_matrix,
        method="pearson",
        minimum_overlap=overlap_floor,
        overlap_counts=early_overlap,
    )
    late_correlation = minimum_overlap_correlation(
        late_matrix,
        method="pearson",
        minimum_overlap=overlap_floor,
        overlap_counts=late_overlap,
    )
    joint_valid = early_correlation.notna() & late_correlation.notna()
    complete_core = deterministic_complete_pair_core(
        joint_valid, candidates=common_eligible
    )

    early_candidate_graph = _subset_graph(
        early_correlation,
        early_overlap,
        common_eligible,
        overlap_floor=overlap_floor,
    )
    late_candidate_graph = _subset_graph(
        late_correlation,
        late_overlap,
        common_eligible,
        overlap_floor=overlap_floor,
    )
    graph_core = deterministic_joint_connected_core(
        early_candidate_graph.edge_mask,
        late_candidate_graph.edge_mask,
        candidates=common_eligible,
    )
    rows: list[dict[str, object]] = []
    for protocol in PROTOCOL_NAMES:
        protocol_symbols = (
            graph_core
            if protocol == "spectral_positive_correlation_graph"
            else complete_core
        )
        ordered = list(protocol_symbols)
        if protocol == "spectral_positive_correlation_graph":
            early_graph = _subset_graph(
                early_correlation,
                early_overlap,
                ordered,
                overlap_floor=overlap_floor,
            )
            late_graph = _subset_graph(
                late_correlation,
                late_overlap,
                ordered,
                overlap_floor=overlap_floor,
            )
            early_builder = lambda count: spectral_graph_labels(  # noqa: E731
                early_graph, cluster_count=count
            )
            late_builder = lambda count: spectral_graph_labels(  # noqa: E731
                late_graph, cluster_count=count
            )
        else:
            linkage = protocol.removeprefix("hierarchical_")
            early_distance = correlation_to_distance(
                early_correlation.loc[ordered, ordered]
            )
            late_distance = correlation_to_distance(
                late_correlation.loc[ordered, ordered]
            )
            early_builder = lambda count: hierarchical_labels(  # noqa: E731
                early_distance, linkage=linkage, cluster_count=count
            )
            late_builder = lambda count: hierarchical_labels(  # noqa: E731
                late_distance, linkage=linkage, cluster_count=count
            )
        for cluster_count in cluster_counts:
            if cluster_count < 2 or cluster_count >= len(ordered):
                rows.append(
                    {
                        "overlap_floor": overlap_floor,
                        "protocol": protocol,
                        "requested_clusters": int(cluster_count),
                        "temporal_common_symbol_count": len(ordered),
                        "temporal_symbol_retention": 0.0,
                        **{
                            column: np.nan
                            for column in TEMPORAL_COLUMNS[5:-1]
                        },
                        "status": "insufficient_temporal_symbols",
                    }
                )
                continue
            early_labels = early_builder(int(cluster_count))
            late_labels = late_builder(int(cluster_count))
            early_snapshot = _cluster_size_snapshot(early_labels)
            late_snapshot = _cluster_size_snapshot(late_labels)
            full_size = full_protocol_sizes[protocol]
            rows.append(
                {
                    "overlap_floor": overlap_floor,
                    "protocol": protocol,
                    "requested_clusters": int(cluster_count),
                    "temporal_common_symbol_count": len(ordered),
                    "temporal_symbol_retention": (
                        len(ordered) / full_size if full_size else 0.0
                    ),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(early_labels, late_labels)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(
                            early_labels, late_labels
                        )
                    ),
                    "early_largest_cluster_fraction": early_snapshot[
                        "largest_cluster_fraction"
                    ],
                    "late_largest_cluster_fraction": late_snapshot[
                        "largest_cluster_fraction"
                    ],
                    "largest_cluster_fraction_change": abs(
                        early_snapshot["largest_cluster_fraction"]
                        - late_snapshot["largest_cluster_fraction"]
                    ),
                    "early_minimum_cluster_size": early_snapshot[
                        "minimum_cluster_size"
                    ],
                    "late_minimum_cluster_size": late_snapshot[
                        "minimum_cluster_size"
                    ],
                    "early_singleton_cluster_count": early_snapshot[
                        "singleton_cluster_count"
                    ],
                    "late_singleton_cluster_count": late_snapshot[
                        "singleton_cluster_count"
                    ],
                    "early_tiny_cluster_count": early_snapshot[
                        "tiny_cluster_count"
                    ],
                    "late_tiny_cluster_count": late_snapshot[
                        "tiny_cluster_count"
                    ],
                    "early_cluster_size_coefficient_of_variation": early_snapshot[
                        "cluster_size_coefficient_of_variation"
                    ],
                    "late_cluster_size_coefficient_of_variation": late_snapshot[
                        "cluster_size_coefficient_of_variation"
                    ],
                    "status": "completed_train_window_stability",
                }
            )
    return pd.DataFrame(rows, columns=TEMPORAL_COLUMNS)


def mark_pareto_nondominated(decision_table: pd.DataFrame) -> pd.DataFrame:
    """Mark evidence-table rows without collapsing evidence into one score."""

    maximize = (
        "clustered_symbol_count",
        "silhouette",
        "cohesion_separation_gap",
        "adjusted_rand_index",
        "normalized_mutual_information",
        "minimum_cluster_size",
    )
    minimize = (
        "largest_cluster_fraction",
        "cluster_size_coefficient_of_variation",
        "singleton_cluster_count",
        "tiny_cluster_count",
    )
    required = set(maximize).union(minimize)
    missing = sorted(required.difference(decision_table.columns))
    if missing:
        raise ClusteringProtocolError(
            "Decision table is missing Pareto fields: " + ", ".join(missing)
        )
    result = decision_table.copy(deep=True).reset_index(drop=True)
    values = result.loc[:, [*maximize, *minimize]].apply(
        pd.to_numeric, errors="coerce"
    )
    normalized = values.copy(deep=True)
    for column in maximize:
        normalized[column] = normalized[column].fillna(-np.inf)
    for column in minimize:
        normalized[column] = -normalized[column].fillna(np.inf)
    array = normalized.to_numpy(dtype="float64")
    nondominated = np.ones(len(result), dtype=bool)
    for index in range(len(result)):
        other_at_least = np.all(array >= array[index], axis=1)
        other_strict = np.any(array > array[index], axis=1)
        other_at_least[index] = False
        if bool(np.any(other_at_least & other_strict)):
            nondominated[index] = False
    result["pareto_nondominated"] = nondominated
    return result


def build_unassigned_reason_table(
    identity_symbols: Sequence[str],
    symbol_diagnostics: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    clustered_symbols: Sequence[str],
    *,
    overlap_floor: int,
    protocol: str,
    minimum_active_span_coverage: float = MINIMUM_ACTIVE_SPAN_COVERAGE,
    minimum_overlapping_peers: int = MINIMUM_OVERLAPPING_PEERS,
) -> pd.DataFrame:
    """Record one deterministic TRAIN-only status for every identity symbol."""

    diagnostics = symbol_diagnostics.set_index("symbol").copy(deep=True)
    qualifies = overlap_counts.ge(overlap_floor)
    diagonal_qualifies = pd.Series(
        np.diag(qualifies.to_numpy(dtype=bool)),
        index=qualifies.index,
        dtype="int64",
    )
    peers = (
        qualifies.sum(axis=1).astype("int64") - diagonal_qualifies
    ).astype("int64")
    clustered = set(map(str, clustered_symbols))
    rows: list[dict[str, object]] = []
    for symbol in sorted(set(map(str, identity_symbols))):
        if symbol not in diagnostics.index:
            raise ClusteringProtocolError(
                f"Identity symbol {symbol} is absent from diagnostics"
            )
        observations = int(
            diagnostics.loc[symbol, "training_return_observations"]
        )
        coverage = float(diagnostics.loc[symbol, "active_span_coverage"])
        peer_count = int(peers.get(symbol, 0))
        if symbol in clustered:
            reason = "clustered"
        elif observations == 0:
            reason = "no_usable_train_returns"
        elif observations < overlap_floor:
            reason = "insufficient_return_observations"
        elif coverage < minimum_active_span_coverage:
            reason = "insufficient_active_span_coverage"
        elif peer_count < minimum_overlapping_peers:
            reason = "insufficient_pairwise_overlap"
        elif protocol.startswith("hierarchical_"):
            reason = "incomplete_pairwise_distance_matrix"
        elif protocol == "spectral_positive_correlation_graph":
            reason = "disconnected_similarity_graph"
        else:
            reason = "protocol_exclusion"
        rows.append(
            {
                "symbol": symbol,
                "training_return_observations": observations,
                "active_span_coverage": coverage,
                "overlapping_peer_count": peer_count,
                "clustered_for_reference_protocol": symbol in clustered,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=UNASSIGNED_COLUMNS)


def _variant_labels(
    correlation: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    symbols: Sequence[str],
    *,
    overlap_floor: int,
    protocol: str,
    cluster_count: int,
) -> np.ndarray:
    ordered = list(map(str, symbols))
    selected = correlation.loc[ordered, ordered]
    if protocol.startswith("hierarchical_"):
        if selected.isna().any().any():
            raise ClusteringProtocolError(
                "Robustness correlation is incomplete on the reference core"
            )
        return hierarchical_labels(
            correlation_to_distance(selected),
            linkage=protocol.removeprefix("hierarchical_"),
            cluster_count=cluster_count,
        )
    graph = _subset_graph(
        correlation,
        overlap_counts,
        ordered,
        overlap_floor=overlap_floor,
    )
    return spectral_graph_labels(graph, cluster_count=cluster_count)


def evaluate_return_correlation_robustness(
    log_return_matrix: pd.DataFrame,
    simple_return_matrix: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    symbols: Sequence[str],
    *,
    overlap_floor: int,
    protocol: str,
    cluster_count: int,
) -> pd.DataFrame:
    """Compare primary labels with simple-return and Spearman alternatives."""

    variants = (
        ("log_return", "pearson", log_return_matrix),
        ("simple_return", "pearson", simple_return_matrix),
        ("log_return", "spearman", log_return_matrix),
    )
    rows: list[dict[str, object]] = []
    primary_labels: np.ndarray | None = None
    for return_type, method, matrix in variants:
        correlation = minimum_overlap_correlation(
            matrix,
            method=method,
            minimum_overlap=overlap_floor,
            overlap_counts=overlap_counts,
        )
        try:
            labels = _variant_labels(
                correlation,
                overlap_counts,
                symbols,
                overlap_floor=overlap_floor,
                protocol=protocol,
                cluster_count=cluster_count,
            )
            if primary_labels is None:
                primary_labels = labels
            ari = float(adjusted_rand_score(primary_labels, labels))
            nmi = float(normalized_mutual_info_score(primary_labels, labels))
            status = (
                "primary"
                if return_type == "log_return" and method == "pearson"
                else "completed"
            )
        except (ClusteringProtocolError, ValueError) as exc:
            ari = np.nan
            nmi = np.nan
            status = f"unavailable: {exc}"
        rows.append(
            {
                "return_representation": return_type,
                "correlation_method": method,
                "protocol": protocol,
                "overlap_floor": overlap_floor,
                "requested_clusters": cluster_count,
                "common_symbol_count": len(symbols),
                "adjusted_rand_index_vs_primary": ari,
                "normalized_mutual_information_vs_primary": nmi,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)


def _run_floor_comparison(
    log_matrix: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    symbol_diagnostics: pd.DataFrame,
    *,
    overlap_floor: int,
    identity_count: int,
    train_capable_count: int,
    cluster_counts: Sequence[int],
    sector_by_symbol: Mapping[str, str],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    dict[str, tuple[str, ...]],
]:
    eligible, _ = eligible_symbols_for_overlap_floor(
        symbol_diagnostics,
        overlap_counts,
        overlap_floor=overlap_floor,
    )
    eligible_list = list(eligible)
    eligible_overlap = overlap_counts.loc[eligible_list, eligible_list]
    correlation = minimum_overlap_correlation(
        log_matrix.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=overlap_floor,
        overlap_counts=eligible_overlap,
    )
    complete_core = _complete_core(correlation, eligible)
    graph = build_positive_correlation_graph(
        correlation,
        eligible_overlap,
        overlap_floor=overlap_floor,
    )
    graph_symbols = graph.largest_component_symbols
    reliability = _pair_reliability(
        eligible_overlap,
        correlation,
        overlap_floor=overlap_floor,
    )
    retention = {
        "overlap_floor": overlap_floor,
        "eligible_symbol_count": len(eligible),
        "eligible_identity_retention": len(eligible) / identity_count,
        "eligible_train_capable_retention": len(eligible) / train_capable_count,
        "complete_core_count": len(complete_core),
        "complete_core_identity_retention": len(complete_core) / identity_count,
        "complete_core_train_capable_retention": len(complete_core)
        / train_capable_count,
        "graph_component_count": graph.component_count,
        "graph_largest_component_count": len(graph_symbols),
        "graph_identity_retention": len(graph_symbols) / identity_count,
        "graph_train_capable_retention": len(graph_symbols) / train_capable_count,
        "graph_isolated_symbol_count": graph.isolated_symbol_count,
        "graph_weak_symbol_count": graph.weak_symbol_count,
        "graph_edge_count": graph.edge_count,
        "graph_density": graph.density,
        **reliability,
    }
    protocol_symbols = {
        "hierarchical_average": complete_core,
        "hierarchical_complete": complete_core,
        "spectral_positive_correlation_graph": graph_symbols,
    }
    evidence_frames = []
    for protocol, clustered in protocol_symbols.items():
        evidence, _ = _evaluate_protocol(
            correlation,
            eligible_overlap,
            clustered,
            overlap_floor=overlap_floor,
            protocol=protocol,
            cluster_counts=cluster_counts,
            identity_count=identity_count,
            train_capable_count=train_capable_count,
            sector_by_symbol=sector_by_symbol,
        )
        evidence_frames.append(evidence)
    return (
        retention,
        pd.concat(evidence_frames, ignore_index=True),
        protocol_symbols,
    )


def run_clustering_protocol_selection_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
    overlap_floors: Sequence[int] = CANDIDATE_OVERLAP_FLOORS,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
    temporal_window_fraction: float = TEMPORAL_WINDOW_FRACTION,
    robustness_overlap_floor: int = ROBUSTNESS_REFERENCE_OVERLAP,
    robustness_protocol: str = ROBUSTNESS_REFERENCE_PROTOCOL,
    robustness_cluster_count: int = ROBUSTNESS_REFERENCE_K,
) -> ProtocolSelectionResult:
    """Run the read-only, TRAIN-fitted final-protocol comparison."""

    floors = tuple(int(value) for value in overlap_floors)
    if not floors or any(value < 2 for value in floors):
        raise ValueError("overlap_floors must contain values of at least 2")
    if robustness_overlap_floor not in floors:
        raise ValueError("robustness_overlap_floor must be one of overlap_floors")
    if robustness_protocol not in PROTOCOL_NAMES:
        raise ValueError("robustness_protocol is unsupported")
    equity: EquityUniverseResult = run_equity_universe(
        parquet_path=parquet_path,
        registry_path=registry_path,
        listing_snapshot_path=listing_snapshot_path,
    )
    resolved_parquet = resolve_market_parquet_path(parquet_path)
    market = load_market_data(resolved_parquet)
    partitions: TemporalDatePartitions = canonical_temporal_date_partitions(
        market["market_date"]
    )
    training_market = training_market_view(market, partitions)
    symbols = tuple(equity.records["symbol"].astype(str))
    returns = construct_close_returns(
        training_market,
        symbols,
        global_market_dates=partitions.training_dates,
    )
    log_matrix = build_return_matrix(
        returns.returns, value_column="log_return", symbols=symbols
    )
    simple_matrix = build_return_matrix(
        returns.returns, value_column="simple_return", symbols=symbols
    )
    overlap_counts = pairwise_overlap_counts(log_matrix)
    symbol_diagnostics = build_training_symbol_diagnostics(
        training_market, symbols, log_matrix
    )
    train_capable_count = int(
        symbol_diagnostics["training_return_observations"].gt(0).sum()
    )
    sector_map = equity.records.set_index("symbol")["sector"].astype(str).to_dict()
    windows = deterministic_temporal_windows(
        partitions.training_dates, fraction=temporal_window_fraction
    )

    retention_rows: list[dict[str, object]] = []
    evidence_frames: list[pd.DataFrame] = []
    temporal_frames: list[pd.DataFrame] = []
    symbols_by_floor_protocol: dict[tuple[int, str], tuple[str, ...]] = {}
    for floor in floors:
        retention, evidence, protocol_symbols = _run_floor_comparison(
            log_matrix,
            overlap_counts,
            symbol_diagnostics,
            overlap_floor=floor,
            identity_count=len(symbols),
            train_capable_count=train_capable_count,
            cluster_counts=cluster_counts,
            sector_by_symbol=sector_map,
        )
        retention_rows.append(retention)
        evidence_frames.append(evidence)
        for protocol, selected_symbols in protocol_symbols.items():
            symbols_by_floor_protocol[(floor, protocol)] = selected_symbols
        temporal_frames.append(
            _temporal_protocol_rows(
                training_market,
                symbols,
                windows,
                overlap_floor=floor,
                cluster_counts=cluster_counts,
                full_protocol_sizes={
                    protocol: len(selected_symbols)
                    for protocol, selected_symbols in protocol_symbols.items()
                },
            )
        )
    retention_frame = pd.DataFrame(retention_rows, columns=RETENTION_COLUMNS)
    evidence_frame = pd.concat(evidence_frames, ignore_index=True)
    temporal_frame = pd.concat(temporal_frames, ignore_index=True)
    decision = evidence_frame.merge(
        temporal_frame,
        on=["overlap_floor", "protocol", "requested_clusters"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_temporal"),
    )
    decision = mark_pareto_nondominated(decision)

    reference_symbols = symbols_by_floor_protocol[
        (robustness_overlap_floor, robustness_protocol)
    ]
    robustness = evaluate_return_correlation_robustness(
        log_matrix,
        simple_matrix,
        overlap_counts,
        reference_symbols,
        overlap_floor=robustness_overlap_floor,
        protocol=robustness_protocol,
        cluster_count=robustness_cluster_count,
    )
    unassigned = build_unassigned_reason_table(
        symbols,
        symbol_diagnostics,
        overlap_counts,
        reference_symbols,
        overlap_floor=robustness_overlap_floor,
        protocol=robustness_protocol,
    )
    summary = ProtocolSelectionSummary(
        protocol_selection_version=PROTOCOL_SELECTION_VERSION,
        identity_universe_count=len(symbols),
        train_return_capable_count=train_capable_count,
        identity_universe_hash=equity.summary.universe_hash,
        training_date_range=(partitions.training_start, partitions.training_end),
        validation_date_range=(
            partitions.validation_start,
            partitions.validation_end,
        ),
        test_date_range=(partitions.test_start, partitions.test_end),
        candidate_overlap_floors=floors,
        active_span_coverage_floor=MINIMUM_ACTIVE_SPAN_COVERAGE,
        minimum_overlapping_peers=MINIMUM_OVERLAPPING_PEERS,
        graph_correlation_threshold=GRAPH_CORRELATION_THRESHOLD,
        temporal_window_fraction=windows.fraction,
        temporal_window_ranges=(
            (windows.early_start, windows.early_end),
            (windows.late_start, windows.late_end),
        ),
        robustness_reference={
            "overlap_floor": robustness_overlap_floor,
            "protocol": robustness_protocol,
            "clusters": robustness_cluster_count,
            "clustered_symbols": len(reference_symbols),
        },
        protocol_selection_decision=PROTOCOL_SELECTION_DECISION,
        final_protocol_selected=False,
        validation_returns_used_for_fitting=False,
        test_returns_used_for_clustering=False,
        final_assignments_written=False,
    )
    return ProtocolSelectionResult(
        summary=summary,
        retention_comparison=retention_frame,
        evidence_table=evidence_frame,
        temporal_stability=temporal_frame,
        decision_table=decision,
        robustness_comparison=robustness,
        reference_unassigned_reasons=unassigned,
        parquet_path=resolved_parquet,
    )


def _print_summary(result: ProtocolSelectionResult) -> None:
    summary = result.summary
    print(f"Protocol selection version: {summary.protocol_selection_version}")
    print(f"Identity universe: {summary.identity_universe_count}")
    print(f"TRAIN-return capable: {summary.train_return_capable_count}")
    print(f"Identity hash: {summary.identity_universe_hash}")
    print(f"Protocol decision: {summary.protocol_selection_decision}")
    print(
        f"TRAIN: {summary.training_date_range[0]} to "
        f"{summary.training_date_range[1]}"
    )
    print(
        f"VALIDATION excluded from fitting: {summary.validation_date_range[0]} "
        f"to {summary.validation_date_range[1]}"
    )
    print(
        f"TEST sealed: {summary.test_date_range[0]} to "
        f"{summary.test_date_range[1]}"
    )
    print(
        "Temporal TRAIN windows: "
        + json.dumps(summary.temporal_window_ranges)
    )
    print("Retention/reliability comparison:")
    print(result.retention_comparison.to_string(index=False))
    selected_columns = [
        "overlap_floor",
        "protocol",
        "requested_clusters",
        "clustered_symbol_count",
        "silhouette",
        "cohesion_separation_gap",
        "largest_cluster_fraction",
        "minimum_cluster_size",
        "tiny_cluster_count",
        "adjusted_rand_index",
        "normalized_mutual_information",
        "temporal_common_symbol_count",
        "pareto_nondominated",
    ]
    print("Protocol evidence (assignments are not retained):")
    print(result.decision_table.loc[:, selected_columns].to_string(index=False))
    print("Return/correlation robustness reference:")
    print(result.robustness_comparison.to_string(index=False))
    print("Reference protocol statuses:")
    print(
        result.reference_unassigned_reasons["reason"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(
        "Safety: "
        f"validation_fit={summary.validation_returns_used_for_fitting}, "
        f"test_used={summary.test_returns_used_for_clustering}, "
        f"assignments_written={summary.final_assignments_written}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only Phase 1 final-protocol diagnostics."
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry", default=str(COMPANY_REGISTRY_PATH)
    )
    parser.add_argument(
        "--listing-snapshot", default=str(CURRENT_LISTINGS_PATH)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_clustering_protocol_selection_audit(
            parquet_path=args.path,
            registry_path=args.company_registry,
            listing_snapshot_path=args.listing_snapshot,
        )
        _print_summary(result)
        return 0
    except (
        ClusteringMethodologyError,
        ClusteringProtocolError,
        EquityUniverseError,
        ListingsUnavailableError,
        MarketParquetError,
        UniverseAuditError,
        UniverseMethodologyError,
        ValueError,
        TypeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_OVERLAP_FLOORS",
    "ClusteringProtocolError",
    "GRAPH_CORRELATION_THRESHOLD",
    "MINIMUM_ACTIVE_SPAN_COVERAGE",
    "MINIMUM_OVERLAPPING_PEERS",
    "PROTOCOL_SELECTION_DECISION",
    "PROTOCOL_SELECTION_VERSION",
    "ProtocolSelectionResult",
    "ProtocolSelectionSummary",
    "SimilarityGraph",
    "TEMPORAL_WINDOW_FRACTION",
    "TemporalWindows",
    "build_positive_correlation_graph",
    "build_unassigned_reason_table",
    "cluster_quality_diagnostics",
    "deterministic_joint_connected_core",
    "deterministic_temporal_windows",
    "eligible_symbols_for_overlap_floor",
    "evaluate_return_correlation_robustness",
    "graph_geodesic_distance",
    "hierarchical_labels",
    "main",
    "mark_pareto_nondominated",
    "run_clustering_protocol_selection_audit",
    "spectral_graph_labels",
)
