"""Bounded TRAIN-only sector-informed multi-view clustering audit.

The return view remains dominant.  Current authoritative sector tags are used
only through the predeclared convex weights in ``SECTOR_LAMBDAS``.  This module
never writes assignments and never loads validation or TEST market values.
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .clustering_market_mode import (
    FROZEN_TRAIN_END,
    FROZEN_TRAIN_START,
    REFERENCE_LINKAGE,
    REFERENCE_OVERLAP_FLOOR,
    build_equal_weight_market_factor,
    load_authoritative_current_equity_identity,
    load_train_only_market_values,
    residualize_static_market_factor,
)
from .clustering_methodology import (
    CANDIDATE_CLUSTER_COUNTS,
    ClusteringMethodologyError,
    build_return_matrix,
    build_training_symbol_diagnostics,
    construct_close_returns,
    correlation_to_distance,
    deterministic_complete_pair_core,
    minimum_overlap_correlation,
    pairwise_overlap_counts,
)
from .clustering_protocol import (
    ClusteringProtocolError,
    TemporalWindows,
    cluster_quality_diagnostics,
    deterministic_temporal_windows,
    eligible_symbols_for_overlap_floor,
    hierarchical_labels,
)
from .config import COMPANY_REGISTRY_PATH, CURRENT_LISTINGS_PATH
from .equity_universe import EquityUniverseError
from .official_listings import ListingsUnavailableError
from .parquet_market_data import (
    MarketParquetError,
    resolve_market_parquet_path,
)
from .universe_audit import UniverseAuditError
from .universe_methodology import UniverseMethodologyError


MULTIVIEW_AUDIT_VERSION = "phase1_sector_informed_multiview_audit_v1"
FINAL_DECISION = "BLOCKED_WEAK_CLUSTER_STRUCTURE"
RECOMMENDED_NEXT_ACTION = "ARCHITECTURE_REVIEW_REQUIRED"
SECTOR_LAMBDAS = (0.0, 0.10, 0.20, 0.30)
RETURN_DISTANCE_THEORETICAL_MAXIMUM = 2.0
ROBUSTNESS_LAMBDA = 0.10
# Lowest nonzero sector weight; within k=10..15, k=12 is the highest-temporal-
# ARI candidate that remains below the explicit sector-domination flag.  This
# selection is used only for the two bounded robustness checks, not as a freeze.
ROBUSTNESS_K = 12
SECTOR_DOMINATION_NMI_FLOOR = 0.80
SECTOR_DOMINATION_PURITY_FLOOR = 0.80
MATERIAL_RETURN_GAP_ABSOLUTE = 0.01
MATERIAL_RETURN_GAP_RELATIVE = 0.10

PRIMARY_COLUMNS = (
    "sector_lambda",
    "requested_clusters",
    "actual_clusters",
    "clustered_symbol_count",
    "silhouette_combined_distance",
    "mean_within_return_correlation",
    "mean_between_return_correlation",
    "return_cohesion_separation_gap",
    "return_gap_change_vs_lambda_zero",
    "material_return_gap_improvement",
    "minimum_cluster_size",
    "maximum_cluster_size",
    "largest_cluster_fraction",
    "cluster_size_coefficient_of_variation",
    "singleton_cluster_count",
    "tiny_cluster_count",
    "sector_nmi",
    "sector_purity",
    "normalized_within_cluster_sector_entropy",
    "sector_nmi_change_vs_lambda_zero",
    "sector_purity_change_vs_lambda_zero",
    "sector_domination_flag",
    "status",
)

TEMPORAL_COLUMNS = (
    "sector_lambda",
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
    "early_cluster_size_cv",
    "late_cluster_size_cv",
    "early_sector_nmi",
    "late_sector_nmi",
    "early_sector_purity",
    "late_sector_purity",
    "status",
)

ROBUSTNESS_COLUMNS = (
    "sector_lambda",
    "requested_clusters",
    "variant",
    "common_symbol_count",
    "adjusted_rand_index_vs_primary",
    "normalized_mutual_information_vs_primary",
    "status",
)

VIEW_COMPARISON_COLUMNS = (
    "sector_lambda",
    "requested_clusters",
    "common_symbol_count",
    "raw_view_silhouette",
    "residual_view_silhouette",
    "raw_labels_raw_return_gap",
    "residual_labels_raw_return_gap",
    "residual_labels_residual_return_gap",
    "raw_largest_cluster_fraction",
    "residual_largest_cluster_fraction",
    "raw_sector_nmi",
    "residual_sector_nmi",
    "raw_sector_purity",
    "residual_sector_purity",
    "raw_vs_residual_ari",
    "raw_vs_residual_nmi",
    "status",
)


class MultiViewAuditError(RuntimeError):
    """Raised when the bounded audit cannot proceed without unsafe assumptions."""


@dataclass(frozen=True)
class SectorMetadataProvenance:
    snapshot_date: str
    authoritative_sources: tuple[str, ...]
    identity_symbol_count: int
    sector_count: int
    missing_sector_count: int
    future_return_leakage: bool
    temporal_metadata_assessment: str
    limitation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiViewAuditSummary:
    audit_version: str
    final_decision: str
    recommended_next_action: str
    identity_universe_count: int
    train_return_capable_count: int
    eligible_symbol_count: int
    fixed_complete_core_count: int
    temporal_common_core_count: int
    sector_lambdas: tuple[float, ...]
    cluster_counts: tuple[int, ...]
    return_view: str
    sector_view: str
    combination_rule: str
    selected_robustness_lambda: float
    selected_robustness_k: int
    training_date_range: tuple[str, str]
    validation_values_loaded: bool
    test_dates_loaded: bool
    test_values_loaded: bool
    final_assignments_written: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiViewAuditResult:
    summary: MultiViewAuditSummary
    sector_provenance: SectorMetadataProvenance
    primary_evidence: pd.DataFrame = field(repr=False, compare=False)
    temporal_evidence: pd.DataFrame = field(repr=False, compare=False)
    robustness_evidence: pd.DataFrame = field(repr=False, compare=False)
    raw_residual_comparison: pd.DataFrame = field(repr=False, compare=False)
    parquet_path: Path


def _ordered_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(sorted(map(str, symbols)))
    if not ordered or len(set(ordered)) != len(ordered):
        raise MultiViewAuditError("Symbols must be non-empty and unique")
    return ordered


def _validate_lambda(sector_lambda: float) -> float:
    value = float(sector_lambda)
    if value not in SECTOR_LAMBDAS:
        raise ValueError(
            f"sector_lambda must be one of the predeclared values {SECTOR_LAMBDAS}"
        )
    return value


def build_sector_dissimilarity(
    symbols: Sequence[str],
    sector_by_symbol: Mapping[str, str],
) -> pd.DataFrame:
    """Construct categorical 0/1 sector dissimilarity without ordinal encoding."""

    ordered = _ordered_symbols(symbols)
    sectors: list[str] = []
    for symbol in ordered:
        raw = sector_by_symbol.get(symbol)
        sector = "" if raw is None else str(raw).strip()
        if not sector or sector.upper() == "UNKNOWN":
            raise MultiViewAuditError(
                f"Authoritative sector metadata is unavailable for {symbol}"
            )
        sectors.append(sector)
    values = np.not_equal.outer(
        np.asarray(sectors, dtype=object), np.asarray(sectors, dtype=object)
    ).astype("float64")
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=ordered, columns=ordered)


def normalized_return_dissimilarity(correlation: pd.DataFrame) -> pd.DataFrame:
    """Map angular-chord return distance from its [0,2] range into [0,1]."""

    distance = correlation_to_distance(correlation)
    normalized = distance / RETURN_DISTANCE_THEORETICAL_MAXIMUM
    values = normalized.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1 + 1e-12:
        raise MultiViewAuditError("Normalized return distance is outside [0, 1]")
    return normalized


def combine_multiview_dissimilarities(
    normalized_return_distance: pd.DataFrame,
    sector_distance: pd.DataFrame,
    *,
    sector_lambda: float,
) -> pd.DataFrame:
    """Combine return and categorical sector views with one declared lambda."""

    weight = _validate_lambda(sector_lambda)
    if list(normalized_return_distance.index) != list(
        normalized_return_distance.columns
    ):
        raise MultiViewAuditError("Return distance ordering is inconsistent")
    sector = sector_distance.reindex(
        index=normalized_return_distance.index,
        columns=normalized_return_distance.columns,
    )
    if sector.isna().any().any():
        raise MultiViewAuditError("Sector distance does not cover return symbols")
    return_values = normalized_return_distance.to_numpy(dtype="float64")
    sector_values = sector.to_numpy(dtype="float64")
    if not set(np.unique(sector_values)).issubset({0.0, 1.0}):
        raise MultiViewAuditError("Sector distance must be strictly categorical 0/1")
    combined = (1.0 - weight) * return_values + weight * sector_values
    if not np.allclose(combined, combined.T, atol=1e-12):
        raise MultiViewAuditError("Combined distance is not symmetric")
    np.fill_diagonal(combined, 0.0)
    return pd.DataFrame(
        combined,
        index=normalized_return_distance.index.copy(),
        columns=normalized_return_distance.columns.copy(),
    )


def sector_alignment_diagnostics(
    labels: Sequence[int] | np.ndarray,
    sectors: Sequence[str],
) -> dict[str, float]:
    """Return descriptive NMI, purity, and normalized within-cluster entropy."""

    labels_array = np.asarray(labels, dtype="int64")
    sector_values = np.asarray([str(value) for value in sectors], dtype=object)
    if len(labels_array) != len(sector_values) or not len(labels_array):
        raise MultiViewAuditError("Labels and sectors must align and be non-empty")
    unique_sectors = np.unique(sector_values)
    purity_numerator = 0
    weighted_entropy = 0.0
    entropy_scale = math.log(len(unique_sectors)) if len(unique_sectors) > 1 else 1.0
    for label in np.unique(labels_array):
        cluster_sectors = sector_values[labels_array == label]
        _, counts = np.unique(cluster_sectors, return_counts=True)
        purity_numerator += int(counts.max())
        probabilities = counts.astype("float64") / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        weighted_entropy += len(cluster_sectors) * entropy / entropy_scale
    return {
        "sector_nmi": float(
            normalized_mutual_info_score(sector_values, labels_array)
        ),
        "sector_purity": float(purity_numerator / len(labels_array)),
        "normalized_within_cluster_sector_entropy": float(
            weighted_entropy / len(labels_array)
        ),
    }


def sector_domination_diagnostics(
    *,
    sector_nmi: float,
    sector_purity: float,
    return_gap: float,
    baseline_return_gap: float,
) -> dict[str, float | bool]:
    """Flag strong metadata conformity without material return-gap improvement."""

    change = float(return_gap - baseline_return_gap)
    threshold = max(
        MATERIAL_RETURN_GAP_ABSOLUTE,
        MATERIAL_RETURN_GAP_RELATIVE * abs(float(baseline_return_gap)),
    )
    material = bool(change >= threshold)
    dominated = bool(
        (
            sector_nmi >= SECTOR_DOMINATION_NMI_FLOOR
            or sector_purity >= SECTOR_DOMINATION_PURITY_FLOOR
        )
        and not material
    )
    return {
        "return_gap_change_vs_lambda_zero": change,
        "material_return_gap_improvement": material,
        "sector_domination_flag": dominated,
    }


def _cluster_size_snapshot(labels: np.ndarray) -> dict[str, float | int]:
    sizes = pd.Series(labels).value_counts().sort_index().to_numpy(dtype="int64")
    return {
        "largest_cluster_fraction": float(sizes.max() / sizes.sum()),
        "minimum_cluster_size": int(sizes.min()),
        "singleton_cluster_count": int((sizes == 1).sum()),
        "tiny_cluster_count": int((sizes < 3).sum()),
        "cluster_size_cv": float(sizes.std(ddof=0) / sizes.mean()),
    }


def evaluate_primary_multiview_grid(
    return_correlation: pd.DataFrame,
    sector_by_symbol: Mapping[str, str],
    *,
    sector_lambdas: Sequence[float] = SECTOR_LAMBDAS,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> tuple[pd.DataFrame, dict[tuple[float, int], np.ndarray]]:
    """Evaluate the exact bounded lambda/k grid on one fixed complete core."""

    if tuple(map(float, sector_lambdas)) != SECTOR_LAMBDAS:
        raise ValueError(f"sector_lambdas must be exactly {SECTOR_LAMBDAS}")
    symbols = _ordered_symbols(return_correlation.index)
    correlation = return_correlation.loc[list(symbols), list(symbols)]
    if correlation.isna().any().any():
        raise MultiViewAuditError("Primary correlation core must be complete")
    return_distance = normalized_return_dissimilarity(correlation)
    sector_distance = build_sector_dissimilarity(symbols, sector_by_symbol)
    sectors = [sector_by_symbol[symbol] for symbol in symbols]
    rows: list[dict[str, object]] = []
    labels_by_candidate: dict[tuple[float, int], np.ndarray] = {}
    for sector_lambda in SECTOR_LAMBDAS:
        combined = combine_multiview_dissimilarities(
            return_distance,
            sector_distance,
            sector_lambda=sector_lambda,
        )
        for cluster_count in cluster_counts:
            count = int(cluster_count)
            if count < 2 or count >= len(symbols):
                continue
            labels = hierarchical_labels(
                combined,
                linkage=REFERENCE_LINKAGE,
                cluster_count=count,
            )
            labels_by_candidate[(sector_lambda, count)] = labels
            quality = cluster_quality_diagnostics(
                correlation,
                combined,
                labels,
                sector_by_symbol=sector_by_symbol,
            )
            alignment = sector_alignment_diagnostics(labels, sectors)
            rows.append(
                {
                    "sector_lambda": sector_lambda,
                    "requested_clusters": count,
                    "actual_clusters": quality["actual_clusters"],
                    "clustered_symbol_count": len(symbols),
                    "silhouette_combined_distance": quality["silhouette"],
                    "mean_within_return_correlation": quality[
                        "mean_within_cluster_correlation"
                    ],
                    "mean_between_return_correlation": quality[
                        "mean_between_cluster_correlation"
                    ],
                    "return_cohesion_separation_gap": quality[
                        "cohesion_separation_gap"
                    ],
                    "minimum_cluster_size": quality["minimum_cluster_size"],
                    "maximum_cluster_size": quality["maximum_cluster_size"],
                    "largest_cluster_fraction": quality[
                        "largest_cluster_fraction"
                    ],
                    "cluster_size_coefficient_of_variation": quality[
                        "cluster_size_coefficient_of_variation"
                    ],
                    "singleton_cluster_count": quality[
                        "singleton_cluster_count"
                    ],
                    "tiny_cluster_count": quality["tiny_cluster_count"],
                    **alignment,
                    "status": "completed_bounded_train_only_multiview",
                }
            )
    frame = pd.DataFrame(rows)
    baselines = frame.loc[frame["sector_lambda"].eq(0.0)].set_index(
        "requested_clusters"
    )
    augmented: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        baseline = baselines.loc[int(row["requested_clusters"])]
        domination = sector_domination_diagnostics(
            sector_nmi=float(row["sector_nmi"]),
            sector_purity=float(row["sector_purity"]),
            return_gap=float(row["return_cohesion_separation_gap"]),
            baseline_return_gap=float(
                baseline["return_cohesion_separation_gap"]
            ),
        )
        row.update(domination)
        row["sector_nmi_change_vs_lambda_zero"] = float(
            row["sector_nmi"] - baseline["sector_nmi"]
        )
        row["sector_purity_change_vs_lambda_zero"] = float(
            row["sector_purity"] - baseline["sector_purity"]
        )
        augmented.append(row)
    result = pd.DataFrame(augmented, columns=PRIMARY_COLUMNS)
    result = result.sort_values(
        ["sector_lambda", "requested_clusters"], kind="mergesort"
    ).reset_index(drop=True)
    return result, labels_by_candidate


def _window_return_matrix(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
    *,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_set = set(pd.DatetimeIndex(dates))
    subset = training_market.loc[
        pd.to_datetime(training_market["market_date"]).isin(date_set)
    ].copy(deep=True)
    built = construct_close_returns(
        subset,
        symbols,
        global_market_dates=dates,
    )
    matrix = build_return_matrix(
        built.returns,
        value_column=value_column,
        symbols=symbols,
    ).reindex(pd.DatetimeIndex(dates[1:]))
    return subset, matrix


def evaluate_temporal_multiview_grid(
    training_market: pd.DataFrame,
    identity_symbols: Sequence[str],
    full_core: Sequence[str],
    windows: TemporalWindows,
    sector_by_symbol: Mapping[str, str],
    *,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Evaluate fixed-sector multi-view labels in established TRAIN windows."""

    window_data: dict[str, dict[str, object]] = {}
    for name, dates in (
        ("early", windows.early_dates),
        ("late", windows.late_dates),
    ):
        market, matrix = _window_return_matrix(
            training_market,
            identity_symbols,
            dates,
            value_column="log_return",
        )
        overlap = pairwise_overlap_counts(matrix)
        diagnostics = build_training_symbol_diagnostics(
            market,
            identity_symbols,
            matrix,
        )
        eligible, _ = eligible_symbols_for_overlap_floor(
            diagnostics,
            overlap,
            overlap_floor=REFERENCE_OVERLAP_FLOOR,
        )
        correlation = minimum_overlap_correlation(
            matrix,
            method="pearson",
            minimum_overlap=REFERENCE_OVERLAP_FLOOR,
            overlap_counts=overlap,
        )
        window_data[name] = {
            "eligible": eligible,
            "correlation": correlation,
        }
    common_candidates = tuple(
        sorted(
            set(map(str, full_core))
            .intersection(window_data["early"]["eligible"])
            .intersection(window_data["late"]["eligible"])
        )
    )
    jointly_valid = (
        window_data["early"]["correlation"].notna()
        & window_data["late"]["correlation"].notna()
    )
    core = deterministic_complete_pair_core(
        jointly_valid,
        candidates=common_candidates,
    )
    if len(core) <= max(map(int, cluster_counts)):
        raise MultiViewAuditError("Temporal common core is too small")
    sector_distance = build_sector_dissimilarity(core, sector_by_symbol)
    sectors = [sector_by_symbol[symbol] for symbol in core]
    return_distances = {
        name: normalized_return_dissimilarity(
            window_data[name]["correlation"].loc[list(core), list(core)]
        )
        for name in ("early", "late")
    }
    rows: list[dict[str, object]] = []
    for sector_lambda in SECTOR_LAMBDAS:
        combined = {
            name: combine_multiview_dissimilarities(
                return_distances[name],
                sector_distance,
                sector_lambda=sector_lambda,
            )
            for name in ("early", "late")
        }
        for cluster_count in cluster_counts:
            count = int(cluster_count)
            early_labels = hierarchical_labels(
                combined["early"],
                linkage=REFERENCE_LINKAGE,
                cluster_count=count,
            )
            late_labels = hierarchical_labels(
                combined["late"],
                linkage=REFERENCE_LINKAGE,
                cluster_count=count,
            )
            early_size = _cluster_size_snapshot(early_labels)
            late_size = _cluster_size_snapshot(late_labels)
            early_sector = sector_alignment_diagnostics(early_labels, sectors)
            late_sector = sector_alignment_diagnostics(late_labels, sectors)
            rows.append(
                {
                    "sector_lambda": sector_lambda,
                    "requested_clusters": count,
                    "temporal_common_symbol_count": len(core),
                    "temporal_symbol_retention": len(core) / len(full_core),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(early_labels, late_labels)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(early_labels, late_labels)
                    ),
                    "early_largest_cluster_fraction": early_size[
                        "largest_cluster_fraction"
                    ],
                    "late_largest_cluster_fraction": late_size[
                        "largest_cluster_fraction"
                    ],
                    "largest_cluster_fraction_change": abs(
                        early_size["largest_cluster_fraction"]
                        - late_size["largest_cluster_fraction"]
                    ),
                    "early_minimum_cluster_size": early_size[
                        "minimum_cluster_size"
                    ],
                    "late_minimum_cluster_size": late_size[
                        "minimum_cluster_size"
                    ],
                    "early_singleton_cluster_count": early_size[
                        "singleton_cluster_count"
                    ],
                    "late_singleton_cluster_count": late_size[
                        "singleton_cluster_count"
                    ],
                    "early_tiny_cluster_count": early_size[
                        "tiny_cluster_count"
                    ],
                    "late_tiny_cluster_count": late_size[
                        "tiny_cluster_count"
                    ],
                    "early_cluster_size_cv": early_size["cluster_size_cv"],
                    "late_cluster_size_cv": late_size["cluster_size_cv"],
                    "early_sector_nmi": early_sector["sector_nmi"],
                    "late_sector_nmi": late_sector["sector_nmi"],
                    "early_sector_purity": early_sector["sector_purity"],
                    "late_sector_purity": late_sector["sector_purity"],
                    "status": "completed_fixed_sector_train_window_stability",
                }
            )
    return (
        pd.DataFrame(rows, columns=TEMPORAL_COLUMNS).sort_values(
            ["sector_lambda", "requested_clusters"], kind="mergesort"
        ).reset_index(drop=True),
        core,
    )


def evaluate_selected_representation_robustness(
    primary_correlation: pd.DataFrame,
    simple_correlation: pd.DataFrame,
    spearman_correlation: pd.DataFrame,
    symbols: Sequence[str],
    sector_by_symbol: Mapping[str, str],
    *,
    sector_lambda: float,
    cluster_count: int,
) -> pd.DataFrame:
    """Evaluate exactly one already-selected lambda/k across return definitions."""

    weight = _validate_lambda(sector_lambda)
    if weight == 0:
        raise ValueError("Robustness candidate must use a nonzero sector lambda")
    ordered = _ordered_symbols(symbols)
    sector_distance = build_sector_dissimilarity(ordered, sector_by_symbol)
    correlations = {
        "primary_log_pearson": primary_correlation,
        "simple_return_pearson": simple_correlation,
        "log_return_spearman": spearman_correlation,
    }
    labels: dict[str, np.ndarray] = {}
    for name, correlation in correlations.items():
        selected = correlation.loc[list(ordered), list(ordered)]
        if selected.isna().any().any():
            raise MultiViewAuditError(
                f"Robustness correlation is incomplete for {name}"
            )
        combined = combine_multiview_dissimilarities(
            normalized_return_dissimilarity(selected),
            sector_distance,
            sector_lambda=weight,
        )
        labels[name] = hierarchical_labels(
            combined,
            linkage=REFERENCE_LINKAGE,
            cluster_count=int(cluster_count),
        )
    primary = labels["primary_log_pearson"]
    rows: list[dict[str, object]] = []
    for variant in ("simple_return_pearson", "log_return_spearman"):
        rows.append(
            {
                "sector_lambda": weight,
                "requested_clusters": int(cluster_count),
                "variant": variant,
                "common_symbol_count": len(ordered),
                "adjusted_rand_index_vs_primary": float(
                    adjusted_rand_score(primary, labels[variant])
                ),
                "normalized_mutual_information_vs_primary": float(
                    normalized_mutual_info_score(primary, labels[variant])
                ),
                "status": "completed_same_lambda_fixed_core_robustness",
            }
        )
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)


def compare_raw_and_residual_multiview(
    raw_correlation: pd.DataFrame,
    residual_correlation: pd.DataFrame,
    symbols: Sequence[str],
    sector_by_symbol: Mapping[str, str],
    *,
    sector_lambda: float,
    cluster_count: int,
) -> pd.DataFrame:
    """Compare raw and residual views using exactly the same lambda/core/k."""

    weight = _validate_lambda(sector_lambda)
    if weight == 0:
        raise ValueError("Raw/residual comparison requires a nonzero lambda")
    ordered = _ordered_symbols(symbols)
    raw = raw_correlation.loc[list(ordered), list(ordered)]
    residual = residual_correlation.loc[list(ordered), list(ordered)]
    if raw.isna().any().any() or residual.isna().any().any():
        raise MultiViewAuditError("Raw/residual comparison core must be complete")
    sector_distance = build_sector_dissimilarity(ordered, sector_by_symbol)
    raw_distance = combine_multiview_dissimilarities(
        normalized_return_dissimilarity(raw),
        sector_distance,
        sector_lambda=weight,
    )
    residual_distance = combine_multiview_dissimilarities(
        normalized_return_dissimilarity(residual),
        sector_distance,
        sector_lambda=weight,
    )
    raw_labels = hierarchical_labels(
        raw_distance,
        linkage=REFERENCE_LINKAGE,
        cluster_count=int(cluster_count),
    )
    residual_labels = hierarchical_labels(
        residual_distance,
        linkage=REFERENCE_LINKAGE,
        cluster_count=int(cluster_count),
    )
    raw_quality = cluster_quality_diagnostics(raw, raw_distance, raw_labels)
    residual_quality = cluster_quality_diagnostics(
        residual,
        residual_distance,
        residual_labels,
    )
    residual_raw_quality = cluster_quality_diagnostics(
        raw,
        residual_distance,
        residual_labels,
    )
    sectors = [sector_by_symbol[symbol] for symbol in ordered]
    raw_sector = sector_alignment_diagnostics(raw_labels, sectors)
    residual_sector = sector_alignment_diagnostics(residual_labels, sectors)
    row = {
        "sector_lambda": weight,
        "requested_clusters": int(cluster_count),
        "common_symbol_count": len(ordered),
        "raw_view_silhouette": raw_quality["silhouette"],
        "residual_view_silhouette": residual_quality["silhouette"],
        "raw_labels_raw_return_gap": raw_quality["cohesion_separation_gap"],
        "residual_labels_raw_return_gap": residual_raw_quality[
            "cohesion_separation_gap"
        ],
        "residual_labels_residual_return_gap": residual_quality[
            "cohesion_separation_gap"
        ],
        "raw_largest_cluster_fraction": raw_quality[
            "largest_cluster_fraction"
        ],
        "residual_largest_cluster_fraction": residual_quality[
            "largest_cluster_fraction"
        ],
        "raw_sector_nmi": raw_sector["sector_nmi"],
        "residual_sector_nmi": residual_sector["sector_nmi"],
        "raw_sector_purity": raw_sector["sector_purity"],
        "residual_sector_purity": residual_sector["sector_purity"],
        "raw_vs_residual_ari": float(
            adjusted_rand_score(raw_labels, residual_labels)
        ),
        "raw_vs_residual_nmi": float(
            normalized_mutual_info_score(raw_labels, residual_labels)
        ),
        "status": "completed_same_lambda_core_k_view_comparison",
    }
    return pd.DataFrame([row], columns=VIEW_COMPARISON_COLUMNS)


def _sector_provenance(identity: pd.DataFrame) -> SectorMetadataProvenance:
    snapshot_dates = sorted(set(identity["snapshot_date"].astype(str)))
    if len(snapshot_dates) != 1:
        raise MultiViewAuditError("Sector records do not share one snapshot date")
    sources = tuple(sorted(set(identity["source"].astype(str))))
    sectors = identity["sector"].astype("string").str.strip()
    missing = int(sectors.isna().sum() + sectors.eq("").sum())
    return SectorMetadataProvenance(
        snapshot_date=snapshot_dates[0],
        authoritative_sources=sources,
        identity_symbol_count=len(identity),
        sector_count=int(sectors.nunique(dropna=True)),
        missing_sector_count=missing,
        future_return_leakage=False,
        temporal_metadata_assessment=(
            "acceptable_only_for_the_frozen_current_universe_with_explicit_"
            "current_metadata_limitation"
        ),
        limitation=(
            "The 2026 official listing snapshot is static current-company metadata, "
            "not point-in-time proof of sector membership throughout 2016-2023. "
            "It introduces a temporal metadata/look-ahead limitation but no future "
            "return values. Results cannot be described as survivorship-free or as "
            "historically effective-dated sector clustering."
        ),
    )


def run_sector_multiview_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
    robustness_lambda: float = ROBUSTNESS_LAMBDA,
    robustness_k: int = ROBUSTNESS_K,
) -> MultiViewAuditResult:
    """Run the bounded sector multi-view experiment on TRAIN observations only."""

    if tuple(map(int, cluster_counts)) != tuple(CANDIDATE_CLUSTER_COUNTS):
        raise ValueError(
            f"cluster_counts must be exactly {tuple(CANDIDATE_CLUSTER_COUNTS)}"
        )
    selected_lambda = _validate_lambda(robustness_lambda)
    if selected_lambda == 0:
        raise ValueError("Selected robustness lambda must be nonzero")
    if int(robustness_k) not in set(map(int, cluster_counts)):
        raise ValueError("Selected robustness k must be in the bounded grid")

    identity = load_authoritative_current_equity_identity(
        registry_path=registry_path,
        listing_snapshot_path=listing_snapshot_path,
    )
    resolved_parquet = resolve_market_parquet_path(parquet_path)
    identity_symbols = tuple(identity["symbol"].astype(str))
    partitions, training_market = load_train_only_market_values(
        resolved_parquet,
        identity_symbols,
    )
    built = construct_close_returns(
        training_market,
        identity_symbols,
        global_market_dates=partitions.training_dates,
    )
    return_index = pd.DatetimeIndex(partitions.training_dates[1:])
    log_matrix = build_return_matrix(
        built.returns,
        value_column="log_return",
        symbols=identity_symbols,
    ).reindex(return_index)
    simple_matrix = build_return_matrix(
        built.returns,
        value_column="simple_return",
        symbols=identity_symbols,
    ).reindex(return_index)
    overlap = pairwise_overlap_counts(log_matrix)
    diagnostics = build_training_symbol_diagnostics(
        training_market,
        identity_symbols,
        log_matrix,
    )
    eligible, _ = eligible_symbols_for_overlap_floor(
        diagnostics,
        overlap,
        overlap_floor=REFERENCE_OVERLAP_FLOOR,
    )
    eligible_list = list(eligible)
    raw_correlation = minimum_overlap_correlation(
        log_matrix.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=overlap.loc[eligible_list, eligible_list],
    )
    core = deterministic_complete_pair_core(
        raw_correlation.notna(),
        candidates=eligible,
    )
    sector_by_symbol = identity.set_index("symbol")["sector"].astype(str).to_dict()
    primary, _ = evaluate_primary_multiview_grid(
        raw_correlation.loc[list(core), list(core)],
        sector_by_symbol,
        cluster_counts=cluster_counts,
    )
    windows = deterministic_temporal_windows(partitions.training_dates)
    temporal, temporal_core = evaluate_temporal_multiview_grid(
        training_market,
        identity_symbols,
        core,
        windows,
        sector_by_symbol,
        cluster_counts=cluster_counts,
    )

    eligible_overlap = overlap.loc[eligible_list, eligible_list]
    simple_correlation = minimum_overlap_correlation(
        simple_matrix.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=eligible_overlap,
    )
    spearman_correlation = minimum_overlap_correlation(
        log_matrix.loc[:, eligible_list],
        method="spearman",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=eligible_overlap,
    )
    robustness = evaluate_selected_representation_robustness(
        raw_correlation,
        simple_correlation,
        spearman_correlation,
        core,
        sector_by_symbol,
        sector_lambda=selected_lambda,
        cluster_count=int(robustness_k),
    )

    factor = build_equal_weight_market_factor(log_matrix, eligible)
    residual = residualize_static_market_factor(
        log_matrix,
        factor.factor,
        eligible,
        factor_name="train_equal_weight_current_equity_factor",
    )
    residual_overlap = pairwise_overlap_counts(
        residual.residuals.loc[:, eligible_list]
    )
    residual_correlation = minimum_overlap_correlation(
        residual.residuals.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=residual_overlap,
    )
    comparison_core = deterministic_complete_pair_core(
        raw_correlation.notna() & residual_correlation.notna(),
        candidates=core,
    )
    raw_residual = compare_raw_and_residual_multiview(
        raw_correlation,
        residual_correlation,
        comparison_core,
        sector_by_symbol,
        sector_lambda=selected_lambda,
        cluster_count=int(robustness_k),
    )
    train_capable = int(
        diagnostics["training_return_observations"].gt(0).sum()
    )
    summary = MultiViewAuditSummary(
        audit_version=MULTIVIEW_AUDIT_VERSION,
        final_decision=FINAL_DECISION,
        recommended_next_action=RECOMMENDED_NEXT_ACTION,
        identity_universe_count=len(identity_symbols),
        train_return_capable_count=train_capable,
        eligible_symbol_count=len(eligible),
        fixed_complete_core_count=len(core),
        temporal_common_core_count=len(temporal_core),
        sector_lambdas=SECTOR_LAMBDAS,
        cluster_counts=tuple(map(int, cluster_counts)),
        return_view="TRAIN log close-to-close Pearson angular-chord / 2",
        sector_view="current authoritative sector categorical 0/1",
        combination_rule="(1-lambda)*normalized_return + lambda*sector",
        selected_robustness_lambda=selected_lambda,
        selected_robustness_k=int(robustness_k),
        training_date_range=(FROZEN_TRAIN_START, FROZEN_TRAIN_END),
        validation_values_loaded=False,
        test_dates_loaded=False,
        test_values_loaded=False,
        final_assignments_written=False,
    )
    return MultiViewAuditResult(
        summary=summary,
        sector_provenance=_sector_provenance(identity),
        primary_evidence=primary,
        temporal_evidence=temporal,
        robustness_evidence=robustness,
        raw_residual_comparison=raw_residual,
        parquet_path=resolved_parquet,
    )


def _print_result(result: MultiViewAuditResult) -> None:
    print("Summary:")
    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    print("Sector provenance:")
    print(json.dumps(result.sector_provenance.to_dict(), indent=2, sort_keys=True))
    print("Primary lambda/k evidence:")
    print(result.primary_evidence.to_string(index=False))
    print("Temporal evidence:")
    print(result.temporal_evidence.to_string(index=False))
    print("Selected representation robustness:")
    print(result.robustness_evidence.to_string(index=False))
    print("Raw versus residual same-lambda/core/k comparison:")
    print(result.raw_residual_comparison.to_string(index=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded TRAIN-only sector multi-view diagnostics."
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument("--company-registry", default=str(COMPANY_REGISTRY_PATH))
    parser.add_argument("--listing-snapshot", default=str(CURRENT_LISTINGS_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_sector_multiview_audit(
            parquet_path=args.path,
            registry_path=args.company_registry,
            listing_snapshot_path=args.listing_snapshot,
        )
        _print_result(result)
        return 0
    except (
        ClusteringMethodologyError,
        ClusteringProtocolError,
        EquityUniverseError,
        ListingsUnavailableError,
        MarketParquetError,
        MultiViewAuditError,
        UniverseAuditError,
        UniverseMethodologyError,
        ValueError,
        TypeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = (
    "FINAL_DECISION",
    "MULTIVIEW_AUDIT_VERSION",
    "MultiViewAuditError",
    "MultiViewAuditResult",
    "MultiViewAuditSummary",
    "RECOMMENDED_NEXT_ACTION",
    "ROBUSTNESS_K",
    "ROBUSTNESS_LAMBDA",
    "SECTOR_LAMBDAS",
    "SectorMetadataProvenance",
    "build_sector_dissimilarity",
    "combine_multiview_dissimilarities",
    "compare_raw_and_residual_multiview",
    "evaluate_primary_multiview_grid",
    "evaluate_selected_representation_robustness",
    "evaluate_temporal_multiview_grid",
    "main",
    "normalized_return_dissimilarity",
    "run_sector_multiview_audit",
    "sector_alignment_diagnostics",
    "sector_domination_diagnostics",
)
