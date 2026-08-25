"""TRAIN-only diagnostics for Phase 1 correlation-clustering methodology.

This module never persists or freezes cluster assignments.  It separates the
authoritative identity universe from experiment-specific clustering
eligibility and keeps validation/test returns outside all fitted diagnostics.
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from feature_engineering.splitting import chronological_split

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
from .universe_audit import UniverseAuditError, build_symbol_universe_audit
from .universe_methodology import (
    UniverseMethodologyError,
    add_active_span_coverage,
)


CLUSTERING_METHODOLOGY_VERSION = "phase1_clustering_methodology_audit_v1"
PRIMARY_RETURN_REPRESENTATION = "log_return"
PRIMARY_CORRELATION_METHOD = "pearson"
ROBUSTNESS_CORRELATION_METHOD = "spearman"
PRIMARY_DISTANCE_TRANSFORM = "angular_chord"
PRIMARY_CLUSTERING_METHOD = "hierarchical_average_precomputed"

OVERLAP_THRESHOLDS = (60, 120, 252, 504, 756)
ACTIVE_COVERAGE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
MINIMUM_OVERLAPPING_PEERS = (5, 10, 20)
EXPLORATORY_MINIMUM_OVERLAP = 252
CANDIDATE_CLUSTER_COUNTS = tuple(range(8, 21))
STABILITY_WINDOW_FRACTION = 0.70

RETURN_COLUMNS = (
    "market_date",
    "symbol",
    "simple_return",
    "log_return",
)
OVERLAP_COLUMNS = (
    "symbol_a",
    "symbol_b",
    "shared_return_observations",
    "overlap_ratio_min_history",
    "overlap_ratio_union",
)
ELIGIBILITY_COLUMNS = (
    "minimum_return_observations",
    "minimum_active_span_coverage",
    "minimum_overlapping_peers",
    "peer_overlap_requirement",
    "eligible_symbol_count",
)
CLUSTER_DIAGNOSTIC_COLUMNS = (
    "linkage",
    "requested_clusters",
    "actual_clusters",
    "symbol_count",
    "silhouette",
    "mean_within_cluster_correlation",
    "mean_between_cluster_correlation",
    "cohesion_separation_gap",
    "minimum_cluster_size",
    "maximum_cluster_size",
    "cluster_size_coefficient_of_variation",
    "singleton_cluster_count",
    "tiny_cluster_count",
    "sector_nmi_posthoc",
    "status",
)
STABILITY_COLUMNS = (
    "linkage",
    "requested_clusters",
    "actual_early_clusters",
    "actual_late_clusters",
    "common_symbol_count",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "status",
)


class ClusteringMethodologyError(RuntimeError):
    """Raised when leakage-safe clustering diagnostics cannot be computed."""


class NonPositiveCloseError(ClusteringMethodologyError):
    """Raised when close-return construction encounters an invalid close."""


@dataclass(frozen=True)
class TemporalDatePartitions:
    training_dates: tuple[pd.Timestamp, ...] = field(repr=False)
    validation_dates: tuple[pd.Timestamp, ...] = field(repr=False)
    test_dates: tuple[pd.Timestamp, ...] = field(repr=False)
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class ReturnConstructionResult:
    returns: pd.DataFrame = field(repr=False, compare=False)
    source_rows: int
    symbols_requested: int
    symbols_with_returns: int
    valid_return_rows: int
    first_observation_rows_rejected: int
    gap_spanning_rows_rejected: int
    non_positive_close_rows: int


@dataclass(frozen=True)
class ClusteringMethodologySummary:
    methodology_version: str
    identity_universe_count: int
    identity_universe_hash: str
    training_date_count: int
    training_date_range: tuple[str, str]
    validation_date_count: int
    validation_date_range: tuple[str, str]
    test_date_count: int
    test_date_range: tuple[str, str]
    return_representation: str
    valid_return_rows: int
    symbols_with_training_returns: int
    first_observation_rows_rejected: int
    gap_spanning_rows_rejected: int
    overlap_pair_count: int
    overlap_count_quantiles: Mapping[str, float | None]
    overlap_ratio_quantiles: Mapping[str, float | None]
    overlap_threshold_pair_counts: Mapping[str, int]
    exploratory_minimum_overlap: int
    finite_pearson_pair_count: int
    finite_spearman_pair_count: int
    missing_pair_count_at_exploratory_overlap: int
    pearson_vs_spearman: Mapping[str, float | int | None]
    simple_vs_log_pearson: Mapping[str, float | int | None]
    complete_overlap_core_count: int
    complete_overlap_core_symbols: tuple[str, ...]
    stability_common_symbol_count: int
    test_returns_used: bool
    final_clusters_frozen: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["complete_overlap_core_symbols"] = list(
            self.complete_overlap_core_symbols
        )
        return payload


@dataclass(frozen=True)
class ClusteringMethodologyResult:
    summary: ClusteringMethodologySummary
    overlap_pairs: pd.DataFrame = field(repr=False, compare=False)
    eligibility_sensitivity: pd.DataFrame = field(repr=False, compare=False)
    cluster_count_diagnostics: pd.DataFrame = field(repr=False, compare=False)
    stability_diagnostics: pd.DataFrame = field(repr=False, compare=False)
    parquet_path: Path


def canonical_temporal_date_partitions(
    market_dates: Sequence[object] | pd.Series | pd.Index,
) -> TemporalDatePartitions:
    """Apply the existing canonical 70/15/15 global-date split policy."""

    dates = pd.DatetimeIndex(pd.to_datetime(market_dates, errors="coerce"))
    if dates.isna().any():
        raise ClusteringMethodologyError("Market date set contains invalid dates")
    dates = dates.unique().sort_values()
    if len(dates) < 3:
        raise ClusteringMethodologyError(
            "At least three global market dates are required"
        )
    boundary_frame = pd.DataFrame(
        {"symbol": "__GLOBAL_DATE_BOUNDARY__", "date": dates}
    )
    split = chronological_split(boundary_frame, scope="master")

    def values(frame: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
        return tuple(pd.DatetimeIndex(frame["date"]).sort_values())

    training = values(split.train)
    validation = values(split.validation)
    test = values(split.test)
    return TemporalDatePartitions(
        training_dates=training,
        validation_dates=validation,
        test_dates=test,
        training_start=training[0].date().isoformat(),
        training_end=training[-1].date().isoformat(),
        validation_start=validation[0].date().isoformat(),
        validation_end=validation[-1].date().isoformat(),
        test_start=test[0].date().isoformat(),
        test_end=test[-1].date().isoformat(),
    )


def training_market_view(
    market: pd.DataFrame,
    partitions: TemporalDatePartitions,
) -> pd.DataFrame:
    """Return only canonical TRAIN rows; validation and test remain excluded."""

    if "market_date" not in market:
        raise ClusteringMethodologyError("Market data is missing market_date")
    dates = pd.to_datetime(market["market_date"], errors="coerce")
    if dates.isna().any():
        raise ClusteringMethodologyError("Market data contains invalid dates")
    training_dates = set(partitions.training_dates)
    result = market.loc[dates.isin(training_dates)].copy(deep=True)
    result["market_date"] = dates.loc[result.index]
    result = result.sort_values(
        ["market_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if not result.empty and result["market_date"].max() > pd.Timestamp(
        partitions.training_end
    ):
        raise ClusteringMethodologyError("Training market view crossed its boundary")
    return result


def construct_close_returns(
    market: pd.DataFrame,
    symbols: Sequence[str],
    *,
    global_market_dates: Sequence[object] | pd.Series | pd.Index | None = None,
) -> ReturnConstructionResult:
    """Construct strict one-market-session simple and log close returns."""

    required = {"market_date", "symbol", "close"}
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ClusteringMethodologyError(
            "Market data is missing return columns: " + ", ".join(missing)
        )
    selected_symbols = tuple(sorted({str(symbol).strip() for symbol in symbols}))
    if not selected_symbols or any(not symbol for symbol in selected_symbols):
        raise ClusteringMethodologyError("At least one non-blank symbol is required")
    frame = market.loc[
        market["symbol"].astype("string").isin(selected_symbols),
        ["market_date", "symbol", "close"],
    ].copy(deep=True)
    frame["symbol"] = frame["symbol"].astype("string").str.strip()
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["market_date"].isna().any() or frame["close"].isna().any():
        raise ClusteringMethodologyError(
            "Return source contains invalid dates or missing/non-numeric close"
        )
    duplicates = frame.duplicated(["market_date", "symbol"], keep=False)
    if duplicates.any():
        raise ClusteringMethodologyError(
            "Return source contains duplicate (market_date, symbol) rows"
        )
    non_positive = frame["close"] <= 0
    if non_positive.any():
        examples = ", ".join(
            frame.loc[non_positive, "symbol"].astype(str).drop_duplicates().head(10)
        )
        raise NonPositiveCloseError(
            f"Close returns require positive close; found {int(non_positive.sum())} "
            f"rows (symbols: {examples})"
        )
    frame = frame.sort_values(
        ["symbol", "market_date"], kind="mergesort"
    ).reset_index(drop=True)
    calendar_source = (
        frame["market_date"]
        if global_market_dates is None
        else global_market_dates
    )
    calendar = pd.DatetimeIndex(
        pd.to_datetime(calendar_source, errors="coerce")
    )
    if calendar.isna().any():
        raise ClusteringMethodologyError("Global return calendar has invalid dates")
    calendar = calendar.unique().sort_values()
    date_rank = pd.Series(
        np.arange(len(calendar), dtype="int64"), index=calendar
    )
    frame["date_rank"] = frame["market_date"].map(date_rank)
    if frame["date_rank"].isna().any():
        raise ClusteringMethodologyError(
            "A source date is absent from the global return calendar"
        )
    grouped = frame.groupby("symbol", sort=False, observed=True)
    frame["previous_close"] = grouped["close"].shift(1)
    frame["previous_date_rank"] = grouped["date_rank"].shift(1)
    first_rows = frame["previous_close"].isna()
    consecutive = frame["date_rank"].eq(frame["previous_date_rank"] + 1)
    gap_rows = ~first_rows & ~consecutive
    valid = ~first_rows & consecutive
    ratio = frame.loc[valid, "close"] / frame.loc[valid, "previous_close"]
    output = frame.loc[valid, ["market_date", "symbol"]].copy()
    output["simple_return"] = ratio.to_numpy(dtype="float64") - 1.0
    output["log_return"] = np.log(ratio.to_numpy(dtype="float64"))
    if not np.isfinite(
        output[["simple_return", "log_return"]].to_numpy(dtype="float64")
    ).all():
        raise ClusteringMethodologyError("Return construction produced NaN/Inf")
    output = output.loc[:, RETURN_COLUMNS].sort_values(
        ["symbol", "market_date"], kind="mergesort"
    ).reset_index(drop=True)
    return ReturnConstructionResult(
        returns=output,
        source_rows=len(frame),
        symbols_requested=len(selected_symbols),
        symbols_with_returns=int(output["symbol"].nunique()),
        valid_return_rows=len(output),
        first_observation_rows_rejected=int(first_rows.sum()),
        gap_spanning_rows_rejected=int(gap_rows.sum()),
        non_positive_close_rows=0,
    )


def build_return_matrix(
    returns: pd.DataFrame,
    *,
    value_column: str,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Build a date-by-symbol matrix without filling any missing return."""

    if value_column not in {"simple_return", "log_return"}:
        raise ValueError("value_column must be simple_return or log_return")
    missing = sorted(
        {"market_date", "symbol", value_column}.difference(returns.columns)
    )
    if missing:
        raise ClusteringMethodologyError(
            "Returns are missing matrix columns: " + ", ".join(missing)
        )
    if returns.duplicated(["market_date", "symbol"]).any():
        raise ClusteringMethodologyError("Returns contain duplicate symbol dates")
    ordered_symbols = tuple(sorted({str(symbol) for symbol in symbols}))
    matrix = returns.pivot(
        index="market_date", columns="symbol", values=value_column
    )
    return matrix.reindex(columns=ordered_symbols).sort_index()


def pairwise_overlap_counts(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the exact shared-observation matrix for all symbols."""

    observed = return_matrix.notna().to_numpy(dtype="int64")
    counts = observed.T @ observed
    return pd.DataFrame(
        counts,
        index=return_matrix.columns.copy(),
        columns=return_matrix.columns.copy(),
        dtype="int64",
    )


def pairwise_overlap_table(
    return_matrix: pd.DataFrame,
    overlap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one deterministic record for every unordered stock pair."""

    counts = (
        pairwise_overlap_counts(return_matrix)
        if overlap_counts is None
        else overlap_counts
    )
    symbols = list(return_matrix.columns)
    individual = return_matrix.notna().sum(axis=0).astype(int)
    rows: list[dict[str, object]] = []
    for left_index, symbol_a in enumerate(symbols):
        for symbol_b in symbols[left_index + 1 :]:
            shared = int(counts.loc[symbol_a, symbol_b])
            minimum = min(int(individual[symbol_a]), int(individual[symbol_b]))
            union = (
                int(individual[symbol_a])
                + int(individual[symbol_b])
                - shared
            )
            rows.append(
                {
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "shared_return_observations": shared,
                    "overlap_ratio_min_history": (
                        shared / minimum if minimum > 0 else np.nan
                    ),
                    "overlap_ratio_union": shared / union if union > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)


def minimum_overlap_correlation(
    return_matrix: pd.DataFrame,
    *,
    method: str,
    minimum_overlap: int,
    overlap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute pairwise-complete Pearson/Spearman correlation with a hard floor."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    if minimum_overlap < 2:
        raise ValueError("minimum_overlap must be at least 2")
    counts = (
        pairwise_overlap_counts(return_matrix)
        if overlap_counts is None
        else overlap_counts
    )
    correlation = return_matrix.corr(
        method=method,
        min_periods=minimum_overlap,
    )
    correlation = correlation.mask(counts < minimum_overlap)
    return correlation.reindex(
        index=return_matrix.columns, columns=return_matrix.columns
    )


def correlation_to_distance(
    correlation: pd.DataFrame,
    *,
    transform: str = PRIMARY_DISTANCE_TRANSFORM,
) -> pd.DataFrame:
    """Transform a complete correlation matrix into a symmetric dissimilarity."""

    if correlation.shape[0] != correlation.shape[1]:
        raise ClusteringMethodologyError("Correlation matrix must be square")
    if list(correlation.index) != list(correlation.columns):
        raise ClusteringMethodologyError(
            "Correlation matrix index/columns must have identical ordering"
        )
    values = correlation.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ClusteringMethodologyError(
            "Distance transformation requires complete finite correlations"
        )
    values = np.clip((values + values.T) / 2.0, -1.0, 1.0)
    if transform == "angular_chord":
        distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - values)))
    elif transform == "one_minus":
        distance = 1.0 - values
    else:
        raise ValueError("transform must be angular_chord or one_minus")
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(
        distance,
        index=correlation.index.copy(),
        columns=correlation.columns.copy(),
    )


def deterministic_complete_pair_core(
    valid_pair_matrix: pd.DataFrame,
    *,
    candidates: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Find a deterministic complete-pair diagnostic core by degree pruning."""

    if list(valid_pair_matrix.index) != list(valid_pair_matrix.columns):
        raise ClusteringMethodologyError("Valid-pair matrix ordering is inconsistent")
    available = set(str(symbol) for symbol in valid_pair_matrix.index)
    remaining = set(available if candidates is None else map(str, candidates))
    unknown = sorted(remaining.difference(available))
    if unknown:
        raise ClusteringMethodologyError(
            "Core candidates are absent from the pair matrix: " + ", ".join(unknown)
        )
    valid = valid_pair_matrix.astype(bool)
    while len(remaining) > 1:
        ordered = sorted(remaining)
        submatrix = valid.loc[ordered, ordered].to_numpy(
            dtype=bool, copy=True
        )
        np.fill_diagonal(submatrix, True)
        if bool(submatrix.all()):
            break
        degree = pd.Series(
            submatrix.sum(axis=1).astype(int) - 1,
            index=ordered,
        )
        minimum_degree = int(degree.min())
        # Remove the lexically last tied symbol for reproducibility while
        # preserving the largest degree-supported prefix of the candidate set.
        remove = sorted(degree.index[degree == minimum_degree])[-1]
        remaining.remove(str(remove))
    return tuple(sorted(remaining))


def build_training_symbol_diagnostics(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    return_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build TRAIN-only coverage/return diagnostics for every identity member."""

    selected = training_market.loc[
        training_market["symbol"].astype("string").isin(symbols)
    ].copy(deep=True)
    if selected.empty:
        raise ClusteringMethodologyError("No identity-universe rows exist in TRAIN")
    observed = build_symbol_universe_audit(selected)
    observed = add_active_span_coverage(observed, training_market["market_date"])
    base = pd.DataFrame({"symbol": tuple(sorted(set(map(str, symbols))))})
    quality = observed.loc[
        :, ["symbol", "observation_count", "active_span_coverage"]
    ].rename(columns={"observation_count": "training_price_observations"})
    result = base.merge(
        quality, on="symbol", how="left", validate="one_to_one", sort=False
    )
    result["training_price_observations"] = (
        result["training_price_observations"].fillna(0).astype("int64")
    )
    result["active_span_coverage"] = result["active_span_coverage"].fillna(0.0)
    return_counts = return_matrix.notna().sum(axis=0).astype("int64")
    result["training_return_observations"] = (
        result["symbol"].map(return_counts).fillna(0).astype("int64")
    )
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def build_eligibility_sensitivity(
    symbol_diagnostics: pd.DataFrame,
    overlap_counts: pd.DataFrame,
    *,
    overlap_thresholds: Sequence[int] = OVERLAP_THRESHOLDS,
    active_coverage_thresholds: Sequence[float] = ACTIVE_COVERAGE_THRESHOLDS,
    minimum_peers: Sequence[int] = MINIMUM_OVERLAPPING_PEERS,
) -> pd.DataFrame:
    """Count candidate clustering cohorts without freezing any eligibility rule."""

    required = {
        "symbol",
        "training_return_observations",
        "active_span_coverage",
    }
    missing = sorted(required.difference(symbol_diagnostics.columns))
    if missing:
        raise ClusteringMethodologyError(
            "Symbol diagnostics are missing eligibility columns: "
            + ", ".join(missing)
        )
    diagnostics = symbol_diagnostics.set_index("symbol")
    rows: list[dict[str, object]] = []
    for overlap_requirement in overlap_thresholds:
        if overlap_requirement < 2:
            raise ValueError("Overlap thresholds must be at least 2")
        peer_counts = (overlap_counts >= overlap_requirement).sum(axis=1) - 1
        peer_counts = peer_counts.reindex(diagnostics.index).fillna(0).astype(int)
        for coverage in active_coverage_thresholds:
            for peers in minimum_peers:
                mask = (
                    diagnostics["training_return_observations"].ge(
                        overlap_requirement
                    )
                    & diagnostics["active_span_coverage"].ge(coverage)
                    & peer_counts.ge(peers)
                )
                rows.append(
                    {
                        "minimum_return_observations": overlap_requirement,
                        "minimum_active_span_coverage": float(coverage),
                        "minimum_overlapping_peers": int(peers),
                        "peer_overlap_requirement": overlap_requirement,
                        "eligible_symbol_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows, columns=ELIGIBILITY_COLUMNS)


def _pair_vector(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype="float64")
    indices = np.triu_indices_from(values, k=1)
    return values[indices]


def compare_correlation_structures(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, float | int | None]:
    """Compare two aligned pairwise correlation structures."""

    if list(primary.index) != list(comparison.index):
        raise ClusteringMethodologyError("Correlation structures are not aligned")
    left = _pair_vector(primary)
    right = _pair_vector(comparison)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return {
            "shared_finite_pairs": 0,
            "mean_absolute_difference": None,
            "median_absolute_difference": None,
            "p90_absolute_difference": None,
            "structure_pearson_correlation": None,
        }
    differences = np.abs(left[valid] - right[valid])
    structural = (
        float(np.corrcoef(left[valid], right[valid])[0, 1])
        if int(valid.sum()) >= 2
        and np.std(left[valid]) > 0
        and np.std(right[valid]) > 0
        else None
    )
    return {
        "shared_finite_pairs": int(valid.sum()),
        "mean_absolute_difference": float(differences.mean()),
        "median_absolute_difference": float(np.median(differences)),
        "p90_absolute_difference": float(np.quantile(differences, 0.90)),
        "structure_pearson_correlation": structural,
    }


def _cluster_labels(
    distance: pd.DataFrame,
    *,
    linkage: str,
    cluster_count: int,
) -> np.ndarray:
    if linkage == "ward":
        raise ClusteringMethodologyError(
            "Ward linkage is not valid for an arbitrary precomputed distance matrix"
        )
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


def evaluate_cluster_counts(
    correlation: pd.DataFrame,
    distance: pd.DataFrame,
    *,
    sector_by_symbol: Mapping[str, str] | None = None,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
    linkages: Sequence[str] = ("average", "complete"),
) -> pd.DataFrame:
    """Evaluate non-final cluster-count diagnostics without retaining assignments."""

    if list(correlation.index) != list(distance.index):
        raise ClusteringMethodologyError("Correlation/distance symbols differ")
    rows: list[dict[str, object]] = []
    correlation_values = correlation.to_numpy(dtype="float64")
    upper = np.triu_indices_from(correlation_values, k=1)
    pair_correlations = correlation_values[upper]
    for linkage in linkages:
        for requested in cluster_counts:
            if requested < 2 or requested >= len(distance):
                rows.append(
                    {
                        "linkage": linkage,
                        "requested_clusters": int(requested),
                        "actual_clusters": 0,
                        "symbol_count": len(distance),
                        "silhouette": np.nan,
                        "mean_within_cluster_correlation": np.nan,
                        "mean_between_cluster_correlation": np.nan,
                        "cohesion_separation_gap": np.nan,
                        "minimum_cluster_size": 0,
                        "maximum_cluster_size": 0,
                        "cluster_size_coefficient_of_variation": np.nan,
                        "singleton_cluster_count": 0,
                        "tiny_cluster_count": 0,
                        "sector_nmi_posthoc": np.nan,
                        "status": "insufficient_symbols",
                    }
                )
                continue
            labels = _cluster_labels(
                distance, linkage=linkage, cluster_count=int(requested)
            )
            label_pairs_equal = labels[upper[0]] == labels[upper[1]]
            within = pair_correlations[label_pairs_equal]
            between = pair_correlations[~label_pairs_equal]
            sizes = pd.Series(labels).value_counts().sort_index().to_numpy()
            sector_nmi = np.nan
            if sector_by_symbol is not None:
                sectors = [
                    sector_by_symbol.get(str(symbol), "UNKNOWN")
                    for symbol in distance.index
                ]
                sector_nmi = float(normalized_mutual_info_score(sectors, labels))
            mean_within = float(within.mean()) if len(within) else np.nan
            mean_between = float(between.mean()) if len(between) else np.nan
            rows.append(
                {
                    "linkage": linkage,
                    "requested_clusters": int(requested),
                    "actual_clusters": int(len(np.unique(labels))),
                    "symbol_count": len(distance),
                    "silhouette": float(
                        silhouette_score(
                            distance.to_numpy(dtype="float64"),
                            labels,
                            metric="precomputed",
                        )
                    ),
                    "mean_within_cluster_correlation": mean_within,
                    "mean_between_cluster_correlation": mean_between,
                    "cohesion_separation_gap": mean_within - mean_between,
                    "minimum_cluster_size": int(sizes.min()),
                    "maximum_cluster_size": int(sizes.max()),
                    "cluster_size_coefficient_of_variation": float(
                        sizes.std(ddof=0) / sizes.mean()
                    ),
                    "singleton_cluster_count": int((sizes == 1).sum()),
                    "tiny_cluster_count": int((sizes < 3).sum()),
                    "sector_nmi_posthoc": sector_nmi,
                    "status": "completed_non_final_diagnostic",
                }
            )
    return pd.DataFrame(rows, columns=CLUSTER_DIAGNOSTIC_COLUMNS)


def evaluate_temporal_stability(
    early_correlation: pd.DataFrame,
    late_correlation: pd.DataFrame,
    *,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
    linkages: Sequence[str] = ("average", "complete"),
) -> pd.DataFrame:
    """Compare two temporal-window clusterings on an identical symbol set."""

    if list(early_correlation.index) != list(late_correlation.index):
        raise ClusteringMethodologyError(
            "Temporal stability correlations must use identical symbols"
        )
    early_distance = correlation_to_distance(early_correlation)
    late_distance = correlation_to_distance(late_correlation)
    rows: list[dict[str, object]] = []
    for linkage in linkages:
        for requested in cluster_counts:
            if requested < 2 or requested >= len(early_distance):
                rows.append(
                    {
                        "linkage": linkage,
                        "requested_clusters": int(requested),
                        "actual_early_clusters": 0,
                        "actual_late_clusters": 0,
                        "common_symbol_count": len(early_distance),
                        "adjusted_rand_index": np.nan,
                        "normalized_mutual_information": np.nan,
                        "status": "insufficient_symbols",
                    }
                )
                continue
            early_labels = _cluster_labels(
                early_distance, linkage=linkage, cluster_count=int(requested)
            )
            late_labels = _cluster_labels(
                late_distance, linkage=linkage, cluster_count=int(requested)
            )
            rows.append(
                {
                    "linkage": linkage,
                    "requested_clusters": int(requested),
                    "actual_early_clusters": int(len(np.unique(early_labels))),
                    "actual_late_clusters": int(len(np.unique(late_labels))),
                    "common_symbol_count": len(early_distance),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(early_labels, late_labels)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(early_labels, late_labels)
                    ),
                    "status": "completed_non_final_diagnostic",
                }
            )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    labels = ("min", "p10", "p25", "p50", "p75", "p90", "max")
    probabilities = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {label: None for label in labels}
    measured = numeric.quantile(probabilities, interpolation="linear")
    return {
        label: float(measured.iloc[index])
        for index, label in enumerate(labels)
    }


def _finite_pair_count(correlation: pd.DataFrame) -> int:
    return int(np.isfinite(_pair_vector(correlation)).sum())


def _temporal_window_correlations(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    training_dates: Sequence[pd.Timestamp],
    *,
    minimum_overlap: int,
    window_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    if not 0.5 < window_fraction < 1.0:
        raise ValueError("window_fraction must be between 0.5 and 1")
    window_length = max(3, int(math.ceil(len(training_dates) * window_fraction)))
    early_dates = tuple(training_dates[:window_length])
    late_dates = tuple(training_dates[-window_length:])

    def correlation_for(
        dates: tuple[pd.Timestamp, ...],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        subset = training_market.loc[
            training_market["market_date"].isin(set(dates))
        ].copy(deep=True)
        built = construct_close_returns(
            subset,
            symbols,
            global_market_dates=dates,
        )
        matrix = build_return_matrix(
            built.returns, value_column="log_return", symbols=symbols
        )
        overlap = pairwise_overlap_counts(matrix)
        correlation = minimum_overlap_correlation(
            matrix,
            method=PRIMARY_CORRELATION_METHOD,
            minimum_overlap=minimum_overlap,
            overlap_counts=overlap,
        )
        return correlation, overlap

    early_correlation, early_overlap = correlation_for(early_dates)
    late_correlation, late_overlap = correlation_for(late_dates)
    jointly_valid = (
        early_correlation.notna()
        & late_correlation.notna()
        & early_overlap.ge(minimum_overlap)
        & late_overlap.ge(minimum_overlap)
    )
    early_counts = pd.Series(np.diag(early_overlap), index=early_overlap.index)
    late_counts = pd.Series(np.diag(late_overlap), index=late_overlap.index)
    candidates = sorted(
        set(early_counts.index[early_counts >= minimum_overlap]).intersection(
            late_counts.index[late_counts >= minimum_overlap]
        )
    )
    core = deterministic_complete_pair_core(jointly_valid, candidates=candidates)
    return (
        early_correlation.loc[list(core), list(core)],
        late_correlation.loc[list(core), list(core)],
        core,
    )


def run_clustering_methodology_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
    exploratory_minimum_overlap: int = EXPLORATORY_MINIMUM_OVERLAP,
    overlap_thresholds: Sequence[int] = OVERLAP_THRESHOLDS,
    candidate_cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
    stability_window_fraction: float = STABILITY_WINDOW_FRACTION,
) -> ClusteringMethodologyResult:
    """Run empirical TRAIN-only methodology diagnostics on the frozen universe."""

    if exploratory_minimum_overlap < 2:
        raise ValueError("exploratory_minimum_overlap must be at least 2")
    equity: EquityUniverseResult = run_equity_universe(
        parquet_path=parquet_path,
        registry_path=registry_path,
        listing_snapshot_path=listing_snapshot_path,
    )
    resolved_parquet = resolve_market_parquet_path(parquet_path)
    market = load_market_data(resolved_parquet)
    partitions = canonical_temporal_date_partitions(market["market_date"])
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
    overlap = pairwise_overlap_counts(log_matrix)
    overlap_pairs = pairwise_overlap_table(log_matrix, overlap)
    symbol_diagnostics = build_training_symbol_diagnostics(
        training_market, symbols, log_matrix
    )
    eligibility = build_eligibility_sensitivity(
        symbol_diagnostics,
        overlap,
        overlap_thresholds=overlap_thresholds,
    )
    pearson = minimum_overlap_correlation(
        log_matrix,
        method="pearson",
        minimum_overlap=exploratory_minimum_overlap,
        overlap_counts=overlap,
    )
    spearman = minimum_overlap_correlation(
        log_matrix,
        method="spearman",
        minimum_overlap=exploratory_minimum_overlap,
        overlap_counts=overlap,
    )
    simple_pearson = minimum_overlap_correlation(
        simple_matrix,
        method="pearson",
        minimum_overlap=exploratory_minimum_overlap,
        overlap_counts=overlap,
    )
    return_counts = log_matrix.notna().sum(axis=0)
    core_candidates = tuple(
        sorted(return_counts.index[return_counts >= exploratory_minimum_overlap])
    )
    valid_pairs = pearson.notna() & overlap.ge(exploratory_minimum_overlap)
    core = deterministic_complete_pair_core(
        valid_pairs, candidates=core_candidates
    )
    if len(core) < 3:
        raise ClusteringMethodologyError(
            "Exploratory complete-overlap core has fewer than three symbols"
        )
    core_correlation = pearson.loc[list(core), list(core)]
    core_distance = correlation_to_distance(core_correlation)
    sector_map = equity.records.set_index("symbol")["sector"].astype(str).to_dict()
    cluster_diagnostics = evaluate_cluster_counts(
        core_correlation,
        core_distance,
        sector_by_symbol=sector_map,
        cluster_counts=candidate_cluster_counts,
    )
    early_correlation, late_correlation, stability_core = (
        _temporal_window_correlations(
            training_market,
            symbols,
            partitions.training_dates,
            minimum_overlap=exploratory_minimum_overlap,
            window_fraction=stability_window_fraction,
        )
    )
    if len(stability_core) < 3:
        raise ClusteringMethodologyError(
            "Temporal stability core has fewer than three symbols"
        )
    stability = evaluate_temporal_stability(
        early_correlation,
        late_correlation,
        cluster_counts=candidate_cluster_counts,
    )
    total_pairs = len(overlap_pairs)
    pearson_finite = _finite_pair_count(pearson)
    spearman_finite = _finite_pair_count(spearman)
    summary = ClusteringMethodologySummary(
        methodology_version=CLUSTERING_METHODOLOGY_VERSION,
        identity_universe_count=len(symbols),
        identity_universe_hash=equity.summary.universe_hash,
        training_date_count=len(partitions.training_dates),
        training_date_range=(partitions.training_start, partitions.training_end),
        validation_date_count=len(partitions.validation_dates),
        validation_date_range=(
            partitions.validation_start,
            partitions.validation_end,
        ),
        test_date_count=len(partitions.test_dates),
        test_date_range=(partitions.test_start, partitions.test_end),
        return_representation=PRIMARY_RETURN_REPRESENTATION,
        valid_return_rows=returns.valid_return_rows,
        symbols_with_training_returns=returns.symbols_with_returns,
        first_observation_rows_rejected=returns.first_observation_rows_rejected,
        gap_spanning_rows_rejected=returns.gap_spanning_rows_rejected,
        overlap_pair_count=total_pairs,
        overlap_count_quantiles=_quantiles(
            overlap_pairs["shared_return_observations"]
        ),
        overlap_ratio_quantiles=_quantiles(
            overlap_pairs["overlap_ratio_min_history"]
        ),
        overlap_threshold_pair_counts={
            str(threshold): int(
                overlap_pairs["shared_return_observations"].ge(threshold).sum()
            )
            for threshold in overlap_thresholds
        },
        exploratory_minimum_overlap=exploratory_minimum_overlap,
        finite_pearson_pair_count=pearson_finite,
        finite_spearman_pair_count=spearman_finite,
        missing_pair_count_at_exploratory_overlap=total_pairs - pearson_finite,
        pearson_vs_spearman=compare_correlation_structures(pearson, spearman),
        simple_vs_log_pearson=compare_correlation_structures(
            simple_pearson, pearson
        ),
        complete_overlap_core_count=len(core),
        complete_overlap_core_symbols=core,
        stability_common_symbol_count=len(stability_core),
        test_returns_used=False,
        final_clusters_frozen=False,
    )
    return ClusteringMethodologyResult(
        summary=summary,
        overlap_pairs=overlap_pairs,
        eligibility_sensitivity=eligibility,
        cluster_count_diagnostics=cluster_diagnostics,
        stability_diagnostics=stability,
        parquet_path=resolved_parquet,
    )


def _print_eligibility(sensitivity: pd.DataFrame) -> None:
    print("Eligibility sensitivity (peer minimum 10 shown; no liquidity filter):")
    subset = sensitivity.loc[sensitivity["minimum_overlapping_peers"] == 10]
    for overlap in sorted(subset["minimum_return_observations"].unique()):
        selected = subset.loc[subset["minimum_return_observations"] == overlap]
        values = ", ".join(
            f"coverage>={row.minimum_active_span_coverage:.0%}: "
            f"{int(row.eligible_symbol_count)}"
            for row in selected.itertuples(index=False)
        )
        print(f"  returns/peer-overlap >= {int(overlap)}: {values}")


def _print_cluster_diagnostics(frame: pd.DataFrame) -> None:
    print("Exploratory cluster-count diagnostics (not final assignments):")
    for row in frame.itertuples(index=False):
        if row.status != "completed_non_final_diagnostic":
            continue
        print(
            f"  {row.linkage} k={row.requested_clusters}: "
            f"silhouette={row.silhouette:.4f}, "
            f"within={row.mean_within_cluster_correlation:.4f}, "
            f"between={row.mean_between_cluster_correlation:.4f}, "
            f"size={row.minimum_cluster_size}-{row.maximum_cluster_size}, "
            f"tiny={row.tiny_cluster_count}, sector_nmi={row.sector_nmi_posthoc:.4f}"
        )


def _print_stability(frame: pd.DataFrame) -> None:
    print("Temporal stability diagnostics (TRAIN subwindows only):")
    for row in frame.itertuples(index=False):
        if row.status != "completed_non_final_diagnostic":
            continue
        print(
            f"  {row.linkage} k={row.requested_clusters}: "
            f"ARI={row.adjusted_rand_index:.4f}, "
            f"NMI={row.normalized_mutual_information:.4f}, "
            f"symbols={row.common_symbol_count}"
        )


def _print_summary(result: ClusteringMethodologyResult) -> None:
    summary = result.summary
    print(f"Methodology version: {summary.methodology_version}")
    print(f"Identity universe: {summary.identity_universe_count}")
    print(f"Identity hash: {summary.identity_universe_hash}")
    print(
        f"TRAIN dates: {summary.training_date_count} "
        f"({summary.training_date_range[0]} to {summary.training_date_range[1]})"
    )
    print(
        f"VALIDATION dates (unused): {summary.validation_date_count} "
        f"({summary.validation_date_range[0]} to {summary.validation_date_range[1]})"
    )
    print(
        f"TEST dates (sealed): {summary.test_date_count} "
        f"({summary.test_date_range[0]} to {summary.test_date_range[1]})"
    )
    print(f"Valid strict daily log returns: {summary.valid_return_rows}")
    print(f"Symbols with TRAIN returns: {summary.symbols_with_training_returns}")
    print(f"Gap-spanning rows rejected: {summary.gap_spanning_rows_rejected}")
    print(
        "Overlap-count quantiles: "
        + json.dumps(dict(summary.overlap_count_quantiles))
    )
    print(
        "Overlap-ratio quantiles: "
        + json.dumps(dict(summary.overlap_ratio_quantiles))
    )
    print(
        "Pairs satisfying overlap candidates: "
        + json.dumps(dict(summary.overlap_threshold_pair_counts))
    )
    print(
        "Pearson vs Spearman: "
        + json.dumps(dict(summary.pearson_vs_spearman))
    )
    print(
        "Simple vs log Pearson: "
        + json.dumps(dict(summary.simple_vs_log_pearson))
    )
    print(
        f"Exploratory complete-overlap core: "
        f"{summary.complete_overlap_core_count}"
    )
    print(
        f"Temporal stability common core: "
        f"{summary.stability_common_symbol_count}"
    )
    print(f"TEST returns used: {summary.test_returns_used}")
    print(f"Final clusters frozen: {summary.final_clusters_frozen}")
    _print_eligibility(result.eligibility_sensitivity)
    _print_cluster_diagnostics(result.cluster_count_diagnostics)
    _print_stability(result.stability_diagnostics)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TRAIN-only non-final Phase 1 clustering diagnostics."
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
        result = run_clustering_methodology_audit(
            parquet_path=args.path,
            registry_path=args.company_registry,
            listing_snapshot_path=args.listing_snapshot,
        )
        _print_summary(result)
        return 0
    except (
        ClusteringMethodologyError,
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
    "ACTIVE_COVERAGE_THRESHOLDS",
    "CANDIDATE_CLUSTER_COUNTS",
    "CLUSTERING_METHODOLOGY_VERSION",
    "ClusteringMethodologyError",
    "ClusteringMethodologyResult",
    "ClusteringMethodologySummary",
    "EXPLORATORY_MINIMUM_OVERLAP",
    "MINIMUM_OVERLAPPING_PEERS",
    "NonPositiveCloseError",
    "OVERLAP_THRESHOLDS",
    "PRIMARY_CLUSTERING_METHOD",
    "PRIMARY_CORRELATION_METHOD",
    "PRIMARY_DISTANCE_TRANSFORM",
    "PRIMARY_RETURN_REPRESENTATION",
    "ROBUSTNESS_CORRELATION_METHOD",
    "ReturnConstructionResult",
    "STABILITY_WINDOW_FRACTION",
    "TemporalDatePartitions",
    "build_eligibility_sensitivity",
    "build_return_matrix",
    "build_training_symbol_diagnostics",
    "canonical_temporal_date_partitions",
    "compare_correlation_structures",
    "construct_close_returns",
    "correlation_to_distance",
    "deterministic_complete_pair_core",
    "evaluate_cluster_counts",
    "evaluate_temporal_stability",
    "main",
    "minimum_overlap_correlation",
    "pairwise_overlap_counts",
    "pairwise_overlap_table",
    "run_clustering_methodology_audit",
    "training_market_view",
)
