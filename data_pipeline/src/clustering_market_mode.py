"""TRAIN-only market-mode and correlation-noise audit for Phase 1 clusters.

The experiment family is deliberately limited to raw returns, one static
single-factor residualization, Ledoit-Wolf shrinkage when a strict complete
matrix exists, and their combination.  It never writes cluster assignments.
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
from sklearn.covariance import LedoitWolf
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .clustering_methodology import (
    CANDIDATE_CLUSTER_COUNTS,
    ClusteringMethodologyError,
    TemporalDatePartitions,
    build_return_matrix,
    build_training_symbol_diagnostics,
    construct_close_returns,
    correlation_to_distance,
    deterministic_complete_pair_core,
    minimum_overlap_correlation,
    pairwise_overlap_counts,
    training_market_view,
)
from .clustering_protocol import (
    ClusteringProtocolError,
    TemporalWindows,
    cluster_quality_diagnostics,
    deterministic_temporal_windows,
    eligible_symbols_for_overlap_floor,
    hierarchical_labels,
)
from .config import (
    COMPANY_REGISTRY_PATH,
    CURRENT_LISTINGS_PATH,
    INDICES_MASTER_PATH,
)
from .equity_universe import (
    ALLOWED_COMMON_EQUITY_SECURITY_TYPES,
    EquityUniverseError,
)
from .instrument_audit import load_registry_evidence
from .official_listings import ListingsUnavailableError, load_listing_snapshot
from .parquet_market_data import (
    MarketParquetError,
    load_market_calendar,
    load_market_data,
    resolve_market_parquet_path,
)
from .universe_audit import UniverseAuditError
from .universe_methodology import UniverseMethodologyError


MARKET_MODE_AUDIT_VERSION = "phase1_market_mode_noise_audit_v1"
FINAL_DECISION = "BLOCKED_WEAK_CLUSTER_STRUCTURE"
REFERENCE_OVERLAP_FLOOR = 120
REFERENCE_ACTIVE_SPAN_COVERAGE = 0.50
REFERENCE_MINIMUM_PEERS = 20
REGRESSION_MINIMUM_OVERLAP = 120
MINIMUM_CROSS_SECTIONAL_CONSTITUENTS = 20
MINIMUM_EXTERNAL_INDEX_COVERAGE = 0.90
MINIMUM_SHRINKAGE_COMMON_DATES = 120
REFERENCE_LINKAGE = "complete"
REFERENCE_K = 15
# Reuse the canonical 7C.2 split verbatim.  This prevents a later-partition
# read merely to rediscover boundaries that the preceding milestone froze.
FROZEN_TRAIN_START = "2016-07-26"
FROZEN_TRAIN_END = "2023-08-03"
FROZEN_VALIDATION_START = "2023-08-04"
FROZEN_VALIDATION_END = "2025-02-04"
FROZEN_TEST_START = "2025-02-06"
FROZEN_TEST_END = "2026-08-20"

RAW_REPRESENTATION = "raw_log_pearson"
RESIDUAL_REPRESENTATION = "equity_market_residual_log_pearson"
SHRINKAGE_REPRESENTATION = "ledoit_wolf_raw"
COMBINED_REPRESENTATION = "equity_market_residual_ledoit_wolf"

STRUCTURE_COLUMNS = (
    "representation",
    "symbol_count",
    "finite_pair_count",
    "mean_pairwise_correlation",
    "minimum_correlation",
    "p10_correlation",
    "p25_correlation",
    "median_pairwise_correlation",
    "p75_correlation",
    "p90_correlation",
    "maximum_correlation",
    "largest_eigenvalue",
    "largest_eigenvalue_share_of_trace",
    "negative_eigenvalue_count",
    "eigen_diagnostic_note",
)

CLUSTER_COLUMNS = (
    "representation",
    "requested_clusters",
    "actual_clusters",
    "clustered_symbol_count",
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
    "representation",
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
    "early_tiny_cluster_count",
    "late_tiny_cluster_count",
    "early_cluster_size_cv",
    "late_cluster_size_cv",
    "status",
)

ROBUSTNESS_COLUMNS = (
    "representation",
    "variant",
    "requested_clusters",
    "common_symbol_count",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "status",
)


class MarketModeAuditError(RuntimeError):
    """Raised when the constrained market-mode audit cannot proceed safely."""


@dataclass(frozen=True)
class IndexFactorEvidence:
    index_code: str
    index_name: str | None
    path: Path
    source: str | None
    fetched_at: str | None
    level_rows_in_train: int
    return_rows_in_train: int
    expected_train_return_dates: int
    missing_train_return_dates: int
    train_return_coverage: float
    first_level_date: str | None
    last_level_date: str | None
    first_return_date: str | None
    last_return_date: str | None
    adequate_for_full_train_static_factor: bool
    reason: str
    returns: pd.Series = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("returns", None)
        payload["path"] = str(self.path)
        return payload


@dataclass(frozen=True)
class CrossSectionalFactorResult:
    factor: pd.Series = field(repr=False, compare=False)
    constituent_counts: pd.Series = field(repr=False, compare=False)
    eligible_symbol_count: int
    minimum_constituents: int
    available_date_count: int
    possible_date_count: int
    coverage: float
    minimum_constituents_observed: int
    median_constituents_observed: float
    maximum_constituents_observed: int
    provenance: str
    limitation: str


@dataclass(frozen=True)
class ResidualizationResult:
    residuals: pd.DataFrame = field(repr=False, compare=False)
    coefficients: pd.DataFrame = field(repr=False, compare=False)
    factor_name: str
    minimum_overlap: int
    fitted_symbol_count: int
    insufficient_symbol_count: int


@dataclass(frozen=True)
class ShrinkageAssessment:
    representation: str
    requested_symbol_count: int
    common_date_count: int
    union_date_count: int
    common_date_retention: float
    minimum_common_dates: int
    feasible: bool
    reason: str
    shrinkage: float | None
    correlation: pd.DataFrame | None = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("correlation", None)
        return payload


@dataclass(frozen=True)
class MarketModeAuditSummary:
    audit_version: str
    final_decision: str
    identity_universe_count: int
    train_return_capable_count: int
    eligible_symbol_count: int
    common_complete_core_count: int
    overlap_floor: int
    training_date_range: tuple[str, str]
    validation_date_range: tuple[str, str]
    test_date_range: tuple[str, str]
    temporal_window_ranges: tuple[tuple[str, str], tuple[str, str]]
    external_index_adequate: bool
    selected_factor: str
    shrinkage_feasible: bool
    combined_feasible: bool
    reference_k: int
    final_assignments_written: bool
    validation_returns_used_for_fitting: bool
    validation_observation_values_loaded: bool
    test_returns_used: bool
    test_observation_values_loaded: bool
    full_calendar_dates_loaded_for_partition_boundaries: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarketModeAuditResult:
    summary: MarketModeAuditSummary
    index_factor: IndexFactorEvidence
    cross_sectional_factor: CrossSectionalFactorResult
    beta_distribution: Mapping[str, float | int | None]
    correlation_structure: pd.DataFrame = field(repr=False, compare=False)
    shrinkage_assessments: tuple[ShrinkageAssessment, ...]
    clustering_evidence: pd.DataFrame = field(repr=False, compare=False)
    temporal_stability: pd.DataFrame = field(repr=False, compare=False)
    robustness: pd.DataFrame = field(repr=False, compare=False)
    parquet_path: Path


def load_authoritative_current_equity_identity(
    *,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
) -> pd.DataFrame:
    """Load the frozen current common-equity identity without market history.

    The identity policy is the same official-listing-backed rule used by the
    frozen Milestone 7B.3 universe.  Reading identity independently lets this
    audit predicate-push the subsequent market load to TRAIN only.
    """

    registry = load_registry_evidence(registry_path)
    if registry is None:
        raise MarketModeAuditError(
            f"Company registry evidence is unavailable: {registry_path}"
        )
    listings = load_listing_snapshot(Path(listing_snapshot_path))
    required = {
        "symbol",
        "company_name",
        "security_type",
        "sector",
        "officially_listed",
        "source",
        "snapshot_date",
    }
    missing = sorted(required.difference(listings.columns))
    if missing:
        raise MarketModeAuditError(
            "Official listings are missing identity fields: " + ", ".join(missing)
        )
    evidence = listings.loc[:, sorted(required)].copy(deep=True)
    evidence["symbol"] = (
        evidence["symbol"].astype("string").str.strip().str.upper()
    )
    evidence["security_type"] = (
        evidence["security_type"].astype("string").str.strip().str.lower()
    )
    evidence["sector"] = evidence["sector"].astype("string").str.strip()
    listed = evidence["officially_listed"]
    if not pd.api.types.is_bool_dtype(listed):
        listed = listed.astype("string").str.strip().str.lower().isin({"true", "1"})
    if not listed.all():
        raise MarketModeAuditError(
            "Current official listing snapshot contains a non-listed record"
        )
    upper_sector = evidence["sector"].str.upper()
    non_common_sector = (
        upper_sector.str.contains("FUND", regex=False)
        | upper_sector.str.contains("REAL ESTATE INVESTMENT TRUST", regex=False)
        | upper_sector.str.contains("REIT", regex=False)
        | upper_sector.eq("MODARABAS")
    )
    selected = evidence.loc[
        evidence["security_type"].isin(ALLOWED_COMMON_EQUITY_SECURITY_TYPES)
        & ~non_common_sector
    ].copy(deep=True)
    if selected.empty or selected["symbol"].duplicated().any():
        raise MarketModeAuditError(
            "Authoritative current common-equity identity is empty or duplicated"
        )

    registry_check = registry.loc[
        :, ["symbol", "security_type", "sector", "officially_listed", "source"]
    ].copy(deep=True)
    registry_check["symbol"] = (
        registry_check["symbol"].astype("string").str.strip().str.upper()
    )
    registry_check = registry_check.rename(
        columns={
            column: f"registry_{column}"
            for column in registry_check
            if column != "symbol"
        }
    )
    selected = selected.merge(
        registry_check,
        on="symbol",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    invalid_registry = selected["_merge"].ne("both")
    registry_listed = selected["registry_officially_listed"]
    if not pd.api.types.is_bool_dtype(registry_listed):
        registry_listed = (
            registry_listed.astype("string")
            .str.strip()
            .str.lower()
            .isin({"true", "1"})
        )
    invalid_registry |= ~registry_listed.fillna(False).astype(bool)
    invalid_registry |= (
        selected["registry_security_type"].astype("string").str.lower()
        != selected["security_type"]
    )
    invalid_registry |= (
        selected["registry_sector"].astype("string").str.strip()
        != selected["sector"]
    )
    invalid_registry |= (
        selected["registry_source"].astype("string").str.strip()
        != selected["source"].astype("string").str.strip()
    )
    if invalid_registry.any():
        symbols = sorted(selected.loc[invalid_registry, "symbol"].astype(str))
        raise MarketModeAuditError(
            "Frozen identity conflicts with company-registry evidence: "
            + ", ".join(symbols)
        )
    return selected.loc[
        :,
        [
            "symbol",
            "company_name",
            "sector",
            "security_type",
            "source",
            "snapshot_date",
        ],
    ].sort_values("symbol", kind="mergesort").reset_index(drop=True)


def load_train_only_market_values(
    parquet_path: str | os.PathLike[str],
    identity_symbols: Sequence[str],
    *,
    training_start: str = FROZEN_TRAIN_START,
    training_end: str = FROZEN_TRAIN_END,
) -> tuple[TemporalDatePartitions, pd.DataFrame]:
    """Load only the already-frozen 7C.2 TRAIN calendar and observations."""

    calendar = load_market_calendar(
        parquet_path,
        start_date=training_start,
        end_date=training_end,
    )
    if calendar.empty:
        raise MarketModeAuditError("Frozen TRAIN calendar is empty")
    actual_start = calendar[0].date().isoformat()
    actual_end = calendar[-1].date().isoformat()
    if (actual_start, actual_end) != (training_start, training_end):
        raise MarketModeAuditError(
            "Frozen TRAIN boundary is absent from the source calendar: "
            f"expected {training_start}..{training_end}, "
            f"found {actual_start}..{actual_end}"
        )
    partitions = TemporalDatePartitions(
        training_dates=tuple(calendar),
        validation_dates=(),
        test_dates=(),
        training_start=training_start,
        training_end=training_end,
        validation_start=FROZEN_VALIDATION_START,
        validation_end=FROZEN_VALIDATION_END,
        test_start=FROZEN_TEST_START,
        test_end=FROZEN_TEST_END,
    )
    market = load_market_data(
        parquet_path,
        start_date=training_start,
        end_date=training_end,
        symbols=tuple(sorted(map(str, identity_symbols))),
    )
    market_dates = pd.to_datetime(market["market_date"], errors="coerce")
    if market_dates.isna().any():
        raise MarketModeAuditError("TRAIN-only market load contains invalid dates")
    if not market.empty and market_dates.max() > pd.Timestamp(partitions.training_end):
        raise MarketModeAuditError(
            "Predicate-pushed market load crossed the TRAIN boundary"
        )
    return partitions, market


def audit_local_kse100_factor(
    index_path: str | os.PathLike[str],
    training_dates: Sequence[object] | pd.Series | pd.Index,
    *,
    minimum_coverage: float = MINIMUM_EXTERNAL_INDEX_COVERAGE,
) -> IndexFactorEvidence:
    """Audit locally persisted official KSE-100 history without downloading."""

    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    path = Path(index_path).expanduser().resolve()
    dates = pd.DatetimeIndex(
        pd.to_datetime(list(training_dates), errors="coerce")
    )
    if dates.isna().any():
        raise MarketModeAuditError("TRAIN date set contains invalid values")
    dates = dates.unique().sort_values()
    if len(dates) < 2:
        raise MarketModeAuditError("At least two TRAIN dates are required")
    if not path.is_file():
        return IndexFactorEvidence(
            index_code="KSE100",
            index_name=None,
            path=path,
            source=None,
            fetched_at=None,
            level_rows_in_train=0,
            return_rows_in_train=0,
            expected_train_return_dates=len(dates) - 1,
            missing_train_return_dates=len(dates) - 1,
            train_return_coverage=0.0,
            first_level_date=None,
            last_level_date=None,
            first_return_date=None,
            last_return_date=None,
            adequate_for_full_train_static_factor=False,
            reason="local_index_file_missing",
            returns=pd.Series(dtype="float64", name="kse100_log_return"),
        )
    frame = pd.read_csv(path, dtype={"index_code": "string"})
    required = {"index_code", "date", "value"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise MarketModeAuditError(
            "Index file is missing required columns: "
            + ", ".join(missing_columns)
        )
    frame = frame.loc[frame["index_code"].eq("KSE100")].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["date"].isna().any() or frame["value"].isna().any():
        raise MarketModeAuditError("KSE100 history contains invalid dates/values")
    if (frame["value"] <= 0).any():
        raise MarketModeAuditError("KSE100 factor requires positive index levels")
    if frame["date"].duplicated().any():
        raise MarketModeAuditError("KSE100 history contains duplicate dates")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    frame = frame.loc[frame["date"].isin(set(dates))].reset_index(drop=True)
    date_rank = pd.Series(np.arange(len(dates), dtype="int64"), index=dates)
    frame["date_rank"] = frame["date"].map(date_rank)
    frame["previous_rank"] = frame["date_rank"].shift(1)
    frame["previous_value"] = frame["value"].shift(1)
    valid = frame["date_rank"].eq(frame["previous_rank"] + 1)
    returns = pd.Series(
        np.log(
            frame.loc[valid, "value"].to_numpy(dtype="float64")
            / frame.loc[valid, "previous_value"].to_numpy(dtype="float64")
        ),
        index=pd.DatetimeIndex(frame.loc[valid, "date"]),
        name="kse100_log_return",
        dtype="float64",
    ).sort_index()
    expected = len(dates) - 1
    coverage = len(returns) / expected
    adequate = coverage >= minimum_coverage

    def first_text(column: str) -> str | None:
        if column not in frame or frame[column].dropna().empty:
            return None
        return str(frame[column].dropna().iloc[0])

    def date_text(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).date().isoformat()

    reason = (
        "adequate_authoritative_train_coverage"
        if adequate
        else "authoritative_index_begins_too_late_for_full_train_static_factor"
    )
    return IndexFactorEvidence(
        index_code="KSE100",
        index_name=first_text("index_name"),
        path=path,
        source=first_text("source"),
        fetched_at=first_text("fetched_at"),
        level_rows_in_train=len(frame),
        return_rows_in_train=len(returns),
        expected_train_return_dates=expected,
        missing_train_return_dates=expected - len(returns),
        train_return_coverage=coverage,
        first_level_date=date_text(frame["date"].min() if not frame.empty else None),
        last_level_date=date_text(frame["date"].max() if not frame.empty else None),
        first_return_date=date_text(returns.index.min() if not returns.empty else None),
        last_return_date=date_text(returns.index.max() if not returns.empty else None),
        adequate_for_full_train_static_factor=adequate,
        reason=reason,
        returns=returns,
    )


def build_equal_weight_market_factor(
    return_matrix: pd.DataFrame,
    symbols: Sequence[str],
    *,
    minimum_constituents: int = MINIMUM_CROSS_SECTIONAL_CONSTITUENTS,
) -> CrossSectionalFactorResult:
    """Build one contemporaneous TRAIN-only equal-weight equity factor."""

    if minimum_constituents < 2:
        raise ValueError("minimum_constituents must be at least 2")
    ordered = tuple(sorted(set(map(str, symbols))))
    if not ordered:
        raise MarketModeAuditError("Market factor requires at least one symbol")
    missing = sorted(set(ordered).difference(map(str, return_matrix.columns)))
    if missing:
        raise MarketModeAuditError(
            "Factor symbols are absent from return matrix: " + ", ".join(missing)
        )
    source = return_matrix.loc[:, list(ordered)]
    counts = source.notna().sum(axis=1).astype("int64")
    factor = source.mean(axis=1, skipna=True).where(
        counts >= minimum_constituents
    )
    factor.name = "equal_weight_current_equity_market_factor"
    available_counts = counts.loc[factor.notna()]
    return CrossSectionalFactorResult(
        factor=factor,
        constituent_counts=counts,
        eligible_symbol_count=len(ordered),
        minimum_constituents=minimum_constituents,
        available_date_count=int(factor.notna().sum()),
        possible_date_count=len(factor),
        coverage=float(factor.notna().mean()) if len(factor) else 0.0,
        minimum_constituents_observed=(
            int(available_counts.min()) if not available_counts.empty else 0
        ),
        median_constituents_observed=(
            float(available_counts.median()) if not available_counts.empty else 0.0
        ),
        maximum_constituents_observed=(
            int(available_counts.max()) if not available_counts.empty else 0
        ),
        provenance=(
            "TRAIN-only equal-weight mean of eligible current "
            "common-equity returns"
        ),
        limitation=(
            "Uses the frozen current common-equity identity universe with variable "
            "observed-date composition; it is survivorship-limited and is not an "
            "externally observed historical PSX market index."
        ),
    )


def residualize_static_market_factor(
    return_matrix: pd.DataFrame,
    factor: pd.Series,
    symbols: Sequence[str],
    *,
    minimum_overlap: int = REGRESSION_MINIMUM_OVERLAP,
    factor_name: str = "market_factor",
) -> ResidualizationResult:
    """Fit static per-stock TRAIN OLS and preserve all missing observations."""

    if minimum_overlap < 3:
        raise ValueError("minimum_overlap must be at least 3")
    ordered = tuple(sorted(set(map(str, symbols))))
    aligned_factor = pd.to_numeric(
        factor.reindex(return_matrix.index), errors="coerce"
    )
    residuals = pd.DataFrame(
        np.nan,
        index=return_matrix.index.copy(),
        columns=return_matrix.columns.copy(),
        dtype="float64",
    )
    rows: list[dict[str, object]] = []
    for symbol in ordered:
        values = pd.to_numeric(return_matrix[symbol], errors="coerce")
        valid = values.notna() & aligned_factor.notna()
        overlap = int(valid.sum())
        if overlap < minimum_overlap:
            rows.append(
                {
                    "symbol": symbol,
                    "overlap_observations": overlap,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "factor_variance": np.nan,
                    "status": "insufficient_factor_overlap",
                }
            )
            continue
        x = aligned_factor.loc[valid].to_numpy(dtype="float64")
        y = values.loc[valid].to_numpy(dtype="float64")
        variance = float(np.var(x, ddof=0))
        if not np.isfinite(variance) or variance <= 0:
            rows.append(
                {
                    "symbol": symbol,
                    "overlap_observations": overlap,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "factor_variance": variance,
                    "status": "constant_or_invalid_factor",
                }
            )
            continue
        design = np.column_stack([np.ones(overlap, dtype="float64"), x])
        alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = alpha + beta * x
        residual = y - fitted
        if not np.isfinite(residual).all():
            raise MarketModeAuditError("Residualization produced NaN/Inf")
        residuals.loc[valid, symbol] = residual
        rows.append(
            {
                "symbol": symbol,
                "overlap_observations": overlap,
                "alpha": float(alpha),
                "beta": float(beta),
                "factor_variance": variance,
                "status": "fitted_train_only_static_ols",
            }
        )
    coefficients = pd.DataFrame(rows).sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
    fitted_count = int(coefficients["status"].eq("fitted_train_only_static_ols").sum())
    return ResidualizationResult(
        residuals=residuals,
        coefficients=coefficients,
        factor_name=factor_name,
        minimum_overlap=minimum_overlap,
        fitted_symbol_count=fitted_count,
        insufficient_symbol_count=len(coefficients) - fitted_count,
    )


def evaluate_ledoit_wolf_shrinkage(
    return_matrix: pd.DataFrame,
    symbols: Sequence[str],
    *,
    representation: str,
    minimum_common_dates: int = MINIMUM_SHRINKAGE_COMMON_DATES,
) -> ShrinkageAssessment:
    """Fit Ledoit-Wolf only on a strict complete matrix; never fill missing."""

    if minimum_common_dates < 2:
        raise ValueError("minimum_common_dates must be at least 2")
    ordered = tuple(sorted(set(map(str, symbols))))
    source = return_matrix.loc[:, list(ordered)]
    source_copy = source.copy(deep=True)
    common = source.dropna(axis=0, how="any")
    union_count = int(source.notna().any(axis=1).sum())
    retention = len(common) / union_count if union_count else 0.0
    if len(common) < minimum_common_dates:
        assessment = ShrinkageAssessment(
            representation=representation,
            requested_symbol_count=len(ordered),
            common_date_count=len(common),
            union_date_count=union_count,
            common_date_retention=retention,
            minimum_common_dates=minimum_common_dates,
            feasible=False,
            reason="insufficient_strict_common_return_dates_without_imputation",
            shrinkage=None,
            correlation=None,
        )
    else:
        estimator = LedoitWolf(assume_centered=False).fit(
            common.to_numpy(dtype="float64")
        )
        covariance = estimator.covariance_
        scale = np.sqrt(np.diag(covariance))
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise MarketModeAuditError("Ledoit-Wolf produced invalid variances")
        correlation_values = covariance / np.outer(scale, scale)
        np.fill_diagonal(correlation_values, 1.0)
        correlation = pd.DataFrame(
            correlation_values, index=ordered, columns=ordered
        )
        assessment = ShrinkageAssessment(
            representation=representation,
            requested_symbol_count=len(ordered),
            common_date_count=len(common),
            union_date_count=union_count,
            common_date_retention=retention,
            minimum_common_dates=minimum_common_dates,
            feasible=True,
            reason="fitted_on_strict_complete_common_return_matrix",
            shrinkage=float(estimator.shrinkage_),
            correlation=correlation,
        )
    pd.testing.assert_frame_equal(source, source_copy)
    return assessment


def correlation_structure_diagnostics(
    representation: str,
    correlation: pd.DataFrame,
) -> dict[str, object]:
    """Measure market-mode magnitude without claiming pairwise matrices are PSD."""

    if correlation.shape[0] != correlation.shape[1]:
        raise MarketModeAuditError("Correlation matrix must be square")
    values = correlation.to_numpy(dtype="float64", copy=True)
    upper = values[np.triu_indices_from(values, k=1)]
    finite = upper[np.isfinite(upper)]
    if not len(finite):
        raise MarketModeAuditError("Correlation matrix has no finite pairs")
    symmetric = (values + values.T) / 2.0
    if not np.isfinite(symmetric).all():
        largest = np.nan
        share = np.nan
        negative = -1
        note = "eigen diagnostic unavailable because matrix is incomplete"
    else:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        largest = float(eigenvalues[-1])
        trace = float(np.trace(symmetric))
        share = largest / trace if trace > 0 else np.nan
        negative = int((eigenvalues < -1e-8).sum())
        note = (
            "Descriptive pairwise-correlation eigenspectrum only; differing "
            "pair samples can produce negative eigenvalues."
        )
    quantiles = np.quantile(finite, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        "representation": representation,
        "symbol_count": len(correlation),
        "finite_pair_count": len(finite),
        "mean_pairwise_correlation": float(finite.mean()),
        "minimum_correlation": float(quantiles[0]),
        "p10_correlation": float(quantiles[1]),
        "p25_correlation": float(quantiles[2]),
        "median_pairwise_correlation": float(quantiles[3]),
        "p75_correlation": float(quantiles[4]),
        "p90_correlation": float(quantiles[5]),
        "maximum_correlation": float(quantiles[6]),
        "largest_eigenvalue": largest,
        "largest_eigenvalue_share_of_trace": share,
        "negative_eigenvalue_count": negative,
        "eigen_diagnostic_note": note,
    }


def _cluster_size_snapshot(labels: np.ndarray) -> dict[str, float | int]:
    sizes = pd.Series(labels).value_counts().sort_index().to_numpy()
    return {
        "largest_cluster_fraction": float(sizes.max() / sizes.sum()),
        "minimum_cluster_size": int(sizes.min()),
        "tiny_cluster_count": int((sizes < 3).sum()),
        "cluster_size_cv": float(sizes.std(ddof=0) / sizes.mean()),
    }


def evaluate_representation_clustering(
    correlations: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    *,
    sector_by_symbol: Mapping[str, str],
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Evaluate complete-linkage clustering on a common fixed symbol set."""

    ordered = list(sorted(map(str, symbols)))
    rows: list[dict[str, object]] = []
    labels: dict[tuple[str, int], np.ndarray] = {}
    for representation, correlation in correlations.items():
        selected = correlation.loc[ordered, ordered]
        if selected.isna().any().any():
            raise MarketModeAuditError(
                f"{representation} is incomplete on the fixed comparison core"
            )
        distance = correlation_to_distance(selected)
        for cluster_count in cluster_counts:
            if cluster_count < 2 or cluster_count >= len(ordered):
                continue
            fitted = hierarchical_labels(
                distance,
                linkage=REFERENCE_LINKAGE,
                cluster_count=int(cluster_count),
            )
            labels[(representation, int(cluster_count))] = fitted
            quality = cluster_quality_diagnostics(
                selected,
                distance,
                fitted,
                sector_by_symbol=sector_by_symbol,
            )
            rows.append(
                {
                    "representation": representation,
                    "requested_clusters": int(cluster_count),
                    "clustered_symbol_count": len(ordered),
                    **quality,
                    "status": "completed_non_final_transformation_diagnostic",
                }
            )
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS), labels


def _window_return_matrix(
    training_market: pd.DataFrame,
    symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
    *,
    value_column: str,
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
        built.returns, value_column=value_column, symbols=symbols
    ).reindex(pd.DatetimeIndex(dates[1:]))
    return subset, matrix


def evaluate_temporal_representations(
    training_market: pd.DataFrame,
    identity_symbols: Sequence[str],
    windows: TemporalWindows,
    *,
    full_core_count: int,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> pd.DataFrame:
    """Refit factors/regressions separately inside two TRAIN-only windows."""

    window_data: dict[str, dict[str, object]] = {}
    for name, dates in (
        ("early", windows.early_dates),
        ("late", windows.late_dates),
    ):
        market, raw = _window_return_matrix(
            training_market,
            identity_symbols,
            dates,
            value_column="log_return",
        )
        overlap = pairwise_overlap_counts(raw)
        diagnostics = build_training_symbol_diagnostics(
            market, identity_symbols, raw
        )
        eligible, _ = eligible_symbols_for_overlap_floor(
            diagnostics,
            overlap,
            overlap_floor=REFERENCE_OVERLAP_FLOOR,
        )
        factor = build_equal_weight_market_factor(raw, eligible)
        residual = residualize_static_market_factor(
            raw,
            factor.factor,
            eligible,
            minimum_overlap=REGRESSION_MINIMUM_OVERLAP,
            factor_name=f"{name}_train_equal_weight_factor",
        )
        window_data[name] = {
            "raw": raw,
            "residual": residual.residuals,
            "overlap": overlap,
            "eligible": eligible,
        }
    common_eligible = tuple(
        sorted(
            set(window_data["early"]["eligible"]).intersection(
                window_data["late"]["eligible"]
            )
        )
    )
    rows: list[dict[str, object]] = []
    for representation, matrix_key in (
        (RAW_REPRESENTATION, "raw"),
        (RESIDUAL_REPRESENTATION, "residual"),
    ):
        correlations: dict[str, pd.DataFrame] = {}
        for name in ("early", "late"):
            matrix = window_data[name][matrix_key]
            overlap = window_data[name]["overlap"]
            correlations[name] = minimum_overlap_correlation(
                matrix,
                method="pearson",
                minimum_overlap=REFERENCE_OVERLAP_FLOOR,
                overlap_counts=overlap,
            )
        joint_valid = (
            correlations["early"].notna() & correlations["late"].notna()
        )
        core = deterministic_complete_pair_core(
            joint_valid, candidates=common_eligible
        )
        ordered = list(core)
        early_distance = correlation_to_distance(
            correlations["early"].loc[ordered, ordered]
        )
        late_distance = correlation_to_distance(
            correlations["late"].loc[ordered, ordered]
        )
        for cluster_count in cluster_counts:
            early_labels = hierarchical_labels(
                early_distance,
                linkage=REFERENCE_LINKAGE,
                cluster_count=int(cluster_count),
            )
            late_labels = hierarchical_labels(
                late_distance,
                linkage=REFERENCE_LINKAGE,
                cluster_count=int(cluster_count),
            )
            early_snapshot = _cluster_size_snapshot(early_labels)
            late_snapshot = _cluster_size_snapshot(late_labels)
            rows.append(
                {
                    "representation": representation,
                    "requested_clusters": int(cluster_count),
                    "temporal_common_symbol_count": len(core),
                    "temporal_symbol_retention": len(core) / full_core_count,
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
                    "early_tiny_cluster_count": early_snapshot[
                        "tiny_cluster_count"
                    ],
                    "late_tiny_cluster_count": late_snapshot[
                        "tiny_cluster_count"
                    ],
                    "early_cluster_size_cv": early_snapshot[
                        "cluster_size_cv"
                    ],
                    "late_cluster_size_cv": late_snapshot[
                        "cluster_size_cv"
                    ],
                    "status": "completed_train_only_window_refit",
                }
            )
    return pd.DataFrame(rows, columns=TEMPORAL_COLUMNS)


def evaluate_representation_robustness(
    correlations: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    *,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> pd.DataFrame:
    """Quantify simple/Pearson and log/Spearman label sensitivity."""

    ordered = list(sorted(map(str, symbols)))
    specifications = (
        (
            RAW_REPRESENTATION,
            "raw_simple_pearson",
            "raw_simple_pearson",
        ),
        (
            RAW_REPRESENTATION,
            "raw_log_spearman",
            "raw_log_spearman",
        ),
        (
            RESIDUAL_REPRESENTATION,
            "residual_simple_pearson",
            "residual_simple_pearson",
        ),
        (
            RESIDUAL_REPRESENTATION,
            "residual_log_spearman",
            "residual_log_spearman",
        ),
        (
            RAW_REPRESENTATION,
            "residual_log_pearson_transformation",
            RESIDUAL_REPRESENTATION,
        ),
    )
    label_cache: dict[tuple[str, int], np.ndarray] = {}

    def labels_for(name: str, cluster_count: int) -> np.ndarray:
        key = (name, cluster_count)
        if key not in label_cache:
            selected = correlations[name].loc[ordered, ordered]
            if selected.isna().any().any():
                raise MarketModeAuditError(
                    f"Robustness representation {name} is incomplete"
                )
            label_cache[key] = hierarchical_labels(
                correlation_to_distance(selected),
                linkage=REFERENCE_LINKAGE,
                cluster_count=cluster_count,
            )
        return label_cache[key]

    rows: list[dict[str, object]] = []
    for primary, variant_label, variant_matrix in specifications:
        for cluster_count in cluster_counts:
            primary_labels = labels_for(primary, int(cluster_count))
            variant_labels = labels_for(variant_matrix, int(cluster_count))
            rows.append(
                {
                    "representation": primary,
                    "variant": variant_label,
                    "requested_clusters": int(cluster_count),
                    "common_symbol_count": len(ordered),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(primary_labels, variant_labels)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(
                            primary_labels, variant_labels
                        )
                    ),
                    "status": "completed_fixed_core_robustness",
                }
            )
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)


def _coefficient_distribution(
    coefficients: pd.DataFrame,
) -> dict[str, float | int | None]:
    beta = pd.to_numeric(
        coefficients.loc[
            coefficients["status"].eq("fitted_train_only_static_ols"), "beta"
        ],
        errors="coerce",
    ).dropna()
    if beta.empty:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    measured = beta.quantile(
        [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    ).to_numpy(dtype="float64")
    return {
        "count": len(beta),
        "min": float(measured[0]),
        "p10": float(measured[1]),
        "p25": float(measured[2]),
        "median": float(measured[3]),
        "p75": float(measured[4]),
        "p90": float(measured[5]),
        "max": float(measured[6]),
    }


def run_market_mode_noise_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
    index_path: str | os.PathLike[str] = INDICES_MASTER_PATH,
    cluster_counts: Sequence[int] = CANDIDATE_CLUSTER_COUNTS,
) -> MarketModeAuditResult:
    """Run the constrained raw/residual/shrinkage TRAIN-only experiment."""

    identity = load_authoritative_current_equity_identity(
        registry_path=registry_path,
        listing_snapshot_path=listing_snapshot_path,
    )
    resolved_parquet = resolve_market_parquet_path(parquet_path)
    identity_symbols = tuple(identity["symbol"].astype(str))
    partitions, market = load_train_only_market_values(
        resolved_parquet, identity_symbols
    )
    training_market = training_market_view(market, partitions)
    built = construct_close_returns(
        training_market,
        identity_symbols,
        global_market_dates=partitions.training_dates,
    )
    log_matrix = build_return_matrix(
        built.returns, value_column="log_return", symbols=identity_symbols
    ).reindex(pd.DatetimeIndex(partitions.training_dates[1:]))
    simple_matrix = build_return_matrix(
        built.returns, value_column="simple_return", symbols=identity_symbols
    ).reindex(pd.DatetimeIndex(partitions.training_dates[1:]))
    overlap = pairwise_overlap_counts(log_matrix)
    diagnostics = build_training_symbol_diagnostics(
        training_market, identity_symbols, log_matrix
    )
    eligible, _ = eligible_symbols_for_overlap_floor(
        diagnostics,
        overlap,
        overlap_floor=REFERENCE_OVERLAP_FLOOR,
    )
    eligible_list = list(eligible)
    eligible_overlap = overlap.loc[eligible_list, eligible_list]
    raw_log_correlation = minimum_overlap_correlation(
        log_matrix.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=eligible_overlap,
    )
    raw_core = deterministic_complete_pair_core(
        raw_log_correlation.notna(), candidates=eligible
    )

    index_evidence = audit_local_kse100_factor(
        index_path, partitions.training_dates
    )
    factor = build_equal_weight_market_factor(log_matrix, eligible)
    residual = residualize_static_market_factor(
        log_matrix,
        factor.factor,
        eligible,
        factor_name="train_equal_weight_current_equity_factor",
    )
    simple_factor = build_equal_weight_market_factor(simple_matrix, eligible)
    simple_residual = residualize_static_market_factor(
        simple_matrix,
        simple_factor.factor,
        eligible,
        factor_name="train_equal_weight_current_equity_simple_factor",
    )
    residual_log_correlation = minimum_overlap_correlation(
        residual.residuals.loc[:, eligible_list],
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=eligible_overlap,
    )
    common_core = deterministic_complete_pair_core(
        raw_log_correlation.notna() & residual_log_correlation.notna(),
        candidates=raw_core,
    )
    core = list(common_core)

    raw_shrinkage = evaluate_ledoit_wolf_shrinkage(
        log_matrix,
        common_core,
        representation=SHRINKAGE_REPRESENTATION,
    )
    residual_shrinkage = evaluate_ledoit_wolf_shrinkage(
        residual.residuals,
        common_core,
        representation=COMBINED_REPRESENTATION,
    )
    primary_correlations: dict[str, pd.DataFrame] = {
        RAW_REPRESENTATION: raw_log_correlation,
        RESIDUAL_REPRESENTATION: residual_log_correlation,
    }
    if raw_shrinkage.feasible and raw_shrinkage.correlation is not None:
        primary_correlations[SHRINKAGE_REPRESENTATION] = raw_shrinkage.correlation
    if residual_shrinkage.feasible and residual_shrinkage.correlation is not None:
        primary_correlations[COMBINED_REPRESENTATION] = residual_shrinkage.correlation

    structure = pd.DataFrame(
        [
            correlation_structure_diagnostics(
                name, correlation.loc[core, core]
            )
            for name, correlation in primary_correlations.items()
        ],
        columns=STRUCTURE_COLUMNS,
    )
    sector_map = identity.set_index("symbol")["sector"].astype(str).to_dict()
    cluster_evidence, _ = evaluate_representation_clustering(
        primary_correlations,
        common_core,
        sector_by_symbol=sector_map,
        cluster_counts=cluster_counts,
    )
    windows = deterministic_temporal_windows(partitions.training_dates)
    temporal = evaluate_temporal_representations(
        training_market,
        identity_symbols,
        windows,
        full_core_count=len(common_core),
        cluster_counts=cluster_counts,
    )

    robustness_correlations = {
        RAW_REPRESENTATION: raw_log_correlation,
        "raw_simple_pearson": minimum_overlap_correlation(
            simple_matrix.loc[:, eligible_list],
            method="pearson",
            minimum_overlap=REFERENCE_OVERLAP_FLOOR,
            overlap_counts=eligible_overlap,
        ),
        "raw_log_spearman": minimum_overlap_correlation(
            log_matrix.loc[:, eligible_list],
            method="spearman",
            minimum_overlap=REFERENCE_OVERLAP_FLOOR,
            overlap_counts=eligible_overlap,
        ),
        RESIDUAL_REPRESENTATION: residual_log_correlation,
        "residual_simple_pearson": minimum_overlap_correlation(
            simple_residual.residuals.loc[:, eligible_list],
            method="pearson",
            minimum_overlap=REFERENCE_OVERLAP_FLOOR,
            overlap_counts=eligible_overlap,
        ),
        "residual_log_spearman": minimum_overlap_correlation(
            residual.residuals.loc[:, eligible_list],
            method="spearman",
            minimum_overlap=REFERENCE_OVERLAP_FLOOR,
            overlap_counts=eligible_overlap,
        ),
    }
    robustness = evaluate_representation_robustness(
        robustness_correlations,
        common_core,
        cluster_counts=cluster_counts,
    )
    train_capable = int(
        diagnostics["training_return_observations"].gt(0).sum()
    )
    summary = MarketModeAuditSummary(
        audit_version=MARKET_MODE_AUDIT_VERSION,
        final_decision=FINAL_DECISION,
        identity_universe_count=len(identity_symbols),
        train_return_capable_count=train_capable,
        eligible_symbol_count=len(eligible),
        common_complete_core_count=len(common_core),
        overlap_floor=REFERENCE_OVERLAP_FLOOR,
        training_date_range=(partitions.training_start, partitions.training_end),
        validation_date_range=(
            partitions.validation_start,
            partitions.validation_end,
        ),
        test_date_range=(partitions.test_start, partitions.test_end),
        temporal_window_ranges=(
            (windows.early_start, windows.early_end),
            (windows.late_start, windows.late_end),
        ),
        external_index_adequate=index_evidence.adequate_for_full_train_static_factor,
        selected_factor="train_equal_weight_current_equity_factor",
        shrinkage_feasible=raw_shrinkage.feasible,
        combined_feasible=residual_shrinkage.feasible,
        reference_k=REFERENCE_K,
        final_assignments_written=False,
        validation_returns_used_for_fitting=False,
        validation_observation_values_loaded=False,
        test_returns_used=False,
        test_observation_values_loaded=False,
        full_calendar_dates_loaded_for_partition_boundaries=False,
    )
    return MarketModeAuditResult(
        summary=summary,
        index_factor=index_evidence,
        cross_sectional_factor=factor,
        beta_distribution=_coefficient_distribution(residual.coefficients),
        correlation_structure=structure,
        shrinkage_assessments=(raw_shrinkage, residual_shrinkage),
        clustering_evidence=cluster_evidence,
        temporal_stability=temporal,
        robustness=robustness,
        parquet_path=resolved_parquet,
    )


def _print_summary(result: MarketModeAuditResult) -> None:
    print(f"Audit version: {result.summary.audit_version}")
    print(f"Final decision: {result.summary.final_decision}")
    print("Index factor evidence:")
    print(json.dumps(result.index_factor.to_dict(), indent=2, sort_keys=True))
    factor = result.cross_sectional_factor
    print(
        "Fallback factor: "
        f"dates={factor.available_date_count}/{factor.possible_date_count}, "
        f"constituents={factor.minimum_constituents_observed}/"
        f"{factor.median_constituents_observed:.1f}/"
        f"{factor.maximum_constituents_observed} min/median/max"
    )
    print("Beta distribution:")
    print(json.dumps(dict(result.beta_distribution), indent=2, sort_keys=True))
    print("Correlation structure:")
    print(result.correlation_structure.to_string(index=False))
    print("Shrinkage feasibility:")
    for assessment in result.shrinkage_assessments:
        print(json.dumps(assessment.to_dict(), sort_keys=True))
    print("Clustering evidence:")
    print(result.clustering_evidence.to_string(index=False))
    print("Temporal stability:")
    print(result.temporal_stability.to_string(index=False))
    print("Robustness:")
    print(result.robustness.to_string(index=False))
    print(
        "Safety: "
        f"validation_fit={result.summary.validation_returns_used_for_fitting}, "
        "validation_values_loaded="
        f"{result.summary.validation_observation_values_loaded}, "
        f"test_used={result.summary.test_returns_used}, "
        f"test_values_loaded={result.summary.test_observation_values_loaded}, "
        "full_calendar_loaded="
        f"{result.summary.full_calendar_dates_loaded_for_partition_boundaries}, "
        f"assignments_written={result.summary.final_assignments_written}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TRAIN-only market-mode and shrinkage diagnostics."
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry", default=str(COMPANY_REGISTRY_PATH)
    )
    parser.add_argument(
        "--listing-snapshot", default=str(CURRENT_LISTINGS_PATH)
    )
    parser.add_argument("--index-path", default=str(INDICES_MASTER_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_market_mode_noise_audit(
            parquet_path=args.path,
            registry_path=args.company_registry,
            listing_snapshot_path=args.listing_snapshot,
            index_path=args.index_path,
        )
        _print_summary(result)
        return 0
    except (
        ClusteringMethodologyError,
        ClusteringProtocolError,
        EquityUniverseError,
        ListingsUnavailableError,
        MarketModeAuditError,
        MarketParquetError,
        UniverseAuditError,
        UniverseMethodologyError,
        ValueError,
        TypeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = (
    "COMBINED_REPRESENTATION",
    "CrossSectionalFactorResult",
    "FINAL_DECISION",
    "IndexFactorEvidence",
    "MARKET_MODE_AUDIT_VERSION",
    "MINIMUM_CROSS_SECTIONAL_CONSTITUENTS",
    "MINIMUM_EXTERNAL_INDEX_COVERAGE",
    "MINIMUM_SHRINKAGE_COMMON_DATES",
    "MarketModeAuditError",
    "MarketModeAuditResult",
    "MarketModeAuditSummary",
    "RAW_REPRESENTATION",
    "REFERENCE_K",
    "REFERENCE_OVERLAP_FLOOR",
    "REGRESSION_MINIMUM_OVERLAP",
    "RESIDUAL_REPRESENTATION",
    "ResidualizationResult",
    "SHRINKAGE_REPRESENTATION",
    "ShrinkageAssessment",
    "audit_local_kse100_factor",
    "build_equal_weight_market_factor",
    "correlation_structure_diagnostics",
    "evaluate_ledoit_wolf_shrinkage",
    "evaluate_representation_clustering",
    "evaluate_representation_robustness",
    "evaluate_temporal_representations",
    "load_authoritative_current_equity_identity",
    "load_train_only_market_values",
    "main",
    "residualize_static_market_factor",
    "run_market_mode_noise_audit",
)
