"""TRAIN-only soft relationship representation and investable-decoder audit.

The module deliberately replaces hard cluster labels with deterministic
nonnegative matrix-factorization loadings, overlapping row-normalized
memberships, and a long-only prototype decoder.  Current sector metadata is
used only after fitting for interpretation.  VALIDATION and TEST observations
are never accepted by the data-loading boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import normalized_mutual_info_score

from feature_engineering.storage import atomic_write_dataframe, atomic_write_json

from .clustering_market_mode import (
    FROZEN_TRAIN_END,
    FROZEN_TRAIN_START,
    REFERENCE_OVERLAP_FLOOR,
    load_authoritative_current_equity_identity,
    load_train_only_market_values,
)
from .clustering_methodology import (
    build_return_matrix,
    build_training_symbol_diagnostics,
    construct_close_returns,
    deterministic_complete_pair_core,
    minimum_overlap_correlation,
    pairwise_overlap_counts,
)
from .clustering_protocol import (
    deterministic_temporal_windows,
    eligible_symbols_for_overlap_floor,
)
from .config import (
    COMPANY_REGISTRY_PATH,
    CURRENT_LISTINGS_PATH,
    PROJECT_ROOT,
    SOFT_RELATIONSHIP_REPRESENTATION_DIR,
)
from .equity_universe import (
    CLASSIFICATION_POLICY_VERSION,
    EQUITY_UNIVERSE_VERSION,
    IDENTITY_POLICY,
    deterministic_universe_identity,
)
from .instrument_audit import COMMON_EQUITY, OFFICIAL_LISTING_DERIVED
from .parquet_market_data import resolve_market_parquet_path


SOFT_REPRESENTATION_VERSION = "soft_relationship_nmf_v1"
SOFT_CONTRACT_VERSION = "soft_relationship_contract_v1"
SOFT_ALIGNMENT_VERSION = "soft_prototype_hungarian_decoder_cosine_v1"
CANDIDATE_DIMENSIONS = (8, 10, 12, 15, 20, 25, 30)
PRIMARY_REPRESENTATION = "log_pearson"
ROBUSTNESS_REPRESENTATIONS = ("simple_pearson", "log_spearman")
NMF_RANDOM_SEED = 42
NMF_MAX_ITERATIONS = 2_000
NMF_TOLERANCE = 1e-5
PROJECTION_MINIMUM_CORE_RELATIONSHIPS = 60
PROJECTION_MINIMUM_CONFIDENCE = 0.10

READY_DECISION = "READY_TO_FREEZE_SOFT_REPRESENTATION"
BLOCKED_DECISION = "BLOCKED_SOFT_REPRESENTATION"

# Predeclared evidence gates.  They are intentionally independent of any RL or
# validation-period return and are not expanded after viewing real results.
MIN_SUPPORTED_IDENTITY_FRACTION = 0.80
MIN_TEMPORAL_SUBSPACE_STABILITY = 0.80
MIN_TEMPORAL_MEMBERSHIP_COSINE = 0.75
MIN_TEMPORAL_DECODER_OVERLAP = 0.55
MIN_ROBUSTNESS_SUBSPACE_STABILITY = 0.85
MIN_ROBUSTNESS_MEMBERSHIP_COSINE = 0.80
MIN_ROBUSTNESS_DECODER_OVERLAP = 0.60
MAX_SECTOR_NMI = 0.75
MAX_MEAN_SECTOR_PURITY = 0.75
MAX_MEMBERSHIP_MASS_MULTIPLIER = 2.50
MIN_NORMALIZED_MEMBERSHIP_ENTROPY = 0.15
MAX_NORMALIZED_MEMBERSHIP_ENTROPY = 0.95
MIN_MEDIAN_DECODER_EFFECTIVE_STOCKS = 5.0
MAX_DECODER_TOP_STOCK_WEIGHT = 0.25
MAX_RELATIONSHIP_CONDITION_NUMBER = 1_000_000.0
MAX_PLATEAU_RECONSTRUCTION_GAIN = 0.03


class SoftRelationshipError(RuntimeError):
    """Raised when a soft representation would violate its research contract."""


@dataclass(frozen=True)
class SoftPrototypeFit:
    """One deterministic factorization and its canonical prototype artifacts."""

    k: int
    symbols: tuple[str, ...]
    relationship_vectors: pd.DataFrame = field(repr=False, compare=False)
    memberships: pd.DataFrame = field(repr=False, compare=False)
    decoder: pd.DataFrame = field(repr=False, compare=False)
    basis: pd.DataFrame = field(repr=False, compare=False)
    reconstruction_error: float
    condition_number: float
    iterations: int
    converged: bool
    relationship_hash: str
    membership_hash: str
    decoder_hash: str


@dataclass(frozen=True)
class ExpandedSoftRepresentation:
    """Fit-core plus projected and explicitly unsupported identity members."""

    k: int
    identity_symbols: tuple[str, ...]
    relationship_vectors: pd.DataFrame = field(repr=False, compare=False)
    memberships: pd.DataFrame = field(repr=False, compare=False)
    decoder: pd.DataFrame = field(repr=False, compare=False)
    eligibility: pd.DataFrame = field(repr=False, compare=False)
    fitted_count: int
    projected_count: int
    unsupported_count: int
    representation_hash: str
    membership_hash: str
    decoder_hash: str


@dataclass(frozen=True)
class SoftAuditSummary:
    representation_version: str
    contract_version: str
    alignment_version: str
    decision: str
    decision_reason: str
    selected_k: int | None
    candidate_dimensions: tuple[int, ...]
    identity_count: int
    train_return_capable_count: int
    eligible_count: int
    fit_core_count: int
    temporal_core_count: int
    fitted_count: int
    projected_count: int
    unsupported_count: int
    train_start: str
    train_end: str
    overlap_floor: int
    projection_minimum_core_relationships: int
    universe_hash: str
    source_parquet_sha256: str
    representation_hash: str | None
    decoder_hash: str | None
    validation_values_loaded: bool = False
    test_values_loaded: bool = False
    sector_used_for_primary_fit: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_dimensions"] = list(self.candidate_dimensions)
        return payload


@dataclass(frozen=True)
class SoftRelationshipAuditResult:
    summary: SoftAuditSummary
    k_comparison: pd.DataFrame = field(repr=False, compare=False)
    temporal_stability: pd.DataFrame = field(repr=False, compare=False)
    robustness: pd.DataFrame = field(repr=False, compare=False)
    sector_diagnostics: pd.DataFrame = field(repr=False, compare=False)
    selected: ExpandedSoftRepresentation | None = field(
        default=None, repr=False, compare=False
    )
    parquet_path: Path = field(default=Path(), compare=False)


def validate_candidate_dimensions(values: Sequence[int]) -> tuple[int, ...]:
    """Require the exact supervisor-approved K grid without adaptive expansion."""

    normalized = tuple(int(value) for value in values)
    if normalized != CANDIDATE_DIMENSIONS:
        raise ValueError(
            f"candidate dimensions must be exactly {CANDIDATE_DIMENSIONS}"
        )
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_payload(value: object) -> str | None:
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return format(round(numeric, 12), ".12g")


def deterministic_frame_hash(frame: pd.DataFrame) -> str:
    """Hash stable labels and rounded numeric content, representing NaN as null."""

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    payload = {
        "index": [str(value) for value in frame.index],
        "columns": [str(value) for value in frame.columns],
        "values": [
            [_float_payload(value) for value in row]
            for row in numeric.to_numpy(dtype="float64")
        ],
    }
    return _canonical_hash(payload)


def authoritative_identity_hash(identity: pd.DataFrame) -> str:
    """Reproduce the frozen universe identity without reading future market rows."""

    required = {"symbol", "sector", "security_type", "source", "snapshot_date"}
    missing = sorted(required.difference(identity.columns))
    if missing:
        raise SoftRelationshipError(
            "identity is missing provenance columns: " + ", ".join(missing)
        )
    ordered = identity.sort_values("symbol", kind="mergesort")
    snapshots = sorted(set(ordered["snapshot_date"].astype(str)))
    if len(snapshots) != 1:
        raise SoftRelationshipError("identity must have one listing snapshot date")
    members = [
        {
            "symbol": str(row.symbol),
            "instrument_category": COMMON_EQUITY,
            "classification_basis": OFFICIAL_LISTING_DERIVED,
            "security_type": str(row.security_type),
            "sector": str(row.sector),
            "authoritative_source": str(row.source),
        }
        for row in ordered.itertuples(index=False)
    ]
    return deterministic_universe_identity(
        {
            "universe_version": EQUITY_UNIVERSE_VERSION,
            "identity_policy": IDENTITY_POLICY,
            "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
            "listing_snapshot_date": snapshots[0],
            "members": members,
        }
    )


def positive_correlation_affinity(correlation: pd.DataFrame) -> pd.DataFrame:
    """Return complete long-only comovement affinity without masking missing data.

    Negative finite correlation is defined as zero long-only prototype affinity;
    missing correlation remains an error.  This is not missing-value zero fill.
    """

    if correlation.shape[0] != correlation.shape[1]:
        raise SoftRelationshipError("correlation must be square")
    if list(correlation.index) != list(correlation.columns):
        raise SoftRelationshipError("correlation labels are not aligned")
    values = correlation.to_numpy(dtype="float64", copy=True)
    if not np.isfinite(values).all():
        raise SoftRelationshipError("fit affinity requires complete correlations")
    values = np.clip((values + values.T) / 2.0, -1.0, 1.0)
    affinity = np.maximum(values, 0.0)
    np.fill_diagonal(affinity, 1.0)
    return pd.DataFrame(
        affinity, index=correlation.index.copy(), columns=correlation.columns.copy()
    )


def _prototype_columns(k: int) -> tuple[str, ...]:
    return tuple(f"prototype_{index + 1:02d}" for index in range(k))


def _canonical_prototype_order(
    memberships: np.ndarray, symbols: Sequence[str]
) -> np.ndarray:
    keys: list[tuple[str, str, int]] = []
    for index in range(memberships.shape[1]):
        column = memberships[:, index]
        anchor = str(symbols[int(np.argmax(column))])
        content = hashlib.sha256(np.round(column, 12).tobytes()).hexdigest()
        keys.append((anchor, content, index))
    return np.asarray([item[2] for item in sorted(keys)], dtype="int64")


def _normalize_decoder(memberships: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    weighted = memberships * confidence[:, None]
    totals = weighted.sum(axis=0)
    if not np.isfinite(weighted).all() or np.any(weighted < 0) or np.any(totals <= 0):
        raise SoftRelationshipError("decoder cannot normalize empty/nonfinite columns")
    decoder = weighted / totals
    if not np.allclose(decoder.sum(axis=0), 1.0, atol=1e-10):
        raise SoftRelationshipError("decoder columns do not conserve capital")
    return decoder


def fit_soft_prototypes(
    correlation: pd.DataFrame,
    *,
    k: int,
) -> SoftPrototypeFit:
    """Fit one deterministic NMF representation on a complete fit core."""

    if k not in CANDIDATE_DIMENSIONS:
        raise ValueError(f"k must be one of {CANDIDATE_DIMENSIONS}")
    affinity = positive_correlation_affinity(correlation)
    if k >= len(affinity):
        raise SoftRelationshipError("k must be smaller than fit-core size")
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        tol=NMF_TOLERANCE,
        max_iter=NMF_MAX_ITERATIONS,
        random_state=NMF_RANDOM_SEED,
        shuffle=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        raw_loadings = model.fit_transform(affinity.to_numpy(dtype="float64"))
    raw_basis = model.components_.copy()
    basis_norms = np.linalg.norm(raw_basis, axis=1)
    if np.any(basis_norms <= 0) or not np.isfinite(basis_norms).all():
        raise SoftRelationshipError("NMF produced an empty prototype basis")
    relationship = raw_loadings * basis_norms[None, :]
    basis = raw_basis / basis_norms[:, None]
    row_totals = relationship.sum(axis=1)
    if np.any(row_totals <= 0) or not np.isfinite(relationship).all():
        raise SoftRelationshipError("NMF produced an empty relationship vector")
    memberships = relationship / row_totals[:, None]
    order = _canonical_prototype_order(memberships, affinity.index)
    relationship = relationship[:, order]
    memberships = memberships[:, order]
    basis = basis[order, :]
    columns = _prototype_columns(k)
    symbols = tuple(map(str, affinity.index))
    relationship_frame = pd.DataFrame(
        relationship, index=symbols, columns=columns
    )
    membership_frame = pd.DataFrame(
        memberships, index=symbols, columns=columns
    )
    decoder_frame = pd.DataFrame(
        _normalize_decoder(memberships, np.ones(len(symbols))),
        index=symbols,
        columns=columns,
    )
    basis_frame = pd.DataFrame(basis, index=columns, columns=symbols)
    reconstructed = relationship @ basis
    denominator = np.linalg.norm(affinity.to_numpy(dtype="float64"))
    error = float(
        np.linalg.norm(affinity.to_numpy(dtype="float64") - reconstructed)
        / denominator
    )
    singular_values = np.linalg.svd(relationship, compute_uv=False)
    condition = float(
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > np.finfo("float64").eps
        else math.inf
    )
    converged = not any(isinstance(item.message, ConvergenceWarning) for item in caught)
    return SoftPrototypeFit(
        k=k,
        symbols=symbols,
        relationship_vectors=relationship_frame,
        memberships=membership_frame,
        decoder=decoder_frame,
        basis=basis_frame,
        reconstruction_error=error,
        condition_number=condition,
        iterations=int(model.n_iter_),
        converged=converged and int(model.n_iter_) < NMF_MAX_ITERATIONS,
        relationship_hash=deterministic_frame_hash(relationship_frame),
        membership_hash=deterministic_frame_hash(membership_frame),
        decoder_hash=deterministic_frame_hash(decoder_frame),
    )


def project_relationship_vector(
    correlations_to_core: pd.Series,
    fit: SoftPrototypeFit,
) -> tuple[np.ndarray | None, float, int, float | None, str | None]:
    """Project one sparse stock using NNLS over observed core relationships only."""

    aligned = pd.to_numeric(
        correlations_to_core.reindex(fit.symbols), errors="coerce"
    )
    observed = np.isfinite(aligned.to_numpy(dtype="float64"))
    observed_count = int(observed.sum())
    if observed_count < PROJECTION_MINIMUM_CORE_RELATIONSHIPS:
        return None, 0.0, observed_count, None, "insufficient_core_relationships"
    affinity = np.maximum(aligned.to_numpy(dtype="float64")[observed], 0.0)
    if not np.any(affinity > 0):
        return None, 0.0, observed_count, None, "no_positive_core_affinity"
    design = fit.basis.to_numpy(dtype="float64")[:, observed].T
    coefficients, residual_norm = nnls(design, affinity)
    coefficient_sum = float(coefficients.sum())
    if coefficient_sum <= 0 or not np.isfinite(coefficients).all():
        return None, 0.0, observed_count, None, "nnls_empty_projection"
    denominator = float(np.linalg.norm(affinity))
    relative_error = float(residual_norm / denominator) if denominator > 0 else 1.0
    coverage = observed_count / len(fit.symbols)
    confidence = float(np.clip(coverage * max(0.0, 1.0 - relative_error), 0.0, 1.0))
    if confidence < PROJECTION_MINIMUM_CONFIDENCE:
        return (
            None,
            confidence,
            observed_count,
            relative_error,
            "projection_confidence_below_floor",
        )
    return coefficients, confidence, observed_count, relative_error, None


def expand_to_identity(
    fit: SoftPrototypeFit,
    all_correlations: pd.DataFrame,
    identity_symbols: Sequence[str],
) -> ExpandedSoftRepresentation:
    """Represent every identity as fitted, projected, or explicitly unsupported."""

    symbols = tuple(sorted(map(str, identity_symbols)))
    if len(symbols) != len(set(symbols)):
        raise SoftRelationshipError("identity symbols must be unique")
    correlations = all_correlations.reindex(index=symbols, columns=fit.symbols)
    relationship = pd.DataFrame(
        np.nan, index=symbols, columns=fit.relationship_vectors.columns
    )
    memberships = relationship.copy(deep=True)
    confidence = pd.Series(0.0, index=symbols, dtype="float64")
    records: list[dict[str, object]] = []
    fit_set = set(fit.symbols)
    for symbol in symbols:
        if symbol in fit_set:
            relationship.loc[symbol] = fit.relationship_vectors.loc[symbol]
            memberships.loc[symbol] = fit.memberships.loc[symbol]
            confidence.loc[symbol] = 1.0
            records.append(
                {
                    "symbol": symbol,
                    "status": "fitted",
                    "reason": "complete_fit_core",
                    "observed_core_relationships": len(fit.symbols),
                    "projection_relative_error": 0.0,
                    "confidence": 1.0,
                }
            )
            continue
        coefficients, score, observed, error, reason = project_relationship_vector(
            correlations.loc[symbol], fit
        )
        if coefficients is None:
            records.append(
                {
                    "symbol": symbol,
                    "status": "unsupported",
                    "reason": reason,
                    "observed_core_relationships": observed,
                    "projection_relative_error": error,
                    "confidence": score,
                }
            )
            continue
        relationship.loc[symbol] = coefficients
        memberships.loc[symbol] = coefficients / coefficients.sum()
        confidence.loc[symbol] = score
        records.append(
            {
                "symbol": symbol,
                "status": "projected",
                "reason": "observed_core_nnls_projection",
                "observed_core_relationships": observed,
                "projection_relative_error": error,
                "confidence": score,
            }
        )
    eligibility = pd.DataFrame(records).sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
    if len(eligibility) != len(symbols) or eligibility["symbol"].duplicated().any():
        raise SoftRelationshipError("identity eligibility lost or duplicated symbols")
    supported = eligibility.loc[
        eligibility["status"].isin(("fitted", "projected")), "symbol"
    ].astype(str)
    supported_memberships = memberships.loc[supported]
    supported_confidence = confidence.loc[supported].to_numpy(dtype="float64")
    decoder_values = _normalize_decoder(
        supported_memberships.to_numpy(dtype="float64"), supported_confidence
    )
    decoder = pd.DataFrame(
        0.0, index=symbols, columns=fit.memberships.columns, dtype="float64"
    )
    decoder.loc[supported] = decoder_values
    if not np.allclose(decoder.sum(axis=0), 1.0, atol=1e-10):
        raise SoftRelationshipError("expanded decoder does not conserve capital")
    fitted_count = int(eligibility["status"].eq("fitted").sum())
    projected_count = int(eligibility["status"].eq("projected").sum())
    unsupported_count = int(eligibility["status"].eq("unsupported").sum())
    representation_hash = deterministic_frame_hash(relationship)
    membership_hash = deterministic_frame_hash(memberships)
    decoder_hash = deterministic_frame_hash(decoder)
    return ExpandedSoftRepresentation(
        k=fit.k,
        identity_symbols=symbols,
        relationship_vectors=relationship,
        memberships=memberships,
        decoder=decoder,
        eligibility=eligibility,
        fitted_count=fitted_count,
        projected_count=projected_count,
        unsupported_count=unsupported_count,
        representation_hash=representation_hash,
        membership_hash=membership_hash,
        decoder_hash=decoder_hash,
    )


def decode_prototype_allocations(
    decoder: pd.DataFrame, allocation: Sequence[float]
) -> pd.Series:
    """Decode nonnegative prototype capital and verify exact conservation."""

    values = np.asarray(allocation, dtype="float64")
    if values.shape != (decoder.shape[1],):
        raise ValueError("allocation length must equal decoder prototype count")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("allocation must be finite and nonnegative")
    matrix = decoder.to_numpy(dtype="float64")
    if np.any(matrix < -1e-12) or not np.allclose(
        matrix.sum(axis=0), 1.0, atol=1e-10
    ):
        raise SoftRelationshipError("decoder is not nonnegative/column-normalized")
    budgets = matrix @ values
    if not math.isclose(float(budgets.sum()), float(values.sum()), abs_tol=1e-10):
        raise SoftRelationshipError("prototype decoder does not conserve capital")
    return pd.Series(budgets, index=decoder.index, dtype="float64")


def align_soft_prototypes(
    reference: SoftPrototypeFit,
    candidate: SoftPrototypeFit,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align candidate columns to reference using decoder cosine similarity."""

    if reference.k != candidate.k:
        raise ValueError("prototype alignment requires equal K")
    common = tuple(sorted(set(reference.symbols).intersection(candidate.symbols)))
    if len(common) <= reference.k:
        raise SoftRelationshipError("prototype alignment has too few common symbols")
    left = reference.decoder.loc[list(common)].to_numpy(
        dtype="float64", copy=True
    )
    right = candidate.decoder.loc[list(common)].to_numpy(
        dtype="float64", copy=True
    )
    left /= np.maximum(np.linalg.norm(left, axis=0), np.finfo("float64").eps)
    right /= np.maximum(np.linalg.norm(right, axis=0), np.finfo("float64").eps)
    similarity = np.clip(left.T @ right, 0.0, 1.0)
    rows, columns = linear_sum_assignment(-similarity)
    if not np.array_equal(rows, np.arange(reference.k)):
        raise SoftRelationshipError("prototype alignment did not cover reference order")
    diagnostics = pd.DataFrame(
        {
            "reference_prototype": reference.decoder.columns,
            "candidate_prototype": candidate.decoder.columns[columns],
            "decoder_cosine_similarity": similarity[rows, columns],
        }
    )
    return columns.astype("int64"), diagnostics


def _subspace_stability(left: pd.DataFrame, right: pd.DataFrame) -> float:
    common = tuple(sorted(set(left.index).intersection(right.index)))
    left_values = left.loc[list(common)].to_numpy(dtype="float64")
    right_values = right.loc[list(common)].to_numpy(dtype="float64")
    left_q, _ = np.linalg.qr(left_values)
    right_q, _ = np.linalg.qr(right_values)
    singular = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    return float(np.clip(singular, 0.0, 1.0).mean())


def compare_soft_fits(
    reference: SoftPrototypeFit,
    candidate: SoftPrototypeFit,
) -> dict[str, float | int]:
    """Return alignment-aware continuous representation and decoder stability."""

    order, alignment = align_soft_prototypes(reference, candidate)
    common = tuple(sorted(set(reference.symbols).intersection(candidate.symbols)))
    reference_membership = reference.memberships.loc[list(common)].to_numpy(
        dtype="float64"
    )
    candidate_membership = candidate.memberships.loc[list(common)].to_numpy(
        dtype="float64"
    )[:, order]
    denominator = np.linalg.norm(reference_membership, axis=1) * np.linalg.norm(
        candidate_membership, axis=1
    )
    row_cosine = np.divide(
        (reference_membership * candidate_membership).sum(axis=1),
        denominator,
        out=np.zeros(len(common), dtype="float64"),
        where=denominator > 0,
    )
    reference_decoder = reference.decoder.loc[list(common)].to_numpy(
        dtype="float64", copy=True
    )
    candidate_decoder = candidate.decoder.loc[list(common)].to_numpy(
        dtype="float64", copy=True
    )[:, order]
    reference_decoder /= reference_decoder.sum(axis=0)
    candidate_decoder /= candidate_decoder.sum(axis=0)
    decoder_overlap = np.minimum(reference_decoder, candidate_decoder).sum(axis=0)
    return {
        "common_symbol_count": len(common),
        "subspace_stability": _subspace_stability(
            reference.relationship_vectors.loc[list(common)],
            candidate.relationship_vectors.loc[list(common)],
        ),
        "mean_membership_cosine": float(row_cosine.mean()),
        "p10_membership_cosine": float(np.quantile(row_cosine, 0.10)),
        "mean_alignment_cosine": float(
            alignment["decoder_cosine_similarity"].mean()
        ),
        "minimum_alignment_cosine": float(
            alignment["decoder_cosine_similarity"].min()
        ),
        "mean_decoder_overlap": float(decoder_overlap.mean()),
        "minimum_decoder_overlap": float(decoder_overlap.min()),
    }


def _fit_diagnostics(expanded: ExpandedSoftRepresentation) -> dict[str, float]:
    supported_symbols = expanded.eligibility.loc[
        expanded.eligibility["status"].isin(("fitted", "projected")), "symbol"
    ].astype(str)
    membership = expanded.memberships.loc[supported_symbols].to_numpy(dtype="float64")
    entropy = -(membership * np.log(np.maximum(membership, 1e-15))).sum(axis=1)
    normalized_entropy = entropy / math.log(expanded.k)
    mass = membership.sum(axis=0)
    mass /= mass.sum()
    decoder = expanded.decoder.to_numpy(dtype="float64")
    effective_stocks = 1.0 / np.maximum((decoder**2).sum(axis=0), 1e-15)
    top_weight = decoder.max(axis=0)
    return {
        "mean_normalized_membership_entropy": float(normalized_entropy.mean()),
        "median_effective_memberships": float(np.median(np.exp(entropy))),
        "maximum_membership_mass_share": float(mass.max()),
        "membership_mass_cv": float(mass.std(ddof=0) / mass.mean()),
        "median_decoder_effective_stocks": float(np.median(effective_stocks)),
        "minimum_decoder_effective_stocks": float(effective_stocks.min()),
        "maximum_decoder_top_stock_weight": float(top_weight.max()),
    }


def sector_correspondence_diagnostics(
    expanded: ExpandedSoftRepresentation,
    sector_by_symbol: Mapping[str, str],
) -> dict[str, float | bool]:
    """Describe current-sector correspondence after, never during, fitting."""

    supported = expanded.eligibility.loc[
        expanded.eligibility["status"].isin(("fitted", "projected")), "symbol"
    ].astype(str)
    sectors = [sector_by_symbol.get(symbol, "UNKNOWN") for symbol in supported]
    labels = np.argmax(
        expanded.memberships.loc[supported].to_numpy(dtype="float64"), axis=1
    )
    nmi = float(normalized_mutual_info_score(sectors, labels))
    purities: list[float] = []
    sector_values = np.asarray(sectors, dtype=object)
    decoder = expanded.decoder.loc[supported].to_numpy(dtype="float64", copy=True)
    decoder /= decoder.sum(axis=0)
    for prototype in range(expanded.k):
        weights = decoder[:, prototype]
        totals: dict[str, float] = {}
        for sector, weight in zip(sector_values, weights, strict=True):
            totals[str(sector)] = totals.get(str(sector), 0.0) + float(weight)
        purities.append(max(totals.values()))
    mean_purity = float(np.mean(purities))
    maximum_purity = float(np.max(purities))
    dominated = nmi >= MAX_SECTOR_NMI or mean_purity >= MAX_MEAN_SECTOR_PURITY
    return {
        "sector_nmi_argmax_posthoc": nmi,
        "mean_decoder_sector_purity": mean_purity,
        "maximum_decoder_sector_purity": maximum_purity,
        "sector_domination": bool(dominated),
    }


def _window_correlation(
    training_market: pd.DataFrame,
    identity_symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_dates = pd.DatetimeIndex(dates)
    window = training_market.loc[
        pd.to_datetime(training_market["market_date"]).isin(selected_dates)
    ].copy(deep=True)
    built = construct_close_returns(
        window, identity_symbols, global_market_dates=selected_dates
    )
    matrix = build_return_matrix(
        built.returns, value_column="log_return", symbols=identity_symbols
    ).reindex(selected_dates[1:])
    overlap = pairwise_overlap_counts(matrix)
    correlation = minimum_overlap_correlation(
        matrix,
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=overlap,
    )
    return correlation, overlap


def _readiness_gates(
    row: Mapping[str, object], *, k: int
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []

    def minimum(field: str, threshold: float) -> None:
        if float(row[field]) < threshold:
            failures.append(f"{field}<{threshold}")

    def maximum(field: str, threshold: float) -> None:
        if float(row[field]) > threshold:
            failures.append(f"{field}>{threshold}")

    minimum("supported_identity_fraction", MIN_SUPPORTED_IDENTITY_FRACTION)
    minimum("temporal_subspace_stability", MIN_TEMPORAL_SUBSPACE_STABILITY)
    minimum("temporal_membership_cosine", MIN_TEMPORAL_MEMBERSHIP_COSINE)
    minimum("temporal_decoder_overlap", MIN_TEMPORAL_DECODER_OVERLAP)
    minimum("worst_robustness_subspace", MIN_ROBUSTNESS_SUBSPACE_STABILITY)
    minimum("worst_robustness_membership_cosine", MIN_ROBUSTNESS_MEMBERSHIP_COSINE)
    minimum("worst_robustness_decoder_overlap", MIN_ROBUSTNESS_DECODER_OVERLAP)
    minimum(
        "mean_normalized_membership_entropy",
        MIN_NORMALIZED_MEMBERSHIP_ENTROPY,
    )
    maximum(
        "mean_normalized_membership_entropy",
        MAX_NORMALIZED_MEMBERSHIP_ENTROPY,
    )
    maximum(
        "maximum_membership_mass_share", MAX_MEMBERSHIP_MASS_MULTIPLIER / k
    )
    minimum(
        "median_decoder_effective_stocks",
        MIN_MEDIAN_DECODER_EFFECTIVE_STOCKS,
    )
    maximum("maximum_decoder_top_stock_weight", MAX_DECODER_TOP_STOCK_WEIGHT)
    maximum("condition_number", MAX_RELATIONSHIP_CONDITION_NUMBER)
    if bool(row["sector_domination"]):
        failures.append("sector_domination")
    if not bool(row["converged"]):
        failures.append("nmf_not_converged")
    return not failures, tuple(failures)


def run_soft_relationship_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
    candidate_dimensions: Sequence[int] = CANDIDATE_DIMENSIONS,
) -> SoftRelationshipAuditResult:
    """Run the single predeclared TRAIN-only soft-representation experiment."""

    dimensions = validate_candidate_dimensions(candidate_dimensions)
    identity = load_authoritative_current_equity_identity(
        registry_path=registry_path,
        listing_snapshot_path=listing_snapshot_path,
    )
    identity_symbols = tuple(identity["symbol"].astype(str))
    if len(identity_symbols) != len(set(identity_symbols)):
        raise SoftRelationshipError("authoritative identity contains duplicates")
    resolved_parquet = resolve_market_parquet_path(parquet_path)
    partitions, training_market = load_train_only_market_values(
        resolved_parquet,
        identity_symbols,
        training_start=FROZEN_TRAIN_START,
        training_end=FROZEN_TRAIN_END,
    )
    market_dates = pd.to_datetime(training_market["market_date"], errors="coerce")
    if market_dates.isna().any() or (
        not training_market.empty
        and market_dates.max() > pd.Timestamp(FROZEN_TRAIN_END)
    ):
        raise SoftRelationshipError("TRAIN-only market boundary was crossed")
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
        diagnostics, overlap, overlap_floor=REFERENCE_OVERLAP_FLOOR
    )
    primary_all = minimum_overlap_correlation(
        log_matrix,
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=overlap,
    )
    fit_core = deterministic_complete_pair_core(
        primary_all.loc[list(eligible), list(eligible)].notna(),
        candidates=eligible,
    )
    if len(fit_core) <= max(dimensions):
        raise SoftRelationshipError("fit core is too small for the candidate K grid")
    primary_core = primary_all.loc[list(fit_core), list(fit_core)]

    windows = deterministic_temporal_windows(partitions.training_dates)
    early_correlation, _ = _window_correlation(
        training_market, identity_symbols, windows.early_dates
    )
    late_correlation, _ = _window_correlation(
        training_market, identity_symbols, windows.late_dates
    )
    temporal_core = deterministic_complete_pair_core(
        (
            early_correlation.loc[list(fit_core), list(fit_core)].notna()
            & late_correlation.loc[list(fit_core), list(fit_core)].notna()
        ),
        candidates=fit_core,
    )
    if len(temporal_core) <= max(dimensions):
        raise SoftRelationshipError("temporal common core is too small")

    simple_all = minimum_overlap_correlation(
        simple_matrix,
        method="pearson",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=overlap,
    )
    spearman_all = minimum_overlap_correlation(
        log_matrix,
        method="spearman",
        minimum_overlap=REFERENCE_OVERLAP_FLOOR,
        overlap_counts=overlap,
    )
    sector_map = identity.set_index("symbol")["sector"].astype(str).to_dict()
    expanded_by_k: dict[int, ExpandedSoftRepresentation] = {}
    comparison_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []

    for k in dimensions:
        primary_fit = fit_soft_prototypes(primary_core, k=k)
        expanded = expand_to_identity(primary_fit, primary_all, identity_symbols)
        expanded_by_k[k] = expanded
        early_fit = fit_soft_prototypes(
            early_correlation.loc[list(temporal_core), list(temporal_core)], k=k
        )
        late_fit = fit_soft_prototypes(
            late_correlation.loc[list(temporal_core), list(temporal_core)], k=k
        )
        temporal_metrics = compare_soft_fits(early_fit, late_fit)
        temporal_rows.append(
            {
                "k": k,
                "early_start": windows.early_start,
                "early_end": windows.early_end,
                "late_start": windows.late_start,
                "late_end": windows.late_end,
                "window_date_count": windows.window_date_count,
                "shared_date_count": windows.shared_date_count,
                "temporal_core_count": len(temporal_core),
                **temporal_metrics,
            }
        )
        variant_metrics: list[dict[str, float | int]] = []
        for name, correlation in (
            ("simple_pearson", simple_all),
            ("log_spearman", spearman_all),
        ):
            variant_fit = fit_soft_prototypes(
                correlation.loc[list(fit_core), list(fit_core)], k=k
            )
            metrics = compare_soft_fits(primary_fit, variant_fit)
            variant_metrics.append(metrics)
            robustness_rows.append({"k": k, "variant": name, **metrics})
        sector_metrics = sector_correspondence_diagnostics(expanded, sector_map)
        sector_rows.append({"k": k, **sector_metrics})
        fit_metrics = _fit_diagnostics(expanded)
        row: dict[str, object] = {
            "k": k,
            "fitted_count": expanded.fitted_count,
            "projected_count": expanded.projected_count,
            "unsupported_count": expanded.unsupported_count,
            "supported_identity_fraction": (
                expanded.fitted_count + expanded.projected_count
            )
            / len(identity_symbols),
            "reconstruction_error": primary_fit.reconstruction_error,
            "condition_number": primary_fit.condition_number,
            "iterations": primary_fit.iterations,
            "converged": primary_fit.converged,
            "temporal_subspace_stability": temporal_metrics["subspace_stability"],
            "temporal_membership_cosine": temporal_metrics[
                "mean_membership_cosine"
            ],
            "temporal_alignment_cosine": temporal_metrics[
                "mean_alignment_cosine"
            ],
            "temporal_decoder_overlap": temporal_metrics[
                "mean_decoder_overlap"
            ],
            "worst_robustness_subspace": min(
                float(item["subspace_stability"]) for item in variant_metrics
            ),
            "worst_robustness_membership_cosine": min(
                float(item["mean_membership_cosine"]) for item in variant_metrics
            ),
            "worst_robustness_decoder_overlap": min(
                float(item["mean_decoder_overlap"]) for item in variant_metrics
            ),
            **fit_metrics,
            **sector_metrics,
            "sequential_100k_minutes": k * 3.6,
            "sequential_250k_minutes": k * 8.8,
        }
        passes, failures = _readiness_gates(row, k=k)
        row["passes_evidence_gates"] = passes
        row["gate_failures"] = ";".join(failures)
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows).sort_values("k").reset_index(drop=True)
    gains: list[float | None] = []
    errors = comparison["reconstruction_error"].to_numpy(dtype="float64")
    for index, error in enumerate(errors):
        if index == len(errors) - 1:
            gains.append(None)
        else:
            gains.append(float((error - errors[index + 1]) / error))
    comparison["relative_reconstruction_gain_to_next_k"] = gains
    comparison["plateau_after_k"] = comparison[
        "relative_reconstruction_gain_to_next_k"
    ].map(
        lambda value: bool(
            pd.notna(value) and float(value) <= MAX_PLATEAU_RECONSTRUCTION_GAIN
        )
    )
    comparison["selection_eligible"] = (
        comparison["passes_evidence_gates"] & comparison["plateau_after_k"]
    )
    selectable = comparison.loc[comparison["selection_eligible"]]
    selected_k = int(selectable.iloc[0]["k"]) if not selectable.empty else None
    decision = READY_DECISION if selected_k is not None else BLOCKED_DECISION
    if selected_k is not None:
        decision_reason = (
            f"K={selected_k} is the smallest candidate passing every predeclared "
            "evidence gate at the reconstruction plateau."
        )
        selected = expanded_by_k[selected_k]
    else:
        failures = sorted(
            set(
                item
                for value in comparison["gate_failures"].astype(str)
                for item in value.split(";")
                if item
            )
        )
        decision_reason = (
            "No candidate K passed every predeclared evidence and plateau gate: "
            + ", ".join(failures)
        )
        selected = None
    reference_counts = expanded_by_k[dimensions[0]]
    universe_hash = authoritative_identity_hash(identity)
    parquet_hash = _sha256_file(resolved_parquet)
    summary = SoftAuditSummary(
        representation_version=SOFT_REPRESENTATION_VERSION,
        contract_version=SOFT_CONTRACT_VERSION,
        alignment_version=SOFT_ALIGNMENT_VERSION,
        decision=decision,
        decision_reason=decision_reason,
        selected_k=selected_k,
        candidate_dimensions=dimensions,
        identity_count=len(identity_symbols),
        train_return_capable_count=int(
            diagnostics["training_return_observations"].gt(0).sum()
        ),
        eligible_count=len(eligible),
        fit_core_count=len(fit_core),
        temporal_core_count=len(temporal_core),
        fitted_count=(selected.fitted_count if selected else reference_counts.fitted_count),
        projected_count=(
            selected.projected_count if selected else reference_counts.projected_count
        ),
        unsupported_count=(
            selected.unsupported_count if selected else reference_counts.unsupported_count
        ),
        train_start=partitions.training_start,
        train_end=partitions.training_end,
        overlap_floor=REFERENCE_OVERLAP_FLOOR,
        projection_minimum_core_relationships=PROJECTION_MINIMUM_CORE_RELATIONSHIPS,
        universe_hash=universe_hash,
        source_parquet_sha256=parquet_hash,
        representation_hash=(selected.representation_hash if selected else None),
        decoder_hash=(selected.decoder_hash if selected else None),
    )
    return SoftRelationshipAuditResult(
        summary=summary,
        k_comparison=comparison,
        temporal_stability=pd.DataFrame(temporal_rows).sort_values("k").reset_index(
            drop=True
        ),
        robustness=pd.DataFrame(robustness_rows).sort_values(
            ["k", "variant"], kind="mergesort"
        ).reset_index(drop=True),
        sector_diagnostics=pd.DataFrame(sector_rows).sort_values("k").reset_index(
            drop=True
        ),
        selected=selected,
        parquet_path=resolved_parquet,
    )


def write_soft_relationship_artifacts(
    result: SoftRelationshipAuditResult,
    output_dir: Path = SOFT_RELATIONSHIP_REPRESENTATION_DIR,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Persist a READY representation atomically; BLOCKED audits cannot freeze."""

    if result.summary.decision != READY_DECISION or result.selected is None:
        raise SoftRelationshipError("BLOCKED representation cannot be frozen")
    destination = Path(output_dir)
    expected = (
        "manifest.json",
        "relationship_vectors.csv",
        "soft_memberships.csv",
        "decoder.csv",
        "eligibility.csv",
        "k_comparison.csv",
        "temporal_stability.csv",
        "robustness.csv",
        "sector_diagnostics.csv",
    )
    existing = [destination / name for name in expected if (destination / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "soft relationship artifacts already exist; pass overwrite=True explicitly"
        )
    destination.mkdir(parents=True, exist_ok=True)
    selected = result.selected
    frame_outputs = {
        "relationship_vectors.csv": selected.relationship_vectors.rename_axis(
            "symbol"
        ).reset_index(),
        "soft_memberships.csv": selected.memberships.rename_axis("symbol").reset_index(),
        "decoder.csv": selected.decoder.rename_axis("symbol").reset_index(),
        "eligibility.csv": selected.eligibility,
        "k_comparison.csv": result.k_comparison,
        "temporal_stability.csv": result.temporal_stability,
        "robustness.csv": result.robustness,
        "sector_diagnostics.csv": result.sector_diagnostics,
    }
    for name, frame in frame_outputs.items():
        atomic_write_dataframe(frame, destination / name)
    manifest = {
        **result.summary.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prototype_ordering": (
            "ascending anchor symbol, then canonical membership-content hash"
        ),
        "alignment_policy": SOFT_ALIGNMENT_VERSION,
        "decoder_contract": {
            "nonnegative": True,
            "column_sums": 1.0,
            "stock_budget_formula": "b = P @ a",
            "capital_conservation": "sum(b) = sum(a)",
        },
        "files": {name: name for name in expected if name != "manifest.json"},
        "portable_paths_only": True,
    }
    atomic_write_json(manifest, destination / "manifest.json")
    return tuple(destination / name for name in expected)


def _print_result(result: SoftRelationshipAuditResult) -> None:
    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    columns = [
        "k",
        "fitted_count",
        "projected_count",
        "unsupported_count",
        "reconstruction_error",
        "temporal_subspace_stability",
        "temporal_membership_cosine",
        "temporal_decoder_overlap",
        "worst_robustness_subspace",
        "worst_robustness_membership_cosine",
        "worst_robustness_decoder_overlap",
        "mean_normalized_membership_entropy",
        "maximum_membership_mass_share",
        "sector_nmi_argmax_posthoc",
        "mean_decoder_sector_purity",
        "passes_evidence_gates",
        "relative_reconstruction_gain_to_next_k",
        "selection_eligible",
        "gate_failures",
    ]
    print(result.k_comparison.loc[:, columns].to_string(index=False))
    print("Robustness:")
    print(result.robustness.to_string(index=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit TRAIN-only soft PSX relationship representations"
    )
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run_soft_relationship_audit(parquet_path=args.parquet)
        _print_result(result)
        if args.output_dir is not None:
            outputs = write_soft_relationship_artifacts(
                result, args.output_dir, overwrite=args.overwrite
            )
            print("Artifacts:")
            for path in outputs:
                print(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)
        return 0 if result.summary.decision == READY_DECISION else 2
    except Exception as exc:
        print(f"Soft relationship audit failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKED_DECISION",
    "CANDIDATE_DIMENSIONS",
    "READY_DECISION",
    "SOFT_ALIGNMENT_VERSION",
    "SOFT_CONTRACT_VERSION",
    "SOFT_REPRESENTATION_VERSION",
    "ExpandedSoftRepresentation",
    "SoftPrototypeFit",
    "SoftRelationshipAuditResult",
    "SoftRelationshipError",
    "align_soft_prototypes",
    "authoritative_identity_hash",
    "compare_soft_fits",
    "decode_prototype_allocations",
    "deterministic_frame_hash",
    "expand_to_identity",
    "fit_soft_prototypes",
    "positive_correlation_affinity",
    "project_relationship_vector",
    "run_soft_relationship_audit",
    "sector_correspondence_diagnostics",
    "validate_candidate_dimensions",
    "write_soft_relationship_artifacts",
]
