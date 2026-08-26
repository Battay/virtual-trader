"""Focused offline tests for the Milestone 7D soft-representation contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.clustering_market_mode import (
    FROZEN_TRAIN_END,
    FROZEN_TRAIN_START,
)
from data_pipeline.src.soft_relationship_representation import (
    CANDIDATE_DIMENSIONS,
    PROJECTION_MINIMUM_CORE_RELATIONSHIPS,
    SoftRelationshipError,
    align_soft_prototypes,
    decode_prototype_allocations,
    deterministic_frame_hash,
    expand_to_identity,
    fit_soft_prototypes,
    positive_correlation_affinity,
    run_soft_relationship_audit,
    sector_correspondence_diagnostics,
    validate_candidate_dimensions,
)


def _structured_correlation(size: int = 72, latent_size: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(20260826)
    latent = rng.uniform(0.05, 1.0, size=(size, latent_size))
    affinity = latent @ latent.T
    scales = np.sqrt(np.diag(affinity))
    correlation = affinity / np.outer(scales, scales)
    np.fill_diagonal(correlation, 1.0)
    symbols = [f"S{index:03d}" for index in range(size)]
    return pd.DataFrame(correlation, index=symbols, columns=symbols)


@pytest.fixture(scope="module")
def fitted_representation():
    return fit_soft_prototypes(_structured_correlation(), k=8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_grid_is_exact_and_cannot_expand_after_results() -> None:
    assert validate_candidate_dimensions(CANDIDATE_DIMENSIONS) == (
        8,
        10,
        12,
        15,
        20,
        25,
        30,
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_candidate_dimensions((*CANDIDATE_DIMENSIONS, 35))
    with pytest.raises(ValueError, match="exactly"):
        validate_candidate_dimensions(tuple(reversed(CANDIDATE_DIMENSIONS)))


def test_affinity_never_treats_missing_correlation_as_zero() -> None:
    correlation = pd.DataFrame(
        [[1.0, -0.3], [-0.3, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    affinity = positive_correlation_affinity(correlation)
    # A finite negative relationship has defined zero long-only affinity.
    assert affinity.loc["A", "B"] == 0.0

    correlation.loc["A", "B"] = np.nan
    correlation.loc["B", "A"] = np.nan
    with pytest.raises(SoftRelationshipError, match="complete correlations"):
        positive_correlation_affinity(correlation)


def test_fit_is_deterministic_nonnegative_and_soft_memberships_normalize() -> None:
    correlation = _structured_correlation()
    first = fit_soft_prototypes(correlation, k=8)
    second = fit_soft_prototypes(correlation, k=8)

    assert first.relationship_hash == second.relationship_hash
    assert first.membership_hash == second.membership_hash
    assert first.decoder_hash == second.decoder_hash
    assert np.allclose(first.relationship_vectors, second.relationship_vectors)
    assert (first.relationship_vectors.to_numpy() >= 0).all()
    assert (first.memberships.to_numpy() >= 0).all()
    assert np.allclose(first.memberships.sum(axis=1), 1.0)
    assert (first.memberships.max(axis=1) < 1.0 - 1e-12).any()


def test_decoder_is_nonnegative_normalized_and_conserves_capital(
    fitted_representation,
) -> None:
    decoder = fitted_representation.decoder
    assert (decoder.to_numpy() >= 0).all()
    assert np.allclose(decoder.sum(axis=0), 1.0)

    allocation = np.arange(1, 9, dtype="float64") / 36.0
    stock_budget = decode_prototype_allocations(decoder, allocation)
    assert (stock_budget >= 0).all()
    assert stock_budget.sum() == pytest.approx(allocation.sum(), abs=1e-10)


def test_alignment_is_deterministic_and_recovers_a_prototype_permutation(
    fitted_representation,
) -> None:
    fit = fitted_representation
    permutation = np.asarray([3, 0, 7, 1, 6, 5, 2, 4])
    candidate = replace(
        fit,
        relationship_vectors=fit.relationship_vectors.iloc[:, permutation].copy(),
        memberships=fit.memberships.iloc[:, permutation].copy(),
        decoder=fit.decoder.iloc[:, permutation].copy(),
        basis=fit.basis.iloc[permutation].copy(),
    )

    first_order, first_diagnostics = align_soft_prototypes(fit, candidate)
    second_order, second_diagnostics = align_soft_prototypes(fit, candidate)

    assert first_order.tolist() == np.argsort(permutation).tolist()
    assert first_order.tolist() == second_order.tolist()
    pd.testing.assert_frame_equal(first_diagnostics, second_diagnostics)
    assert np.allclose(first_diagnostics["decoder_cosine_similarity"], 1.0)


def test_sparse_stock_is_projected_or_explicitly_unsupported(
    fitted_representation,
) -> None:
    fit = fitted_representation
    identity = [*fit.symbols, "PROJECTED", "UNSUPPORTED"]
    correlations = pd.DataFrame(
        np.nan, index=identity, columns=fit.symbols, dtype="float64"
    )
    correlations.loc[list(fit.symbols), list(fit.symbols)] = (
        _structured_correlation().to_numpy()
    )
    # This is generated by the fitted decoder basis and is exactly projectable.
    target_coefficients = np.linspace(0.2, 1.0, fit.k)
    projected = target_coefficients @ fit.basis.to_numpy(dtype="float64")
    correlations.loc["PROJECTED", list(fit.symbols)] = projected
    insufficient = PROJECTION_MINIMUM_CORE_RELATIONSHIPS - 1
    correlations.loc[
        "UNSUPPORTED", list(fit.symbols[:insufficient])
    ] = projected[:insufficient]

    expanded = expand_to_identity(fit, correlations, identity)
    status = expanded.eligibility.set_index("symbol")

    assert len(status) == len(identity)
    assert status.index.is_unique
    assert status.loc["PROJECTED", "status"] == "projected"
    assert status.loc["UNSUPPORTED", "status"] == "unsupported"
    assert (
        status.loc["UNSUPPORTED", "reason"] == "insufficient_core_relationships"
    )
    assert expanded.fitted_count == len(fit.symbols)
    assert expanded.projected_count == 1
    assert expanded.unsupported_count == 1
    assert np.allclose(expanded.decoder.sum(axis=0), 1.0)


def test_sector_values_are_posthoc_and_do_not_change_primary_fit(
    fitted_representation,
) -> None:
    fit = fitted_representation
    correlations = _structured_correlation()
    expanded = expand_to_identity(fit, correlations, fit.symbols)
    alternating = {
        symbol: ("BANK" if index % 2 else "TEXTILE")
        for index, symbol in enumerate(fit.symbols)
    }
    all_one_sector = {symbol: "ONE" for symbol in fit.symbols}

    first = sector_correspondence_diagnostics(expanded, alternating)
    second = sector_correspondence_diagnostics(expanded, all_one_sector)

    assert first != second
    assert expanded.representation_hash == deterministic_frame_hash(
        fit.relationship_vectors
    )
    assert expanded.membership_hash == fit.membership_hash
    assert expanded.decoder_hash == fit.decoder_hash


def test_representation_hash_is_order_and_value_sensitive() -> None:
    frame = pd.DataFrame(
        [[0.25, 0.75], [0.60, 0.40]],
        index=["A", "B"],
        columns=["prototype_01", "prototype_02"],
    )
    same = frame.copy(deep=True)
    changed = frame.copy(deep=True)
    changed.loc["A", "prototype_01"] += 0.01

    assert deterministic_frame_hash(frame) == deterministic_frame_hash(same)
    assert deterministic_frame_hash(frame) != deterministic_frame_hash(changed)
    assert deterministic_frame_hash(frame) != deterministic_frame_hash(
        frame.iloc[::-1]
    )


def test_audit_requests_only_frozen_train_and_does_not_mutate_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import data_pipeline.src.soft_relationship_representation as module

    parquet = tmp_path / "market.parquet"
    parquet.write_bytes(b"immutable-source-sentinel")
    before = _sha256(parquet)
    symbols = [f"S{index:03d}" for index in range(40)]
    identity = pd.DataFrame(
        {
            "symbol": symbols,
            "company_name": symbols,
            "sector": "SECTOR",
            "security_type": "COMMON_EQUITY",
            "source": "fixture",
            "snapshot_date": "2026-08-05",
        }
    )
    captured: dict[str, object] = {}

    class BoundaryObserved(RuntimeError):
        pass

    def intercept(path, requested_symbols, *, training_start, training_end):
        captured.update(
            path=path,
            symbols=tuple(requested_symbols),
            training_start=training_start,
            training_end=training_end,
        )
        raise BoundaryObserved

    monkeypatch.setattr(
        module, "load_authoritative_current_equity_identity", lambda **_: identity
    )
    monkeypatch.setattr(module, "resolve_market_parquet_path", lambda _: parquet)
    monkeypatch.setattr(module, "load_train_only_market_values", intercept)

    with pytest.raises(BoundaryObserved):
        run_soft_relationship_audit(parquet_path=parquet)

    assert captured == {
        "path": parquet,
        "symbols": tuple(symbols),
        "training_start": FROZEN_TRAIN_START,
        "training_end": FROZEN_TRAIN_END,
    }
    assert _sha256(parquet) == before

