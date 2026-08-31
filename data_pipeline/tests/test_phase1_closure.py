"""Offline tests for the deterministic Phase-1 closure decision."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import data_pipeline.src.phase1_closure as closure_module
from data_pipeline.src.phase1_closure import (
    FINAL_PHASE1_DECISION,
    HARD_CLUSTERING_DECISION,
    PHASE1_CLOSURE_ARTIFACT,
    Phase1ClosureError,
    SOFT_RELATIONSHIP_DECISION,
    decision_evidence_hash,
    load_phase1_closure,
    validate_phase1_closure,
)
from data_pipeline.src.soft_relationship_representation import (
    CANDIDATE_DIMENSIONS,
    MAX_DECODER_TOP_STOCK_WEIGHT,
    MAX_MEAN_SECTOR_PURITY,
    MAX_MEMBERSHIP_MASS_MULTIPLIER,
    MAX_NORMALIZED_MEMBERSHIP_ENTROPY,
    MAX_PLATEAU_RECONSTRUCTION_GAIN,
    MAX_RELATIONSHIP_CONDITION_NUMBER,
    MAX_SECTOR_NMI,
    MIN_MEDIAN_DECODER_EFFECTIVE_STOCKS,
    MIN_NORMALIZED_MEMBERSHIP_ENTROPY,
    MIN_ROBUSTNESS_DECODER_OVERLAP,
    MIN_ROBUSTNESS_MEMBERSHIP_COSINE,
    MIN_ROBUSTNESS_SUBSPACE_STABILITY,
    MIN_SUPPORTED_IDENTITY_FRACTION,
    MIN_TEMPORAL_DECODER_OVERLAP,
    MIN_TEMPORAL_MEMBERSHIP_COSINE,
    MIN_TEMPORAL_SUBSPACE_STABILITY,
)


def test_frozen_closure_loads_with_reconciled_negative_decisions() -> None:
    payload = load_phase1_closure()

    assert payload["hard_clustering_decision"] == HARD_CLUSTERING_DECISION
    assert payload["soft_relationship_decision"] == SOFT_RELATIONSHIP_DECISION
    assert payload["final_architecture_decision"] == FINAL_PHASE1_DECISION
    assert payload["universe"]["identity_count"] == 508
    assert payload["hard_clustering_evidence"]["specific_protocol_supported"] is False
    assert payload["soft_relationship_evidence"]["selected_k"] is None


def test_clustering_temporal_protocol_is_explicitly_not_rl_partition_v1() -> None:
    temporal = load_phase1_closure()["temporal_protocol"]

    assert temporal["training_start"] == "2016-07-26"
    assert temporal["training_end"] == "2023-08-03"
    assert temporal["fitting_partition"] == "TRAIN_ONLY_COMMON_CALENDAR"
    assert temporal["rl_partition_v1_applies"] is False
    assert "each symbol's own" in temporal["rl_partition_distinction"]


def test_validation_test_and_model_work_remain_excluded() -> None:
    payload = load_phase1_closure()
    safety = payload["safety"]

    assert payload["temporal_protocol"]["validation_observations_loaded_for_fitting"] is False
    assert payload["temporal_protocol"]["test_observations_loaded"] is False
    assert safety == {
        "validation_used_for_fitting": False,
        "test_status": "SEALED",
        "test_observations_loaded": False,
        "model_training_performed": False,
        "lead_agent_work_performed": False,
        "phase3_recurrent_agents_modified": False,
        "final_cluster_assignments_written": False,
        "soft_representation_frozen": False,
    }
    source = Path(closure_module.__file__).read_text(encoding="utf-8")
    assert "reinforcement_learning" not in source


def test_soft_acceptance_thresholds_match_predeclared_source_constants() -> None:
    thresholds = load_phase1_closure()["soft_relationship_acceptance_thresholds"]

    assert thresholds == {
        "minimum_supported_identity_fraction": MIN_SUPPORTED_IDENTITY_FRACTION,
        "minimum_temporal_subspace_stability": MIN_TEMPORAL_SUBSPACE_STABILITY,
        "minimum_temporal_membership_cosine": MIN_TEMPORAL_MEMBERSHIP_COSINE,
        "minimum_temporal_decoder_overlap": MIN_TEMPORAL_DECODER_OVERLAP,
        "minimum_robustness_subspace_stability": MIN_ROBUSTNESS_SUBSPACE_STABILITY,
        "minimum_robustness_membership_cosine": MIN_ROBUSTNESS_MEMBERSHIP_COSINE,
        "minimum_robustness_decoder_overlap": MIN_ROBUSTNESS_DECODER_OVERLAP,
        "maximum_sector_nmi": MAX_SECTOR_NMI,
        "maximum_mean_sector_purity": MAX_MEAN_SECTOR_PURITY,
        "maximum_membership_mass_multiplier": MAX_MEMBERSHIP_MASS_MULTIPLIER,
        "minimum_normalized_membership_entropy": MIN_NORMALIZED_MEMBERSHIP_ENTROPY,
        "maximum_normalized_membership_entropy": MAX_NORMALIZED_MEMBERSHIP_ENTROPY,
        "minimum_median_decoder_effective_stocks": MIN_MEDIAN_DECODER_EFFECTIVE_STOCKS,
        "maximum_decoder_top_stock_weight": MAX_DECODER_TOP_STOCK_WEIGHT,
        "maximum_relationship_condition_number": MAX_RELATIONSHIP_CONDITION_NUMBER,
        "maximum_plateau_reconstruction_gain": MAX_PLATEAU_RECONSTRUCTION_GAIN,
        "convergence_required": True,
    }
    assert tuple(load_phase1_closure()["soft_relationship_evidence"]["candidate_k"]) == CANDIDATE_DIMENSIONS


def test_evidence_hash_is_deterministic_and_ignores_only_archival_timestamp() -> None:
    payload = json.loads(PHASE1_CLOSURE_ARTIFACT.read_text(encoding="utf-8"))
    first = decision_evidence_hash(payload)
    second = decision_evidence_hash(deepcopy(payload))
    with_new_timestamp = deepcopy(payload)
    with_new_timestamp["frozen_at"] = "2099-01-01T00:00:00Z"

    assert first == second == payload["decision_evidence_hash"]
    assert decision_evidence_hash(with_new_timestamp) == first


def test_evidence_hash_changes_with_scientific_content() -> None:
    payload = load_phase1_closure()
    changed = deepcopy(payload)
    changed["soft_relationship_evidence"]["k8"]["temporal_subspace_stability"] = 0.9

    assert decision_evidence_hash(changed) != payload["decision_evidence_hash"]
    with pytest.raises(Phase1ClosureError, match="hash verification"):
        validate_phase1_closure(changed)


def test_closure_rejects_test_leakage_or_decision_drift() -> None:
    payload = load_phase1_closure()
    leaked = deepcopy(payload)
    leaked["temporal_protocol"]["test_observations_loaded"] = True
    leaked["decision_evidence_hash"] = decision_evidence_hash(leaked)
    with pytest.raises(Phase1ClosureError, match="TEST observations"):
        validate_phase1_closure(leaked)

    drifted = deepcopy(payload)
    drifted["final_architecture_decision"] = "ACCEPTED_HARD_CLUSTER_STRUCTURE"
    drifted["decision_evidence_hash"] = decision_evidence_hash(drifted)
    with pytest.raises(Phase1ClosureError, match="final_architecture_decision"):
        validate_phase1_closure(drifted)


def test_load_fails_closed_for_missing_or_malformed_artifact(tmp_path: Path) -> None:
    with pytest.raises(Phase1ClosureError, match="unavailable"):
        load_phase1_closure(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(Phase1ClosureError, match="must be an object"):
        load_phase1_closure(malformed)
