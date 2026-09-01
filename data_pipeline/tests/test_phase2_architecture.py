"""Offline safety tests for the Phase-2 Lead-Agent architecture audit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import data_pipeline.src.phase2_architecture as architecture_module
from data_pipeline.src.phase1_closure import (
    FINAL_PHASE1_DECISION,
    PHASE1_CLOSURE_VERSION,
    Phase1ClosureError,
    load_phase1_closure,
)
from data_pipeline.src.phase2_architecture import (
    PHASE2_ACTION_PROPOSAL,
    PHASE2_ARCHITECTURE_ARTIFACT,
    PHASE2_ARCHITECTURE_DECISION,
    PHASE2_DATA_CONTRACT_PROPOSAL,
    PHASE2_ENVIRONMENT_PROPOSAL,
    PHASE2_REWARD_PROPOSAL,
    PHASE2_TEMPORAL_POLICY_PROPOSAL,
    PHASE2_TRAINING_READINESS,
    Phase2ArchitectureError,
    architecture_evidence_hash,
    load_phase2_architecture,
    validate_phase2_architecture,
)


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["architecture_evidence_hash"] = architecture_evidence_hash(payload)
    return payload


def _all_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [text for child in value.values() for text in _all_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _all_strings(child)]
    return [value] if isinstance(value, str) else []


def test_p21_artifact_is_architecture_ready_but_training_blocked() -> None:
    payload = load_phase2_architecture()

    assert payload["phase"] == "PHASE_2"
    assert payload["milestone"] == "P2.1"
    assert payload["architecture_decision"] == PHASE2_ARCHITECTURE_DECISION
    assert payload["training_readiness"] == PHASE2_TRAINING_READINESS
    assert PHASE2_ARCHITECTURE_DECISION == "READY_PHASE2_ARCHITECTURE"
    assert PHASE2_TRAINING_READINESS == "BLOCKED_PENDING_P2_2_DATA_CONTRACT"


def test_phase1_rejection_is_an_exact_binding_constraint() -> None:
    payload = load_phase2_architecture()
    inherited = payload["phase1_inherited_constraint"]
    phase1 = load_phase1_closure()

    assert inherited["phase1_decision"] == FINAL_PHASE1_DECISION
    assert inherited["source_artifact_version"] == PHASE1_CLOSURE_VERSION
    assert inherited["source_decision_evidence_hash"] == phase1["decision_evidence_hash"]
    assert inherited["requires_clusters"] is False
    assert inherited["requires_soft_prototypes"] is False
    assert inherited["sectors_relabelled_as_clusters"] is False


@pytest.mark.parametrize(
    ("field", "unsafe_value", "message"),
    (
        ("requires_clusters", True, "hard clusters"),
        ("requires_soft_prototypes", True, "soft prototypes"),
        ("sectors_relabelled_as_clusters", True, "relabelled"),
    ),
)
def test_cluster_dependency_drift_fails_closed(
    field: str,
    unsafe_value: bool,
    message: str,
) -> None:
    payload = deepcopy(load_phase2_architecture())
    payload["phase1_inherited_constraint"][field] = unsafe_value
    _rehash(payload)

    with pytest.raises(Phase2ArchitectureError, match=message):
        validate_phase2_architecture(payload)


def test_scalar_action_is_long_only_and_relationship_independent() -> None:
    action = load_phase2_architecture()["recommended_action_space"]

    assert action["contract_version"] == PHASE2_ACTION_PROPOSAL
    assert action["gymnasium_space"] == "Box"
    assert action["shape"] == [1]
    assert (action["low"], action["high"], action["dtype"]) == (
        0.0,
        1.0,
        "float32",
    )
    assert action["shorting_allowed"] is False
    assert action["leverage_allowed"] is False
    assert action["per_symbol_allocation"] is False
    assert action["cluster_allocation"] is False


def test_common_calendar_is_distinct_from_phase1_and_symbol_rl() -> None:
    temporal = load_phase2_architecture()["temporal_policy"]

    assert temporal["policy_version"] == PHASE2_TEMPORAL_POLICY_PROPOSAL
    assert temporal["calendar_scope"] == "COMMON_MARKET_MACRO_CALENDAR"
    assert temporal["split_fractions"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert temporal["phase1_common_window_reused"] is False
    assert temporal["rl_partition_v1_applies"] is False
    assert temporal["exact_boundaries_frozen"] is False


def test_validation_normalization_and_test_boundaries_are_fail_closed() -> None:
    payload = load_phase2_architecture()
    temporal = payload["temporal_policy"]
    contract = payload["proposed_data_contract"]

    assert temporal["validation_observations_used_for_fitting"] is False
    assert temporal["normalization_fit_partition"] == "TRAIN_ONLY"
    assert temporal["test_status"] == "SEALED"
    assert temporal["test_observations_loaded"] is False
    assert contract["normalization_fit_partition"] == "TRAIN_ONLY"
    assert contract["validation_rows_used_for_fitting"] is False
    assert contract["test_rows_loaded"] is False

    leaked = deepcopy(payload)
    leaked["proposed_data_contract"]["validation_rows_used_for_fitting"] = True
    _rehash(leaked)
    with pytest.raises(Phase2ArchitectureError, match="VALIDATION rows"):
        validate_phase2_architecture(leaked)


def test_point_in_time_decision_and_return_interval_are_causal() -> None:
    payload = load_phase2_architecture()
    timing = payload["proposed_data_contract"]["decision_timing"]

    assert timing["market_information_latest_allowed"] == "previous_trading_session"
    assert (
        timing["macro_availability_rule"]
        == "release_timestamp_at_or_before_decision_cutoff"
    )
    assert timing["same_session_close_visible"] is False
    assert timing["action_return_interval"].startswith("benchmark close t to close t+1")
    assert "close-t to close-t+1" in payload["environment_proposal"]["execution_proxy"]


def test_feature_schema_is_unique_causal_and_has_no_relationship_identity() -> None:
    payload = load_phase2_architecture()
    features = payload["proposed_data_contract"]["features"]
    required = {
        "name",
        "category",
        "source",
        "frequency",
        "availability_lag",
        "transformation",
        "normalization_scope",
        "missing_data_treatment",
        "causal_property",
        "status",
    }
    names = [feature["name"] for feature in features]

    assert len(names) == len(set(names))
    assert all(required <= set(feature) for feature in features)
    assert not {"cluster", "sector", "soft_relationship", "symbol"}.intersection(
        {feature["category"] for feature in features}
    )
    assert not any("cluster" in name or "prototype" in name for name in names)

    duplicate = deepcopy(payload)
    duplicate["proposed_data_contract"]["features"][1]["name"] = names[0]
    _rehash(duplicate)
    with pytest.raises(Phase2ArchitectureError, match="not unique"):
        validate_phase2_architecture(duplicate)


def test_missing_macro_data_blocks_contract_freeze_and_training() -> None:
    payload = load_phase2_architecture()
    contract = payload["proposed_data_contract"]
    macro_features = [
        feature for feature in contract["features"] if feature["category"] == "macro"
    ]

    assert payload["macro_data_inventory"]["local_authoritative_macro_dataset"] == "MISSING"
    assert macro_features
    assert {feature["status"] for feature in macro_features} == {"MISSING_BLOCKER"}
    assert contract["feature_shape_frozen"] is False
    assert payload["training_readiness"] == "BLOCKED_PENDING_P2_2_DATA_CONTRACT"


def test_reward_is_auditable_and_not_validation_tuned() -> None:
    payload = load_phase2_architecture()
    reward = payload["recommended_reward"]

    assert reward["contract_version"] == PHASE2_REWARD_PROPOSAL
    assert reward["primary_objective"] == "net_log_portfolio_growth"
    assert "absolute_change_in_target_exposure" in reward["turnover_cost_model"]
    assert reward["drawdown_penalty_weight_v1"] == 0.0
    assert reward["volatility_penalty_weight_v1"] == 0.0
    assert reward["validation_tuning_used"] is False
    assert reward["coefficient_changes_require_predeclaration"] is True


def test_environment_is_proposal_only_and_cannot_open_test_or_phase3() -> None:
    environment = load_phase2_architecture()["environment_proposal"]

    assert environment["environment_version"] == PHASE2_ENVIRONMENT_PROPOSAL
    assert environment["implemented"] is False
    assert environment["test_partition_accepted"] is False
    assert environment["phase3_communication"] is False
    assert environment["recurrent_policy_required"] is False


def test_algorithm_comparison_is_bounded_fair_and_nonrecurrent() -> None:
    algorithm = load_phase2_architecture()["algorithm_recommendation"]

    assert algorithm["recommendation"] == "BOUNDED_PPO_VS_SAC_COMPARISON"
    assert algorithm["experiment_seed_set"] == [42, 43, 44]
    assert "Same TRAIN environment transitions" in algorithm["comparison_protocol"]
    assert algorithm["validation_hyperparameter_tuning"] is False
    assert algorithm["recurrent_initial_baseline"] is False
    assert algorithm["selected_algorithm"] is None


def test_gap_matrix_and_phase2_sequence_are_complete() -> None:
    payload = load_phase2_architecture()
    capabilities = {item["capability"] for item in payload["gap_matrix"]}
    expected = {
        "Official PSX index dataset",
        "Pakistan macro dataset",
        "Point-in-time alignment",
        "Lead-Agent feature engineering",
        "Normalization",
        "Lead-Agent environment",
        "Reward",
        "Action space",
        "PPO",
        "SAC",
        "Persistence/registry",
        "Validation and baselines",
        "Dashboard controls",
    }

    assert capabilities == expected
    assert [item["milestone"] for item in payload["phase2_milestones"]] == [
        "P2.1",
        "P2.2",
        "P2.3",
        "P2.4",
        "P2.5",
        "P2.6",
    ]
    assert not any("P3" in text for text in _all_strings(payload["phase2_milestones"]))


def test_architecture_hash_is_deterministic_and_ignores_only_timestamp() -> None:
    payload = json.loads(PHASE2_ARCHITECTURE_ARTIFACT.read_text(encoding="utf-8"))
    first = architecture_evidence_hash(payload)
    changed_timestamp = deepcopy(payload)
    changed_timestamp["generated_at"] = "2099-01-01T00:00:00Z"

    assert first == architecture_evidence_hash(deepcopy(payload))
    assert first == payload["architecture_evidence_hash"]
    assert architecture_evidence_hash(changed_timestamp) == first

    changed = deepcopy(payload)
    changed["recommended_action_space"]["high"] = 0.9
    assert architecture_evidence_hash(changed) != first


def test_loader_fails_closed_for_missing_malformed_or_tampered_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(Phase2ArchitectureError, match="unavailable"):
        load_phase2_architecture(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(Phase2ArchitectureError, match="must be an object"):
        load_phase2_architecture(malformed)

    tampered = deepcopy(load_phase2_architecture())
    tampered["recommended_reward"]["primary_objective"] = "validation_sharpe"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Phase2ArchitectureError, match="Primary reward"):
        _rehash(tampered)
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        load_phase2_architecture(tampered_path)


def test_main_reports_phase1_failure_as_phase2_block(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_phase1() -> dict[str, object]:
        raise Phase1ClosureError("simulated closure drift")

    monkeypatch.setattr(architecture_module, "load_phase1_closure", fail_phase1)

    assert architecture_module.main([]) == 2
    assert "BLOCKED_PHASE2_ARCHITECTURE" in capsys.readouterr().out


def test_artifact_is_portable_and_audit_module_has_no_execution_path() -> None:
    payload = load_phase2_architecture()
    strings = _all_strings(payload)
    source = Path(architecture_module.__file__).read_text(encoding="utf-8")

    assert not any("/Users/" in value or "\\Users\\" in value for value in strings)
    for forbidden in (
        ".learn(",
        "read_parquet(",
        "read_csv(",
        "requests.",
        "load_rl_partition(",
        "load_recurrent_partition(",
        "stable_baselines3",
        "model_registry",
        "to_parquet(",
        "to_csv(",
    ):
        assert forbidden not in source
    assert payload["safety"] == {
        "lead_agent_training_performed": False,
        "test_observations_loaded": False,
        "phase3_agents_modified": False,
        "joint_fine_tuning_started": False,
        "phase4_integration_started": False,
        "network_fetch_performed": False,
        "model_artifacts_modified": False,
        "training_authorized": False,
    }
