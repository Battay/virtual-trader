"""Validation for the Phase-2 Lead-Agent architecture audit artifact.

This module deliberately validates documentation/configuration only.  It does
not load market observations, construct an environment, train an agent, or
write model state.  Executable Phase-2 contracts belong to later milestones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PROJECT_ROOT
from .phase1_closure import (
    FINAL_PHASE1_DECISION,
    PHASE1_CLOSURE_VERSION,
    Phase1ClosureError,
    load_phase1_closure,
)


PHASE2_ARCHITECTURE_VERSION = "phase2_lead_agent_architecture_audit_v1"
PHASE2_DATA_CONTRACT_PROPOSAL = "lead_agent_market_macro_contract_v1"
PHASE2_TEMPORAL_POLICY_PROPOSAL = "lead_agent_common_calendar_split_v1"
PHASE2_ENVIRONMENT_PROPOSAL = "lead_agent_market_risk_env_v1"
PHASE2_ACTION_PROPOSAL = "lead_agent_scalar_market_exposure_v1"
PHASE2_REWARD_PROPOSAL = "lead_agent_net_growth_reward_v1"
PHASE2_ARCHITECTURE_DECISION = "READY_PHASE2_ARCHITECTURE"
PHASE2_TRAINING_READINESS = "BLOCKED_PENDING_P2_2_DATA_CONTRACT"
PHASE2_ARCHITECTURE_ARTIFACT = (
    PROJECT_ROOT
    / "docs"
    / "config"
    / "phase2_lead_agent_architecture_audit_v1.json"
)
_HASH_EXCLUDED_FIELDS = frozenset({"generated_at", "architecture_evidence_hash"})
_ALLOWED_FEATURE_STATUSES = frozenset(
    {"AVAILABLE_DERIVABLE", "MISSING_BLOCKER", "PROPOSED"}
)
_FORBIDDEN_FEATURE_CATEGORIES = frozenset(
    {"cluster", "sector", "soft_relationship", "symbol"}
)
_REQUIRED_GAP_CAPABILITIES = frozenset(
    {
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
)


class Phase2ArchitectureError(RuntimeError):
    """Raised when the Phase-2 architecture audit has drifted or is unsafe."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def architecture_evidence_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash of nonvolatile architecture evidence."""

    evidence = {
        key: value
        for key, value in payload.items()
        if key not in _HASH_EXCLUDED_FIELDS
    }
    return hashlib.sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()


def _require_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise Phase2ArchitectureError(f"Phase-2 {field} is missing")
    return value


def validate_phase2_architecture(payload: Mapping[str, Any]) -> None:
    """Fail closed on leakage, cluster dependence, or architecture drift."""

    expected = {
        "artifact_version": PHASE2_ARCHITECTURE_VERSION,
        "phase": "PHASE_2",
        "milestone": "P2.1",
        "architecture_decision": PHASE2_ARCHITECTURE_DECISION,
        "training_readiness": PHASE2_TRAINING_READINESS,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise Phase2ArchitectureError(f"Phase-2 {field} is incompatible")

    inherited = _require_mapping(payload, "phase1_inherited_constraint")
    if inherited.get("phase1_decision") != FINAL_PHASE1_DECISION:
        raise Phase2ArchitectureError("Phase-1 decision is incompatible")
    if inherited.get("source_artifact_version") != PHASE1_CLOSURE_VERSION:
        raise Phase2ArchitectureError("Phase-1 artifact version is incompatible")
    try:
        phase1 = load_phase1_closure()
    except Phase1ClosureError as exc:
        raise Phase2ArchitectureError(
            "Tracked Phase-1 closure is unavailable or incompatible"
        ) from exc
    if (
        inherited.get("source_decision_evidence_hash")
        != phase1.get("decision_evidence_hash")
    ):
        raise Phase2ArchitectureError("Phase-1 evidence identity is incompatible")
    if inherited.get("requires_clusters") is not False:
        raise Phase2ArchitectureError("Lead Agent cannot require hard clusters")
    if inherited.get("requires_soft_prototypes") is not False:
        raise Phase2ArchitectureError("Lead Agent cannot require soft prototypes")
    if inherited.get("sectors_relabelled_as_clusters") is not False:
        raise Phase2ArchitectureError("Sectors cannot be relabelled as clusters")

    action = _require_mapping(payload, "recommended_action_space")
    if action.get("contract_version") != PHASE2_ACTION_PROPOSAL:
        raise Phase2ArchitectureError("Phase-2 action contract changed")
    if action.get("gymnasium_space") != "Box":
        raise Phase2ArchitectureError("Lead Agent action must be continuous Box")
    if action.get("shape") != [1] or action.get("low") != 0.0 or action.get("high") != 1.0:
        raise Phase2ArchitectureError("Lead Agent action must be scalar [0,1]")
    if action.get("per_symbol_allocation") is not False:
        raise Phase2ArchitectureError("Lead Agent cannot allocate per symbol")
    if action.get("cluster_allocation") is not False:
        raise Phase2ArchitectureError("Lead Agent cannot allocate to clusters")
    if action.get("shorting_allowed") is not False or action.get("leverage_allowed") is not False:
        raise Phase2ArchitectureError("Initial Lead Agent must be long-only and unlevered")

    reward = _require_mapping(payload, "recommended_reward")
    if reward.get("contract_version") != PHASE2_REWARD_PROPOSAL:
        raise Phase2ArchitectureError("Phase-2 reward contract changed")
    if reward.get("primary_objective") != "net_log_portfolio_growth":
        raise Phase2ArchitectureError("Primary reward is not auditable net growth")
    if reward.get("validation_tuning_used") is not False:
        raise Phase2ArchitectureError("Reward was selected using validation")

    temporal = _require_mapping(payload, "temporal_policy")
    if temporal.get("policy_version") != PHASE2_TEMPORAL_POLICY_PROPOSAL:
        raise Phase2ArchitectureError("Phase-2 temporal policy changed")
    if temporal.get("calendar_scope") != "COMMON_MARKET_MACRO_CALENDAR":
        raise Phase2ArchitectureError("Lead Agent requires a common calendar")
    if temporal.get("rl_partition_v1_applies") is not False:
        raise Phase2ArchitectureError("Symbol RL and Lead-Agent splits were conflated")
    if temporal.get("test_status") != "SEALED":
        raise Phase2ArchitectureError("Phase-2 TEST is not sealed")
    if temporal.get("test_observations_loaded") is not False:
        raise Phase2ArchitectureError("Phase-2 TEST observations were loaded")
    if temporal.get("validation_observations_used_for_fitting") is not False:
        raise Phase2ArchitectureError("VALIDATION entered Phase-2 fitting")
    if temporal.get("normalization_fit_partition") != "TRAIN_ONLY":
        raise Phase2ArchitectureError("Phase-2 normalization is not TRAIN-only")
    if temporal.get("exact_boundaries_frozen") is not False:
        raise Phase2ArchitectureError("P2.1 must not invent final split boundaries")
    if temporal.get("split_fractions") != {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }:
        raise Phase2ArchitectureError("Phase-2 split proposal changed")

    contract = _require_mapping(payload, "proposed_data_contract")
    if contract.get("contract_version") != PHASE2_DATA_CONTRACT_PROPOSAL:
        raise Phase2ArchitectureError("Phase-2 data-contract proposal changed")
    if contract.get("feature_shape_frozen") is not False:
        raise Phase2ArchitectureError("P2.1 cannot freeze unavailable macro features")
    if contract.get("normalization_fit_partition") != "TRAIN_ONLY":
        raise Phase2ArchitectureError("Lead-Agent scaler must be fitted on TRAIN only")
    if contract.get("validation_rows_used_for_fitting") is not False:
        raise Phase2ArchitectureError("VALIDATION rows entered data-contract fitting")
    if contract.get("test_rows_loaded") is not False:
        raise Phase2ArchitectureError("TEST rows entered the data-contract audit")
    timing = _require_mapping(contract, "decision_timing")
    if timing.get("market_information_latest_allowed") != "previous_trading_session":
        raise Phase2ArchitectureError("Same-session market observations are unsafe")
    if timing.get("macro_availability_rule") != "release_timestamp_at_or_before_decision_cutoff":
        raise Phase2ArchitectureError("Macro release timing is not point-in-time safe")
    if timing.get("action_return_interval") != (
        "benchmark close t to close t+1 after decision, with predeclared "
        "turnover friction"
    ):
        raise Phase2ArchitectureError("Lead-Agent action/return timing changed")
    if timing.get("same_session_close_visible") is not False:
        raise Phase2ArchitectureError("Same-session close leaked into observation")

    features = contract.get("features")
    if not isinstance(features, list) or not features:
        raise Phase2ArchitectureError("Phase-2 feature proposal is empty")
    required_feature_fields = {
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
    feature_names: list[str] = []
    for feature in features:
        if not isinstance(feature, Mapping) or required_feature_fields.difference(feature):
            raise Phase2ArchitectureError("Phase-2 feature schema is incomplete")
        if any(
            not isinstance(feature.get(field), str)
            or not str(feature[field]).strip()
            for field in required_feature_fields
        ):
            raise Phase2ArchitectureError("Phase-2 feature metadata is empty")
        name = str(feature["name"])
        feature_names.append(name)
        if feature.get("status") not in _ALLOWED_FEATURE_STATUSES:
            raise Phase2ArchitectureError("Phase-2 feature status is incompatible")
        if str(feature.get("category")).lower() in _FORBIDDEN_FEATURE_CATEGORIES:
            raise Phase2ArchitectureError("Rejected relationship identity entered features")
        lowered_name = name.lower()
        if any(token in lowered_name for token in ("cluster", "prototype", "sector_id")):
            raise Phase2ArchitectureError("Rejected relationship feature entered contract")
        if feature.get("category") == "macro" and feature.get("status") == "AVAILABLE":
            raise Phase2ArchitectureError("P2.1 found no available macro series")
        if (
            feature.get("category") == "macro"
            and feature.get("status") != "MISSING_BLOCKER"
        ):
            raise Phase2ArchitectureError("Macro data cannot be marked available in P2.1")
    if len(feature_names) != len(set(feature_names)):
        raise Phase2ArchitectureError("Phase-2 feature names are not unique")

    environment = _require_mapping(payload, "environment_proposal")
    if environment.get("environment_version") != PHASE2_ENVIRONMENT_PROPOSAL:
        raise Phase2ArchitectureError("Phase-2 environment proposal changed")
    if environment.get("implemented") is not False:
        raise Phase2ArchitectureError("P2.1 must not implement the environment")
    if environment.get("test_partition_accepted") is not False:
        raise Phase2ArchitectureError("Environment must reject TEST")
    if environment.get("phase3_communication") is not False:
        raise Phase2ArchitectureError("Phase-3 communication entered Phase 2")
    if environment.get("recurrent_policy_required") is not False:
        raise Phase2ArchitectureError("P2.1 initial environment cannot require recurrence")

    algorithm = _require_mapping(payload, "algorithm_recommendation")
    if algorithm.get("recommendation") != "BOUNDED_PPO_VS_SAC_COMPARISON":
        raise Phase2ArchitectureError("Phase-2 algorithm recommendation changed")
    if algorithm.get("recurrent_initial_baseline") is not False:
        raise Phase2ArchitectureError("Initial Lead-Agent baseline must remain nonrecurrent")
    if algorithm.get("experiment_seed_set") != [42, 43, 44]:
        raise Phase2ArchitectureError("Phase-2 algorithm seed protocol changed")
    if algorithm.get("validation_hyperparameter_tuning") is not False:
        raise Phase2ArchitectureError("Algorithm settings were tuned on VALIDATION")

    gap_matrix = payload.get("gap_matrix")
    if not isinstance(gap_matrix, list):
        raise Phase2ArchitectureError("Phase-2 gap matrix is missing")
    gap_capabilities = [
        str(item.get("capability"))
        for item in gap_matrix
        if isinstance(item, Mapping)
    ]
    if len(gap_capabilities) != len(gap_matrix) or set(gap_capabilities) != _REQUIRED_GAP_CAPABILITIES:
        raise Phase2ArchitectureError("Phase-2 gap matrix is incomplete")
    if len(gap_capabilities) != len(set(gap_capabilities)):
        raise Phase2ArchitectureError("Phase-2 gap matrix contains duplicates")

    milestones = payload.get("phase2_milestones")
    if not isinstance(milestones, list) or [
        item.get("milestone") if isinstance(item, Mapping) else None
        for item in milestones
    ] != ["P2.1", "P2.2", "P2.3", "P2.4", "P2.5", "P2.6"]:
        raise Phase2ArchitectureError("Phase-2 milestone sequence is incomplete")

    market_inventory = _require_mapping(payload, "market_level_data_inventory")
    index_inventory = _require_mapping(market_inventory, "official_index_master")
    if index_inventory.get("rows") != 5012 or set(index_inventory.get("index_codes", ())) != {
        "KSE100",
        "KSE30",
        "KMI30",
        "ALLSHR",
    }:
        raise Phase2ArchitectureError("Official index inventory changed")
    macro_inventory = _require_mapping(payload, "macro_data_inventory")
    if macro_inventory.get("local_authoritative_macro_dataset") != "MISSING":
        raise Phase2ArchitectureError("P2.1 macro inventory changed")

    safety = _require_mapping(payload, "safety")
    required_false = (
        "lead_agent_training_performed",
        "test_observations_loaded",
        "phase3_agents_modified",
        "joint_fine_tuning_started",
        "phase4_integration_started",
        "network_fetch_performed",
        "model_artifacts_modified",
        "training_authorized",
    )
    if any(safety.get(field) is not False for field in required_false):
        raise Phase2ArchitectureError("Phase-2 safety declaration is not fail-closed")

    serialized = _canonical_json(payload)
    if "/Users/" in serialized or "\\\\Users\\" in serialized:
        raise Phase2ArchitectureError("Phase-2 evidence contains a machine-specific path")

    recorded_hash = payload.get("architecture_evidence_hash")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise Phase2ArchitectureError("Phase-2 architecture hash is malformed")
    if recorded_hash != architecture_evidence_hash(payload):
        raise Phase2ArchitectureError("Phase-2 architecture hash verification failed")


def load_phase2_architecture(
    path: str | Path = PHASE2_ARCHITECTURE_ARTIFACT,
) -> dict[str, Any]:
    """Load and validate the tracked P2.1 audit artifact."""

    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2ArchitectureError(
            f"Phase-2 architecture artifact is unavailable: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase2ArchitectureError("Phase-2 architecture artifact must be an object")
    # Re-verify the tracked Phase-1 decision without loading market values.
    try:
        phase1 = load_phase1_closure()
    except Phase1ClosureError as exc:
        raise Phase2ArchitectureError(
            "Tracked Phase-1 closure is unavailable or incompatible"
        ) from exc
    if phase1["final_architecture_decision"] != FINAL_PHASE1_DECISION:
        raise Phase2ArchitectureError("Tracked Phase-1 closure is incompatible")
    validate_phase2_architecture(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the Phase-2 Lead-Agent architecture audit artifact."
    )
    parser.add_argument("--artifact", default=str(PHASE2_ARCHITECTURE_ARTIFACT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = load_phase2_architecture(args.artifact)
    except (Phase2ArchitectureError, Phase1ClosureError) as exc:
        print(f"BLOCKED_PHASE2_ARCHITECTURE: {exc}")
        return 2
    print(
        f"{payload['architecture_decision']} "
        f"({payload['architecture_evidence_hash']}); "
        f"training={payload['training_readiness']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = (
    "PHASE2_ACTION_PROPOSAL",
    "PHASE2_ARCHITECTURE_ARTIFACT",
    "PHASE2_ARCHITECTURE_DECISION",
    "PHASE2_ARCHITECTURE_VERSION",
    "PHASE2_DATA_CONTRACT_PROPOSAL",
    "PHASE2_ENVIRONMENT_PROPOSAL",
    "PHASE2_REWARD_PROPOSAL",
    "PHASE2_TEMPORAL_POLICY_PROPOSAL",
    "PHASE2_TRAINING_READINESS",
    "Phase2ArchitectureError",
    "architecture_evidence_hash",
    "load_phase2_architecture",
    "main",
    "validate_phase2_architecture",
)
