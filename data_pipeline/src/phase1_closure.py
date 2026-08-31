"""Validation helpers for the frozen Phase-1 relationship-structure decision.

The closure artifact consolidates existing clustering evidence; it does not fit
clustering models, load market observations, or create downstream assignments.
The deterministic evidence identity deliberately excludes only the archival
timestamp so the scientific decision can be reproduced byte-independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .clustering_market_mode import (
    FINAL_DECISION as MARKET_MODE_DECISION,
    FROZEN_TRAIN_END,
    FROZEN_TRAIN_START,
    MARKET_MODE_AUDIT_VERSION,
)
from .clustering_methodology import CLUSTERING_METHODOLOGY_VERSION
from .clustering_multiview import (
    FINAL_DECISION as MULTIVIEW_DECISION,
    MULTIVIEW_AUDIT_VERSION,
)
from .clustering_protocol import (
    PROTOCOL_SELECTION_DECISION,
    PROTOCOL_SELECTION_VERSION,
)
from .config import PROJECT_ROOT
from .identity_universe_policy import (
    FROZEN_RESEARCH_IDENTITY_COUNT,
    FROZEN_RESEARCH_IDENTITY_SNAPSHOT,
    FROZEN_RESEARCH_UNIVERSE_HASH,
)
from .soft_relationship_representation import (
    BLOCKED_DECISION,
    SOFT_CONTRACT_VERSION,
    SOFT_REPRESENTATION_VERSION,
)


PHASE1_CLOSURE_VERSION = "phase1_clustering_closure_v1"
HARD_CLUSTERING_DECISION = "REJECT_HARD_CLUSTERING"
SOFT_RELATIONSHIP_DECISION = "REJECT_SOFT_RELATIONSHIP"
FINAL_PHASE1_DECISION = "REJECTED_CLUSTER_STRUCTURE"
PHASE1_CLOSURE_ARTIFACT = (
    PROJECT_ROOT / "docs" / "config" / "phase1_clustering_closure_v1.json"
)
_HASH_EXCLUDED_FIELDS = frozenset({"frozen_at", "decision_evidence_hash"})


class Phase1ClosureError(RuntimeError):
    """Raised when the frozen Phase-1 decision is absent or has drifted."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def decision_evidence_hash(payload: Mapping[str, Any]) -> str:
    """Return the deterministic identity of nonvolatile decision evidence."""

    evidence = {
        key: value
        for key, value in payload.items()
        if key not in _HASH_EXCLUDED_FIELDS
    }
    return hashlib.sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()


def validate_phase1_closure(payload: Mapping[str, Any]) -> None:
    """Fail closed when the versioned Phase-1 artifact is inconsistent."""

    expected = {
        "artifact_version": PHASE1_CLOSURE_VERSION,
        "phase": "PHASE_1",
        "hard_clustering_decision": HARD_CLUSTERING_DECISION,
        "soft_relationship_decision": SOFT_RELATIONSHIP_DECISION,
        "final_architecture_decision": FINAL_PHASE1_DECISION,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise Phase1ClosureError(f"Phase-1 closure {field} is incompatible")

    universe = payload.get("universe")
    if not isinstance(universe, Mapping):
        raise Phase1ClosureError("Phase-1 closure universe is missing")
    if universe.get("identity_count") != FROZEN_RESEARCH_IDENTITY_COUNT:
        raise Phase1ClosureError("Phase-1 identity count changed")
    if universe.get("universe_hash") != FROZEN_RESEARCH_UNIVERSE_HASH:
        raise Phase1ClosureError("Phase-1 identity hash changed")
    if universe.get("snapshot_date") != FROZEN_RESEARCH_IDENTITY_SNAPSHOT:
        raise Phase1ClosureError("Phase-1 identity snapshot changed")

    temporal = payload.get("temporal_protocol")
    if not isinstance(temporal, Mapping):
        raise Phase1ClosureError("Phase-1 temporal protocol is missing")
    if temporal.get("training_start") != FROZEN_TRAIN_START:
        raise Phase1ClosureError("Phase-1 TRAIN start changed")
    if temporal.get("training_end") != FROZEN_TRAIN_END:
        raise Phase1ClosureError("Phase-1 TRAIN end changed")
    if temporal.get("validation_observations_loaded_for_fitting") is not False:
        raise Phase1ClosureError("VALIDATION observations entered Phase-1 fitting")
    if temporal.get("test_observations_loaded") is not False:
        raise Phase1ClosureError("TEST observations entered Phase-1 work")
    if temporal.get("rl_partition_v1_applies") is not False:
        raise Phase1ClosureError("Clustering and RL temporal protocols were conflated")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 5:
        raise Phase1ClosureError("Phase-1 experiment inventory is incomplete")
    versions = {str(item.get("version")) for item in experiments if isinstance(item, Mapping)}
    expected_versions = {
        CLUSTERING_METHODOLOGY_VERSION,
        PROTOCOL_SELECTION_VERSION,
        MARKET_MODE_AUDIT_VERSION,
        MULTIVIEW_AUDIT_VERSION,
        SOFT_REPRESENTATION_VERSION,
    }
    if versions != expected_versions:
        raise Phase1ClosureError("Phase-1 experiment versions changed")
    if PROTOCOL_SELECTION_DECISION != "blocked_weak_cluster_structure":
        raise Phase1ClosureError("Hard protocol decision changed in source")
    if MARKET_MODE_DECISION != "BLOCKED_WEAK_CLUSTER_STRUCTURE":
        raise Phase1ClosureError("Market-mode decision changed in source")
    if MULTIVIEW_DECISION != "BLOCKED_WEAK_CLUSTER_STRUCTURE":
        raise Phase1ClosureError("Multiview decision changed in source")
    if BLOCKED_DECISION != "BLOCKED_SOFT_REPRESENTATION":
        raise Phase1ClosureError("Soft representation decision changed in source")

    safety = payload.get("safety")
    if not isinstance(safety, Mapping):
        raise Phase1ClosureError("Phase-1 safety declaration is missing")
    required_false = (
        "validation_used_for_fitting",
        "test_observations_loaded",
        "model_training_performed",
        "lead_agent_work_performed",
        "phase3_recurrent_agents_modified",
        "final_cluster_assignments_written",
        "soft_representation_frozen",
    )
    if any(safety.get(field) is not False for field in required_false):
        raise Phase1ClosureError("Phase-1 safety declaration is not fail-closed")
    if safety.get("test_status") != "SEALED":
        raise Phase1ClosureError("TEST is not declared sealed")

    recorded_hash = payload.get("decision_evidence_hash")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise Phase1ClosureError("Phase-1 decision evidence hash is malformed")
    if recorded_hash != decision_evidence_hash(payload):
        raise Phase1ClosureError("Phase-1 decision evidence hash verification failed")


def load_phase1_closure(
    path: str | Path = PHASE1_CLOSURE_ARTIFACT,
) -> dict[str, Any]:
    """Load and validate the tracked Phase-1 closure artifact."""

    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase1ClosureError(
            f"Phase-1 closure artifact is unavailable: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase1ClosureError("Phase-1 closure artifact must be an object")
    validate_phase1_closure(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Phase-1 clustering closure artifact."
    )
    parser.add_argument("--artifact", default=str(PHASE1_CLOSURE_ARTIFACT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = load_phase1_closure(args.artifact)
    except Phase1ClosureError as exc:
        print(f"BLOCKED_PHASE1_CLOSURE: {exc}")
        return 2
    print(
        f"{payload['final_architecture_decision']} "
        f"({payload['decision_evidence_hash']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = (
    "FINAL_PHASE1_DECISION",
    "HARD_CLUSTERING_DECISION",
    "PHASE1_CLOSURE_ARTIFACT",
    "PHASE1_CLOSURE_VERSION",
    "Phase1ClosureError",
    "SOFT_RELATIONSHIP_DECISION",
    "decision_evidence_hash",
    "load_phase1_closure",
    "main",
    "validate_phase1_closure",
)
