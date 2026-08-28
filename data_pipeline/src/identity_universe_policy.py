"""Explicit frozen-research and current-operational identity policies.

The live PSX listing snapshot is allowed to evolve.  Reproducible research and
full recurrent-run preparation instead load an immutable, tracked snapshot and
verify its deterministic identity hash before returning any members.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .equity_universe import (
    CLASSIFICATION_POLICY_VERSION,
    EQUITY_UNIVERSE_VERSION,
    IDENTITY_POLICY,
    deterministic_universe_identity,
)


FROZEN_RESEARCH_IDENTITY_MANIFEST_VERSION = (
    "frozen_research_common_equity_identity_v1"
)
FROZEN_RESEARCH_UNIVERSE = "FROZEN_RESEARCH_UNIVERSE"
CURRENT_OPERATIONAL_IDENTITY = "CURRENT_OPERATIONAL_IDENTITY"
FROZEN_RESEARCH_IDENTITY_COUNT = 508
FROZEN_RESEARCH_IDENTITY_SNAPSHOT = "2026-08-02"
FROZEN_RESEARCH_UNIVERSE_HASH = (
    "571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5"
)
FROZEN_RESEARCH_TRAINING_POLICY = (
    "TRAINABLE_MEMBERS_OF_FROZEN_RESEARCH_UNIVERSE_V1"
)
CURRENT_OPERATIONAL_TRAINING_POLICY = (
    "TRAINABLE_MEMBERS_OF_CURRENT_OPERATIONAL_IDENTITY_V1"
)
FROZEN_RESEARCH_IDENTITY_MANIFEST_PATH = (
    PROJECT_ROOT
    / "docs"
    / "config"
    / "frozen_research_common_equity_identity_v1.json"
)


class IdentityUniversePolicyError(RuntimeError):
    """Raised when a frozen identity cannot be reproduced exactly."""


def load_frozen_research_identity(
    manifest_path: str | Path = FROZEN_RESEARCH_IDENTITY_MANIFEST_PATH,
) -> pd.DataFrame:
    """Load the immutable research identity and verify all provenance/hash data."""

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IdentityUniversePolicyError(
            f"Frozen research identity manifest is unavailable: {path}"
        ) from exc
    if payload.get("manifest_version") != FROZEN_RESEARCH_IDENTITY_MANIFEST_VERSION:
        raise IdentityUniversePolicyError("Frozen identity manifest version is incompatible")
    if payload.get("identity_role") != FROZEN_RESEARCH_UNIVERSE:
        raise IdentityUniversePolicyError("Frozen identity role is incompatible")
    if payload.get("universe_version") != EQUITY_UNIVERSE_VERSION:
        raise IdentityUniversePolicyError("Frozen universe version is incompatible")
    if payload.get("identity_policy") != IDENTITY_POLICY:
        raise IdentityUniversePolicyError("Frozen identity methodology changed")
    if payload.get("classification_policy_version") != CLASSIFICATION_POLICY_VERSION:
        raise IdentityUniversePolicyError("Frozen classification methodology changed")
    if payload.get("listing_snapshot_date") != FROZEN_RESEARCH_IDENTITY_SNAPSHOT:
        raise IdentityUniversePolicyError("Frozen identity snapshot changed")
    if payload.get("identity_count") != FROZEN_RESEARCH_IDENTITY_COUNT:
        raise IdentityUniversePolicyError("Frozen identity count changed")
    if payload.get("universe_hash") != FROZEN_RESEARCH_UNIVERSE_HASH:
        raise IdentityUniversePolicyError("Frozen identity hash changed")

    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise IdentityUniversePolicyError("Frozen identity members are missing")
    required = {
        "symbol",
        "company_name",
        "instrument_category",
        "classification_basis",
        "security_type",
        "sector",
        "authoritative_source",
    }
    if any(not isinstance(item, dict) or required.difference(item) for item in members):
        raise IdentityUniversePolicyError("Frozen identity member schema is incomplete")
    ordered = sorted(members, key=lambda item: str(item["symbol"]))
    symbols = [str(item["symbol"]) for item in ordered]
    if len(set(symbols)) != len(symbols) or symbols != sorted(symbols):
        raise IdentityUniversePolicyError("Frozen identity symbols are not unique")
    if payload.get("identity_count") != len(ordered):
        raise IdentityUniversePolicyError("Frozen identity count does not reconcile")

    identity_payload = {
        "universe_version": payload["universe_version"],
        "identity_policy": payload["identity_policy"],
        "classification_policy_version": payload["classification_policy_version"],
        "listing_snapshot_date": payload["listing_snapshot_date"],
        "members": [
            {
                "symbol": str(item["symbol"]),
                "instrument_category": str(item["instrument_category"]),
                "classification_basis": str(item["classification_basis"]),
                "security_type": str(item["security_type"]),
                "sector": str(item["sector"]),
                "authoritative_source": str(item["authoritative_source"]),
            }
            for item in ordered
        ],
    }
    calculated = deterministic_universe_identity(identity_payload)
    if calculated != payload.get("universe_hash"):
        raise IdentityUniversePolicyError("Frozen identity hash verification failed")

    frame = pd.DataFrame(
        [
            {
                "symbol": str(item["symbol"]),
                "company_name": str(item["company_name"]),
                "sector": str(item["sector"]),
                "security_type": str(item["security_type"]),
                "source": str(item["authoritative_source"]),
                "snapshot_date": str(payload["listing_snapshot_date"]),
            }
            for item in ordered
        ]
    )
    frame.attrs.update(
        identity_role=FROZEN_RESEARCH_UNIVERSE,
        execution_training_policy=FROZEN_RESEARCH_TRAINING_POLICY,
        universe_hash=calculated,
        manifest_path=str(path),
    )
    return frame


__all__ = [
    "CURRENT_OPERATIONAL_IDENTITY",
    "CURRENT_OPERATIONAL_TRAINING_POLICY",
    "FROZEN_RESEARCH_IDENTITY_MANIFEST_PATH",
    "FROZEN_RESEARCH_IDENTITY_MANIFEST_VERSION",
    "FROZEN_RESEARCH_IDENTITY_COUNT",
    "FROZEN_RESEARCH_IDENTITY_SNAPSHOT",
    "FROZEN_RESEARCH_TRAINING_POLICY",
    "FROZEN_RESEARCH_UNIVERSE",
    "FROZEN_RESEARCH_UNIVERSE_HASH",
    "IdentityUniversePolicyError",
    "load_frozen_research_identity",
]
