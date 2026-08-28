"""Offline regression tests for the immutable research identity policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.src.identity_universe_policy import (
    FROZEN_RESEARCH_IDENTITY_MANIFEST_PATH,
    FROZEN_RESEARCH_UNIVERSE,
    IdentityUniversePolicyError,
    load_frozen_research_identity,
)
from reinforcement_learning.training.recurrent_orchestrator import (
    _identity_universe_hash,
)


FROZEN_HASH = "571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5"


def test_frozen_research_identity_is_immutable_and_hash_validated() -> None:
    identity = load_frozen_research_identity()

    assert len(identity) == 508
    assert identity["symbol"].tolist() == sorted(identity["symbol"].tolist())
    assert not identity["symbol"].duplicated().any()
    assert set(identity["snapshot_date"]) == {"2026-08-02"}
    assert not {"GCWLPRS", "TISL"}.intersection(identity["symbol"])
    assert identity.attrs["identity_role"] == FROZEN_RESEARCH_UNIVERSE
    assert _identity_universe_hash(identity) == FROZEN_HASH


def test_frozen_research_identity_rejects_membership_or_hash_change(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        FROZEN_RESEARCH_IDENTITY_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    payload["members"][0]["sector"] = "CHANGED SECTOR"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IdentityUniversePolicyError, match="hash verification"):
        load_frozen_research_identity(changed)
