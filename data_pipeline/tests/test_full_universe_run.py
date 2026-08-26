"""Offline tests for full-universe recurrent run preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reinforcement_learning.training.full_universe_run import (
    FullUniverseTrainingSpec,
    build_full_universe_plan,
    dry_run_progress_summary,
    estimate_full_run_storage,
    evaluate_full_run_gates,
)
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    UNSUPPORTED,
    RecurrentUniverseDiscovery,
)


def _discovery() -> RecurrentUniverseDiscovery:
    symbols = tuple(f"S{index:03d}" for index in range(508))
    eligible = set(symbols[:435])
    records = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "sector": "TEST",
                "security_type": "ordinary_equity",
                "category": ELIGIBLE_TRAINABLE if symbol in eligible else UNSUPPORTED,
                "reason": (
                    "canonical_mature_recurrent_contract"
                    if symbol in eligible
                    else "not_active_recently_traded"
                ),
                "compatibility_error": "",
                "recurrent_contract_version": (
                    "rl_recurrent_partition_v1" if symbol in eligible else ""
                ),
                "environment_version": (
                    "single_symbol_env_v1" if symbol in eligible else ""
                ),
                "feature_version": "feature_v1" if symbol in eligible else "",
                "source_data_hash": "a" * 64 if symbol in eligible else "",
                "train_rows": 1_000 if symbol in eligible else 0,
                "train_start": "2017-01-01" if symbol in eligible else "",
                "train_end": "2023-01-01" if symbol in eligible else "",
                "validation_available": symbol in eligible,
            }
            for symbol in reversed(symbols)
        ]
    ).sort_values("symbol", kind="mergesort").reset_index(drop=True)
    return RecurrentUniverseDiscovery(
        records=records,
        universe_version="current_common_equity_universe_v1",
        universe_hash="b" * 64,
        identity_count=508,
        category_counts={ELIGIBLE_TRAINABLE: 435, UNSUPPORTED: 73},
        source_inventory_hash="c" * 64,
    )


def test_full_plan_accounts_for_508_with_435_trainable_and_73_explicit_exclusions() -> None:
    identity, plan = build_full_universe_plan(
        _discovery(), spec=FullUniverseTrainingSpec()
    )

    assert identity["identity_count"] == 508
    assert identity["trainable_count"] == 435
    assert identity["ineligible_count"] == 73
    assert plan["trainability"].value_counts().to_dict() == {
        "eligible": 435,
        "ineligible": 73,
    }
    assert plan.loc[plan["trainability"].eq("ineligible"), "trainability_reason"].str.len().gt(0).all()
    assert not plan["test_partition_loaded"].any()


def test_full_plan_is_deterministic_and_paths_are_isolated() -> None:
    discovery = _discovery()
    spec = FullUniverseTrainingSpec()
    first_identity, first = build_full_universe_plan(discovery, spec=spec)
    second_identity, second = build_full_universe_plan(discovery, spec=spec)

    pd.testing.assert_frame_equal(first, second)
    assert first_identity == second_identity
    assert first["symbol"].tolist() == sorted(first["symbol"].tolist())
    assert first["model_path"].is_unique
    assert first["model_path"].str.match(r"models/S\d{3}/attempt_000/model\.zip").all()
    assert first_identity["execution_authorized"] is False


def test_dry_run_does_not_create_artifacts_or_invent_eta(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    _, plan = build_full_universe_plan(_discovery(), spec=FullUniverseTrainingSpec())
    summary = dry_run_progress_summary(plan)

    assert tuple(tmp_path.iterdir()) == before
    assert summary["queued"] == 435
    assert summary["ineligible"] == 73
    assert summary["overall_progress_percent"] == 0.0
    assert summary["estimated_remaining_seconds"] is None


def test_storage_gate_is_conservative_and_blocks_insufficient_space(tmp_path: Path) -> None:
    safe = estimate_full_run_storage(
        trainable_jobs=435,
        identity_jobs=508,
        storage_root=tmp_path,
        available_bytes=30 * 1024**3,
    )
    blocked = estimate_full_run_storage(
        trainable_jobs=435,
        identity_jobs=508,
        storage_root=tmp_path,
        available_bytes=1024,
    )

    assert safe.safe
    assert safe.required_safety_bytes >= 2 * safe.conservative_total_bytes
    assert not blocked.safe
    assert blocked.safety_margin_bytes < 0


def test_safety_gates_keep_full_execution_blocked_until_cuda_and_budget_freeze(tmp_path: Path) -> None:
    identity, plan = build_full_universe_plan(
        _discovery(), spec=FullUniverseTrainingSpec()
    )
    storage = estimate_full_run_storage(
        trainable_jobs=435,
        identity_jobs=508,
        storage_root=tmp_path,
        available_bytes=30 * 1024**3,
    )
    result = evaluate_full_run_gates(
        identity=identity,
        plan=plan,
        storage=storage,
        cuda_benchmark_completed=False,
        training_budget_frozen=False,
    )

    assert result["ready_to_benchmark_cuda"] is True
    assert result["full_run_authorized"] is False
    assert result["gates"]["cuda_benchmarked_for_cuda_run"] is False
    assert result["gates"]["training_budget_frozen"] is False


def test_full_spec_rejects_test_or_unsafe_worker_device() -> None:
    with pytest.raises(ValueError, match="TEST"):
        FullUniverseTrainingSpec(test_partition_loaded=True)
    with pytest.raises(ValueError, match="exactly one worker"):
        FullUniverseTrainingSpec(requested_device="cpu", worker_count=2)
