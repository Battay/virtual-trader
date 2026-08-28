"""Offline contracts for selective recurrent training and global coverage."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from feature_engineering.storage import atomic_write_json
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.training.job_state import (
    COMPLETED,
    TRAINING,
    VALIDATING,
    transition_job,
)
from reinforcement_learning.training.production_control import (
    PRODUCTION_RUN_KIND,
    classify_run,
    launch_production_controller,
    production_plan,
)
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    INSUFFICIENT_DATA,
    RecurrentUniverseDiscovery,
)
from reinforcement_learning.training.selective_training import (
    SELECTED_RUN_KIND,
    TRAINED,
    UNTRAINED,
    SelectiveTrainingError,
    build_global_model_coverage,
    completed_job_is_trained,
    filter_symbol_coverage,
    load_selected_run_metadata,
    prepare_selected_run,
    selected_membership_hash,
    validate_selected_run,
)


NOW = "2026-08-29T00:00:00+00:00"


def _discovery() -> RecurrentUniverseDiscovery:
    plan = production_plan()
    records = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "sector": sector,
                "security_type": "ordinary_equity",
                "category": category,
                "reason": reason,
                "compatibility_error": "",
                "recurrent_contract_version": "rl_recurrent_partition_v1",
                "environment_version": "single_symbol_env_v1",
                "feature_version": "ai_features_v1",
                "source_data_hash": source_hash,
                "train_rows": rows,
                "train_start": "2018-01-01",
                "train_end": "2025-12-31",
                "validation_available": category == ELIGIBLE_TRAINABLE,
            }
            for symbol, sector, category, reason, source_hash, rows in (
                ("AAA", "BANKS", ELIGIBLE_TRAINABLE, "compatible", "a" * 64, 500),
                ("BBB", "CEMENT", ELIGIBLE_TRAINABLE, "compatible", "b" * 64, 500),
                ("CCC", "TEXTILE", INSUFFICIENT_DATA, "too_short", "", 50),
            )
        ]
    )
    return RecurrentUniverseDiscovery(
        records=records,
        universe_version="current_common_equity_universe_v1",
        universe_hash=plan.universe_hash,
        identity_count=3,
        category_counts={ELIGIBLE_TRAINABLE: 2, INSUFFICIENT_DATA: 1},
        source_inventory_hash="d" * 64,
        identity_policy=plan.identity_policy,
        identity_snapshot=plan.identity_snapshot,
        execution_training_policy=plan.execution_training_policy,
    )


def _coverage(*, aaa_trained: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "company_name": "AAA Limited",
                "sector": "BANKS",
                "coverage_status": TRAINED if aaa_trained else UNTRAINED,
                "trained": aaa_trained,
            },
            {
                "symbol": "BBB",
                "company_name": "BBB Limited",
                "sector": "CEMENT",
                "coverage_status": UNTRAINED,
                "trained": False,
            },
        ]
    )


def _complete_job(store, symbol: str) -> None:
    job = transition_job(store.read_job(symbol), TRAINING, timestamp=NOW)
    job = replace(job, completed_timesteps=100_352, effective_device="cpu")
    job = transition_job(job, VALIDATING, timestamp=NOW)
    model_path = store.run_directory / "models" / symbol / "attempt_000" / "model.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"verified recurrent model fixture")
    validation_path = (
        store.run_directory / "validation" / symbol / "attempt_000.json"
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {
            "symbol": symbol,
            "evaluation_partition": "validation",
            "environment_version": job.environment_version,
            "feature_version": job.feature_version,
            "recurrent_contract_version": job.data_contract_version,
            "model_parameters_unchanged": True,
        },
        validation_path,
    )
    job = transition_job(
        job,
        COMPLETED,
        timestamp=NOW,
        completed_timesteps=100_352,
        completed_at=NOW,
        model_path=str(model_path.relative_to(store.run_directory)),
        model_sha256=sha256_file(model_path),
        validation_status="completed",
        validation_metrics_reference=str(
            validation_path.relative_to(store.run_directory)
        ),
    )
    store.write_job(job)


def test_selected_run_skips_trained_by_default_and_is_immutable(tmp_path: Path) -> None:
    store, metadata, created = prepare_selected_run(
        ["BBB", "AAA"],
        runs_root=tmp_path,
        coverage=_coverage(aaa_trained=True),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )

    assert created
    assert metadata.requested_symbols == ("AAA", "BBB")
    assert metadata.selected_symbols == ("BBB",)
    assert metadata.skipped_trained_symbols == ("AAA",)
    assert tuple(job.symbol for job in store.list_jobs()) == ("BBB",)
    assert classify_run(store.read_manifest(), store.run_directory) == SELECTED_RUN_KIND
    assert validate_selected_run(store) == metadata

    payload = metadata.to_dict()
    payload["selected_symbols"] = ["AAA"]
    atomic_write_json(payload, store.run_directory / "selected_run.json")
    with pytest.raises(SelectiveTrainingError):
        validate_selected_run(store)


def test_existing_default_selection_is_reused_without_duplicate_queue(
    tmp_path: Path,
) -> None:
    first, first_metadata, first_created = prepare_selected_run(
        ["AAA"],
        runs_root=tmp_path,
        coverage=_coverage(),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )
    second, second_metadata, second_created = prepare_selected_run(
        ["AAA"],
        runs_root=tmp_path,
        coverage=_coverage(),
        frozen_discovery=_discovery(),
        created_at="2026-08-29T00:01:00+00:00",
    )

    assert first_created and not second_created
    assert second.run_directory == first.run_directory
    assert second_metadata == first_metadata


def test_explicit_retraining_creates_new_isolated_attempt(tmp_path: Path) -> None:
    first, first_metadata, _ = prepare_selected_run(
        ["AAA"],
        retrain_trained=True,
        runs_root=tmp_path,
        coverage=_coverage(aaa_trained=True),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )
    second, second_metadata, _ = prepare_selected_run(
        ["AAA"],
        retrain_trained=True,
        runs_root=tmp_path,
        coverage=_coverage(aaa_trained=True),
        frozen_discovery=_discovery(),
        created_at="2026-08-29T00:01:00+00:00",
    )

    assert first_metadata.attempt_version == 0
    assert second_metadata.attempt_version == 1
    assert first.run_directory != second.run_directory
    assert first.read_job("AAA").model_path is None
    assert second.read_job("AAA").model_path is None


def test_selected_controller_launch_accepts_only_qualified_persisted_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, metadata, _ = prepare_selected_run(
        ["AAA"],
        runs_root=tmp_path,
        coverage=_coverage(),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_: object) -> object:
        calls.append(command)
        return SimpleNamespace(pid=54321)

    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._pid_alive",
        lambda pid: pid == 54321,
    )
    status = launch_production_controller(
        store, popen=fake_popen, python_executable="/project/.venv/bin/python"
    )

    assert status.state == "STARTING"
    assert status.alive
    assert calls and calls[0][-1] == str(store.run_directory)
    assert metadata.requested_device == "cpu"
    assert metadata.worker_count == 4
    assert metadata.cpu_threads_per_worker == 2
    assert not metadata.cuda_execution_authorized
    assert not metadata.test_partition_loaded


@pytest.mark.parametrize("symbols", [(), ("CCC",), ("MISSING",)])
def test_empty_or_ineligible_selection_is_rejected(
    tmp_path: Path, symbols: tuple[str, ...]
) -> None:
    with pytest.raises(SelectiveTrainingError):
        prepare_selected_run(
            symbols,
            runs_root=tmp_path,
            coverage=_coverage(),
            frozen_discovery=_discovery(),
        )


def test_verified_model_and_validation_are_required_for_trained(tmp_path: Path) -> None:
    store, _, _ = prepare_selected_run(
        ["AAA"],
        runs_root=tmp_path,
        coverage=_coverage(),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )
    _complete_job(store, "AAA")

    assert completed_job_is_trained(store, store.read_job("AAA")) == (
        True,
        "verified",
    )
    model_path = store.resolve_artifact(store.read_job("AAA").model_path or "")
    model_path.unlink()
    assert completed_job_is_trained(store, store.read_job("AAA")) == (
        False,
        "model_artifact_missing",
    )


def test_global_coverage_survives_refresh_and_spans_run_history(tmp_path: Path) -> None:
    store, _, _ = prepare_selected_run(
        ["AAA"],
        runs_root=tmp_path,
        coverage=_coverage(),
        frozen_discovery=_discovery(),
        created_at=NOW,
    )
    _complete_job(store, "AAA")

    first, first_summary = build_global_model_coverage(
        runs_root=tmp_path, frozen_discovery=_discovery()
    )
    second, second_summary = build_global_model_coverage(
        runs_root=tmp_path, frozen_discovery=_discovery()
    )

    assert first.equals(second)
    assert first_summary == second_summary
    assert first_summary.eligible == 2
    assert first_summary.trained == 1
    assert first_summary.untrained == 1
    assert first.set_index("symbol").loc["AAA", "coverage_status"] == TRAINED
    assert first.set_index("symbol").loc["BBB", "coverage_status"] == UNTRAINED


def test_coverage_filters_are_deterministic_and_untrained_is_semantic() -> None:
    coverage = _coverage(aaa_trained=True)
    coverage.loc[coverage["symbol"].eq("BBB"), "coverage_status"] = "FAILED"

    trained = filter_symbol_coverage(coverage, statuses=[TRAINED])
    untrained = filter_symbol_coverage(coverage, statuses=[UNTRAINED])
    searched = filter_symbol_coverage(
        coverage, sectors=["CEMENT"], search="bbb limited"
    )

    assert trained["symbol"].tolist() == ["AAA"]
    assert untrained["symbol"].tolist() == ["BBB"]
    assert searched["symbol"].tolist() == ["BBB"]
    assert selected_membership_hash(["BBB", "AAA"]) == selected_membership_hash(
        ["AAA", "BBB", "AAA"]
    )


def test_full_production_contract_is_unchanged_and_distinct() -> None:
    plan = production_plan()

    assert PRODUCTION_RUN_KIND == "FULL_PRODUCTION"
    assert plan.identity_count == 508
    assert plan.trainable_count == 435
    assert plan.universe_hash == (
        "571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5"
    )
    assert plan.trainable_symbol_hash == (
        "44efa67c6c1aa5ac27d559f85835493206617a63fa24c25648e2da0d9f38a4a2"
    )


def test_selective_source_has_no_test_partition_load() -> None:
    source = Path(
        "reinforcement_learning/training/selective_training.py"
    ).read_text(encoding="utf-8")

    forbidden = ("test_rl.csv", 'partition="test"', "load_test")
    assert not [value for value in forbidden if value in source]
    assert load_selected_run_metadata
