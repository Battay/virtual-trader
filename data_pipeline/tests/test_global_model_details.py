"""Read-only global recurrent-model inventory and boundary audit tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from feature_engineering.storage import atomic_write_json
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.training.job_state import (
    COMPLETED,
    TRAINING,
    VALIDATING,
    transition_job,
)
from reinforcement_learning.training.model_details import (
    PartitionBoundary,
    RESEARCH_PARTITION_POLICY_VERSION,
    SINGLE_SYMBOL_RL_PARTITION_CONTRACT,
    SymbolPartitionManifest,
    audit_model_contract_compatibility,
    build_global_verified_model_inventory,
    read_symbol_partition_manifest,
    single_symbol_rl_partition_protocol,
)
from reinforcement_learning.training.production_control import production_plan
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    RecurrentUniverseDiscovery,
)
from reinforcement_learning.training.selective_training import prepare_selected_run


NOW = "2026-08-30T00:00:00+00:00"


def test_single_symbol_partition_protocol_is_distinct_and_test_sealed() -> None:
    protocol = single_symbol_rl_partition_protocol()

    assert protocol["name"] == "Single-symbol RL partition protocol"
    assert protocol["contract"] == SINGLE_SYMBOL_RL_PARTITION_CONTRACT
    assert protocol["contract"] == "rl_partition_v1"
    assert protocol["train"].startswith("first floor(70%)")
    assert protocol["validation"].startswith("next floor(15%)")
    assert "SEALED" in protocol["test"]
    assert protocol["rule"].startswith("symbol-specific chronological 70% TRAIN")
    assert "different common frozen temporal protocol" in protocol["clustering_note"]
    assert "do not define rl_partition_v1" in protocol["clustering_note"]


def test_fixed_window_reports_scope_cutoff_as_clustering_only() -> None:
    reports = Path(__file__).resolve().parents[2] / "docs" / "report_logs"
    reconciliation = (
        reports / "milestone_7c3h_frozen_vs_current_identity_reconciliation.md"
    ).read_text(encoding="utf-8")
    relationship = (
        reports / "milestone_7d_soft_relationship_representation_audit.md"
    ).read_text(encoding="utf-8")

    assert "The frozen TRAIN boundary ends" not in reconciliation
    assert "clustering/relationship research" in reconciliation
    assert "does not define the single-symbol `rl_partition_v1`" in reconciliation
    assert "Clustering/relationship TRAIN interval" in relationship
    assert "does not define the single-symbol `rl_partition_v1`" in relationship


def _write_partition_manifest(root: Path, market_root: Path, symbol: str = "AAA") -> None:
    directory = root / "symbols" / symbol
    recurrent = directory / "recurrent"
    recurrent.mkdir(parents=True)
    split_partitions = {
        "training": {"start": "2020-01-01", "end": "2020-01-07", "rows": 7},
        "validation": {"start": "2020-01-08", "end": "2020-01-08", "rows": 1},
        "testing": {"start": "2020-01-09", "end": "2020-01-10", "rows": 2},
    }
    atomic_write_json(
        {
            "scope": "symbol",
            "feature_version": "feature-v1",
            "proportions": {"training": 0.70, "validation": 0.15, "testing": 0.15},
            **split_partitions,
        },
        directory / "metadata.json",
    )
    atomic_write_json(
        {
            "artifact_schema_version": "rl_recurrent_partition_v1",
            "source_rl_contract_version": "rl_partition_v1",
            "symbol": symbol,
            "feature_version": "feature-v1",
            "environment_version": "single_symbol_env_v1",
            "normalization": {"fit_partition": "train"},
            "sequence": {
                "episode_strategy": "full_partition",
                "fixed_windows_enabled": False,
            },
            "partitions": {
                "train": {**split_partitions["training"], "sealed": False},
                "validation": {**split_partitions["validation"], "sealed": False},
                "test": {
                    **split_partitions["testing"],
                    "sealed": True,
                    "frame_access": "sealed_metadata_only",
                },
            },
            "test_sealing": {
                "sealed": True,
                "metadata_only": True,
                "evaluation_performed": False,
                "frame_loaded_during_build": False,
            },
        },
        recurrent / "recurrent_contract.json",
    )
    atomic_write_json(
        {
            "artifact_schema_version": "rl_recurrent_episode_boundaries_v1",
            "recurrent_contract_version": "rl_recurrent_partition_v1",
            "symbol": symbol,
            "episode_strategy": "full_partition",
            "partitions": {
                "train": [
                    {
                        "symbol": symbol,
                        "partition": "train",
                        "start_row": 0,
                        "end_row": 6,
                        "rows": 7,
                        "start": "2020-01-01",
                        "end": "2020-01-07",
                    }
                ],
                "validation": [
                    {
                        "symbol": symbol,
                        "partition": "validation",
                        "start_row": 0,
                        "end_row": 0,
                        "rows": 1,
                        "start": "2020-01-08",
                        "end": "2020-01-08",
                    }
                ],
            },
        },
        recurrent / "episode_boundaries.json",
    )
    atomic_write_json(
        {"training_rows": 7, "scaled_features": ["simple_return"]},
        directory / "rl_observation_scaler.json",
    )
    market_root.mkdir(parents=True)
    pd.DataFrame(
        {"market_date": ["2019-12-01", "2020-01-10", "2020-02-01"]}
    ).to_csv(market_root / f"{symbol}.csv", index=False)


def _discovery() -> RecurrentUniverseDiscovery:
    plan = production_plan()
    rows = []
    for symbol, source_hash in (("AAA", "a" * 64), ("BBB", "b" * 64)):
        rows.append(
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "sector": "BANKS",
                "security_type": "ordinary_equity",
                "category": ELIGIBLE_TRAINABLE,
                "reason": "compatible",
                "compatibility_error": "",
                "recurrent_contract_version": "rl_recurrent_partition_v1",
                "environment_version": "single_symbol_env_v1",
                "feature_version": "feature-v1",
                "source_data_hash": source_hash,
                "train_rows": 7,
                "train_start": "2020-01-01",
                "train_end": "2020-01-07",
                "validation_available": True,
            }
        )
    return RecurrentUniverseDiscovery(
        records=pd.DataFrame(rows),
        universe_version="current_common_equity_universe_v1",
        universe_hash=plan.universe_hash,
        identity_count=2,
        category_counts={ELIGIBLE_TRAINABLE: 2},
        source_inventory_hash="c" * 64,
        identity_policy=plan.identity_policy,
        identity_snapshot=plan.identity_snapshot,
        execution_training_policy=plan.execution_training_policy,
    )


def _complete(store, symbol: str) -> None:
    job = transition_job(store.read_job(symbol), TRAINING, timestamp=NOW)
    job = replace(job, completed_timesteps=100_352, effective_device="cpu")
    job = transition_job(job, VALIDATING, timestamp=NOW)
    model = store.run_directory / "models" / symbol / "attempt_000" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(f"model-{symbol}".encode())
    validation = store.run_directory / "validation" / symbol / "attempt_000.json"
    validation.parent.mkdir(parents=True)
    atomic_write_json(
        {
            "symbol": symbol,
            "evaluation_partition": "validation",
            "environment_version": job.environment_version,
            "feature_version": job.feature_version,
            "recurrent_contract_version": job.data_contract_version,
            "model_parameters_unchanged": True,
            "test_evaluated": False,
        },
        validation,
    )
    log = store.run_directory / "logs" / symbol / "attempt_000.json"
    log.parent.mkdir(parents=True)
    atomic_write_json(
        {
            "status": "completed",
            "test_partition_loaded": False,
            "training_start": "2020-01-01",
            "training_end": "2020-01-07",
            "training_rows": 7,
            "training_diagnostics": {"approximate_kl": 0.01},
        },
        log,
    )
    job = transition_job(
        job,
        COMPLETED,
        timestamp=NOW,
        completed_timesteps=100_352,
        completed_at=NOW,
        model_path=str(model.relative_to(store.run_directory)),
        model_sha256=sha256_file(model),
        validation_status="completed",
        validation_metrics_reference=str(validation.relative_to(store.run_directory)),
    )
    store.write_job(job)


def _partition(symbol: str, source_hash: str) -> SymbolPartitionManifest:
    return SymbolPartitionManifest(
        symbol=symbol,
        raw_available_start="2019-12-01",
        raw_available_end="2020-02-01",
        raw_available_rows=12,
        usable_feature_start="2020-01-01",
        usable_feature_end="2020-01-10",
        usable_feature_rows=10,
        train=PartitionBoundary("2020-01-01", "2020-01-07", 7),
        validation=PartitionBoundary("2020-01-08", "2020-01-08", 1),
        test=PartitionBoundary("2020-01-09", "2020-01-10", 2),
        partition_contract_version="rl_partition_v1",
        recurrent_contract_version="rl_recurrent_partition_v1",
        feature_version="feature-v1",
        environment_version="single_symbol_env_v1",
        scaler_fit_partition="train",
        split_policy_version=RESEARCH_PARTITION_POLICY_VERSION,
        test_sealed=True,
        test_metadata_only=True,
        contract_path=f"/{symbol}/recurrent_contract.json",
        source_contract_sha256=source_hash,
    )


def test_partition_manifest_uses_metadata_only_and_raw_date_column(
    monkeypatch, tmp_path: Path
) -> None:
    splits = tmp_path / "splits"
    market = tmp_path / "market"
    _write_partition_manifest(splits, market)
    calls: list[tuple[Path, object]] = []
    original = pd.read_csv

    def spy(path, *args, **kwargs):
        calls.append((Path(path), kwargs.get("usecols")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    result = read_symbol_partition_manifest(
        "AAA", splits_dir=splits, market_symbols_dir=market
    )

    assert result.train.rows == 7
    assert result.validation.rows == 1
    assert result.test.rows == 2
    assert result.raw_available_start == "2019-12-01"
    assert result.raw_available_end == "2020-02-01"
    assert result.test_sealed and result.test_metadata_only
    assert calls == [(market / "AAA.csv", ["market_date"])]
    assert not any("test.csv" in str(path) or "test_rl.csv" in str(path) for path, _ in calls)


def test_global_inventory_spans_runs_and_excludes_unverified_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    discovery = _discovery()
    coverage = pd.DataFrame(
        [
            {"symbol": symbol, "company_name": f"{symbol} Limited", "sector": "BANKS", "trained": True}
            for symbol in ("AAA", "BBB")
        ]
    )
    creation_coverage = coverage.assign(trained=False)
    store_a, _, _ = prepare_selected_run(
        ["AAA"], runs_root=tmp_path, coverage=creation_coverage, frozen_discovery=discovery, created_at=NOW
    )
    store_b, _, _ = prepare_selected_run(
        ["BBB"], runs_root=tmp_path, coverage=creation_coverage, frozen_discovery=discovery, created_at=NOW
    )
    _complete(store_a, "AAA")
    _complete(store_b, "BBB")
    run_types = {
        store_a.run_directory.resolve(): "FULL_PRODUCTION",
        store_b.run_directory.resolve(): "SELECTED",
    }
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control.classify_run",
        lambda _manifest, path: run_types[Path(path).resolve()],
    )
    hashes = {"AAA": "a" * 64, "BBB": "b" * 64}
    loader = lambda symbol, **_: _partition(symbol, hashes[symbol])

    inventory = build_global_verified_model_inventory(
        coverage=coverage, runs_root=tmp_path, partition_loader=loader
    )

    assert inventory["symbol"].tolist() == ["AAA", "BBB"]
    assert inventory.set_index("symbol")["run_type"].to_dict() == {
        "AAA": "FULL_PRODUCTION",
        "BBB": "SELECTED",
    }
    assert inventory["attempt"].tolist() == [1, 1]
    assert inventory["validation_after_training"].all()
    assert not inventory["test_partition_loaded"].any()

    store_b.resolve_artifact(store_b.read_job("BBB").model_path or "").unlink()
    reduced = build_global_verified_model_inventory(
        coverage=coverage, runs_root=tmp_path, partition_loader=loader
    )
    assert reduced["symbol"].tolist() == ["AAA"]


def test_contract_compatibility_detects_overlap_and_methodology_drift() -> None:
    frame = pd.DataFrame(
        [
            {
                **{
                    field: value
                    for field, value in (
                        ("partition_contract_version", "rl_partition_v1"),
                        ("recurrent_contract_version", "rl_recurrent_partition_v1"),
                        ("feature_version", "feature-v1"),
                        ("environment_version", "single_symbol_env_v1"),
                        ("algorithm", "RecurrentPPO"),
                        ("policy", "MlpLstmPolicy"),
                        ("trainer_version", "recurrent_ppo_single_symbol_v1"),
                        ("recurrent_config_version", "recurrent_ppo_single_symbol_v1"),
                        ("requested_timesteps", 100_000),
                        ("seed", 42),
                        ("hyperparameters_hash", "a" * 64),
                        ("split_policy_version", RESEARCH_PARTITION_POLICY_VERSION),
                        ("scaler_fit_partition", "train"),
                    )
                },
                "run_type": run_type,
                "train_end": "2023-01-01",
                "validation_start": "2023-01-02",
                "validation_end": "2024-01-01",
                "test_start": "2024-01-02",
                "test_partition_loaded": False,
            }
            for run_type in ("FULL_PRODUCTION", "SELECTED")
        ]
    )
    clean = audit_model_contract_compatibility(frame)
    assert clean["compatible"] is True
    assert clean["partition_overlap"] is False

    frame.loc[1, "feature_version"] = "feature-v2"
    frame.loc[1, "validation_start"] = "2023-01-01"
    drift = audit_model_contract_compatibility(frame)
    assert drift["compatible"] is False
    assert drift["differences"] == {"feature_version": ["feature-v1", "feature-v2"]}
    assert drift["partition_overlap"] is True
