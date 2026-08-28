"""Offline tests for persistent multi-symbol recurrent job orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from feature_engineering.schemas import FEATURE_VERSION
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    RecurrentDataContractError,
)
from reinforcement_learning.training.callbacks import TrainingProgress
from reinforcement_learning.training.devices import TorchDeviceResolution
from reinforcement_learning.training.job_state import (
    COMPLETED,
    FAILED,
    INELIGIBLE,
    INTERRUPTED,
    QUEUED,
    STALE,
    TRAINING,
    VALIDATING,
    TrainingJobStateError,
    progress_snapshot,
    transition_job,
)
from reinforcement_learning.training.recurrent_config import RecurrentPPOConfig
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    INSUFFICIENT_DATA,
    UNSUPPORTED,
    RecurrentOrchestratorError,
    TrainingRunStore,
    build_training_run,
    completed_job_compatibility,
    create_training_run,
    discover_recurrent_training_universe,
    execute_queued_jobs,
    explicitly_requeue_job,
    mark_stale_jobs,
    recover_interrupted_jobs,
    training_status_table,
    validate_recurrent_device_resolution,
)


def _identity(symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "company_name": [f"{symbol} Limited" for symbol in symbols],
            "sector": ["COMMERCIAL BANKS"] * len(symbols),
            "security_type": ["ordinary_equity"] * len(symbols),
            "source": ["https://dps.psx.com.pk/listings-table/main/nc"] * len(symbols),
            "snapshot_date": ["2026-08-02"] * len(symbols),
        }
    )


def _metadata(symbol: str, contract_path: Path):
    return SimpleNamespace(
        symbol=symbol,
        contract_path=contract_path,
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        history=SimpleNamespace(independent_recurrent_ready=True),
        training_scope="symbol",
        constituent_symbols=(symbol,),
        train=SimpleNamespace(rows=1_000, start="2017-01-02", end="2023-08-03"),
    )


@pytest.fixture
def discovery_fixture(tmp_path: Path):
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    metadata = {}
    for symbol in ("AAA", "BBB"):
        path = contracts / f"{symbol}.json"
        path.write_text(f'{{"symbol":"{symbol}"}}', encoding="utf-8")
        metadata[symbol] = _metadata(symbol, path)

    readiness = tmp_path / "readiness.csv"
    pd.DataFrame(
        {
            "symbol": ["CCC", "DDD"],
            "history_class": ["INSUFFICIENT", "NOT_APPLICABLE"],
            "exclusion_reason": ["insufficient_history", "not_active_recently_traded"],
        }
    ).to_csv(readiness, index=False)

    def loader(symbol: str, **_: object):
        if symbol in metadata:
            return metadata[symbol]
        raise RecurrentDataContractError(f"recurrent contract is missing for {symbol}")

    discovery = discover_recurrent_training_universe(
        identity=_identity(),
        splits_dir=tmp_path,
        readiness_evidence_path=readiness,
        metadata_loader=loader,
    )
    return discovery, metadata, loader


def test_discovery_accounts_for_every_identity_with_explicit_categories(
    discovery_fixture,
) -> None:
    discovery, _, _ = discovery_fixture

    assert discovery.records["symbol"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert discovery.identity_count == 4
    assert discovery.category_counts == {
        ELIGIBLE_TRAINABLE: 2,
        INSUFFICIENT_DATA: 1,
        UNSUPPORTED: 1,
    }
    assert discovery.eligible_count == 2
    assert discovery.ineligible_count == 2


def test_run_and_job_identifiers_are_deterministic_and_every_identity_is_kept(
    discovery_fixture,
) -> None:
    discovery, _, _ = discovery_fixture
    config = RecurrentPPOConfig(total_timesteps=512, seed=42, device="cpu")
    first_manifest, first_jobs = build_training_run(
        discovery, config=config, created_at="2026-08-26T00:00:00+00:00"
    )
    second_manifest, second_jobs = build_training_run(
        discovery, config=config, created_at="2026-08-27T00:00:00+00:00"
    )

    assert first_manifest.run_id == second_manifest.run_id
    assert first_manifest.run_fingerprint == second_manifest.run_fingerprint
    assert [job.job_id for job in first_jobs] == [job.job_id for job in second_jobs]
    assert len(first_jobs) == first_manifest.identity_count == 4
    assert [job.status for job in first_jobs] == [QUEUED, QUEUED, INELIGIBLE, INELIGIBLE]
    assert all(job.requested_timesteps == 512 for job in first_jobs)
    assert first_manifest.test_partition_loaded is False
    assert first_manifest.identity_policy == discovery.identity_policy
    assert first_manifest.identity_snapshot == "2026-08-02"
    assert first_manifest.trainable_symbol_count == 2
    assert first_manifest.trainable_symbol_hash == discovery.trainable_symbol_hash


def test_resumed_run_rejects_incompatible_identity_hash(
    discovery_fixture, tmp_path: Path
) -> None:
    discovery, _, _ = discovery_fixture
    store = create_training_run(
        discovery,
        config=RecurrentPPOConfig(total_timesteps=512),
        runs_root=tmp_path / "runs",
        validation_enabled=False,
    )
    original = store.read_job("AAA")
    store.write_job(replace(original, universe_hash="f" * 64))

    with pytest.raises(RecurrentOrchestratorError, match="identity hash/version"):
        store.list_jobs()


def test_state_machine_accepts_only_explicit_legal_transitions(discovery_fixture) -> None:
    discovery, _, _ = discovery_fixture
    _, jobs = build_training_run(
        discovery,
        config=RecurrentPPOConfig(total_timesteps=512),
        created_at="2026-08-26T00:00:00+00:00",
    )
    queued = jobs[0]
    training = transition_job(queued, TRAINING, timestamp="2026-08-26T00:01:00+00:00")
    validating = transition_job(training, VALIDATING, timestamp="2026-08-26T00:02:00+00:00")
    completed = transition_job(
        validating,
        COMPLETED,
        timestamp="2026-08-26T00:03:00+00:00",
        model_path="models/AAA/model.zip",
        model_sha256="a" * 64,
    )

    assert completed.status == COMPLETED
    assert [event["to"] for event in completed.state_history[-3:]] == [
        TRAINING,
        VALIDATING,
        COMPLETED,
    ]
    with pytest.raises(TrainingJobStateError, match="illegal"):
        transition_job(queued, COMPLETED)
    with pytest.raises(TrainingJobStateError, match="illegal"):
        transition_job(jobs[2], QUEUED)


def test_progress_eta_requires_observed_nonzero_progress(discovery_fixture) -> None:
    discovery, _, _ = discovery_fixture
    _, jobs = build_training_run(
        discovery,
        config=RecurrentPPOConfig(total_timesteps=1_000),
        created_at="2026-08-26T00:00:00+00:00",
    )
    training = transition_job(jobs[0], TRAINING)
    no_progress = progress_snapshot(training, observed_elapsed_seconds=5)
    halfway = progress_snapshot(
        replace(training, completed_timesteps=500), observed_elapsed_seconds=10
    )

    assert no_progress["estimated_remaining_seconds"] is None
    assert halfway["progress_percent"] == 50.0
    assert halfway["estimated_remaining_seconds"] == pytest.approx(10.0)


class _FakeModel:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def save(self, path: Path) -> None:
        Path(path).write_bytes(f"model:{self.symbol}".encode())


def _fake_result(symbol: str, model: _FakeModel, timesteps: int = 512):
    return SimpleNamespace(
        symbol=symbol,
        status="completed",
        succeeded=True,
        model=model,
        device="cpu",
        actual_timesteps=timesteps,
        message="complete",
        error=None,
    )


def test_failure_isolated_model_paths_unique_completed_jobs_skip_and_registry_untouched(
    discovery_fixture, tmp_path: Path
) -> None:
    discovery, _, loader = discovery_fixture
    config = RecurrentPPOConfig(total_timesteps=512, seed=42, device="cpu")
    store = create_training_run(
        discovery,
        config=config,
        runs_root=tmp_path / "runs",
        validation_enabled=True,
        created_at="2026-08-26T00:00:00+00:00",
    )
    registry = tmp_path / "model_registry.csv"
    registry.write_text("sentinel\n", encoding="utf-8")

    def trainer(symbol: str, *, progress_callback, **_: object):
        progress_callback(
            TrainingProgress(
                symbol=symbol,
                phase="progress",
                current_timesteps=256,
                requested_timesteps=512,
                progress_percent=50.0,
                timestamp="2026-08-26T00:01:00+00:00",
            )
        )
        if symbol == "AAA":
            raise RuntimeError("simulated isolated failure")
        return _fake_result(symbol, _FakeModel(symbol))

    outcomes = execute_queued_jobs(
        store,
        config=config,
        max_jobs=2,
        trainer=trainer,
        evaluator=lambda *_args, **_kwargs: {
            "evaluation_partition": "validation",
            "test_evaluated": False,
        },
        device_resolver=lambda _: TorchDeviceResolution(
            "cpu", "cpu", False, False
        ),
        metadata_loader=loader,
        registry_path=registry,
    )

    assert [job.status for job in outcomes] == [FAILED, COMPLETED]
    assert "simulated isolated failure" in str(outcomes[0].failure_error_message)
    completed = outcomes[1]
    assert completed.model_available
    assert completed.model_path == "models/BBB/attempt_000/model.zip"
    assert store.resolve_artifact(completed.model_path).read_bytes() == b"model:BBB"
    assert completed.validation_status == "completed"
    assert registry.read_text(encoding="utf-8") == "sentinel\n"
    # Completed work is not re-requested; only the failed symbol remains terminal.
    assert execute_queued_jobs(
        store,
        config=config,
        max_jobs=2,
        trainer=lambda *_args, **_kwargs: pytest.fail("completed job retrained"),
        metadata_loader=loader,
        registry_path=registry,
    ) == ()
    table = training_status_table(store)
    assert len(table) == 4
    assert bool(table.loc[table["symbol"] == "BBB", "model_available"].iloc[0])


def test_interrupted_recovery_and_retry_are_explicit_restart_not_resume(
    discovery_fixture, tmp_path: Path
) -> None:
    discovery, _, _ = discovery_fixture
    config = RecurrentPPOConfig(total_timesteps=512)
    store = create_training_run(
        discovery,
        config=config,
        runs_root=tmp_path / "runs",
        validation_enabled=False,
    )
    store.update_job("AAA", lambda job: transition_job(job, TRAINING))

    assert recover_interrupted_jobs(store) == ("AAA",)
    interrupted = store.read_job("AAA")
    assert interrupted.status == INTERRUPTED
    assert "optimizer state was not checkpointed" in str(
        interrupted.failure_error_message
    )

    restarted = explicitly_requeue_job(store, "AAA")
    assert restarted.status == QUEUED
    assert restarted.completed_timesteps == 0
    assert restarted.retry_count == 1
    assert restarted.checkpoint_path == "checkpoints/AAA/attempt_001"
    assert "restart_from_zero" in str(restarted.state_history[-1]["message"])


def test_completed_compatibility_and_stale_detection(discovery_fixture, tmp_path: Path) -> None:
    discovery, metadata, loader = discovery_fixture
    config = RecurrentPPOConfig(total_timesteps=512)
    store = create_training_run(
        discovery,
        config=config,
        runs_root=tmp_path / "runs",
        validation_enabled=False,
    )
    model_path = store.resolve_artifact("models/AAA/attempt_000/model.zip")
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"stable-model")
    import hashlib

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    store.update_job("AAA", lambda job: transition_job(job, TRAINING))
    store.update_job(
        "AAA",
        lambda job: transition_job(
            job,
            COMPLETED,
            model_path="models/AAA/attempt_000/model.zip",
            model_sha256=model_hash,
        ),
    )
    compatible, reason = completed_job_compatibility(
        store, store.read_job("AAA"), metadata_loader=loader, splits_dir=tmp_path
    )
    assert compatible and reason == "completed_compatible_model"

    metadata["AAA"].contract_path.write_text("changed", encoding="utf-8")
    assert mark_stale_jobs(
        store, metadata_loader=loader, splits_dir=tmp_path
    ) == ("AAA",)
    assert store.read_job("AAA").status == STALE
    assert "source_contract_hash_changed" in str(
        store.read_job("AAA").failure_error_message
    )


def test_device_resolution_contract_denies_silent_fallback_and_auto_mps() -> None:
    validate_recurrent_device_resolution(
        TorchDeviceResolution("auto", "cuda", False, False, True, 1, "GPU")
    )
    validate_recurrent_device_resolution(
        TorchDeviceResolution("auto", "cpu", True, True, False, 0, None)
    )
    with pytest.raises(RecurrentOrchestratorError, match="never MPS"):
        validate_recurrent_device_resolution(
            TorchDeviceResolution("auto", "mps", True, True)
        )
    with pytest.raises(RecurrentOrchestratorError, match="silent fallback denied"):
        validate_recurrent_device_resolution(
            TorchDeviceResolution("mps", "cpu", True, True)
        )


def test_run_artifact_paths_are_isolated_and_portable(discovery_fixture) -> None:
    discovery, _, _ = discovery_fixture
    _, jobs = build_training_run(
        discovery,
        config=RecurrentPPOConfig(total_timesteps=512),
        created_at="2026-08-26T00:00:00+00:00",
    )
    paths = [job.checkpoint_path for job in jobs]

    assert len(paths) == len(set(paths))
    assert all(path and not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
    assert all("test" not in str(path).lower() for path in paths)
