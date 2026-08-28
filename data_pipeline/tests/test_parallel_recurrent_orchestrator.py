"""Offline process-concurrency tests for recurrent symbol orchestration."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
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
)
from reinforcement_learning.training.callbacks import TrainingProgress
from reinforcement_learning.training.devices import TorchDeviceResolution
from reinforcement_learning.training.job_state import (
    COMPLETED,
    FAILED,
    INTERRUPTED,
    QUEUED,
    TRAINING,
    VALIDATING,
    canonical_hash,
)
from reinforcement_learning.training.parallel_worker import (
    PARALLEL_WORKER_PROTOCOL_VERSION,
)
from reinforcement_learning.training.recurrent_config import RecurrentPPOConfig
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    RecurrentOrchestratorError,
    RecurrentUniverseDiscovery,
    TrainingRunStore,
    build_training_run,
    execute_queued_jobs,
    explicitly_requeue_job,
    training_status_summary,
    training_status_table,
)


def _fake_process_worker(request: dict[str, object], connection: object) -> None:
    """Spawn-safe worker double that never imports or runs PPO."""

    symbol = str(request["symbol"])
    started = time.time()
    try:
        forbidden = {
            "job_path",
            "manifest_path",
            "registry_path",
            "test_path",
            "test_partition",
        }
        if forbidden.intersection(request):
            raise AssertionError("worker received parent-owned or TEST paths")
        if request.get("test_partition_loaded") is not False:
            raise AssertionError("worker request exposed TEST")
        workspace = Path(str(request["workspace"]))
        workspace.mkdir(parents=True, exist_ok=False)
        connection.send(
            {
                "type": "progress",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "actual_timesteps": 256,
                "elapsed_seconds": 0.01,
                "timestamp": "2026-08-28T00:00:01+00:00",
            }
        )
        if symbol == "FAIL":
            time.sleep(0.1)
            connection.send(
                {
                    "type": "result",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "status": "failed",
                    "error": "RuntimeError: isolated failure",
                    "actual_timesteps": 256,
                    "effective_device": "cpu",
                    "duration_seconds": time.time() - started,
                    "worker_pid": os.getpid(),
                    "worker_started_epoch": started,
                    "worker_finished_epoch": time.time(),
                    "test_partition_loaded": False,
                }
            )
            return
        if symbol == "MALFORMED":
            connection.send(
                {
                    "type": "result",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "status": "completed",
                    "actual_timesteps": 512,
                    "effective_device": "cpu",
                    "duration_seconds": time.time() - started,
                    "worker_pid": os.getpid(),
                    "test_partition_loaded": False,
                }
            )
            return
        if symbol == "WRONGDEVICE":
            effective_device = "mps"
        else:
            effective_device = "cpu"
        time.sleep(0.25)
        model = workspace / "model.zip"
        model.write_bytes(f"fake-model:{symbol}".encode())
        validation_file = None
        if bool(request["validation_enabled"]):
            connection.send(
                {
                    "type": "stage",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "stage": "validating",
                    "actual_timesteps": 512,
                    "elapsed_seconds": time.time() - started,
                }
            )
            validation = workspace / "validation.json"
            validation.write_text(
                json.dumps(
                    {
                        "evaluation_partition": "validation",
                        "test_evaluated": False,
                    }
                ),
                encoding="utf-8",
            )
            validation_file = validation.name
        finished = time.time()
        connection.send(
            {
                "type": "result",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "status": "completed",
                "error": None,
                "actual_timesteps": 512,
                "requested_timesteps": int(request["config"]["total_timesteps"]),
                "effective_device": effective_device,
                "duration_seconds": finished - started,
                "model_file": model.name,
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "validation_file": validation_file,
                "worker_pid": os.getpid(),
                "worker_started_epoch": started,
                "worker_finished_epoch": finished,
                "worker_peak_rss_bytes": 1,
                "cpu_thread_policy": {
                    "requested_threads_per_worker": request.get(
                        "cpu_threads_per_worker"
                    ),
                    "torch_intraop_threads": request.get(
                        "cpu_threads_per_worker"
                    ),
                    "environment": {
                        name: os.environ.get(name)
                        for name in (
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        )
                    },
                },
                "test_partition_loaded": False,
            }
        )
    finally:
        connection.close()


def _slow_process_worker(request: dict[str, object], connection: object) -> None:
    workspace = Path(str(request["workspace"]))
    workspace.mkdir(parents=True, exist_ok=False)
    time.sleep(10)
    connection.close()


def _metadata(symbol: str, contract_path: Path) -> SimpleNamespace:
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
        validation_available=True,
        train=SimpleNamespace(rows=1_000, start="2017-01-02", end="2023-08-03"),
    )


def _store(
    tmp_path: Path,
    symbols: tuple[str, ...],
    *,
    validation_enabled: bool = True,
) -> tuple[TrainingRunStore, RecurrentPPOConfig, object, Path]:
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    metadata: dict[str, SimpleNamespace] = {}
    rows = []
    for symbol in symbols:
        path = contracts / f"{symbol}.json"
        path.write_text(json.dumps({"symbol": symbol}), encoding="utf-8")
        metadata[symbol] = _metadata(symbol, path)
        rows.append(
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "sector": "COMMERCIAL BANKS",
                "security_type": "ordinary_equity",
                "category": ELIGIBLE_TRAINABLE,
                "reason": "canonical_mature_recurrent_contract",
                "compatibility_error": "",
                "recurrent_contract_version": RL_RECURRENT_PARTITION_SCHEMA_VERSION,
                "environment_version": ENVIRONMENT_VERSION,
                "feature_version": FEATURE_VERSION,
                "source_data_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "train_rows": 1_000,
                "train_start": "2017-01-02",
                "train_end": "2023-08-03",
                "validation_available": True,
            }
        )
    records = pd.DataFrame(rows).sort_values("symbol", kind="mergesort")
    discovery = RecurrentUniverseDiscovery(
        records=records,
        universe_version="current_common_equity_universe_v1",
        universe_hash="a" * 64,
        identity_count=len(symbols),
        category_counts={ELIGIBLE_TRAINABLE: len(symbols)},
        source_inventory_hash="b" * 64,
    )
    config = RecurrentPPOConfig(total_timesteps=512, seed=42, device="cpu")
    manifest, jobs = build_training_run(
        discovery,
        config=config,
        validation_enabled=validation_enabled,
        created_at="2026-08-28T00:00:00+00:00",
    )
    store = TrainingRunStore(tmp_path / "run")
    store.initialize(manifest, jobs)

    def loader(symbol: str, **_: object) -> SimpleNamespace:
        return metadata[symbol]

    registry = tmp_path / "model_registry.csv"
    registry.write_text("sentinel\n", encoding="utf-8")
    return store, config, loader, registry


def _run_parallel(
    store: TrainingRunStore,
    config: RecurrentPPOConfig,
    loader: object,
    registry: Path,
    *,
    workers: int,
    max_jobs: int,
    process_worker: object = _fake_process_worker,
    cancellation_requested: object | None = None,
):
    return execute_queued_jobs(
        store,
        config=config,
        max_jobs=max_jobs,
        workers=workers,
        cpu_threads_per_worker=1,
        process_worker=process_worker,
        force_process_workers=True,
        metadata_loader=loader,
        registry_path=registry,
        device_resolver=lambda _: TorchDeviceResolution("cpu", "cpu", False, False),
        cancellation_requested=cancellation_requested,
    )


def _maximum_overlap(logs: list[dict[str, object]]) -> int:
    events = []
    for payload in logs:
        events.append((float(payload["worker_started_epoch"]), 1))
        events.append((float(payload["worker_finished_epoch"]), -1))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


@pytest.mark.parametrize("workers", [2, 4])
def test_process_pool_respects_worker_bound_and_completes_deterministically(
    tmp_path: Path, workers: int
) -> None:
    symbols = tuple(f"S{value}" for value in range(1, 6))
    store, config, loader, registry = _store(tmp_path, symbols)

    outcomes = _run_parallel(
        store,
        config,
        loader,
        registry,
        workers=workers,
        max_jobs=len(symbols),
    )

    assert [job.symbol for job in outcomes] == sorted(symbols)
    assert all(job.status == COMPLETED for job in outcomes)
    logs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((store.run_directory / "logs").glob("*/*.json"))
        if path.parent.name != "invocations"
    ]
    assert 1 < _maximum_overlap(logs) <= workers
    assert {payload["effective_device"] for payload in logs} == {"cpu"}
    assert all(payload["test_partition_loaded"] is False for payload in logs)
    assert all(
        set(payload["cpu_thread_policy"]["environment"].values()) == {"1"}
        for payload in logs
    )
    assert store.read_active_workers() == {}
    assert registry.read_text(encoding="utf-8") == "sentinel\n"
    invocation_logs = sorted(
        (store.run_directory / "logs" / "invocations").glob("*.json")
    )
    assert len(invocation_logs) == 1
    invocation = json.loads(invocation_logs[0].read_text(encoding="utf-8"))
    assert invocation["workers"] == workers
    assert invocation["status"] == "completed"
    assert invocation["test_partition_loaded"] is False


def test_max_jobs_remains_job_limit_and_progress_is_parent_owned(tmp_path: Path) -> None:
    store, config, loader, registry = _store(
        tmp_path, ("AAA", "BBB", "CCC", "DDD")
    )

    outcomes = _run_parallel(
        store, config, loader, registry, workers=2, max_jobs=3
    )

    assert len(outcomes) == 3
    assert [job.status for job in store.list_jobs()].count(COMPLETED) == 3
    assert [job.status for job in store.list_jobs()].count(QUEUED) == 1
    assert training_status_summary(store) == {
        "total": 4,
        "eligible": 4,
        "queued": 1,
        "active": 0,
        "completed": 3,
        "failed": 0,
        "interrupted": 0,
    }
    assert training_status_table(store)["worker_process_id"].isna().all()
    for job in outcomes:
        assert [event["to"] for event in job.state_history[-3:]] == [
            TRAINING,
            VALIDATING,
            COMPLETED,
        ]


def test_parallel_failure_isolated_and_retry_remains_explicit(tmp_path: Path) -> None:
    store, config, loader, registry = _store(tmp_path, ("AAA", "FAIL", "ZZZ"))

    outcomes = _run_parallel(
        store, config, loader, registry, workers=2, max_jobs=3
    )

    assert {job.symbol: job.status for job in outcomes} == {
        "AAA": COMPLETED,
        "FAIL": FAILED,
        "ZZZ": COMPLETED,
    }
    assert "isolated failure" in str(store.read_job("FAIL").failure_error_message)
    retried = explicitly_requeue_job(store, "FAIL")
    assert retried.status == QUEUED
    assert retried.retry_count == 1
    assert retried.completed_timesteps == 0


@pytest.mark.parametrize("symbol", ["MALFORMED", "WRONGDEVICE"])
def test_malformed_or_non_cpu_worker_result_fails_closed(
    tmp_path: Path, symbol: str
) -> None:
    store, config, loader, registry = _store(tmp_path, (symbol,))

    outcomes = _run_parallel(
        store, config, loader, registry, workers=1, max_jobs=1
    )

    assert outcomes[0].status == FAILED
    assert store.read_job(symbol).status == FAILED
    assert store.read_active_workers() == {}
    assert not any((store.run_directory / "models").rglob("model.zip"))


def test_parallel_interrupt_joins_workers_and_marks_launched_job(tmp_path: Path) -> None:
    store, config, loader, registry = _store(tmp_path, ("AAA", "BBB", "CCC"))
    calls = 0

    def cancel_after_launch() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    outcomes = _run_parallel(
        store,
        config,
        loader,
        registry,
        workers=2,
        max_jobs=3,
        process_worker=_slow_process_worker,
        cancellation_requested=cancel_after_launch,
    )

    assert outcomes
    assert all(job.status == INTERRUPTED for job in outcomes)
    assert not any(job.status in {TRAINING, VALIDATING} for job in store.list_jobs())
    assert store.read_active_workers() == {}
    assert not [
        child
        for child in multiprocessing.active_children()
        if child.name.startswith("rppo-")
    ]


def test_run_lock_and_cpu_only_parallel_contract(tmp_path: Path) -> None:
    store, config, loader, registry = _store(tmp_path, ("AAA",))
    with store.execution_lock():
        with pytest.raises(RecurrentOrchestratorError, match="already owns"):
            _run_parallel(
                store, config, loader, registry, workers=1, max_jobs=1
            )

    non_cpu = RecurrentPPOConfig(total_timesteps=512, seed=42, device="mps")
    with pytest.raises(RecurrentOrchestratorError, match="explicit device=cpu"):
        _run_parallel(
            store, non_cpu, loader, registry, workers=1, max_jobs=1
        )


def test_default_worker_one_preserves_legacy_sequential_path(tmp_path: Path) -> None:
    store, config, loader, registry = _store(
        tmp_path, ("AAA", "BBB"), validation_enabled=False
    )
    active = maximum = 0

    class Model:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"sequential")

    def trainer(symbol: str, *, progress_callback, **_: object):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        progress_callback(
            TrainingProgress(
                symbol=symbol,
                phase="progress",
                current_timesteps=512,
                requested_timesteps=512,
                progress_percent=100.0,
                timestamp="2026-08-28T00:00:01+00:00",
            )
        )
        active -= 1
        return SimpleNamespace(
            status="completed",
            succeeded=True,
            model=Model(),
            device="cpu",
            actual_timesteps=512,
            message="complete",
            error=None,
        )

    outcomes = execute_queued_jobs(
        store,
        config=config,
        max_jobs=2,
        trainer=trainer,
        metadata_loader=loader,
        registry_path=registry,
        device_resolver=lambda _: TorchDeviceResolution("cpu", "cpu", False, False),
    )

    assert maximum == 1
    assert all(job.status == COMPLETED for job in outcomes)
    assert not (store.run_directory / "active_workers.json").exists()


def test_parallel_more_than_one_requires_explicit_thread_policy(tmp_path: Path) -> None:
    store, config, loader, registry = _store(tmp_path, ("AAA", "BBB"))
    with pytest.raises(RecurrentOrchestratorError, match="explicit"):
        execute_queued_jobs(
            store,
            config=config,
            max_jobs=2,
            workers=2,
            process_worker=_fake_process_worker,
            metadata_loader=loader,
            registry_path=registry,
            device_resolver=lambda _: TorchDeviceResolution(
                "cpu", "cpu", False, False
            ),
        )


def test_parallel_job_configuration_hash_is_unchanged_by_worker_count(
    tmp_path: Path,
) -> None:
    store, _, _, _ = _store(tmp_path, ("AAA",))
    manifest = store.read_manifest()
    assert manifest.worker_limit == 1
    assert manifest.run_fingerprint == canonical_hash(
        {
            "orchestrator_version": manifest.orchestrator_version,
            "universe_version": manifest.universe_version,
            "universe_hash": manifest.universe_hash,
            "source_inventory_hash": manifest.source_inventory_hash,
            "agent_version": manifest.agent_version,
            "hyperparameters_hash": manifest.hyperparameters_hash,
            "requested_timesteps": manifest.requested_timesteps,
            "seed": manifest.seed,
            "requested_device": manifest.requested_device,
            "validation_enabled": manifest.validation_enabled,
            "worker_limit": 1,
        }
    )
