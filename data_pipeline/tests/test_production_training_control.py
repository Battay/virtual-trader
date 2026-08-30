"""Offline contracts for the recurrent production control plane."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pandas as pd
import pytest

from reinforcement_learning.training.job_state import (
    COMPLETED,
    FAILED,
    INELIGIBLE,
    INTERRUPTED,
    QUEUED,
    TRAINING,
    TRAINING_JOB_SCHEMA_VERSION,
    TRAINING_ORCHESTRATOR_VERSION,
    TRAINING_RUN_SCHEMA_VERSION,
    TrainingJobRecord,
    TrainingRunManifest,
    canonical_hash,
    transition_job,
)
from reinforcement_learning.training.production_control import (
    BENCHMARK_RUN_KIND,
    CONTROLLER_STATE_FILENAME,
    PRODUCTION_CONTROL_VERSION,
    PRODUCTION_RUN_KIND,
    ControllerStatus,
    ProductionControlError,
    RunCatalogEntry,
    aggregate_training_progress,
    build_job_table,
    calculate_training_progress,
    classify_run,
    controller_status,
    default_run_selection,
    launch_production_controller,
    list_run_catalog,
    load_run_snapshot,
    load_validation_metrics,
    job_training_progress,
    latest_job_diagnostics,
    production_plan,
    recent_orchestration_events,
    recover_dead_controller,
    registry_view,
    request_interrupt,
    request_stop_after_current,
    requeue_jobs,
    summarize_run,
)
from reinforcement_learning.training.recurrent_orchestrator import TrainingRunStore


NOW = "2026-08-28T00:00:00+00:00"


def _job(symbol: str = "AAA", *, status: str = QUEUED) -> TrainingJobRecord:
    eligible = status != INELIGIBLE
    return TrainingJobRecord(
        schema_version=TRAINING_JOB_SCHEMA_VERSION,
        job_id=f"job-{symbol}",
        run_id="run-small",
        symbol=symbol,
        trainability="eligible" if eligible else "ineligible",
        trainability_reason=("canonical" if eligible else "insufficient_history"),
        agent_version="recurrent_ppo_single_symbol_v1",
        environment_version="single_symbol_env_v1",
        data_contract_version="rl_recurrent_partition_v1",
        feature_version="ai_features_v1",
        universe_version="current_common_equity_universe_v1",
        universe_hash="a" * 64,
        source_data_hash="b" * 64,
        requested_timesteps=100_000,
        completed_timesteps=100_000 if status == COMPLETED else 0,
        seed=42,
        hyperparameters_hash="c" * 64,
        requested_device="cpu",
        effective_device="cpu" if status in {TRAINING, COMPLETED} else None,
        device_name=None,
        status=status,
        created_at=NOW,
        started_at=NOW if status in {TRAINING, COMPLETED, FAILED, INTERRUPTED} else None,
        updated_at=NOW,
        completed_at=NOW if status in {COMPLETED, FAILED, INTERRUPTED, INELIGIBLE} else None,
        wall_clock_duration_seconds=120.0 if status == COMPLETED else 0.0,
        checkpoint_path=None,
        model_path=(f"models/{symbol}/model.zip" if status == COMPLETED else None),
        model_sha256=("d" * 64 if status == COMPLETED else None),
        validation_status="completed" if status == COMPLETED else "not_requested",
        validation_metrics_reference=None,
        failure_error_message="RuntimeError: isolated" if status == FAILED else None,
        retry_count=0,
    )


def _controller(*, state: str = "NOT_STARTED", alive: bool = False) -> ControllerStatus:
    return ControllerStatus(
        state=state,
        pid=123 if alive else None,
        alive=alive,
        started_at=NOW if alive else None,
        updated_at=NOW,
        completed_at=None,
        message="",
        log_path=None,
    )


def _store(tmp_path: Path, status: str = QUEUED) -> TrainingRunStore:
    job = _job(status=status)
    manifest = TrainingRunManifest(
        schema_version=TRAINING_RUN_SCHEMA_VERSION,
        orchestrator_version=TRAINING_ORCHESTRATOR_VERSION,
        run_id=job.run_id,
        run_fingerprint="e" * 64,
        universe_version=job.universe_version,
        universe_hash=job.universe_hash,
        identity_count=1,
        eligible_count=1,
        ineligible_count=0,
        agent_version=job.agent_version,
        requested_timesteps=job.requested_timesteps,
        seed=job.seed,
        requested_device=job.requested_device,
        hyperparameters_hash=job.hyperparameters_hash,
        source_inventory_hash="f" * 64,
        validation_enabled=True,
        worker_limit=1,
        resume_capability="restart_from_zero_only",
        created_at=NOW,
    )
    store = TrainingRunStore(tmp_path / "run")
    store.initialize(manifest, (job,))
    return store


def test_production_plan_is_exact_and_not_ui_mutable() -> None:
    plan = production_plan()

    assert plan.identity_count == 508
    assert plan.identity_snapshot == "2026-08-02"
    assert plan.frozen_universe_version == (
        "frozen_research_common_equity_identity_v1"
    )
    assert plan.universe_version == "current_common_equity_universe_v1"
    assert plan.trainable_count == 435
    assert plan.excluded_count == 73
    assert plan.requested_timesteps == 100_000
    assert plan.seed == 42
    assert plan.requested_device == "cpu"
    assert plan.worker_count == 4
    assert plan.cpu_threads_per_worker == 2
    assert plan.test_status == "SEALED"
    assert len(plan.universe_hash) == len(plan.trainable_symbol_hash) == 64


def test_progress_states_and_eta_require_real_completed_history() -> None:
    prepared = summarize_run((_job(),), _controller())
    running = summarize_run((_job(status=TRAINING),), _controller(state="RUNNING", alive=True))
    failed = summarize_run((_job(status=FAILED),), _controller())
    interrupted = summarize_run((_job(status=INTERRUPTED),), _controller())
    completed = summarize_run((_job(status=COMPLETED),), _controller())

    assert prepared.system_status == "NOT_STARTED"
    assert prepared.estimated_remaining_seconds is None
    assert running.system_status == "RUNNING"
    assert failed.system_status == "FAILED"
    assert interrupted.system_status == "INTERRUPTED"
    assert completed.system_status == "COMPLETED"
    assert completed.progress_percent == 100.0

    jobs = (
        replace(_job("AAA", status=COMPLETED), started_at="2026-08-28T00:00:00+00:00", completed_at="2026-08-28T00:02:00+00:00"),
        replace(_job("BBB", status=COMPLETED), started_at="2026-08-28T00:00:00+00:00", completed_at="2026-08-28T00:04:00+00:00"),
        _job("CCC"),
    )
    evidenced = summarize_run(jobs, _controller(state="RUNNING", alive=True))
    assert evidenced.agents_per_hour == pytest.approx(30.0)
    assert evidenced.estimated_remaining_seconds == pytest.approx(120.0)


def test_stopped_after_current_is_not_not_started_and_has_no_eta() -> None:
    jobs = (
        replace(
            _job("DONE1", status=COMPLETED),
            started_at="2026-08-28T00:00:00+00:00",
            completed_at="2026-08-28T00:02:00+00:00",
        ),
        replace(
            _job("DONE2", status=COMPLETED),
            started_at="2026-08-28T00:00:00+00:00",
            completed_at="2026-08-28T00:04:00+00:00",
        ),
        _job("WAITING"),
        _job("NOPE", status=INELIGIBLE),
    )
    stopped = summarize_run(
        jobs, _controller(state="STOPPED_AFTER_CURRENT", alive=False)
    )

    assert stopped.system_status == "STOPPED_AFTER_CURRENT"
    assert stopped.completed == 2
    assert stopped.queued == 1
    assert stopped.ineligible == 1
    assert stopped.active == 0
    assert stopped.agents_per_hour == pytest.approx(30.0)
    assert stopped.estimated_remaining_seconds is None


def test_persisted_stopped_status_is_shared_by_controller_catalog_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    state_path = store.run_directory / CONTROLLER_STATE_FILENAME
    state_path.write_text(
        json.dumps(
            {
                "schema_version": PRODUCTION_CONTROL_VERSION,
                "run_id": "run-small",
                "state": "STOPPED_AFTER_CURRENT",
                "pid": 123,
                "started_at": NOW,
                "updated_at": NOW,
                "completed_at": NOW,
                "message": "deliberate stop",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._identity_lookup",
        lambda: pd.DataFrame(
            {"symbol": ["AAA"], "company_name": ["AAA Limited"], "sector": ["BANKS"]}
        ),
    )

    controller = controller_status(store.run_directory)
    catalog = list_run_catalog(runs_root=tmp_path)
    snapshot = load_run_snapshot(store.run_directory, recover_dead=False)

    assert controller.state == "STOPPED_AFTER_CURRENT"
    assert catalog[0].status == "STOPPED_AFTER_CURRENT"
    assert snapshot.progress.system_status == "STOPPED_AFTER_CURRENT"
    assert snapshot.controller.state == snapshot.progress.system_status


def test_default_run_selection_preserves_explicit_then_active_then_newest() -> None:
    production = RunCatalogEntry(
        run_id="production",
        run_directory=Path("production"),
        run_kind="FULL_PRODUCTION",
        created_at="2026-08-28T00:00:00+00:00",
        status="STOPPED_AFTER_CURRENT",
        identity_count=508,
        eligible_count=435,
        universe_hash="a" * 64,
    )
    selected = replace(
        production,
        run_id="selected",
        run_directory=Path("selected"),
        run_kind="SELECTED",
        created_at="2026-08-30T00:00:00+00:00",
        status="COMPLETED",
        identity_count=12,
        eligible_count=12,
    )
    active = replace(
        selected,
        run_id="active",
        run_directory=Path("active"),
        created_at="2026-08-29T00:00:00+00:00",
        status="RUNNING",
    )

    assert default_run_selection((production, selected, active), "production") == (
        "production"
    )
    assert default_run_selection((production, selected, active), None) == "active"
    stopping = replace(
        active,
        run_id="stopping",
        run_directory=Path("stopping"),
        created_at="2026-08-28T12:00:00+00:00",
        status="STOPPING_AFTER_CURRENT",
    )
    assert default_run_selection((production, selected, stopping), None) == "stopping"
    assert default_run_selection((production, selected), None) == "selected"
    assert default_run_selection((), None) is None


def test_latest_diagnostics_exposes_canonical_approximate_kl_without_retraining(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, status=COMPLETED)
    log_path = store.run_directory / "logs" / "AAA" / "attempt_000.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "training_diagnostics": {
                    "timesteps": 100_352,
                    "approximate_kl": 0.0166595,
                    "clip_fraction": 0.12,
                },
                "test_partition_loaded": False,
            }
        ),
        encoding="utf-8",
    )

    diagnostics = latest_job_diagnostics(store, "AAA")

    assert diagnostics["approximate_kl"] == pytest.approx(0.0166595)
    assert "approx_kl" not in diagnostics


@pytest.mark.parametrize(
    ("actual", "expected_percent"),
    [(0, 0.0), (63_488, 63.488), (100_000, 100.0), (100_352, 100.0)],
)
def test_symbol_progress_is_persisted_ratio_with_safe_clamping(
    actual: int, expected_percent: float
) -> None:
    progress = calculate_training_progress(actual, 100_000)

    assert progress.actual_timesteps == actual
    assert progress.requested_timesteps == 100_000
    assert progress.clamped_timesteps == min(actual, 100_000)
    assert progress.percent == pytest.approx(expected_percent)


@pytest.mark.parametrize(
    ("actual", "requested"),
    [(0, 0), (-1, 100_000), (None, 100_000), (1, None), (1.5, 100_000)],
)
def test_malformed_or_zero_budget_progress_fails_closed(
    actual: object, requested: object
) -> None:
    with pytest.raises(ProductionControlError):
        calculate_training_progress(actual, requested)


def test_status_progress_preserves_partial_failure_and_requires_completed_training() -> None:
    failed = replace(_job("FAIL", status=FAILED), completed_timesteps=47_104)
    interrupted = replace(
        _job("STOP", status=INTERRUPTED), completed_timesteps=32_768
    )
    validating = replace(
        _job("VALID", status=TRAINING),
        status="VALIDATING",
        completed_timesteps=100_352,
    )
    completed = replace(_job("DONE", status=COMPLETED), completed_timesteps=100_352)

    assert job_training_progress(failed).percent == pytest.approx(47.104)
    assert job_training_progress(interrupted).percent == pytest.approx(32.768)
    assert job_training_progress(validating).percent == 100.0
    assert job_training_progress(completed).percent == 100.0
    with pytest.raises(ProductionControlError, match="before its training budget"):
        job_training_progress(
            replace(validating, completed_timesteps=99_999)
        )
    with pytest.raises(ProductionControlError, match="ineligible"):
        job_training_progress(_job("NOPE", status=INELIGIBLE))


def test_aggregate_progress_excludes_ineligible_and_counts_partial_work() -> None:
    jobs = (
        replace(_job("AAA", status=TRAINING), completed_timesteps=50_000),
        _job("BBB", status=COMPLETED),
        _job("NOPE", status=INELIGIBLE),
    )

    progress = aggregate_training_progress(jobs)

    assert progress.eligible_jobs == 2
    assert progress.completed_timesteps == 150_000
    assert progress.requested_timesteps == 200_000
    assert progress.percent == pytest.approx(75.0)
    summarized = summarize_run(jobs, _controller(state="RUNNING", alive=True))
    assert summarized.completed == 1
    assert summarized.completed_training_timesteps == 150_000
    assert summarized.requested_training_timesteps == 200_000
    assert summarized.progress_percent == pytest.approx(75.0)


def test_job_table_refresh_reloads_persisted_progress_and_hides_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._identity_lookup",
        lambda: pd.DataFrame(
            {"symbol": ["AAA"], "company_name": ["AAA Limited"], "sector": ["BANKS"]}
        ),
    )
    first = build_job_table(store)
    training = transition_job(store.read_job("AAA"), TRAINING, timestamp=NOW)
    store.write_job(replace(training, completed_timesteps=50_000))
    refreshed = build_job_table(store)

    assert first.iloc[0]["progress_percent"] == 0.0
    assert refreshed.iloc[0]["progress_percent"] == 50.0

    class MixedStore:
        def list_jobs(self):
            return (_job("AAA"), _job("NOPE", status=INELIGIBLE))

        def read_active_workers(self):
            return {}

        def resolve_artifact(self, relative_path: str) -> Path:
            return tmp_path / relative_path

    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._identity_lookup",
        lambda: pd.DataFrame(
            {
                "symbol": ["AAA", "NOPE"],
                "company_name": ["AAA Limited", "Nope Limited"],
                "sector": ["BANKS", "UNKNOWN"],
            }
        ),
    )
    mixed = build_job_table(MixedStore())
    assert pd.isna(
        mixed.loc[mixed["symbol"].eq("NOPE"), "progress_percent"].iloc[0]
    )


def test_production_and_benchmark_runs_are_distinguished(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = _store(tmp_path).read_manifest()
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._validate_production_manifest",
        lambda _: None,
    )
    assert classify_run(manifest, tmp_path / "run") == PRODUCTION_RUN_KIND

    def reject(_: object) -> None:
        raise ProductionControlError("not production")

    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._validate_production_manifest",
        reject,
    )
    benchmark = replace(manifest, requested_timesteps=50_000)
    assert classify_run(benchmark, tmp_path / "benchmark") == BENCHMARK_RUN_KIND


def test_detached_launch_persists_pid_and_prevents_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_popen(command: list[str], **kwargs: object) -> object:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(pid=54321)

    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._validate_production_manifest",
        lambda _: None,
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._pid_alive",
        lambda pid: pid == 54321,
    )
    status = launch_production_controller(
        store, popen=fake_popen, python_executable="/project/.venv/bin/python"
    )

    assert status.state == "STARTING"
    assert status.pid == 54321 and status.alive
    assert calls[0]["start_new_session"] is True
    assert calls[0]["stdin"] is not None
    assert calls[0]["command"][-2:] == ["--run-directory", str(store.run_directory)]
    with pytest.raises(ProductionControlError, match="already running"):
        launch_production_controller(store, popen=fake_popen)


def test_stop_interrupt_and_retry_are_explicit_restart_from_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    state_path = store.run_directory / CONTROLLER_STATE_FILENAME
    state_path.write_text(
        json.dumps(
            {
                "schema_version": PRODUCTION_CONTROL_VERSION,
                "state": "RUNNING",
                "pid": 123,
                "started_at": NOW,
                "updated_at": NOW,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._pid_alive", lambda _: True
    )
    stopped = request_stop_after_current(store)
    assert stopped.state == "STOP_AFTER_CURRENT_REQUESTED"
    signals: list[tuple[int, int]] = []
    interrupted = request_interrupt(store, kill=lambda pid, sig: signals.append((pid, sig)))
    assert interrupted.state == "INTERRUPT_REQUESTED"
    assert signals == [(123, signal.SIGINT)]

    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._pid_alive", lambda _: False
    )
    failed = transition_job(store.read_job("AAA"), TRAINING, timestamp=NOW)
    failed = transition_job(failed, FAILED, timestamp=NOW, failure_error_message="boom")
    store.write_job(failed)
    restarted = requeue_jobs(store, statuses=frozenset({FAILED}))
    retried = store.read_job("AAA")
    assert restarted == ("AAA",)
    assert retried.status == QUEUED
    assert retried.completed_timesteps == 0
    assert retried.retry_count == 1
    assert retried.state_history[-1]["message"] == (
        "explicit_restart_from_zero_not_checkpoint_resume"
    )


def test_dead_controller_recovery_marks_active_job_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    store.write_job(transition_job(store.read_job("AAA"), TRAINING, timestamp=NOW))
    store.write_active_workers({"AAA": {"worker_pid": 456, "status": TRAINING}})
    (store.run_directory / CONTROLLER_STATE_FILENAME).write_text(
        json.dumps(
            {"state": "RUNNING", "pid": 123, "started_at": NOW, "updated_at": NOW}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control._pid_alive", lambda _: False
    )
    killed: list[tuple[int, int]] = []
    recovered = recover_dead_controller(store, kill=lambda pid, sig: killed.append((pid, sig)))

    assert recovered == ("AAA",)
    assert killed == [(456, signal.SIGTERM)]
    assert store.read_job("AAA").status == INTERRUPTED
    assert controller_status(store.run_directory).state == "RECOVERED_INTERRUPTED"


def test_validation_view_rejects_test_and_registry_contract_is_read_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    validation = store.run_directory / "validation" / "AAA" / "attempt_000.json"
    validation.parent.mkdir(parents=True, exist_ok=True)
    validation.write_text(json.dumps({"test_evaluated": True}), encoding="utf-8")
    store.write_job(
        replace(
            store.read_job("AAA"),
            validation_metrics_reference=str(validation.relative_to(store.run_directory)),
        )
    )
    with pytest.raises(ProductionControlError, match="TEST access"):
        load_validation_metrics(store, "AAA")

    missing_registry = tmp_path / "missing_registry.csv"
    view = registry_view(missing_registry)
    assert view.empty
    assert view.columns.tolist() == [
        "model_id", "model_family", "symbol", "run_id", "algorithm", "policy",
        "training_status", "validation_status", "created_at",
        "model_artifact_status", "metadata_integrity",
    ]


def test_source_contract_contains_no_test_partition_load() -> None:
    source = Path(
        "reinforcement_learning/training/production_control.py"
    ).read_text(encoding="utf-8")
    forbidden = ("test_rl.csv", 'load_rl_partition(symbol, "test"', 'partition="test"')
    assert not [value for value in forbidden if value in source]


def test_recent_event_view_is_bounded_and_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    training = transition_job(
        store.read_job("AAA"), TRAINING, timestamp="2026-08-28T00:01:00+00:00"
    )
    failed = transition_job(
        training,
        FAILED,
        timestamp="2026-08-28T00:02:00+00:00",
        failure_error_message="boom",
    )
    store.write_job(failed)

    events = recent_orchestration_events(store, limit=2)

    assert events["event"].tolist() == [FAILED, TRAINING]
    assert events["symbol"].tolist() == ["AAA", "AAA"]
