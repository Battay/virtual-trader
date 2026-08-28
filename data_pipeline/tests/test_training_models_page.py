"""Streamlit acceptance states for the recurrent Training & Models page."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from reinforcement_learning.training.production_control import (
    PRODUCTION_RUN_KIND,
    ControllerStatus,
    RunCatalogEntry,
    RunProgress,
    RunSnapshot,
)
from reinforcement_learning.training.selective_training import CoverageSummary


PAGE_PATH = Path(__file__).resolve().parents[2] / "app_pages" / "6_Training_and_Models.py"


class _Store:
    def __init__(self, root: Path) -> None:
        self.run_directory = root

    def read_job(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(validation_metrics_reference=None)


def _job_table(state: str) -> pd.DataFrame:
    active = state in {"TRAINING", "VALIDATING"}
    completed = state == "COMPLETED"
    failed = state == "FAILED"
    interrupted = state == "INTERRUPTED"
    actual_timesteps = (
        100_000
        if completed or state == "VALIDATING"
        else 63_488
        if state == "TRAINING"
        else 47_104
        if failed
        else 32_768
        if interrupted
        else 0
    )
    progress_percent = min(100.0, actual_timesteps / 100_000 * 100.0)
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "company_name": "AAA Limited",
                "sector": "COMMERCIAL BANKS",
                "eligibility": "eligible",
                "state": state,
                "exclusion_reason": "",
                "requested_timesteps": 100_000,
                "actual_timesteps": actual_timesteps,
                "progress_percent": progress_percent,
                "validation_status": "completed" if completed else "not_requested",
                "runtime_seconds": 240.0 if completed else 30.0 if active else 0.0,
                "started_at": "2026-08-28T00:00:00+00:00" if state not in {"QUEUED", "INELIGIBLE"} else None,
                "updated_at": "2026-08-28T00:01:00+00:00",
                "completed_at": "2026-08-28T00:04:00+00:00" if completed or failed or interrupted else None,
                "attempts": 1,
                "last_error": "RuntimeError: isolated failure" if failed else None,
                "error_type": "RuntimeError" if failed else "",
                "model_path": "models/AAA/model.zip" if completed else None,
                "model_artifact_status": "available" if completed else "not_created",
                "effective_device": "cpu" if active or completed else None,
                "worker_pid": 321 if active else None,
                "worker_slot": 1 if active else None,
                "cpu_threads": 2 if active else None,
            }
        ]
    )


def _snapshot(tmp_path: Path, system_status: str, job_state: str) -> RunSnapshot:
    jobs = _job_table(job_state)
    completed = int(job_state == "COMPLETED")
    failed = int(job_state == "FAILED")
    interrupted = int(job_state == "INTERRUPTED")
    active = int(job_state in {"TRAINING", "VALIDATING"})
    queued = int(job_state == "QUEUED")
    controller = ControllerStatus(
        state="RUNNING" if active else system_status,
        pid=123 if active else None,
        alive=bool(active),
        started_at="2026-08-28T00:00:00+00:00" if active else None,
        updated_at="2026-08-28T00:01:00+00:00",
        completed_at=None,
        message="fixture",
        log_path="logs/production_controller.log",
    )
    manifest = SimpleNamespace(
        run_id="run-production",
        identity_policy="current_common_equity_universe_v1",
        identity_snapshot="2026-08-02",
        universe_hash="a" * 64,
        trainable_symbol_hash="b" * 64,
        test_partition_loaded=False,
    )
    progress = RunProgress(
        system_status=system_status,
        eligible=1,
        completed=completed,
        active=active,
        training=int(job_state == "TRAINING"),
        validating=int(job_state == "VALIDATING"),
        queued=queued,
        failed=failed,
        interrupted=interrupted,
        stale=0,
        ineligible=0,
        completed_training_timesteps=int(jobs.iloc[0]["actual_timesteps"]),
        requested_training_timesteps=100_000,
        progress_percent=float(jobs.iloc[0]["progress_percent"]),
        elapsed_seconds=240.0 if completed else None,
        agents_per_hour=15.0 if completed else None,
        estimated_remaining_seconds=None,
    )
    return RunSnapshot(
        store=_Store(tmp_path),
        manifest=manifest,
        run_kind=PRODUCTION_RUN_KIND,
        controller=controller,
        progress=progress,
        jobs=jobs,
    )


def _patch_page_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: RunSnapshot | None,
) -> None:
    module = "reinforcement_learning.training.production_control"
    if snapshot is None:
        monkeypatch.setattr(f"{module}.list_run_catalog", lambda: ())
    else:
        entry = RunCatalogEntry(
            run_id="run-production",
            run_directory=tmp_path,
            run_kind=PRODUCTION_RUN_KIND,
            created_at="2026-08-28T00:00:00+00:00",
            status=snapshot.progress.system_status,
            identity_count=508,
            eligible_count=435,
            universe_hash="a" * 64,
        )
        monkeypatch.setattr(f"{module}.list_run_catalog", lambda: (entry,))
        monkeypatch.setattr(f"{module}.load_run_snapshot", lambda _: snapshot)
    monkeypatch.setattr(f"{module}.registry_view", lambda: pd.DataFrame())
    monkeypatch.setattr(f"{module}.symbol_contract_summary", lambda _: {})
    monkeypatch.setattr(f"{module}.latest_job_diagnostics", lambda *_: {})
    monkeypatch.setattr(f"{module}.load_validation_metrics", lambda *_: {})
    monkeypatch.setattr(
        f"{module}.recent_orchestration_events", lambda *_: pd.DataFrame()
    )
    coverage = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "company_name": "AAA Limited",
                "sector": "COMMERCIAL BANKS",
                "coverage_status": "TRAINED",
                "trained": True,
                "latest_progress_percent": 100.0,
                "actual_timesteps": 100_000,
                "requested_timesteps": 100_000,
                "model_status": "verified",
                "validation_status": "completed",
                "latest_run_id": "run-production",
                "latest_run_kind": PRODUCTION_RUN_KIND,
                "latest_attempt": 0,
                "latest_successful_run": "run-production",
                "model_path": "models/AAA/model.zip",
                "currently_training": False,
                "currently_validating": False,
            },
            {
                "symbol": "BBB",
                "company_name": "BBB Limited",
                "sector": "CEMENT",
                "coverage_status": "UNTRAINED",
                "trained": False,
                "latest_progress_percent": 0.0,
                "actual_timesteps": 0,
                "requested_timesteps": 0,
                "model_status": "not_created",
                "validation_status": "not_requested",
                "latest_run_id": "",
                "latest_run_kind": "",
                "latest_attempt": 0,
                "latest_successful_run": "",
                "model_path": None,
                "currently_training": False,
                "currently_validating": False,
            },
        ]
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.selective_training.build_global_model_coverage",
        lambda: (
            coverage,
            CoverageSummary(
                eligible=2,
                trained=1,
                untrained=1,
                training=0,
                validating=0,
                failed=0,
                interrupted=0,
            ),
        ),
    )


def test_page_loads_in_pre_run_state_without_creating_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_page_backend(monkeypatch, tmp_path, None)
    before = list(tmp_path.iterdir())

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    assert [item.value for item in app.title] == ["Training & Models"]
    assert "Prepare production run" in [item.label for item in app.button]
    assert not app.get("progress")
    captions = "\n".join(item.value for item in app.caption)
    assert "frozen 508-identity research snapshot" in captions
    assert "current operational identity universe is outside this run" in captions
    assert list(tmp_path.iterdir()) == before


def test_selective_ui_has_persistent_coverage_and_no_default_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_page_backend(monkeypatch, tmp_path, None)

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Eligible symbols"] == "2"
    assert metrics["Trained"] == "1"
    assert metrics["Untrained"] == "1"
    selector = next(
        item
        for item in app.multiselect
        if item.label == "Selected symbols in this filtered view"
    )
    assert selector.value == []
    train_button = next(
        item for item in app.button if item.label == "Train selected symbols"
    )
    assert train_button.disabled


@pytest.mark.parametrize(
    ("system_status", "job_state", "expected_control"),
    [
        ("NOT_STARTED", "QUEUED", "Start / continue run"),
        ("RUNNING", "TRAINING", "Stop after current jobs"),
        ("RUNNING", "VALIDATING", "Stop after current jobs"),
        ("PARTIAL_FAILURE", "FAILED", "Retry all failed"),
        ("PAUSED/INTERRUPTED", "INTERRUPTED", "Requeue interrupted"),
        ("COMPLETED", "COMPLETED", "Refresh status"),
    ],
)
def test_page_loads_prepared_running_failure_and_completed_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    system_status: str,
    job_state: str,
    expected_control: str,
) -> None:
    snapshot = _snapshot(tmp_path, system_status, job_state)
    _patch_page_backend(monkeypatch, tmp_path, snapshot)

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    assert expected_control in [item.label for item in app.button]
    rendered = "\n".join(
        str(item.value)
        for kind in ("caption", "info", "success", "warning", "error", "markdown")
        for item in app.get(kind)
    )
    assert system_status in rendered
    assert "TEST sealed" in rendered or "TEST remains sealed" in rendered
    if job_state == "TRAINING":
        progress_elements = app.get("progress")
        assert [item.value for item in progress_elements] == [63, 63]
        assert progress_elements[-1].text == (
            "63,488 / 100,000 timesteps · 63.5%"
        )
    if job_state == "VALIDATING":
        progress_elements = app.get("progress")
        assert [item.value for item in progress_elements] == [100, 100]
        assert "Training complete — validating" in rendered
    if job_state == "COMPLETED":
        assert "No active TRAINING or VALIDATING jobs." in rendered


def test_active_card_clamps_rollout_overage_but_preserves_actual_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot(tmp_path, "RUNNING", "TRAINING")
    jobs = snapshot.jobs.copy(deep=True)
    jobs.loc[:, "actual_timesteps"] = 100_352
    jobs.loc[:, "progress_percent"] = 100.0
    snapshot = replace(
        snapshot,
        jobs=jobs,
        progress=replace(
            snapshot.progress,
            completed_training_timesteps=100_000,
            progress_percent=100.0,
        ),
    )
    _patch_page_backend(monkeypatch, tmp_path, snapshot)

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    progress_elements = app.get("progress")
    assert [item.value for item in progress_elements] == [100, 100]
    assert progress_elements[-1].text == (
        "100,352 / 100,000 timesteps · 100.0%"
    )
