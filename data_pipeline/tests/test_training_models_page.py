"""Streamlit acceptance states for the recurrent Training & Models page."""

from __future__ import annotations

from dataclasses import replace
import json
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
from reinforcement_learning.training.selective_training import (
    SELECTED_RUN_KIND,
    CoverageSummary,
)


PAGE_PATH = Path(__file__).resolve().parents[2] / "app_pages" / "6_Training_and_Models.py"


class _Store:
    def __init__(self, root: Path) -> None:
        self.run_directory = root

    def read_job(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(validation_metrics_reference=None)


def _job_table(state: str, *, symbols: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
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
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
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
                "model_path": f"models/{symbol}/model.zip" if completed else None,
                "model_artifact_status": "available" if completed else "not_created",
                "effective_device": "cpu" if active or completed else None,
                "worker_pid": 321 if active else None,
                "worker_slot": 1 if active else None,
                "cpu_threads": 2 if active else None,
            }
            for symbol in symbols
        ]
    )


def _snapshot(
    tmp_path: Path,
    system_status: str,
    job_state: str,
    *,
    run_kind: str = PRODUCTION_RUN_KIND,
    run_id: str = "run-production",
    symbols: tuple[str, ...] = ("AAA",),
    controller_state: str | None = None,
) -> RunSnapshot:
    jobs = _job_table(job_state, symbols=symbols)
    count = len(symbols)
    completed = count if job_state == "COMPLETED" else 0
    failed = count if job_state == "FAILED" else 0
    interrupted = count if job_state == "INTERRUPTED" else 0
    active = count if job_state in {"TRAINING", "VALIDATING"} else 0
    queued = count if job_state == "QUEUED" else 0
    controller = ControllerStatus(
        state=controller_state or ("RUNNING" if active else system_status),
        pid=123 if active else None,
        alive=bool(active),
        started_at="2026-08-28T00:00:00+00:00" if active else None,
        updated_at="2026-08-28T00:01:00+00:00",
        completed_at=None,
        message="fixture",
        log_path="logs/production_controller.log",
    )
    manifest = SimpleNamespace(
        run_id=run_id,
        identity_policy="current_common_equity_universe_v1",
        identity_snapshot="2026-08-02",
        universe_hash="a" * 64,
        trainable_symbol_hash="b" * 64,
        test_partition_loaded=False,
    )
    progress = RunProgress(
        system_status=system_status,
        eligible=count,
        completed=completed,
        active=active,
        training=count if job_state == "TRAINING" else 0,
        validating=count if job_state == "VALIDATING" else 0,
        queued=queued,
        failed=failed,
        interrupted=interrupted,
        stale=0,
        ineligible=0,
        completed_training_timesteps=int(jobs["actual_timesteps"].sum()),
        requested_training_timesteps=100_000 * count,
        progress_percent=float(jobs.iloc[0]["progress_percent"]),
        elapsed_seconds=240.0 if completed else None,
        agents_per_hour=15.0 if completed else None,
        estimated_remaining_seconds=None,
    )
    return RunSnapshot(
        store=_Store(tmp_path),
        manifest=manifest,
        run_kind=run_kind,
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
            run_id=snapshot.manifest.run_id,
            run_directory=tmp_path,
            run_kind=snapshot.run_kind,
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
                eligible=435,
                trained=16,
                untrained=419,
                training=0,
                validating=0,
                failed=0,
                interrupted=0,
            ),
        ),
    )
    verified_inventory = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "company_name": "AAA Limited",
                "sector": "COMMERCIAL BANKS",
                "run_type": PRODUCTION_RUN_KIND,
                "run_id": "run-production",
                "attempt": 1,
                "training_status": "COMPLETED",
                "validation_status": "completed",
                "artifact_verification": "verified",
                "algorithm": "RecurrentPPO",
                "policy": "MlpLstmPolicy",
                "trainer_version": "recurrent_ppo_single_symbol_v1",
                "partition_contract_version": "rl_partition_v1",
                "recurrent_contract_version": "rl_recurrent_partition_v1",
                "feature_version": "feature-v1",
                "environment_version": "single_symbol_env_v1",
                "split_policy_version": "split-v1",
                "scaler_fit_partition": "train",
                "source_contract_sha256": "c" * 64,
                "hyperparameters_hash": "h" * 64,
                "seed": 42,
                "model_sha256": "m" * 64,
                "requested_timesteps": 100_000,
                "actual_timesteps": 100_352,
                "effective_device": "cpu",
                "runtime_seconds": 240.0,
                "raw_available_start": "2016-07-26",
                "raw_available_end": "2026-08-28",
                "raw_available_rows": 2_450,
                "usable_feature_start": "2016-10-06",
                "usable_feature_end": "2026-08-05",
                "usable_feature_rows": 2_400,
                "train_start": "2016-10-06",
                "train_end": "2023-08-23",
                "train_rows": 1_680,
                "validation_start": "2023-08-24",
                "validation_end": "2025-02-12",
                "validation_rows": 360,
                "test_start": "2025-02-13",
                "test_end": "2026-08-05",
                "test_rows": 360,
                "validation_partition": "validation",
                "validation_metrics_reference": "validation/AAA/attempt_000.json",
                "model_path": "models/AAA/attempt_000/model.zip",
                "run_directory": str(tmp_path),
            }
        ]
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.model_details.build_global_verified_model_inventory",
        lambda **_: verified_inventory,
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
    assert metrics["Eligible symbols"] == "435"
    assert metrics["Trained"] == "16"
    assert metrics["Untrained"] == "419"
    editor = next(
        item
        for item in app.dataframe
        if item.key == "selective_symbol_checkbox_editor"
    )
    assert not editor.value["selected"].any()
    assert app.session_state["selective_training_symbols"] == []
    column_contract = json.loads(editor.proto.columns)
    assert column_contract["selected"]["type_config"]["type"] == "checkbox"
    assert not column_contract["selected"].get("disabled", False)
    read_only = set(editor.value.columns).difference({"selected"})
    assert all(column_contract[column]["disabled"] for column in read_only)
    train_button = next(
        item for item in app.button if item.label == "Train selected symbols"
    )
    assert train_button.disabled


def test_checkbox_edits_accumulate_across_filter_reruns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_page_backend(monkeypatch, tmp_path, None)
    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    app.text_input(key="selective_symbol_search").set_value("AAA").run()
    app.session_state["selective_symbol_checkbox_editor"] = {
        "edited_rows": {0: {"selected": True}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert app.session_state["selective_training_symbols"] == ["AAA"]

    app.text_input(key="selective_symbol_search").set_value("BBB").run()
    app.session_state["selective_symbol_checkbox_editor"] = {
        "edited_rows": {0: {"selected": True}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert app.session_state["selective_training_symbols"] == ["AAA", "BBB"]

    app.text_input(key="selective_symbol_search").set_value("").run().run()
    editor = next(
        item
        for item in app.dataframe
        if item.key == "selective_symbol_checkbox_editor"
    )
    assert editor.value.set_index("symbol")["selected"].to_dict() == {
        "AAA": True,
        "BBB": True,
    }


def test_select_visible_clear_visible_and_clear_all_buttons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_page_backend(monkeypatch, tmp_path, None)
    launched: list[object] = []
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control.launch_production_controller",
        lambda store: launched.append(store),
    )
    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    app.text_input(key="selective_symbol_search").set_value("AAA").run()
    next(button for button in app.button if button.label.startswith("Select visible")).click().run()
    assert app.session_state["selective_training_symbols"] == ["AAA"]

    app.text_input(key="selective_symbol_search").set_value("BBB").run()
    next(button for button in app.button if button.label.startswith("Select visible")).click().run()
    assert app.session_state["selective_training_symbols"] == ["AAA", "BBB"]

    next(button for button in app.button if button.label == "Clear visible").click().run()
    assert app.session_state["selective_training_symbols"] == ["AAA"]
    next(
        button for button in app.button if button.label == "Clear all selection"
    ).click().run()
    assert app.session_state["selective_training_symbols"] == []
    assert launched == []


@pytest.mark.parametrize(
    ("system_status", "job_state", "expected_control"),
    [
        ("NOT_STARTED", "QUEUED", "Start / continue run"),
        ("RUNNING", "TRAINING", "Stop after current jobs"),
        ("RUNNING", "VALIDATING", "Stop after current jobs"),
        ("FAILED", "FAILED", "Retry all failed"),
        ("INTERRUPTED", "INTERRUPTED", "Requeue interrupted"),
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


def test_latest_completed_selected_run_remains_selected_and_shows_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    symbols = tuple(f"S{index:02d}" for index in range(12))
    selected = _snapshot(
        tmp_path / "selected",
        "COMPLETED",
        "COMPLETED",
        run_kind=SELECTED_RUN_KIND,
        run_id="run-selected",
        symbols=symbols,
        controller_state="COMPLETED",
    )
    production = _snapshot(
        tmp_path / "production",
        "STOPPED_AFTER_CURRENT",
        "QUEUED",
        run_id="run-production",
        controller_state="STOPPED_AFTER_CURRENT",
    )
    _patch_page_backend(monkeypatch, tmp_path, selected)
    catalog = (
        RunCatalogEntry(
            run_id="run-selected",
            run_directory=selected.store.run_directory,
            run_kind=SELECTED_RUN_KIND,
            created_at="2026-08-30T18:05:30+00:00",
            status="COMPLETED",
            identity_count=12,
            eligible_count=12,
            universe_hash="a" * 64,
            selected_count=12,
            selected_symbol_hash="c" * 64,
        ),
        RunCatalogEntry(
            run_id="run-production",
            run_directory=production.store.run_directory,
            run_kind=PRODUCTION_RUN_KIND,
            created_at="2026-08-28T00:00:00+00:00",
            status="STOPPED_AFTER_CURRENT",
            identity_count=508,
            eligible_count=435,
            universe_hash="a" * 64,
        ),
    )
    backend = "reinforcement_learning.training.production_control"
    monkeypatch.setattr(f"{backend}.list_run_catalog", lambda: catalog)
    monkeypatch.setattr(
        f"{backend}.load_run_snapshot",
        lambda path: selected if Path(path).name == "selected" else production,
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.selective_training.load_selected_run_metadata",
        lambda _: SimpleNamespace(
            selected_symbols=symbols,
            selected_symbol_hash="c" * 64,
            attempt_version=0,
        ),
    )

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    assert app.session_state["training_control_run_id"] == "run-selected"
    assert app.selectbox(key="training_control_run_id").value == "run-selected"
    metrics = [(item.label, item.value) for item in app.metric]
    assert ("Run type", "SELECTED") in metrics
    assert ("Completed", "12 / 12") in metrics
    assert ("Failed", "0") in metrics
    assert ("Interrupted", "0") in metrics
    assert ("Validation completed", "12 / 12") in metrics
    assert ("Final status", "COMPLETED") in metrics
    rendered = "\n".join(
        str(item.value)
        for kind in ("caption", "success", "markdown")
        for item in app.get(kind)
    )
    assert "Membership hash" in rendered
    app.run()
    assert app.session_state["training_control_run_id"] == "run-selected"
    assert app.selectbox(key="training_control_run_id").value == "run-selected"


def test_stopped_production_status_is_consistent_and_has_no_active_eta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot(
        tmp_path,
        "STOPPED_AFTER_CURRENT",
        "QUEUED",
        controller_state="STOPPED_AFTER_CURRENT",
    )
    snapshot = replace(
        snapshot,
        progress=replace(
            snapshot.progress,
            eligible=435,
            completed=4,
            queued=431,
            ineligible=73,
            completed_training_timesteps=400_000,
            requested_training_timesteps=43_500_000,
            progress_percent=400_000 / 43_500_000 * 100.0,
            elapsed_seconds=313.59,
            agents_per_hour=45.92,
            estimated_remaining_seconds=None,
        ),
    )
    _patch_page_backend(monkeypatch, tmp_path, snapshot)

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    rendered = "\n".join(
        str(item.value)
        for kind in ("caption", "warning", "markdown")
        for item in app.get(kind)
    )
    assert "Execution status: STOPPED_AFTER_CURRENT" in rendered
    assert "Remaining work: 431 agents" in rendered
    assert "ETA unavailable while run is stopped" in rendered
    assert "Active estimated remaining time" not in rendered
    assert "NOT_STARTED" not in rendered
    history = next(
        item
        for item in app.dataframe
        if "run_type" in item.value.columns and "status" in item.value.columns
    )
    assert history.value.iloc[0]["status"] == "STOPPED_AFTER_CURRENT"
    assert app.selectbox(key="training_control_run_id").options[0].endswith(
        "STOPPED_AFTER_CURRENT"
    )


def test_empty_promotion_registry_still_reports_run_isolated_model_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot(tmp_path, "COMPLETED", "COMPLETED")
    _patch_page_backend(monkeypatch, tmp_path, snapshot)

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Verified recurrent models"] == "16"
    assert metrics["Registry-promoted models"] == "0"
    captions = "\n".join(item.value for item in app.caption)
    assert "optional promoted model registry contains no rows" in captions
    assert "No model registry records exist" not in captions


def test_symbol_details_uses_all_verified_models_across_run_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: list[object] = []
    selected_symbols = (
        "ENGROH", "FFC", "HUBC", "LUCK", "MARI", "MCB", "OGDC", "PSO",
        "SYS", "TRG", "UBL", "UNITY",
    )
    full_symbols = ("786", "AABS", "AATM", "ABL")
    all_symbols = tuple(sorted((*full_symbols, *selected_symbols)))
    snapshot = _snapshot(
        tmp_path / "selected",
        "COMPLETED",
        "COMPLETED",
        run_kind=SELECTED_RUN_KIND,
        run_id="run-selected",
        symbols=selected_symbols,
        controller_state="COMPLETED",
    )
    _patch_page_backend(monkeypatch, tmp_path, snapshot)
    monkeypatch.setattr(
        "reinforcement_learning.training.production_control.launch_production_controller",
        lambda store: launched.append(store),
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.selective_training.load_selected_run_metadata",
        lambda _: SimpleNamespace(
            selected_symbols=selected_symbols,
            selected_symbol_hash="c" * 64,
            attempt_version=0,
        ),
    )

    rows = []
    for index, symbol in enumerate(all_symbols):
        run_type = PRODUCTION_RUN_KIND if symbol in full_symbols else SELECTED_RUN_KIND
        validation_reference = f"validation/{symbol}/attempt_000.json"
        row = {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "sector": "COMMERCIAL BANKS",
                "run_type": run_type,
                "run_id": "run-production" if run_type == PRODUCTION_RUN_KIND else "run-selected",
                "attempt": 1,
                "training_status": "COMPLETED",
                "validation_status": "completed",
                "artifact_verification": "verified",
                "algorithm": "RecurrentPPO",
                "policy": "MlpLstmPolicy",
                "trainer_version": "recurrent_ppo_single_symbol_v1",
                "partition_contract_version": "rl_partition_v1",
                "recurrent_contract_version": "rl_recurrent_partition_v1",
                "feature_version": "feature-v1",
                "environment_version": "single_symbol_env_v1",
                "split_policy_version": "split-v1",
                "scaler_fit_partition": "train",
                "source_contract_sha256": "c" * 64,
                "hyperparameters_hash": "h" * 64,
                "seed": 42,
                "model_sha256": "m" * 64,
                "requested_timesteps": 100_000,
                "actual_timesteps": 100_352,
                "effective_device": "cpu",
                "runtime_seconds": 300.0,
                "raw_available_start": "2016-07-26",
                "raw_available_end": "2026-08-28",
                "raw_available_rows": 2_450,
                "usable_feature_start": "2016-10-06",
                "usable_feature_end": "2026-08-05",
                "usable_feature_rows": 2_435,
                "train_start": "2016-10-06",
                "train_end": "2023-08-23",
                "train_rows": 1_704,
                "validation_start": "2023-08-24",
                "validation_end": "2025-02-12",
                "validation_rows": 365,
                "test_start": "2025-02-13",
                "test_end": "2026-08-05",
                "test_rows": 366,
                "validation_partition": "validation",
                "validation_metrics_reference": validation_reference,
                "model_path": f"models/{symbol}/attempt_000/model.zip",
                "run_directory": str(tmp_path),
            }
        rows.append(row)
        validation_path = tmp_path / validation_reference
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "evaluation_partition": "validation",
                    "validation_start": row["validation_start"],
                    "validation_end": row["validation_end"],
                    "validation_rows": row["validation_rows"],
                    "recurrent_contract_version": row[
                        "recurrent_contract_version"
                    ],
                    "feature_version": row["feature_version"],
                    "environment_version": row["environment_version"],
                    "model_parameters_unchanged": True,
                    "parameter_hash_before": "p" * 64,
                    "parameter_hash_after": "p" * 64,
                    "model_timesteps_before": 100_352,
                    "model_timesteps_after": 100_352,
                    "strategy_result": {
                        "strategy": "RecurrentPPO",
                        "metrics": {
                            "total_return": index / 100,
                            "sharpe_ratio": index / 10,
                            "sortino_ratio": index / 8,
                            "maximum_drawdown": 0.05 + index / 100,
                            "annualized_volatility": 0.20,
                            "number_of_trades": 5,
                            "completed_trade_win_rate": 0.60,
                            "final_portfolio_value": 1_000_000 + index,
                            "daily_returns": [0.0, index / 100],
                            "metric_warnings": [],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        training_log = tmp_path / "logs" / symbol / "attempt_000.json"
        training_log.parent.mkdir(parents=True, exist_ok=True)
        training_log.write_text(
            json.dumps({"training_diagnostics": {"approximate_kl": 0.01}}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "reinforcement_learning.training.model_details.build_global_verified_model_inventory",
        lambda **_: pd.DataFrame(rows),
    )

    app = AppTest.from_file(str(PAGE_PATH), default_timeout=20).run()

    assert not app.exception
    selector = app.selectbox(key="training_detail_symbol")
    assert tuple(selector.options) == all_symbols
    rendered = "\n".join(
        str(item.value)
        for kind in ("caption", "markdown", "json", "info")
        for item in app.get(kind)
    )
    assert "Single-symbol RL partition protocol" in rendered
    assert "rl_partition_v1" in rendered
    assert "first floor(70%)" in rendered
    assert "next floor(15%)" in rendered
    assert "SEALED" in rendered
    assert "different common frozen temporal protocol" in rendered
    assert "fixed research cutoffs do not define" in rendered
    ranges = next(
        item
        for item in app.dataframe
        if "Range" in item.value.columns
    )
    assert ranges.value["Range"].tolist() == [
        "Current raw availability (date column only)",
        "Usable post-feature history",
        "Model-observed TRAIN range",
        "Model-observed VALIDATION range",
    ]
    detail_json = "\n".join(str(item.value) for item in app.get("json"))
    assert "Partition contract" in detail_json
    assert "Partition rule" in detail_json
    assert "TEST rows (metadata only)" in detail_json
    assert "TEST status" in detail_json
    assert "test_start" not in detail_json
    assert "test_end" not in detail_json
    assert not app.error, [item.value for item in app.error]
    comparison = next(
        item
        for item in app.dataframe
        if "validation_artifact_status" in item.value.columns
        and "validation_total_return" in item.value.columns
    )
    assert len(comparison.value) == 16
    assert comparison.value["validation_artifact_status"].eq("VALID").all()
    assert comparison.value.iloc[0]["symbol"] == "UNITY"
    page_text = "\n".join(
        str(item.value)
        for kind in ("caption", "markdown", "info")
        for item in app.get(kind)
    )
    assert "VALIDATION ONLY" in page_text
    assert "TEST SEALED" in page_text

    app.selectbox(key="training_detail_symbol").set_value("ENGROH").run()
    metrics = [(item.label, item.value) for item in app.metric]
    assert ("Run type", SELECTED_RUN_KIND) in metrics
    assert launched == []
    page_source = PAGE_PATH.read_text(encoding="utf-8")
    assert "load_rl_partition" not in page_source
    assert "load_recurrent_partition" not in page_source
    assert "test_rl.csv" not in page_source
