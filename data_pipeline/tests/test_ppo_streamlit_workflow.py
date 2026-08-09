"""Offline tests for the safe, explicit Streamlit PPO workflow helpers."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt
import pytest

from dashboard.presentation import MISSING_VALUE
from dashboard.ppo_workflow import (
    CPU_MPS_SPEED_MULTIPLIER,
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    DEVICE_OPTIONS,
    DEVICE_RECOMMENDATION,
    PILOT_SYMBOLS,
    PPO_WORKFLOW_SESSION_KEY,
    TIMESTEP_PRESETS,
    PPOWorkflowIdentity,
    build_ready_symbol_catalog,
    build_workflow_identity,
    claim_workflow_job,
    initialize_workflow_session,
    mark_candidate_persisted,
    persistence_availability,
    pilot_readiness_table,
    registry_history_table,
    release_workflow_job,
    run_persistence_action,
    run_training_action,
    run_validation_action,
    selected_symbol_summary,
    sync_workflow_identity,
    training_availability,
    validation_availability,
    validation_chart_frames,
    validation_metrics_table,
)
from reinforcement_learning.data_contract import (
    RLContractMetadata,
    RLPartitionMetadata,
)
from reinforcement_learning.training.config import PPO_CONFIG_VERSION


def _metadata(symbol: str, contract_path: Path) -> RLContractMetadata:
    return RLContractMetadata(
        symbol=symbol,
        contract_path=contract_path,
        rl_contract_version="rl_partition_v1",
        environment_version="single_symbol_env_v1",
        feature_version="psx_ai_features_v1",
        observation_features=("return_1d", "sma_20"),
        dynamic_portfolio_features=("cash_fraction",),
        observation_shape=(17,),
        scaler_fit_partition="train",
        train=RLPartitionMetadata(
            name="train", rows=1_704, start="2016-10-06", end="2023-08-23"
        ),
        validation=RLPartitionMetadata(
            name="validation", rows=365, start="2023-08-24", end="2025-02-12"
        ),
        test=RLPartitionMetadata(
            name="test", rows=366, start="2025-02-13", end="2026-08-05"
        ),
    )


def _status_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["0786", "MCB", "OLD"], dtype="string"),
            "eligible": [True, True, False],
            "readiness_status": ["Ready", "Ready", "Insufficient History"],
            "company_name": ["Numeric Symbol", "MCB Bank", "Old Limited"],
            "train_rows": [1_704, 1_704, 10],
            "validation_rows": [365, 365, 2],
            "test_rows": [366, 366, 2],
            "first_usable_date": ["2016-10-06", "2016-10-06", "2026-01-01"],
            "last_usable_date": ["2026-08-05", "2026-08-05", "2026-01-20"],
        }
    )


def _catalog(tmp_path: Path):
    contracts: dict[str, Path] = {}
    for symbol in ("0786", "MCB"):
        path = tmp_path / symbol / "rl_contract.json"
        path.parent.mkdir(parents=True)
        path.write_text(f'{{"symbol": "{symbol}"}}', encoding="utf-8")
        (path.parent / "rl_observation_scaler.joblib").write_bytes(b"scaler")
        (path.parent / "rl_observation_scaler.json").write_text(
            '{"fit_partition": "train"}',
            encoding="utf-8",
        )
        for name in (
            "train.csv",
            "train_rl.csv",
            "validation.csv",
            "validation_rl.csv",
        ):
            (path.parent / name).write_text("fixture", encoding="utf-8")
        contracts[symbol] = path

    calls: list[tuple[str, Path]] = []

    def loader(symbol: str, *, splits_dir: Path) -> RLContractMetadata:
        calls.append((symbol, splits_dir))
        return _metadata(symbol, contracts[symbol])

    status = _status_table()
    original = status.copy(deep=True)
    catalog = build_ready_symbol_catalog(
        status,
        splits_dir=tmp_path / "splits",
        metadata_loader=loader,
    )
    pdt.assert_frame_equal(status, original)
    return catalog, calls


def _identity(symbol: str = "MCB", *, digest: str = "a" * 64, **overrides):
    values = {
        "symbol": symbol,
        "requested_timesteps": 10_000,
        "seed": 42,
        "requested_device": "cpu",
        "ppo_config_version": PPO_CONFIG_VERSION,
        "contract_sha256": digest,
        "observation_scaler_sha256": "d" * 64,
        "observation_scaler_metadata_sha256": "e" * 64,
        "train_validation_artifact_fingerprint": "1" * 64,
    }
    values.update(overrides)
    return PPOWorkflowIdentity(**values)


def _training(identity: PPOWorkflowIdentity, *, status: str = "completed"):
    return SimpleNamespace(
        status=status,
        model=object() if status == "completed" else None,
        symbol=identity.symbol,
        seed=identity.seed,
        requested_timesteps=identity.requested_timesteps,
        requested_device=identity.requested_device,
        ppo_config_version=identity.ppo_config_version,
        source_rl_contract_sha256=identity.contract_sha256,
        source_observation_scaler_sha256=identity.observation_scaler_sha256,
        source_observation_scaler_metadata_sha256=(
            identity.observation_scaler_metadata_sha256
        ),
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "execution_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "portfolio_value": [100_000.0, 101_000.0],
            "drawdown": [0.0, 0.01],
        }
    )


def _strategy(**metric_overrides):
    metrics = {
        "final_portfolio_value": 101_000.0,
        "total_return": 0.0123,
        "annualized_return": 0.10,
        "annualized_volatility": 0.20,
        "sharpe_ratio": 0.5,
        "sortino_ratio": None,
        "maximum_drawdown": 0.045,
        "number_of_trades": 3,
        "total_transaction_costs": 125.0,
        "exposure_percentage": 75.0,
        "realized_profit_loss": 900.0,
        "final_unrealized_profit_loss": 100.0,
    }
    metrics.update(metric_overrides)
    return SimpleNamespace(metrics=metrics, history=_history())


def _validation(
    identity: PPOWorkflowIdentity,
    *,
    status: str = "validation_pass",
):
    return SimpleNamespace(
        symbol=identity.symbol,
        evaluation_partition="validation",
        candidate_decision=SimpleNamespace(status=status, reasons=("fixture",)),
        ppo_parameter_hash_after="b" * 64,
        ppo=_strategy(),
        buy_and_hold=_strategy(),
        always_hold=_strategy(),
        random=_strategy(),
    )


def test_ready_symbol_catalog_intersects_readiness_and_valid_metadata(
    tmp_path: Path,
) -> None:
    catalog, calls = _catalog(tmp_path)

    assert catalog.ready_symbols == ("0786", "MCB")
    assert tuple(symbol for symbol, _ in calls) == ("0786", "MCB")
    assert all(path == tmp_path / "splits" for _, path in calls)
    assert catalog.rejected_reasons == {"OLD": "Insufficient History"}
    assert selected_symbol_summary(catalog, "0786").symbol == "0786"


def test_selected_symbol_summary_uses_metadata_only_and_keeps_test_sealed(
    tmp_path: Path,
) -> None:
    catalog, calls = _catalog(tmp_path)
    summary = selected_symbol_summary(catalog, "MCB")

    assert len(calls) == 2
    assert summary.train_rows == 1_704
    assert (summary.validation_start, summary.validation_end) == (
        "2023-08-24",
        "2025-02-12",
    )
    assert (summary.test_rows, summary.test_start, summary.test_end) == (
        366,
        "2025-02-13",
        "2026-08-05",
    )
    assert summary.observation_shape == (17,)
    assert len(summary.contract_sha256) == 64


def test_non_ready_symbol_fails_with_saved_reason(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)

    with pytest.raises(ValueError, match="Insufficient History"):
        selected_symbol_summary(catalog, "OLD")


def test_stale_contract_counts_are_excluded_from_ready_catalog(tmp_path: Path) -> None:
    status = _status_table()
    status.loc[status["symbol"].eq("MCB"), "train_rows"] = 1_705
    contracts: dict[str, Path] = {}
    for symbol in ("0786", "MCB"):
        path = tmp_path / symbol / "rl_contract.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        (path.parent / "rl_observation_scaler.joblib").write_bytes(b"scaler")
        (path.parent / "rl_observation_scaler.json").write_text("{}", encoding="utf-8")
        for name in (
            "train.csv",
            "train_rl.csv",
            "validation.csv",
            "validation_rl.csv",
        ):
            (path.parent / name).write_text("fixture", encoding="utf-8")
        contracts[symbol] = path

    catalog = build_ready_symbol_catalog(
        status,
        splits_dir=tmp_path,
        metadata_loader=lambda symbol, **_: _metadata(symbol, contracts[symbol]),
    )

    assert "MCB" not in catalog.ready_symbols
    assert "row counts differ" in catalog.rejected_reasons["MCB"]


def test_workflow_identity_uses_requested_config_and_contract_sha(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalog(tmp_path)
    summary = selected_symbol_summary(catalog, "MCB")

    identity = build_workflow_identity(summary, 25_000, 7, "MPS")

    assert identity == PPOWorkflowIdentity(
        symbol="MCB",
        requested_timesteps=25_000,
        seed=7,
        requested_device="mps",
        ppo_config_version=PPO_CONFIG_VERSION,
        contract_sha256=summary.contract_sha256,
        observation_scaler_sha256=summary.observation_scaler_sha256,
        observation_scaler_metadata_sha256=(
            summary.observation_scaler_metadata_sha256
        ),
        train_validation_artifact_fingerprint=(
            summary.train_validation_artifact_fingerprint
        ),
    )


def test_session_initialization_has_no_automatic_actions() -> None:
    session: dict[str, object] = {}

    state = initialize_workflow_session(session)

    assert session[PPO_WORKFLOW_SESSION_KEY] is state
    assert state["job_phase"] == "idle"
    assert state["training_result"] is None
    assert state["validation_result"] is None
    assert state["persisted_bundle"] is None


@pytest.mark.parametrize(
    "changed",
    [
        {"symbol": "OGDC"},
        {"requested_timesteps": 25_000},
        {"seed": 43},
        {"requested_device": "auto"},
        {"digest": "c" * 64},
        {"observation_scaler_sha256": "f" * 64},
        {"observation_scaler_metadata_sha256": "0" * 64},
        {"train_validation_artifact_fingerprint": "2" * 64},
    ],
)
def test_config_change_invalidates_stale_training_and_validation(changed) -> None:
    session: dict[str, object] = {}
    original = _identity()
    sync_workflow_identity(session, original)
    state = initialize_workflow_session(session)
    state["training_result"] = _training(original)
    state["validation_result"] = _validation(original)
    state["persisted_bundle"] = object()
    state["persisted_candidate_key"] = object()

    replacement = _identity(**changed)
    assert sync_workflow_identity(session, replacement)

    assert state["identity"] == replacement
    assert state["training_result"] is None
    assert state["validation_result"] is None
    assert state["persisted_bundle"] is None
    assert state["persisted_candidate_key"] is None


def test_overlapping_job_and_duplicate_training_are_blocked() -> None:
    session: dict[str, object] = {}
    identity = _identity()
    sync_workflow_identity(session, identity)

    assert claim_workflow_job(session, identity, "training").allowed
    assert not claim_workflow_job(session, identity, "validating").allowed
    assert not training_availability(session, identity).allowed
    with pytest.raises(RuntimeError, match="while a job is running"):
        sync_workflow_identity(session, _identity(seed=9))

    release_workflow_job(session)
    initialize_workflow_session(session)["training_result"] = _training(identity)
    availability = training_availability(session, identity)
    assert not availability.allowed
    assert "already" in availability.reason


def test_job_claim_enforces_phase_prerequisites_and_replay_guards() -> None:
    session: dict[str, object] = {}
    identity = _identity()
    sync_workflow_identity(session, identity)

    assert not claim_workflow_job(session, identity, "validating").allowed
    assert not claim_workflow_job(session, identity, "persisting").allowed
    state = initialize_workflow_session(session)
    state["training_result"] = _training(identity)
    assert not claim_workflow_job(session, identity, "training").allowed
    assert claim_workflow_job(session, identity, "validating").allowed
    release_workflow_job(session)
    state["validation_result"] = _validation(identity)
    assert not claim_workflow_job(session, identity, "validating").allowed
    assert claim_workflow_job(session, identity, "persisting").allowed
    release_workflow_job(session)


def test_validation_is_available_only_after_matching_completed_training() -> None:
    session: dict[str, object] = {}
    identity = _identity()
    sync_workflow_identity(session, identity)

    assert not validation_availability(session, identity).allowed
    initialize_workflow_session(session)["training_result"] = _training(identity)
    assert validation_availability(session, identity).allowed
    initialize_workflow_session(session)["validation_result"] = _validation(identity)
    assert not validation_availability(session, identity).allowed


@pytest.mark.parametrize(
    ("decision", "expected"),
    [("validation_pass", True), ("validation_fail", False), ("evaluation_error", False)],
)
def test_persistence_availability_requires_validation_pass(
    decision: str,
    expected: bool,
) -> None:
    session: dict[str, object] = {}
    identity = _identity()
    sync_workflow_identity(session, identity)
    state = initialize_workflow_session(session)
    state["training_result"] = _training(identity)
    state["validation_result"] = _validation(identity, status=decision)

    assert persistence_availability(session, identity).allowed is expected


def test_persisted_candidate_marker_prevents_duplicate_save() -> None:
    session: dict[str, object] = {}
    identity = _identity()
    validation = _validation(identity)
    sync_workflow_identity(session, identity)
    state = initialize_workflow_session(session)
    state["training_result"] = _training(identity)
    state["validation_result"] = validation

    mark_candidate_persisted(session, identity, validation, bundle="saved")

    assert state["persisted_bundle"] == "saved"
    availability = persistence_availability(session, identity)
    assert not availability.allowed
    assert "already persisted" in availability.reason


def test_training_wrapper_uses_production_train_boundary_only(tmp_path: Path) -> None:
    identity = _identity(requested_timesteps=25_000, seed=11, requested_device="auto")
    captured: dict[str, object] = {}

    def trainer(symbol: str, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)
        return "trained"

    result = run_training_action(identity, splits_dir=tmp_path, trainer=trainer)

    assert result == "trained"
    assert captured["symbol"] == "MCB"
    config = captured["config"]
    assert config.seed == 11
    assert config.total_timesteps == 25_000
    assert config.device == "auto"
    assert captured["splits_dir"] == tmp_path
    assert captured["output_dir"] is None
    assert captured["smoke_test"] is False
    assert "partition" not in captured
    assert "validation" not in captured
    assert "test" not in captured


def test_validation_wrapper_exposes_only_validation_api(tmp_path: Path) -> None:
    identity = _identity()
    training = _training(identity)
    captured: dict[str, object] = {}

    def evaluator(model, symbol: str, **kwargs):
        captured["model"] = model
        captured["symbol"] = symbol
        captured.update(kwargs)
        return "validated"

    result = run_validation_action(
        training,
        identity,
        splits_dir=tmp_path,
        evaluator=evaluator,
    )

    assert result == "validated"
    assert captured["model"] is training.model
    assert captured["symbol"] == "MCB"
    assert captured["trainer_result"] is training
    assert captured["deterministic_seed"] == 42
    assert captured["random_seed"] == 42
    assert captured["splits_dir"] == tmp_path
    assert "partition" not in captured
    assert all("test" not in str(key).lower() for key in captured)


def test_validation_fail_cannot_call_production_persister(tmp_path: Path) -> None:
    identity = _identity()
    called = False

    def persister(*args, **kwargs):
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="requires validation_pass"):
        run_persistence_action(
            _training(identity),
            _validation(identity, status="validation_fail"),
            identity,
            registry_path=tmp_path / "registry.csv",
            saved_models_dir=tmp_path / "models",
            splits_dir=tmp_path / "splits",
            persister=persister,
        )

    assert not called


def test_validation_pass_invokes_candidate_persistence_without_promotion(
    tmp_path: Path,
) -> None:
    identity = _identity()
    training = _training(identity)
    validation = _validation(identity)
    captured: dict[str, object] = {}

    def persister(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "bundle"

    result = run_persistence_action(
        training,
        validation,
        identity,
        registry_path=tmp_path / "registry.csv",
        saved_models_dir=tmp_path / "models",
        splits_dir=tmp_path / "splits",
        persister=persister,
    )

    assert result == "bundle"
    assert captured["args"] == (training, validation)
    assert captured["symbol"] == "MCB"
    assert captured["registry_path"] == tmp_path / "registry.csv"
    assert captured["saved_models_dir"] == tmp_path / "models"
    assert "promotion" not in captured


def test_pilot_readiness_uses_fixed_order_and_no_train_all_action() -> None:
    table = pilot_readiness_table({"OGDC", "FFC", "TRG"})

    assert tuple(table["Symbol"]) == PILOT_SYMBOLS
    assert table.loc[table["Symbol"] == "OGDC", "Status"].item() == "Ready"
    assert table.loc[table["Symbol"] == "UBL", "Status"].item() == "Not Ready"
    assert "Train" not in " ".join(table.columns)


def test_device_recommendation_and_safe_presets_are_explicit() -> None:
    assert DEFAULT_DEVICE == "cpu"
    assert DEFAULT_SEED == 42
    assert DEVICE_OPTIONS == ("cpu", "mps", "auto")
    assert TIMESTEP_PRESETS == (10_000, 25_000, 50_000, 100_000)
    assert CPU_MPS_SPEED_MULTIPLIER == pytest.approx(6.81)
    assert "recommended" in DEVICE_RECOMMENDATION.lower()


def test_empty_registry_history_has_stable_columns() -> None:
    display = registry_history_table(pd.DataFrame())

    assert display.empty
    assert {"Model ID", "Validation Status", "Promotion Status", "Model Status"}.issubset(
        display.columns
    )


def test_registry_history_preserves_all_versions_and_source() -> None:
    registry = pd.DataFrame(
        {
            "model_id": ["ppo-symbol-MCB-v0001", "ppo-symbol-MCB-v0002"],
            "symbol": pd.Series(["MCB", "MCB"], dtype="string"),
            "model_version": [1, 2],
            "algorithm": ["PPO", "PPO"],
            "validation_status": ["validation_pass", "validation_pass"],
            "promotion_status": ["candidate", "candidate"],
            "environment_version": ["single_symbol_env_v1"] * 2,
            "feature_version": ["psx_ai_features_v1"] * 2,
            "created_at": ["2026-08-08T00:00:00+00:00", "2026-08-09T00:00:00+00:00"],
            "last_trained_at": ["2026-08-08T00:00:00+00:00", "2026-08-09T00:00:00+00:00"],
            "training_data_start": ["2016-10-06"] * 2,
            "training_data_end": ["2023-08-23"] * 2,
            "validation_data_start": ["2023-08-24"] * 2,
            "validation_data_end": ["2025-02-12"] * 2,
            "random_seed": [42, 7],
            "model_status": ["candidate", "candidate"],
        }
    )
    original = registry.copy(deep=True)

    display = registry_history_table(registry)

    pdt.assert_frame_equal(registry, original)
    assert len(display) == 2
    assert display["Model ID"].tolist() == [
        "ppo-symbol-MCB-v0002",
        "ppo-symbol-MCB-v0001",
    ]


def test_validation_metrics_format_fractions_but_not_exposure_percent() -> None:
    identity = _identity()
    result = _validation(identity)
    result.ppo.metrics["sortino_ratio"] = float("nan")

    table = validation_metrics_table(result).set_index("Metric")

    assert table.loc["Total Return", "PPO"] == "1.23%"
    assert table.loc["Annualized Return", "PPO"] == "10.00%"
    assert table.loc["Annualized Volatility", "PPO"] == "20.00%"
    assert table.loc["Maximum Drawdown", "PPO"] == "4.50%"
    assert table.loc["Exposure", "PPO"] == "75.00%"
    assert table.loc["Sortino", "PPO"] == MISSING_VALUE


def test_validation_chart_frames_are_copies_and_do_not_mutate_histories() -> None:
    result = _validation(_identity())
    originals = {
        name: getattr(result, name).history.copy(deep=True)
        for name in ("ppo", "buy_and_hold", "always_hold", "random")
    }

    portfolio, drawdown = validation_chart_frames(result)
    portfolio.loc[0, "Portfolio Value"] = -1
    drawdown.loc[0, "Drawdown"] = -1

    for name, original in originals.items():
        pdt.assert_frame_equal(getattr(result, name).history, original)
    assert set(portfolio["Strategy"]) == {
        "PPO",
        "Buy & Hold",
        "Always Hold",
        "Random (seed 42)",
    }
    assert len(portfolio) == len(drawdown) == 8
