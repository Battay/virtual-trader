"""Offline, short-running tests for the 6C RecurrentPPO baseline."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.environments.config import DEFAULT_OBSERVATION_FEATURES
from reinforcement_learning.evaluation.recurrent_comparison import (
    compare_recurrent_and_mlp_on_validation,
)
from reinforcement_learning.evaluation.recurrent_evaluator import (
    evaluate_recurrent_on_validation,
)
from reinforcement_learning.evaluation.ppo_evaluator import ValidationEvaluationError
from reinforcement_learning.model_management.recurrent_persistence import (
    verify_temporary_recurrent_round_trip,
)
from reinforcement_learning.recurrent_data_contract import (
    load_recurrent_partition,
    persist_recurrent_contract,
)
from reinforcement_learning.training.config import PPOConfig
from reinforcement_learning.training.ppo_trainer import (
    create_training_vector_environment,
    train_single_symbol,
)
from reinforcement_learning.training.recurrent_config import (
    RECURRENT_PPO_CONFIG_VERSION,
    RecurrentPPOConfig,
)
from reinforcement_learning.training.recurrent_benchmark import (
    benchmark_recurrent_cpu_vs_mps,
)
from reinforcement_learning.training.recurrent_results import RecurrentDeviceRun
from reinforcement_learning.training.recurrent_trainer import (
    RECURRENT_TRAINER_VERSION,
    train_recurrent_single_symbol,
)


def _processed(rows: int = 180, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2020-01-01", periods=rows),
        "open": 100.0 + index,
        "high": 102.0 + index,
        "low": 99.0 + index,
        "close": 101.0 + index,
        "volume": 1_000.0 + 10 * index,
    }
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = (feature_index + 1) * 0.1 + index
    return pd.DataFrame(data)


def _artifacts(tmp_path: Path, *, symbol: str = "MCB") -> None:
    split = chronological_split(_processed(symbol=symbol), scope="symbol")
    persist_split_artifacts(split, tmp_path / "symbols" / symbol)
    persist_recurrent_contract(
        symbol,
        company=f"{symbol} Limited",
        sector="Commercial Banks",
        sector_verified=True,
        usable_observations=180,
        splits_dir=tmp_path,
    )


def _recurrent_config(*, timesteps: int = 8, device: str = "cpu") -> RecurrentPPOConfig:
    return RecurrentPPOConfig(
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        total_timesteps=timesteps,
        device=device,
    )


def _train(tmp_path: Path, *, timesteps: int = 8):
    _artifacts(tmp_path)
    result = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(timesteps=timesteps),
        splits_dir=tmp_path,
        smoke_test=timesteps <= 1_024,
    )
    assert result.succeeded, result.error
    return result


def test_dependency_and_versions_are_available() -> None:
    assert importlib.metadata.version("sb3-contrib") == "2.9.0"
    assert importlib.metadata.version("stable-baselines3") == "2.9.0"
    assert RECURRENT_PPO_CONFIG_VERSION == "recurrent_ppo_single_symbol_v1"
    assert RECURRENT_TRAINER_VERSION == "recurrent_ppo_single_symbol_v1"


def test_recurrent_config_is_explicit_and_rejects_invalid_lstm_combinations() -> None:
    config = RecurrentPPOConfig()
    kwargs = config.policy_kwargs()
    assert config.policy == "MlpLstmPolicy"
    assert kwargs["lstm_hidden_size"] == 64
    assert kwargs["n_lstm_layers"] == 1
    assert kwargs["shared_lstm"] is False
    assert kwargs["enable_critic_lstm"] is True
    assert kwargs["net_arch"] == {"pi": [64], "vf": [64]}
    assert kwargs["activation_fn"] is torch.nn.Tanh
    assert kwargs["ortho_init"] is True
    with pytest.raises(ValueError, match="cannot both be true"):
        RecurrentPPOConfig(shared_lstm=True, enable_critic_lstm=True)
    with pytest.raises(ValueError, match="batch_size must divide"):
        RecurrentPPOConfig(n_steps=10, batch_size=6)
    with pytest.raises(ValueError, match="activation_fn"):
        RecurrentPPOConfig(activation_fn="ELU")


def test_mature_symbol_smoke_uses_train_only_and_preserves_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifacts(tmp_path)
    source = load_recurrent_partition("MCB", "train", splits_dir=tmp_path)
    calls: list[str] = []
    original = load_recurrent_partition

    def guarded(symbol: str, partition: str, **kwargs: object):
        calls.append(partition)
        if partition != "train":
            raise AssertionError("optimizer accessed a non-TRAIN partition")
        return original(symbol, partition, **kwargs)

    monkeypatch.setattr(
        "reinforcement_learning.training.recurrent_trainer.load_recurrent_partition",
        guarded,
    )
    result = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(),
        splits_dir=tmp_path,
        smoke_test=True,
    )

    assert result.succeeded, result.error
    assert calls == ["train"]
    assert result.first_episode_start is True
    assert result.observation_shape == (17,)
    assert result.observation_features == DEFAULT_OBSERVATION_FEATURES
    assert result.training_rows == len(source.data)
    raw = pd.read_csv(tmp_path / "symbols" / "MCB" / "train.csv")
    for column in ("open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(source.data[column], raw[column])


@pytest.mark.parametrize(
    ("history_class", "usable", "independent", "transfer"),
    (("COLD_START", 100, False, True), ("INSUFFICIENT", 99, False, False)),
)
def test_non_mature_independent_training_is_rejected(
    tmp_path: Path,
    history_class: str,
    usable: int,
    independent: bool,
    transfer: bool,
) -> None:
    _artifacts(tmp_path)
    contract_path = (
        tmp_path / "symbols" / "MCB" / "recurrent" / "recurrent_contract.json"
    )
    contract = json.loads(contract_path.read_text())
    contract["history_policy"].update(
        {
            "history_class": history_class,
            "history_class_label": history_class.replace("_", " ").title(),
            "usable_observations": usable,
            "recurrent_artifact_eligible": False,
            "independent_recurrent_ready": independent,
            "transfer_fine_tune_eligible": transfer,
        }
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    result = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert result.status == "failed"
    assert result.model is None
    assert "Mature" in str(result.error)


def test_missing_recurrent_contract_fails_without_mlp_fallback(tmp_path: Path) -> None:
    split = chronological_split(_processed(), scope="symbol")
    persist_split_artifacts(split, tmp_path / "symbols" / "MCB")
    result = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert result.status == "failed"
    assert "recurrent RL contract is missing" in str(result.error)


def test_rollout_boundary_preserves_state_and_real_episode_auto_reset(
    tmp_path: Path,
) -> None:
    result = _train(tmp_path, timesteps=256)
    assert result.rollout_boundaries_observed == 32
    assert result.rollout_continuity_checks == 31
    assert result.rollout_continuity_verified
    assert result.rollout_start_episode_flags[0] is True
    assert any(flag is False for flag in result.rollout_start_episode_flags[1:])
    assert result.environment_episode_resets >= 2


def test_dummy_vec_env_auto_reset_preserves_terminal_observation(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    loaded = load_recurrent_partition("MCB", "train", splits_dir=tmp_path)
    vector = create_training_vector_environment(loaded.data, seed=42)
    try:
        first = vector.reset().copy()
        done = np.asarray([False])
        info = None
        for _ in range(len(loaded.data)):
            observation, _, done, infos = vector.step(np.asarray([0]))
            if bool(done[0]):
                info = infos[0]
                break
        assert bool(done[0])
        assert info is not None and "terminal_observation" in info
        np.testing.assert_array_equal(observation, first)
        assert not np.array_equal(info["terminal_observation"], observation[0])
    finally:
        vector.close()


def test_recurrent_validation_propagates_state_and_starts_fresh_each_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _train(tmp_path)
    original_predict = result.model.predict
    calls: list[tuple[bool, bool]] = []

    def spy(observation: object, *, state=None, episode_start=None, deterministic=False):
        calls.append((state is None, bool(np.asarray(episode_start)[0])))
        return original_predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    monkeypatch.setattr(result.model, "predict", spy)
    first = evaluate_recurrent_on_validation(
        result.model,
        "MCB",
        trainer_result=result,
        splits_dir=tmp_path,
    )
    first_call_count = len(calls)
    second = evaluate_recurrent_on_validation(
        result.model,
        "MCB",
        trainer_result=result,
        splits_dir=tmp_path,
    )

    assert calls[0] == (True, True)
    assert calls[1] == (False, False)
    assert calls[first_call_count] == (True, True)
    assert first.episode_reset_count == second.episode_reset_count == 1
    assert first.recurrent_state_steps == first.validation_rows - 2
    assert first.final_state_available
    assert not first.train_state_supplied
    assert first.model_parameters_unchanged
    pd.testing.assert_frame_equal(
        first.strategy_result.history,
        second.strategy_result.history,
    )


def test_recurrent_validation_rejects_cross_symbol_state_or_model_use(
    tmp_path: Path,
) -> None:
    result = _train(tmp_path)
    _artifacts(tmp_path, symbol="OGDC")
    with pytest.raises(ValidationEvaluationError, match="symbols differ"):
        evaluate_recurrent_on_validation(
            result.model,
            "OGDC",
            trainer_result=result,
            splits_dir=tmp_path,
        )


def test_training_validation_and_mlp_comparison_never_load_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifacts(tmp_path)
    original_read_csv = pd.read_csv
    reads: list[str] = []

    def guarded(path: object, *args: object, **kwargs: object):
        name = Path(path).name
        reads.append(name)
        if name in {"test.csv", "test_rl.csv"}:
            raise AssertionError("TEST frame accessed")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    recurrent = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    mlp = train_single_symbol(
        "MCB",
        config=PPOConfig(
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            total_timesteps=8,
        ),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    comparison = compare_recurrent_and_mlp_on_validation(
        recurrent,
        mlp,
        splits_dir=tmp_path,
    )
    assert len(comparison.strategies) == 5
    assert comparison.same_training_budget
    assert not comparison.test_evaluated
    assert "test.csv" not in reads and "test_rl.csv" not in reads


def test_temporary_recurrent_save_reload_is_exact_and_registry_free(
    tmp_path: Path,
) -> None:
    result = _train(tmp_path / "splits")
    registry = tmp_path / "model_registry.csv"
    registry.write_text("model_id\n", encoding="utf-8")
    before = registry.read_bytes()
    round_trip = verify_temporary_recurrent_round_trip(
        result,
        temporary_root=tmp_path / "outside_project_bundle",
        splits_dir=tmp_path / "splits",
        registry_path=registry,
    )
    assert round_trip.model_path.is_file()
    assert round_trip.policy_class == "RecurrentActorCriticPolicy"
    assert round_trip.saved_parameter_count == round_trip.loaded_parameter_count
    assert round_trip.deterministic_action_match
    assert round_trip.recurrent_state_match
    assert round_trip.architecture_match
    assert not round_trip.registry_touched
    assert registry.read_bytes() == before


def test_existing_mlp_trainer_remains_operational(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    result = train_single_symbol(
        "MCB",
        config=PPOConfig(
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            total_timesteps=8,
        ),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert result.succeeded, result.error
    assert result.algorithm == "PPO"
    assert result.ppo_config_version == "ppo_single_symbol_v1"


def test_cpu_path_uses_cpu(tmp_path: Path) -> None:
    result = _train(tmp_path)
    assert result.requested_device == result.resolved_device == "cpu"
    assert result.device.split(":", 1)[0] == "cpu"


def test_isolated_benchmark_recommends_measured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = {"symbol": "MCB", "seed": 42, "contract": "same"}

    def worker(**kwargs: object) -> RecurrentDeviceRun:
        device = str(kwargs["device"])
        duration = 10.0 if device == "cpu" else 5.0
        return RecurrentDeviceRun(
            device=device,
            status="completed",
            actual_device=device,
            requested_timesteps=5_120,
            actual_timesteps=5_120,
            duration_seconds=duration,
            timesteps_per_second=5_120 / duration,
            parameter_count=51_076,
            provenance=provenance,
        )

    monkeypatch.setattr(
        "reinforcement_learning.training.recurrent_benchmark._run_isolated_worker",
        worker,
    )
    result = benchmark_recurrent_cpu_vs_mps("MCB")
    assert result.isolated_subprocesses
    assert result.speedup_cpu_over_mps == 2.0
    assert result.recommended_device == "mps"
    assert not result.test_evaluated


def test_isolated_benchmark_preserves_cpu_when_mps_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def worker(**kwargs: object) -> RecurrentDeviceRun:
        device = str(kwargs["device"])
        return RecurrentDeviceRun(
            device=device,
            status="completed" if device == "cpu" else "failed",
            actual_device="cpu" if device == "cpu" else None,
            requested_timesteps=5_120,
            actual_timesteps=5_120 if device == "cpu" else 0,
            duration_seconds=10.0 if device == "cpu" else None,
            timesteps_per_second=512.0 if device == "cpu" else None,
            parameter_count=51_076 if device == "cpu" else 0,
            error=None if device == "cpu" else "native MPS failure",
        )

    monkeypatch.setattr(
        "reinforcement_learning.training.recurrent_benchmark._run_isolated_worker",
        worker,
    )
    result = benchmark_recurrent_cpu_vs_mps("MCB")
    assert result.cpu.succeeded
    assert not result.mps.succeeded
    assert result.recommended_device == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS is unavailable to this test process",
)
def test_real_mps_smoke_uses_mps_without_fallback(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    result = train_recurrent_single_symbol(
        "MCB",
        config=_recurrent_config(device="mps"),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert result.succeeded, result.error
    assert result.device.split(":", 1)[0] == "mps"
