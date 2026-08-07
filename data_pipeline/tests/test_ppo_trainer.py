"""Offline tests for the single-symbol Stable-Baselines3 PPO trainer core."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.config import MODEL_REGISTRY_PATH, SAVED_MODELS_DIR
from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.data_contract import (
    EXECUTION_ACCOUNTING_COLUMNS,
    RL_PARTITION_SCHEMA_VERSION,
    load_rl_partition,
)
from reinforcement_learning.environments.config import ENVIRONMENT_VERSION
from reinforcement_learning.model_management import registry as model_registry
from reinforcement_learning.training.config import PPO_CONFIG_VERSION, PPOConfig
from reinforcement_learning.training.ppo_trainer import (
    MAX_SMOKE_TIMESTEPS,
    create_training_vector_environment,
    train_single_symbol,
)


def _processed(rows: int = 30, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2025-01-01", periods=rows),
        "open": 100.0 + index,
        "high": 102.0 + index,
        "low": 99.0 + index,
        "close": 101.0 + index,
        "volume": 10_000.0 + 100 * index,
    }
    for position, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = (position + 1) * 0.25 + index
    return pd.DataFrame(data)


@pytest.fixture
def rl_splits(tmp_path: Path):
    source = _processed()
    split = chronological_split(source, scope="symbol")
    result = persist_split_artifacts(split, tmp_path / "symbols" / "MCB")
    return tmp_path, source, split, result


def _hash_files(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tiny_config(**overrides) -> PPOConfig:
    return replace(
        PPOConfig(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        total_timesteps=16,
        **overrides,
    )


def test_ppo_config_defaults_are_versioned_and_exact() -> None:
    config = PPOConfig()
    assert config.to_dict() == {
        "config_version": PPO_CONFIG_VERSION,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.01,
        "vf_coef": 0.50,
        "max_grad_norm": 0.50,
        "seed": 42,
        "total_timesteps": 100_000,
        "device": "cpu",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"config_version": "old"}, "config_version"),
        ({"policy": "CnnPolicy"}, "MlpPolicy"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"n_steps": 1}, "n_steps"),
        ({"n_steps": 8, "batch_size": 3}, "divide"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"gamma": 1.1}, "gamma"),
        ({"gae_lambda": -0.1}, "gae_lambda"),
        ({"clip_range": 0}, "clip_range"),
        ({"ent_coef": -0.1}, "ent_coef"),
        ({"max_grad_norm": 0}, "max_grad_norm"),
        ({"seed": -1}, "seed"),
        ({"total_timesteps": 0}, "total_timesteps"),
        ({"device": "auto"}, "CPU"),
    ),
)
def test_ppo_config_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PPOConfig(**overrides)


def test_training_vector_environment_is_seeded_and_preserves_real_prices(
    rl_splits,
) -> None:
    splits_dir, _, split, _ = rl_splits
    loaded = load_rl_partition("MCB", "train", splits_dir=splits_dir)
    first = create_training_vector_environment(loaded.data, seed=13)
    second = create_training_vector_environment(loaded.data, seed=13)
    try:
        first_observation = first.reset()
        second_observation = second.reset()
        assert first.num_envs == 1
        assert first_observation.shape == (1, 17)
        np.testing.assert_allclose(first_observation, second_observation)
        assert first.action_space.sample() == second.action_space.sample()
        inner_data = first.envs[0].unwrapped._data
        for column in EXECUTION_ACCOUNTING_COLUMNS:
            np.testing.assert_array_equal(inner_data[column], split.train[column])
    finally:
        first.close()
        second.close()


def test_real_tiny_ppo_uses_canonical_train_partition_and_returns_result(
    rl_splits,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    splits_dir, _, split, _ = rl_splits
    source_artifacts = splits_dir / "symbols" / "MCB"
    before = _hash_files(source_artifacts)
    original_loader = load_rl_partition
    loader_calls: list[tuple[str, str, Path]] = []

    def guarded_loader(symbol: str, partition: str, *, splits_dir: Path):
        loader_calls.append((symbol, partition, Path(splits_dir)))
        if partition != "train":
            raise AssertionError("validation/test data must remain sealed")
        return original_loader(symbol, partition, splits_dir=splits_dir)

    save_calls: list[object] = []

    def forbidden_save(*args, **kwargs):
        save_calls.append((args, kwargs))
        raise AssertionError("5B-1 must not save a PPO model")

    def forbidden_registry(*args, **kwargs):
        raise AssertionError("5B-1 must not write the model registry")

    monkeypatch.setattr(
        "reinforcement_learning.training.ppo_trainer.load_rl_partition",
        guarded_loader,
    )
    monkeypatch.setattr(
        "reinforcement_learning.training.ppo_trainer.PPO.save", forbidden_save
    )
    monkeypatch.setattr(model_registry, "append_model_version", forbidden_registry)
    output_dir = tmp_path / "developer-output"
    output_dir.mkdir()
    sentinel = output_dir / "existing-production-like-model.zip"
    sentinel.write_bytes(b"do not overwrite")

    config = _tiny_config(seed=7)
    result = train_single_symbol(
        "MCB",
        config=config,
        output_dir=output_dir,
        splits_dir=splits_dir,
        smoke_test=True,
    )

    assert result.succeeded
    assert result.status == "completed"
    assert result.symbol == "MCB"
    assert result.algorithm == "PPO"
    assert result.ppo_config_version == PPO_CONFIG_VERSION
    assert result.environment_version == ENVIRONMENT_VERSION
    assert result.rl_contract_version == RL_PARTITION_SCHEMA_VERSION
    assert result.seed == 7
    assert result.requested_timesteps == 16
    assert result.actual_timesteps == 16
    assert result.training_rows == len(split.train)
    assert result.training_start == split.train["date"].min().date().isoformat()
    assert result.training_end == split.train["date"].max().date().isoformat()
    assert result.observation_shape == (17,)
    assert result.device == "cpu"
    assert result.duration_seconds >= 0
    assert result.model is not None
    assert result.model.seed == 7
    assert "model" not in result.to_dict()
    assert loader_calls == [("MCB", "train", splits_dir)]
    assert save_calls == []
    assert sentinel.read_bytes() == b"do not overwrite"
    assert _hash_files(source_artifacts) == before


def test_invalid_and_incompatible_symbols_fail_before_training(rl_splits) -> None:
    splits_dir, _, _, result = rl_splits
    missing = train_single_symbol(
        "NOT_READY",
        config=_tiny_config(),
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert missing.status == "failed"
    assert missing.model is None
    assert "RL contract is missing" in str(missing.error)

    contract_path = result.rl_artifacts.contract_path
    original = contract_path.read_text(encoding="utf-8")
    contract_path.write_text(
        original.replace(RL_PARTITION_SCHEMA_VERSION, "rl_partition_stale", 1),
        encoding="utf-8",
    )
    stale = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert stale.status == "failed"
    assert stale.model is None
    assert "Incompatible RL artifact schema" in str(stale.error)


def test_trainer_rejects_a_non_train_loader_result(
    rl_splits,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits_dir, _, _, _ = rl_splits

    def wrong_partition(symbol: str, partition: str, *, splits_dir: Path):
        assert partition == "train"
        return load_rl_partition(symbol, "validation", splits_dir=splits_dir)

    monkeypatch.setattr(
        "reinforcement_learning.training.ppo_trainer.load_rl_partition",
        wrong_partition,
    )
    result = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert result.status == "failed"
    assert result.model is None
    assert "non-training partition" in str(result.error)


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    ((KeyboardInterrupt(), "interrupted"), (RuntimeError("boom"), "failed")),
)
def test_interruption_and_failure_never_write_models_or_registry(
    rl_splits,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_status: str,
) -> None:
    splits_dir, _, _, _ = rl_splits
    registry_before = Path(MODEL_REGISTRY_PATH).read_bytes()
    models_before = _hash_files(Path(SAVED_MODELS_DIR))

    def fail_learning(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        "reinforcement_learning.training.ppo_trainer.PPO.learn", fail_learning
    )
    result = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        splits_dir=splits_dir,
        smoke_test=True,
    )

    assert result.status == expected_status
    assert result.model is None
    assert Path(MODEL_REGISTRY_PATH).read_bytes() == registry_before
    assert _hash_files(Path(SAVED_MODELS_DIR)) == models_before


def test_callback_cancellation_is_interrupted_without_a_model(rl_splits) -> None:
    splits_dir, _, _, _ = rl_splits
    events = []

    def cancel_after_first_progress(event):
        events.append(event)
        return False if event.phase == "progress" else None

    result = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        progress_callback=cancel_after_first_progress,
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert result.status == "interrupted"
    assert result.model is None
    assert any(event.phase == "progress" for event in events)


def test_production_output_roots_are_rejected_without_writing(rl_splits) -> None:
    splits_dir, _, _, _ = rl_splits
    before = _hash_files(Path(SAVED_MODELS_DIR))
    result = train_single_symbol(
        "MCB",
        config=_tiny_config(),
        output_dir=Path(SAVED_MODELS_DIR) / "symbol_models" / "MCB" / "v9999",
        splits_dir=splits_dir,
        smoke_test=True,
    )
    assert result.status == "failed"
    assert result.model is None
    assert "production model directory" in str(result.error)
    assert _hash_files(Path(SAVED_MODELS_DIR)) == before


def test_smoke_mode_refuses_long_runs_before_loading(monkeypatch) -> None:
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("over-limit smoke run must fail before data loading")

    monkeypatch.setattr(
        "reinforcement_learning.training.ppo_trainer.load_rl_partition",
        forbidden_loader,
    )
    result = train_single_symbol(
        "MCB",
        config=replace(PPOConfig(), total_timesteps=MAX_SMOKE_TIMESTEPS + 1),
        smoke_test=True,
    )
    assert result.status == "failed"
    assert "capped" in str(result.error)
