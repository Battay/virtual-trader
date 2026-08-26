"""Offline tests for the single-symbol Stable-Baselines3 PPO trainer core."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data_pipeline.src.config import MODEL_REGISTRY_PATH, SAVED_MODELS_DIR
from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.data_contract import (
    EXECUTION_ACCOUNTING_COLUMNS,
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
    RL_PARTITION_SCHEMA_VERSION,
    load_rl_partition,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.model_management import registry as model_registry
from reinforcement_learning.training.config import PPO_CONFIG_VERSION, PPOConfig
from reinforcement_learning.training.devices import (
    TorchDeviceError,
    TorchDeviceResolution,
    resolve_torch_device,
    synchronize_torch_device,
    torch_devices_equivalent,
)
import reinforcement_learning.training.devices as training_devices
import reinforcement_learning.training.ppo_trainer as ppo_trainer_module
from reinforcement_learning.training.ppo_trainer import (
    MAX_SMOKE_TIMESTEPS,
    create_training_vector_environment,
    train_single_symbol,
)
from reinforcement_learning.training.results import PPOTrainingDiagnostics


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


def test_training_diagnostics_copy_only_finite_whitelisted_sb3_values() -> None:
    diagnostics = PPOTrainingDiagnostics.from_sb3_logger_values(
        {
            "train/approx_kl": np.float32(0.0125),
            "train/clip_fraction": np.float64(0.25),
            "train/entropy_loss": np.float64(-1.05),
            "train/explained_variance": np.float64(0.4),
            "train/policy_gradient_loss": np.float64(-0.02),
            "train/value_loss": np.nan,
            "train/learning_rate": np.float64(3e-4),
            "train/n_updates": np.int64(20),
            "train/unrelated_internal_value": 999,
        },
        timesteps=1_024,
    )

    assert diagnostics.to_dict() == {
        "timesteps": 1_024,
        "updates": 20,
        "approximate_kl": pytest.approx(0.0125),
        "clip_fraction": pytest.approx(0.25),
        "entropy_loss": pytest.approx(-1.05),
        "explained_variance": pytest.approx(0.4),
        "policy_gradient_loss": pytest.approx(-0.02),
        "value_loss": None,
        "learning_rate": pytest.approx(3e-4),
    }


def test_training_diagnostics_do_not_fabricate_unavailable_values() -> None:
    diagnostics = PPOTrainingDiagnostics.from_sb3_logger_values(
        None,
        timesteps=512,
    )

    assert diagnostics.timesteps == 512
    assert diagnostics.updates is None
    assert all(
        value is None
        for name, value in diagnostics.to_dict().items()
        if name != "timesteps"
    )


def test_sha256_file_reads_in_bounded_chunks(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    payload = bytes(range(256)) * 20
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=17) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="positive integer"):
        sha256_file(path, chunk_size=0)


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
        ({"device": "xpu"}, "device"),
    ),
)
def test_ppo_config_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PPOConfig(**overrides)


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps", "auto"))
def test_ppo_config_accepts_supported_device_requests(device: str) -> None:
    config = PPOConfig(device=device)

    assert config.device == device
    if device == "auto":
        with pytest.raises(ValueError, match="must be resolved"):
            config.model_kwargs()
        assert config.model_kwargs(resolved_device="cpu")["device"] == "cpu"
    else:
        assert config.model_kwargs()["device"] == device


@pytest.mark.parametrize("device", (None, 7, True, "cuda:1", "mps:0"))
def test_ppo_config_rejects_bad_device_types_and_unsupported_devices(device) -> None:
    with pytest.raises(ValueError, match="auto, cpu, cuda, mps"):
        PPOConfig(device=device)


@pytest.mark.parametrize("mps_state", ((False, False), (True, True)))
def test_cpu_device_resolution_is_explicit_and_accelerator_independent(
    monkeypatch: pytest.MonkeyPatch,
    mps_state: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: mps_state)
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))

    resolution = resolve_torch_device("cpu")

    assert resolution.requested_device == "cpu"
    assert resolution.resolved_device == "cpu"
    assert (resolution.mps_built, resolution.mps_available) == mps_state
    assert not resolution.accelerator_selected


def test_explicit_mps_resolves_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, True))
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))

    resolution = resolve_torch_device("mps")

    assert resolution.requested_device == "mps"
    assert resolution.resolved_device == "mps"
    assert resolution.accelerator_selected
    assert torch_devices_equivalent("mps", "mps:0")


def test_mps_resolution_rejects_external_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, True))
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    with pytest.raises(TorchDeviceError, match="unsupported operations could be hidden"):
        resolve_torch_device("mps")


def test_mps_synchronization_failure_is_reported_as_device_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, True))
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))
    monkeypatch.setattr(
        training_devices.torch.mps,
        "synchronize",
        lambda: (_ for _ in ()).throw(RuntimeError("asynchronous kernel failed")),
    )

    with pytest.raises(TorchDeviceError, match="cannot be reported as successful"):
        synchronize_torch_device("mps")


@pytest.mark.parametrize("mps_state", ((False, False), (True, False)))
def test_explicit_mps_unavailable_fails_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    mps_state: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: mps_state)
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))

    with pytest.raises(TorchDeviceError, match="CPU fallback is disabled"):
        resolve_torch_device("mps")


@pytest.mark.parametrize(
    ("cuda_state", "mps_state", "expected"),
    (
        ((False, 0, None), (False, False), "cpu"),
        ((False, 0, None), (True, True), "cpu"),
        ((True, 1, "Mock CUDA"), (True, True), "cuda"),
    ),
)
def test_auto_device_resolution_prefers_cuda_then_cpu_never_mps(
    monkeypatch: pytest.MonkeyPatch,
    cuda_state: tuple[bool, int, str | None],
    mps_state: tuple[bool, bool],
    expected: str,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: mps_state)
    monkeypatch.setattr(training_devices, "_cuda_state", lambda: cuda_state)

    resolution = resolve_torch_device("auto")

    assert resolution.requested_device == "auto"
    assert resolution.resolved_device == expected


def test_explicit_cuda_resolves_only_when_available_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, True))
    monkeypatch.setattr(
        training_devices, "_cuda_state", lambda: (True, 1, "Mock CUDA GPU")
    )

    resolution = resolve_torch_device("cuda")

    assert resolution.resolved_device == "cuda"
    assert resolution.device_name == "Mock CUDA GPU"
    assert resolution.accelerator_selected
    assert torch_devices_equivalent("cuda", "cuda:0")

    monkeypatch.setattr(training_devices, "_cuda_state", lambda: (False, 0, None))
    with pytest.raises(TorchDeviceError, match="fallback is disabled"):
        resolve_torch_device("cuda")


@pytest.mark.parametrize("requested", (None, 3, True, "cuda:0", "xpu"))
def test_device_resolution_rejects_bad_types_and_unknown_devices(requested) -> None:
    with pytest.raises(TorchDeviceError, match="auto, cpu, cuda, mps"):
        resolve_torch_device(requested)


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
    contract_path = (source_artifacts / RL_CONTRACT_FILENAME).resolve()
    scaler_path = (source_artifacts / RL_OBSERVATION_SCALER_FILENAME).resolve()
    scaler_metadata_path = scaler_path.with_suffix(".json")
    assert result.source_rl_contract_path == str(contract_path)
    assert result.source_rl_contract_sha256 == sha256_file(contract_path)
    assert result.source_observation_scaler_path == str(scaler_path)
    assert result.source_observation_scaler_sha256 == sha256_file(scaler_path)
    assert result.source_observation_scaler_metadata_path == str(
        scaler_metadata_path
    )
    assert result.source_observation_scaler_metadata_sha256 == sha256_file(
        scaler_metadata_path
    )
    assert result.observation_features == DEFAULT_OBSERVATION_FEATURES
    assert result.seed == 7
    assert result.requested_timesteps == 16
    assert result.actual_timesteps == 16
    assert result.training_rows == len(split.train)
    assert result.training_start == split.train["date"].min().date().isoformat()
    assert result.training_end == split.train["date"].max().date().isoformat()
    assert result.observation_shape == (17,)
    assert result.device == "cpu"
    assert result.requested_device == "cpu"
    assert result.resolved_device == "cpu"
    assert torch_devices_equivalent(result.model.device, result.resolved_device)
    assert result.duration_seconds >= 0
    assert result.model is not None
    assert result.model.seed == 7
    assert result.training_diagnostics is not None
    diagnostics = result.training_diagnostics
    assert diagnostics.timesteps == result.actual_timesteps
    assert diagnostics.updates == 2
    assert diagnostics.learning_rate == pytest.approx(config.learning_rate)
    assert diagnostics.approximate_kl is not None
    assert diagnostics.clip_fraction is not None
    assert 0.0 <= diagnostics.clip_fraction <= 1.0
    assert diagnostics.entropy_loss is not None
    assert diagnostics.explained_variance is not None
    assert diagnostics.policy_gradient_loss is not None
    assert diagnostics.value_loss is not None
    serialized = result.to_dict()
    assert "model" not in serialized
    assert serialized["requested_device"] == "cpu"
    assert serialized["resolved_device"] == "cpu"
    assert serialized["device"] == "cpu"
    assert serialized["training_diagnostics"] == diagnostics.to_dict()
    assert loader_calls == [("MCB", "train", splits_dir)]
    assert save_calls == []
    assert sentinel.read_bytes() == b"do not overwrite"
    assert _hash_files(source_artifacts) == before


def test_auto_resolution_is_passed_to_ppo_and_recorded_in_result(
    rl_splits,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits_dir, _, _, _ = rl_splits
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, False))
    original_ppo = ppo_trainer_module.PPO
    constructor_devices: list[str] = []

    def capture_ppo_device(*args, **kwargs):
        constructor_devices.append(str(kwargs.get("device")))
        return original_ppo(*args, **kwargs)

    monkeypatch.setattr(ppo_trainer_module, "PPO", capture_ppo_device)

    result = train_single_symbol(
        "MCB",
        config=_tiny_config(device="auto"),
        splits_dir=splits_dir,
        smoke_test=True,
    )

    assert result.succeeded
    assert constructor_devices == ["cpu"]
    assert result.requested_device == "auto"
    assert result.resolved_device == "cpu"
    assert result.device == "cpu"
    assert result.ppo_config["device"] == "auto"
    assert torch_devices_equivalent(result.model.device, "cpu")


def test_trainer_explicit_mps_unavailable_never_loads_data_or_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_devices, "_mps_state", lambda: (True, False))
    loader_calls: list[object] = []

    def forbidden_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        raise AssertionError("unavailable explicit MPS must fail before data loading")

    monkeypatch.setattr(ppo_trainer_module, "load_rl_partition", forbidden_loader)

    result = train_single_symbol(
        "MCB",
        config=_tiny_config(device="mps"),
        smoke_test=True,
    )

    assert result.status == "failed"
    assert result.requested_device == "mps"
    assert result.resolved_device is None
    assert "CPU fallback is disabled" in str(result.error)
    assert loader_calls == []


def test_actual_device_mismatch_aborts_before_learning(
    rl_splits,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits_dir, _, _, _ = rl_splits
    learn_calls: list[object] = []

    class MismatchedPPO:
        device = torch.device("mps")
        num_timesteps = 0

        def __init__(self, *args, **kwargs) -> None:
            self.policy = object()

        def learn(self, *args, **kwargs):
            learn_calls.append((args, kwargs))
            raise AssertionError("learn must not run after a device mismatch")

    monkeypatch.setattr(ppo_trainer_module, "PPO", MismatchedPPO)

    result = train_single_symbol(
        "MCB",
        config=_tiny_config(device="cpu"),
        splits_dir=splits_dir,
        smoke_test=True,
    )

    assert result.status == "failed"
    assert result.model is None
    assert result.requested_device == "cpu"
    assert result.resolved_device == "cpu"
    assert "does not match the resolved device" in str(result.error)
    assert learn_calls == []


def test_mps_specific_seed_is_called_only_for_resolved_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_seed_calls: list[int] = []
    mps_seed_calls: list[int] = []
    sb3_seed_calls: list[tuple[int, bool]] = []
    resolution_calls: list[str] = []

    monkeypatch.setattr(
        ppo_trainer_module.torch,
        "manual_seed",
        lambda seed: torch_seed_calls.append(seed),
    )
    monkeypatch.setattr(
        ppo_trainer_module.torch.mps,
        "manual_seed",
        lambda seed: mps_seed_calls.append(seed),
    )
    monkeypatch.setattr(
        ppo_trainer_module,
        "set_random_seed",
        lambda seed, using_cuda: sb3_seed_calls.append((seed, using_cuda)),
    )

    def resolve_mps(requested: str) -> TorchDeviceResolution:
        resolution_calls.append(requested)
        return TorchDeviceResolution(
            requested_device="mps",
            resolved_device="mps",
            mps_built=True,
            mps_available=True,
        )

    monkeypatch.setattr(ppo_trainer_module, "resolve_torch_device", resolve_mps)

    ppo_trainer_module._seed_everything(11, resolved_device="cpu")
    assert torch_seed_calls == [11]
    assert mps_seed_calls == []
    assert resolution_calls == []

    ppo_trainer_module._seed_everything(13, resolved_device="mps")
    assert torch_seed_calls == [11, 13]
    assert mps_seed_calls == [13]
    assert resolution_calls == ["mps"]
    assert sb3_seed_calls == [(11, False), (13, False)]


def _real_mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_built() and backend.is_available())
    except (AttributeError, RuntimeError):
        return False


@pytest.mark.skipif(not _real_mps_available(), reason="Apple MPS is unavailable")
def test_real_tiny_mps_smoke_training_uses_mps_hardware(rl_splits) -> None:
    splits_dir, _, _, _ = rl_splits

    result = train_single_symbol(
        "MCB",
        config=_tiny_config(device="mps"),
        splits_dir=splits_dir,
        smoke_test=True,
    )

    assert result.succeeded, result.error
    assert result.requested_device == "mps"
    assert result.resolved_device == "mps"
    assert torch_devices_equivalent(result.device, "mps")
    assert result.model is not None
    assert torch_devices_equivalent(result.model.device, "mps")
    assert all(
        torch_devices_equivalent(parameter.device, "mps")
        for parameter in result.model.policy.parameters()
    )


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
    assert missing.source_rl_contract_path is None
    assert missing.source_rl_contract_sha256 is None
    assert missing.source_observation_scaler_path is None
    assert missing.source_observation_scaler_sha256 is None
    assert missing.source_observation_scaler_metadata_path is None
    assert missing.source_observation_scaler_metadata_sha256 is None
    assert missing.observation_features == ()

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
    assert result.training_diagnostics is None
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
    assert result.training_diagnostics is None
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
