"""Deterministic offline tests for the CPU-versus-MPS PPO benchmark."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.evaluation import ppo_evaluator
from reinforcement_learning.model_management import persistence, registry
from reinforcement_learning.training import device_benchmark as benchmark
from reinforcement_learning.training.config import PPOConfig
from reinforcement_learning.training.devices import TorchDeviceResolution
from reinforcement_learning.training.results import PPOTrainingResult


class _InMemoryModel:
    """Minimal marker object proving the benchmark received an in-memory model."""


def _tiny_config(**overrides: object) -> PPOConfig:
    return replace(
        PPOConfig(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        total_timesteps=16,
        **overrides,
    )


def _training_result(
    config: PPOConfig,
    *,
    status: str = "completed",
    error: str | None = None,
    seed: int | None = None,
    ppo_config: dict[str, object] | None = None,
) -> PPOTrainingResult:
    resolved = config.device
    if resolved == "auto":
        resolved = "mps"
    succeeded = status == "completed"
    effective_seed = config.seed if seed is None else seed
    return PPOTrainingResult(
        symbol="MCB",
        algorithm="PPO",
        ppo_config_version=config.config_version,
        ppo_config=config.to_dict() if ppo_config is None else ppo_config,
        environment_version=ENVIRONMENT_VERSION,
        rl_contract_version="rl_partition_v1",
        feature_version="feature_v1",
        source_rl_contract_path="/fixture/rl_contract.json",
        source_rl_contract_sha256="a" * 64,
        source_observation_scaler_path="/fixture/rl_observation_scaler.joblib",
        source_observation_scaler_sha256="b" * 64,
        source_observation_scaler_metadata_path=(
            "/fixture/rl_observation_scaler.json"
        ),
        source_observation_scaler_metadata_sha256="c" * 64,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        seed=effective_seed,
        requested_timesteps=config.total_timesteps,
        actual_timesteps=config.total_timesteps if succeeded else 0,
        training_start="2020-01-02",
        training_end="2024-12-31",
        training_rows=1_200,
        duration_seconds=99.0,
        device=resolved,
        observation_shape=(17,),
        status=status,
        started_at="2026-08-09T00:00:00+00:00",
        completed_at="2026-08-09T00:00:01+00:00",
        message="completed in memory" if succeeded else "failed safely",
        error=error,
        model=_InMemoryModel() if succeeded else None,
        requested_device=config.device,
        resolved_device=resolved,
    )


@pytest.fixture
def benchmark_paths(tmp_path: Path) -> dict[str, Path]:
    split = tmp_path / "splits" / "symbols" / "MCB"
    split.mkdir(parents=True)
    artifacts = {
        "train.csv": b"raw train bytes\n",
        "train_rl.csv": b"scaled observation train bytes\n",
        "rl_contract.json": b'{"artifact_schema_version":"rl_partition_v1"}\n',
        "rl_observation_scaler.joblib": b"fixture scaler bytes",
        "rl_observation_scaler.json": b'{"scaled_features":[]}\n',
    }
    for name, payload in artifacts.items():
        (split / name).write_bytes(payload)

    registry_path = tmp_path / "models" / "model_registry.csv"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(b"model_id,status\n")
    saved_models = tmp_path / "saved_models"
    saved_models.mkdir()
    (saved_models / "sentinel.zip").write_bytes(b"production sentinel")
    return {
        "splits": tmp_path / "splits",
        "split": split,
        "registry": registry_path,
        "models": saved_models,
    }


def _available_resolution(requested_device: str) -> TorchDeviceResolution:
    requested = str(requested_device)
    resolved = "mps" if requested in {"mps", "auto"} else "cpu"
    return TorchDeviceResolution(
        requested_device=requested,
        resolved_device=resolved,
        mps_built=True,
        mps_available=True,
    )


def _fixed_telemetry() -> benchmark.MPSRuntimeInfo:
    return benchmark.MPSRuntimeInfo(
        built=True,
        available=True,
        device_name=None,
        gpu_core_count=None,
        device_count=1,
        current_allocated_bytes=100,
        driver_allocated_bytes=200,
        recommended_max_bytes=1_000,
    )


def _install_successful_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock_values: tuple[float, ...] = (10.0, 14.0, 20.0, 22.0),
) -> list[tuple[str, object]]:
    calls: list[tuple[str, object]] = []
    clocks = iter(clock_values)

    def fake_train(symbol: str, **kwargs: object) -> PPOTrainingResult:
        config = kwargs["config"]
        assert isinstance(config, PPOConfig)
        calls.append(("train", (symbol, config, kwargs)))
        return _training_result(config)

    def fake_clock() -> float:
        value = next(clocks)
        calls.append(("clock", value))
        return value

    def fake_sync(device: str) -> None:
        calls.append(("sync", device))

    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "train_single_symbol", fake_train)
    monkeypatch.setattr(benchmark, "synchronize_torch_device", fake_sync)
    monkeypatch.setattr(benchmark.time, "perf_counter", fake_clock)
    monkeypatch.setattr(benchmark, "collect_mps_runtime_info", _fixed_telemetry)
    return calls


def _run_successful_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
    *,
    clock_values: tuple[float, ...] = (10.0, 14.0, 20.0, 22.0),
) -> tuple[benchmark.DeviceBenchmarkResult, list[tuple[str, object]]]:
    calls = _install_successful_runtime(monkeypatch, clock_values=clock_values)
    result = benchmark.benchmark_cpu_vs_mps(
        "MCB",
        timesteps=16,
        warmup_timesteps=8,
        seed=42,
        config=_tiny_config(),
        splits_dir=benchmark_paths["splits"],
        registry_path=benchmark_paths["registry"],
        saved_models_dir=benchmark_paths["models"],
    )
    return result, calls


def test_benchmark_uses_identical_config_data_seed_and_train_only(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("validation, TEST, persistence, and saves are forbidden")

    monkeypatch.setattr(ppo_evaluator, "evaluate_ppo_validation", forbidden)
    monkeypatch.setattr(persistence, "persist_ppo_candidate", forbidden)
    monkeypatch.setattr(persistence, "persist_developer_smoke_bundle", forbidden)
    monkeypatch.setattr(registry, "append_model_version", forbidden)
    monkeypatch.setattr("stable_baselines3.PPO.save", forbidden)
    registry_before = benchmark_paths["registry"].read_bytes()
    model_before = (benchmark_paths["models"] / "sentinel.zip").read_bytes()

    result, calls = _run_successful_benchmark(monkeypatch, benchmark_paths)

    training_calls = [value for kind, value in calls if kind == "train"]
    assert len(training_calls) == 4
    assert [item[1].device for item in training_calls] == [
        "cpu",
        "cpu",
        "mps",
        "mps",
    ]
    assert [item[1].total_timesteps for item in training_calls] == [8, 16, 8, 16]
    assert {item[1].seed for item in training_calls} == {42}
    assert {item[0] for item in training_calls} == {"MCB"}
    assert {Path(item[2]["splits_dir"]) for item in training_calls} == {
        benchmark_paths["splits"]
    }
    assert [item[2]["smoke_test"] for item in training_calls] == [
        True,
        False,
        True,
        False,
    ]
    cpu_timed = training_calls[1][1].to_dict()
    mps_timed = training_calls[3][1].to_dict()
    assert {
        key: value for key, value in cpu_timed.items() if key != "device"
    } == {key: value for key, value in mps_timed.items() if key != "device"}
    assert result.cpu.training_partition == "train"
    assert result.mps.training_partition == "train"
    assert not result.cpu.validation_accessed
    assert not result.cpu.test_accessed
    assert not result.mps.validation_accessed
    assert not result.mps.test_accessed
    assert not result.test_evaluated
    assert result.registry_unchanged
    assert result.production_models_unchanged
    assert result.training_artifacts_unchanged
    assert benchmark_paths["registry"].read_bytes() == registry_before
    assert (benchmark_paths["models"] / "sentinel.zip").read_bytes() == model_before
    assert sorted(path.name for path in benchmark_paths["models"].iterdir()) == [
        "sentinel.zip"
    ]


def test_warmup_is_excluded_and_outer_timing_is_synchronized(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    result, calls = _run_successful_benchmark(monkeypatch, benchmark_paths)

    assert result.cpu.wall_clock_seconds == pytest.approx(4.0)
    assert result.mps.wall_clock_seconds == pytest.approx(2.0)
    assert result.cpu.timesteps_per_second == pytest.approx(4.0)
    assert result.mps.timesteps_per_second == pytest.approx(8.0)
    assert result.speedup_cpu_over_mps == pytest.approx(2.0)
    assert result.recommended_device == "mps"

    compact: list[str] = []
    for kind, value in calls:
        if kind == "train":
            _, config, _ = value
            compact.append(f"train-{config.device}-{config.total_timesteps}")
        elif kind == "sync":
            compact.append(f"sync-{value}")
        else:
            compact.append(f"clock-{value:g}")
    assert compact == [
        "train-cpu-8",
        "sync-cpu",
        "sync-cpu",
        "clock-10",
        "train-cpu-16",
        "sync-cpu",
        "clock-14",
        "train-mps-8",
        "sync-mps",
        "sync-mps",
        "clock-20",
        "train-mps-16",
        "sync-mps",
        "clock-22",
    ]


def test_benchmark_rejects_silent_mps_fallback_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fallback guard must run before training or resolution")

    monkeypatch.setattr(benchmark, "train_single_symbol", forbidden)
    monkeypatch.setattr(benchmark, "resolve_torch_device", forbidden)
    with pytest.raises(benchmark.DeviceBenchmarkError, match="silently executed"):
        benchmark.benchmark_cpu_vs_mps("MCB", timesteps=512)


def test_protected_artifact_mutation_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "synchronize_torch_device", lambda device: None)
    monkeypatch.setattr(
        benchmark.time,
        "perf_counter",
        iter((1.0, 2.0, 3.0, 4.0)).__next__,
    )
    mutated = False

    def mutating_train(symbol: str, **kwargs: object) -> PPOTrainingResult:
        nonlocal mutated
        config = kwargs["config"]
        assert isinstance(config, PPOConfig)
        if not mutated:
            (benchmark_paths["models"] / "forbidden.zip").write_bytes(b"bad")
            mutated = True
        return _training_result(config)

    monkeypatch.setattr(benchmark, "train_single_symbol", mutating_train)
    with pytest.raises(benchmark.DeviceBenchmarkError, match="changed protected"):
        benchmark.benchmark_cpu_vs_mps(
            "MCB",
            timesteps=16,
            warmup_timesteps=8,
            config=_tiny_config(),
            splits_dir=benchmark_paths["splits"],
            registry_path=benchmark_paths["registry"],
            saved_models_dir=benchmark_paths["models"],
        )


def test_timed_mps_failure_is_visible_and_prevents_speedup(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "synchronize_torch_device", lambda device: None)
    monkeypatch.setattr(
        benchmark.time,
        "perf_counter",
        iter((10.0, 14.0, 20.0, 21.0)).__next__,
    )

    def failing_mps_train(symbol: str, **kwargs: object) -> PPOTrainingResult:
        config = kwargs["config"]
        assert isinstance(config, PPOConfig)
        if config.device == "mps" and not kwargs["smoke_test"]:
            return _training_result(
                config,
                status="failed",
                error="NotImplementedError: unsupported MPS operation",
            )
        return _training_result(config)

    monkeypatch.setattr(benchmark, "train_single_symbol", failing_mps_train)
    monkeypatch.setattr(benchmark, "collect_mps_runtime_info", _fixed_telemetry)
    result = benchmark.benchmark_cpu_vs_mps(
        "MCB",
        timesteps=16,
        warmup_timesteps=8,
        config=_tiny_config(),
        splits_dir=benchmark_paths["splits"],
        registry_path=benchmark_paths["registry"],
        saved_models_dir=benchmark_paths["models"],
    )

    assert result.cpu.succeeded
    assert not result.mps.succeeded
    assert result.mps.status == "failed"
    assert result.mps.error == "NotImplementedError: unsupported MPS operation"
    assert result.mps.failure_phase == "timed"
    assert result.mps.actual_device is None
    assert result.mps.wall_clock_seconds == pytest.approx(1.0)
    assert result.speedup_cpu_over_mps is None
    assert result.recommended_device == "cpu"
    assert not result.succeeded


def test_warmup_failure_does_not_masquerade_as_timed_device_work(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "synchronize_torch_device", lambda device: None)
    monkeypatch.setattr(
        benchmark.time,
        "perf_counter",
        iter((10.0, 14.0)).__next__,
    )

    def fail_mps_warmup(symbol: str, **kwargs: object) -> PPOTrainingResult:
        config = kwargs["config"]
        assert isinstance(config, PPOConfig)
        if config.device == "mps":
            return _training_result(
                config,
                status="failed",
                error="RuntimeError: warm-up failed",
            )
        return _training_result(config)

    monkeypatch.setattr(benchmark, "train_single_symbol", fail_mps_warmup)
    monkeypatch.setattr(benchmark, "collect_mps_runtime_info", _fixed_telemetry)

    result = benchmark.benchmark_cpu_vs_mps(
        "MCB",
        timesteps=16,
        warmup_timesteps=8,
        config=_tiny_config(),
        splits_dir=benchmark_paths["splits"],
        registry_path=benchmark_paths["registry"],
        saved_models_dir=benchmark_paths["models"],
    )

    assert result.cpu.succeeded
    assert not result.mps.succeeded
    assert result.mps.failure_phase == "warmup"
    assert result.mps.requested_timesteps == 16
    assert result.mps.actual_timesteps == 0
    assert result.mps.actual_device is None
    assert result.mps.wall_clock_seconds is None
    assert "Warm-up failed" in str(result.mps.error)


def test_protected_mutation_is_reported_even_when_training_raises(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)

    def mutate_then_raise(*args: object, **kwargs: object) -> PPOTrainingResult:
        (benchmark_paths["models"] / "forbidden.zip").write_bytes(b"bad")
        raise RuntimeError("unexpected trainer exception")

    monkeypatch.setattr(benchmark, "train_single_symbol", mutate_then_raise)

    with pytest.raises(benchmark.DeviceBenchmarkError, match="changed protected"):
        benchmark.benchmark_cpu_vs_mps(
            "MCB",
            timesteps=16,
            warmup_timesteps=8,
            config=_tiny_config(),
            splits_dir=benchmark_paths["splits"],
            registry_path=benchmark_paths["registry"],
            saved_models_dir=benchmark_paths["models"],
        )


def test_cpu_and_mps_result_seed_or_config_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_paths: dict[str, Path],
) -> None:
    """Returned trainer provenance, not only requested arguments, must agree."""
    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "synchronize_torch_device", lambda device: None)
    monkeypatch.setattr(
        benchmark.time,
        "perf_counter",
        iter((1.0, 2.0, 3.0, 4.0)).__next__,
    )

    def drifting_train(symbol: str, **kwargs: object) -> PPOTrainingResult:
        config = kwargs["config"]
        assert isinstance(config, PPOConfig)
        if config.device == "mps" and not kwargs["smoke_test"]:
            changed = config.to_dict()
            changed["gamma"] = 0.5
            return _training_result(config, seed=99, ppo_config=changed)
        return _training_result(config)

    monkeypatch.setattr(benchmark, "train_single_symbol", drifting_train)
    with pytest.raises(benchmark.DeviceBenchmarkError, match="identical training"):
        benchmark.benchmark_cpu_vs_mps(
            "MCB",
            timesteps=16,
            warmup_timesteps=8,
            seed=42,
            config=_tiny_config(),
            splits_dir=benchmark_paths["splits"],
            registry_path=benchmark_paths["registry"],
            saved_models_dir=benchmark_paths["models"],
        )


def test_optional_telemetry_failures_are_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(label: str):
        def fail():
            raise RuntimeError(label)

        return fail

    backend = SimpleNamespace(
        is_built=lambda: True,
        is_available=lambda: True,
        get_name=unavailable("name failed"),
        get_core_count=unavailable("core count failed"),
    )
    mps = SimpleNamespace(
        device_count=unavailable("device count failed"),
        current_allocated_memory=unavailable("allocated failed"),
        driver_allocated_memory=unavailable("driver failed"),
        recommended_max_memory=unavailable("recommended failed"),
    )
    monkeypatch.setattr(benchmark.torch.backends, "mps", backend)
    monkeypatch.setattr(benchmark.torch, "mps", mps)

    info = benchmark.collect_mps_runtime_info()

    assert info.built
    assert info.available
    assert info.device_name is None
    assert info.gpu_core_count is None
    assert info.device_count is None
    assert info.current_allocated_bytes is None
    assert info.driver_allocated_bytes is None
    assert info.recommended_max_bytes is None
    assert len(info.warnings) == 6
    assert all("unavailable: RuntimeError" in item for item in info.warnings)


@pytest.mark.parametrize(
    "argv",
    (
        [],
        ["--symbol", "MCB", "--timesteps", "0"],
        ["--symbol", "MCB", "--seed", "-1"],
    ),
)
def test_cli_parser_bounds_exit_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        benchmark.main(argv)
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "argv",
    (
        ["--symbol", "MCB", "--timesteps", "10241"],
        ["--symbol", "MCB", "--timesteps", "513"],
        ["--symbol", "MCB", "--warmup-timesteps", "1025"],
        ["--symbol", "MCB", "--warmup-timesteps", "513"],
    ),
)
def test_cli_runtime_bounds_exit_two_before_training(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid CLI bounds must fail before training")

    monkeypatch.setattr(benchmark, "resolve_torch_device", _available_resolution)
    monkeypatch.setattr(benchmark, "train_single_symbol", forbidden)
    with pytest.raises(SystemExit) as raised:
        benchmark.main(argv)
    assert raised.value.code == 2
