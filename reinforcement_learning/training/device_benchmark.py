"""Safe, single-symbol CPU-versus-Apple-MPS PPO developer benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import time
import warnings

import torch

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    PROCESSED_SPLITS_DIR,
    SAVED_MODELS_DIR,
)
from feature_engineering.storage import safe_path_component
from reinforcement_learning.data_contract import (
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
)
from reinforcement_learning.integrity import sha256_file

from .config import PPOConfig
from .devices import (
    TorchDeviceError,
    resolve_torch_device,
    synchronize_torch_device,
)
from .ppo_trainer import train_single_symbol
from .results import PPOTrainingResult


LOGGER = logging.getLogger(__name__)
DEFAULT_BENCHMARK_TIMESTEPS = 5_120
DEFAULT_WARMUP_TIMESTEPS = 512
MATERIAL_MPS_SPEEDUP = 1.10
MAX_BENCHMARK_TIMESTEPS = 10_240
MAX_WARMUP_TIMESTEPS = 1_024


class DeviceBenchmarkError(RuntimeError):
    """Raised when a CPU/MPS comparison cannot be performed safely."""


@dataclass(frozen=True)
class MPSRuntimeInfo:
    """Optional Apple-MPS telemetry; correctness never depends on it."""

    built: bool
    available: bool
    device_name: str | None
    gpu_core_count: int | None
    device_count: int | None
    current_allocated_bytes: int | None
    driver_allocated_bytes: int | None
    recommended_max_bytes: int | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceBenchmarkRun:
    """One timed PPO run after a same-device warm-up."""

    requested_device: str
    resolved_device: str
    actual_device: str | None
    symbol: str
    training_partition: str
    train_rows: int
    train_start: str | None
    train_end: str | None
    requested_timesteps: int
    actual_timesteps: int
    wall_clock_seconds: float | None
    timesteps_per_second: float | None
    observation_shape: tuple[int, ...] | None
    status: str
    error: str | None
    warnings: tuple[str, ...]
    model_valid: bool
    failure_phase: str | None = None
    validation_accessed: bool = False
    test_accessed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.model_valid

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceBenchmarkResult:
    """Complete CPU/MPS benchmark and its conservative device recommendation."""

    symbol: str
    seed: int
    warmup_timesteps: int
    benchmark_timesteps: int
    ppo_config_version: str
    environment_version: str
    rl_contract_version: str
    feature_version: str
    cpu: DeviceBenchmarkRun
    mps: DeviceBenchmarkRun
    speedup_cpu_over_mps: float | None
    recommended_device: str
    recommendation_reason: str
    mps_info: MPSRuntimeInfo
    registry_unchanged: bool
    production_models_unchanged: bool
    training_artifacts_unchanged: bool
    train_rl_sha256: str
    test_evaluated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.cpu.succeeded
            and self.mps.succeeded
            and self.registry_unchanged
            and self.production_models_unchanged
            and self.training_artifacts_unchanged
            and not self.test_evaluated
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["succeeded"] = self.succeeded
        return payload


def _optional_call(label: str, callable_object, warnings_out: list[str]):
    try:
        return callable_object()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        warnings_out.append(f"{label} unavailable: {type(exc).__name__}: {exc}")
        return None


def collect_mps_runtime_info() -> MPSRuntimeInfo:
    """Collect guarded telemetry without making the benchmark depend on it."""
    backend = getattr(torch.backends, "mps", None)
    telemetry_warnings: list[str] = []
    built_value = _optional_call(
        "is_built",
        getattr(backend, "is_built", lambda: False),
        telemetry_warnings,
    )
    available_value = _optional_call(
        "is_available",
        getattr(backend, "is_available", lambda: False),
        telemetry_warnings,
    )
    built = bool(built_value)
    available = bool(available_value)
    if not available:
        return MPSRuntimeInfo(
            built=built,
            available=False,
            device_name=None,
            gpu_core_count=None,
            device_count=0,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            warnings=tuple(telemetry_warnings),
        )

    def optional(owner, name: str):
        member = getattr(owner, name, None)
        if not callable(member):
            telemetry_warnings.append(f"{name} is not exposed by this PyTorch build")
            return None
        return _optional_call(name, member, telemetry_warnings)

    return MPSRuntimeInfo(
        built=built,
        available=available,
        device_name=optional(backend, "get_name"),
        gpu_core_count=optional(backend, "get_core_count"),
        device_count=optional(torch.mps, "device_count"),
        current_allocated_bytes=optional(torch.mps, "current_allocated_memory"),
        driver_allocated_bytes=optional(torch.mps, "driver_allocated_memory"),
        recommended_max_bytes=optional(torch.mps, "recommended_max_memory"),
        warnings=tuple(telemetry_warnings),
    )


def _registry_snapshot(path: Path) -> bytes | None:
    destination = Path(path)
    return destination.read_bytes() if destination.is_file() else None


def _file_tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    directory = Path(root)
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): (path.stat().st_size, sha256_file(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _training_artifact_snapshot(
    splits_dir: Path,
    symbol: str,
) -> dict[str, tuple[int, str]]:
    directory = Path(splits_dir) / "symbols" / safe_path_component(symbol)
    names = (
        "train.csv",
        "train_rl.csv",
        RL_CONTRACT_FILENAME,
        RL_OBSERVATION_SCALER_FILENAME,
        RL_OBSERVATION_SCALER_FILENAME.replace(".joblib", ".json"),
    )
    snapshot: dict[str, tuple[int, str]] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise DeviceBenchmarkError(f"Required training artifact is missing: {path}")
        snapshot[name] = (path.stat().st_size, sha256_file(path))
    return snapshot


def _unique_warning_messages(captured: Sequence[warnings.WarningMessage]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item.message) for item in captured))


def _failed_run(
    *,
    requested_device: str,
    resolved_device: str,
    symbol: str,
    requested_timesteps: int,
    result: PPOTrainingResult | None,
    error: str,
    warning_messages: tuple[str, ...],
    failure_phase: str,
    wall_clock_seconds: float | None = None,
) -> DeviceBenchmarkRun:
    timed_failure = failure_phase == "timed"
    actual_timesteps = (
        result.actual_timesteps if timed_failure and result is not None else 0
    )
    return DeviceBenchmarkRun(
        requested_device=requested_device,
        resolved_device=resolved_device,
        # Failed trainer results cannot expose a model, so a configured or resolved
        # device string is not proof that SB3 actually constructed a policy there.
        actual_device=None,
        symbol=symbol,
        training_partition="train",
        train_rows=result.training_rows if result is not None else 0,
        train_start=result.training_start if result is not None else None,
        train_end=result.training_end if result is not None else None,
        requested_timesteps=requested_timesteps,
        actual_timesteps=actual_timesteps,
        wall_clock_seconds=wall_clock_seconds,
        timesteps_per_second=(
            actual_timesteps / wall_clock_seconds
            if actual_timesteps > 0
            and wall_clock_seconds is not None
            and wall_clock_seconds > 0
            else None
        ),
        observation_shape=result.observation_shape if result is not None else None,
        status="failed",
        error=error,
        warnings=warning_messages,
        model_valid=False,
        failure_phase=failure_phase,
    )


def _run_device_benchmark(
    *,
    symbol: str,
    requested_device: str,
    base_config: PPOConfig,
    warmup_timesteps: int,
    benchmark_timesteps: int,
    splits_dir: Path,
) -> tuple[DeviceBenchmarkRun, PPOTrainingResult | None]:
    resolution = resolve_torch_device(requested_device)
    warmup_config = replace(
        base_config,
        device=requested_device,
        total_timesteps=warmup_timesteps,
    )
    timed_config = replace(
        base_config,
        device=requested_device,
        total_timesteps=benchmark_timesteps,
    )
    captured_messages: list[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warmup = train_single_symbol(
            symbol,
            config=warmup_config,
            splits_dir=Path(splits_dir),
            smoke_test=warmup_timesteps <= 1_024,
        )
    captured_messages.extend(_unique_warning_messages(captured))
    if not warmup.succeeded or warmup.model is None:
        return (
            _failed_run(
                requested_device=requested_device,
                resolved_device=resolution.resolved_device,
                symbol=symbol,
                requested_timesteps=benchmark_timesteps,
                result=warmup,
                error=f"Warm-up failed: {warmup.error or warmup.message}",
                warning_messages=tuple(captured_messages),
                failure_phase="warmup",
            ),
            None,
        )
    if (
        warmup.requested_device != requested_device
        or warmup.resolved_device != resolution.resolved_device
    ):
        raise DeviceBenchmarkError("Warm-up device resolution changed unexpectedly")
    synchronize_torch_device(resolution.resolved_device)
    del warmup

    synchronize_torch_device(resolution.resolved_device)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            timed = train_single_symbol(
                symbol,
                config=timed_config,
                splits_dir=Path(splits_dir),
                smoke_test=False,
            )
        synchronize_torch_device(resolution.resolved_device)
    except TorchDeviceError as exc:
        duration = time.perf_counter() - started
        return (
            _failed_run(
                requested_device=requested_device,
                resolved_device=resolution.resolved_device,
                symbol=symbol,
                requested_timesteps=benchmark_timesteps,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
                warning_messages=tuple(captured_messages),
                failure_phase="timed",
                wall_clock_seconds=duration,
            ),
            None,
        )
    duration = time.perf_counter() - started
    captured_messages.extend(_unique_warning_messages(captured))
    warning_messages = tuple(dict.fromkeys(captured_messages))
    if not timed.succeeded or timed.model is None:
        return (
            _failed_run(
                requested_device=requested_device,
                resolved_device=resolution.resolved_device,
                symbol=symbol,
                requested_timesteps=benchmark_timesteps,
                result=timed,
                error=timed.error or timed.message,
                warning_messages=warning_messages,
                failure_phase="timed",
                wall_clock_seconds=duration,
            ),
            None,
        )
    if (
        timed.requested_device != requested_device
        or timed.resolved_device != resolution.resolved_device
    ):
        raise DeviceBenchmarkError("Timed-run device resolution changed unexpectedly")
    throughput = timed.actual_timesteps / duration if duration > 0 else None
    return (
        DeviceBenchmarkRun(
            requested_device=requested_device,
            resolved_device=resolution.resolved_device,
            actual_device=timed.device,
            symbol=timed.symbol,
            training_partition="train",
            train_rows=timed.training_rows,
            train_start=timed.training_start,
            train_end=timed.training_end,
            requested_timesteps=benchmark_timesteps,
            actual_timesteps=timed.actual_timesteps,
            wall_clock_seconds=duration,
            timesteps_per_second=throughput,
            observation_shape=timed.observation_shape,
            status=timed.status,
            error=None,
            warnings=warning_messages,
            model_valid=True,
        ),
        timed,
    )


def _research_configuration(config: PPOConfig) -> Mapping[str, object]:
    payload = config.to_dict()
    return {
        key: value
        for key, value in payload.items()
        if key != "device"
    }


def _training_provenance(result: PPOTrainingResult) -> dict[str, object]:
    effective_config = dict(result.ppo_config)
    effective_config.pop("device", None)
    return {
        "symbol": result.symbol,
        "seed": result.seed,
        "requested_timesteps": result.requested_timesteps,
        "effective_ppo_config_except_device": effective_config,
        "training_rows": result.training_rows,
        "training_start": result.training_start,
        "training_end": result.training_end,
        "observation_shape": result.observation_shape,
        "observation_features": result.observation_features,
        "environment_version": result.environment_version,
        "rl_contract_version": result.rl_contract_version,
        "feature_version": result.feature_version,
        "rl_contract_sha256": result.source_rl_contract_sha256,
        "observation_scaler_sha256": result.source_observation_scaler_sha256,
        "observation_scaler_metadata_sha256": (
            result.source_observation_scaler_metadata_sha256
        ),
    }


def benchmark_cpu_vs_mps(
    symbol: str,
    *,
    timesteps: int = DEFAULT_BENCHMARK_TIMESTEPS,
    warmup_timesteps: int = DEFAULT_WARMUP_TIMESTEPS,
    seed: int = 42,
    config: PPOConfig | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
) -> DeviceBenchmarkResult:
    """Benchmark one explicit symbol without validation, TEST, or persistence."""
    symbol_text = str(symbol).strip()
    if not symbol_text:
        raise DeviceBenchmarkError("symbol is required")
    for name, value in (
        ("timesteps", timesteps),
        ("warmup_timesteps", warmup_timesteps),
        ("seed", seed),
    ):
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise DeviceBenchmarkError(f"{name} must be an integer of at least {minimum}")
    base = (config or PPOConfig()).with_runtime_overrides(
        seed=seed,
        total_timesteps=timesteps,
        device="cpu",
    )
    if timesteps > MAX_BENCHMARK_TIMESTEPS:
        raise DeviceBenchmarkError(
            f"timesteps cannot exceed {MAX_BENCHMARK_TIMESTEPS} for this developer benchmark"
        )
    if warmup_timesteps > MAX_WARMUP_TIMESTEPS:
        raise DeviceBenchmarkError(
            f"warmup_timesteps cannot exceed {MAX_WARMUP_TIMESTEPS}"
        )
    if timesteps < base.n_steps or timesteps % base.n_steps:
        raise DeviceBenchmarkError(
            "timesteps must be at least one PPO rollout and divisible by n_steps"
        )
    if warmup_timesteps < base.n_steps or warmup_timesteps % base.n_steps:
        raise DeviceBenchmarkError(
            "warmup_timesteps must be at least one PPO rollout and divisible by n_steps"
        )
    fallback_setting = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    if fallback_setting in {"1", "true", "yes", "on"}:
        raise DeviceBenchmarkError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; benchmark aborted because "
            "unsupported MPS operations could be silently executed on CPU"
        )
    # Fail before doing the CPU work when an actual two-device comparison is
    # impossible. Explicit MPS never degrades into a CPU-vs-CPU benchmark.
    resolve_torch_device("mps")
    cpu_research = _research_configuration(replace(base, device="cpu"))
    mps_research = _research_configuration(replace(base, device="mps"))
    if cpu_research != mps_research:
        raise DeviceBenchmarkError("CPU and MPS research configurations differ")

    registry_before = _registry_snapshot(Path(registry_path))
    models_before = _file_tree_snapshot(Path(saved_models_dir))
    training_before = _training_artifact_snapshot(Path(splits_dir), symbol_text)
    cpu_run: DeviceBenchmarkRun | None = None
    mps_run: DeviceBenchmarkRun | None = None
    cpu_training: PPOTrainingResult | None = None
    mps_training: PPOTrainingResult | None = None
    cpu_provenance: dict[str, object] | None = None
    mps_provenance: dict[str, object] | None = None
    pending_error: BaseException | None = None
    try:
        cpu_run, cpu_training = _run_device_benchmark(
            symbol=symbol_text,
            requested_device="cpu",
            base_config=base,
            warmup_timesteps=warmup_timesteps,
            benchmark_timesteps=timesteps,
            splits_dir=Path(splits_dir),
        )
        if cpu_training is not None:
            cpu_provenance = _training_provenance(cpu_training)
        training_after_cpu = _training_artifact_snapshot(
            Path(splits_dir), symbol_text
        )
        if training_after_cpu != training_before:
            raise DeviceBenchmarkError(
                "CPU benchmark changed protected training artifacts"
            )
        # Do not retain the CPU policy in unified memory while timing MPS.
        cpu_training = None
        mps_run, mps_training = _run_device_benchmark(
            symbol=symbol_text,
            requested_device="mps",
            base_config=base,
            warmup_timesteps=warmup_timesteps,
            benchmark_timesteps=timesteps,
            splits_dir=Path(splits_dir),
        )
        if mps_training is not None:
            mps_provenance = _training_provenance(mps_training)
    except BaseException as exc:
        pending_error = exc
    finally:
        registry_unchanged = _registry_snapshot(Path(registry_path)) == registry_before
        models_unchanged = (
            _file_tree_snapshot(Path(saved_models_dir)) == models_before
        )
        training_unchanged = (
            _training_artifact_snapshot(Path(splits_dir), symbol_text)
            == training_before
        )
    if not registry_unchanged or not models_unchanged or not training_unchanged:
        raise DeviceBenchmarkError(
            "Benchmark changed protected registry, model, or training artifacts"
        ) from pending_error
    if pending_error is not None:
        raise pending_error.with_traceback(pending_error.__traceback__)
    if cpu_run is None or mps_run is None:
        raise DeviceBenchmarkError("Device benchmark ended without both run results")
    if cpu_run.succeeded and mps_run.succeeded and cpu_provenance != mps_provenance:
        raise DeviceBenchmarkError(
            "CPU and MPS runs did not use identical training data and contracts"
        )

    speedup = None
    if (
        cpu_run.succeeded
        and mps_run.succeeded
        and cpu_run.wall_clock_seconds is not None
        and mps_run.wall_clock_seconds is not None
        and mps_run.wall_clock_seconds > 0
    ):
        speedup = cpu_run.wall_clock_seconds / mps_run.wall_clock_seconds
    if speedup is not None and speedup >= MATERIAL_MPS_SPEEDUP:
        recommended = "mps"
        reason = (
            f"MPS completed stably with a measured {speedup:.3f}x speedup, "
            f"meeting the {MATERIAL_MPS_SPEEDUP:.2f}x materiality threshold."
        )
    elif speedup is not None:
        recommended = "cpu"
        if speedup < 1:
            reason = f"CPU was faster; measured CPU/MPS speedup was {speedup:.3f}x."
        else:
            reason = (
                f"MPS speedup was only {speedup:.3f}x, below the "
                f"{MATERIAL_MPS_SPEEDUP:.2f}x threshold; retain conservative CPU default."
            )
    else:
        recommended = "cpu"
        reason = "One or both timed device runs did not complete; retain conservative CPU default."

    provenance = mps_training
    if provenance is None and cpu_provenance is None:
        environment_version = ""
        rl_contract_version = ""
        feature_version = ""
    elif provenance is not None:
        environment_version = provenance.environment_version
        rl_contract_version = provenance.rl_contract_version
        feature_version = provenance.feature_version
    else:
        environment_version = str(cpu_provenance["environment_version"])
        rl_contract_version = str(cpu_provenance["rl_contract_version"])
        feature_version = str(cpu_provenance["feature_version"])
    return DeviceBenchmarkResult(
        symbol=symbol_text,
        seed=seed,
        warmup_timesteps=warmup_timesteps,
        benchmark_timesteps=timesteps,
        ppo_config_version=base.config_version,
        environment_version=environment_version,
        rl_contract_version=rl_contract_version,
        feature_version=feature_version,
        cpu=cpu_run,
        mps=mps_run,
        speedup_cpu_over_mps=speedup,
        recommended_device=recommended,
        recommendation_reason=reason,
        mps_info=collect_mps_runtime_info(),
        registry_unchanged=registry_unchanged,
        production_models_unchanged=models_unchanged,
        training_artifacts_unchanged=training_unchanged,
        train_rl_sha256=training_before["train_rl.csv"][1],
        test_evaluated=False,
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _benchmark_timesteps(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed < 512 or parsed > MAX_BENCHMARK_TIMESTEPS or parsed % 512:
        raise argparse.ArgumentTypeError(
            "timesteps must be a multiple of 512 from 512 through 10240"
        )
    return parsed


def _warmup_timesteps(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed > MAX_WARMUP_TIMESTEPS or parsed % 512:
        raise argparse.ArgumentTypeError(
            "warmup timesteps must be 512 or 1024"
        )
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one PSX PPO symbol on CPU and Apple MPS in memory only; "
            "no validation, TEST, registry, or model persistence is used."
        )
    )
    parser.add_argument("--symbol", required=True, help="One RL-ready PSX symbol")
    parser.add_argument(
        "--timesteps",
        type=_benchmark_timesteps,
        default=DEFAULT_BENCHMARK_TIMESTEPS,
    )
    parser.add_argument(
        "--warmup-timesteps",
        type=_warmup_timesteps,
        default=DEFAULT_WARMUP_TIMESTEPS,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.warning(
        "Developer runtime benchmark only: no profitability inference, persistence, "
        "registry write, or model promotion."
    )
    try:
        result = benchmark_cpu_vs_mps(
            args.symbol,
            timesteps=args.timesteps,
            warmup_timesteps=args.warmup_timesteps,
            seed=args.seed,
        )
    except (DeviceBenchmarkError, TorchDeviceError) as exc:
        LOGGER.error("device_benchmark_failed error=%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
