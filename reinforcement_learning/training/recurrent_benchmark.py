"""Isolated CPU/MPS benchmark for the single-symbol recurrent baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

from data_pipeline.src.config import PROCESSED_SPLITS_DIR, PROJECT_ROOT

from .devices import resolve_torch_device, synchronize_torch_device
from .recurrent_config import RecurrentPPOConfig
from .recurrent_results import (
    RecurrentDeviceBenchmarkResult,
    RecurrentDeviceRun,
)
from .recurrent_trainer import train_recurrent_single_symbol


DEFAULT_RECURRENT_BENCHMARK_TIMESTEPS = 5_120
DEFAULT_RECURRENT_WARMUP_TIMESTEPS = 512
BENCHMARK_MARKER = "RECURRENT_BENCHMARK_RESULT="
MATERIAL_SPEEDUP = 1.10


class RecurrentBenchmarkError(RuntimeError):
    """Raised when benchmark inputs or subprocess results are unsafe."""


def _worker_payload(
    *,
    symbol: str,
    device: str,
    seed: int,
    warmup_timesteps: int,
    timesteps: int,
    splits_dir: Path,
) -> dict[str, object]:
    warning_messages: list[str] = []
    try:
        resolution = resolve_torch_device(device)
        warmup_config = RecurrentPPOConfig(
            seed=seed,
            total_timesteps=warmup_timesteps,
            device=device,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warmup = train_recurrent_single_symbol(
                symbol,
                config=warmup_config,
                splits_dir=splits_dir,
                smoke_test=warmup_timesteps <= 1_024,
            )
        warning_messages.extend(str(item.message) for item in captured)
        if not warmup.succeeded:
            raise RecurrentBenchmarkError(warmup.error or warmup.message)
        del warmup
        config = RecurrentPPOConfig(
            seed=seed,
            total_timesteps=timesteps,
            device=device,
        )
        synchronize_torch_device(resolution.resolved_device)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = train_recurrent_single_symbol(
                symbol,
                config=config,
                splits_dir=splits_dir,
                smoke_test=False,
            )
        synchronize_torch_device(resolution.resolved_device)
        duration = time.perf_counter() - started
        warning_messages.extend(str(item.message) for item in captured)
        if not result.succeeded:
            raise RecurrentBenchmarkError(result.error or result.message)
        comparable_config = dict(result.config)
        comparable_config.pop("device", None)
        return {
            "device": device,
            "status": "completed",
            "actual_device": result.device,
            "requested_timesteps": timesteps,
            "actual_timesteps": result.actual_timesteps,
            "duration_seconds": duration,
            "timesteps_per_second": result.actual_timesteps / duration,
            "parameter_count": result.parameter_count,
            "warning": "; ".join(dict.fromkeys(warning_messages)) or None,
            "error": None,
            "provenance": {
                "symbol": result.symbol,
                "seed": result.seed,
                "config_except_device": comparable_config,
                "training_rows": result.training_rows,
                "training_start": result.training_start,
                "training_end": result.training_end,
                "recurrent_contract_version": result.recurrent_contract_version,
                "recurrent_contract_sha256": result.source_recurrent_contract_sha256,
                "observation_features": list(result.observation_features),
            },
        }
    except BaseException as exc:
        return {
            "device": device,
            "status": "failed",
            "actual_device": None,
            "requested_timesteps": timesteps,
            "actual_timesteps": 0,
            "duration_seconds": None,
            "timesteps_per_second": None,
            "parameter_count": 0,
            "warning": "; ".join(dict.fromkeys(warning_messages)) or None,
            "error": f"{type(exc).__name__}: {exc}",
            "provenance": {},
        }


def _run_isolated_worker(
    *,
    symbol: str,
    device: str,
    seed: int,
    warmup_timesteps: int,
    timesteps: int,
    splits_dir: Path,
    timeout_seconds: int,
) -> RecurrentDeviceRun:
    command = [
        sys.executable,
        "-m",
        "reinforcement_learning.training.recurrent_benchmark",
        "--worker",
        "--symbol",
        symbol,
        "--device",
        device,
        "--seed",
        str(seed),
        "--warmup-timesteps",
        str(warmup_timesteps),
        "--timesteps",
        str(timesteps),
        "--splits-dir",
        str(Path(splits_dir).resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RecurrentDeviceRun(
            device=device,
            status="failed",
            actual_device=None,
            requested_timesteps=timesteps,
            actual_timesteps=0,
            duration_seconds=None,
            timesteps_per_second=None,
            parameter_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    marker_lines = [
        line[len(BENCHMARK_MARKER) :]
        for line in completed.stdout.splitlines()
        if line.startswith(BENCHMARK_MARKER)
    ]
    if completed.returncode != 0 or len(marker_lines) != 1:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return RecurrentDeviceRun(
            device=device,
            status="failed",
            actual_device=None,
            requested_timesteps=timesteps,
            actual_timesteps=0,
            duration_seconds=None,
            timesteps_per_second=None,
            parameter_count=0,
            error=(
                f"isolated {device} worker exited {completed.returncode}: "
                f"{detail[-1000:]}"
            ),
        )
    try:
        payload = json.loads(marker_lines[0])
        return RecurrentDeviceRun(**payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RecurrentBenchmarkError(
            f"isolated {device} worker returned invalid JSON: {exc}"
        ) from exc


def benchmark_recurrent_cpu_vs_mps(
    symbol: str,
    *,
    timesteps: int = DEFAULT_RECURRENT_BENCHMARK_TIMESTEPS,
    warmup_timesteps: int = DEFAULT_RECURRENT_WARMUP_TIMESTEPS,
    seed: int = 42,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    timeout_seconds: int = 1_800,
) -> RecurrentDeviceBenchmarkResult:
    """Run CPU and MPS in separate child processes, preserving either result."""

    symbol_text = str(symbol).strip()
    if not symbol_text:
        raise RecurrentBenchmarkError("symbol is required")
    for label, value in (
        ("timesteps", timesteps),
        ("warmup_timesteps", warmup_timesteps),
        ("seed", seed),
        ("timeout_seconds", timeout_seconds),
    ):
        minimum = 0 if label == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RecurrentBenchmarkError(
                f"{label} must be an integer of at least {minimum}"
            )
    if timesteps % 512 or warmup_timesteps % 512:
        raise RecurrentBenchmarkError(
            "benchmark and warm-up timesteps must be multiples of 512"
        )
    cpu = _run_isolated_worker(
        symbol=symbol_text,
        device="cpu",
        seed=seed,
        warmup_timesteps=warmup_timesteps,
        timesteps=timesteps,
        splits_dir=Path(splits_dir),
        timeout_seconds=timeout_seconds,
    )
    mps = _run_isolated_worker(
        symbol=symbol_text,
        device="mps",
        seed=seed,
        warmup_timesteps=warmup_timesteps,
        timesteps=timesteps,
        splits_dir=Path(splits_dir),
        timeout_seconds=timeout_seconds,
    )
    if cpu.succeeded and mps.succeeded and cpu.provenance != mps.provenance:
        raise RecurrentBenchmarkError("CPU and MPS benchmark provenance differs")
    speedup = None
    if (
        cpu.succeeded
        and mps.succeeded
        and cpu.duration_seconds
        and mps.duration_seconds
    ):
        speedup = cpu.duration_seconds / mps.duration_seconds
    if speedup is not None and speedup >= MATERIAL_SPEEDUP:
        recommended = "mps"
        reason = f"MPS was {speedup:.3f}x faster and completed stably."
    elif speedup is not None:
        recommended = "cpu"
        reason = (
            f"Measured CPU/MPS speedup was {speedup:.3f}x; MPS did not meet "
            f"the {MATERIAL_SPEEDUP:.2f}x material threshold."
        )
    else:
        recommended = "cpu"
        reason = "MPS did not complete stably; the successful CPU result is retained."
    return RecurrentDeviceBenchmarkResult(
        symbol=symbol_text,
        seed=seed,
        warmup_timesteps=warmup_timesteps,
        benchmark_timesteps=timesteps,
        cpu=cpu,
        mps=mps,
        speedup_cpu_over_mps=speedup,
        recommended_device=recommended,
        recommendation_reason=reason,
        isolated_subprocesses=True,
        test_evaluated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-timesteps", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=5_120)
    parser.add_argument("--splits-dir", type=Path, default=PROCESSED_SPLITS_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.device is None:
            raise SystemExit("--device is required for --worker")
        payload = _worker_payload(
            symbol=args.symbol,
            device=args.device,
            seed=args.seed,
            warmup_timesteps=args.warmup_timesteps,
            timesteps=args.timesteps,
            splits_dir=args.splits_dir,
        )
        print(BENCHMARK_MARKER + json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "completed" else 1
    result = benchmark_recurrent_cpu_vs_mps(
        args.symbol,
        timesteps=args.timesteps,
        warmup_timesteps=args.warmup_timesteps,
        seed=args.seed,
        splits_dir=args.splits_dir,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.cpu.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
