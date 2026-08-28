"""Fail-closed CUDA benchmark contract for recurrent PPO.

CUDA work is always explicit and runs in isolated subprocesses.  Merely
building or printing a contract never trains a model.  CPU fallback is not a
valid outcome and is rejected both before and after training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Mapping, Sequence

import torch

from data_pipeline.src.config import PROCESSED_SPLITS_DIR, PROJECT_ROOT
from reinforcement_learning.environments.config import ENVIRONMENT_VERSION

from .devices import (
    TorchDeviceError,
    resolve_torch_device,
    synchronize_torch_device,
    torch_devices_equivalent,
)
from .job_state import canonical_hash
from .recurrent_config import RecurrentPPOConfig
from .recurrent_trainer import RECURRENT_TRAINER_VERSION, train_recurrent_single_symbol


CUDA_BENCHMARK_CONTRACT_VERSION = "recurrent_cuda_benchmark_v1"
CUDA_CONCURRENCY_CONTRACT_VERSION = "recurrent_cuda_concurrency_benchmark_v1"
CUDA_BENCHMARK_SYMBOL = "PIAHCLA"
CUDA_BENCHMARK_SYMBOL_RULE = "milestone_7c3a_train_quality_coverage_medoid_v1"
CUDA_BENCHMARK_BUDGETS = (50_000, 100_000, 250_000)
CUDA_CONCURRENCY_WORKERS = (1, 2, 4)
CUDA_BENCHMARK_MARKER = "CUDA_RECURRENT_BENCHMARK_RESULT="


class CudaBenchmarkError(RuntimeError):
    """Raised if CUDA availability, execution, or reporting is unsafe."""


@dataclass(frozen=True)
class CudaBenchmarkContract:
    contract_version: str = CUDA_BENCHMARK_CONTRACT_VERSION
    symbol: str = CUDA_BENCHMARK_SYMBOL
    representative_selection_rule: str = CUDA_BENCHMARK_SYMBOL_RULE
    requested_timesteps: tuple[int, ...] = CUDA_BENCHMARK_BUDGETS
    seed: int = 42
    requested_device: str = "cuda"
    warmup_timesteps: int = 512
    worker_candidates: tuple[int, ...] = CUDA_CONCURRENCY_WORKERS
    test_partition_loaded: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != CUDA_BENCHMARK_CONTRACT_VERSION:
            raise ValueError("CUDA benchmark contract version is incompatible")
        if not self.symbol.strip() or self.requested_device != "cuda":
            raise ValueError("CUDA benchmark requires a symbol and explicit cuda")
        if self.seed < 0 or self.warmup_timesteps < 1:
            raise ValueError("CUDA benchmark seed/warm-up is invalid")
        if not self.requested_timesteps or any(value < 1 for value in self.requested_timesteps):
            raise ValueError("CUDA benchmark budgets must be positive")
        if tuple(self.worker_candidates) != CUDA_CONCURRENCY_WORKERS:
            raise ValueError("CUDA concurrency candidates must be 1, 2, 4")
        if self.test_partition_loaded:
            raise ValueError("TEST cannot enter CUDA benchmarking")

    def to_dict(self) -> dict[str, object]:
        config = RecurrentPPOConfig(
            seed=self.seed,
            total_timesteps=self.requested_timesteps[0],
            device="cuda",
        )
        return {
            **asdict(self),
            "requested_timesteps": list(self.requested_timesteps),
            "worker_candidates": list(self.worker_candidates),
            "trainer_version": RECURRENT_TRAINER_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "ppo_lstm_config_except_budget": {
                key: value
                for key, value in config.to_dict().items()
                if key != "total_timesteps"
            },
            "concurrency_contract_version": CUDA_CONCURRENCY_CONTRACT_VERSION,
            "no_cpu_fallback": True,
            "validation_evaluated": False,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())


def cuda_hardware_preflight() -> dict[str, object]:
    """Return CUDA telemetry or fail closed; never resolves to CPU."""

    try:
        resolution = resolve_torch_device("cuda")
    except TorchDeviceError as exc:
        raise CudaBenchmarkError(str(exc)) from exc
    if not torch_devices_equivalent(resolution.resolved_device, "cuda"):
        raise CudaBenchmarkError("explicit CUDA resolution did not resolve to CUDA")
    try:
        properties = torch.cuda.get_device_properties(0)
        memory = int(properties.total_memory)
        capability = list(torch.cuda.get_device_capability(0))
    except (AttributeError, RuntimeError) as exc:
        raise CudaBenchmarkError(f"CUDA device telemetry failed: {exc}") from exc
    return {
        "requested_device": "cuda",
        "effective_device": resolution.resolved_device,
        "gpu_model": resolution.device_name,
        "gpu_count": resolution.cuda_device_count,
        "gpu_total_memory_bytes": memory,
        "compute_capability": capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def benchmark_command(
    *,
    contract: CudaBenchmarkContract,
    output_json: Path,
    workers: int = 1,
) -> tuple[str, ...]:
    """Build an exact command; it is never executed by this helper."""

    if workers not in contract.worker_candidates:
        raise ValueError("workers must be one of the predeclared candidates")
    return (
        sys.executable,
        "-m",
        "reinforcement_learning.training.cuda_benchmark",
        "--run",
        "--device",
        "cuda",
        "--symbol",
        contract.symbol,
        "--seed",
        str(contract.seed),
        "--warmup-timesteps",
        str(contract.warmup_timesteps),
        "--budgets",
        *(str(value) for value in contract.requested_timesteps),
        "--workers",
        str(workers),
        "--output-json",
        str(Path(output_json)),
    )


def _peak_cpu_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _worker_payload(
    *, symbol: str, seed: int, warmup_timesteps: int, timesteps: int, splits_dir: Path
) -> dict[str, object]:
    hardware = cuda_hardware_preflight()
    torch.cuda.reset_peak_memory_stats(0)
    warmup = train_recurrent_single_symbol(
        symbol,
        config=RecurrentPPOConfig(seed=seed, total_timesteps=warmup_timesteps, device="cuda"),
        splits_dir=splits_dir,
        smoke_test=warmup_timesteps <= 1_024,
    )
    if not warmup.succeeded or not torch_devices_equivalent(warmup.device, "cuda"):
        raise CudaBenchmarkError("CUDA warm-up failed or reported a non-CUDA device")
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    synchronize_torch_device("cuda")
    started = time.perf_counter()
    result = train_recurrent_single_symbol(
        symbol,
        config=RecurrentPPOConfig(seed=seed, total_timesteps=timesteps, device="cuda"),
        splits_dir=splits_dir,
        smoke_test=False,
    )
    synchronize_torch_device("cuda")
    duration = time.perf_counter() - started
    if not result.succeeded or not torch_devices_equivalent(result.device, "cuda"):
        raise CudaBenchmarkError("CUDA benchmark failed or reported a non-CUDA device")
    return {
        "contract_version": CUDA_BENCHMARK_CONTRACT_VERSION,
        "symbol": symbol,
        "seed": seed,
        "status": "completed",
        "requested_device": "cuda",
        "effective_device": result.device,
        "requested_timesteps": timesteps,
        "actual_timesteps": result.actual_timesteps,
        "duration_seconds": duration,
        "steps_per_second": result.actual_timesteps / duration,
        "parameter_count": result.parameter_count,
        "peak_cpu_rss_bytes": _peak_cpu_rss_bytes(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "training_rows": result.training_rows,
        "training_start": result.training_start,
        "training_end": result.training_end,
        "recurrent_contract_version": result.recurrent_contract_version,
        "test_partition_loaded": False,
        "hardware": hardware,
    }


def _worker_command(
    *, symbol: str, seed: int, warmup_timesteps: int, timesteps: int, splits_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "reinforcement_learning.training.cuda_benchmark",
        "--worker",
        "--device",
        "cuda",
        "--symbol",
        symbol,
        "--seed",
        str(seed),
        "--warmup-timesteps",
        str(warmup_timesteps),
        "--budgets",
        str(timesteps),
        "--splits-dir",
        str(Path(splits_dir).resolve()),
    ]


def _parse_worker_output(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    markers = [
        line[len(CUDA_BENCHMARK_MARKER) :]
        for line in completed.stdout.splitlines()
        if line.startswith(CUDA_BENCHMARK_MARKER)
    ]
    if completed.returncode or len(markers) != 1:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CudaBenchmarkError(
            f"isolated CUDA worker failed closed (exit={completed.returncode}): {detail[-1500:]}"
        )
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError as exc:
        raise CudaBenchmarkError("isolated CUDA worker returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise CudaBenchmarkError("isolated CUDA worker returned a malformed payload")
    if payload.get("status") != "completed":
        raise CudaBenchmarkError("isolated CUDA worker did not complete successfully")
    if not torch_devices_equivalent(payload.get("effective_device"), "cuda"):
        raise CudaBenchmarkError("non-CUDA result cannot be reported as CUDA performance")
    return dict(payload)


def run_cuda_contract(
    contract: CudaBenchmarkContract,
    *,
    workers: int = 1,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    timeout_seconds: int = 7_200,
) -> dict[str, object]:
    """Execute explicit CUDA benchmarks, including bounded concurrency."""

    hardware = cuda_hardware_preflight()
    if workers not in contract.worker_candidates:
        raise CudaBenchmarkError("worker count is outside the predeclared contract")
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    for budget in contract.requested_timesteps:
        commands = [
            _worker_command(
                symbol=contract.symbol,
                seed=contract.seed,
                warmup_timesteps=contract.warmup_timesteps,
                timesteps=budget,
                splits_dir=splits_dir,
            )
            for _ in range(workers)
        ]
        processes = [
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            for command in commands
        ]
        budget_results: list[dict[str, object]] = []
        try:
            for process in processes:
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    raise CudaBenchmarkError(
                        "CUDA worker timed out and all peers were terminated"
                    ) from exc
                budget_results.append(
                    _parse_worker_output(
                        subprocess.CompletedProcess(
                            process.args, process.returncode, stdout, stderr
                        )
                    )
                )
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        results.extend(budget_results)
    wall = time.perf_counter() - started
    return {
        "contract": contract.to_dict(),
        "contract_fingerprint": contract.fingerprint,
        "workers": workers,
        "hardware": hardware,
        "wall_clock_seconds": wall,
        "aggregate_steps_per_second": sum(int(item["actual_timesteps"]) for item in results) / wall,
        "per_worker_results": results,
        "oom_or_failures": 0,
        "gpu_utilization": "not_collected_no_optional_monitoring_dependency",
        "thermal_observation": "not_available_from_portable_torch_api",
        "test_partition_loaded": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed recurrent CUDA benchmark")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--symbol", default=CUDA_BENCHMARK_SYMBOL)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-timesteps", type=int, default=512)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(CUDA_BENCHMARK_BUDGETS))
    parser.add_argument("--workers", type=int, choices=CUDA_CONCURRENCY_WORKERS, default=1)
    parser.add_argument("--splits-dir", type=Path, default=PROCESSED_SPLITS_DIR)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.preflight:
            print(json.dumps(cuda_hardware_preflight(), indent=2, sort_keys=True))
            return 0
        contract = CudaBenchmarkContract(
            symbol=args.symbol,
            seed=args.seed,
            warmup_timesteps=args.warmup_timesteps,
            requested_timesteps=tuple(args.budgets),
        )
        if args.worker:
            if len(args.budgets) != 1:
                raise CudaBenchmarkError("isolated worker accepts exactly one budget")
            payload = _worker_payload(
                symbol=args.symbol,
                seed=args.seed,
                warmup_timesteps=args.warmup_timesteps,
                timesteps=args.budgets[0],
                splits_dir=args.splits_dir,
            )
            print(CUDA_BENCHMARK_MARKER + json.dumps(payload, sort_keys=True))
            return 0
        payload = run_cuda_contract(
            contract, workers=args.workers, splits_dir=args.splits_dir
        )
        if args.output_json is not None:
            path = Path(args.output_json)
            if path.exists():
                raise FileExistsError(f"benchmark result exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (CudaBenchmarkError, TorchDeviceError, OSError, ValueError) as exc:
        print(f"CUDA benchmark blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUDA_BENCHMARK_BUDGETS",
    "CUDA_BENCHMARK_CONTRACT_VERSION",
    "CUDA_BENCHMARK_SYMBOL",
    "CUDA_CONCURRENCY_CONTRACT_VERSION",
    "CUDA_CONCURRENCY_WORKERS",
    "CudaBenchmarkContract",
    "CudaBenchmarkError",
    "benchmark_command",
    "cuda_hardware_preflight",
    "run_cuda_contract",
]
