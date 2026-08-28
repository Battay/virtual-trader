"""Bounded CPU process-concurrency benchmark for recurrent symbol agents.

The benchmark invokes the production run store, process worker, trainer, and
validation path. It writes models only below a temporary benchmark directory,
removes that directory afterward, and rejects any protected-artifact change.
TEST is neither selected nor exposed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
from typing import Mapping, Sequence

from data_pipeline.src.config import (
    CANONICAL_RECURRENT_TRAIN_V2_DIR,
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    SAVED_MODELS_DIR,
)
from reinforcement_learning.integrity import sha256_file

from .job_state import COMPLETED, FAILED, INTERRUPTED, canonical_hash
from .recurrent_config import RecurrentPPOConfig
from .recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    SUPPORTED_PROCESS_WORKERS,
    RecurrentUniverseDiscovery,
    create_training_run,
    discover_recurrent_training_universe,
    execute_queued_jobs,
)


CPU_PARALLEL_BENCHMARK_VERSION = "recurrent_cpu_parallel_benchmark_v1"
CPU_PARALLEL_SELECTION_POLICY = "train_row_depth_quantiles_v1"
DEFAULT_CPU_BENCHMARK_TIMESTEPS = 100_000
DEFAULT_BENCHMARK_SYMBOL_COUNT = 4
MATERIAL_THROUGHPUT_GAIN = 0.10
LOWER_WORKER_HEADROOM_TOLERANCE = 0.05


class CPUParallelBenchmarkError(RuntimeError):
    """Raised when a CPU concurrency result cannot be reported safely."""


@dataclass(frozen=True)
class CPUParallelBenchmarkContract:
    benchmark_version: str
    symbols: tuple[str, ...]
    symbol_selection_policy: str
    worker_candidates: tuple[int, ...]
    thread_policy: tuple[tuple[int, int], ...]
    requested_timesteps_per_symbol: int
    seed: int
    requested_device: str = "cpu"
    validation_enabled: bool = True
    test_partition_loaded: bool = False

    def __post_init__(self) -> None:
        if self.benchmark_version != CPU_PARALLEL_BENCHMARK_VERSION:
            raise ValueError("CPU parallel benchmark version is incompatible")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("benchmark symbols must be nonempty and unique")
        if tuple(sorted(self.symbols)) != self.symbols:
            raise ValueError("benchmark symbols must be deterministically ordered")
        if (
            not self.worker_candidates
            or len(set(self.worker_candidates)) != len(self.worker_candidates)
            or any(value not in SUPPORTED_PROCESS_WORKERS for value in self.worker_candidates)
        ):
            raise ValueError("worker candidates must be unique values from 1, 2, 4")
        policy = dict(self.thread_policy)
        if set(policy) != set(self.worker_candidates) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in policy.values()
        ):
            raise ValueError("thread policy must define a positive value per candidate")
        if self.requested_timesteps_per_symbol < 1 or self.seed < 0:
            raise ValueError("benchmark timesteps/seed are invalid")
        if self.requested_device != "cpu":
            raise ValueError("parallel benchmark requires explicit CPU")
        if self.test_partition_loaded:
            raise ValueError("TEST cannot enter CPU parallel benchmarking")

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["worker_candidates"] = list(self.worker_candidates)
        payload["thread_policy"] = {
            str(workers): threads for workers, threads in self.thread_policy
        }
        payload["ppo_lstm_config"] = RecurrentPPOConfig(
            total_timesteps=self.requested_timesteps_per_symbol,
            seed=self.seed,
            device="cpu",
        ).to_dict()
        return payload


def select_representative_symbols(
    discovery: RecurrentUniverseDiscovery,
    *,
    count: int = DEFAULT_BENCHMARK_SYMBOL_COUNT,
) -> tuple[str, ...]:
    """Select deterministic TRAIN-depth quantile representatives."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("representative symbol count must be positive")
    eligible_mask = discovery.records["category"].eq(ELIGIBLE_TRAINABLE)
    if "validation_available" in discovery.records.columns:
        eligible_mask &= discovery.records["validation_available"].eq(True)
    eligible = discovery.records.loc[
        eligible_mask,
        ["symbol", "train_rows"],
    ].copy()
    eligible["train_rows"] = eligible["train_rows"].astype("int64")
    eligible = eligible.sort_values(
        ["train_rows", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(eligible) < count:
        raise CPUParallelBenchmarkError(
            f"benchmark requires {count} trainable symbols; found {len(eligible)}"
        )
    indexes = [min(len(eligible) - 1, ((2 * rank + 1) * len(eligible)) // (2 * count)) for rank in range(count)]
    symbols = tuple(sorted(str(eligible.iloc[index]["symbol"]) for index in indexes))
    if len(set(symbols)) != count:
        raise CPUParallelBenchmarkError("representative selection produced duplicates")
    return symbols


def _file_snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    directory = Path(root)
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): (path.stat().st_size, sha256_file(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _training_source_metadata_snapshot(
    symbols: Sequence[str], *, splits_dir: Path
) -> dict[str, tuple[int, int]]:
    """Snapshot metadata only; sealed TEST file contents are never read."""

    roots = (
        Path(splits_dir) / "symbols",
        Path(CANONICAL_RECURRENT_TRAIN_V2_DIR),
    )
    result: dict[str, tuple[int, int]] = {}
    for root in roots:
        for symbol in symbols:
            directory = root / symbol
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    result[str(path.resolve(strict=False))] = (
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                    )
    return result


def _system_memory_observation() -> dict[str, int | None]:
    def value(name: str) -> int | None:
        try:
            return int(os.sysconf(name))
        except (OSError, TypeError, ValueError):
            return None

    page_size = value("SC_PAGE_SIZE")
    physical_pages = value("SC_PHYS_PAGES")
    available_pages = value("SC_AVPHYS_PAGES")
    return {
        "total_bytes": (
            page_size * physical_pages
            if page_size is not None and physical_pages is not None
            else None
        ),
        "available_bytes": (
            page_size * available_pages
            if page_size is not None and available_pages is not None
            else None
        ),
    }


def _parent_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _candidate_summary(
    *,
    store: object,
    outcomes: Sequence[object],
    symbols: tuple[str, ...],
    workers: int,
    threads_per_worker: int,
    wall_clock_seconds: float,
) -> dict[str, object]:
    jobs = {job.symbol: job for job in store.list_jobs() if job.symbol in symbols}
    logs: dict[str, Mapping[str, object]] = {}
    for symbol in symbols:
        paths = sorted((store.run_directory / "logs" / symbol).glob("*.json"))
        if len(paths) != 1:
            raise CPUParallelBenchmarkError(
                f"expected one benchmark log for {symbol}; found {len(paths)}"
            )
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CPUParallelBenchmarkError("benchmark worker log is malformed")
        logs[symbol] = payload
    completed = sum(job.status == COMPLETED for job in jobs.values())
    failed = sum(job.status == FAILED for job in jobs.values())
    interrupted = sum(job.status == INTERRUPTED for job in jobs.values())
    actual_timesteps = sum(job.completed_timesteps for job in jobs.values())
    devices = sorted(
        {str(job.effective_device) for job in jobs.values() if job.effective_device}
    )
    test_loaded = any(payload.get("test_partition_loaded") is not False for payload in logs.values())
    per_agent = []
    for symbol in symbols:
        job = jobs[symbol]
        payload = logs[symbol]
        duration = float(payload.get("duration_seconds", 0.0))
        per_agent.append(
            {
                "symbol": symbol,
                "status": job.status,
                "requested_timesteps": job.requested_timesteps,
                "actual_timesteps": job.completed_timesteps,
                "duration_seconds": duration,
                "training_duration_seconds": payload.get(
                    "training_duration_seconds"
                ),
                "validation_duration_seconds": payload.get(
                    "validation_duration_seconds"
                ),
                "steps_per_second": (
                    job.completed_timesteps / duration if duration > 0 else None
                ),
                "effective_device": job.effective_device,
                "validation_status": job.validation_status,
                "worker_pid": payload.get("worker_pid"),
                "worker_peak_rss_bytes": payload.get("worker_peak_rss_bytes"),
                "training_rows": payload.get("training_rows"),
                "training_start": payload.get("training_start"),
                "training_end": payload.get("training_end"),
                "training_diagnostics": payload.get("training_diagnostics"),
                "test_partition_loaded": payload.get("test_partition_loaded"),
            }
        )
    safe = (
        len(outcomes) == len(symbols)
        and completed == len(symbols)
        and failed == 0
        and interrupted == 0
        and devices == ["cpu"]
        and not test_loaded
        and all(job.validation_status == "completed" for job in jobs.values())
        and store.read_active_workers() == {}
    )
    worker_rss = [
        int(value["worker_peak_rss_bytes"])
        for value in logs.values()
        if isinstance(value.get("worker_peak_rss_bytes"), int)
    ]
    return {
        "workers": workers,
        "cpu_threads_per_worker": threads_per_worker,
        "symbols": list(symbols),
        "wall_clock_seconds": wall_clock_seconds,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "interrupted_jobs": interrupted,
        "aggregate_actual_timesteps": actual_timesteps,
        "aggregate_steps_per_second": (
            actual_timesteps / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
        "agents_per_hour": (
            completed * 3600.0 / wall_clock_seconds
            if wall_clock_seconds > 0
            else None
        ),
        "effective_devices": devices,
        "validation_behavior": "validation_only_after_train",
        "test_partition_loaded": test_loaded,
        "peak_parent_rss_bytes": _parent_peak_rss_bytes(),
        "maximum_worker_peak_rss_bytes": max(worker_rss, default=None),
        "sum_worker_peak_rss_bytes": sum(worker_rss) if worker_rss else None,
        "system_memory_after": _system_memory_observation(),
        "per_agent": per_agent,
        "safe": safe,
    }


def recommend_worker_count(
    candidates: Sequence[Mapping[str, object]],
    *,
    full_budget: bool,
) -> dict[str, object]:
    """Apply a predeclared throughput/headroom rule, never validation returns."""

    safe = [item for item in candidates if item.get("safe") is True]
    if not full_budget:
        return {
            "worker_count": None,
            "status": "qualification_only_not_permanent_selection",
            "reason": "A sub-100k smoke does not select the Mac production worker count.",
        }
    baseline = next((item for item in safe if item.get("workers") == 1), None)
    if baseline is None:
        return {
            "worker_count": None,
            "status": "blocked",
            "reason": "No safe one-worker baseline completed.",
        }
    baseline_rate = float(baseline["aggregate_steps_per_second"])
    material = [
        item
        for item in safe
        if float(item["aggregate_steps_per_second"])
        >= baseline_rate * (1.0 + MATERIAL_THROUGHPUT_GAIN)
    ]
    if not material:
        return {
            "worker_count": 1,
            "status": "selected",
            "reason": "No safe parallel candidate improved aggregate throughput by 10%.",
        }
    best = max(material, key=lambda item: float(item["agents_per_hour"]))
    lower = [item for item in material if int(item["workers"]) < int(best["workers"])]
    if lower:
        next_lower = max(lower, key=lambda item: int(item["workers"]))
        if float(next_lower["agents_per_hour"]) >= float(best["agents_per_hour"]) * (
            1.0 - LOWER_WORKER_HEADROOM_TOLERANCE
        ):
            best = next_lower
    return {
        "worker_count": int(best["workers"]),
        "status": "selected",
        "reason": (
            "Highest safe agents/hour, preferring the lower worker count when "
            "within 5% for memory/CPU headroom."
        ),
    }


def run_cpu_parallel_benchmark(
    *,
    timesteps: int = DEFAULT_CPU_BENCHMARK_TIMESTEPS,
    seed: int = 42,
    worker_candidates: tuple[int, ...] = SUPPORTED_PROCESS_WORKERS,
    thread_policy: Mapping[int, int],
    symbol_count: int = DEFAULT_BENCHMARK_SYMBOL_COUNT,
    discovery: RecurrentUniverseDiscovery | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    registry_path: Path = MODEL_REGISTRY_PATH,
    saved_models_dir: Path = SAVED_MODELS_DIR,
    models_data_dir: Path = MODELS_DATA_DIR,
) -> dict[str, object]:
    """Execute isolated 1/2/4 candidate runs and return bounded telemetry."""

    resolved_discovery = discovery or discover_recurrent_training_universe(
        splits_dir=Path(splits_dir)
    )
    symbols = select_representative_symbols(
        resolved_discovery, count=symbol_count
    )
    missing_thread_policy = set(worker_candidates).difference(thread_policy)
    if missing_thread_policy:
        raise CPUParallelBenchmarkError(
            "thread policy is missing worker candidates: "
            + ", ".join(map(str, sorted(missing_thread_policy)))
        )
    contract = CPUParallelBenchmarkContract(
        benchmark_version=CPU_PARALLEL_BENCHMARK_VERSION,
        symbols=symbols,
        symbol_selection_policy=CPU_PARALLEL_SELECTION_POLICY,
        worker_candidates=tuple(worker_candidates),
        thread_policy=tuple(
            (workers, int(thread_policy[workers])) for workers in worker_candidates
        ),
        requested_timesteps_per_symbol=timesteps,
        seed=seed,
    )
    registry_before = _file_snapshot(Path(registry_path))
    saved_before = _tree_snapshot(Path(saved_models_dir))
    models_before = _tree_snapshot(Path(models_data_dir))
    sources_before = _training_source_metadata_snapshot(
        symbols, splits_dir=Path(splits_dir)
    )
    candidates: list[dict[str, object]] = []
    system_memory_before = _system_memory_observation()
    with tempfile.TemporaryDirectory(prefix="virtual-trader-cpu-parallel-") as root_name:
        root = Path(root_name)
        for workers in contract.worker_candidates:
            config = RecurrentPPOConfig(
                total_timesteps=timesteps,
                seed=seed,
                device="cpu",
            )
            store = create_training_run(
                resolved_discovery,
                config=config,
                runs_root=root / f"workers_{workers}" / "runs",
                validation_enabled=True,
            )
            started = time.perf_counter()
            outcomes = execute_queued_jobs(
                store,
                config=config,
                max_jobs=len(symbols),
                workers=workers,
                cpu_threads_per_worker=dict(contract.thread_policy)[workers],
                symbols=symbols,
                splits_dir=Path(splits_dir),
                registry_path=Path(registry_path),
                force_process_workers=True,
            )
            wall = time.perf_counter() - started
            candidates.append(
                _candidate_summary(
                    store=store,
                    outcomes=outcomes,
                    symbols=symbols,
                    workers=workers,
                    threads_per_worker=dict(contract.thread_policy)[workers],
                    wall_clock_seconds=wall,
                )
            )
    registry_unchanged = _file_snapshot(Path(registry_path)) == registry_before
    saved_unchanged = _tree_snapshot(Path(saved_models_dir)) == saved_before
    models_unchanged = _tree_snapshot(Path(models_data_dir)) == models_before
    sources_unchanged = _training_source_metadata_snapshot(
        symbols, splits_dir=Path(splits_dir)
    ) == sources_before
    if not (
        registry_unchanged
        and saved_unchanged
        and models_unchanged
        and sources_unchanged
    ):
        raise CPUParallelBenchmarkError(
            "CPU benchmark changed a protected registry/model/source artifact"
        )
    full_budget = timesteps >= DEFAULT_CPU_BENCHMARK_TIMESTEPS
    return {
        "benchmark_version": CPU_PARALLEL_BENCHMARK_VERSION,
        "contract": contract.to_dict(),
        "contract_fingerprint": contract.fingerprint,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "system_memory_before": system_memory_before,
        "candidates": candidates,
        "recommendation": recommend_worker_count(candidates, full_budget=full_budget),
        "selection_authoritative_for_100k": full_budget,
        "registry_unchanged": registry_unchanged,
        "production_saved_models_unchanged": saved_unchanged,
        "data_models_unchanged": models_unchanged,
        "training_source_artifacts_unchanged": sources_unchanged,
        "temporary_runs_cleaned": True,
        "test_partition_loaded": False,
    }


def _parse_thread_policy(values: Sequence[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for raw in values:
        try:
            workers_text, threads_text = raw.split(":", maxsplit=1)
            workers, threads = int(workers_text), int(threads_text)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "thread policy entries must use WORKERS:THREADS"
            ) from exc
        if workers in result:
            raise argparse.ArgumentTypeError("thread policy repeats a worker count")
        result[workers] = threads
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit bounded CPU parallel recurrent benchmark"
    )
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CPU_BENCHMARK_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        choices=SUPPORTED_PROCESS_WORKERS,
        default=list(SUPPORTED_PROCESS_WORKERS),
    )
    parser.add_argument(
        "--thread-policy",
        nargs="+",
        required=True,
        metavar="WORKERS:THREADS",
    )
    parser.add_argument("--symbol-count", type=int, default=DEFAULT_BENCHMARK_SYMBOL_COUNT)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.output_json is not None and Path(args.output_json).exists():
            raise FileExistsError(f"benchmark output exists: {args.output_json}")
        thread_policy = _parse_thread_policy(args.thread_policy)
        result = run_cpu_parallel_benchmark(
            timesteps=args.timesteps,
            seed=args.seed,
            worker_candidates=tuple(args.workers),
            thread_policy=thread_policy,
            symbol_count=args.symbol_count,
        )
        if args.output_json is not None:
            destination = Path(args.output_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if all(item["safe"] for item in result["candidates"]) else 1
    except (CPUParallelBenchmarkError, OSError, ValueError) as exc:
        print(
            f"CPU parallel benchmark blocked: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_PARALLEL_BENCHMARK_VERSION",
    "CPU_PARALLEL_SELECTION_POLICY",
    "CPUParallelBenchmarkContract",
    "CPUParallelBenchmarkError",
    "recommend_worker_count",
    "run_cpu_parallel_benchmark",
    "select_representative_symbols",
]
