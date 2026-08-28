"""Isolated process worker for one recurrent symbol-training job.

The worker owns only its symbol-specific temporary workspace. It never receives
the run store, job-record path, manifest path, registry path, or TEST path.
Canonical job transitions and artifact promotion remain parent responsibilities.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Mapping


PARALLEL_WORKER_PROTOCOL_VERSION = "recurrent_parallel_worker_v1"
CPU_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _send(connection: object, payload: Mapping[str, object]) -> None:
    connection.send(dict(payload))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.sys.platform == "darwin" else value * 1024


def configure_cpu_thread_policy(threads: int | None) -> dict[str, object]:
    """Apply an explicit per-process Torch/BLAS policy when requested."""

    if threads is not None and (isinstance(threads, bool) or threads < 1):
        raise ValueError("CPU threads per worker must be a positive integer")
    if threads is not None:
        for name in CPU_THREAD_ENVIRONMENT_VARIABLES:
            os.environ[name] = str(threads)

    # Import after setting BLAS/OpenMP hints. The spawned process has not loaded
    # the training stack before this point.
    import torch

    if threads is not None:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A fresh spawned worker should permit this. Fail if an unexpected
            # initialization order made the requested policy unenforceable.
            if int(torch.get_num_interop_threads()) != 1:
                raise
    return {
        "requested_threads_per_worker": threads,
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "environment": {
            name: os.environ.get(name) for name in CPU_THREAD_ENVIRONMENT_VARIABLES
        },
    }


def _config_from_payload(payload: Mapping[str, object]):
    from .recurrent_config import RecurrentPPOConfig

    allowed = {item.name for item in fields(RecurrentPPOConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if "net_arch" in values:
        values["net_arch"] = tuple(values["net_arch"])
    return RecurrentPPOConfig(**values)


def _save_worker_model(model: object, workspace: Path) -> tuple[Path, str]:
    from reinforcement_learning.integrity import sha256_file

    destination = workspace / "model.zip"
    if destination.exists():
        raise FileExistsError(f"worker model already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=workspace, prefix=".model.", suffix=".zip"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        model.save(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("worker trainer did not create a model archive")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, sha256_file(destination)


def run_recurrent_process_worker(
    request: Mapping[str, object], connection: object
) -> None:
    """Train and optionally validate one symbol, returning bounded telemetry."""

    started_clock = time.perf_counter()
    started_epoch = time.time()
    symbol = str(request.get("symbol") or "")
    actual_timesteps = 0
    effective_device: str | None = None
    thread_policy: dict[str, object] = {}
    try:
        if request.get("protocol_version") != PARALLEL_WORKER_PROTOCOL_VERSION:
            raise ValueError("parallel worker protocol is incompatible")
        if not symbol:
            raise ValueError("parallel worker symbol is required")
        if request.get("requested_device") != "cpu":
            raise ValueError("parallel recurrent workers require explicit CPU")
        if request.get("test_partition_loaded") is not False:
            raise ValueError("TEST cannot enter a recurrent worker request")
        workspace = Path(str(request["workspace"])).resolve(strict=False)
        workspace.mkdir(parents=True, exist_ok=False)
        thread_policy = configure_cpu_thread_policy(
            request.get("cpu_threads_per_worker")
        )

        from feature_engineering.storage import atomic_write_json
        from reinforcement_learning.evaluation.recurrent_evaluator import (
            evaluate_recurrent_on_validation,
        )
        from reinforcement_learning.training.recurrent_trainer import (
            train_recurrent_single_symbol,
        )

        config = _config_from_payload(dict(request["config"]))

        def progress(event: object) -> bool:
            _send(
                connection,
                {
                    "type": "progress",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "phase": str(event.phase),
                    "actual_timesteps": int(event.current_timesteps),
                    "timestamp": str(event.timestamp),
                    "elapsed_seconds": time.perf_counter() - started_clock,
                },
            )
            return True

        result = train_recurrent_single_symbol(
            symbol,
            config=config,
            device="cpu",
            total_timesteps=config.total_timesteps,
            seed=config.seed,
            progress_callback=progress,
            splits_dir=Path(str(request["splits_dir"])),
            smoke_test=config.total_timesteps <= 1_024,
        )
        actual_timesteps = int(result.actual_timesteps)
        effective_device = str(result.device)
        if result.status == "interrupted":
            _send(
                connection,
                {
                    "type": "result",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "status": "interrupted",
                    "error": result.error or result.message,
                    "actual_timesteps": actual_timesteps,
                    "effective_device": effective_device,
                    "duration_seconds": time.perf_counter() - started_clock,
                    "worker_pid": os.getpid(),
                    "worker_started_epoch": started_epoch,
                    "worker_finished_epoch": time.time(),
                    "worker_peak_rss_bytes": _peak_rss_bytes(),
                    "cpu_thread_policy": thread_policy,
                    "test_partition_loaded": False,
                },
            )
            return
        if not result.succeeded or result.model is None:
            raise RuntimeError(result.error or result.message or "trainer failed")
        from .devices import torch_devices_equivalent

        if not torch_devices_equivalent(effective_device, "cpu"):
            raise RuntimeError(
                f"CPU worker reported unexpected device {effective_device!r}"
            )
        model_path, model_sha256 = _save_worker_model(result.model, workspace)
        validation_file: str | None = None
        validation_started = time.perf_counter()
        if bool(request.get("validation_enabled")):
            _send(
                connection,
                {
                    "type": "stage",
                    "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                    "symbol": symbol,
                    "stage": "validating",
                    "actual_timesteps": actual_timesteps,
                    "elapsed_seconds": time.perf_counter() - started_clock,
                },
            )
            validation = evaluate_recurrent_on_validation(
                result.model,
                symbol,
                trainer_result=result,
                seed=config.seed,
                splits_dir=Path(str(request["splits_dir"])),
            )
            validation_payload = validation.to_dict(include_history=False)
            if validation_payload.get("test_evaluated") is True:
                raise RuntimeError("worker evaluator reported TEST access")
            validation_path = workspace / "validation.json"
            atomic_write_json(validation_payload, validation_path)
            validation_file = validation_path.name
        validation_duration = (
            time.perf_counter() - validation_started
            if bool(request.get("validation_enabled"))
            else 0.0
        )

        duration = time.perf_counter() - started_clock
        if not math.isfinite(duration):
            raise RuntimeError("worker duration is not finite")
        _send(
            connection,
            {
                "type": "result",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "status": "completed",
                "error": None,
                "actual_timesteps": actual_timesteps,
                "requested_timesteps": int(result.requested_timesteps),
                "effective_device": effective_device,
                "duration_seconds": duration,
                "training_duration_seconds": float(result.duration_seconds),
                "validation_duration_seconds": validation_duration,
                "training_rows": int(result.training_rows),
                "training_start": result.training_start,
                "training_end": result.training_end,
                "parameter_count": int(result.parameter_count),
                "training_diagnostics": (
                    result.training_diagnostics.to_dict()
                    if result.training_diagnostics is not None
                    else None
                ),
                "model_file": model_path.name,
                "model_sha256": model_sha256,
                "validation_file": validation_file,
                "worker_pid": os.getpid(),
                "worker_started_epoch": started_epoch,
                "worker_finished_epoch": time.time(),
                "worker_peak_rss_bytes": _peak_rss_bytes(),
                "cpu_thread_policy": thread_policy,
                "test_partition_loaded": False,
            },
        )
    except KeyboardInterrupt:
        _send(
            connection,
            {
                "type": "result",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "status": "interrupted",
                "error": "worker received KeyboardInterrupt",
                "actual_timesteps": actual_timesteps,
                "effective_device": effective_device,
                "duration_seconds": time.perf_counter() - started_clock,
                "worker_pid": os.getpid(),
                "worker_started_epoch": started_epoch,
                "worker_finished_epoch": time.time(),
                "worker_peak_rss_bytes": _peak_rss_bytes(),
                "cpu_thread_policy": thread_policy,
                "test_partition_loaded": False,
            },
        )
    except BaseException as exc:
        _send(
            connection,
            {
                "type": "result",
                "protocol_version": PARALLEL_WORKER_PROTOCOL_VERSION,
                "symbol": symbol,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "actual_timesteps": actual_timesteps,
                "effective_device": effective_device,
                "duration_seconds": max(0.0, time.perf_counter() - started_clock),
                "worker_pid": os.getpid(),
                "worker_started_epoch": started_epoch,
                "worker_finished_epoch": time.time(),
                "worker_peak_rss_bytes": _peak_rss_bytes(),
                "cpu_thread_policy": thread_policy,
                "test_partition_loaded": False,
            },
        )
    finally:
        connection.close()


__all__ = [
    "CPU_THREAD_ENVIRONMENT_VARIABLES",
    "PARALLEL_WORKER_PROTOCOL_VERSION",
    "configure_cpu_thread_policy",
    "run_recurrent_process_worker",
]
