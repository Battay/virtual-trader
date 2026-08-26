"""Offline tests for the explicit CUDA recurrent benchmark contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforcement_learning.training import cuda_benchmark
from reinforcement_learning.training.cuda_benchmark import (
    CudaBenchmarkContract,
    CudaBenchmarkError,
    benchmark_command,
    cuda_hardware_preflight,
)
from reinforcement_learning.training.devices import TorchDeviceError, TorchDeviceResolution


def test_cuda_preflight_fails_closed_without_cpu_fallback(monkeypatch) -> None:
    def unavailable(_requested: str):
        raise TorchDeviceError(
            "CUDA was explicitly requested but is unavailable; CPU fallback disabled"
        )

    monkeypatch.setattr(cuda_benchmark, "resolve_torch_device", unavailable)
    with pytest.raises(CudaBenchmarkError, match="fallback"):
        cuda_hardware_preflight()


def test_cuda_preflight_records_hardware_and_effective_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        cuda_benchmark,
        "resolve_torch_device",
        lambda _requested: TorchDeviceResolution(
            requested_device="cuda",
            resolved_device="cuda",
            mps_built=False,
            mps_available=False,
            cuda_available=True,
            cuda_device_count=1,
            device_name="Test NVIDIA GPU",
        ),
    )
    monkeypatch.setattr(
        cuda_benchmark.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=24 * 1024**3),
    )
    monkeypatch.setattr(
        cuda_benchmark.torch.cuda, "get_device_capability", lambda _index: (8, 9)
    )

    result = cuda_hardware_preflight()

    assert result["requested_device"] == "cuda"
    assert result["effective_device"] == "cuda"
    assert result["gpu_model"] == "Test NVIDIA GPU"
    assert result["gpu_total_memory_bytes"] == 24 * 1024**3


def test_cuda_contract_and_command_are_deterministic_and_explicit(tmp_path: Path) -> None:
    contract = CudaBenchmarkContract()
    first = benchmark_command(
        contract=contract, output_json=tmp_path / "result.json", workers=2
    )
    second = benchmark_command(
        contract=contract, output_json=tmp_path / "result.json", workers=2
    )

    assert first == second
    assert contract.fingerprint == CudaBenchmarkContract().fingerprint
    assert "--run" in first
    assert "--device" in first and first[first.index("--device") + 1] == "cuda"
    assert "--workers" in first and first[first.index("--workers") + 1] == "2"
    assert tuple(contract.requested_timesteps) == (50_000, 100_000, 250_000)
    assert contract.to_dict()["no_cpu_fallback"] is True
    assert contract.to_dict()["test_partition_loaded"] is False


def test_non_cuda_worker_result_cannot_be_reported_as_cuda() -> None:
    completed = cuda_benchmark.subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=(
            cuda_benchmark.CUDA_BENCHMARK_MARKER
            + '{"status":"completed","effective_device":"cpu"}\n'
        ),
        stderr="",
    )
    with pytest.raises(CudaBenchmarkError, match="non-CUDA"):
        cuda_benchmark._parse_worker_output(completed)


def test_contract_rejects_test_and_unapproved_workers() -> None:
    with pytest.raises(ValueError, match="TEST"):
        CudaBenchmarkContract(test_partition_loaded=True)
    with pytest.raises(ValueError, match="workers"):
        benchmark_command(
            contract=CudaBenchmarkContract(), output_json=Path("result.json"), workers=3
        )
