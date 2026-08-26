"""Explicit CPU/CUDA/Apple-MPS device resolution for PPO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import torch


SUPPORTED_TORCH_DEVICES = frozenset({"cpu", "cuda", "mps", "auto"})


class TorchDeviceError(RuntimeError):
    """Raised when a requested PPO device cannot be used safely."""


@dataclass(frozen=True)
class TorchDeviceResolution:
    """The explicit request and the concrete device selected for SB3."""

    requested_device: str
    resolved_device: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool = False
    cuda_device_count: int = 0
    device_name: str | None = None

    @property
    def accelerator_selected(self) -> bool:
        return self.resolved_device in {"cuda", "mps"}

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mps_state() -> tuple[bool, bool]:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False, False
    try:
        built = bool(backend.is_built())
    except (AttributeError, RuntimeError):
        built = False
    try:
        available = bool(backend.is_available())
    except (AttributeError, RuntimeError):
        available = False
    return built, available


def _cuda_state() -> tuple[bool, int, str | None]:
    """Return bounded CUDA discovery without starting a training run."""

    try:
        available = bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        available = False
    if not available:
        return False, 0, None
    try:
        count = int(torch.cuda.device_count())
        name = str(torch.cuda.get_device_name(0)) if count > 0 else None
    except (AttributeError, RuntimeError):
        return False, 0, None
    return count > 0, count, name


def resolve_torch_device(requested_device: str) -> TorchDeviceResolution:
    """Resolve CPU/CUDA/MPS/AUTO without hiding accelerator fallback.

    Recurrent MPS training is known to be unstable on the measured Apple-M2
    environment, so AUTO prefers CUDA and otherwise selects CPU. MPS remains an
    explicit opt-in diagnostic path.
    """
    if not isinstance(requested_device, str):
        raise TorchDeviceError(
            "requested device must be one of: auto, cpu, cuda, mps"
        )
    requested = requested_device.strip().lower()
    if requested not in SUPPORTED_TORCH_DEVICES:
        raise TorchDeviceError(
            "requested device must be one of: auto, cpu, cuda, mps"
        )
    mps_built, mps_available = _mps_state()
    cuda_available, cuda_device_count, cuda_name = _cuda_state()
    if requested == "cpu":
        resolved = "cpu"
    elif requested == "cuda":
        if not cuda_available:
            raise TorchDeviceError(
                "CUDA was explicitly requested but is unavailable; "
                "CPU fallback is disabled for explicit CUDA requests"
            )
        resolved = "cuda"
    elif requested == "mps":
        if not mps_available:
            raise TorchDeviceError(
                "MPS was explicitly requested but is unavailable "
                f"(built={mps_built}, available={mps_available}); "
                "CPU fallback is disabled for explicit MPS requests"
            )
        resolved = "mps"
    else:
        resolved = "cuda" if cuda_available else "cpu"
    fallback_setting = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    if resolved == "mps" and fallback_setting in {"1", "true", "yes", "on"}:
        raise TorchDeviceError(
            "MPS was selected while PYTORCH_ENABLE_MPS_FALLBACK is enabled; "
            "training aborted because unsupported operations could be hidden on CPU"
        )
    return TorchDeviceResolution(
        requested_device=requested,
        resolved_device=resolved,
        mps_built=mps_built,
        mps_available=mps_available,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        device_name=(cuda_name if resolved == "cuda" else None),
    )


def torch_device_type(device: object) -> str:
    """Return the canonical torch device type or fail with useful context."""
    try:
        return torch.device(str(device)).type
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TorchDeviceError(f"Invalid torch device value: {device!r}") from exc


def torch_devices_equivalent(actual: object, expected: object) -> bool:
    """Treat implicit CPU/MPS index and explicit index zero as equivalent."""
    try:
        actual_device = torch.device(str(actual))
        expected_device = torch.device(str(expected))
    except (RuntimeError, TypeError, ValueError):
        return False
    if actual_device.type != expected_device.type:
        return False
    if actual_device.type in {"cpu", "cuda", "mps"}:
        return actual_device.index in {None, 0} and expected_device.index in {None, 0}
    return actual_device == expected_device


def verify_sb3_model_device(
    model: Any,
    resolution: TorchDeviceResolution,
) -> str:
    """Verify both SB3's model device and its actual policy parameter device."""
    model_device = str(getattr(model, "device", ""))
    if not torch_devices_equivalent(model_device, resolution.resolved_device):
        raise TorchDeviceError(
            "Stable-Baselines3 model device does not match the resolved device: "
            f"requested={resolution.requested_device}, "
            f"resolved={resolution.resolved_device}, actual={model_device}"
        )
    policy = getattr(model, "policy", None)
    if policy is None:
        raise TorchDeviceError("Stable-Baselines3 PPO model has no policy")
    policy_reported_device = str(getattr(policy, "device", ""))
    if not torch_devices_equivalent(
        policy_reported_device, resolution.resolved_device
    ):
        raise TorchDeviceError(
            "Stable-Baselines3 policy reports a device that does not match the "
            f"resolved device: requested={resolution.requested_device}, "
            f"resolved={resolution.resolved_device}, actual={policy_reported_device}"
        )
    try:
        tensors = tuple(policy.parameters()) + tuple(policy.buffers())
    except (TypeError, AttributeError) as exc:
        raise TorchDeviceError("Stable-Baselines3 policy tensors are unavailable") from exc
    if not tensors:
        raise TorchDeviceError("Stable-Baselines3 policy has no parameters or buffers")
    tensor_devices = {str(tensor.device) for tensor in tensors}
    incompatible = sorted(
        device
        for device in tensor_devices
        if not torch_devices_equivalent(device, resolution.resolved_device)
    )
    if incompatible:
        raise TorchDeviceError(
            "Stable-Baselines3 policy tensors do not match the resolved device: "
            f"requested={resolution.requested_device}, "
            f"resolved={resolution.resolved_device}, actual={incompatible}"
        )
    if not all(torch_devices_equivalent(model_device, item) for item in tensor_devices):
        raise TorchDeviceError(
            "Stable-Baselines3 model and policy tensors report different devices"
        )
    return policy_reported_device


def synchronize_torch_device(device: str) -> None:
    """Synchronize accelerator work; CPU requires no synchronization."""

    device_type = torch_device_type(device)
    if device_type == "cpu":
        return
    if device_type == "cuda":
        try:
            torch.cuda.synchronize()
        except (AttributeError, RuntimeError) as exc:
            raise TorchDeviceError(f"CUDA synchronization failed: {exc}") from exc
        return
    resolution = resolve_torch_device("mps")
    if resolution.resolved_device != "mps":  # defensive; explicit MPS never falls back
        raise TorchDeviceError("MPS synchronization requested without MPS")
    synchronize = getattr(torch.mps, "synchronize", None)
    if not callable(synchronize):
        raise TorchDeviceError("This PyTorch build lacks torch.mps.synchronize()")
    try:
        synchronize()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TorchDeviceError(
            "MPS synchronization failed; asynchronous accelerator work may have "
            f"failed and the run cannot be reported as successful: {exc}"
        ) from exc
