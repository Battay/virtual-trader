"""Versioned Stable-Baselines3 PPO configuration for single-symbol training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real

from .devices import SUPPORTED_TORCH_DEVICES


PPO_CONFIG_VERSION = "ppo_single_symbol_v1"


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


@dataclass(frozen=True)
class PPOConfig:
    """Frozen pilot configuration with an explicit runtime device request."""

    config_version: str = PPO_CONFIG_VERSION
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    seed: int = 42
    total_timesteps: int = 100_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.config_version != PPO_CONFIG_VERSION:
            raise ValueError(f"config_version must be {PPO_CONFIG_VERSION!r}")
        if self.policy != "MlpPolicy":
            raise ValueError("5B-1 supports only the MlpPolicy PPO policy")
        _require_integer("n_steps", self.n_steps, minimum=2)
        _require_integer("batch_size", self.batch_size, minimum=2)
        _require_integer("n_epochs", self.n_epochs, minimum=1)
        _require_integer("seed", self.seed, minimum=0)
        _require_integer("total_timesteps", self.total_timesteps, minimum=1)
        if self.batch_size > self.n_steps:
            raise ValueError("batch_size cannot exceed n_steps for one environment")
        if self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps for deterministic rollouts")

        learning_rate = _require_finite("learning_rate", self.learning_rate)
        gamma = _require_finite("gamma", self.gamma)
        gae_lambda = _require_finite("gae_lambda", self.gae_lambda)
        clip_range = _require_finite("clip_range", self.clip_range)
        ent_coef = _require_finite("ent_coef", self.ent_coef)
        vf_coef = _require_finite("vf_coef", self.vf_coef)
        max_grad_norm = _require_finite("max_grad_norm", self.max_grad_norm)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be greater than 0 and at most 1")
        if not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if not 0 < clip_range <= 1:
            raise ValueError("clip_range must be greater than 0 and at most 1")
        if ent_coef < 0 or vf_coef < 0:
            raise ValueError("ent_coef and vf_coef cannot be negative")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if (
            not isinstance(self.device, str)
            or self.device not in SUPPORTED_TORCH_DEVICES
        ):
            raise ValueError("device must be one of: auto, cpu, cuda, mps")

    def with_runtime_overrides(
        self,
        *,
        seed: int | None = None,
        total_timesteps: int | None = None,
        device: str | None = None,
    ) -> "PPOConfig":
        """Return a validated configuration with explicit run overrides."""
        return replace(
            self,
            seed=self.seed if seed is None else seed,
            total_timesteps=(
                self.total_timesteps if total_timesteps is None else total_timesteps
            ),
            device=self.device if device is None else device,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete effective configuration as plain values."""
        return asdict(self)

    def model_kwargs(self, *, resolved_device: str | None = None) -> dict[str, object]:
        """Return SB3 parameters with AUTO already resolved by the trainer."""
        selected_device = self.device if resolved_device is None else resolved_device
        if selected_device == "auto":
            raise ValueError("auto device must be resolved before constructing PPO")
        if selected_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("resolved_device must be cpu, cuda, or mps")
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "device": selected_device,
        }
