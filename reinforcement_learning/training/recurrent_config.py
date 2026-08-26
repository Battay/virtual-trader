"""Explicit configuration for the single-symbol RecurrentPPO baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real

from torch import nn

from .devices import SUPPORTED_TORCH_DEVICES


RECURRENT_PPO_CONFIG_VERSION = "recurrent_ppo_single_symbol_v1"
SUPPORTED_ACTIVATIONS = {"Tanh": nn.Tanh, "ReLU": nn.ReLU}


def _integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """Frozen, reproducible RecurrentPPO + MlpLstmPolicy configuration."""

    config_version: str = RECURRENT_PPO_CONFIG_VERSION
    algorithm: str = "RecurrentPPO"
    policy: str = "MlpLstmPolicy"
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
    lstm_hidden_size: int = 64
    n_lstm_layers: int = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True
    share_features_extractor: bool = True
    net_arch: tuple[int, ...] = (64,)
    activation_fn: str = "Tanh"
    ortho_init: bool = True

    def __post_init__(self) -> None:
        if self.config_version != RECURRENT_PPO_CONFIG_VERSION:
            raise ValueError(
                f"config_version must be {RECURRENT_PPO_CONFIG_VERSION!r}"
            )
        if self.algorithm != "RecurrentPPO" or self.policy != "MlpLstmPolicy":
            raise ValueError(
                "6C supports only RecurrentPPO with MlpLstmPolicy"
            )
        for name, value, minimum in (
            ("n_steps", self.n_steps, 2),
            ("batch_size", self.batch_size, 2),
            ("n_epochs", self.n_epochs, 1),
            ("seed", self.seed, 0),
            ("total_timesteps", self.total_timesteps, 1),
            ("lstm_hidden_size", self.lstm_hidden_size, 1),
            ("n_lstm_layers", self.n_lstm_layers, 1),
        ):
            _integer(name, value, minimum)
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps for one environment")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("gamma", self.gamma),
            ("gae_lambda", self.gae_lambda),
            ("clip_range", self.clip_range),
            ("ent_coef", self.ent_coef),
            ("vf_coef", self.vf_coef),
            ("max_grad_norm", self.max_grad_norm),
        ):
            _finite(name, value)
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma/gae_lambda are outside supported bounds")
        if not 0 < self.clip_range <= 1:
            raise ValueError("clip_range must be greater than 0 and at most 1")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("ent_coef and vf_coef cannot be negative")
        if not isinstance(self.device, str) or self.device not in SUPPORTED_TORCH_DEVICES:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if not all(isinstance(value, bool) for value in (
            self.shared_lstm,
            self.enable_critic_lstm,
            self.share_features_extractor,
            self.ortho_init,
        )):
            raise ValueError("recurrent policy flags must be boolean")
        if self.shared_lstm and self.enable_critic_lstm:
            raise ValueError(
                "shared_lstm and enable_critic_lstm cannot both be true"
            )
        if self.shared_lstm and not self.share_features_extractor:
            raise ValueError(
                "a shared LSTM requires a shared features extractor"
            )
        if (
            not isinstance(self.net_arch, tuple)
            or not self.net_arch
            or any(
                isinstance(width, bool)
                or not isinstance(width, Integral)
                or width < 1
                for width in self.net_arch
            )
        ):
            raise ValueError("net_arch must contain positive integer widths")
        if self.activation_fn not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                "activation_fn must be one of: "
                + ", ".join(sorted(SUPPORTED_ACTIVATIONS))
            )

    def with_runtime_overrides(
        self,
        *,
        seed: int | None = None,
        total_timesteps: int | None = None,
        device: str | None = None,
    ) -> "RecurrentPPOConfig":
        return replace(
            self,
            seed=self.seed if seed is None else seed,
            total_timesteps=(
                self.total_timesteps if total_timesteps is None else total_timesteps
            ),
            device=self.device if device is None else device,
        )

    def policy_kwargs(self) -> dict[str, object]:
        """Return every policy choice explicitly, including SB3 defaults used."""

        return {
            "lstm_hidden_size": self.lstm_hidden_size,
            "n_lstm_layers": self.n_lstm_layers,
            "shared_lstm": self.shared_lstm,
            "enable_critic_lstm": self.enable_critic_lstm,
            "share_features_extractor": self.share_features_extractor,
            "net_arch": {
                "pi": list(self.net_arch),
                "vf": list(self.net_arch),
            },
            "activation_fn": SUPPORTED_ACTIVATIONS[self.activation_fn],
            "ortho_init": self.ortho_init,
        }

    def model_kwargs(self, *, resolved_device: str) -> dict[str, object]:
        if resolved_device not in {"cpu", "cuda", "mps"}:
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
            "device": resolved_device,
            "policy_kwargs": self.policy_kwargs(),
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["net_arch"] = list(self.net_arch)
        payload["policy_kwargs"] = {
            **self.policy_kwargs(),
            "activation_fn": self.activation_fn,
        }
        return payload
