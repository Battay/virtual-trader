"""Structured results for recurrent training, validation, and benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from reinforcement_learning.evaluation.results import (
    StrategyEvaluationResult,
    ValidationComparisonResult,
)

from .devices import torch_devices_equivalent
from .results import PPOTrainingDiagnostics


@dataclass(frozen=True)
class RecurrentPPOTrainingResult:
    symbol: str
    algorithm: str
    policy: str
    trainer_version: str
    recurrent_contract_version: str
    environment_version: str
    feature_version: str
    config: Mapping[str, object]
    seed: int
    requested_timesteps: int
    actual_timesteps: int
    training_rows: int
    training_start: str | None
    training_end: str | None
    duration_seconds: float
    requested_device: str
    resolved_device: str | None
    device: str
    observation_shape: tuple[int, ...] | None
    observation_features: tuple[str, ...]
    lstm_hidden_size: int
    n_lstm_layers: int
    shared_lstm: bool
    enable_critic_lstm: bool
    parameter_count: int
    source_recurrent_contract_path: str | None
    source_recurrent_contract_sha256: str | None
    source_episode_boundaries_path: str | None
    source_episode_boundaries_sha256: str | None
    status: str
    started_at: str
    completed_at: str
    message: str
    error: str | None = None
    model: Any | None = field(default=None, repr=False, compare=False)
    training_diagnostics: PPOTrainingDiagnostics | None = None
    first_episode_start: bool | None = None
    rollout_boundaries_observed: int = 0
    rollout_continuity_checks: int = 0
    rollout_continuity_verified: bool = False
    environment_episode_resets: int = 0
    rollout_start_episode_flags: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm != "RecurrentPPO" or self.policy != "MlpLstmPolicy":
            raise ValueError("invalid recurrent algorithm or policy")
        if self.status not in {"completed", "interrupted", "failed"}:
            raise ValueError("invalid recurrent training status")
        if min(
            self.actual_timesteps,
            self.training_rows,
            self.parameter_count,
            self.rollout_boundaries_observed,
            self.rollout_continuity_checks,
            self.environment_episode_resets,
        ) < 0 or self.duration_seconds < 0:
            raise ValueError("recurrent training counts/duration cannot be negative")
        if self.status != "completed" and self.model is not None:
            raise ValueError("failed/interrupted recurrent results cannot expose a model")
        if self.status == "completed":
            if self.model is None or self.resolved_device is None:
                raise ValueError("completed recurrent training requires model/device")
            if not torch_devices_equivalent(self.device, self.resolved_device):
                raise ValueError("actual recurrent device differs from resolution")
            required = (
                self.source_recurrent_contract_path,
                self.source_recurrent_contract_sha256,
                self.source_episode_boundaries_path,
                self.source_episode_boundaries_sha256,
            )
            if not all(required) or not self.observation_features:
                raise ValueError("completed recurrent training lacks provenance")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            if item.name == "model":
                continue
            value = getattr(self, item.name)
            if item.name == "config":
                payload[item.name] = dict(value)
            elif item.name == "training_diagnostics":
                payload[item.name] = value.to_dict() if value else None
            else:
                payload[item.name] = value
        return payload


@dataclass(frozen=True)
class RecurrentValidationResult:
    symbol: str
    evaluation_partition: str
    validation_start: str
    validation_end: str
    validation_rows: int
    recurrent_contract_version: str
    environment_version: str
    feature_version: str
    deterministic_seed: int
    strategy_result: StrategyEvaluationResult
    parameter_hash_before: str
    parameter_hash_after: str
    model_timesteps_before: int
    model_timesteps_after: int
    first_episode_start: bool
    episode_reset_count: int
    recurrent_state_steps: int
    final_state_available: bool
    train_state_supplied: bool

    def __post_init__(self) -> None:
        if self.evaluation_partition != "validation" or self.validation_rows < 2:
            raise ValueError("recurrent evaluation must use complete validation")
        if not self.first_episode_start or self.episode_reset_count != 1:
            raise ValueError("validation must begin with exactly one recurrent reset")
        if self.train_state_supplied:
            raise ValueError("TRAIN hidden state cannot enter validation")

    @property
    def model_parameters_unchanged(self) -> bool:
        return (
            self.parameter_hash_before == self.parameter_hash_after
            and self.model_timesteps_before == self.model_timesteps_after
        )

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        payload = asdict(self)
        payload["strategy_result"] = self.strategy_result.to_dict(
            include_history=include_history
        )
        payload["model_parameters_unchanged"] = self.model_parameters_unchanged
        return payload


@dataclass(frozen=True)
class RecurrentMLPComparisonResult:
    symbol: str
    recurrent: RecurrentValidationResult
    mlp_comparison: ValidationComparisonResult
    recurrent_training: Mapping[str, object]
    mlp_training: Mapping[str, object]
    same_training_budget: bool
    same_validation_partition: bool
    test_evaluated: bool = False

    def __post_init__(self) -> None:
        if self.test_evaluated:
            raise ValueError("TEST cannot be evaluated in the 6C comparison")
        if not self.same_training_budget or not self.same_validation_partition:
            raise ValueError("comparison inputs are not equivalent")

    @property
    def strategies(self) -> tuple[StrategyEvaluationResult, ...]:
        return (
            self.recurrent.strategy_result,
            self.mlp_comparison.ppo,
            self.mlp_comparison.buy_and_hold,
            self.mlp_comparison.always_hold,
            self.mlp_comparison.random,
        )

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "recurrent": self.recurrent.to_dict(include_history=include_history),
            "mlp_comparison": self.mlp_comparison.to_dict(
                include_history=include_history
            ),
            "recurrent_training": dict(self.recurrent_training),
            "mlp_training": dict(self.mlp_training),
            "same_training_budget": self.same_training_budget,
            "same_validation_partition": self.same_validation_partition,
            "test_evaluated": self.test_evaluated,
        }


@dataclass(frozen=True)
class RecurrentDeviceRun:
    device: str
    status: str
    actual_device: str | None
    requested_timesteps: int
    actual_timesteps: int
    duration_seconds: float | None
    timesteps_per_second: float | None
    parameter_count: int
    warning: str | None = None
    error: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.actual_device is not None


@dataclass(frozen=True)
class RecurrentDeviceBenchmarkResult:
    symbol: str
    seed: int
    warmup_timesteps: int
    benchmark_timesteps: int
    cpu: RecurrentDeviceRun
    mps: RecurrentDeviceRun
    speedup_cpu_over_mps: float | None
    recommended_device: str
    recommendation_reason: str
    isolated_subprocesses: bool
    test_evaluated: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporaryRecurrentPersistenceResult:
    model_path: Path
    metadata_path: Path
    policy_class: str
    saved_parameter_count: int
    loaded_parameter_count: int
    deterministic_action_match: bool
    recurrent_state_match: bool
    architecture_match: bool
    registry_touched: bool
