"""Structured validation-evaluation and baseline-comparison results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd


def _json_safe(value):
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class StrategyEvaluationResult:
    """History and metrics for one policy on a fixed environment partition."""

    strategy: str
    history: pd.DataFrame
    metrics: Mapping[str, object]
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        object.__setattr__(self, "history", self.history.copy(deep=True))
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "strategy": self.strategy,
            "metrics": _json_safe(self.metrics),
            "duration_seconds": self.duration_seconds,
        }
        if include_history:
            payload["history"] = _json_safe(self.history.to_dict(orient="records"))
        return payload


@dataclass(frozen=True)
class PPOValidationResult:
    """Deterministic PPO result and provenance for validation only."""

    symbol: str
    evaluation_partition: str
    validation_start: str
    validation_end: str
    validation_rows: int
    environment_version: str
    rl_contract_version: str
    feature_version: str
    source_rl_contract_path: str
    source_rl_contract_sha256: str
    source_observation_scaler_path: str
    source_observation_scaler_sha256: str
    source_observation_scaler_metadata_path: str
    source_observation_scaler_metadata_sha256: str
    observation_features: tuple[str, ...]
    deterministic_seed: int
    environment_config: Mapping[str, object]
    strategy_result: StrategyEvaluationResult
    parameter_hash_before: str
    parameter_hash_after: str
    model_timesteps_before: int
    model_timesteps_after: int

    def __post_init__(self) -> None:
        if self.evaluation_partition != "validation":
            raise ValueError("PPO evaluation partition must be validation")
        if self.validation_rows < 2:
            raise ValueError("validation evaluation requires at least two rows")

    @property
    def model_parameters_unchanged(self) -> bool:
        return (
            self.parameter_hash_before == self.parameter_hash_after
            and self.model_timesteps_before == self.model_timesteps_after
        )

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "evaluation_partition": self.evaluation_partition,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_rows": self.validation_rows,
            "environment_version": self.environment_version,
            "rl_contract_version": self.rl_contract_version,
            "feature_version": self.feature_version,
            "source_rl_contract_path": self.source_rl_contract_path,
            "source_rl_contract_sha256": self.source_rl_contract_sha256,
            "source_observation_scaler_path": self.source_observation_scaler_path,
            "source_observation_scaler_sha256": (
                self.source_observation_scaler_sha256
            ),
            "source_observation_scaler_metadata_path": (
                self.source_observation_scaler_metadata_path
            ),
            "source_observation_scaler_metadata_sha256": (
                self.source_observation_scaler_metadata_sha256
            ),
            "observation_features": list(self.observation_features),
            "deterministic_seed": self.deterministic_seed,
            "environment_config": _json_safe(self.environment_config),
            "strategy_result": self.strategy_result.to_dict(
                include_history=include_history
            ),
            "parameter_hash_before": self.parameter_hash_before,
            "parameter_hash_after": self.parameter_hash_after,
            "model_timesteps_before": self.model_timesteps_before,
            "model_timesteps_after": self.model_timesteps_after,
            "model_parameters_unchanged": self.model_parameters_unchanged,
        }


@dataclass(frozen=True)
class CandidateValidationDecision:
    """Conservative validation decision that never promotes a model."""

    status: str
    passed: bool
    reasons: tuple[str, ...]
    criteria_version: str
    thresholds: Mapping[str, object]

    def __post_init__(self) -> None:
        allowed = {
            "validation_pass",
            "validation_fail",
            "insufficient_validation_data",
            "evaluation_error",
        }
        if self.status not in allowed:
            raise ValueError(f"Unsupported candidate validation status: {self.status}")
        if self.passed != (self.status == "validation_pass"):
            raise ValueError("passed must be true only for validation_pass")
        if not self.reasons:
            raise ValueError("candidate validation decision requires a reason")

    def to_dict(self) -> dict[str, object]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ValidationComparisonResult:
    """PPO and three baselines evaluated on one identical validation frame."""

    symbol: str
    evaluation_partition: str
    validation_start: str
    validation_end: str
    validation_rows: int
    environment_version: str
    rl_contract_version: str
    feature_version: str
    source_rl_contract_path: str
    source_rl_contract_sha256: str
    source_observation_scaler_path: str
    source_observation_scaler_sha256: str
    source_observation_scaler_metadata_path: str
    source_observation_scaler_metadata_sha256: str
    observation_features: tuple[str, ...]
    initial_cash: float
    commission_rate: float
    slippage_rate: float
    environment_config: Mapping[str, object]
    ppo_training_metadata: Mapping[str, object]
    ppo_parameter_hash_before: str
    ppo_parameter_hash_after: str
    ppo_model_timesteps_before: int
    ppo_model_timesteps_after: int
    ppo: StrategyEvaluationResult
    buy_and_hold: StrategyEvaluationResult
    always_hold: StrategyEvaluationResult
    random: StrategyEvaluationResult
    ppo_return_difference_vs_buy_and_hold: float
    ppo_sharpe_difference_vs_buy_and_hold: float | None
    ppo_max_drawdown_difference_vs_buy_and_hold: float
    deterministic_seed: int
    random_seed: int
    evaluation_duration_seconds: float
    candidate_decision: CandidateValidationDecision
    warnings: tuple[str, ...]
    status: str = "completed"

    def __post_init__(self) -> None:
        if self.evaluation_partition != "validation":
            raise ValueError("comparison partition must be validation")
        if self.validation_rows < 2:
            raise ValueError("comparison requires at least two validation rows")
        if self.evaluation_duration_seconds < 0:
            raise ValueError("evaluation duration cannot be negative")
        if self.status != "completed":
            raise ValueError("completed comparison results must use completed status")

    @property
    def ppo_model_unchanged(self) -> bool:
        return (
            self.ppo_parameter_hash_before == self.ppo_parameter_hash_after
            and self.ppo_model_timesteps_before == self.ppo_model_timesteps_after
        )

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "evaluation_partition": self.evaluation_partition,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_rows": self.validation_rows,
            "environment_version": self.environment_version,
            "rl_contract_version": self.rl_contract_version,
            "feature_version": self.feature_version,
            "source_rl_contract_path": self.source_rl_contract_path,
            "source_rl_contract_sha256": self.source_rl_contract_sha256,
            "source_observation_scaler_path": self.source_observation_scaler_path,
            "source_observation_scaler_sha256": (
                self.source_observation_scaler_sha256
            ),
            "source_observation_scaler_metadata_path": (
                self.source_observation_scaler_metadata_path
            ),
            "source_observation_scaler_metadata_sha256": (
                self.source_observation_scaler_metadata_sha256
            ),
            "observation_features": list(self.observation_features),
            "initial_cash": self.initial_cash,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "environment_config": _json_safe(self.environment_config),
            "ppo_training_metadata": _json_safe(self.ppo_training_metadata),
            "ppo_parameter_hash_before": self.ppo_parameter_hash_before,
            "ppo_parameter_hash_after": self.ppo_parameter_hash_after,
            "ppo_model_timesteps_before": self.ppo_model_timesteps_before,
            "ppo_model_timesteps_after": self.ppo_model_timesteps_after,
            "ppo_model_unchanged": self.ppo_model_unchanged,
            "ppo": self.ppo.to_dict(include_history=include_history),
            "buy_and_hold": self.buy_and_hold.to_dict(
                include_history=include_history
            ),
            "always_hold": self.always_hold.to_dict(
                include_history=include_history
            ),
            "random": self.random.to_dict(include_history=include_history),
            "ppo_return_difference_vs_buy_and_hold": (
                self.ppo_return_difference_vs_buy_and_hold
            ),
            "ppo_sharpe_difference_vs_buy_and_hold": (
                self.ppo_sharpe_difference_vs_buy_and_hold
            ),
            "ppo_max_drawdown_difference_vs_buy_and_hold": (
                self.ppo_max_drawdown_difference_vs_buy_and_hold
            ),
            "deterministic_seed": self.deterministic_seed,
            "random_seed": self.random_seed,
            "evaluation_duration_seconds": self.evaluation_duration_seconds,
            "candidate_decision": self.candidate_decision.to_dict(),
            "warnings": list(self.warnings),
            "status": self.status,
        }
