"""Structured sector pretraining, validation, and persistence results."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from reinforcement_learning.evaluation.results import StrategyEvaluationResult

from .results import PPOTrainingDiagnostics


@dataclass(frozen=True)
class SectorRecurrentTrainingResult:
    sector_id: str
    sector_name: str
    trainer_version: str
    algorithm: str
    policy: str
    taxonomy_version: str
    sector_universe_hash: str
    recurrent_contract_version: str
    environment_version: str
    feature_version: str
    constituent_symbols: tuple[str, ...]
    sampling_strategy: str
    normalization_scope: str
    config: Mapping[str, object]
    seed: int
    requested_timesteps: int
    actual_timesteps: int
    duration_seconds: float
    requested_device: str
    resolved_device: str | None
    device: str | None
    observation_shape: tuple[int, ...] | None
    observation_features: tuple[str, ...]
    parameter_count: int
    total_referenced_train_rows: int
    total_episodes_started: int
    total_episodes_completed: int
    episode_counts_started: Mapping[str, int]
    episode_counts_completed: Mapping[str, int]
    timesteps_by_symbol: Mapping[str, int]
    timestep_contribution_percentages: Mapping[str, float]
    constituents_sampled: tuple[str, ...]
    constituents_never_sampled: tuple[str, ...]
    sampling_sequence: tuple[str, ...]
    sampling_sequence_digest: str
    termination_reasons: Mapping[str, int]
    reset_snapshots: tuple[Mapping[str, object], ...]
    first_episode_start: bool | None
    symbol_boundary_episode_start_verified: bool
    portfolio_reset_verified: bool
    rollout_boundaries_observed: int
    rollout_continuity_checks: int
    rollout_continuity_verified: bool
    environment_episode_resets: int
    training_diagnostics: PPOTrainingDiagnostics | None
    reproducibility_fingerprint: Mapping[str, object]
    warnings: tuple[str, ...]
    status: str
    started_at: str
    completed_at: str
    message: str
    error: str | None = None
    model: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "interrupted", "failed"}:
            raise ValueError("unsupported sector training status")
        if self.status == "completed" and self.model is None:
            raise ValueError("completed sector training requires an in-memory model")
        if self.status != "completed" and self.model is not None:
            raise ValueError("failed/interrupted sector training cannot retain a model")
        if self.actual_timesteps < 0 or self.duration_seconds < 0:
            raise ValueError("sector training counters cannot be negative")
        if set(self.timesteps_by_symbol).difference(self.constituent_symbols):
            raise ValueError("training exposure contains a non-constituent symbol")
        if sum(self.timesteps_by_symbol.values()) != self.actual_timesteps:
            raise ValueError("per-symbol timesteps do not reconcile")
        if self.status == "completed" and self.first_episode_start is not True:
            raise ValueError("sector recurrent training must start a new episode")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            if item.name == "model":
                continue
            value = getattr(self, item.name)
            if item.name == "training_diagnostics":
                payload[item.name] = value.to_dict() if value else None
            elif isinstance(value, Mapping):
                payload[item.name] = dict(value)
            elif isinstance(value, tuple):
                payload[item.name] = list(value)
            else:
                payload[item.name] = value
        return payload


@dataclass(frozen=True)
class SectorSymbolValidationResult:
    symbol: str
    validation_start: str
    validation_end: str
    validation_rows: int
    initial_capital: float
    ppo: StrategyEvaluationResult
    buy_and_hold: StrategyEvaluationResult
    always_hold: StrategyEvaluationResult
    random: StrategyEvaluationResult
    first_episode_start: bool
    recurrent_state_steps: int
    portfolio_reset_verified: bool
    model_parameters_unchanged: bool
    action_counts: Mapping[str, int]
    action_percentages: Mapping[str, float]
    action_pattern_digest: str

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_rows": self.validation_rows,
            "initial_capital": self.initial_capital,
            "first_episode_start": self.first_episode_start,
            "recurrent_state_steps": self.recurrent_state_steps,
            "portfolio_reset_verified": self.portfolio_reset_verified,
            "model_parameters_unchanged": self.model_parameters_unchanged,
            "action_counts": dict(self.action_counts),
            "action_percentages": dict(self.action_percentages),
            "action_pattern_digest": self.action_pattern_digest,
            "ppo": self.ppo.to_dict(include_history=include_history),
            "buy_and_hold": self.buy_and_hold.to_dict(include_history=include_history),
            "always_hold": self.always_hold.to_dict(include_history=include_history),
            "random": self.random.to_dict(include_history=include_history),
        }


@dataclass(frozen=True)
class SectorValidationResult:
    sector_id: str
    sector_name: str
    universe_hash: str
    evaluation_partition: str
    independent_symbol_capital: bool
    initial_capital_per_symbol: float
    symbols_requested: tuple[str, ...]
    symbol_results: tuple[SectorSymbolValidationResult, ...]
    failures: Mapping[str, str]
    aggregate_metrics: Mapping[str, object]
    collapse_diagnostics: Mapping[str, object]
    parameter_hash_before: str
    parameter_hash_after: str
    model_timesteps_before: int
    model_timesteps_after: int
    duration_seconds: float
    test_evaluated: bool = False

    def __post_init__(self) -> None:
        if self.evaluation_partition != "validation" or self.test_evaluated:
            raise ValueError("sector evaluation must be validation-only")
        if not self.independent_symbol_capital:
            raise ValueError("sector validation capital must be independent")

    @property
    def model_parameters_unchanged(self) -> bool:
        return (
            self.parameter_hash_before == self.parameter_hash_after
            and self.model_timesteps_before == self.model_timesteps_after
        )

    def to_dict(self, *, include_history: bool = False) -> dict[str, object]:
        return {
            "sector_id": self.sector_id,
            "sector_name": self.sector_name,
            "universe_hash": self.universe_hash,
            "evaluation_partition": self.evaluation_partition,
            "independent_symbol_capital": self.independent_symbol_capital,
            "initial_capital_per_symbol": self.initial_capital_per_symbol,
            "symbols_requested": list(self.symbols_requested),
            "symbol_results": [
                result.to_dict(include_history=include_history)
                for result in self.symbol_results
            ],
            "failures": dict(self.failures),
            "aggregate_metrics": dict(self.aggregate_metrics),
            "collapse_diagnostics": dict(self.collapse_diagnostics),
            "parameter_hash_before": self.parameter_hash_before,
            "parameter_hash_after": self.parameter_hash_after,
            "model_timesteps_before": self.model_timesteps_before,
            "model_timesteps_after": self.model_timesteps_after,
            "model_parameters_unchanged": self.model_parameters_unchanged,
            "duration_seconds": self.duration_seconds,
            "test_evaluated": self.test_evaluated,
        }


@dataclass(frozen=True)
class TemporarySectorPersistenceResult:
    model_path: Path
    metadata_path: Path
    metadata_sha256: str
    model_sha256: str
    deterministic_action_match: bool
    recurrent_state_match: bool
    metadata_integrity_verified: bool
    registry_touched: bool


__all__ = (
    "SectorRecurrentTrainingResult",
    "SectorSymbolValidationResult",
    "SectorValidationResult",
    "TemporarySectorPersistenceResult",
)
