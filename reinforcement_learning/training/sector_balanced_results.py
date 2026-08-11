"""Structured outputs for balanced-window sector methodology runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .results import PPOTrainingDiagnostics


@dataclass(frozen=True)
class BalancedSectorTrainingResult:
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
    target_symbol: str | None
    target_excluded_from_pretraining: bool
    sampling_version: str
    reward_version: str
    action_validity_version: str
    invalid_action_mode: str
    invalid_action_mode_version: str
    normalization_scope: str
    observation_shape: tuple[int, ...] | None
    observation_features: tuple[str, ...]
    symbol_identity_in_observation: bool
    model_seed: int
    data_schedule_seed: int
    experiment_seed_set: tuple[int, ...]
    window_transition_count: int
    source_rows_per_window: int
    balanced_rounds: int
    scheduled_window_count: int
    schedule_digest: str
    expected_transitions_per_symbol: int
    expected_total_scheduled_transitions: int
    actual_scheduled_transitions: int
    optimizer_requested_timesteps: int
    optimizer_actual_timesteps: int
    rollout_padding_timesteps: int
    duration_seconds: float
    requested_device: str
    resolved_device: str | None
    device: str | None
    parameter_count: int
    expected_windows_by_symbol: Mapping[str, int]
    completed_windows_by_symbol: Mapping[str, int]
    expected_transitions_by_symbol: Mapping[str, int]
    actual_transitions_by_symbol: Mapping[str, int]
    contribution_percentages: Mapping[str, float]
    coverage_statistics: tuple[Mapping[str, object], ...]
    termination_reasons: Mapping[str, int]
    reset_snapshots: tuple[Mapping[str, object], ...]
    passive_post_schedule_resets: int
    first_episode_start: bool | None
    window_boundary_episode_start_verified: bool
    portfolio_reset_verified: bool
    recurrent_reset_verified: bool
    rollout_boundaries_observed: int
    rollout_continuity_checks: int
    rollout_continuity_verified: bool
    environment_episode_resets: int
    reward_action_diagnostics: Mapping[str, object]
    collapse_diagnostics: Mapping[str, object]
    training_diagnostics: PPOTrainingDiagnostics | None
    periodic_training_diagnostics: tuple[Mapping[str, object], ...]
    reproducibility_fingerprint: Mapping[str, object]
    config: Mapping[str, object]
    warnings: tuple[str, ...]
    status: str
    started_at: str
    completed_at: str
    message: str
    error: str | None = None
    model: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "interrupted", "failed"}:
            raise ValueError("unsupported balanced sector training status")
        if self.duration_seconds < 0:
            raise ValueError("duration cannot be negative")
        counters = (
            self.expected_total_scheduled_transitions,
            self.actual_scheduled_transitions,
            self.optimizer_requested_timesteps,
            self.optimizer_actual_timesteps,
            self.rollout_padding_timesteps,
        )
        if any(value < 0 for value in counters):
            raise ValueError("balanced training counters cannot be negative")
        expected_symbols = set(self.constituent_symbols)
        for mapping in (
            self.expected_windows_by_symbol,
            self.completed_windows_by_symbol,
            self.expected_transitions_by_symbol,
            self.actual_transitions_by_symbol,
            self.contribution_percentages,
        ):
            if set(mapping) != expected_symbols:
                raise ValueError("per-symbol result population is inconsistent")
        if sum(self.actual_transitions_by_symbol.values()) != self.actual_scheduled_transitions:
            raise ValueError("actual per-symbol transitions do not reconcile")
        if self.target_symbol is not None:
            if not self.target_excluded_from_pretraining:
                raise ValueError("result target exclusion is inconsistent")
            if self.target_symbol in expected_symbols:
                raise ValueError("target appears in its own training result")
        if self.status == "completed":
            if self.model is None:
                raise ValueError("completed balanced training requires a model")
            if self.actual_scheduled_transitions != self.expected_total_scheduled_transitions:
                raise ValueError("completed run did not consume the exact schedule")
            if self.optimizer_actual_timesteps != self.optimizer_requested_timesteps:
                raise ValueError("completed run contains PPO rollout padding")
            if self.rollout_padding_timesteps != 0:
                raise ValueError("completed run cannot relabel padded timesteps")
            if self.passive_post_schedule_resets != 1:
                raise ValueError(
                    "completed vector run requires one non-counted passive reset"
                )
            if self.first_episode_start is not True:
                raise ValueError("balanced recurrent training must start a new episode")
            if not all(
                (
                    self.window_boundary_episode_start_verified,
                    self.portfolio_reset_verified,
                    self.recurrent_reset_verified,
                )
            ):
                raise ValueError("completed run did not prove state isolation")
            if self.expected_transitions_by_symbol != self.actual_transitions_by_symbol:
                raise ValueError("completed run exposure is unequal")
            if self.expected_windows_by_symbol != self.completed_windows_by_symbol:
                raise ValueError("completed run window counts are unequal")
        elif self.model is not None:
            raise ValueError("failed/interrupted run cannot retain a model")

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


__all__ = ("BalancedSectorTrainingResult",)
