"""Structured, serialization-safe PPO trainer results."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping


TRAINING_RESULT_STATUSES = frozenset({"completed", "interrupted", "failed"})


@dataclass(frozen=True)
class PPOTrainingResult:
    """Outcome and provenance for one in-memory single-symbol PPO run."""

    symbol: str
    algorithm: str
    ppo_config_version: str
    ppo_config: Mapping[str, object]
    environment_version: str
    rl_contract_version: str
    feature_version: str
    source_rl_contract_path: str | None
    source_rl_contract_sha256: str | None
    source_observation_scaler_path: str | None
    source_observation_scaler_sha256: str | None
    source_observation_scaler_metadata_path: str | None
    source_observation_scaler_metadata_sha256: str | None
    observation_features: tuple[str, ...]
    seed: int
    requested_timesteps: int
    actual_timesteps: int
    training_start: str | None
    training_end: str | None
    training_rows: int
    duration_seconds: float
    device: str
    observation_shape: tuple[int, ...] | None
    status: str
    started_at: str
    completed_at: str
    message: str
    error: str | None = None
    output_directory: str | None = None
    model: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.algorithm != "PPO":
            raise ValueError("algorithm must be 'PPO'")
        if self.status not in TRAINING_RESULT_STATUSES:
            raise ValueError(f"Unsupported training result status: {self.status}")
        if self.actual_timesteps < 0 or self.training_rows < 0:
            raise ValueError("training counts cannot be negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if self.status != "completed" and self.model is not None:
            raise ValueError("failed or interrupted results cannot expose a model")
        if self.status == "completed":
            required_provenance = (
                self.source_rl_contract_path,
                self.source_rl_contract_sha256,
                self.source_observation_scaler_path,
                self.source_observation_scaler_sha256,
                self.source_observation_scaler_metadata_path,
                self.source_observation_scaler_metadata_sha256,
            )
            if not all(required_provenance) or not self.observation_features:
                raise ValueError(
                    "completed training results require complete RL source provenance"
                )

    @property
    def succeeded(self) -> bool:
        """Return whether training completed successfully."""
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata without the in-memory PPO model."""
        payload: dict[str, object] = {}
        for item in fields(self):
            if item.name == "model":
                continue
            value = getattr(self, item.name)
            payload[item.name] = dict(value) if item.name == "ppo_config" else value
        return payload
