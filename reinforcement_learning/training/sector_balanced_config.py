"""Frozen configuration for the repaired balanced-window sector method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, Sequence

from reinforcement_learning.environments.action_validity import (
    MASK_ACTION_VALIDITY_VERSION,
    MASK_MODE,
    PENALTY_ACTION_VALIDITY_VERSION,
    PENALTY_MODE,
    SECTOR_ACTION_VALIDITY_VERSION,
    normalize_invalid_action_mode,
)
from reinforcement_learning.environments.reward import SectorRewardConfig
from reinforcement_learning.sector_universe import (
    SECTOR_NORMALIZATION_POLICY_VERSION,
    SECTOR_TAXONOMY_VERSION,
    SECTOR_UNIVERSE_SCHEMA_VERSION,
)

from .recurrent_config import RecurrentPPOConfig
from .sector_balanced_windows import (
    DEFAULT_BALANCED_ROUNDS,
    DEFAULT_DATA_SCHEDULE_SEED,
    DEFAULT_WINDOW_TRANSITIONS,
    PREDECLARED_MODEL_SEEDS,
    SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
)
from .sector_recurrent_config import (
    COMMERCIAL_BANKS_SECTOR_ID,
    COMMERCIAL_BANKS_SECTOR_NAME,
)


BALANCED_SECTOR_TRAINER_VERSION = "recurrent_ppo_sector_balanced_v1"
BALANCED_SECTOR_CONFIG_VERSION = "sector_recurrent_balanced_config_v1"
METHODOLOGY_SMOKE_PURPOSE = "methodology_smoke"
PREDECLARED_RESEARCH_PURPOSE = "predeclared_fair_research"
SUPPORTED_EXECUTION_PURPOSES = (
    METHODOLOGY_SMOKE_PURPOSE,
    PREDECLARED_RESEARCH_PURPOSE,
)


def _canonical_symbols(values: Sequence[object]) -> tuple[str, ...]:
    symbols = tuple(str(value).strip().upper() for value in values)
    if not symbols or any(not value for value in symbols):
        raise ValueError("balanced sector config requires non-empty symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("balanced sector config symbols must be unique")
    return symbols


@dataclass(frozen=True)
class BalancedSectorRecurrentPPOConfig:
    """Immutable methodology identity for one scheduled sector run.

    ``model_seed`` affects optimization and environment stochasticity.
    ``data_schedule_seed`` independently fixes window locations and per-round
    order so model seeds 42/43/44 see the same data schedule.
    """

    config_version: str = BALANCED_SECTOR_CONFIG_VERSION
    trainer_version: str = BALANCED_SECTOR_TRAINER_VERSION
    sector_id: str = COMMERCIAL_BANKS_SECTOR_ID
    sector_name: str = COMMERCIAL_BANKS_SECTOR_NAME
    sector_universe_hash: str = ""
    taxonomy_version: str = SECTOR_TAXONOMY_VERSION
    manifest_version: str = SECTOR_UNIVERSE_SCHEMA_VERSION
    constituent_symbols: tuple[str, ...] = ()
    target_symbol: str | None = None
    target_excluded_from_pretraining: bool = False
    sampling_version: str = SECTOR_BALANCED_WINDOW_SAMPLING_VERSION
    window_transition_count: int = DEFAULT_WINDOW_TRANSITIONS
    balanced_rounds: int = DEFAULT_BALANCED_ROUNDS
    data_schedule_seed: int = DEFAULT_DATA_SCHEDULE_SEED
    experiment_seed_set: tuple[int, ...] = PREDECLARED_MODEL_SEEDS
    normalization_scope: str = "symbol"
    normalization_policy_version: str = SECTOR_NORMALIZATION_POLICY_VERSION
    observation_includes_symbol_identity: bool = False
    action_validity_version: str = SECTOR_ACTION_VALIDITY_VERSION
    invalid_action_mode: str = PENALTY_MODE
    invalid_action_mode_version: str = PENALTY_ACTION_VALIDITY_VERSION
    reward: SectorRewardConfig = field(default_factory=SectorRewardConfig)
    execution_purpose: str = METHODOLOGY_SMOKE_PURPOSE
    ppo: RecurrentPPOConfig = field(default_factory=RecurrentPPOConfig)

    def __post_init__(self) -> None:
        if self.config_version != BALANCED_SECTOR_CONFIG_VERSION:
            raise ValueError("balanced sector config version is incompatible")
        if self.trainer_version != BALANCED_SECTOR_TRAINER_VERSION:
            raise ValueError("balanced sector trainer version is incompatible")
        if (
            self.sector_id != COMMERCIAL_BANKS_SECTOR_ID
            or self.sector_name != COMMERCIAL_BANKS_SECTOR_NAME
        ):
            raise ValueError("6E.2 supports Commercial Banks only")
        if self.taxonomy_version != SECTOR_TAXONOMY_VERSION:
            raise ValueError("sector taxonomy version is incompatible")
        if self.manifest_version != SECTOR_UNIVERSE_SCHEMA_VERSION:
            raise ValueError("sector manifest version is incompatible")
        if self.sampling_version != SECTOR_BALANCED_WINDOW_SAMPLING_VERSION:
            raise ValueError("balanced sampling version is incompatible")
        if len(self.sector_universe_hash) != 64:
            raise ValueError("sector universe hash must be a SHA-256")
        try:
            int(self.sector_universe_hash, 16)
        except ValueError as exc:
            raise ValueError("sector universe hash must be hexadecimal") from exc
        symbols = _canonical_symbols(self.constituent_symbols)
        if symbols != self.constituent_symbols:
            raise ValueError("constituent symbols must already be canonical")
        for label, value in (
            ("window_transition_count", self.window_transition_count),
            ("balanced_rounds", self.balanced_rounds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.data_schedule_seed, bool)
            or not isinstance(self.data_schedule_seed, int)
            or self.data_schedule_seed < 0
        ):
            raise ValueError("data_schedule_seed must be a non-negative integer")
        if self.experiment_seed_set != PREDECLARED_MODEL_SEEDS:
            raise ValueError("future experiment seeds must remain (42, 43, 44)")
        if self.normalization_scope != "symbol":
            raise ValueError("6E.2 retains per-symbol TRAIN-fitted scaling")
        if self.normalization_policy_version != SECTOR_NORMALIZATION_POLICY_VERSION:
            raise ValueError("normalization policy version is incompatible")
        if self.observation_includes_symbol_identity:
            raise ValueError("6E.2 keeps symbol/sector identity out of observations")
        if self.action_validity_version != SECTOR_ACTION_VALIDITY_VERSION:
            raise ValueError("action validity semantic version is incompatible")
        mode = normalize_invalid_action_mode(self.invalid_action_mode)
        expected_mode_version = (
            PENALTY_ACTION_VALIDITY_VERSION
            if mode == PENALTY_MODE
            else MASK_ACTION_VALIDITY_VERSION
        )
        if self.invalid_action_mode_version != expected_mode_version:
            raise ValueError("invalid-action mode/version are inconsistent")
        if self.execution_purpose not in SUPPORTED_EXECUTION_PURPOSES:
            raise ValueError("unsupported balanced sector execution purpose")
        target = (
            str(self.target_symbol).strip().upper()
            if self.target_symbol is not None
            else None
        )
        if target != self.target_symbol:
            raise ValueError("target symbol must already be canonical")
        if target is None and self.target_excluded_from_pretraining:
            raise ValueError("target exclusion requires a target symbol")
        if target is not None:
            if not self.target_excluded_from_pretraining:
                raise ValueError("declared transfer target must be excluded")
            if target in symbols:
                raise ValueError("target appears in its own pretraining universe")
        expected_timesteps = (
            len(symbols) * self.balanced_rounds * self.window_transition_count
        )
        if self.ppo.total_timesteps != expected_timesteps:
            raise ValueError(
                "PPO total_timesteps must equal the exact scheduled exposure"
            )
        if self.ppo.seed not in self.experiment_seed_set:
            raise ValueError("model seed must come from the frozen seed set")
        if expected_timesteps % self.ppo.n_steps:
            raise ValueError(
                "scheduled transitions must divide exactly into PPO rollouts"
            )

    @property
    def expected_transitions_per_symbol(self) -> int:
        return self.balanced_rounds * self.window_transition_count

    @property
    def expected_total_transitions(self) -> int:
        return (
            len(self.constituent_symbols) * self.expected_transitions_per_symbol
        )

    @property
    def model_seed(self) -> int:
        return self.ppo.seed

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        constituent_symbols: Sequence[str] | None = None,
        target_symbol: str | None = None,
        rounds: int = 2,
        window_transition_count: int = DEFAULT_WINDOW_TRANSITIONS,
        data_schedule_seed: int = DEFAULT_DATA_SCHEDULE_SEED,
        model_seed: int = 42,
        n_steps: int = 128,
        batch_size: int = 64,
        device: str = "cpu",
        invalid_action_mode: str = PENALTY_MODE,
        execution_purpose: str = METHODOLOGY_SMOKE_PURPOSE,
    ) -> "BalancedSectorRecurrentPPOConfig":
        """Construct an exact schedule-bound config without starting training."""

        sector = manifest.get("sector")
        experiment = manifest.get("experiment_mode")
        if not isinstance(sector, Mapping) or not isinstance(experiment, Mapping):
            raise ValueError("sector manifest lacks identity metadata")
        approved = _canonical_symbols(
            tuple(experiment.get("pretraining_constituent_symbols", ()))
        )
        if constituent_symbols is None:
            selected = approved
        else:
            requested = _canonical_symbols(constituent_symbols)
            if not set(requested).issubset(approved):
                raise ValueError("config requested a non-manifest constituent")
            # Canonical manifest order avoids caller-order fingerprints.
            selected = tuple(symbol for symbol in approved if symbol in requested)
        target = str(target_symbol).strip().upper() if target_symbol else None
        if target is not None:
            if target not in approved:
                raise ValueError("leave-one-out target is not an approved constituent")
            selected = tuple(symbol for symbol in selected if symbol != target)
        expected_timesteps = len(selected) * rounds * window_transition_count
        base = RecurrentPPOConfig(
            seed=model_seed,
            total_timesteps=expected_timesteps,
            device=device,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        mode = normalize_invalid_action_mode(invalid_action_mode)
        return cls(
            sector_id=str(sector["sector_id"]),
            sector_name=str(sector["sector_name"]),
            sector_universe_hash=str(manifest["universe_hash"]),
            taxonomy_version=str(manifest["taxonomy_version"]),
            manifest_version=str(manifest["artifact_schema_version"]),
            constituent_symbols=selected,
            target_symbol=target,
            target_excluded_from_pretraining=target is not None,
            balanced_rounds=rounds,
            window_transition_count=window_transition_count,
            data_schedule_seed=data_schedule_seed,
            invalid_action_mode=mode,
            invalid_action_mode_version=(
                PENALTY_ACTION_VALIDITY_VERSION
                if mode == PENALTY_MODE
                else MASK_ACTION_VALIDITY_VERSION
            ),
            execution_purpose=execution_purpose,
            ppo=base,
        )

    @classmethod
    def predeclared_research_from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        model_seed: int,
        device: str = "cpu",
    ) -> "BalancedSectorRecurrentPPOConfig":
        """Build—but never authorize—the frozen 19 x 20 x 512 run."""

        return cls.from_manifest(
            manifest,
            rounds=DEFAULT_BALANCED_ROUNDS,
            window_transition_count=DEFAULT_WINDOW_TRANSITIONS,
            data_schedule_seed=DEFAULT_DATA_SCHEDULE_SEED,
            model_seed=model_seed,
            n_steps=512,
            device=device,
            execution_purpose=PREDECLARED_RESEARCH_PURPOSE,
        )

    def with_model_seed(self, model_seed: int) -> "BalancedSectorRecurrentPPOConfig":
        return replace(
            self,
            ppo=self.ppo.with_runtime_overrides(seed=model_seed),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["constituent_symbols"] = list(self.constituent_symbols)
        payload["experiment_seed_set"] = list(self.experiment_seed_set)
        payload["reward"] = self.reward.to_metadata()
        payload["ppo"] = self.ppo.to_dict()
        return payload


__all__ = (
    "BALANCED_SECTOR_CONFIG_VERSION",
    "BALANCED_SECTOR_TRAINER_VERSION",
    "BalancedSectorRecurrentPPOConfig",
    "METHODOLOGY_SMOKE_PURPOSE",
    "PREDECLARED_RESEARCH_PURPOSE",
    "SUPPORTED_EXECUTION_PURPOSES",
)
