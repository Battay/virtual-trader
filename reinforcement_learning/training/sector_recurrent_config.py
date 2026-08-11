"""Versioned Commercial Banks sector RecurrentPPO configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Mapping

from reinforcement_learning.environments.sector_training_env import (
    EQUAL_SYMBOL_EPISODE_SAMPLING,
)
from reinforcement_learning.sector_universe import (
    SECTOR_NORMALIZATION_POLICY_VERSION,
    SECTOR_TAXONOMY_VERSION,
    SECTOR_UNIVERSE_SCHEMA_VERSION,
)

from .recurrent_config import RecurrentPPOConfig


SECTOR_RECURRENT_TRAINER_VERSION = "recurrent_ppo_sector_v1"
COMMERCIAL_BANKS_SECTOR_ID = "commercial_banks"
COMMERCIAL_BANKS_SECTOR_NAME = "Commercial Banks"


@dataclass(frozen=True)
class SectorRecurrentPPOConfig:
    """Complete immutable identity for one general sector foundation run."""

    trainer_version: str = SECTOR_RECURRENT_TRAINER_VERSION
    sector_id: str = COMMERCIAL_BANKS_SECTOR_ID
    sector_name: str = COMMERCIAL_BANKS_SECTOR_NAME
    sector_universe_hash: str = ""
    taxonomy_version: str = SECTOR_TAXONOMY_VERSION
    manifest_version: str = SECTOR_UNIVERSE_SCHEMA_VERSION
    constituent_symbols: tuple[str, ...] = ()
    sampling_strategy: str = EQUAL_SYMBOL_EPISODE_SAMPLING
    normalization_scope: str = "symbol"
    normalization_policy_version: str = SECTOR_NORMALIZATION_POLICY_VERSION
    episode_strategy: str = "full_partition_per_symbol"
    symbol_reset_behavior: str = "full_portfolio_and_recurrent_reset"
    ppo: RecurrentPPOConfig = field(default_factory=RecurrentPPOConfig)

    def __post_init__(self) -> None:
        if self.trainer_version != SECTOR_RECURRENT_TRAINER_VERSION:
            raise ValueError("sector trainer version is incompatible")
        if (
            self.sector_id != COMMERCIAL_BANKS_SECTOR_ID
            or self.sector_name != COMMERCIAL_BANKS_SECTOR_NAME
        ):
            raise ValueError("6E supports Commercial Banks only")
        if self.taxonomy_version != SECTOR_TAXONOMY_VERSION:
            raise ValueError("sector taxonomy version is incompatible")
        if self.manifest_version != SECTOR_UNIVERSE_SCHEMA_VERSION:
            raise ValueError("sector manifest version is incompatible")
        if self.sampling_strategy != EQUAL_SYMBOL_EPISODE_SAMPLING:
            raise ValueError("6E supports equal-symbol episode sampling only")
        if self.normalization_scope != "symbol":
            raise ValueError("6E requires per-symbol normalization")
        if self.normalization_policy_version != SECTOR_NORMALIZATION_POLICY_VERSION:
            raise ValueError("sector normalization policy is incompatible")
        if self.episode_strategy != "full_partition_per_symbol":
            raise ValueError("sector episodes must use full symbol partitions")
        if self.symbol_reset_behavior != "full_portfolio_and_recurrent_reset":
            raise ValueError("sector symbol reset behavior is incompatible")
        if not self.constituent_symbols or any(not value for value in self.constituent_symbols):
            raise ValueError("sector config requires constituent symbols")
        if len(set(self.constituent_symbols)) != len(self.constituent_symbols):
            raise ValueError("sector config constituents must be unique")
        if len(self.sector_universe_hash) != 64:
            raise ValueError("sector config requires a SHA-256 universe hash")
        try:
            int(self.sector_universe_hash, 16)
        except ValueError as exc:
            raise ValueError("sector universe hash must be hexadecimal") from exc

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        seed: int = 42,
        total_timesteps: int = 25_000,
        device: str = "cpu",
    ) -> "SectorRecurrentPPOConfig":
        sector = manifest.get("sector")
        experiment = manifest.get("experiment_mode")
        normalization = manifest.get("normalization")
        sampling = manifest.get("sampling")
        if not all(isinstance(value, Mapping) for value in (
            sector, experiment, normalization, sampling
        )):
            raise ValueError("sector manifest is missing configuration metadata")
        return cls(
            sector_id=str(sector["sector_id"]),
            sector_name=str(sector["sector_name"]),
            sector_universe_hash=str(manifest["universe_hash"]),
            taxonomy_version=str(manifest["taxonomy_version"]),
            manifest_version=str(manifest["artifact_schema_version"]),
            constituent_symbols=tuple(experiment["pretraining_constituent_symbols"]),
            sampling_strategy=str(sampling["policy_version"]),
            normalization_scope="symbol",
            normalization_policy_version=str(normalization["policy_version"]),
            ppo=RecurrentPPOConfig(
                seed=seed,
                total_timesteps=total_timesteps,
                device=device,
            ),
        )

    def with_runtime_overrides(
        self,
        *,
        seed: int | None = None,
        total_timesteps: int | None = None,
        device: str | None = None,
    ) -> "SectorRecurrentPPOConfig":
        return replace(
            self,
            ppo=self.ppo.with_runtime_overrides(
                seed=seed,
                total_timesteps=total_timesteps,
                device=device,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["constituent_symbols"] = list(self.constituent_symbols)
        payload["ppo"] = self.ppo.to_dict()
        return payload


__all__ = (
    "COMMERCIAL_BANKS_SECTOR_ID",
    "COMMERCIAL_BANKS_SECTOR_NAME",
    "SECTOR_RECURRENT_TRAINER_VERSION",
    "SectorRecurrentPPOConfig",
)
