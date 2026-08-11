"""TRAIN-only multi-symbol RecurrentPPO sector foundation trainer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from data_pipeline.src.config import PROCESSED_SPLITS_DIR, PROJECT_ROOT
from feature_engineering.schemas import FEATURE_VERSION
from reinforcement_learning.environments import SingleSymbolEnvConfig
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.environments.sector_training_env import SectorTrainingEnv
from reinforcement_learning.history_policy import HistoryClass
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    LoadedRecurrentPartition,
    load_recurrent_partition,
)
from reinforcement_learning.sector_universe import (
    SECTOR_TAXONOMY_VERSION,
    SECTOR_UNIVERSE_SCHEMA_VERSION,
    deterministic_universe_hash,
)

from .callbacks import ProgressHandler
from .devices import resolve_torch_device, synchronize_torch_device, verify_sb3_model_device
from .ppo_trainer import EXPECTED_OBSERVATION_SHAPE, _seed_everything
from .recurrent_callbacks import RecurrentProgressCallback
from .recurrent_trainer import count_model_parameters
from .sector_recurrent_config import (
    COMMERCIAL_BANKS_SECTOR_ID,
    SECTOR_RECURRENT_TRAINER_VERSION,
    SectorRecurrentPPOConfig,
)
from .sector_recurrent_results import SectorRecurrentTrainingResult


COMMERCIAL_BANKS_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_universes"
    / COMMERCIAL_BANKS_SECTOR_ID
    / "universe_manifest.json"
)
MAX_SECTOR_SMOKE_TIMESTEPS = 2_048


class SectorRecurrentTrainerError(RuntimeError):
    """Raised when a sector run cannot satisfy its immutable contract."""


@dataclass(frozen=True)
class LoadedSectorTrainingUniverse:
    manifest_path: Path
    manifest: Mapping[str, object]
    universe_hash: str
    symbols: tuple[str, ...]
    train_data: Mapping[str, pd.DataFrame]
    train_rows: Mapping[str, int]
    total_train_rows: int
    recurrent_contract_version: str
    feature_version: str
    environment_version: str
    observation_features: tuple[str, ...]
    observation_shape: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SectorRecurrentTrainerError(f"Could not read sector manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise SectorRecurrentTrainerError("sector manifest must be a JSON object")
    if value.get("artifact_schema_version") != SECTOR_UNIVERSE_SCHEMA_VERSION:
        raise SectorRecurrentTrainerError("sector manifest version is incompatible")
    identity = value.get("deterministic_identity")
    if not isinstance(identity, Mapping):
        raise SectorRecurrentTrainerError("sector manifest lacks deterministic identity")
    if deterministic_universe_hash(identity) != value.get("universe_hash"):
        raise SectorRecurrentTrainerError("sector universe hash is stale")
    return value


def _portable_source_path(value: object, *, project_root: Path) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise SectorRecurrentTrainerError("sector source path is not portable")
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise SectorRecurrentTrainerError("sector source escaped the project") from exc
    if not path.is_file():
        raise SectorRecurrentTrainerError(f"sector source artifact is missing: {relative}")
    return path


def load_sector_training_universe(
    manifest_path: Path = COMMERCIAL_BANKS_MANIFEST_PATH,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
) -> LoadedSectorTrainingUniverse:
    """Fail-closed audit and load of every canonical constituent TRAIN frame."""

    manifest = _read_manifest(Path(manifest_path))
    sector = manifest.get("sector")
    experiment = manifest.get("experiment_mode")
    compatibility = manifest.get("compatibility")
    normalization = manifest.get("normalization")
    data_access = manifest.get("data_access")
    if not all(isinstance(value, Mapping) for value in (
        sector, experiment, compatibility, normalization, data_access
    )):
        raise SectorRecurrentTrainerError("sector manifest metadata is incomplete")
    if sector.get("sector_id") != COMMERCIAL_BANKS_SECTOR_ID:
        raise SectorRecurrentTrainerError("6E accepts Commercial Banks only")
    if manifest.get("taxonomy_version") != SECTOR_TAXONOMY_VERSION:
        raise SectorRecurrentTrainerError("sector taxonomy version is incompatible")
    if experiment.get("mode") != "standard_sector_pretraining":
        raise SectorRecurrentTrainerError("6E requires the standard foundation universe")
    if experiment.get("target_symbol") is not None:
        raise SectorRecurrentTrainerError("general sector pretraining cannot declare a target")
    if normalization.get("implemented_scope") != "per_symbol_train_fitted_scalers":
        raise SectorRecurrentTrainerError("sector normalization scope is incompatible")
    if data_access.get("training_partition") != "train_only":
        raise SectorRecurrentTrainerError("sector manifest is not TRAIN-only")
    if data_access.get("validation_frames_referenced") is not False:
        raise SectorRecurrentTrainerError("validation cannot enter sector pretraining")
    if data_access.get("test_frame_access") != "prohibited":
        raise SectorRecurrentTrainerError("TEST sealing is incompatible")

    constituents = manifest.get("constituents")
    if not isinstance(constituents, list) or not constituents:
        raise SectorRecurrentTrainerError("sector manifest has no constituents")
    symbols = tuple(str(item.get("symbol", "")).strip() for item in constituents)
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise SectorRecurrentTrainerError("sector constituent symbols are invalid")
    expected_symbols = tuple(experiment.get("pretraining_constituent_symbols", ()))
    if symbols != expected_symbols:
        raise SectorRecurrentTrainerError("manifest constituent ordering is inconsistent")

    frames: dict[str, pd.DataFrame] = {}
    rows: dict[str, int] = {}
    errors: list[str] = []
    root = Path(project_root).resolve()
    expected_shape = (
        len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
    )
    for item in constituents:
        symbol = str(item["symbol"])
        try:
            if item.get("history_class") != "MATURE":
                raise SectorRecurrentTrainerError("history class is not Mature")
            if item.get("eligible_at_cutoff") is not True:
                raise SectorRecurrentTrainerError("constituent is not cutoff eligible")
            if item.get("sector_verified_current") is not True:
                raise SectorRecurrentTrainerError("current sector evidence is unverified")
            source_fields = (
                ("recurrent_contract_path", "recurrent_contract_sha256"),
                ("train_rl_path", "train_rl_sha256"),
                ("scaler_path", "scaler_sha256"),
                ("scaler_metadata_path", "scaler_metadata_sha256"),
            )
            sources: dict[str, Path] = {}
            for path_field, hash_field in source_fields:
                path = _portable_source_path(item[path_field], project_root=root)
                if sha256_file(path) != item[hash_field]:
                    raise SectorRecurrentTrainerError(f"{path_field} hash is stale")
                sources[path_field] = path
            loaded = partition_loader(symbol, "train", splits_dir=Path(splits_dir))
            metadata = loaded.metadata
            if loaded.partition != "train":
                raise SectorRecurrentTrainerError("partition loader returned non-TRAIN data")
            if metadata.history.history_class is not HistoryClass.MATURE:
                raise SectorRecurrentTrainerError("recurrent history is not Mature")
            if not metadata.history.independent_recurrent_ready:
                raise SectorRecurrentTrainerError("recurrent contract is not independently ready")
            if metadata.recurrent_contract_version != RL_RECURRENT_PARTITION_SCHEMA_VERSION:
                raise SectorRecurrentTrainerError("recurrent contract version differs")
            if metadata.feature_version != FEATURE_VERSION:
                raise SectorRecurrentTrainerError("feature version differs")
            if metadata.environment_version != ENVIRONMENT_VERSION:
                raise SectorRecurrentTrainerError("environment version differs")
            if metadata.observation_features != DEFAULT_OBSERVATION_FEATURES:
                raise SectorRecurrentTrainerError("observation ordering differs")
            if metadata.observation_shape != expected_shape:
                raise SectorRecurrentTrainerError("observation shape differs")
            if metadata.normalization_scope != "symbol" or metadata.scaler_fit_partition != "train":
                raise SectorRecurrentTrainerError("symbol TRAIN scaler provenance differs")
            if sha256_file(metadata.contract_path) != item["recurrent_contract_sha256"]:
                raise SectorRecurrentTrainerError("loaded recurrent contract differs")
            if sha256_file(loaded.source_artifact_path) != item["train_rl_sha256"]:
                raise SectorRecurrentTrainerError("loaded TRAIN artifact differs")
            if (
                len(loaded.data) != int(item["rows_available_at_cutoff"])
                or metadata.train.start != item["train_start"]
                or metadata.train.end != item["last_train_date"]
            ):
                raise SectorRecurrentTrainerError("TRAIN rows/date range differs")
            if not bool(loaded.episode_start[0]) or int(loaded.episode_start.sum()) != 1:
                raise SectorRecurrentTrainerError("TRAIN reset mask differs")
            observation = loaded.data.loc[:, DEFAULT_OBSERVATION_FEATURES].to_numpy(float)
            execution = loaded.data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(float)
            if not np.isfinite(observation).all() or not np.isfinite(execution).all():
                raise SectorRecurrentTrainerError("TRAIN values contain non-finite data")
            if (execution[:, :4] <= 0).any() or (execution[:, 4] < 0).any():
                raise SectorRecurrentTrainerError("execution OHLCV is invalid")
            frames[symbol] = loaded.data.copy(deep=True)
            rows[symbol] = len(loaded.data)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    if errors:
        raise SectorRecurrentTrainerError(
            "Commercial Banks manifest audit failed; no symbol was dropped: "
            + " | ".join(errors)
        )
    return LoadedSectorTrainingUniverse(
        manifest_path=Path(manifest_path).resolve(),
        manifest=manifest,
        universe_hash=str(manifest["universe_hash"]),
        symbols=symbols,
        train_data=frames,
        train_rows=rows,
        total_train_rows=sum(rows.values()),
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        observation_shape=expected_shape,
    )


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sb3_contrib": version("sb3-contrib"),
        "stable_baselines3": version("stable-baselines3"),
        "torch": version("torch"),
        "gymnasium": version("gymnasium"),
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _assert_finite_training_state(model: RecurrentPPO) -> None:
    if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
        raise SectorRecurrentTrainerError("model parameters became non-finite")
    values = getattr(getattr(model, "logger", None), "name_to_value", {})
    for key in (
        "train/approx_kl", "train/clip_fraction", "train/entropy_loss",
        "train/explained_variance", "train/policy_gradient_loss",
        "train/value_loss", "train/learning_rate",
    ):
        value = values.get(key) if isinstance(values, Mapping) else None
        if isinstance(value, Real) and not math.isfinite(float(value)):
            raise SectorRecurrentTrainerError(f"non-finite training diagnostic: {key}")


def train_sector_recurrent_ppo(
    *,
    manifest_path: Path = COMMERCIAL_BANKS_MANIFEST_PATH,
    config: SectorRecurrentPPOConfig | None = None,
    seed: int | None = None,
    total_timesteps: int | None = None,
    device: str | None = None,
    progress_callback: ProgressHandler | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
    smoke_test: bool = False,
) -> SectorRecurrentTrainingResult:
    """Train one general Commercial Banks foundation model from TRAIN only."""

    universe = load_sector_training_universe(
        Path(manifest_path), splits_dir=Path(splits_dir), project_root=Path(project_root),
        partition_loader=partition_loader,
    )
    effective = config or SectorRecurrentPPOConfig.from_manifest(
        universe.manifest, seed=seed or 42, total_timesteps=total_timesteps or 25_000,
        device=device or "cpu",
    )
    effective = effective.with_runtime_overrides(
        seed=seed, total_timesteps=total_timesteps, device=device
    )
    if effective.sector_universe_hash != universe.universe_hash:
        raise SectorRecurrentTrainerError("configuration universe hash differs")
    if effective.constituent_symbols != universe.symbols:
        raise SectorRecurrentTrainerError("configuration constituents differ")
    if smoke_test and effective.ppo.total_timesteps > MAX_SECTOR_SMOKE_TIMESTEPS:
        raise SectorRecurrentTrainerError(
            f"sector smoke is capped at {MAX_SECTOR_SMOKE_TIMESTEPS} timesteps"
        )

    started_at = _utc_now()
    started_clock = time.perf_counter()
    resolution = None
    model: RecurrentPPO | None = None
    vector: DummyVecEnv | None = None
    sector_env: SectorTrainingEnv | None = None
    callback: RecurrentProgressCallback | None = None
    actual_device: str | None = None
    actual_timesteps = 0
    diagnostics = None

    def finish(status: str, message: str, error: str | None = None):
        elapsed = max(0.0, time.perf_counter() - started_clock)
        timestep_counts = {
            symbol: int((sector_env.timesteps_by_symbol if sector_env else {}).get(symbol, 0))
            for symbol in universe.symbols
        }
        started_counts = {
            symbol: int((sector_env.episode_counts_started if sector_env else {}).get(symbol, 0))
            for symbol in universe.symbols
        }
        completed_counts = {
            symbol: int((sector_env.episode_counts_completed if sector_env else {}).get(symbol, 0))
            for symbol in universe.symbols
        }
        sampled = tuple(symbol for symbol in universe.symbols if timestep_counts[symbol] > 0)
        never = tuple(symbol for symbol in universe.symbols if timestep_counts[symbol] == 0)
        contributions = {
            symbol: (100.0 * count / actual_timesteps if actual_timesteps else 0.0)
            for symbol, count in timestep_counts.items()
        }
        warnings: list[str] = []
        if never:
            warnings.append(
                "Not all sector constituents received training exposure: " + ", ".join(never)
            )
        positive = [count for count in timestep_counts.values() if count]
        if positive and max(positive) > 2 * min(positive):
            warnings.append(
                "Equal episode selection produced unequal timestep exposure because full episode lengths differ."
            )
        config_payload = effective.to_dict()
        fingerprint = {
            "git_commit": _git_commit(Path(project_root)),
            "dependencies": _dependency_versions(),
            "taxonomy_version": effective.taxonomy_version,
            "sector_universe_hash": universe.universe_hash,
            "recurrent_contract_version": universe.recurrent_contract_version,
            "feature_version": universe.feature_version,
            "environment_version": universe.environment_version,
            "trainer_version": effective.trainer_version,
            "config": config_payload,
            "seed": effective.ppo.seed,
            "requested_device": effective.ppo.device,
            "resolved_device": resolution.resolved_device if resolution else None,
            "sampling_strategy": effective.sampling_strategy,
            "normalization_scope": effective.normalization_scope,
            "requested_timesteps": effective.ppo.total_timesteps,
            "actual_timesteps": actual_timesteps,
            "constituent_symbols": list(universe.symbols),
            "episode_counts_started": started_counts,
            "episode_counts_completed": completed_counts,
            "timesteps_by_symbol": timestep_counts,
        }
        completed_episodes = sum(completed_counts.values())
        boundary_resets = callback.environment_episode_resets if callback else 0
        boundary_verified = (
            status == "completed" and boundary_resets == completed_episodes
        )
        portfolio_verified = bool(
            sector_env
            and sector_env.reset_snapshots
            and all(
                snapshot["episode_start"] is True
                and snapshot["cash"] == SingleSymbolEnvConfig().initial_cash
                and snapshot["shares_held"] == 0
                and snapshot["realized_profit_loss"] == 0.0
                and snapshot["drawdown"] == 0.0
                for snapshot in sector_env.reset_snapshots
            )
        )
        return SectorRecurrentTrainingResult(
            sector_id=effective.sector_id,
            sector_name=effective.sector_name,
            trainer_version=effective.trainer_version,
            algorithm="RecurrentPPO",
            policy=effective.ppo.policy,
            taxonomy_version=effective.taxonomy_version,
            sector_universe_hash=universe.universe_hash,
            recurrent_contract_version=universe.recurrent_contract_version,
            environment_version=universe.environment_version,
            feature_version=universe.feature_version,
            constituent_symbols=universe.symbols,
            sampling_strategy=effective.sampling_strategy,
            normalization_scope=effective.normalization_scope,
            config=config_payload,
            seed=effective.ppo.seed,
            requested_timesteps=effective.ppo.total_timesteps,
            actual_timesteps=actual_timesteps,
            duration_seconds=elapsed,
            requested_device=effective.ppo.device,
            resolved_device=resolution.resolved_device if resolution else None,
            device=actual_device,
            observation_shape=universe.observation_shape,
            observation_features=universe.observation_features,
            parameter_count=count_model_parameters(model) if model is not None else 0,
            total_referenced_train_rows=universe.total_train_rows,
            total_episodes_started=sum(started_counts.values()),
            total_episodes_completed=completed_episodes,
            episode_counts_started=started_counts,
            episode_counts_completed=completed_counts,
            timesteps_by_symbol=timestep_counts,
            timestep_contribution_percentages=contributions,
            constituents_sampled=sampled,
            constituents_never_sampled=never,
            sampling_sequence=(sector_env.sampling_sequence if sector_env else ()),
            sampling_sequence_digest=(
                sector_env.sampling_sequence_digest if sector_env else hashlib.sha256(b"[]").hexdigest()
            ),
            termination_reasons=dict(sector_env.termination_reasons if sector_env else {}),
            reset_snapshots=tuple(sector_env.reset_snapshots if sector_env else ()),
            first_episode_start=(callback.first_episode_start if callback else None),
            symbol_boundary_episode_start_verified=boundary_verified,
            portfolio_reset_verified=portfolio_verified,
            rollout_boundaries_observed=(callback.rollout_boundaries_observed if callback else 0),
            rollout_continuity_checks=(callback.rollout_continuity_checks if callback else 0),
            rollout_continuity_verified=(callback.rollout_continuity_verified if callback else False),
            environment_episode_resets=(callback.environment_episode_resets if callback else 0),
            training_diagnostics=diagnostics if status == "completed" else None,
            reproducibility_fingerprint=fingerprint,
            warnings=tuple(warnings),
            status=status,
            started_at=started_at,
            completed_at=_utc_now(),
            message=message,
            error=error,
            model=model if status == "completed" else None,
        )

    try:
        resolution = resolve_torch_device(effective.ppo.device)
        _seed_everything(effective.ppo.seed, resolved_device=resolution.resolved_device)
        environment_config = SingleSymbolEnvConfig()
        sector_env = SectorTrainingEnv(
            universe.train_data,
            universe_hash=universe.universe_hash,
            seed=effective.ppo.seed,
            config=environment_config,
        )
        if sector_env.observation_space.shape != EXPECTED_OBSERVATION_SHAPE:
            raise SectorRecurrentTrainerError("sector observation shape is incompatible")
        if tuple(sector_env.observation_feature_names) != (
            *DEFAULT_OBSERVATION_FEATURES,
            *DYNAMIC_PORTFOLIO_FEATURES,
        ):
            raise SectorRecurrentTrainerError("sector observation identity is incompatible")
        sector_env.action_space.seed(effective.ppo.seed)
        vector = DummyVecEnv([lambda: Monitor(sector_env)])
        vector.seed(effective.ppo.seed)
        model = RecurrentPPO(
            effective.ppo.policy,
            vector,
            verbose=0,
            **effective.ppo.model_kwargs(resolved_device=resolution.resolved_device),
        )
        actual_device = verify_sb3_model_device(model, resolution)
        callback = RecurrentProgressCallback(
            symbol=effective.sector_name,
            requested_timesteps=effective.ppo.total_timesteps,
            interval_steps=max(effective.ppo.n_steps, effective.ppo.total_timesteps // 10),
            handler=progress_callback,
        )
        model.learn(
            total_timesteps=effective.ppo.total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        synchronize_torch_device(resolution.resolved_device)
        actual_device = verify_sb3_model_device(model, resolution)
        actual_timesteps = int(model.num_timesteps)
        _assert_finite_training_state(model)
        diagnostics = callback.training_diagnostics
        if callback.cancel_requested:
            model = None
            return finish("interrupted", "Sector training stopped cooperatively; no model retained.")
        return finish(
            "completed",
            f"Completed {actual_timesteps} TRAIN-only sector RecurrentPPO timesteps.",
        )
    except KeyboardInterrupt:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        model = None
        return finish("interrupted", "Sector training interrupted; no model retained.")
    except Exception as exc:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        model = None
        return finish(
            "failed", "Sector training failed safely; no model retained.",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if vector is not None:
            vector.close()


__all__ = (
    "COMMERCIAL_BANKS_MANIFEST_PATH",
    "LoadedSectorTrainingUniverse",
    "MAX_SECTOR_SMOKE_TIMESTEPS",
    "SectorRecurrentTrainerError",
    "load_sector_training_universe",
    "train_sector_recurrent_ppo",
)
