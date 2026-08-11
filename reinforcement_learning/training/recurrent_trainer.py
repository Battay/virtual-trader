"""Production-safe single-symbol RecurrentPPO trainer core."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import numpy as np
from sb3_contrib import RecurrentPPO

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.environments import SingleSymbolTradingEnv
from reinforcement_learning.environments.validation import validate_environment
from reinforcement_learning.history_policy import HistoryClass
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    RecurrentDataContractError,
    load_recurrent_partition,
)

from .callbacks import ProgressHandler
from .devices import (
    TorchDeviceResolution,
    resolve_torch_device,
    synchronize_torch_device,
    verify_sb3_model_device,
)
from .ppo_trainer import (
    EXPECTED_OBSERVATION_SHAPE,
    _seed_everything,
    _validate_output_directory,
    create_training_vector_environment,
)
from .recurrent_callbacks import RecurrentProgressCallback
from .recurrent_config import RecurrentPPOConfig
from .recurrent_results import RecurrentPPOTrainingResult
from .results import PPOTrainingDiagnostics


LOGGER = logging.getLogger(__name__)
RECURRENT_TRAINER_VERSION = "recurrent_ppo_single_symbol_v1"
MAX_RECURRENT_SMOKE_TIMESTEPS = 1_024


class RecurrentPPOTrainerError(RuntimeError):
    """Raised internally when recurrent training would violate its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_model_parameters(model: object) -> int:
    """Return the exact number of trainable and non-trainable policy values."""

    return sum(int(parameter.numel()) for parameter in model.policy.parameters())


def train_recurrent_single_symbol(
    symbol: str,
    *,
    config: RecurrentPPOConfig | None = None,
    seed: int | None = None,
    total_timesteps: int | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    progress_callback: ProgressHandler | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    smoke_test: bool = False,
) -> RecurrentPPOTrainingResult:
    """Train RecurrentPPO from the canonical recurrent TRAIN partition only."""

    effective = (config or RecurrentPPOConfig()).with_runtime_overrides(
        seed=seed,
        total_timesteps=total_timesteps,
        device=device,
    )
    symbol_text = str(symbol).strip()
    started_at = _utc_now()
    started_clock = time.perf_counter()
    vector_environment = None
    model: RecurrentPPO | None = None
    resolution: TorchDeviceResolution | None = None
    actual_device: str | None = None
    actual_timesteps = 0
    training_rows = 0
    training_start: str | None = None
    training_end: str | None = None
    observation_shape: tuple[int, ...] | None = None
    observation_features: tuple[str, ...] = ()
    recurrent_contract_version = ""
    environment_version = ""
    feature_version = ""
    contract_path: Path | None = None
    boundaries_path: Path | None = None
    contract_sha256: str | None = None
    boundaries_sha256: str | None = None
    parameter_count = 0
    diagnostics: PPOTrainingDiagnostics | None = None
    callback: RecurrentProgressCallback | None = None

    def finish(
        status: str,
        *,
        message: str,
        error: str | None = None,
        trained_model: RecurrentPPO | None = None,
    ) -> RecurrentPPOTrainingResult:
        return RecurrentPPOTrainingResult(
            symbol=symbol_text,
            algorithm="RecurrentPPO",
            policy=effective.policy,
            trainer_version=RECURRENT_TRAINER_VERSION,
            recurrent_contract_version=recurrent_contract_version,
            environment_version=environment_version,
            feature_version=feature_version,
            config=effective.to_dict(),
            seed=effective.seed,
            requested_timesteps=effective.total_timesteps,
            actual_timesteps=actual_timesteps,
            training_rows=training_rows,
            training_start=training_start,
            training_end=training_end,
            duration_seconds=max(0.0, time.perf_counter() - started_clock),
            requested_device=effective.device,
            resolved_device=resolution.resolved_device if resolution else None,
            device=actual_device or (resolution.resolved_device if resolution else effective.device),
            observation_shape=observation_shape,
            observation_features=observation_features,
            lstm_hidden_size=effective.lstm_hidden_size,
            n_lstm_layers=effective.n_lstm_layers,
            shared_lstm=effective.shared_lstm,
            enable_critic_lstm=effective.enable_critic_lstm,
            parameter_count=parameter_count,
            source_recurrent_contract_path=(str(contract_path) if contract_path else None),
            source_recurrent_contract_sha256=(
                contract_sha256
            ),
            source_episode_boundaries_path=(
                str(boundaries_path) if boundaries_path else None
            ),
            source_episode_boundaries_sha256=(
                boundaries_sha256
            ),
            status=status,
            started_at=started_at,
            completed_at=_utc_now(),
            message=message,
            error=error,
            model=trained_model,
            training_diagnostics=diagnostics if status == "completed" else None,
            first_episode_start=(callback.first_episode_start if callback else None),
            rollout_boundaries_observed=(
                callback.rollout_boundaries_observed if callback else 0
            ),
            rollout_continuity_checks=(
                callback.rollout_continuity_checks if callback else 0
            ),
            rollout_continuity_verified=(
                callback.rollout_continuity_verified if callback else False
            ),
            environment_episode_resets=(
                callback.environment_episode_resets if callback else 0
            ),
            rollout_start_episode_flags=(
                tuple(callback.rollout_start_episode_flags) if callback else ()
            ),
        )

    try:
        if not symbol_text:
            raise RecurrentPPOTrainerError("symbol is required")
        if smoke_test and effective.total_timesteps > MAX_RECURRENT_SMOKE_TIMESTEPS:
            raise RecurrentPPOTrainerError(
                f"smoke_test is capped at {MAX_RECURRENT_SMOKE_TIMESTEPS} timesteps"
            )
        _validate_output_directory(output_dir)
        resolution = resolve_torch_device(effective.device)

        # Sole market-frame load. This API cannot load TEST.
        loaded = load_recurrent_partition(
            symbol_text,
            "train",
            splits_dir=Path(splits_dir),
        )
        if loaded.partition != "train":
            raise RecurrentPPOTrainerError("recurrent loader returned non-TRAIN data")
        metadata = loaded.metadata
        if metadata.recurrent_contract_version != RL_RECURRENT_PARTITION_SCHEMA_VERSION:
            raise RecurrentPPOTrainerError("recurrent contract version is incompatible")
        if metadata.history.history_class is not HistoryClass.MATURE:
            raise RecurrentPPOTrainerError(
                "independent recurrent training requires Mature history"
            )
        if not metadata.history.independent_recurrent_ready:
            raise RecurrentPPOTrainerError(
                "symbol is not approved for independent recurrent training"
            )
        if metadata.episode_strategy != "full_partition":
            raise RecurrentPPOTrainerError("unsupported recurrent episode strategy")
        if not bool(loaded.episode_start[0]) or int(loaded.episode_start.sum()) != 1:
            raise RecurrentPPOTrainerError("TRAIN reset mask is incompatible")
        recurrent_contract_version = metadata.recurrent_contract_version
        environment_version = metadata.environment_version
        feature_version = metadata.feature_version
        observation_features = metadata.observation_features
        contract_path = metadata.contract_path
        boundaries_path = metadata.boundaries_path
        contract_sha256 = sha256_file(contract_path)
        boundaries_sha256 = sha256_file(boundaries_path)
        training_data = loaded.data
        training_rows = len(training_data)
        training_start = metadata.train.start
        training_end = metadata.train.end

        environment = SingleSymbolTradingEnv(training_data)
        try:
            validation = validate_environment(environment)
        finally:
            environment.close()
        observation_shape = validation.observation_shape
        if not validation.valid or observation_shape != EXPECTED_OBSERVATION_SHAPE:
            raise RecurrentPPOTrainerError(
                "recurrent environment validation failed: "
                + "; ".join(validation.errors)
            )
        _seed_everything(effective.seed, resolved_device=resolution.resolved_device)
        vector_environment = create_training_vector_environment(
            training_data,
            seed=effective.seed,
        )
        model = RecurrentPPO(
            effective.policy,
            vector_environment,
            verbose=0,
            **effective.model_kwargs(resolved_device=resolution.resolved_device),
        )
        actual_device = verify_sb3_model_device(model, resolution)
        parameter_count = count_model_parameters(model)
        callback = RecurrentProgressCallback(
            symbol=symbol_text,
            requested_timesteps=effective.total_timesteps,
            interval_steps=max(
                effective.n_steps,
                effective.total_timesteps // 10,
            ),
            handler=progress_callback,
        )
        LOGGER.info(
            "recurrent_training_start symbol=%s partition=train rows=%s "
            "start=%s end=%s config=%s parameters=%s",
            symbol_text,
            training_rows,
            training_start,
            training_end,
            json.dumps(effective.to_dict(), sort_keys=True),
            parameter_count,
        )
        model.learn(
            total_timesteps=effective.total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        synchronize_torch_device(resolution.resolved_device)
        actual_device = verify_sb3_model_device(model, resolution)
        actual_timesteps = int(model.num_timesteps)
        if callback.cancel_requested:
            return finish(
                "interrupted",
                message="Recurrent training stopped cooperatively; no model retained.",
            )
        diagnostics = callback.training_diagnostics
        return finish(
            "completed",
            message=(
                f"Completed {actual_timesteps} TRAIN-only RecurrentPPO timesteps."
            ),
            trained_model=model,
        )
    except KeyboardInterrupt:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        return finish(
            "interrupted",
            message="Recurrent training interrupted; no model retained.",
        )
    except Exception as exc:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        LOGGER.exception("recurrent_training_failed symbol=%s", symbol_text)
        return finish(
            "failed",
            message="Recurrent training failed safely; no model retained.",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if vector_environment is not None:
            vector_environment.close()
