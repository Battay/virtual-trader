"""Validation-only evaluation with explicit recurrent hidden-state handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import torch

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.environments.config import DEFAULT_OBSERVATION_FEATURES
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    LoadedRecurrentPartition,
    RecurrentDataContractError,
    load_recurrent_partition,
)
from reinforcement_learning.training.recurrent_results import (
    RecurrentPPOTrainingResult,
    RecurrentValidationResult,
)

from .metrics import calculate_episode_metrics
from .ppo_evaluator import (
    ValidationEvaluationError,
    _validate_full_history,
    policy_parameter_hash,
)
from .results import StrategyEvaluationResult


@dataclass(frozen=True)
class RecurrentValidationContext:
    symbol: str
    data: pd.DataFrame
    start: str
    end: str
    rows: int
    recurrent_contract_version: str
    environment_version: str
    feature_version: str
    observation_features: tuple[str, ...]
    contract_path: Path
    contract_sha256: str


def load_recurrent_validation_context(
    symbol: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> RecurrentValidationContext:
    """Load only the canonical recurrent VALIDATION partition."""

    try:
        loaded: LoadedRecurrentPartition = load_recurrent_partition(
            str(symbol).strip(),
            "validation",
            splits_dir=Path(splits_dir),
        )
    except RecurrentDataContractError as exc:
        raise ValidationEvaluationError(
            f"Could not load recurrent validation: {exc}"
        ) from exc
    if loaded.partition != "validation":
        raise ValidationEvaluationError("recurrent loader returned non-validation data")
    if not bool(loaded.episode_start[0]) or int(loaded.episode_start.sum()) != 1:
        raise ValidationEvaluationError("validation reset mask is incompatible")
    metadata = loaded.metadata
    return RecurrentValidationContext(
        symbol=loaded.symbol,
        data=loaded.data.copy(deep=True),
        start=metadata.validation.start,
        end=metadata.validation.end,
        rows=metadata.validation.rows,
        recurrent_contract_version=metadata.recurrent_contract_version,
        environment_version=metadata.environment_version,
        feature_version=metadata.feature_version,
        observation_features=metadata.observation_features,
        contract_path=metadata.contract_path,
        contract_sha256=sha256_file(metadata.contract_path),
    )


def evaluate_recurrent_on_validation(
    model: RecurrentPPO,
    symbol: str,
    *,
    trainer_result: RecurrentPPOTrainingResult,
    seed: int = 42,
    environment_config: SingleSymbolEnvConfig | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> RecurrentValidationResult:
    """Evaluate exactly one complete validation episode with explicit state."""

    if seed < 0:
        raise ValidationEvaluationError("evaluation seed cannot be negative")
    if not isinstance(model, RecurrentPPO):
        raise ValidationEvaluationError("model must be sb3-contrib RecurrentPPO")
    if not trainer_result.succeeded or trainer_result.model is not model:
        raise ValidationEvaluationError(
            "completed matching recurrent trainer result is required"
        )
    context = load_recurrent_validation_context(
        symbol,
        splits_dir=Path(splits_dir),
    )
    if trainer_result.symbol != context.symbol:
        raise ValidationEvaluationError("training/validation symbols differ")
    version_pairs = (
        (trainer_result.recurrent_contract_version, context.recurrent_contract_version),
        (trainer_result.environment_version, context.environment_version),
        (trainer_result.feature_version, context.feature_version),
    )
    if any(training != validation for training, validation in version_pairs):
        raise ValidationEvaluationError("training/validation recurrent versions differ")
    if trainer_result.source_recurrent_contract_sha256 != context.contract_sha256:
        raise ValidationEvaluationError("recurrent contract changed after training")
    if trainer_result.observation_features != context.observation_features:
        raise ValidationEvaluationError("recurrent observation order changed")
    config = environment_config or SingleSymbolEnvConfig()
    if config.max_episode_steps is not None:
        raise ValidationEvaluationError("validation requires an untruncated episode")
    if tuple(config.feature_columns) != DEFAULT_OBSERVATION_FEATURES:
        raise ValidationEvaluationError("validation feature order is incompatible")
    if config.environment_version != context.environment_version:
        raise ValidationEvaluationError("validation environment version differs")

    before_hash = policy_parameter_hash(model)
    before_timesteps = int(model.num_timesteps)
    original_training_mode = bool(model.policy.training)
    environment = SingleSymbolTradingEnv(context.data, config)
    if model.observation_space.shape != environment.observation_space.shape:
        environment.close()
        raise ValidationEvaluationError("recurrent observation space differs")
    if getattr(model.action_space, "n", None) != environment.action_space.n:
        environment.close()
        raise ValidationEvaluationError("recurrent action space differs")

    started = time.perf_counter()
    lstm_state = None
    episode_start = np.asarray([True], dtype=bool)
    episode_reset_count = 0
    recurrent_state_steps = 0
    try:
        observation, _ = environment.reset(seed=seed)
        environment.action_space.seed(seed)
        terminated = truncated = False
        with torch.no_grad():
            while not (terminated or truncated):
                if bool(episode_start[0]):
                    episode_reset_count += 1
                if lstm_state is not None:
                    recurrent_state_steps += 1
                predicted, lstm_state = model.predict(
                    observation,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action = int(np.asarray(predicted).item())
                observation, _, terminated, truncated, _ = environment.step(action)
                episode_start = np.asarray([terminated or truncated], dtype=bool)
        history = environment.get_history()
    finally:
        model.policy.set_training_mode(original_training_mode)
        environment.close()
    duration = time.perf_counter() - started
    # The generic checker validates exact N-1 observation/execution dates.
    generic_context = type(
        "HistoryContext",
        (),
        {"rows": context.rows, "data": context.data},
    )()
    _validate_full_history(history, generic_context)
    after_hash = policy_parameter_hash(model)
    after_timesteps = int(model.num_timesteps)
    if before_hash != after_hash or before_timesteps != after_timesteps:
        raise ValidationEvaluationError(
            "recurrent parameters/timestep state changed during validation"
        )
    strategy = StrategyEvaluationResult(
        strategy="RecurrentPPO",
        history=history,
        metrics=calculate_episode_metrics(history),
        duration_seconds=duration,
    )
    return RecurrentValidationResult(
        symbol=context.symbol,
        evaluation_partition="validation",
        validation_start=context.start,
        validation_end=context.end,
        validation_rows=context.rows,
        recurrent_contract_version=context.recurrent_contract_version,
        environment_version=context.environment_version,
        feature_version=context.feature_version,
        deterministic_seed=seed,
        strategy_result=strategy,
        parameter_hash_before=before_hash,
        parameter_hash_after=after_hash,
        model_timesteps_before=before_timesteps,
        model_timesteps_after=after_timesteps,
        first_episode_start=True,
        episode_reset_count=episode_reset_count,
        recurrent_state_steps=recurrent_state_steps,
        final_state_available=lstm_state is not None,
        train_state_supplied=False,
    )
