"""Deterministic PPO evaluation on the sealed validation partition only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.data_contract import (
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
    LoadedRLPartition,
    load_rl_partition,
)
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.environments.config import DEFAULT_OBSERVATION_FEATURES
from reinforcement_learning.integrity import sha256_file

from .metrics import calculate_episode_metrics
from .results import PPOValidationResult, StrategyEvaluationResult


VALIDATION_PARTITION = "validation"
EXPECTED_HISTORY_COLUMNS = (
    "initial_portfolio_value",
    "observation_date",
    "execution_date",
    "action",
    "action_name",
    "execution_price",
    "shares_traded",
    "transaction_cost",
    "cash",
    "shares_held",
    "portfolio_value",
    "realized_profit_loss",
    "unrealized_profit_loss",
    "drawdown",
    "reward",
)


class ValidationEvaluationError(RuntimeError):
    """Raised when validation evaluation would violate its fixed contract."""


@dataclass(frozen=True)
class ValidationContext:
    """One validated canonical validation frame and its provenance."""

    symbol: str
    data: pd.DataFrame
    start: str
    end: str
    rows: int
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.copy(deep=True))


def policy_parameter_hash(model: PPO) -> str:
    """Hash policy parameters/buffers deterministically without serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def load_validation_context(
    symbol: str,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> ValidationContext:
    """Load exactly one canonical validation partition, with no fallback."""
    symbol_text = str(symbol).strip()
    if not symbol_text:
        raise ValidationEvaluationError("symbol is required for validation")
    loaded: LoadedRLPartition = load_rl_partition(
        symbol_text,
        VALIDATION_PARTITION,
        splits_dir=Path(splits_dir),
    )
    if loaded.partition != VALIDATION_PARTITION:
        raise ValidationEvaluationError(
            "Canonical loader returned a non-validation partition"
        )
    if loaded.contract.get("scaler_fit_partition") != "train":
        raise ValidationEvaluationError(
            "Validation observations are not backed by a train-fitted scaler"
        )
    data = loaded.data
    if len(data) < 2:
        raise ValidationEvaluationError(
            "Validation partition requires at least two observations"
        )
    artifact_directory = loaded.artifact_path.parent
    contract_path = (artifact_directory / RL_CONTRACT_FILENAME).resolve()
    scaler_path = (artifact_directory / RL_OBSERVATION_SCALER_FILENAME).resolve()
    scaler_metadata_path = scaler_path.with_suffix(".json")
    return ValidationContext(
        symbol=symbol_text,
        data=data,
        start=data["date"].min().date().isoformat(),
        end=data["date"].max().date().isoformat(),
        rows=len(data),
        environment_version=str(loaded.contract.get("environment_version", "")),
        rl_contract_version=str(
            loaded.contract.get("artifact_schema_version", "")
        ),
        feature_version=str(loaded.contract.get("feature_version", "")),
        source_rl_contract_path=str(contract_path),
        source_rl_contract_sha256=sha256_file(contract_path),
        source_observation_scaler_path=str(scaler_path),
        source_observation_scaler_sha256=sha256_file(scaler_path),
        source_observation_scaler_metadata_path=str(scaler_metadata_path),
        source_observation_scaler_metadata_sha256=sha256_file(
            scaler_metadata_path
        ),
        observation_features=tuple(
            str(feature) for feature in loaded.contract.get("observation_features", ())
        ),
    )


def _validate_full_history(history: pd.DataFrame, context: ValidationContext) -> None:
    if tuple(history.columns) != EXPECTED_HISTORY_COLUMNS:
        raise ValidationEvaluationError(
            "PPO history schema differs from the environment contract"
        )
    if len(history) != context.rows - 1:
        raise ValidationEvaluationError(
            "PPO validation episode did not consume the complete partition"
        )
    expected_observations = context.data["date"].iloc[:-1].reset_index(drop=True)
    expected_executions = context.data["date"].iloc[1:].reset_index(drop=True)
    actual_observations = pd.to_datetime(history["observation_date"]).reset_index(
        drop=True
    )
    actual_executions = pd.to_datetime(history["execution_date"]).reset_index(
        drop=True
    )
    if not actual_observations.equals(expected_observations):
        raise ValidationEvaluationError("PPO observation dates left validation bounds")
    if not actual_executions.equals(expected_executions):
        raise ValidationEvaluationError("PPO execution dates left validation bounds")


def evaluate_ppo_on_context(
    model: PPO,
    context: ValidationContext,
    *,
    seed: int,
    environment_config: SingleSymbolEnvConfig,
) -> PPOValidationResult:
    """Run deterministic predict-only PPO inference on one loaded context."""
    if seed < 0:
        raise ValidationEvaluationError("evaluation seed cannot be negative")
    if environment_config.max_episode_steps is not None:
        raise ValidationEvaluationError(
            "Validation evaluation requires a complete, untruncated episode"
        )
    if tuple(environment_config.feature_columns) != DEFAULT_OBSERVATION_FEATURES:
        raise ValidationEvaluationError(
            "Validation observation features must match the canonical RL contract "
            "order"
        )
    if environment_config.environment_version != context.environment_version:
        raise ValidationEvaluationError(
            "Validation environment version differs from the loaded RL contract"
        )
    before_hash = policy_parameter_hash(model)
    before_timesteps = int(model.num_timesteps)
    original_training_mode = bool(model.policy.training)
    environment = SingleSymbolTradingEnv(context.data, environment_config)
    if model.observation_space.shape != environment.observation_space.shape:
        environment.close()
        raise ValidationEvaluationError(
            "PPO observation space is incompatible with validation environment"
        )
    if getattr(model.action_space, "n", None) != environment.action_space.n:
        environment.close()
        raise ValidationEvaluationError(
            "PPO action space is incompatible with validation environment"
        )

    started = time.perf_counter()
    try:
        observation, _ = environment.reset(seed=seed)
        environment.action_space.seed(seed)
        terminated = truncated = False
        with torch.no_grad():
            while not (terminated or truncated):
                predicted, _ = model.predict(observation, deterministic=True)
                action = int(np.asarray(predicted).item())
                observation, _, terminated, truncated, _ = environment.step(action)
        history = environment.get_history()
    finally:
        model.policy.set_training_mode(original_training_mode)
        environment.close()
    duration = time.perf_counter() - started
    _validate_full_history(history, context)
    after_hash = policy_parameter_hash(model)
    after_timesteps = int(model.num_timesteps)
    if before_hash != after_hash or before_timesteps != after_timesteps:
        raise ValidationEvaluationError(
            "PPO parameters or training timestep state changed during validation"
        )
    strategy = StrategyEvaluationResult(
        strategy="PPO",
        history=history,
        metrics=calculate_episode_metrics(history),
        duration_seconds=duration,
    )
    return PPOValidationResult(
        symbol=context.symbol,
        evaluation_partition=VALIDATION_PARTITION,
        validation_start=context.start,
        validation_end=context.end,
        validation_rows=context.rows,
        environment_version=context.environment_version,
        rl_contract_version=context.rl_contract_version,
        feature_version=context.feature_version,
        source_rl_contract_path=context.source_rl_contract_path,
        source_rl_contract_sha256=context.source_rl_contract_sha256,
        source_observation_scaler_path=context.source_observation_scaler_path,
        source_observation_scaler_sha256=(
            context.source_observation_scaler_sha256
        ),
        source_observation_scaler_metadata_path=(
            context.source_observation_scaler_metadata_path
        ),
        source_observation_scaler_metadata_sha256=(
            context.source_observation_scaler_metadata_sha256
        ),
        observation_features=context.observation_features,
        deterministic_seed=seed,
        environment_config=asdict(environment_config),
        strategy_result=strategy,
        parameter_hash_before=before_hash,
        parameter_hash_after=after_hash,
        model_timesteps_before=before_timesteps,
        model_timesteps_after=after_timesteps,
    )


def evaluate_ppo_validation(
    model: PPO,
    symbol: str,
    *,
    seed: int = 42,
    environment_config: SingleSymbolEnvConfig | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> PPOValidationResult:
    """Load validation only and evaluate an existing PPO model deterministically."""
    context = load_validation_context(symbol, splits_dir=Path(splits_dir))
    return evaluate_ppo_on_context(
        model,
        context,
        seed=seed,
        environment_config=environment_config or SingleSymbolEnvConfig(),
    )
