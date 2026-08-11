"""Equivalent-budget RecurrentPPO versus MLP PPO validation comparison."""

from __future__ import annotations

from pathlib import Path

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.environments import SingleSymbolEnvConfig
from reinforcement_learning.training.recurrent_results import (
    RecurrentMLPComparisonResult,
    RecurrentPPOTrainingResult,
)
from reinforcement_learning.training.results import PPOTrainingResult

from .comparison import compare_candidate_on_validation
from .ppo_evaluator import ValidationEvaluationError
from .recurrent_evaluator import evaluate_recurrent_on_validation


def compare_recurrent_and_mlp_on_validation(
    recurrent_training: RecurrentPPOTrainingResult,
    mlp_training: PPOTrainingResult,
    *,
    seed: int = 42,
    random_seed: int = 42,
    environment_config: SingleSymbolEnvConfig | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> RecurrentMLPComparisonResult:
    """Evaluate two same-budget in-memory policies and three fixed baselines."""

    if not recurrent_training.succeeded or recurrent_training.model is None:
        raise ValidationEvaluationError("completed recurrent training is required")
    if not mlp_training.succeeded or mlp_training.model is None:
        raise ValidationEvaluationError("completed MLP training is required")
    if recurrent_training.symbol != mlp_training.symbol:
        raise ValidationEvaluationError("recurrent and MLP symbols differ")
    if recurrent_training.seed != mlp_training.seed or recurrent_training.seed != seed:
        raise ValidationEvaluationError("recurrent and MLP seeds differ")
    if (
        recurrent_training.requested_timesteps != mlp_training.requested_timesteps
        or recurrent_training.actual_timesteps != mlp_training.actual_timesteps
    ):
        raise ValidationEvaluationError("recurrent and MLP timestep budgets differ")
    if (
        recurrent_training.training_rows != mlp_training.training_rows
        or recurrent_training.training_start != mlp_training.training_start
        or recurrent_training.training_end != mlp_training.training_end
    ):
        raise ValidationEvaluationError("recurrent and MLP TRAIN partitions differ")
    config = environment_config or SingleSymbolEnvConfig()
    recurrent = evaluate_recurrent_on_validation(
        recurrent_training.model,
        recurrent_training.symbol,
        trainer_result=recurrent_training,
        seed=seed,
        environment_config=config,
        splits_dir=Path(splits_dir),
    )
    mlp = compare_candidate_on_validation(
        mlp_training.model,
        mlp_training.symbol,
        trainer_result=mlp_training,
        environment_config=config,
        deterministic_seed=seed,
        random_seed=random_seed,
        splits_dir=Path(splits_dir),
    )
    same_validation = (
        recurrent.validation_start == mlp.validation_start
        and recurrent.validation_end == mlp.validation_end
        and recurrent.validation_rows == mlp.validation_rows
    )
    return RecurrentMLPComparisonResult(
        symbol=recurrent_training.symbol,
        recurrent=recurrent,
        mlp_comparison=mlp,
        recurrent_training=recurrent_training.to_dict(),
        mlp_training=mlp_training.to_dict(),
        same_training_budget=True,
        same_validation_partition=same_validation,
        test_evaluated=False,
    )
