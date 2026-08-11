"""Validation-only PPO comparison with deterministic non-AI baselines."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import logging
import math
from numbers import Integral, Real
from pathlib import Path
import time

import pandas as pd
from stable_baselines3 import PPO

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.training.config import PPOConfig
from reinforcement_learning.training.ppo_trainer import (
    MAX_SMOKE_TIMESTEPS,
    train_single_symbol,
)
from reinforcement_learning.training.results import PPOTrainingResult

from .baselines import (
    AlwaysHoldPolicy,
    BaselinePolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    run_baseline,
)
from .ppo_evaluator import (
    EXPECTED_HISTORY_COLUMNS,
    ValidationContext,
    ValidationEvaluationError,
    evaluate_ppo_on_context,
    load_validation_context,
)
from .results import (
    CandidateValidationDecision,
    StrategyEvaluationResult,
    ValidationComparisonResult,
)


CANDIDATE_CRITERIA_VERSION = "ppo_validation_criteria_v1"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateValidationCriteria:
    """Explicit conservative thresholds for validation, never promotion."""

    criteria_version: str = CANDIDATE_CRITERIA_VERSION
    minimum_validation_observations: int = 126
    minimum_return_advantage_vs_buy_and_hold: float = 0.0
    minimum_ppo_sharpe: float = 0.0
    minimum_sharpe_advantage_vs_buy_and_hold: float = 0.0
    minimum_ppo_sortino: float = 0.0
    maximum_ppo_drawdown: float = 0.30
    maximum_drawdown_disadvantage_vs_buy_and_hold: float = 0.02

    def __post_init__(self) -> None:
        if self.criteria_version != CANDIDATE_CRITERIA_VERSION:
            raise ValueError(
                f"criteria_version must be {CANDIDATE_CRITERIA_VERSION!r}"
            )
        if (
            isinstance(self.minimum_validation_observations, bool)
            or not isinstance(self.minimum_validation_observations, Integral)
            or self.minimum_validation_observations < 2
        ):
            raise ValueError("minimum_validation_observations must be at least 2")
        numeric = (
            self.minimum_return_advantage_vs_buy_and_hold,
            self.minimum_ppo_sharpe,
            self.minimum_sharpe_advantage_vs_buy_and_hold,
            self.minimum_ppo_sortino,
            self.maximum_ppo_drawdown,
            self.maximum_drawdown_disadvantage_vs_buy_and_hold,
        )
        if not all(
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("candidate validation thresholds must be finite")
        if not 0 <= self.maximum_ppo_drawdown <= 1:
            raise ValueError("maximum_ppo_drawdown must be between 0 and 1")
        if self.maximum_drawdown_disadvantage_vs_buy_and_hold < 0:
            raise ValueError(
                "maximum_drawdown_disadvantage_vs_buy_and_hold cannot be negative"
            )


def _finite_metric(metrics: Mapping[str, object], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decide_candidate_validation(
    ppo_metrics: Mapping[str, object],
    buy_and_hold_metrics: Mapping[str, object],
    *,
    validation_observations: int,
    criteria: CandidateValidationCriteria | None = None,
    evaluation_error: str | None = None,
) -> CandidateValidationDecision:
    """Apply predeclared validation thresholds without model promotion."""
    selected = criteria or CandidateValidationCriteria()
    if (
        isinstance(validation_observations, bool)
        or not isinstance(validation_observations, Integral)
        or validation_observations < 0
    ):
        raise ValueError("validation_observations must be a non-negative integer")
    thresholds = asdict(selected)
    if evaluation_error:
        return CandidateValidationDecision(
            status="evaluation_error",
            passed=False,
            reasons=(evaluation_error,),
            criteria_version=selected.criteria_version,
            thresholds=thresholds,
        )
    if validation_observations < selected.minimum_validation_observations:
        return CandidateValidationDecision(
            status="insufficient_validation_data",
            passed=False,
            reasons=(
                f"Validation has {validation_observations} observations; "
                f"at least {selected.minimum_validation_observations} are required.",
            ),
            criteria_version=selected.criteria_version,
            thresholds=thresholds,
        )

    required = {
        "ppo_total_return": _finite_metric(ppo_metrics, "total_return"),
        "buy_hold_total_return": _finite_metric(
            buy_and_hold_metrics, "total_return"
        ),
        "ppo_sharpe": _finite_metric(ppo_metrics, "sharpe_ratio"),
        "buy_hold_sharpe": _finite_metric(
            buy_and_hold_metrics, "sharpe_ratio"
        ),
        "ppo_sortino": _finite_metric(ppo_metrics, "sortino_ratio"),
        "ppo_drawdown": _finite_metric(ppo_metrics, "maximum_drawdown"),
        "buy_hold_drawdown": _finite_metric(
            buy_and_hold_metrics, "maximum_drawdown"
        ),
    }
    unavailable = tuple(name for name, value in required.items() if value is None)
    if unavailable:
        return CandidateValidationDecision(
            status="evaluation_error",
            passed=False,
            reasons=(
                "Required validation metrics are missing or non-finite: "
                + ", ".join(unavailable),
            ),
            criteria_version=selected.criteria_version,
            thresholds=thresholds,
        )

    values = {name: float(value) for name, value in required.items()}
    return_advantage = (
        values["ppo_total_return"] - values["buy_hold_total_return"]
    )
    sharpe_advantage = values["ppo_sharpe"] - values["buy_hold_sharpe"]
    drawdown_disadvantage = (
        values["ppo_drawdown"] - values["buy_hold_drawdown"]
    )
    reasons: list[str] = []
    if return_advantage < selected.minimum_return_advantage_vs_buy_and_hold:
        reasons.append("PPO return did not meet the Buy-and-Hold advantage threshold.")
    if values["ppo_sharpe"] < selected.minimum_ppo_sharpe:
        reasons.append("PPO Sharpe ratio was below the absolute minimum.")
    if sharpe_advantage < selected.minimum_sharpe_advantage_vs_buy_and_hold:
        reasons.append("PPO Sharpe ratio did not meet the benchmark advantage.")
    if values["ppo_sortino"] < selected.minimum_ppo_sortino:
        reasons.append("PPO Sortino ratio was below the absolute minimum.")
    if values["ppo_drawdown"] > selected.maximum_ppo_drawdown:
        reasons.append("PPO maximum drawdown exceeded the absolute limit.")
    if (
        drawdown_disadvantage
        > selected.maximum_drawdown_disadvantage_vs_buy_and_hold
    ):
        reasons.append("PPO drawdown disadvantage exceeded the benchmark limit.")
    return CandidateValidationDecision(
        status="validation_fail" if reasons else "validation_pass",
        passed=not reasons,
        reasons=tuple(reasons) if reasons else ("All validation criteria passed.",),
        criteria_version=selected.criteria_version,
        thresholds=thresholds,
    )


def _run_baseline_strategy(
    name: str,
    policy: BaselinePolicy,
    context: ValidationContext,
    environment_config: SingleSymbolEnvConfig,
    *,
    seed: int,
) -> StrategyEvaluationResult:
    environment = SingleSymbolTradingEnv(context.data, environment_config)
    started = time.perf_counter()
    try:
        result = run_baseline(environment, policy, seed=seed)
    finally:
        environment.close()
    return StrategyEvaluationResult(
        strategy=name,
        history=result.history,
        metrics=result.metrics,
        duration_seconds=time.perf_counter() - started,
    )


def _assert_apples_to_apples(
    context: ValidationContext,
    strategies: Sequence[StrategyEvaluationResult],
    *,
    initial_cash: float,
) -> None:
    expected_observation_dates = context.data["date"].iloc[:-1].reset_index(drop=True)
    expected_execution_dates = context.data["date"].iloc[1:].reset_index(drop=True)
    for strategy in strategies:
        history = strategy.history
        if not set(EXPECTED_HISTORY_COLUMNS).issubset(history.columns):
            raise ValidationEvaluationError(
                f"{strategy.strategy} history schema is incompatible"
            )
        if len(history) != context.rows - 1:
            raise ValidationEvaluationError(
                f"{strategy.strategy} did not consume full validation history"
            )
        observation_dates = pd.to_datetime(history["observation_date"]).reset_index(
            drop=True
        )
        execution_dates = pd.to_datetime(history["execution_date"]).reset_index(
            drop=True
        )
        if not observation_dates.equals(expected_observation_dates):
            raise ValidationEvaluationError(
                f"{strategy.strategy} observation dates differ"
            )
        if not execution_dates.equals(expected_execution_dates):
            raise ValidationEvaluationError(f"{strategy.strategy} execution dates differ")
        strategy_initial = _finite_metric(
            strategy.metrics, "initial_portfolio_value"
        )
        if strategy_initial is None or not math.isclose(
            strategy_initial, initial_cash
        ):
            raise ValidationEvaluationError(
                f"{strategy.strategy} initial capital differs"
            )


def _training_metadata(
    model: PPO,
    symbol: str,
    context: ValidationContext,
    trainer_result: PPOTrainingResult | None,
) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    if trainer_result is None:
        warnings.append("PPO trainer result metadata was not supplied.")
        return {
            "available": False,
            "model_class": type(model).__name__,
            "model_num_timesteps": int(model.num_timesteps),
        }, warnings
    if not trainer_result.succeeded:
        raise ValidationEvaluationError("PPO trainer result is not completed")
    if trainer_result.symbol != symbol:
        raise ValidationEvaluationError("PPO trainer result symbol does not match")
    if trainer_result.model is not model:
        raise ValidationEvaluationError(
            "PPO trainer result does not describe the supplied in-memory model"
        )
    version_fields = (
        ("environment_version", context.environment_version),
        ("rl_contract_version", context.rl_contract_version),
        ("feature_version", context.feature_version),
    )
    for field_name, validation_value in version_fields:
        if getattr(trainer_result, field_name) != validation_value:
            raise ValidationEvaluationError(
                f"PPO trainer {field_name} does not match validation artifacts"
            )
    provenance_fields = (
        ("source_rl_contract_sha256", context.source_rl_contract_sha256),
        (
            "source_observation_scaler_sha256",
            context.source_observation_scaler_sha256,
        ),
        (
            "source_observation_scaler_metadata_sha256",
            context.source_observation_scaler_metadata_sha256,
        ),
    )
    for field_name, validation_value in provenance_fields:
        training_value = getattr(trainer_result, field_name)
        if not training_value or training_value != validation_value:
            raise ValidationEvaluationError(
                f"PPO trainer {field_name} does not match validation artifacts"
            )
    if tuple(trainer_result.observation_features) != context.observation_features:
        raise ValidationEvaluationError(
            "PPO trainer observation_features do not match validation artifacts"
        )
    if trainer_result.observation_shape != model.observation_space.shape:
        raise ValidationEvaluationError(
            "PPO trainer observation shape does not match the supplied model"
        )
    return trainer_result.to_dict(), warnings


def compare_candidate_on_validation(
    model: PPO,
    symbol: str,
    *,
    trainer_result: PPOTrainingResult | None = None,
    environment_config: SingleSymbolEnvConfig | None = None,
    deterministic_seed: int = 42,
    random_seed: int = 42,
    criteria: CandidateValidationCriteria | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> ValidationComparisonResult:
    """Evaluate PPO and all baselines on one identical validation partition."""
    if deterministic_seed < 0 or random_seed < 0:
        raise ValidationEvaluationError("evaluation seeds cannot be negative")
    config = environment_config or SingleSymbolEnvConfig()
    if config.max_episode_steps is not None:
        raise ValidationEvaluationError(
            "Validation comparison requires complete, untruncated episodes"
        )
    started = time.perf_counter()
    context = load_validation_context(symbol, splits_dir=Path(splits_dir))
    metadata, warnings = _training_metadata(
        model,
        context.symbol,
        context,
        trainer_result,
    )
    ppo_validation = evaluate_ppo_on_context(
        model,
        context,
        seed=deterministic_seed,
        environment_config=config,
    )
    ppo = ppo_validation.strategy_result
    buy_and_hold = _run_baseline_strategy(
        "Buy and Hold",
        BuyAndHoldPolicy(),
        context,
        config,
        seed=deterministic_seed,
    )
    always_hold = _run_baseline_strategy(
        "Always Hold",
        AlwaysHoldPolicy(),
        context,
        config,
        seed=deterministic_seed,
    )
    random_result = _run_baseline_strategy(
        "Random",
        RandomPolicy(seed=random_seed),
        context,
        config,
        seed=random_seed,
    )
    strategies = (ppo, buy_and_hold, always_hold, random_result)
    _assert_apples_to_apples(context, strategies, initial_cash=config.initial_cash)

    ppo_return = float(ppo.metrics["total_return"])
    buy_hold_return = float(buy_and_hold.metrics["total_return"])
    ppo_drawdown = float(ppo.metrics["maximum_drawdown"])
    buy_hold_drawdown = float(buy_and_hold.metrics["maximum_drawdown"])
    ppo_sharpe = _finite_metric(ppo.metrics, "sharpe_ratio")
    buy_hold_sharpe = _finite_metric(buy_and_hold.metrics, "sharpe_ratio")
    sharpe_difference = (
        ppo_sharpe - buy_hold_sharpe
        if ppo_sharpe is not None and buy_hold_sharpe is not None
        else None
    )
    decision = decide_candidate_validation(
        ppo.metrics,
        buy_and_hold.metrics,
        validation_observations=context.rows,
        criteria=criteria,
    )
    for strategy in strategies:
        for warning in strategy.metrics.get("metric_warnings", ()):
            warnings.append(f"{strategy.strategy}: {warning}")
    return ValidationComparisonResult(
        symbol=context.symbol,
        evaluation_partition="validation",
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
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        environment_config=asdict(config),
        ppo_training_metadata=metadata,
        ppo_parameter_hash_before=ppo_validation.parameter_hash_before,
        ppo_parameter_hash_after=ppo_validation.parameter_hash_after,
        ppo_model_timesteps_before=ppo_validation.model_timesteps_before,
        ppo_model_timesteps_after=ppo_validation.model_timesteps_after,
        ppo=ppo,
        buy_and_hold=buy_and_hold,
        always_hold=always_hold,
        random=random_result,
        ppo_return_difference_vs_buy_and_hold=ppo_return - buy_hold_return,
        ppo_sharpe_difference_vs_buy_and_hold=sharpe_difference,
        ppo_max_drawdown_difference_vs_buy_and_hold=(
            ppo_drawdown - buy_hold_drawdown
        ),
        deterministic_seed=deterministic_seed,
        random_seed=random_seed,
        evaluation_duration_seconds=time.perf_counter() - started,
        candidate_decision=decision,
        warnings=tuple(warnings),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one tiny in-memory PPO smoke candidate and compare validation "
            "baselines without saving or promotion."
        )
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--timesteps", type=_positive_integer, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.smoke_test:
        parser.error("--smoke-test is required for the 5B-2 developer CLI")
    if args.timesteps > MAX_SMOKE_TIMESTEPS:
        parser.error(
            f"--smoke-test permits at most {MAX_SMOKE_TIMESTEPS} timesteps"
        )
    if args.seed < 0 or args.random_seed < 0:
        parser.error("seeds cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.warning(
        "Developer validation smoke only: no test evaluation or model promotion."
    )
    training = train_single_symbol(
        args.symbol,
        config=PPOConfig(),
        seed=args.seed,
        total_timesteps=args.timesteps,
        smoke_test=True,
    )
    if not training.succeeded or training.model is None:
        print(json.dumps(training.to_dict(), indent=2, sort_keys=True))
        return 1
    try:
        result = compare_candidate_on_validation(
            training.model,
            args.symbol,
            trainer_result=training,
            deterministic_seed=args.seed,
            random_seed=args.random_seed,
        )
    except Exception as exc:
        print(f"Validation evaluation failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result.to_dict(include_history=False), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
