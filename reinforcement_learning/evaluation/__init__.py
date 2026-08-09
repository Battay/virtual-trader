"""Baseline policies and evaluation metrics."""

from .baselines import (
    AlwaysHoldPolicy,
    BaselineResult,
    BuyAndHoldPolicy,
    RandomPolicy,
    run_baseline,
)
from .metrics import calculate_episode_metrics
from .results import (
    CandidateValidationDecision,
    PPOValidationResult,
    StrategyEvaluationResult,
    ValidationComparisonResult,
)


def __getattr__(name: str):
    """Load PPO evaluation orchestration lazily for clean module CLI use."""
    if name in {
        "CandidateValidationCriteria",
        "compare_candidate_on_validation",
        "decide_candidate_validation",
    }:
        from .comparison import (
            CandidateValidationCriteria,
            compare_candidate_on_validation,
            decide_candidate_validation,
        )

        values = {
            "CandidateValidationCriteria": CandidateValidationCriteria,
            "compare_candidate_on_validation": compare_candidate_on_validation,
            "decide_candidate_validation": decide_candidate_validation,
        }
        return values[name]
    if name in {
        "ValidationEvaluationError",
        "evaluate_ppo_validation",
        "policy_parameter_hash",
    }:
        from .ppo_evaluator import (
            ValidationEvaluationError,
            evaluate_ppo_validation,
            policy_parameter_hash,
        )

        values = {
            "ValidationEvaluationError": ValidationEvaluationError,
            "evaluate_ppo_validation": evaluate_ppo_validation,
            "policy_parameter_hash": policy_parameter_hash,
        }
        return values[name]
    raise AttributeError(name)

__all__ = (
    "AlwaysHoldPolicy",
    "BaselineResult",
    "BuyAndHoldPolicy",
    "CandidateValidationCriteria",
    "CandidateValidationDecision",
    "PPOValidationResult",
    "RandomPolicy",
    "StrategyEvaluationResult",
    "ValidationComparisonResult",
    "ValidationEvaluationError",
    "calculate_episode_metrics",
    "compare_candidate_on_validation",
    "decide_candidate_validation",
    "evaluate_ppo_validation",
    "policy_parameter_hash",
    "run_baseline",
)
