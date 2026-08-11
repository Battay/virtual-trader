"""Independent per-symbol VALIDATION evaluation for a sector foundation model."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from sb3_contrib import RecurrentPPO
import torch

from data_pipeline.src.config import PROCESSED_SPLITS_DIR
from reinforcement_learning.environments import SingleSymbolEnvConfig, SingleSymbolTradingEnv
from reinforcement_learning.evaluation.baselines import (
    AlwaysHoldPolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    run_baseline,
)
from reinforcement_learning.evaluation.metrics import calculate_episode_metrics
from reinforcement_learning.evaluation.ppo_evaluator import (
    ValidationEvaluationError,
    _validate_full_history,
    policy_parameter_hash,
)
from reinforcement_learning.evaluation.results import StrategyEvaluationResult
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    LoadedRecurrentPartition,
    load_recurrent_partition,
)
from reinforcement_learning.training.sector_recurrent_results import (
    SectorRecurrentTrainingResult,
    SectorSymbolValidationResult,
    SectorValidationResult,
)
from reinforcement_learning.training.sector_methodology_diagnostics import (
    detect_action_collapse,
)
from reinforcement_learning.training.sector_recurrent_trainer import (
    COMMERCIAL_BANKS_MANIFEST_PATH,
    _read_manifest,
)


def _strategy(
    name: str,
    history,
    *,
    duration: float,
) -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        strategy=name,
        history=history,
        metrics=calculate_episode_metrics(history),
        duration_seconds=duration,
    )


def _run_recurrent_episode(
    model: RecurrentPPO,
    environment: SingleSymbolTradingEnv,
    *,
    seed: int,
) -> tuple[StrategyEvaluationResult, int, Mapping[str, int], Mapping[str, float], str]:
    observation, info = environment.reset(seed=seed)
    if (
        float(info["cash"]) != environment.config.initial_cash
        or int(info["shares_held"]) != 0
        or float(info["portfolio_value"]) != environment.config.initial_cash
    ):
        raise ValidationEvaluationError("validation portfolio failed to reset")
    environment.action_space.seed(seed)
    state = None
    episode_start = np.asarray([True], dtype=bool)
    recurrent_state_steps = 0
    terminated = truncated = False
    started = time.perf_counter()
    with torch.no_grad():
        while not (terminated or truncated):
            if state is not None:
                recurrent_state_steps += 1
            action, state = model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, reward, terminated, truncated, _ = environment.step(
                int(np.asarray(action).item())
            )
            if not math.isfinite(float(reward)):
                raise ValidationEvaluationError("validation reward is non-finite")
            episode_start = np.asarray([terminated or truncated], dtype=bool)
    duration = time.perf_counter() - started
    history = environment.get_history()
    action_counts = {
        name: int(history["action_name"].eq(name).sum())
        for name in ("Hold", "Buy", "Sell")
    }
    total = len(history)
    percentages = {
        name: (100.0 * count / total if total else 0.0)
        for name, count in action_counts.items()
    }
    sequence = json.dumps(history["action"].astype(int).tolist(), separators=(",", ":"))
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    return (
        _strategy("Sector RecurrentPPO", history, duration=duration),
        recurrent_state_steps,
        action_counts,
        percentages,
        digest,
    )


def _baseline_result(data, policy, name: str, *, seed: int, config):
    environment = SingleSymbolTradingEnv(data, config)
    started = time.perf_counter()
    try:
        result = run_baseline(environment, policy, seed=seed)
    finally:
        environment.close()
    return _strategy(name, result.history, duration=time.perf_counter() - started)


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "minimum", "maximum", "q25", "q75")}
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValidationEvaluationError("aggregate metric contains non-finite values")
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "q25": float(np.quantile(data, 0.25)),
        "q75": float(np.quantile(data, 0.75)),
    }


def aggregate_sector_validation(
    symbol_results: Sequence[SectorSymbolValidationResult],
    failures: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate independent symbol outcomes without constructing a portfolio."""

    metrics: dict[str, object] = {
        "symbols_evaluated": len(symbol_results),
        "evaluation_failures": len(failures),
        "failed_symbols": sorted(failures),
        "aggregation_interpretation": "statistics_of_independent_equal_capital_episodes",
    }
    for strategy_name, attribute in (
        ("ppo", "ppo"),
        ("buy_and_hold", "buy_and_hold"),
        ("always_hold", "always_hold"),
        ("random", "random"),
    ):
        for metric in (
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "maximum_drawdown",
            "number_of_trades", "total_transaction_costs", "exposure_percentage",
        ):
            values = []
            for result in symbol_results:
                value = getattr(result, attribute).metrics.get(metric)
                if value is not None:
                    values.append(float(value))
            metrics[f"{strategy_name}_{metric}"] = _distribution(values)

    beating_return = losing_return = ties = positive = negative = invalid_sharpe = 0
    beating_sharpe = comparable_sharpe = 0
    for result in symbol_results:
        ppo = result.ppo.metrics
        benchmark = result.buy_and_hold.metrics
        ppo_return = float(ppo["total_return"])
        benchmark_return = float(benchmark["total_return"])
        if ppo_return > benchmark_return:
            beating_return += 1
        elif ppo_return < benchmark_return:
            losing_return += 1
        else:
            ties += 1
        positive += int(ppo_return > 0)
        negative += int(ppo_return < 0)
        ppo_sharpe = ppo.get("sharpe_ratio")
        benchmark_sharpe = benchmark.get("sharpe_ratio")
        if ppo_sharpe is None:
            invalid_sharpe += 1
        if ppo_sharpe is not None and benchmark_sharpe is not None:
            comparable_sharpe += 1
            beating_sharpe += int(float(ppo_sharpe) > float(benchmark_sharpe))
    evaluated = len(symbol_results)
    metrics.update(
        {
            "symbols_beating_buy_and_hold_return": beating_return,
            "symbols_losing_to_buy_and_hold_return": losing_return,
            "symbols_tied_buy_and_hold_return": ties,
            "fraction_beating_buy_and_hold_return": beating_return / evaluated if evaluated else None,
            "symbols_with_positive_ppo_return": positive,
            "symbols_with_negative_ppo_return": negative,
            "symbols_with_undefined_ppo_sharpe": invalid_sharpe,
            "symbols_with_comparable_sharpe": comparable_sharpe,
            "symbols_beating_buy_and_hold_sharpe": beating_sharpe,
            "fraction_beating_buy_and_hold_sharpe": (
                beating_sharpe / comparable_sharpe if comparable_sharpe else None
            ),
        }
    )

    total_actions = Counter({"Hold": 0, "Buy": 0, "Sell": 0})
    zero_trade_symbols: list[str] = []
    patterns: dict[str, list[str]] = {}
    pattern_by_symbol: dict[str, str] = {}
    trades_by_symbol: dict[str, int] = {}
    exposure_by_symbol: dict[str, float] = {}
    exposure = []
    invalid_action_count = 0
    observed_action_count = 0
    for result in symbol_results:
        total_actions.update(result.action_counts)
        trade_count = int(result.ppo.metrics["number_of_trades"])
        trades_by_symbol[result.symbol] = trade_count
        if trade_count == 0:
            zero_trade_symbols.append(result.symbol)
        symbol_exposure = float(
            result.ppo.metrics["exposure_percentage"] or 0.0
        )
        exposure_by_symbol[result.symbol] = symbol_exposure
        exposure.append(symbol_exposure)
        patterns.setdefault(result.action_pattern_digest, []).append(result.symbol)
        pattern_by_symbol[result.symbol] = result.action_pattern_digest
        history = result.ppo.history
        if "action_valid" in history:
            validity = history["action_valid"].astype(bool)
            invalid_action_count += int((~validity).sum())
            observed_action_count += len(validity)
    action_total = sum(total_actions.values())
    action_percentages = {
        name: (100.0 * count / action_total if action_total else 0.0)
        for name, count in total_actions.items()
    }
    identical_groups = [symbols for symbols in patterns.values() if len(symbols) > 1]
    fixed_collapse = detect_action_collapse(
        selected_action_counts={
            "hold": int(total_actions["Hold"]),
            "buy": int(total_actions["Buy"]),
            "sell": int(total_actions["Sell"]),
        },
        invalid_action_rate=(
            invalid_action_count / observed_action_count
            if observed_action_count
            else 0.0
        ),
        per_symbol_exposure_percentages=exposure_by_symbol,
        per_symbol_trade_counts=trades_by_symbol,
        per_symbol_action_digests=pattern_by_symbol,
    )
    warnings: list[str] = []
    dominant_action = max(action_percentages, key=action_percentages.get) if action_total else None
    if dominant_action and action_percentages[dominant_action] > 80.0:
        warnings.append(
            f"possible_action_collapse: {dominant_action} exceeds 80% of actions"
        )
    if dominant_action and action_percentages[dominant_action] >= 90.0:
        warnings.append(f"possible policy collapse: {dominant_action} is at least 90% of actions")
    if observed_action_count and invalid_action_count / observed_action_count > 0.80:
        warnings.append(
            "possible_invalid_action_attractor: more than 80% of selections are invalid"
        )
    if evaluated and len(zero_trade_symbols) == evaluated:
        warnings.append("possible policy collapse: every symbol has zero executed trades")
    if exposure and float(np.median(exposure)) < 5.0:
        warnings.append("possible policy collapse: median exposure is at most 5%")
    if identical_groups:
        warnings.append("identical action sequences observed for equal-length validation episodes")
    collapse = {
        "action_counts": dict(total_actions),
        "action_percentages": action_percentages,
        "zero_trade_symbol_count": len(zero_trade_symbols),
        "zero_trade_symbols": zero_trade_symbols,
        "median_exposure_percentage": float(np.median(exposure)) if exposure else None,
        "identical_action_pattern_groups": identical_groups,
        "invalid_action_count": invalid_action_count,
        "invalid_action_rate": (
            invalid_action_count / observed_action_count
            if observed_action_count
            else 0.0
        ),
        "predeclared_warning_diagnostics": fixed_collapse,
        "warnings": warnings,
        "obvious_policy_collapse_flag": bool(warnings),
    }
    return metrics, collapse


def evaluate_sector_recurrent_on_validation(
    training: SectorRecurrentTrainingResult,
    *,
    manifest_path: Path = COMMERCIAL_BANKS_MANIFEST_PATH,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    deterministic_seed: int = 42,
    random_seed: int = 42,
    environment_config: SingleSymbolEnvConfig | None = None,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
) -> SectorValidationResult:
    """Evaluate one shared model on separate complete constituent validations."""

    if not training.succeeded or not isinstance(training.model, RecurrentPPO):
        raise ValidationEvaluationError("completed sector RecurrentPPO training is required")
    manifest = _read_manifest(Path(manifest_path))
    if manifest["universe_hash"] != training.sector_universe_hash:
        raise ValidationEvaluationError("training/validation sector universe differs")
    if tuple(manifest["experiment_mode"]["pretraining_constituent_symbols"]) != training.constituent_symbols:
        raise ValidationEvaluationError("training/validation constituents differ")
    constituent_by_symbol = {item["symbol"]: item for item in manifest["constituents"]}
    config = environment_config or SingleSymbolEnvConfig()
    if config.max_episode_steps is not None:
        raise ValidationEvaluationError("sector validation must be untruncated")
    before_hash = policy_parameter_hash(training.model)
    before_timesteps = int(training.model.num_timesteps)
    original_mode = bool(training.model.policy.training)
    started = time.perf_counter()
    results: list[SectorSymbolValidationResult] = []
    failures: dict[str, str] = {}
    try:
        for index, symbol in enumerate(training.constituent_symbols):
            environment = None
            try:
                loaded = partition_loader(symbol, "validation", splits_dir=Path(splits_dir))
                metadata = loaded.metadata
                source = constituent_by_symbol[symbol]
                if sha256_file(metadata.contract_path) != source["recurrent_contract_sha256"]:
                    raise ValidationEvaluationError("recurrent contract changed after training")
                if (
                    metadata.recurrent_contract_version != training.recurrent_contract_version
                    or metadata.feature_version != training.feature_version
                    or metadata.environment_version != training.environment_version
                    or metadata.observation_features != training.observation_features
                ):
                    raise ValidationEvaluationError("validation contract is incompatible")
                environment = SingleSymbolTradingEnv(loaded.data, config)
                ppo, state_steps, counts, percentages, digest = _run_recurrent_episode(
                    training.model, environment, seed=deterministic_seed
                )
                context = type("Context", (), {"rows": metadata.validation.rows, "data": loaded.data})()
                _validate_full_history(ppo.history, context)
                portfolio_reset = (
                    float(ppo.metrics["initial_portfolio_value"]) == config.initial_cash
                )
                buy_hold = _baseline_result(
                    loaded.data, BuyAndHoldPolicy(), "Buy and Hold",
                    seed=deterministic_seed, config=config,
                )
                always_hold = _baseline_result(
                    loaded.data, AlwaysHoldPolicy(), "Always Hold",
                    seed=deterministic_seed, config=config,
                )
                random = _baseline_result(
                    loaded.data, RandomPolicy(random_seed + index), "Random",
                    seed=random_seed + index, config=config,
                )
                for item in (buy_hold, always_hold, random):
                    _validate_full_history(item.history, context)
                    if float(item.metrics["initial_portfolio_value"]) != config.initial_cash:
                        raise ValidationEvaluationError("baseline capital was not independently reset")
                current_hash = policy_parameter_hash(training.model)
                current_timesteps = int(training.model.num_timesteps)
                results.append(
                    SectorSymbolValidationResult(
                        symbol=symbol,
                        validation_start=metadata.validation.start,
                        validation_end=metadata.validation.end,
                        validation_rows=metadata.validation.rows,
                        initial_capital=config.initial_cash,
                        ppo=ppo,
                        buy_and_hold=buy_hold,
                        always_hold=always_hold,
                        random=random,
                        first_episode_start=True,
                        recurrent_state_steps=state_steps,
                        portfolio_reset_verified=portfolio_reset,
                        model_parameters_unchanged=(
                            current_hash == before_hash and current_timesteps == before_timesteps
                        ),
                        action_counts=counts,
                        action_percentages=percentages,
                        action_pattern_digest=digest,
                    )
                )
            except Exception as exc:
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            finally:
                if environment is not None:
                    environment.close()
    finally:
        training.model.policy.set_training_mode(original_mode)
    after_hash = policy_parameter_hash(training.model)
    after_timesteps = int(training.model.num_timesteps)
    if before_hash != after_hash or before_timesteps != after_timesteps:
        raise ValidationEvaluationError("sector model changed during validation")
    aggregate, collapse = aggregate_sector_validation(results, failures)
    return SectorValidationResult(
        sector_id=training.sector_id,
        sector_name=training.sector_name,
        universe_hash=training.sector_universe_hash,
        evaluation_partition="validation",
        independent_symbol_capital=True,
        initial_capital_per_symbol=config.initial_cash,
        symbols_requested=training.constituent_symbols,
        symbol_results=tuple(results),
        failures=failures,
        aggregate_metrics=aggregate,
        collapse_diagnostics=collapse,
        parameter_hash_before=before_hash,
        parameter_hash_after=after_hash,
        model_timesteps_before=before_timesteps,
        model_timesteps_after=after_timesteps,
        duration_seconds=time.perf_counter() - started,
        test_evaluated=False,
    )


__all__ = (
    "aggregate_sector_validation",
    "evaluate_sector_recurrent_on_validation",
)
