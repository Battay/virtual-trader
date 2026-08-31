"""Deterministic Buy-and-Hold benchmark on canonical VALIDATION only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    LoadedRecurrentPartition,
    load_recurrent_partition,
)

from .baselines import BuyAndHoldPolicy, run_baseline


VALIDATION_BUY_AND_HOLD_VERSION = "validation_buy_and_hold_v1"
VALID_BENCHMARK = "VALID"
INVALID_BENCHMARK = "INVALID"

VALIDATION_BUY_AND_HOLD_CONTRACT = {
    "contract_version": VALIDATION_BUY_AND_HOLD_VERSION,
    "scope": "one symbol's canonical rl_partition_v1 VALIDATION observations only",
    "entry": (
        "observe the first VALIDATION row, then buy the maximum whole-share "
        "position at the next VALIDATION row open"
    ),
    "holding": "hold the position through the remaining VALIDATION observations",
    "exit": (
        "no forced terminal sale; mark the open position at the final VALIDATION close"
    ),
    "missing_dates": (
        "use adjacent stored usable VALIDATION observations; do not fabricate, "
        "forward-fill, or interpolate market bars"
    ),
    "transaction_costs": (
        "use single_symbol_env_v1 defaults: 0.10% commission and 0.05% slippage"
    ),
    "annualization": "existing 252-trading-day episode-metrics methodology",
    "determinism": "BuyAndHoldPolicy with fixed seed 42",
    "test_access": "prohibited; recurrent loader is called only for validation",
}


class ValidationBenchmarkError(RuntimeError):
    """Raised when a benchmark cannot prove VALIDATION-only membership."""


@dataclass(frozen=True)
class ValidationBuyAndHoldResult:
    symbol: str
    benchmark_contract_version: str
    validation_start: str
    validation_end: str
    validation_rows: int
    validation_transition_count: int
    validation_membership_sha256: str
    source_validation_artifact_path: str
    source_validation_artifact_sha256: str
    recurrent_contract_version: str
    environment_version: str
    feature_version: str
    deterministic_seed: int
    initial_cash: float
    commission_rate: float
    slippage_rate: float
    total_return: float | None
    annualized_return: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    maximum_drawdown: float | None
    annualized_volatility: float | None
    final_portfolio_value: float | None
    number_of_trades: int
    total_transaction_costs: float | None
    metric_warnings: tuple[str, ...]
    test_partition_loaded: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validation_membership_hash(data: pd.DataFrame) -> str:
    required = ("symbol", "date", "open", "close")
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValidationBenchmarkError(
            "VALIDATION data is missing benchmark columns: " + ", ".join(missing)
        )
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        raise ValidationBenchmarkError("VALIDATION membership contains invalid dates")
    records = [
        {
            "symbol": str(symbol),
            "date": date_value.date().isoformat(),
            "open": float(open_value),
            "close": float(close_value),
        }
        for symbol, date_value, open_value, close_value in zip(
            data["symbol"], dates, data["open"], data["close"], strict=True
        )
    ]
    serialized = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_validation_buy_and_hold(
    symbol: str,
    *,
    seed: int = 42,
    environment_config: SingleSymbolEnvConfig | None = None,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
) -> ValidationBuyAndHoldResult:
    """Run the canonical baseline on one exact recurrent VALIDATION partition."""

    if seed < 0:
        raise ValidationBenchmarkError("benchmark seed cannot be negative")
    symbol_text = str(symbol).strip()
    loaded = partition_loader(symbol_text, "validation")
    if loaded.partition != "validation":
        raise ValidationBenchmarkError("partition loader returned non-VALIDATION data")
    if loaded.symbol != symbol_text:
        raise ValidationBenchmarkError("partition loader returned a different symbol")
    if not bool(loaded.metadata.test.sealed):
        raise ValidationBenchmarkError("recurrent contract does not keep TEST sealed")
    if len(loaded.data) != loaded.metadata.validation.rows:
        raise ValidationBenchmarkError("VALIDATION row count differs from metadata")
    dates = pd.to_datetime(loaded.data["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValidationBenchmarkError("VALIDATION dates are invalid or unsorted")
    observed_bounds = (
        dates.iloc[0].date().isoformat(),
        dates.iloc[-1].date().isoformat(),
    )
    expected_bounds = (
        loaded.metadata.validation.start,
        loaded.metadata.validation.end,
    )
    if observed_bounds != expected_bounds:
        raise ValidationBenchmarkError("VALIDATION date range differs from metadata")

    config = environment_config or SingleSymbolEnvConfig()
    if config.max_episode_steps is not None:
        raise ValidationBenchmarkError("benchmark requires a complete VALIDATION episode")
    if config.environment_version != loaded.metadata.environment_version:
        raise ValidationBenchmarkError("benchmark environment version is incompatible")

    environment = SingleSymbolTradingEnv(loaded.data, config)
    try:
        baseline = run_baseline(environment, BuyAndHoldPolicy(), seed=seed)
    finally:
        environment.close()
    history = baseline.history
    expected_transitions = len(loaded.data) - 1
    if len(history) != expected_transitions:
        raise ValidationBenchmarkError("benchmark did not consume exact VALIDATION membership")
    observation_dates = pd.to_datetime(history["observation_date"]).reset_index(drop=True)
    execution_dates = pd.to_datetime(history["execution_date"]).reset_index(drop=True)
    if not observation_dates.equals(dates.iloc[:-1].reset_index(drop=True)):
        raise ValidationBenchmarkError("benchmark observation dates differ")
    if not execution_dates.equals(dates.iloc[1:].reset_index(drop=True)):
        raise ValidationBenchmarkError("benchmark execution dates differ")
    if history.empty or int(history["action"].iloc[0]) != 1:
        raise ValidationBenchmarkError("benchmark did not select Buy first")
    if len(history) > 1 and not history["action"].iloc[1:].eq(0).all():
        raise ValidationBenchmarkError("benchmark did not Hold after entry")

    metrics: Mapping[str, object] = baseline.metrics
    warnings = metrics.get("metric_warnings")
    warning_values = (
        tuple(str(value) for value in warnings)
        if isinstance(warnings, (tuple, list))
        else ()
    )
    return ValidationBuyAndHoldResult(
        symbol=symbol_text,
        benchmark_contract_version=VALIDATION_BUY_AND_HOLD_VERSION,
        validation_start=observed_bounds[0],
        validation_end=observed_bounds[1],
        validation_rows=len(loaded.data),
        validation_transition_count=expected_transitions,
        validation_membership_sha256=_validation_membership_hash(loaded.data),
        source_validation_artifact_path=str(loaded.source_artifact_path),
        source_validation_artifact_sha256=sha256_file(loaded.source_artifact_path),
        recurrent_contract_version=loaded.metadata.recurrent_contract_version,
        environment_version=loaded.metadata.environment_version,
        feature_version=loaded.metadata.feature_version,
        deterministic_seed=seed,
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        total_return=_finite(metrics.get("total_return")),
        annualized_return=_finite(metrics.get("annualized_return")),
        sharpe_ratio=_finite(metrics.get("sharpe_ratio")),
        sortino_ratio=_finite(metrics.get("sortino_ratio")),
        maximum_drawdown=_finite(metrics.get("maximum_drawdown")),
        annualized_volatility=_finite(metrics.get("annualized_volatility")),
        final_portfolio_value=_finite(metrics.get("final_portfolio_value")),
        number_of_trades=int(metrics.get("number_of_trades", 0)),
        total_transaction_costs=_finite(metrics.get("total_transaction_costs")),
        metric_warnings=warning_values,
        test_partition_loaded=False,
    )


__all__ = [
    "INVALID_BENCHMARK",
    "VALIDATION_BUY_AND_HOLD_CONTRACT",
    "VALIDATION_BUY_AND_HOLD_VERSION",
    "VALID_BENCHMARK",
    "ValidationBenchmarkError",
    "ValidationBuyAndHoldResult",
    "compute_validation_buy_and_hold",
]
