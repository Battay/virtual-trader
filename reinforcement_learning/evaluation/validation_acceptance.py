"""Frozen pre-TEST acceptance policy for recurrent VALIDATION evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from reinforcement_learning.training.global_validation import (
    VALID,
    rank_validation_inventory,
)

from .comparison import CANDIDATE_CRITERIA_VERSION, CandidateValidationCriteria
from .validation_benchmark import (
    INVALID_BENCHMARK,
    VALIDATION_BUY_AND_HOLD_CONTRACT,
    VALIDATION_BUY_AND_HOLD_VERSION,
    VALID_BENCHMARK,
    ValidationBuyAndHoldResult,
    compute_validation_buy_and_hold,
)


VALIDATION_ACCEPTANCE_POLICY_VERSION = "recurrent_validation_acceptance_v1"
VALIDATION_ACCEPTANCE_FREEZE_SCHEMA = "recurrent_validation_acceptance_freeze_v1"
_EXISTING_CRITERIA = CandidateValidationCriteria()
MINIMUM_VALIDATION_ROWS = _EXISTING_CRITERIA.minimum_validation_observations
MAXIMUM_RL_DRAWDOWN = _EXISTING_CRITERIA.maximum_ppo_drawdown
MAXIMUM_DRAWDOWN_DISADVANTAGE = (
    _EXISTING_CRITERIA.maximum_drawdown_disadvantage_vs_buy_and_hold
)

STRONG_VALIDATION = "STRONG_VALIDATION"
ACCEPTABLE_VALIDATION = "ACCEPTABLE_VALIDATION"
WEAK_VALIDATION = "WEAK_VALIDATION"
INSUFFICIENT_VALIDATION_HISTORY = "INSUFFICIENT_VALIDATION_HISTORY"
INVALID_VALIDATION = "INVALID_VALIDATION"

SUFFICIENT = "SUFFICIENT"
INSUFFICIENT = "INSUFFICIENT"
INVALID_SUFFICIENCY = "INVALID"

DEFAULT_POLICY_FREEZE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "config"
    / "recurrent_validation_acceptance_v1.json"
)

FROZEN_POLICY_RULES = {
    "invalid_validation": (
        "model/validation artifact or Buy-and-Hold benchmark is invalid, "
        "incompatible, contaminated, or missing a required comparison metric"
    ),
    "insufficient_validation_history": (
        "valid comparison with fewer than 126 VALIDATION observations"
    ),
    "strong_validation": (
        "sufficient history; positive RL return, Sharpe, and Sortino; RL maximum "
        "drawdown <= 30%; RL return and Sharpe each meet or exceed Buy-and-Hold; "
        "RL drawdown is no more than 2 percentage points worse than Buy-and-Hold"
    ),
    "acceptable_validation": (
        "sufficient history; positive RL return, Sharpe, and Sortino; RL maximum "
        "drawdown <= 30%; one or more conservative relative Buy-and-Hold gates "
        "may remain unmet and are reported explicitly"
    ),
    "weak_validation": "valid and sufficient comparison that meets neither stronger rule",
}


class ValidationAcceptanceError(RuntimeError):
    """Raised when benchmark/policy evidence cannot remain fail closed."""


@dataclass(frozen=True)
class ValidationAcceptanceSummary:
    strong_validation: int
    acceptable_validation: int
    weak_validation: int
    insufficient_history: int
    invalid: int
    outperforming_buy_and_hold: int
    sharpe_improvement: int
    median_excess_return: float | None
    median_sharpe_delta: float | None


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _benchmark_base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy(deep=True)
    defaults: dict[str, object] = {
        "benchmark_contract_version": VALIDATION_BUY_AND_HOLD_VERSION,
        "benchmark_status": INVALID_BENCHMARK,
        "benchmark_error": None,
        "benchmark_validation_start": None,
        "benchmark_validation_end": None,
        "benchmark_validation_rows": None,
        "benchmark_validation_membership_sha256": None,
        "benchmark_source_artifact_sha256": None,
        "benchmark_total_return": None,
        "benchmark_annualized_return": None,
        "benchmark_sharpe": None,
        "benchmark_sortino": None,
        "benchmark_max_drawdown": None,
        "benchmark_volatility": None,
        "benchmark_final_portfolio_value": None,
        "benchmark_transaction_costs": None,
        "benchmark_metric_warnings": "",
        "excess_return": None,
        "sharpe_delta": None,
        "sortino_delta": None,
        "drawdown_improvement": None,
        "volatility_difference": None,
    }
    for column, default in defaults.items():
        enriched[column] = default
    return enriched


def attach_validation_benchmarks(
    inventory: pd.DataFrame,
    *,
    benchmark_runner: Callable[[str], ValidationBuyAndHoldResult] = (
        compute_validation_buy_and_hold
    ),
) -> pd.DataFrame:
    """Attach deterministic VALIDATION-only benchmarks without touching models."""

    if inventory.empty:
        return _benchmark_base_columns(inventory)
    required = {
        "symbol",
        "validation_artifact_status",
        "validation_start",
        "validation_end",
        "validation_rows",
        "validation_total_return",
        "validation_sharpe",
        "validation_sortino",
        "validation_max_drawdown",
        "validation_volatility",
        "recurrent_contract_version",
        "feature_version",
        "environment_version",
        "test_partition_loaded",
    }
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValidationAcceptanceError(
            "validation inventory is missing benchmark columns: " + ", ".join(missing)
        )
    if inventory["test_partition_loaded"].fillna(False).astype(bool).any():
        raise ValidationAcceptanceError("validation inventory reports TEST access")
    forbidden_test_columns = {
        "test",
        "test_frame",
        "test_dataframe",
        "test_history",
        "test_metrics",
        "test_returns",
        "test_observations",
    }
    contaminated = sorted(
        column
        for column in inventory.columns
        if str(column).strip().casefold() in forbidden_test_columns
    )
    if contaminated:
        raise ValidationAcceptanceError(
            "validation inventory contains prohibited TEST evidence columns: "
            + ", ".join(contaminated)
        )

    enriched = _benchmark_base_columns(inventory)
    for index, row in enriched.iterrows():
        if row["validation_artifact_status"] != VALID:
            enriched.at[index, "benchmark_error"] = (
                "verified validation artifact is not valid; benchmark not computed"
            )
            continue
        try:
            result = benchmark_runner(str(row["symbol"]))
            if result.test_partition_loaded:
                raise ValidationAcceptanceError("benchmark reports TEST access")
            expected = (
                str(row["symbol"]),
                str(row["validation_start"]),
                str(row["validation_end"]),
                int(row["validation_rows"]),
                str(row["recurrent_contract_version"]),
                str(row["feature_version"]),
                str(row["environment_version"]),
            )
            actual = (
                result.symbol,
                result.validation_start,
                result.validation_end,
                result.validation_rows,
                result.recurrent_contract_version,
                result.feature_version,
                result.environment_version,
            )
            if actual != expected:
                raise ValidationAcceptanceError(
                    "benchmark VALIDATION membership/provenance differs"
                )
            required_benchmark = (
                result.total_return,
                result.sharpe_ratio,
                result.maximum_drawdown,
                result.annualized_volatility,
            )
            if any(_finite(value) is None for value in required_benchmark):
                raise ValidationAcceptanceError(
                    "benchmark required comparison metrics are unavailable"
                )
            enriched.at[index, "benchmark_status"] = VALID_BENCHMARK
            enriched.at[index, "benchmark_error"] = None
            enriched.at[index, "benchmark_validation_start"] = result.validation_start
            enriched.at[index, "benchmark_validation_end"] = result.validation_end
            enriched.at[index, "benchmark_validation_rows"] = result.validation_rows
            enriched.at[index, "benchmark_validation_membership_sha256"] = (
                result.validation_membership_sha256
            )
            enriched.at[index, "benchmark_source_artifact_sha256"] = (
                result.source_validation_artifact_sha256
            )
            enriched.at[index, "benchmark_total_return"] = result.total_return
            enriched.at[index, "benchmark_annualized_return"] = result.annualized_return
            enriched.at[index, "benchmark_sharpe"] = result.sharpe_ratio
            enriched.at[index, "benchmark_sortino"] = result.sortino_ratio
            enriched.at[index, "benchmark_max_drawdown"] = result.maximum_drawdown
            enriched.at[index, "benchmark_volatility"] = result.annualized_volatility
            enriched.at[index, "benchmark_final_portfolio_value"] = (
                result.final_portfolio_value
            )
            enriched.at[index, "benchmark_transaction_costs"] = (
                result.total_transaction_costs
            )
            enriched.at[index, "benchmark_metric_warnings"] = "; ".join(
                result.metric_warnings
            )
            pairs = {
                "excess_return": (
                    row["validation_total_return"], result.total_return, "subtract"
                ),
                "sharpe_delta": (
                    row["validation_sharpe"], result.sharpe_ratio, "subtract"
                ),
                "sortino_delta": (
                    row["validation_sortino"], result.sortino_ratio, "subtract"
                ),
                "drawdown_improvement": (
                    result.maximum_drawdown,
                    row["validation_max_drawdown"],
                    "subtract",
                ),
                "volatility_difference": (
                    row["validation_volatility"],
                    result.annualized_volatility,
                    "subtract",
                ),
            }
            for column, (left, right, _) in pairs.items():
                left_value = _finite(left)
                right_value = _finite(right)
                enriched.at[index, column] = (
                    left_value - right_value
                    if left_value is not None and right_value is not None
                    else None
                )
        except (OSError, ValueError, RuntimeError, ValidationAcceptanceError) as exc:
            enriched.at[index, "benchmark_status"] = INVALID_BENCHMARK
            enriched.at[index, "benchmark_error"] = f"{type(exc).__name__}: {exc}"
    if "comparability_warnings" in enriched.columns:
        superseded = {"benchmark_metric_unavailable", "acceptance_flag_unavailable"}
        enriched["comparability_warnings"] = enriched[
            "comparability_warnings"
        ].fillna("").map(
            lambda value: "; ".join(
                warning.strip()
                for warning in str(value).split(";")
                if warning.strip() and warning.strip() not in superseded
            )
        )
    return rank_validation_inventory(enriched)


def _classification(row: Mapping[str, object]) -> tuple[str, str, tuple[str, ...]]:
    if (
        row.get("validation_artifact_status") != VALID
        or row.get("benchmark_status") != VALID_BENCHMARK
    ):
        return (
            INVALID_VALIDATION,
            INVALID_SUFFICIENCY,
            ("Validation or benchmark artifact is invalid or unavailable.",),
        )
    rows = int(row.get("validation_rows", 0))
    if rows < MINIMUM_VALIDATION_ROWS:
        return (
            INSUFFICIENT_VALIDATION_HISTORY,
            INSUFFICIENT,
            (
                f"Validation has {rows} rows; frozen minimum is "
                f"{MINIMUM_VALIDATION_ROWS}.",
            ),
        )

    required_names = (
        "validation_total_return",
        "validation_sharpe",
        "validation_sortino",
        "validation_max_drawdown",
        "benchmark_total_return",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "excess_return",
        "sharpe_delta",
        "drawdown_improvement",
    )
    values = {name: _finite(row.get(name)) for name in required_names}
    unavailable = tuple(name for name, value in values.items() if value is None)
    if unavailable:
        return (
            INVALID_VALIDATION,
            INVALID_SUFFICIENCY,
            ("Required metrics unavailable: " + ", ".join(unavailable),),
        )
    numeric = {name: float(value) for name, value in values.items()}
    gates = {
        "positive RL return": numeric["validation_total_return"] > 0,
        "positive RL Sharpe": numeric["validation_sharpe"] > 0,
        "positive RL Sortino": numeric["validation_sortino"] > 0,
        "RL maximum drawdown <= 30%": (
            numeric["validation_max_drawdown"] <= MAXIMUM_RL_DRAWDOWN
        ),
        "RL return >= Buy-and-Hold": numeric["excess_return"] >= 0,
        "RL Sharpe >= Buy-and-Hold": numeric["sharpe_delta"] >= 0,
        "RL drawdown no more than 2pp worse": (
            numeric["drawdown_improvement"] >= -MAXIMUM_DRAWDOWN_DISADVANTAGE
        ),
    }
    standalone = all(
        gates[name]
        for name in (
            "positive RL return",
            "positive RL Sharpe",
            "positive RL Sortino",
            "RL maximum drawdown <= 30%",
        )
    )
    relative_return = gates["RL return >= Buy-and-Hold"]
    relative_sharpe = gates["RL Sharpe >= Buy-and-Hold"]
    drawdown_safe = gates["RL drawdown no more than 2pp worse"]
    if standalone and relative_return and relative_sharpe and drawdown_safe:
        return (
            STRONG_VALIDATION,
            SUFFICIENT,
            ("All frozen standalone and relative validation gates passed.",),
        )
    if standalone:
        relative_failures = tuple(
            name
            for name in (
                "RL return >= Buy-and-Hold",
                "RL Sharpe >= Buy-and-Hold",
                "RL drawdown no more than 2pp worse",
            )
            if not gates[name]
        )
        return (
            ACCEPTABLE_VALIDATION,
            SUFFICIENT,
            (
                "All frozen standalone validation gates passed. Conservative "
                "relative gates not met: " + ", ".join(relative_failures) + ".",
            ),
        )
    failed = tuple(name for name, passed in gates.items() if not passed)
    return (
        WEAK_VALIDATION,
        SUFFICIENT,
        tuple(f"Failed frozen gate: {name}." for name in failed),
    )


def apply_frozen_acceptance_policy(inventory: pd.DataFrame) -> pd.DataFrame:
    classified = inventory.copy(deep=True)
    classifications = [
        _classification(row)
        for row in classified.to_dict(orient="records")
    ]
    classified["acceptance_classification"] = [item[0] for item in classifications]
    classified["validation_sufficiency"] = [item[1] for item in classifications]
    classified["acceptance_reasons"] = [
        " ".join(item[2]) for item in classifications
    ]
    classified["acceptance_policy_version"] = VALIDATION_ACCEPTANCE_POLICY_VERSION
    return classified


def summarize_acceptance(inventory: pd.DataFrame) -> ValidationAcceptanceSummary:
    counts = inventory["acceptance_classification"].value_counts()
    valid_benchmarks = inventory.loc[inventory["benchmark_status"].eq(VALID_BENCHMARK)]
    excess = pd.to_numeric(valid_benchmarks["excess_return"], errors="coerce").dropna()
    sharpe = pd.to_numeric(valid_benchmarks["sharpe_delta"], errors="coerce").dropna()
    return ValidationAcceptanceSummary(
        strong_validation=int(counts.get(STRONG_VALIDATION, 0)),
        acceptable_validation=int(counts.get(ACCEPTABLE_VALIDATION, 0)),
        weak_validation=int(counts.get(WEAK_VALIDATION, 0)),
        insufficient_history=int(counts.get(INSUFFICIENT_VALIDATION_HISTORY, 0)),
        invalid=int(counts.get(INVALID_VALIDATION, 0)),
        outperforming_buy_and_hold=int(excess.gt(0).sum()),
        sharpe_improvement=int(sharpe.gt(0).sum()),
        median_excess_return=float(excess.median()) if not excess.empty else None,
        median_sharpe_delta=float(sharpe.median()) if not sharpe.empty else None,
    )


def _canonical_hash(records: list[dict[str, object]]) -> str:
    serialized = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def model_inventory_hash(inventory: pd.DataFrame) -> str:
    fields = (
        "symbol",
        "run_type",
        "run_id",
        "attempt",
        "model_sha256",
        "validation_artifact_sha256",
        "validation_start",
        "validation_end",
        "validation_rows",
        "partition_contract_version",
        "recurrent_contract_version",
        "feature_version",
        "environment_version",
    )
    records = [
        {field: row.get(field) for field in fields}
        for row in inventory.sort_values("symbol", kind="mergesort").to_dict(
            orient="records"
        )
    ]
    return _canonical_hash(records)


def benchmark_result_hash(inventory: pd.DataFrame) -> str:
    fields = (
        "symbol",
        "benchmark_contract_version",
        "benchmark_validation_membership_sha256",
        "benchmark_source_artifact_sha256",
        "benchmark_total_return",
        "benchmark_annualized_return",
        "benchmark_sharpe",
        "benchmark_sortino",
        "benchmark_max_drawdown",
        "benchmark_volatility",
    )
    records = [
        {field: row.get(field) for field in fields}
        for row in inventory.sort_values("symbol", kind="mergesort").to_dict(
            orient="records"
        )
    ]
    return _canonical_hash(records)


def classification_result_hash(inventory: pd.DataFrame) -> str:
    records = [
        {
            "symbol": row.get("symbol"),
            "acceptance_classification": row.get("acceptance_classification"),
            "validation_sufficiency": row.get("validation_sufficiency"),
            "acceptance_reasons": row.get("acceptance_reasons"),
        }
        for row in inventory.sort_values("symbol", kind="mergesort").to_dict(
            orient="records"
        )
    ]
    return _canonical_hash(records)


def _frozen_thresholds() -> dict[str, float | int]:
    return {
        "minimum_validation_rows": MINIMUM_VALIDATION_ROWS,
        "rl_return_strictly_greater_than": 0.0,
        "rl_sharpe_strictly_greater_than": (
            _EXISTING_CRITERIA.minimum_ppo_sharpe
        ),
        "rl_sortino_strictly_greater_than": (
            _EXISTING_CRITERIA.minimum_ppo_sortino
        ),
        "maximum_rl_drawdown": MAXIMUM_RL_DRAWDOWN,
        "minimum_excess_return": (
            _EXISTING_CRITERIA.minimum_return_advantage_vs_buy_and_hold
        ),
        "minimum_sharpe_delta": (
            _EXISTING_CRITERIA.minimum_sharpe_advantage_vs_buy_and_hold
        ),
        "minimum_drawdown_improvement": -MAXIMUM_DRAWDOWN_DISADVANTAGE,
    }


def build_policy_freeze_payload(
    inventory: pd.DataFrame,
    *,
    frozen_at: str,
) -> dict[str, object]:
    summary = summarize_acceptance(inventory)
    return {
        "artifact_schema_version": VALIDATION_ACCEPTANCE_FREEZE_SCHEMA,
        "policy_version": VALIDATION_ACCEPTANCE_POLICY_VERSION,
        "source_criteria_version": CANDIDATE_CRITERIA_VERSION,
        "benchmark_contract_version": VALIDATION_BUY_AND_HOLD_VERSION,
        "frozen_at": frozen_at,
        "frozen_before_test": True,
        "test_status_at_freeze": "SEALED",
        "test_observations_accessed": False,
        "thresholds": _frozen_thresholds(),
        "classification_rules": dict(FROZEN_POLICY_RULES),
        "benchmark_methodology": dict(VALIDATION_BUY_AND_HOLD_CONTRACT),
        "model_inventory": {
            "count": len(inventory),
            "hash": model_inventory_hash(inventory),
            "symbols": sorted(inventory["symbol"].astype(str).tolist()),
        },
        "benchmark_result_hash": benchmark_result_hash(inventory),
        "classification_result_hash": classification_result_hash(inventory),
        "classification_counts": asdict(summary),
        "statement": (
            "Policy and classifications were frozen from TRAIN/VALIDATION-only "
            "evidence before any TEST evaluation. No TEST-derived information is "
            "present in this artifact."
        ),
    }


def load_policy_freeze(
    path: Path = DEFAULT_POLICY_FREEZE_PATH,
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationAcceptanceError(f"policy freeze is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationAcceptanceError("policy freeze must contain an object")
    return payload


def validate_policy_freeze(
    payload: Mapping[str, object], inventory: pd.DataFrame
) -> None:
    expected = {
        "artifact_schema_version": VALIDATION_ACCEPTANCE_FREEZE_SCHEMA,
        "policy_version": VALIDATION_ACCEPTANCE_POLICY_VERSION,
        "source_criteria_version": CANDIDATE_CRITERIA_VERSION,
        "benchmark_contract_version": VALIDATION_BUY_AND_HOLD_VERSION,
        "frozen_before_test": True,
        "test_status_at_freeze": "SEALED",
        "test_observations_accessed": False,
        "thresholds": _frozen_thresholds(),
        "classification_rules": dict(FROZEN_POLICY_RULES),
        "benchmark_methodology": dict(VALIDATION_BUY_AND_HOLD_CONTRACT),
        "benchmark_result_hash": benchmark_result_hash(inventory),
        "classification_result_hash": classification_result_hash(inventory),
        "classification_counts": asdict(summarize_acceptance(inventory)),
    }
    differences = [
        key for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    model = payload.get("model_inventory")
    if not isinstance(model, Mapping):
        differences.append("model_inventory")
    else:
        if int(model.get("count", -1)) != len(inventory):
            differences.append("model_inventory.count")
        if model.get("hash") != model_inventory_hash(inventory):
            differences.append("model_inventory.hash")
        if model.get("symbols") != sorted(inventory["symbol"].astype(str).tolist()):
            differences.append("model_inventory.symbols")
    frozen_at = payload.get("frozen_at")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        differences.append("frozen_at")
    if differences:
        raise ValidationAcceptanceError(
            "policy freeze differs from current pre-TEST evidence: "
            + ", ".join(sorted(differences))
        )


__all__ = [
    "ACCEPTABLE_VALIDATION",
    "DEFAULT_POLICY_FREEZE_PATH",
    "FROZEN_POLICY_RULES",
    "INSUFFICIENT_VALIDATION_HISTORY",
    "INVALID_VALIDATION",
    "MINIMUM_VALIDATION_ROWS",
    "STRONG_VALIDATION",
    "VALIDATION_ACCEPTANCE_FREEZE_SCHEMA",
    "VALIDATION_ACCEPTANCE_POLICY_VERSION",
    "ValidationAcceptanceError",
    "ValidationAcceptanceSummary",
    "WEAK_VALIDATION",
    "apply_frozen_acceptance_policy",
    "attach_validation_benchmarks",
    "benchmark_result_hash",
    "build_policy_freeze_payload",
    "classification_result_hash",
    "load_policy_freeze",
    "model_inventory_hash",
    "summarize_acceptance",
    "validate_policy_freeze",
]
