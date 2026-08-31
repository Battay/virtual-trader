"""Read-only comparison of persisted recurrent VALIDATION artifacts.

The comparison never loads a partition dataframe, model, or evaluator. Missing
or invalid artifacts remain visible as explicit inventory rows and are never
silently recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from reinforcement_learning.integrity import sha256_file

from .model_details import build_global_verified_model_inventory


GLOBAL_VALIDATION_COMPARISON_VERSION = "global_recurrent_validation_comparison_v1"
EXISTING_MINIMUM_VALIDATION_OBSERVATIONS = 126

VALID = "VALID"
MISSING = "MISSING"
INVALID = "INVALID"

RANK_COLUMNS = (
    "return_rank",
    "sharpe_rank",
    "sortino_rank",
    "drawdown_rank",
    "excess_return_rank",
)

EXPORT_COLUMNS = (
    "symbol",
    "company_name",
    "sector",
    "run_type",
    "run_id",
    "attempt",
    "algorithm",
    "policy",
    "trainer_version",
    "artifact_verification",
    "validation_status",
    "validation_artifact_status",
    "validation_error",
    "validation_start",
    "validation_end",
    "validation_rows",
    "validation_total_return",
    "benchmark_total_return",
    "excess_return",
    "validation_sharpe",
    "validation_sortino",
    "validation_max_drawdown",
    "validation_volatility",
    "trade_count",
    "turnover",
    "win_rate",
    "average_reward",
    "final_portfolio_value",
    "acceptance_status",
    "acceptance_passed",
    *RANK_COLUMNS,
    "comparability_warnings",
    "training_diagnostics_available",
    "partition_contract_version",
    "recurrent_contract_version",
    "feature_version",
    "environment_version",
    "split_policy_version",
    "scaler_fit_partition",
    "train_start",
    "train_end",
    "train_rows",
    "model_path",
    "model_sha256",
    "validation_metrics_reference",
    "validation_artifact_sha256",
    "source_contract_sha256",
    "hyperparameters_hash",
    "seed",
    "TEST_status",
    "test_partition_loaded",
    "comparison_version",
)


class GlobalValidationError(RuntimeError):
    """Raised for invalid comparison inputs, never for one bad artifact row."""


@dataclass(frozen=True)
class GlobalValidationSummary:
    verified_models: int
    validation_complete: int
    validation_missing_or_invalid: int
    positive_validation_return: int
    positive_excess_return: int | None
    median_return: float | None
    median_sharpe: float | None
    median_max_drawdown: float | None


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalValidationError(f"unreadable JSON artifact: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GlobalValidationError(f"JSON artifact must contain an object: {path}")
    return payload


def _safe_artifact_path(run_directory: object, reference: object) -> Path:
    root = Path(str(run_directory)).resolve()
    reference_text = str(reference or "").strip()
    if not reference_text:
        raise FileNotFoundError("validation artifact reference is missing")
    candidate = (root / reference_text).resolve()
    if candidate != root and root not in candidate.parents:
        raise GlobalValidationError("validation artifact path escapes its run directory")
    if candidate.name in {"test.csv", "test_rl.csv"}:
        raise GlobalValidationError("TEST artifacts are prohibited")
    return candidate


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GlobalValidationError(f"{label} is missing or malformed")
    return value


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_metric(metrics: Mapping[str, object], name: str) -> float | None:
    return _finite_number(metrics.get(name))


def _forbidden_test_payload(payload: object) -> bool:
    """Reject persisted TEST values while allowing explicit false seal flags."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().casefold()
            if normalized in {
                "test",
                "test_frame",
                "test_dataframe",
                "test_history",
                "test_metrics",
                "test_returns",
                "test_observations",
            }:
                return True
            if normalized in {
                "test_evaluated",
                "test_evaluation_performed",
                "test_partition_loaded",
            } and value is True:
                return True
            if _forbidden_test_payload(value):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_forbidden_test_payload(item) for item in payload)
    return False


def _benchmark_metrics(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    for key in ("buy_and_hold", "buy_and_hold_result", "benchmark_result"):
        result = payload.get(key)
        if not isinstance(result, Mapping):
            continue
        metrics = result.get("metrics", result)
        if isinstance(metrics, Mapping):
            return metrics
    return None


def _acceptance(payload: Mapping[str, object]) -> tuple[str | None, bool | None]:
    decision = payload.get("candidate_decision")
    if not isinstance(decision, Mapping):
        return None, None
    status = str(decision.get("status") or "").strip() or None
    passed = decision.get("passed")
    return status, passed if isinstance(passed, bool) else None


def _training_diagnostics(
    row: Mapping[str, object],
    *,
    json_loader: Callable[[Path], dict[str, object]],
) -> bool:
    attempt_index = max(0, int(row.get("attempt", 1)) - 1)
    path = (
        Path(str(row["run_directory"]))
        / "logs"
        / str(row["symbol"])
        / f"attempt_{attempt_index:03d}.json"
    )
    try:
        payload = json_loader(path)
    except (OSError, ValueError, RuntimeError, GlobalValidationError):
        return False
    diagnostics = payload.get("training_diagnostics")
    return isinstance(diagnostics, Mapping) and bool(diagnostics)


def _base_row(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "symbol": str(source.get("symbol", "")),
        "company_name": source.get("company_name"),
        "sector": source.get("sector"),
        "run_type": source.get("run_type"),
        "run_id": source.get("run_id"),
        "attempt": source.get("attempt"),
        "algorithm": source.get("algorithm"),
        "policy": source.get("policy"),
        "trainer_version": source.get("trainer_version"),
        "artifact_verification": source.get("artifact_verification"),
        "validation_status": source.get("validation_status"),
        "validation_artifact_status": MISSING,
        "validation_error": None,
        "validation_start": source.get("validation_start"),
        "validation_end": source.get("validation_end"),
        "validation_rows": source.get("validation_rows"),
        "validation_total_return": None,
        "benchmark_total_return": None,
        "excess_return": None,
        "validation_sharpe": None,
        "validation_sortino": None,
        "validation_max_drawdown": None,
        "validation_volatility": None,
        "trade_count": None,
        "turnover": None,
        "win_rate": None,
        "average_reward": None,
        "final_portfolio_value": None,
        "acceptance_status": None,
        "acceptance_passed": None,
        "comparability_warnings": "",
        "training_diagnostics_available": False,
        "partition_contract_version": source.get("partition_contract_version"),
        "recurrent_contract_version": source.get("recurrent_contract_version"),
        "feature_version": source.get("feature_version"),
        "environment_version": source.get("environment_version"),
        "split_policy_version": source.get("split_policy_version"),
        "scaler_fit_partition": source.get("scaler_fit_partition"),
        "train_start": source.get("train_start"),
        "train_end": source.get("train_end"),
        "train_rows": source.get("train_rows"),
        "model_path": source.get("model_path"),
        "model_sha256": source.get("model_sha256"),
        "validation_metrics_reference": source.get("validation_metrics_reference"),
        "run_directory": source.get("run_directory"),
        "validation_artifact_sha256": None,
        "source_contract_sha256": source.get("source_contract_sha256"),
        "hyperparameters_hash": source.get("hyperparameters_hash"),
        "seed": source.get("seed"),
        "TEST_status": "SEALED",
        "test_partition_loaded": False,
        "comparison_version": GLOBAL_VALIDATION_COMPARISON_VERSION,
    }


def _validate_artifact(
    payload: Mapping[str, object], source: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
    if _forbidden_test_payload(payload):
        raise GlobalValidationError("validation artifact contains or reports TEST data")
    expected = {
        "symbol": source.get("symbol"),
        "evaluation_partition": "validation",
        "recurrent_contract_version": source.get("recurrent_contract_version"),
        "feature_version": source.get("feature_version"),
        "environment_version": source.get("environment_version"),
        "validation_start": source.get("validation_start"),
        "validation_end": source.get("validation_end"),
    }
    differences = [
        key for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    try:
        row_count_matches = int(payload.get("validation_rows", -1)) == int(
            source.get("validation_rows", -2)
        )
    except (TypeError, ValueError):
        row_count_matches = False
    if not row_count_matches:
        differences.append("validation_rows")
    if differences:
        raise GlobalValidationError(
            "validation provenance differs: " + ", ".join(sorted(differences))
        )
    if payload.get("model_parameters_unchanged") is not True:
        raise GlobalValidationError("validation does not prove unchanged model parameters")
    if payload.get("parameter_hash_before") != payload.get("parameter_hash_after"):
        raise GlobalValidationError("validation parameter hashes differ")
    if payload.get("model_timesteps_before") != payload.get("model_timesteps_after"):
        raise GlobalValidationError("validation model timesteps differ")
    strategy = _mapping(payload.get("strategy_result", payload.get("ppo")), label="strategy result")
    metrics = _mapping(strategy.get("metrics"), label="strategy metrics")
    if strategy.get("strategy") not in {"RecurrentPPO", "PPO"}:
        raise GlobalValidationError("validation strategy is not recurrent PPO")
    return metrics, _benchmark_metrics(payload)


def rank_validation_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Add transparent metric-specific ranks and deterministic display order."""

    ranked = inventory.copy(deep=True)
    specifications = {
        "return_rank": ("validation_total_return", False),
        "sharpe_rank": ("validation_sharpe", False),
        "sortino_rank": ("validation_sortino", False),
        "drawdown_rank": ("validation_max_drawdown", True),
        "excess_return_rank": ("excess_return", False),
    }
    valid = ranked["validation_artifact_status"].eq(VALID)
    for rank_column, (metric, ascending) in specifications.items():
        numeric = pd.to_numeric(ranked[metric], errors="coerce").where(valid)
        ranked[rank_column] = numeric.rank(
            method="min", ascending=ascending, na_option="keep"
        ).astype("Int64")
    ranked["_valid_sort"] = valid.astype(int)
    ranked["_sharpe_sort"] = pd.to_numeric(
        ranked["validation_sharpe"], errors="coerce"
    )
    ranked = ranked.sort_values(
        ["_valid_sort", "_sharpe_sort", "symbol"],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_valid_sort", "_sharpe_sort"])
    return ranked.reset_index(drop=True)


def build_global_validation_inventory(
    *,
    verified_inventory: pd.DataFrame | None = None,
    json_loader: Callable[[Path], dict[str, object]] = _read_json_object,
) -> pd.DataFrame:
    """Extract persisted VALIDATION metrics without evaluating or loading data."""

    source = (
        build_global_verified_model_inventory()
        if verified_inventory is None
        else verified_inventory.copy(deep=True)
    )
    if source.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    required = {
        "symbol",
        "run_directory",
        "validation_metrics_reference",
        "validation_start",
        "validation_end",
        "validation_rows",
        "recurrent_contract_version",
        "feature_version",
        "environment_version",
    }
    missing_columns = sorted(required.difference(source.columns))
    if missing_columns:
        raise GlobalValidationError(
            "verified inventory is missing columns: " + ", ".join(missing_columns)
        )
    if source["symbol"].astype(str).duplicated().any():
        raise GlobalValidationError("verified validation inventory contains duplicate symbols")
    if (
        "artifact_verification" in source.columns
        and not source["artifact_verification"].astype(str).eq("verified").all()
    ):
        raise GlobalValidationError("comparison input contains an unverified model artifact")
    if (
        "test_partition_loaded" in source.columns
        and source["test_partition_loaded"].fillna(False).astype(bool).any()
    ):
        raise GlobalValidationError("comparison input reports TEST partition access")

    rows: list[dict[str, object]] = []
    for source_row in source.sort_values("symbol", kind="mergesort").to_dict(
        orient="records"
    ):
        row = _base_row(source_row)
        warnings: list[str] = []
        try:
            artifact_path = _safe_artifact_path(
                source_row["run_directory"],
                source_row["validation_metrics_reference"],
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(f"validation artifact is missing: {artifact_path}")
            payload = json_loader(artifact_path)
            metrics, benchmark = _validate_artifact(payload, source_row)
            row["validation_artifact_status"] = VALID
            row["validation_artifact_sha256"] = sha256_file(artifact_path)
            row["validation_total_return"] = _optional_metric(metrics, "total_return")
            row["validation_sharpe"] = _optional_metric(metrics, "sharpe_ratio")
            row["validation_sortino"] = _optional_metric(metrics, "sortino_ratio")
            row["validation_max_drawdown"] = _optional_metric(
                metrics, "maximum_drawdown"
            )
            row["validation_volatility"] = _optional_metric(
                metrics, "annualized_volatility"
            )
            row["trade_count"] = _optional_metric(metrics, "number_of_trades")
            row["turnover"] = _optional_metric(metrics, "turnover")
            row["win_rate"] = _optional_metric(
                metrics, "completed_trade_win_rate"
            )
            row["average_reward"] = _optional_metric(metrics, "average_reward")
            row["final_portfolio_value"] = _optional_metric(
                metrics, "final_portfolio_value"
            )
            if benchmark is not None:
                row["benchmark_total_return"] = _optional_metric(
                    benchmark, "total_return"
                )
            if (
                row["validation_total_return"] is not None
                and row["benchmark_total_return"] is not None
            ):
                row["excess_return"] = (
                    float(row["validation_total_return"])
                    - float(row["benchmark_total_return"])
                )
            status, passed = _acceptance(payload)
            row["acceptance_status"] = status
            row["acceptance_passed"] = passed
            metric_warnings = metrics.get("metric_warnings")
            if isinstance(metric_warnings, list):
                warnings.extend(f"metric:{value}" for value in metric_warnings)
        except FileNotFoundError as exc:
            row["validation_artifact_status"] = MISSING
            row["validation_error"] = str(exc)
            warnings.append("missing_validation_artifact")
        except (OSError, ValueError, RuntimeError, GlobalValidationError) as exc:
            row["validation_artifact_status"] = INVALID
            row["validation_error"] = f"{type(exc).__name__}: {exc}"
            warnings.append("invalid_validation_artifact")

        diagnostics = _training_diagnostics(source_row, json_loader=json_loader)
        row["training_diagnostics_available"] = diagnostics
        if not diagnostics:
            warnings.append("missing_training_diagnostics")
        validation_rows = _finite_number(row.get("validation_rows"))
        if (
            validation_rows is not None
            and validation_rows < EXISTING_MINIMUM_VALIDATION_OBSERVATIONS
        ):
            warnings.append("short_validation_history_below_existing_126_row_rule")
        if row["validation_artifact_status"] == VALID:
            if row["benchmark_total_return"] is None:
                warnings.append("benchmark_metric_unavailable")
            if row["turnover"] is None:
                warnings.append("turnover_unavailable")
            if row["average_reward"] is None:
                warnings.append("average_reward_unavailable")
            if row["acceptance_status"] is None:
                warnings.append("acceptance_flag_unavailable")
        row["comparability_warnings"] = "; ".join(dict.fromkeys(warnings))
        rows.append(row)

    inventory = pd.DataFrame(rows)
    valid_rows = pd.to_numeric(
        inventory.loc[inventory["validation_artifact_status"].eq(VALID), "validation_rows"],
        errors="coerce",
    )
    median_rows = valid_rows.median() if not valid_rows.empty else float("nan")
    if pd.notna(median_rows) and float(median_rows) > 0:
        for index, value in pd.to_numeric(
            inventory["validation_rows"], errors="coerce"
        ).items():
            if pd.isna(value) or inventory.at[index, "validation_artifact_status"] != VALID:
                continue
            ratio = float(value) / float(median_rows)
            if ratio < 0.75 or ratio > 1.25:
                existing = str(inventory.at[index, "comparability_warnings"] or "")
                warning = "validation_rows_materially_different_from_median"
                inventory.at[index, "comparability_warnings"] = (
                    f"{existing}; {warning}" if existing else warning
                )
    return rank_validation_inventory(inventory)


def summarize_validation_inventory(
    inventory: pd.DataFrame,
) -> GlobalValidationSummary:
    valid = inventory.loc[inventory["validation_artifact_status"].eq(VALID)]

    def median(column: str) -> float | None:
        values = pd.to_numeric(valid[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else None

    excess = pd.to_numeric(valid["excess_return"], errors="coerce").dropna()
    returns = pd.to_numeric(valid["validation_total_return"], errors="coerce")
    return GlobalValidationSummary(
        verified_models=len(inventory),
        validation_complete=len(valid),
        validation_missing_or_invalid=len(inventory) - len(valid),
        positive_validation_return=int(returns.gt(0).sum()),
        positive_excess_return=int(excess.gt(0).sum()) if not excess.empty else None,
        median_return=median("validation_total_return"),
        median_sharpe=median("validation_sharpe"),
        median_max_drawdown=median("validation_max_drawdown"),
    )


def build_sector_validation_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive sector aggregates from valid persisted artifacts."""

    valid = inventory.loc[inventory["validation_artifact_status"].eq(VALID)].copy()
    columns = (
        "sector",
        "model_count",
        "median_validation_return",
        "median_sharpe",
        "median_max_drawdown",
        "positive_return_count",
    )
    if valid.empty:
        return pd.DataFrame(columns=columns)
    valid["sector"] = valid["sector"].fillna("UNKNOWN").astype(str)
    valid["positive_return"] = pd.to_numeric(
        valid["validation_total_return"], errors="coerce"
    ).gt(0)
    result = (
        valid.groupby("sector", sort=True, dropna=False)
        .agg(
            model_count=("symbol", "size"),
            median_validation_return=("validation_total_return", "median"),
            median_sharpe=("validation_sharpe", "median"),
            median_max_drawdown=("validation_max_drawdown", "median"),
            positive_return_count=("positive_return", "sum"),
        )
        .reset_index()
    )
    return result.loc[:, list(columns)].sort_values(
        ["model_count", "sector"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def validation_export_frame(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic, TEST-sealed research-export columns."""

    export = inventory.copy(deep=True)
    for column in EXPORT_COLUMNS:
        if column not in export.columns:
            export[column] = None
    export["TEST_status"] = "SEALED"
    export["test_partition_loaded"] = False
    return export.loc[:, list(EXPORT_COLUMNS)].sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)


def validation_export_csv(inventory: pd.DataFrame) -> bytes:
    return validation_export_frame(inventory).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")


def load_persisted_validation_returns(
    run_directory: object,
    validation_reference: object,
    *,
    json_loader: Callable[[Path], dict[str, object]] = _read_json_object,
) -> pd.DataFrame:
    """Load only a persisted VALIDATION daily-return list for optional plotting."""

    path = _safe_artifact_path(run_directory, validation_reference)
    payload = json_loader(path)
    if _forbidden_test_payload(payload) or payload.get("evaluation_partition") != "validation":
        raise GlobalValidationError("persisted return series is not VALIDATION-only")
    strategy = _mapping(payload.get("strategy_result", payload.get("ppo")), label="strategy result")
    metrics = _mapping(strategy.get("metrics"), label="strategy metrics")
    values = metrics.get("daily_returns")
    if not isinstance(values, list):
        return pd.DataFrame(columns=["validation_step", "cumulative_return"])
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise GlobalValidationError("persisted validation returns are malformed")
    return pd.DataFrame(
        {
            "validation_step": range(1, len(numeric) + 1),
            "cumulative_return": (1.0 + numeric).cumprod() - 1.0,
        }
    )


__all__ = [
    "EXPORT_COLUMNS",
    "GLOBAL_VALIDATION_COMPARISON_VERSION",
    "GlobalValidationError",
    "GlobalValidationSummary",
    "INVALID",
    "MISSING",
    "VALID",
    "build_global_validation_inventory",
    "build_sector_validation_summary",
    "load_persisted_validation_returns",
    "rank_validation_inventory",
    "summarize_validation_inventory",
    "validation_export_csv",
    "validation_export_frame",
]
