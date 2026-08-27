"""Predeclared multi-symbol/multi-seed RecurrentPPO budget study.

The study is deliberately separate from production orchestration.  Every run
trains from TRAIN, evaluates only VALIDATION, returns scalar diagnostics, and
discards the in-memory model.  TEST is not available through this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from data_pipeline.src.config import (
    CANONICAL_RECURRENT_TRAIN_V2_DIR,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
)
from feature_engineering.storage import atomic_write_json, safe_path_component
from reinforcement_learning.canonical_recurrent_artifacts import (
    load_training_recurrent_partition,
)
from reinforcement_learning.evaluation.recurrent_evaluator import (
    evaluate_recurrent_on_validation,
)

from .job_state import canonical_hash
from .recurrent_config import RECURRENT_PPO_CONFIG_VERSION, RecurrentPPOConfig
from .recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    RecurrentUniverseDiscovery,
    discover_recurrent_training_universe,
)
from .recurrent_trainer import RECURRENT_TRAINER_VERSION, train_recurrent_single_symbol


BUDGET_STUDY_VERSION = "recurrent_budget_study_v1"
REPRESENTATIVE_SELECTION_VERSION = "train_regime_archetypes_v1"
BUDGET_DECISION_VERSION = "smallest_technically_mature_v1"
STUDY_SEEDS = (42, 123, 2026)
STUDY_BUDGETS = (50_000, 100_000, 250_000)
STUDY_SYMBOL_COUNT = 6
STUDY_RUN_COUNT = STUDY_SYMBOL_COUNT * len(STUDY_SEEDS) * len(STUDY_BUDGETS)
WORKER_MARKER = "RECURRENT_BUDGET_STUDY_RESULT="
SUCCESS = "completed"
FAILED = "failed"

# Declared before training. Validation return levels are not inputs.
MIN_SUCCESSFUL_RUNS_PER_BUDGET = 17
MIN_SUCCESSFUL_SEEDS_PER_SYMBOL_BUDGET = 2
MATURE_DIAGNOSTIC_FRACTION = 0.90
MAX_MATURE_APPROXIMATE_KL = 0.10
MAX_MATURE_CLIP_FRACTION = 0.30
MIN_MATURE_MEDIAN_EXPLAINED_VARIANCE = 0.0
MATERIAL_EXPLAINED_VARIANCE_DELTA = 0.05
MATERIAL_VALUE_LOSS_REDUCTION = 0.20
WIDESPREAD_PAIR_COUNT = 12

ARCHETYPE_TARGETS: tuple[tuple[str, Mapping[str, float]], ...] = (
    (
        "long_history_high_coverage",
        {"history_rank": 0.95, "coverage_rank": 0.95, "liquidity_rank": 0.55},
    ),
    (
        "medium_history",
        {"history_rank": 0.50, "coverage_rank": 0.70, "liquidity_rank": 0.50},
    ),
    (
        "shorter_valid_history",
        {"history_rank": 0.05, "coverage_rank": 0.75, "liquidity_rank": 0.50},
    ),
    (
        "high_liquidity",
        {"history_rank": 0.70, "coverage_rank": 0.85, "liquidity_rank": 0.95},
    ),
    (
        "medium_liquidity",
        {"history_rank": 0.75, "coverage_rank": 0.90, "liquidity_rank": 0.50},
    ),
    (
        "lower_liquidity_sparse_valid",
        {"history_rank": 0.25, "coverage_rank": 0.40, "liquidity_rank": 0.05},
    ),
)


class RecurrentBudgetStudyError(RuntimeError):
    """Raised when study inputs/results violate the predeclared protocol."""


@dataclass(frozen=True)
class BudgetStudyManifest:
    study_version: str
    selection_version: str
    decision_version: str
    universe_version: str
    universe_hash: str
    source_inventory_hash: str
    symbols: tuple[str, ...]
    symbol_descriptors: tuple[Mapping[str, object], ...]
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    requested_device: str
    config_except_seed_budget: Mapping[str, object]
    run_count: int
    test_partition_loaded: bool = False

    def __post_init__(self) -> None:
        if self.study_version != BUDGET_STUDY_VERSION:
            raise ValueError("budget-study version is incompatible")
        if len(self.symbols) != STUDY_SYMBOL_COUNT or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("budget study requires exactly six unique symbols")
        if self.seeds != STUDY_SEEDS or self.budgets != STUDY_BUDGETS:
            raise ValueError("budget study seeds/budgets differ from protocol")
        if self.requested_device != "cpu":
            raise ValueError("7C.3f is CPU-only")
        if self.run_count != STUDY_RUN_COUNT:
            raise ValueError("budget study must contain exactly 54 runs")
        if self.test_partition_loaded:
            raise ValueError("TEST cannot enter budget-study metadata")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["symbol_descriptors"] = [dict(item) for item in self.symbol_descriptors]
        payload["seeds"] = list(self.seeds)
        payload["budgets"] = list(self.budgets)
        payload["config_except_seed_budget"] = dict(self.config_except_seed_budget)
        payload["selection_targets"] = [
            {"regime": regime, **dict(target)} for regime, target in ARCHETYPE_TARGETS
        ]
        payload["decision_thresholds"] = decision_thresholds()
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())


def decision_thresholds() -> dict[str, object]:
    return {
        "minimum_successful_runs_per_budget": MIN_SUCCESSFUL_RUNS_PER_BUDGET,
        "minimum_successful_seeds_per_symbol_budget": MIN_SUCCESSFUL_SEEDS_PER_SYMBOL_BUDGET,
        "mature_diagnostic_fraction": MATURE_DIAGNOSTIC_FRACTION,
        "maximum_mature_approximate_kl": MAX_MATURE_APPROXIMATE_KL,
        "maximum_mature_clip_fraction": MAX_MATURE_CLIP_FRACTION,
        "minimum_mature_median_explained_variance": MIN_MATURE_MEDIAN_EXPLAINED_VARIANCE,
        "material_explained_variance_delta": MATERIAL_EXPLAINED_VARIANCE_DELTA,
        "material_value_loss_reduction": MATERIAL_VALUE_LOSS_REDUCTION,
        "widespread_pair_count": WIDESPREAD_PAIR_COUNT,
        "validation_profit_used_for_decision": False,
    }


def _train_descriptor(
    symbol: str,
    *,
    splits_dir: Path,
    canonical_artifacts_dir: Path,
) -> tuple[dict[str, object], set[pd.Timestamp]]:
    loaded = load_training_recurrent_partition(
        symbol,
        "train",
        splits_dir=splits_dir,
        canonical_artifacts_dir=canonical_artifacts_dir,
    )
    if loaded.partition != "train":
        raise RecurrentBudgetStudyError("descriptor loader returned non-TRAIN data")
    data = loaded.data
    dates = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    volume = pd.to_numeric(data["volume"], errors="raise")
    ohl = data.loc[:, ["open", "high", "low"]].apply(pd.to_numeric, errors="raise")
    if dates.duplicated().any() or len(data) < 2:
        raise RecurrentBudgetStudyError(f"invalid TRAIN chronology for {symbol}")
    return (
        {
            "symbol": symbol,
            "train_rows": int(len(data)),
            "train_start": dates.min().date().isoformat(),
            "train_end": dates.max().date().isoformat(),
            "median_volume": float(volume.median()),
            "average_volume": float(volume.mean()),
            "zero_volume_ratio": float(volume.eq(0).mean()),
            "zero_ohl_ratio": float(ohl.eq(0).any(axis=1).mean()),
            "non_positive_close_rows": int(
                pd.to_numeric(data["close"], errors="raise").le(0).sum()
            ),
            "validation_available": bool(
                getattr(loaded.metadata, "validation_available", True)
            ),
            "recurrent_contract_version": loaded.metadata.recurrent_contract_version,
            "feature_version": loaded.metadata.feature_version,
            "environment_version": loaded.metadata.environment_version,
        },
        set(dates.tolist()),
    )


def build_train_descriptors(
    discovery: RecurrentUniverseDiscovery,
    *,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    canonical_artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
    descriptor_loader: Callable[..., tuple[dict[str, object], set[pd.Timestamp]]] = _train_descriptor,
) -> pd.DataFrame:
    """Build TRAIN-only descriptors for eligible symbols with VALIDATION."""

    eligible = discovery.records.loc[
        discovery.records["category"].eq(ELIGIBLE_TRAINABLE)
        & discovery.records["validation_available"].astype(bool),
        "symbol",
    ].astype(str)
    rows: list[dict[str, object]] = []
    date_sets: dict[str, set[pd.Timestamp]] = {}
    for symbol in sorted(eligible):
        descriptor, dates = descriptor_loader(
            symbol,
            splits_dir=Path(splits_dir),
            canonical_artifacts_dir=Path(canonical_artifacts_dir),
        )
        rows.append(descriptor)
        date_sets[symbol] = dates
    if len(rows) < STUDY_SYMBOL_COUNT:
        raise RecurrentBudgetStudyError("fewer than six trainable validation symbols")
    global_dates = sorted(set().union(*date_sets.values()))
    for row in rows:
        observed = date_sets[str(row["symbol"])]
        start = min(observed)
        end = max(observed)
        denominator = sum(start <= value <= end for value in global_dates)
        row["active_span_coverage"] = len(observed) / denominator
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(drop=True)


def select_representative_symbols(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Select six deterministic TRAIN-regime medoids without outcomes."""

    required = {
        "symbol",
        "train_rows",
        "active_span_coverage",
        "median_volume",
        "zero_volume_ratio",
        "zero_ohl_ratio",
        "non_positive_close_rows",
    }
    missing = sorted(required.difference(descriptors.columns))
    if missing:
        raise RecurrentBudgetStudyError("descriptor fields missing: " + ", ".join(missing))
    frame = descriptors.copy(deep=True).sort_values("symbol", kind="mergesort").reset_index(drop=True)
    if frame["symbol"].duplicated().any() or len(frame) < STUDY_SYMBOL_COUNT:
        raise RecurrentBudgetStudyError("descriptor symbols must be unique and sufficient")
    frame = frame.loc[frame["non_positive_close_rows"].eq(0)].copy()
    if len(frame) < STUDY_SYMBOL_COUNT:
        raise RecurrentBudgetStudyError("fewer than six symbols pass positive-close safety")
    frame["history_rank"] = frame["train_rows"].rank(method="average", pct=True)
    frame["coverage_rank"] = frame["active_span_coverage"].rank(method="average", pct=True)
    frame["liquidity_rank"] = np.log1p(frame["median_volume"].clip(lower=0)).rank(
        method="average", pct=True
    )
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for regime, targets in ARCHETYPE_TARGETS:
        candidates = frame.loc[~frame["symbol"].isin(used)].copy()
        distance = sum(
            (candidates[column] - target) ** 2 for column, target in targets.items()
        )
        # Quality terms prevent a regime label from selecting an avoidably bad row,
        # but remain lower weight than the declared history/liquidity targets.
        distance += 0.25 * candidates["zero_volume_ratio"] ** 2
        distance += 0.25 * candidates["zero_ohl_ratio"] ** 2
        candidates["selection_distance"] = distance
        chosen = candidates.sort_values(
            ["selection_distance", "symbol"], kind="mergesort"
        ).iloc[0]
        payload = chosen.to_dict()
        payload["selection_regime"] = regime
        selected.append(payload)
        used.add(str(chosen["symbol"]))
    result = pd.DataFrame(selected)
    if len(result) != STUDY_SYMBOL_COUNT or result["symbol"].duplicated().any():
        raise RecurrentBudgetStudyError("representative selection did not yield six symbols")
    return result.reset_index(drop=True)


def build_study_manifest(
    discovery: RecurrentUniverseDiscovery,
    selected: pd.DataFrame,
) -> BudgetStudyManifest:
    config = RecurrentPPOConfig(device="cpu")
    fixed = config.to_dict()
    fixed.pop("seed")
    fixed.pop("total_timesteps")
    descriptors = tuple(
        {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        }
        for row in selected.to_dict(orient="records")
    )
    return BudgetStudyManifest(
        study_version=BUDGET_STUDY_VERSION,
        selection_version=REPRESENTATIVE_SELECTION_VERSION,
        decision_version=BUDGET_DECISION_VERSION,
        universe_version=discovery.universe_version,
        universe_hash=discovery.universe_hash,
        source_inventory_hash=discovery.source_inventory_hash,
        symbols=tuple(selected["symbol"].astype(str)),
        symbol_descriptors=descriptors,
        seeds=STUDY_SEEDS,
        budgets=STUDY_BUDGETS,
        requested_device="cpu",
        config_except_seed_budget=fixed,
        run_count=STUDY_RUN_COUNT,
    )


def study_schedule(manifest: BudgetStudyManifest) -> pd.DataFrame:
    """Return deterministic symbol-major, seed-major, budget-major work."""

    rows = []
    for symbol in manifest.symbols:
        for seed in manifest.seeds:
            for budget in manifest.budgets:
                identity = {
                    "study_fingerprint": manifest.fingerprint,
                    "symbol": symbol,
                    "seed": seed,
                    "requested_timesteps": budget,
                }
                rows.append(
                    {
                        **identity,
                        "run_id": "budget-" + canonical_hash(identity)[:20],
                        "requested_device": "cpu",
                        "status": "PENDING",
                        "test_partition_loaded": False,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != STUDY_RUN_COUNT or result["run_id"].duplicated().any():
        raise RecurrentBudgetStudyError("study schedule does not contain 54 unique runs")
    return result


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (tuple, list)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return value


def _worker_payload(
    *,
    symbol: str,
    seed: int,
    budget: int,
    splits_dir: Path,
    canonical_artifacts_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    result = train_recurrent_single_symbol(
        symbol,
        config=RecurrentPPOConfig(seed=seed, total_timesteps=budget, device="cpu"),
        splits_dir=splits_dir,
        canonical_artifacts_dir=canonical_artifacts_dir,
        smoke_test=False,
    )
    payload: dict[str, object] = {
        "study_version": BUDGET_STUDY_VERSION,
        "symbol": symbol,
        "seed": seed,
        "requested_timesteps": budget,
        "actual_timesteps": result.actual_timesteps,
        "status": result.status,
        "error": result.error,
        "runtime_seconds": result.duration_seconds,
        "steps_per_second": (
            result.actual_timesteps / result.duration_seconds
            if result.duration_seconds > 0
            else None
        ),
        "requested_device": "cpu",
        "effective_device": result.device,
        "parameter_count": result.parameter_count,
        "train_rows": result.training_rows,
        "train_start": result.training_start,
        "train_end": result.training_end,
        "recurrent_contract_version": result.recurrent_contract_version,
        "feature_version": result.feature_version,
        "environment_version": result.environment_version,
        "rollout_boundaries_observed": result.rollout_boundaries_observed,
        "recurrent_continuity_checks": result.rollout_continuity_checks,
        "recurrent_continuity_verified": result.rollout_continuity_verified,
        "validation_status": "not_run",
        "validation_return": None,
        "validation_sharpe": None,
        "validation_max_drawdown": None,
        "validation_trade_count": None,
        "validation_exposure_percentage": None,
        "test_partition_loaded": False,
        "worker_wall_seconds": time.perf_counter() - started,
    }
    diagnostics = result.training_diagnostics.to_dict() if result.training_diagnostics else {}
    for key in (
        "updates",
        "approximate_kl",
        "clip_fraction",
        "entropy_loss",
        "explained_variance",
        "policy_gradient_loss",
        "value_loss",
        "learning_rate",
    ):
        payload[key] = diagnostics.get(key)
    if not result.succeeded:
        return _json_scalar(payload)  # type: ignore[return-value]
    try:
        validation = evaluate_recurrent_on_validation(
            result.model,
            symbol,
            trainer_result=result,
            seed=seed,
            splits_dir=splits_dir,
        )
        metrics = validation.strategy_result.metrics
        payload.update(
            validation_status="completed",
            validation_rows=validation.validation_rows,
            validation_start=validation.validation_start,
            validation_end=validation.validation_end,
            validation_return=metrics.get("total_return"),
            validation_sharpe=metrics.get("sharpe_ratio"),
            validation_max_drawdown=metrics.get("maximum_drawdown"),
            validation_trade_count=metrics.get("number_of_trades"),
            validation_exposure_percentage=metrics.get("exposure_percentage"),
            validation_parameters_unchanged=validation.model_parameters_unchanged,
            validation_first_episode_start=validation.first_episode_start,
        )
    except Exception as exc:
        payload["validation_status"] = "failed"
        payload["validation_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        payload["worker_wall_seconds"] = time.perf_counter() - started
    return _json_scalar(payload)  # type: ignore[return-value]


def _result_path(output_directory: Path, row: Mapping[str, object]) -> Path:
    return (
        output_directory
        / "runs"
        / safe_path_component(str(row["symbol"]))
        / f"seed_{int(row['seed'])}"
        / f"budget_{int(row['requested_timesteps'])}"
        / "result.json"
    )


def run_study(
    manifest: BudgetStudyManifest,
    *,
    output_directory: Path,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    canonical_artifacts_dir: Path = CANONICAL_RECURRENT_TRAIN_V2_DIR,
    timeout_seconds: int = 3_600,
    retry_infrastructure_failures: bool = False,
) -> pd.DataFrame:
    """Execute/resume 54 isolated CPU runs without persisting models."""

    output = Path(output_directory).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(manifest.to_dict()):
            raise RecurrentBudgetStudyError("existing study manifest is incompatible")
    else:
        atomic_write_json(manifest.to_dict(), manifest_path)
    schedule = study_schedule(manifest)
    collected: list[dict[str, object]] = []
    for number, row in enumerate(schedule.to_dict(orient="records"), start=1):
        path = _result_path(output, row)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            retryable = bool(
                retry_infrastructure_failures
                and payload.get("status") == FAILED
                and str(payload.get("error", "")).startswith(
                    "worker_exit_2:usage: recurrent_budget_study.py"
                )
                and int(payload.get("actual_timesteps", 0) or 0) == 0
            )
            if retryable:
                archive = path.parent / "attempts" / "infrastructure_attempt_000.json"
                if archive.exists() and json.loads(archive.read_text(encoding="utf-8")) != payload:
                    raise RecurrentBudgetStudyError(
                        f"infrastructure attempt archive differs: {archive}"
                    )
                if not archive.exists():
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(payload, archive)
                print(
                    f"[{number}/{STUDY_RUN_COUNT}] explicit retry after archived pre-training CLI failure {row['run_id']}",
                    flush=True,
                )
            else:
                collected.append(payload)
                print(f"[{number}/{STUDY_RUN_COUNT}] skip existing record {row['run_id']}", flush=True)
                continue
        command = [
            sys.executable,
            "-m",
            "reinforcement_learning.training.recurrent_budget_study",
            "--worker",
            "--symbol",
            str(row["symbol"]),
            "--seed",
            str(row["seed"]),
            "--budget",
            str(row["requested_timesteps"]),
            "--splits-dir",
            str(Path(splits_dir).resolve()),
            "--canonical-artifacts-dir",
            str(Path(canonical_artifacts_dir).resolve()),
        ]
        print(
            f"[{number}/{STUDY_RUN_COUNT}] train {row['symbol']} seed={row['seed']} budget={row['requested_timesteps']}",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
            markers = [
                line[len(WORKER_MARKER) :]
                for line in completed.stdout.splitlines()
                if line.startswith(WORKER_MARKER)
            ]
            if completed.returncode or len(markers) != 1:
                detail = completed.stderr.strip() or completed.stdout.strip()
                payload = {
                    **row,
                    "status": FAILED,
                    "error": f"worker_exit_{completed.returncode}:{detail[-2000:]}",
                    "validation_status": "not_run",
                    "test_partition_loaded": False,
                }
            else:
                payload = json.loads(markers[0])
        except subprocess.TimeoutExpired as exc:
            payload = {
                **row,
                "status": FAILED,
                "error": f"TimeoutExpired:{exc}",
                "validation_status": "not_run",
                "test_partition_loaded": False,
            }
        payload["run_id"] = row["run_id"]
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(payload, path)
        collected.append(payload)
        print(
            f"[{number}/{STUDY_RUN_COUNT}] {payload.get('status')} validation={payload.get('validation_status')}",
            flush=True,
        )
    results = pd.DataFrame(collected).sort_values(
        ["symbol", "seed", "requested_timesteps"], kind="mergesort"
    ).reset_index(drop=True)
    if len(results) != STUDY_RUN_COUNT or results["run_id"].duplicated().any():
        raise RecurrentBudgetStudyError("persisted study results do not reconcile")
    results.to_csv(output / "per_run_results.csv", index=False, lineterminator="\n")
    return results


RESULT_NUMERIC_COLUMNS = (
    "actual_timesteps",
    "runtime_seconds",
    "steps_per_second",
    "approximate_kl",
    "clip_fraction",
    "entropy_loss",
    "explained_variance",
    "policy_gradient_loss",
    "value_loss",
    "validation_return",
    "validation_sharpe",
    "validation_max_drawdown",
    "validation_trade_count",
    "validation_exposure_percentage",
)


def aggregate_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = results.copy(deep=True)
    for column in RESULT_NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    successful = frame.loc[frame["status"].eq(SUCCESS)]
    symbol_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for (symbol, budget), group in successful.groupby(
        ["symbol", "requested_timesteps"], sort=True
    ):
        payload: dict[str, object] = {
            "symbol": symbol,
            "requested_timesteps": int(budget),
            "successful_runs": len(group),
        }
        for column in RESULT_NUMERIC_COLUMNS:
            payload[f"{column}_mean"] = group[column].mean()
            payload[f"{column}_std"] = group[column].std(ddof=0)
        symbol_rows.append(payload)
    for budget in STUDY_BUDGETS:
        group = successful.loc[successful["requested_timesteps"].eq(budget)]
        payload = {
            "requested_timesteps": budget,
            "successful_runs": len(group),
            "failed_runs": int(
                len(frame.loc[frame["requested_timesteps"].eq(budget)]) - len(group)
            ),
        }
        for column in RESULT_NUMERIC_COLUMNS:
            payload[f"{column}_mean"] = group[column].mean()
            payload[f"{column}_std"] = group[column].std(ddof=0)
            payload[f"{column}_median"] = group[column].median()
        global_rows.append(payload)
    return pd.DataFrame(symbol_rows), pd.DataFrame(global_rows)


def _budget_maturity(frame: pd.DataFrame, budget: int) -> dict[str, object]:
    group = frame.loc[
        frame["requested_timesteps"].eq(budget) & frame["status"].eq(SUCCESS)
    ].copy()
    for column in ("approximate_kl", "clip_fraction", "explained_variance"):
        group[column] = pd.to_numeric(group[column], errors="coerce")
    per_symbol = group.groupby("symbol").size()
    validation_group = group.loc[group["validation_status"].eq("completed")]
    validation_per_symbol = validation_group.groupby("symbol").size()
    diagnostics_complete = group[
        ["approximate_kl", "clip_fraction", "explained_variance"]
    ].notna().all(axis=1)
    continuity = group["recurrent_continuity_verified"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    kl_ok = group["approximate_kl"].between(0, MAX_MATURE_APPROXIMATE_KL)
    clip_ok = group["clip_fraction"].between(0, MAX_MATURE_CLIP_FRACTION)
    fraction_ok = float((diagnostics_complete & kl_ok & clip_ok).mean()) if len(group) else 0.0
    median_ev = float(group["explained_variance"].median()) if len(group) else math.nan
    mature = bool(
        len(group) >= MIN_SUCCESSFUL_RUNS_PER_BUDGET
        and len(per_symbol) == STUDY_SYMBOL_COUNT
        and bool((per_symbol >= MIN_SUCCESSFUL_SEEDS_PER_SYMBOL_BUDGET).all())
        and len(validation_group) >= MIN_SUCCESSFUL_RUNS_PER_BUDGET
        and len(validation_per_symbol) == STUDY_SYMBOL_COUNT
        and bool(
            (
                validation_per_symbol
                >= MIN_SUCCESSFUL_SEEDS_PER_SYMBOL_BUDGET
            ).all()
        )
        and bool(continuity.all())
        and fraction_ok >= MATURE_DIAGNOSTIC_FRACTION
        and math.isfinite(median_ev)
        and median_ev >= MIN_MATURE_MEDIAN_EXPLAINED_VARIANCE
    )
    return {
        "budget": budget,
        "successful_runs": len(group),
        "successful_validation_runs": len(validation_group),
        "continuity_all": bool(continuity.all()) if len(group) else False,
        "diagnostic_fraction_ok": fraction_ok,
        "median_explained_variance": median_ev if math.isfinite(median_ev) else None,
        "technically_mature": mature,
    }


def _widespread_improvement(
    frame: pd.DataFrame, smaller: int, larger: int
) -> dict[str, object]:
    successful = frame.loc[frame["status"].eq(SUCCESS)].copy()
    fields = ["symbol", "seed", "requested_timesteps", "explained_variance", "value_loss"]
    pivot_ev = successful[fields].pivot_table(
        index=["symbol", "seed"], columns="requested_timesteps", values="explained_variance"
    )
    pivot_value = successful[fields].pivot_table(
        index=["symbol", "seed"], columns="requested_timesteps", values="value_loss"
    )
    if smaller not in pivot_ev or larger not in pivot_ev:
        return {"smaller": smaller, "larger": larger, "comparable_pairs": 0, "widespread": False}
    ev_delta = pivot_ev[larger] - pivot_ev[smaller]
    value_reduction = (pivot_value[smaller].abs() - pivot_value[larger].abs()) / pivot_value[
        smaller
    ].abs().replace(0, np.nan)
    comparable = ev_delta.dropna()
    ev_widespread = bool(
        len(comparable) >= MIN_SUCCESSFUL_RUNS_PER_BUDGET
        and float(comparable.median()) >= MATERIAL_EXPLAINED_VARIANCE_DELTA
        and int(comparable.gt(0).sum()) >= WIDESPREAD_PAIR_COUNT
    )
    value_comparable = value_reduction.dropna()
    value_widespread = bool(
        len(value_comparable) >= MIN_SUCCESSFUL_RUNS_PER_BUDGET
        and float(value_comparable.median()) >= MATERIAL_VALUE_LOSS_REDUCTION
        and int(value_comparable.gt(0).sum()) >= WIDESPREAD_PAIR_COUNT
    )
    return {
        "smaller": smaller,
        "larger": larger,
        "comparable_pairs": len(comparable),
        "median_explained_variance_delta": (
            float(comparable.median()) if len(comparable) else None
        ),
        "explained_variance_positive_pairs": int(comparable.gt(0).sum()),
        "median_relative_value_loss_reduction": (
            float(value_comparable.median()) if len(value_comparable) else None
        ),
        "value_loss_improved_pairs": int(value_comparable.gt(0).sum()),
        "widespread": ev_widespread or value_widespread,
    }


def select_budget(results: pd.DataFrame) -> dict[str, object]:
    """Apply the frozen technical rule; validation profit is never read."""

    required = {
        "symbol",
        "seed",
        "requested_timesteps",
        "status",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
        "value_loss",
        "recurrent_continuity_verified",
        "validation_status",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise RecurrentBudgetStudyError("result fields missing: " + ", ".join(missing))
    frame = results.copy(deep=True)
    for column in ("approximate_kl", "clip_fraction", "explained_variance", "value_loss"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    maturity = {budget: _budget_maturity(frame, budget) for budget in STUDY_BUDGETS}
    comparisons = {
        (50_000, 100_000): _widespread_improvement(frame, 50_000, 100_000),
        (100_000, 250_000): _widespread_improvement(frame, 100_000, 250_000),
    }
    decision = "BLOCKED_BUDGET_SELECTION"
    reason = "No budget satisfied the predeclared smallest-mature rule."
    if maturity[50_000]["technically_mature"]:
        if not comparisons[(50_000, 100_000)]["widespread"]:
            decision = "FREEZE_BUDGET_50000"
            reason = "50k was technically mature and 100k showed no widespread material technical improvement."
    elif maturity[100_000]["technically_mature"]:
        if not comparisons[(100_000, 250_000)]["widespread"]:
            decision = "FREEZE_BUDGET_100000"
            reason = "50k was not mature; 100k was mature and 250k showed no widespread material technical improvement."
    elif maturity[250_000]["technically_mature"]:
        decision = "FREEZE_BUDGET_250000"
        reason = "Only 250k satisfied the predeclared technical-maturity rule."
    return {
        "decision_version": BUDGET_DECISION_VERSION,
        "decision": decision,
        "reason": reason,
        "maturity": {str(key): value for key, value in maturity.items()},
        "comparisons": {
            f"{left}_to_{right}": value
            for (left, right), value in comparisons.items()
        },
        "thresholds": decision_thresholds(),
        "validation_profit_used": False,
        "test_partition_loaded": False,
    }


def write_aggregates(results: pd.DataFrame, output_directory: Path) -> dict[str, object]:
    output = Path(output_directory)
    per_symbol, global_summary = aggregate_results(results)
    per_symbol.to_csv(output / "per_symbol_aggregates.csv", index=False, lineterminator="\n")
    global_summary.to_csv(output / "global_aggregates.csv", index=False, lineterminator="\n")
    runtime = global_summary.loc[
        :, ["requested_timesteps", "runtime_seconds_mean", "runtime_seconds_std", "runtime_seconds_median"]
    ].copy()
    runtime["sequential_435_seconds_from_mean"] = runtime["runtime_seconds_mean"] * 435
    runtime["sequential_435_hours_from_mean"] = runtime["sequential_435_seconds_from_mean"] / 3600
    runtime.to_csv(output / "runtime_scaling.csv", index=False, lineterminator="\n")
    decision = select_budget(results)
    atomic_write_json(decision, output / "budget_decision.json")
    return decision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded recurrent budget study")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--symbol")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--splits-dir", type=Path, default=PROCESSED_SPLITS_DIR)
    parser.add_argument(
        "--canonical-artifacts-dir", type=Path, default=CANONICAL_RECURRENT_TRAIN_V2_DIR
    )
    parser.add_argument("--retry-infrastructure-failures", action="store_true")
    return parser


def _prepare(args: argparse.Namespace) -> BudgetStudyManifest:
    discovery = discover_recurrent_training_universe()
    descriptors = build_train_descriptors(
        discovery,
        splits_dir=args.splits_dir,
        canonical_artifacts_dir=args.canonical_artifacts_dir,
    )
    selected = select_representative_symbols(descriptors)
    manifest = build_study_manifest(discovery, selected)
    output = args.output_directory.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(manifest.to_dict()):
            raise RecurrentBudgetStudyError("prepared manifest differs from existing study")
    else:
        atomic_write_json(manifest.to_dict(), manifest_path)
    descriptors.to_csv(output / "train_descriptors.csv", index=False, lineterminator="\n")
    selected.to_csv(output / "selected_symbols.csv", index=False, lineterminator="\n")
    study_schedule(manifest).to_csv(output / "schedule.csv", index=False, lineterminator="\n")
    print(json.dumps({"fingerprint": manifest.fingerprint, **manifest.to_dict()}, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.worker:
            if not args.symbol or args.seed is None or args.budget is None:
                raise RecurrentBudgetStudyError("worker requires symbol, seed, and budget")
            if args.seed not in STUDY_SEEDS or args.budget not in STUDY_BUDGETS:
                raise RecurrentBudgetStudyError("worker seed/budget is outside protocol")
            payload = _worker_payload(
                symbol=args.symbol,
                seed=args.seed,
                budget=args.budget,
                splits_dir=args.splits_dir,
                canonical_artifacts_dir=args.canonical_artifacts_dir,
            )
            print(WORKER_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))
            return 0
        if args.output_directory is None:
            raise RecurrentBudgetStudyError(
                "prepare/run/aggregate requires --output-directory"
            )
        manifest = _prepare(args)
        if args.prepare:
            return 0
        if args.aggregate:
            results = pd.read_csv(args.output_directory / "per_run_results.csv")
            print(json.dumps(write_aggregates(results, args.output_directory), indent=2, sort_keys=True))
            return 0
        results = run_study(
            manifest,
            output_directory=args.output_directory,
            splits_dir=args.splits_dir,
            canonical_artifacts_dir=args.canonical_artifacts_dir,
            retry_infrastructure_failures=args.retry_infrastructure_failures,
        )
        print(json.dumps(write_aggregates(results, args.output_directory), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RecurrentBudgetStudyError) as exc:
        print(f"Budget study failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHETYPE_TARGETS",
    "BUDGET_DECISION_VERSION",
    "BUDGET_STUDY_VERSION",
    "BudgetStudyManifest",
    "RecurrentBudgetStudyError",
    "STUDY_BUDGETS",
    "STUDY_RUN_COUNT",
    "STUDY_SEEDS",
    "aggregate_results",
    "build_study_manifest",
    "build_train_descriptors",
    "decision_thresholds",
    "run_study",
    "select_budget",
    "select_representative_symbols",
    "study_schedule",
    "write_aggregates",
]
