"""Dry-run contract and safety gates for full-universe recurrent training.

This module is intentionally incapable of calling ``model.learn``.  It turns
the canonical recurrent discovery into a deterministic 508-identity plan,
estimates storage, and reports the gates that must pass before a separately
authorized training run may be created.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
from typing import Mapping

import pandas as pd

from data_pipeline.src.config import TRAINING_RUNS_DIR
from feature_engineering.storage import atomic_write_json
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)

from .job_state import INELIGIBLE, QUEUED, canonical_hash
from .recurrent_config import RECURRENT_PPO_CONFIG_VERSION, RecurrentPPOConfig
from .recurrent_orchestrator import (
    RESUME_CAPABILITY,
    RecurrentUniverseDiscovery,
    build_training_run,
    discover_recurrent_training_universe,
)
from .recurrent_trainer import RECURRENT_TRAINER_VERSION


FULL_RUN_SPEC_VERSION = "recurrent_full_universe_run_spec_v1"
FULL_RUN_PLAN_VERSION = "recurrent_full_universe_dry_run_plan_v1"
STORAGE_ESTIMATE_VERSION = "recurrent_full_universe_storage_estimate_v1"
CANDIDATE_TIMESTEP_BUDGET = 100_000
EXPECTED_IDENTITY_COUNT = 508
EXPECTED_TRAINABLE_COUNT = 435
EXPECTED_INELIGIBLE_COUNT = 73
GIB = 1024**3


class FullUniverseRunError(RuntimeError):
    """Raised when a full-run plan cannot be represented safely."""


@dataclass(frozen=True)
class FullUniverseTrainingSpec:
    """Versioned candidate configuration; it does not authorize training."""

    spec_version: str = FULL_RUN_SPEC_VERSION
    requested_timesteps: int = CANDIDATE_TIMESTEP_BUDGET
    budget_status: str = "candidate_cost_budget_research_choice_not_frozen"
    seed_policy: str = "fixed_per_symbol_seed_42_v1"
    seed: int = 42
    requested_device: str = "cuda"
    worker_count: int = 1
    validation_policy: str = "validation_only_after_training_test_sealed_v1"
    model_policy: str = "isolated_atomic_final_model_no_overwrite_v1"
    checkpoint_policy: str = RESUME_CAPABILITY
    artifact_contract_policy: str = "compatible_symbol_recurrent_contract_v1_or_v2"
    test_partition_loaded: bool = False

    def __post_init__(self) -> None:
        if self.spec_version != FULL_RUN_SPEC_VERSION:
            raise ValueError("full-run specification version is incompatible")
        if self.requested_timesteps < 1 or self.seed < 0:
            raise ValueError("timesteps must be positive and seed cannot be negative")
        if self.requested_device not in {"cpu", "cuda"}:
            raise ValueError("full-run candidate device must be explicit CPU or CUDA")
        if self.worker_count != 1:
            raise ValueError(
                "the current production orchestrator supports exactly one worker; "
                "2/4 remain CUDA benchmark candidates only"
            )
        if self.test_partition_loaded:
            raise ValueError("TEST cannot enter a full-run specification")

    @property
    def config(self) -> RecurrentPPOConfig:
        return RecurrentPPOConfig(
            seed=self.seed,
            total_timesteps=self.requested_timesteps,
            device=self.requested_device,
        )

    def to_dict(self) -> dict[str, object]:
        config = self.config
        return {
            **asdict(self),
            "trainer_version": RECURRENT_TRAINER_VERSION,
            "recurrent_ppo_config_version": RECURRENT_PPO_CONFIG_VERSION,
            "algorithm": config.algorithm,
            "policy": config.policy,
            "environment_version": ENVIRONMENT_VERSION,
            "observation_shape": [
                len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES)
            ],
            "observation_features": [
                *DEFAULT_OBSERVATION_FEATURES,
                *DYNAMIC_PORTFOLIO_FEATURES,
            ],
            "dependency_versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "stable_baselines3": version("stable-baselines3"),
                "sb3_contrib": version("sb3-contrib"),
                "torch": version("torch"),
                "gymnasium": version("gymnasium"),
            },
            "ppo_lstm_config": config.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class StorageEstimate:
    estimate_version: str
    trainable_jobs: int
    identity_jobs: int
    sampled_model_bytes: int
    final_models_bytes: int
    validation_bytes: int
    job_state_bytes: int
    logs_bytes: int
    optional_checkpoints_bytes: int
    conservative_total_bytes: int
    available_bytes: int
    required_safety_bytes: int
    safety_margin_bytes: int
    safe: bool
    checkpoint_copies_per_job: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_full_run_storage(
    *,
    trainable_jobs: int,
    identity_jobs: int,
    storage_root: Path = TRAINING_RUNS_DIR,
    sampled_model_bytes: int = 1 * 1024**2,
    validation_bytes_per_job: int = 256 * 1024,
    job_state_bytes_per_identity: int = 64 * 1024,
    logs_bytes_per_job: int = 5 * 1024**2,
    checkpoint_copies_per_job: int = 0,
    available_bytes: int | None = None,
) -> StorageEstimate:
    """Return a conservative estimate without creating any run directory."""

    values = (
        trainable_jobs,
        identity_jobs,
        sampled_model_bytes,
        validation_bytes_per_job,
        job_state_bytes_per_identity,
        logs_bytes_per_job,
        checkpoint_copies_per_job,
    )
    if any(isinstance(value, bool) or int(value) < 0 for value in values):
        raise ValueError("storage estimate inputs cannot be negative")
    root = Path(storage_root).expanduser().resolve(strict=False)
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free if available_bytes is None else available_bytes)
    final_models = trainable_jobs * sampled_model_bytes
    validation = trainable_jobs * validation_bytes_per_job
    job_state = identity_jobs * job_state_bytes_per_identity
    logs = trainable_jobs * logs_bytes_per_job
    checkpoints = trainable_jobs * sampled_model_bytes * checkpoint_copies_per_job
    total = final_models + validation + job_state + logs + checkpoints
    required = max(5 * GIB, 2 * total)
    return StorageEstimate(
        estimate_version=STORAGE_ESTIMATE_VERSION,
        trainable_jobs=trainable_jobs,
        identity_jobs=identity_jobs,
        sampled_model_bytes=sampled_model_bytes,
        final_models_bytes=final_models,
        validation_bytes=validation,
        job_state_bytes=job_state,
        logs_bytes=logs,
        optional_checkpoints_bytes=checkpoints,
        conservative_total_bytes=total,
        available_bytes=free,
        required_safety_bytes=required,
        safety_margin_bytes=free - required,
        safe=free >= required,
        checkpoint_copies_per_job=checkpoint_copies_per_job,
    )


def build_full_universe_plan(
    discovery: RecurrentUniverseDiscovery,
    *,
    spec: FullUniverseTrainingSpec,
    created_at: str = "1970-01-01T00:00:00+00:00",
    expected_identity_count: int = EXPECTED_IDENTITY_COUNT,
    expected_trainable_count: int = EXPECTED_TRAINABLE_COUNT,
) -> tuple[Mapping[str, object], pd.DataFrame]:
    """Materialize an in-memory plan; no run store or trainer is invoked."""

    if discovery.identity_count != expected_identity_count:
        raise FullUniverseRunError(
            f"identity accounting changed: {discovery.identity_count}!={expected_identity_count}"
        )
    if discovery.eligible_count != expected_trainable_count:
        raise FullUniverseRunError(
            f"trainable accounting changed: {discovery.eligible_count}!={expected_trainable_count}"
        )
    manifest, jobs = build_training_run(
        discovery,
        config=spec.config,
        created_at=created_at,
        validation_enabled=True,
    )
    records = discovery.records.set_index("symbol", drop=False)
    rows: list[dict[str, object]] = []
    for job in jobs:
        source = records.loc[job.symbol]
        attempt = f"attempt_{job.retry_count:03d}"
        rows.append(
            {
                "symbol": job.symbol,
                "company_name": str(source["company_name"]),
                "trainability": job.trainability,
                "trainability_category": str(source["category"]),
                "trainability_reason": job.trainability_reason,
                "artifact_contract_version": job.data_contract_version,
                "feature_version": job.feature_version,
                "environment_version": job.environment_version,
                "source_data_hash": job.source_data_hash,
                "requested_timesteps": job.requested_timesteps,
                "seed": job.seed,
                "requested_device": job.requested_device,
                "status": job.status,
                "model_path": f"models/{job.symbol}/{attempt}/model.zip",
                "checkpoint_path": job.checkpoint_path,
                "validation_path": f"validation/{job.symbol}/{attempt}.json",
                "test_partition_loaded": False,
            }
        )
    plan = pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(drop=True)
    if len(plan) != expected_identity_count or plan["symbol"].duplicated().any():
        raise FullUniverseRunError("dry-run plan lost or duplicated an identity")
    status_counts = plan["status"].value_counts().to_dict()
    if status_counts != {QUEUED: expected_trainable_count, INELIGIBLE: expected_identity_count - expected_trainable_count}:
        raise FullUniverseRunError(f"dry-run job statuses do not reconcile: {status_counts}")
    identity = {
        "plan_version": FULL_RUN_PLAN_VERSION,
        "spec_fingerprint": spec.fingerprint,
        "run_id": manifest.run_id,
        "run_fingerprint": manifest.run_fingerprint,
        "universe_version": discovery.universe_version,
        "universe_hash": discovery.universe_hash,
        "source_inventory_hash": discovery.source_inventory_hash,
        "identity_count": len(plan),
        "trainable_count": expected_trainable_count,
        "ineligible_count": len(plan) - expected_trainable_count,
        "queued_count": int(status_counts[QUEUED]),
        "test_partition_loaded": False,
        "plan_content_hash": canonical_hash(
            {"rows": plan.fillna("").to_dict(orient="records")}
        ),
        "execution_authorized": False,
    }
    return identity, plan


def dry_run_progress_summary(plan: pd.DataFrame) -> dict[str, object]:
    """Summarize a dry plan without inventing an ETA at zero progress."""

    counts = plan["status"].value_counts().to_dict()
    return {
        "total_identities": int(len(plan)),
        "trainable": int(plan["trainability"].eq("eligible").sum()),
        "ineligible": int(plan["trainability"].eq("ineligible").sum()),
        "queued": int(counts.get(QUEUED, 0)),
        "training": 0,
        "validating": 0,
        "completed": 0,
        "failed": 0,
        "interrupted": 0,
        "overall_progress_percent": 0.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "effective_device": None,
    }


def evaluate_full_run_gates(
    *,
    identity: Mapping[str, object],
    plan: pd.DataFrame,
    storage: StorageEstimate,
    cuda_benchmark_completed: bool,
    benchmarked_worker_counts: tuple[int, ...] = (),
    requested_worker_count: int = 1,
    training_budget_frozen: bool = False,
    validation_policy_frozen: bool = True,
    incompatible_stale_run: bool = False,
) -> dict[str, object]:
    """Return explicit gates; this function never starts training."""

    gates = {
        "all_identities_accounted": len(plan) == EXPECTED_IDENTITY_COUNT,
        "trainable_jobs_materialized": int(plan["trainability"].eq("eligible").sum()) == EXPECTED_TRAINABLE_COUNT,
        "explicit_ineligible_jobs": int(plan["trainability"].eq("ineligible").sum()) == EXPECTED_INELIGIBLE_COUNT,
        "source_and_universe_hashes_present": all(identity.get(key) for key in ("universe_hash", "source_inventory_hash")),
        "test_sealed": not bool(plan["test_partition_loaded"].any()) and not bool(identity.get("test_partition_loaded")),
        "storage_safe": storage.safe,
        "cuda_benchmarked_for_cuda_run": cuda_benchmark_completed,
        "worker_count_benchmarked": requested_worker_count == 1 or requested_worker_count in benchmarked_worker_counts,
        "model_paths_isolated_unique": plan["model_path"].is_unique and plan["model_path"].str.startswith("models/").all(),
        "no_incompatible_stale_run": not incompatible_stale_run,
        "training_budget_frozen": training_budget_frozen,
        "validation_policy_frozen": validation_policy_frozen,
    }
    preparation_keys = tuple(
        key for key in gates if key not in {"cuda_benchmarked_for_cuda_run", "training_budget_frozen"}
    )
    return {
        "gates": gates,
        "ready_to_benchmark_cuda": all(gates[key] for key in preparation_keys),
        "full_run_authorized": all(gates.values()),
    }


def write_full_run_plan(
    *,
    identity: Mapping[str, object],
    plan: pd.DataFrame,
    spec: FullUniverseTrainingSpec,
    storage: StorageEstimate,
    output_directory: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Persist deterministic planning artifacts, never a runnable job store."""

    output = Path(output_directory)
    csv_path = output / "full_universe_run_plan.csv"
    json_path = output / "full_universe_run_spec.json"
    storage_path = output / "full_universe_storage_estimate.json"
    existing = [path for path in (csv_path, json_path, storage_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("planning artifact exists; use overwrite explicitly")
    output.mkdir(parents=True, exist_ok=True)
    plan.to_csv(csv_path, index=False, lineterminator="\n")
    atomic_write_json({"identity": dict(identity), "spec": spec.to_dict()}, json_path)
    atomic_write_json(storage.to_dict(), storage_path)
    return csv_path, json_path, storage_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run full recurrent universe")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timesteps", type=int, default=CANDIDATE_TIMESTEP_BUDGET)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = FullUniverseTrainingSpec(
        requested_timesteps=args.timesteps,
        requested_device=args.device,
        worker_count=args.workers,
    )
    discovery = discover_recurrent_training_universe()
    identity, plan = build_full_universe_plan(discovery, spec=spec)
    storage = estimate_full_run_storage(
        trainable_jobs=discovery.eligible_count,
        identity_jobs=discovery.identity_count,
    )
    print(json.dumps({**identity, "progress": dry_run_progress_summary(plan), "storage": storage.to_dict()}, indent=2, sort_keys=True))
    if args.output_directory:
        for path in write_full_run_plan(
            identity=identity,
            plan=plan,
            spec=spec,
            storage=storage,
            output_directory=args.output_directory,
            overwrite=args.overwrite,
        ):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_TIMESTEP_BUDGET",
    "EXPECTED_IDENTITY_COUNT",
    "EXPECTED_INELIGIBLE_COUNT",
    "EXPECTED_TRAINABLE_COUNT",
    "FULL_RUN_PLAN_VERSION",
    "FULL_RUN_SPEC_VERSION",
    "FullUniverseRunError",
    "FullUniverseTrainingSpec",
    "StorageEstimate",
    "build_full_universe_plan",
    "dry_run_progress_summary",
    "estimate_full_run_storage",
    "evaluate_full_run_gates",
    "write_full_run_plan",
]
