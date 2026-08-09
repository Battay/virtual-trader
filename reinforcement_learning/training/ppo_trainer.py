"""Production single-symbol Stable-Baselines3 PPO trainer core."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from data_pipeline.src.config import (
    MODELS_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    SAVED_MODELS_DIR,
)
from reinforcement_learning.data_contract import (
    RL_CONTRACT_FILENAME,
    RL_OBSERVATION_SCALER_FILENAME,
    load_rl_partition,
)
from reinforcement_learning.environments import SingleSymbolTradingEnv
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.environments.validation import validate_environment
from reinforcement_learning.integrity import sha256_file

from .callbacks import PPOProgressCallback, ProgressHandler
from .config import PPOConfig
from .devices import (
    TorchDeviceResolution,
    resolve_torch_device,
    synchronize_torch_device,
    verify_sb3_model_device,
)
from .results import PPOTrainingDiagnostics, PPOTrainingResult


LOGGER = logging.getLogger(__name__)
ALGORITHM = "PPO"
MAX_SMOKE_TIMESTEPS = 1_024
EXPECTED_OBSERVATION_SHAPE = (
    len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
)


class PPOTrainerError(RuntimeError):
    """Raised internally when a requested training run is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_output_directory(output_dir: Path | None) -> Path | None:
    """Reject production roots; 5B-1 never writes to an accepted directory."""
    if output_dir is None:
        return None
    resolved = Path(output_dir).expanduser().resolve(strict=False)
    for protected in (Path(SAVED_MODELS_DIR), Path(MODELS_DATA_DIR)):
        protected_resolved = protected.resolve(strict=False)
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise PPOTrainerError(
                "5B-1 output_directory cannot be inside a production model directory"
            )
    if resolved.exists() and not resolved.is_dir():
        raise PPOTrainerError("output_directory must be a directory when it exists")
    return resolved


def _seed_everything(seed: int, *, resolved_device: str) -> None:
    """Seed Python, NumPy, SB3, CPU torch, and the selected MPS backend."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device == "mps":
        # Resolution is repeated deliberately so an explicit MPS request can
        # never become an implicit CPU run between configuration and seeding.
        resolve_torch_device("mps")
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if not callable(manual_seed):
            raise PPOTrainerError(
                "This PyTorch build lacks torch.mps.manual_seed(); MPS training aborted"
            )
        manual_seed(seed)
    set_random_seed(seed, using_cuda=False)


def create_training_vector_environment(
    training_data: pd.DataFrame,
    *,
    seed: int,
) -> DummyVecEnv:
    """Create one deterministically seeded SB3 environment from train data."""
    source = training_data.copy(deep=True)

    def make_environment() -> Monitor:
        environment = SingleSymbolTradingEnv(source)
        environment.action_space.seed(seed)
        return Monitor(environment)

    vector_environment = DummyVecEnv([make_environment])
    vector_environment.seed(seed)
    vector_environment.action_space.seed(seed)
    return vector_environment


def train_single_symbol(
    symbol: str,
    *,
    config: PPOConfig | None = None,
    seed: int | None = None,
    total_timesteps: int | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    progress_callback: ProgressHandler | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    smoke_test: bool = False,
) -> PPOTrainingResult:
    """Train one in-memory PPO model using only its canonical train partition."""
    effective_config = (config or PPOConfig()).with_runtime_overrides(
        seed=seed,
        total_timesteps=total_timesteps,
        device=device,
    )
    symbol_text = str(symbol).strip()
    started_at = _utc_now()
    started_clock = time.perf_counter()
    vector_environment: DummyVecEnv | None = None
    model: PPO | None = None
    actual_timesteps = 0
    training_start: str | None = None
    training_end: str | None = None
    training_rows = 0
    observation_shape: tuple[int, ...] | None = None
    feature_version = ""
    rl_contract_version = ""
    source_rl_contract_path: str | None = None
    source_rl_contract_sha256: str | None = None
    source_observation_scaler_path: str | None = None
    source_observation_scaler_sha256: str | None = None
    source_observation_scaler_metadata_path: str | None = None
    source_observation_scaler_metadata_sha256: str | None = None
    observation_features: tuple[str, ...] = ()
    resolved_output: Path | None = None
    device_resolution: TorchDeviceResolution | None = None
    actual_device: str | None = None
    training_diagnostics: PPOTrainingDiagnostics | None = None

    def finish(
        status: str,
        *,
        message: str,
        error: str | None = None,
        trained_model: PPO | None = None,
    ) -> PPOTrainingResult:
        return PPOTrainingResult(
            symbol=symbol_text,
            algorithm=ALGORITHM,
            ppo_config_version=effective_config.config_version,
            ppo_config=effective_config.to_dict(),
            environment_version=ENVIRONMENT_VERSION,
            rl_contract_version=rl_contract_version,
            feature_version=feature_version,
            source_rl_contract_path=source_rl_contract_path,
            source_rl_contract_sha256=source_rl_contract_sha256,
            source_observation_scaler_path=source_observation_scaler_path,
            source_observation_scaler_sha256=source_observation_scaler_sha256,
            source_observation_scaler_metadata_path=(
                source_observation_scaler_metadata_path
            ),
            source_observation_scaler_metadata_sha256=(
                source_observation_scaler_metadata_sha256
            ),
            observation_features=observation_features,
            seed=effective_config.seed,
            requested_timesteps=effective_config.total_timesteps,
            actual_timesteps=actual_timesteps,
            training_start=training_start,
            training_end=training_end,
            training_rows=training_rows,
            duration_seconds=max(0.0, time.perf_counter() - started_clock),
            requested_device=effective_config.device,
            resolved_device=(
                device_resolution.resolved_device if device_resolution else None
            ),
            device=(
                actual_device
                or (
                    device_resolution.resolved_device
                    if device_resolution
                    else effective_config.device
                )
            ),
            observation_shape=observation_shape,
            status=status,
            started_at=started_at,
            completed_at=_utc_now(),
            message=message,
            error=error,
            output_directory=str(resolved_output) if resolved_output else None,
            model=trained_model,
            training_diagnostics=(
                training_diagnostics if status == "completed" else None
            ),
        )

    try:
        if not symbol_text:
            raise PPOTrainerError("symbol is required for single-symbol PPO training")
        if smoke_test and effective_config.total_timesteps > MAX_SMOKE_TIMESTEPS:
            raise PPOTrainerError(
                f"smoke_test is capped at {MAX_SMOKE_TIMESTEPS} requested timesteps"
            )
        resolved_output = _validate_output_directory(output_dir)
        device_resolution = resolve_torch_device(effective_config.device)
        LOGGER.info(
            "ppo_device_resolution symbol=%s requested_device=%s "
            "resolved_device=%s mps_built=%s mps_available=%s",
            symbol_text,
            device_resolution.requested_device,
            device_resolution.resolved_device,
            device_resolution.mps_built,
            device_resolution.mps_available,
        )

        # This is the sole market-data load. Validation and test rows stay sealed.
        loaded = load_rl_partition(symbol_text, "train", splits_dir=Path(splits_dir))
        if loaded.partition != "train":
            raise PPOTrainerError(
                "Canonical loader returned a non-training partition; training aborted"
            )
        artifact_directory = loaded.artifact_path.parent
        contract_path = (artifact_directory / RL_CONTRACT_FILENAME).resolve()
        scaler_path = (
            artifact_directory / RL_OBSERVATION_SCALER_FILENAME
        ).resolve()
        scaler_metadata_path = scaler_path.with_suffix(".json")
        source_rl_contract_path = str(contract_path)
        source_rl_contract_sha256 = sha256_file(contract_path)
        source_observation_scaler_path = str(scaler_path)
        source_observation_scaler_sha256 = sha256_file(scaler_path)
        source_observation_scaler_metadata_path = str(scaler_metadata_path)
        source_observation_scaler_metadata_sha256 = sha256_file(
            scaler_metadata_path
        )
        observation_features = tuple(
            str(feature) for feature in loaded.contract.get("observation_features", ())
        )
        training_data = loaded.data
        training_rows = len(training_data)
        training_start = training_data["date"].min().date().isoformat()
        training_end = training_data["date"].max().date().isoformat()
        feature_version = str(loaded.contract.get("feature_version", ""))
        rl_contract_version = str(
            loaded.contract.get("artifact_schema_version", "")
        )

        validation_environment = SingleSymbolTradingEnv(training_data)
        try:
            validation = validate_environment(validation_environment)
        finally:
            validation_environment.close()
        observation_shape = validation.observation_shape
        if not validation.valid:
            raise PPOTrainerError(
                "Training environment validation failed: "
                + "; ".join(validation.errors)
            )
        if observation_shape != EXPECTED_OBSERVATION_SHAPE:
            raise PPOTrainerError(
                "Training environment observation shape differs from "
                f"{EXPECTED_OBSERVATION_SHAPE}: {observation_shape}"
            )
        LOGGER.info(
            "ppo_environment_validation symbol=%s valid=%s observation_shape=%s "
            "errors=%s",
            symbol_text,
            validation.valid,
            observation_shape,
            validation.errors,
        )

        _seed_everything(
            effective_config.seed,
            resolved_device=device_resolution.resolved_device,
        )
        vector_environment = create_training_vector_environment(
            training_data,
            seed=effective_config.seed,
        )
        LOGGER.info(
            "ppo_training_data symbol=%s partition=train rows=%s start=%s end=%s "
            "contract=%s environment=%s observation_shape=%s",
            symbol_text,
            training_rows,
            training_start,
            training_end,
            rl_contract_version,
            ENVIRONMENT_VERSION,
            observation_shape,
        )
        LOGGER.info(
            "ppo_training_start symbol=%s started_at=%s config=%s smoke_test=%s",
            symbol_text,
            started_at,
            json.dumps(effective_config.to_dict(), sort_keys=True),
            smoke_test,
        )

        model = PPO(
            effective_config.policy,
            vector_environment,
            verbose=0,
            **effective_config.model_kwargs(
                resolved_device=device_resolution.resolved_device
            ),
        )
        actual_device = verify_sb3_model_device(model, device_resolution)
        LOGGER.info(
            "ppo_device_verified symbol=%s requested_device=%s "
            "resolved_device=%s actual_device=%s",
            symbol_text,
            device_resolution.requested_device,
            device_resolution.resolved_device,
            actual_device,
        )
        interval_steps = max(
            effective_config.n_steps,
            effective_config.total_timesteps // 10,
        )
        callback = PPOProgressCallback(
            symbol=symbol_text,
            requested_timesteps=effective_config.total_timesteps,
            interval_steps=interval_steps,
            handler=progress_callback,
        )
        model.learn(
            total_timesteps=effective_config.total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        # MPS kernels are asynchronous. Synchronizing here surfaces unsupported
        # operations before a run can be reported as completed.
        synchronize_torch_device(device_resolution.resolved_device)
        actual_device = verify_sb3_model_device(model, device_resolution)
        actual_timesteps = int(model.num_timesteps)
        if callback.cancel_requested:
            LOGGER.warning(
                "ppo_training_interrupted symbol=%s timesteps=%s reason=callback",
                symbol_text,
                actual_timesteps,
            )
            return finish(
                "interrupted",
                message="Training was cancelled by the progress callback; no model was saved.",
            )

        training_diagnostics = callback.training_diagnostics
        if training_diagnostics is not None:
            LOGGER.info(
                "ppo_training_diagnostics symbol=%s diagnostics=%s",
                symbol_text,
                json.dumps(training_diagnostics.to_dict(), sort_keys=True),
            )

        LOGGER.info(
            "ppo_training_completed symbol=%s timesteps=%s duration_seconds=%.3f",
            symbol_text,
            actual_timesteps,
            time.perf_counter() - started_clock,
        )
        label = "Developer smoke training" if smoke_test else "Training"
        return finish(
            "completed",
            message=(
                f"{label} completed in memory; no model or registry artifact was written."
            ),
            trained_model=model,
        )
    except KeyboardInterrupt:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        LOGGER.warning(
            "ppo_training_interrupted symbol=%s timesteps=%s reason=keyboard_interrupt",
            symbol_text,
            actual_timesteps,
        )
        return finish(
            "interrupted",
            message="Training was interrupted; no model was saved or registered.",
            error="KeyboardInterrupt",
        )
    except Exception as exc:
        actual_timesteps = int(model.num_timesteps) if model is not None else 0
        LOGGER.error(
            "ppo_training_failed symbol=%s error=%s: %s",
            symbol_text,
            type(exc).__name__,
            exc,
        )
        return finish(
            "failed",
            message="Training failed safely; no model was saved or registered.",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if vector_environment is not None:
            vector_environment.close()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the intentionally single-symbol-only 5B-1 command line."""
    parser = argparse.ArgumentParser(
        description="Run one in-memory PSX PPO training job (no model promotion)."
    )
    parser.add_argument("--symbol", required=True, help="One RL-ready PSX symbol")
    parser.add_argument("--timesteps", type=_positive_integer)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="Explicit torch device request; mps never silently falls back to CPU",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Label and cap a developer smoke run at {MAX_SMOKE_TIMESTEPS} steps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Reserved non-production directory; 5B-1 writes no model artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one safe in-memory job and print its structured result."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.timesteps is None and not args.smoke_test:
        parser.error("--timesteps is required unless --smoke-test is supplied")
    timesteps = args.timesteps if args.timesteps is not None else 512
    if args.smoke_test and timesteps > MAX_SMOKE_TIMESTEPS:
        parser.error(
            f"--smoke-test permits at most {MAX_SMOKE_TIMESTEPS} timesteps"
        )
    if args.seed < 0:
        parser.error("--seed cannot be negative")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.smoke_test:
        LOGGER.warning(
            "Developer smoke test only: no profitability inference or model promotion."
        )
    result = train_single_symbol(
        args.symbol,
        seed=args.seed,
        total_timesteps=timesteps,
        device=args.device,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
