"""TRAIN-only RecurrentPPO trainer for the repaired balanced-window method.

This is deliberately separate from ``recurrent_ppo_sector_v1``.  The original
6E engine remains reproducible; this module implements the predeclared 6E.2
methodology and refuses the full three-seed experiment unless a future caller
provides an exact frozen specification and explicit authorization.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data_pipeline.src.config import PROCESSED_SPLITS_DIR, PROJECT_ROOT
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    action_validity_metadata,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
)
from reinforcement_learning.recurrent_data_contract import (
    LoadedRecurrentPartition,
    load_recurrent_partition,
)

from .callbacks import ProgressHandler
from .devices import (
    resolve_torch_device,
    synchronize_torch_device,
    verify_sb3_model_device,
)
from .ppo_trainer import EXPECTED_OBSERVATION_SHAPE, _seed_everything
from .recurrent_trainer import count_model_parameters
from .sector_balanced_callbacks import BalancedSectorProgressCallback
from .sector_balanced_config import (
    METHODOLOGY_SMOKE_PURPOSE,
    PREDECLARED_RESEARCH_PURPOSE,
    BalancedSectorRecurrentPPOConfig,
)
from .sector_balanced_results import BalancedSectorTrainingResult
from .sector_balanced_windows import (
    BalancedWindowSchedule,
    BalancedWindowTrainingEnv,
    build_balanced_window_schedule,
    canonical_payload_hash,
)
from .sector_recurrent_trainer import (
    COMMERCIAL_BANKS_MANIFEST_PATH,
    LoadedSectorTrainingUniverse,
    _assert_finite_training_state,
    _dependency_versions,
    _git_commit,
    load_sector_training_universe,
)


MAX_METHODOLOGY_SMOKE_SYMBOLS = 3
MAX_METHODOLOGY_SMOKE_ROUNDS = 2
MAX_METHODOLOGY_SMOKE_TRANSITIONS = 3_072


class BalancedSectorTrainerError(RuntimeError):
    """Raised when the 6E.2 methodology cannot be satisfied exactly."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataframe_digest(frame: pd.DataFrame) -> str:
    """Hash values, order, columns, and dtypes without serializing artifacts."""

    value_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy(
        dtype=np.uint64
    )
    identity = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(value_hashes.tobytes())
    return digest.hexdigest()


def _selected_train_data(
    universe: LoadedSectorTrainingUniverse,
    config: BalancedSectorRecurrentPPOConfig,
) -> dict[str, pd.DataFrame]:
    requested = tuple(config.constituent_symbols)
    unknown = sorted(set(requested).difference(universe.symbols))
    if unknown:
        raise BalancedSectorTrainerError(
            "configuration contains non-manifest constituents: "
            + ", ".join(unknown)
        )
    if config.target_symbol is not None and config.target_symbol in requested:
        raise BalancedSectorTrainerError(
            "leave-one-out target appears in its own pretraining universe"
        )
    # Only selected peers are copied.  An excluded target is not passed to the
    # scheduler and cannot contribute observations or normalization metadata.
    return {
        symbol: universe.train_data[symbol].copy(deep=True)
        for symbol in requested
    }


def _build_schedule(
    universe: LoadedSectorTrainingUniverse,
    config: BalancedSectorRecurrentPPOConfig,
) -> tuple[BalancedWindowSchedule, dict[str, pd.DataFrame]]:
    train_data = _selected_train_data(universe, config)
    schedule = build_balanced_window_schedule(
        train_data,
        universe_hash=universe.universe_hash,
        rounds=config.balanced_rounds,
        window_transition_count=config.window_transition_count,
        data_schedule_seed=config.data_schedule_seed,
        target_symbol=config.target_symbol,
    )
    if schedule.symbols != config.constituent_symbols:
        raise BalancedSectorTrainerError(
            "scheduler constituent identity differs from configuration"
        )
    if schedule.expected_total_scheduled_transitions != config.expected_total_transitions:
        raise BalancedSectorTrainerError("scheduled transition total differs")
    if schedule.expected_transitions_per_symbol != config.expected_transitions_per_symbol:
        raise BalancedSectorTrainerError("scheduled per-symbol exposure differs")
    return schedule, train_data


def _load_and_validate_experiment_spec(
    path: Path,
    *,
    expected_hash: str,
    schedule: BalancedWindowSchedule,
    config: BalancedSectorRecurrentPPOConfig,
) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BalancedSectorTrainerError(
            f"could not read frozen experiment specification: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BalancedSectorTrainerError("experiment specification must be an object")
    stored_hash = str(payload.get("experiment_spec_hash", ""))
    if stored_hash != expected_hash:
        raise BalancedSectorTrainerError("experiment specification hash differs")
    identity = dict(payload)
    identity.pop("experiment_spec_hash", None)
    identity.pop("reproducibility_fingerprint", None)
    if canonical_payload_hash(identity) != stored_hash:
        raise BalancedSectorTrainerError("experiment specification is stale")
    checks = {
        "sector_universe_hash": schedule.universe_hash,
        "schedule_digest": schedule.schedule_digest,
        "trainer_version": config.trainer_version,
        "sampling_version": config.sampling_version,
        "reward_version": config.reward.reward_version,
        "action_validity_version": config.action_validity_version,
        "expected_total_scheduled_transitions": (
            schedule.expected_total_scheduled_transitions
        ),
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise BalancedSectorTrainerError(
                f"experiment specification {field} differs"
            )
    if payload.get("reward_configuration") != config.reward.to_metadata():
        raise BalancedSectorTrainerError(
            "experiment specification reward semantics differ"
        )
    if payload.get("action_validity_configuration") != action_validity_metadata(
        config.invalid_action_mode
    ):
        raise BalancedSectorTrainerError(
            "experiment specification action-validity semantics differ"
        )
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or tuple(
        seed_policy.get("experiment_model_seeds", ())
    ) != config.experiment_seed_set:
        raise BalancedSectorTrainerError("experiment seed protocol differs")
    return payload


def _enforce_execution_scope(
    config: BalancedSectorRecurrentPPOConfig,
    schedule: BalancedWindowSchedule,
    *,
    methodology_smoke: bool,
    allow_predeclared_research_run: bool,
    experiment_spec_path: Path | None,
    expected_experiment_spec_hash: str | None,
) -> Mapping[str, object] | None:
    if config.execution_purpose == METHODOLOGY_SMOKE_PURPOSE:
        if not methodology_smoke:
            raise BalancedSectorTrainerError(
                "methodology smoke must be explicitly requested"
            )
        if (
            len(schedule.symbols) > MAX_METHODOLOGY_SMOKE_SYMBOLS
            or schedule.rounds > MAX_METHODOLOGY_SMOKE_ROUNDS
            or schedule.expected_total_scheduled_transitions
            > MAX_METHODOLOGY_SMOKE_TRANSITIONS
        ):
            raise BalancedSectorTrainerError(
                "6E.2 methodology smoke is capped at 3 symbols x 2 x 512"
            )
        if allow_predeclared_research_run:
            raise BalancedSectorTrainerError(
                "smoke mode cannot carry full-research authorization"
            )
        return None
    if config.execution_purpose != PREDECLARED_RESEARCH_PURPOSE:
        raise BalancedSectorTrainerError("unknown execution purpose")
    schedule.assert_full_research_contract()
    if methodology_smoke:
        raise BalancedSectorTrainerError(
            "the frozen 194,560-transition run is not a smoke test"
        )
    if not allow_predeclared_research_run:
        raise BalancedSectorTrainerError(
            "full sector experiment is blocked pending explicit future authorization"
        )
    if experiment_spec_path is None or expected_experiment_spec_hash is None:
        raise BalancedSectorTrainerError(
            "full run requires an exact frozen specification path and hash"
        )
    return _load_and_validate_experiment_spec(
        experiment_spec_path,
        expected_hash=expected_experiment_spec_hash,
        schedule=schedule,
        config=config,
    )


def _empty_methodology_diagnostics() -> dict[str, object]:
    return {
        "reward_and_action": {},
        "selected_action_counts": {"hold": 0, "buy": 0, "sell": 0},
        "per_symbol_exposure_percentages": {},
        "per_symbol_trade_counts": {},
        "per_symbol_action_pattern_digests": {},
        "collapse_diagnostics": {"warnings": []},
    }


def train_balanced_sector_recurrent_ppo(
    config: BalancedSectorRecurrentPPOConfig,
    *,
    manifest_path: Path = COMMERCIAL_BANKS_MANIFEST_PATH,
    progress_callback: ProgressHandler | None = None,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    partition_loader: Callable[..., LoadedRecurrentPartition] = load_recurrent_partition,
    methodology_smoke: bool = False,
    allow_predeclared_research_run: bool = False,
    experiment_spec_path: Path | None = None,
    expected_experiment_spec_hash: str | None = None,
    universe_loader: Callable[..., LoadedSectorTrainingUniverse] = (
        load_sector_training_universe
    ),
) -> BalancedSectorTrainingResult:
    """Train one exact balanced schedule from canonical TRAIN partitions only."""

    if not isinstance(config, BalancedSectorRecurrentPPOConfig):
        raise TypeError("config must be BalancedSectorRecurrentPPOConfig")
    universe = universe_loader(
        Path(manifest_path),
        splits_dir=Path(splits_dir),
        project_root=Path(project_root),
        partition_loader=partition_loader,
    )
    if config.sector_universe_hash != universe.universe_hash:
        raise BalancedSectorTrainerError("configuration universe hash differs")
    if config.taxonomy_version != str(universe.manifest.get("taxonomy_version")):
        raise BalancedSectorTrainerError("configuration taxonomy version differs")
    schedule, train_data = _build_schedule(universe, config)
    authorized_spec = _enforce_execution_scope(
        config,
        schedule,
        methodology_smoke=methodology_smoke,
        allow_predeclared_research_run=allow_predeclared_research_run,
        experiment_spec_path=experiment_spec_path,
        expected_experiment_spec_hash=expected_experiment_spec_hash,
    )
    if config.ppo.total_timesteps % config.ppo.n_steps:
        raise BalancedSectorTrainerError(
            "SB3 rollout length would pad the scheduled transition total"
        )

    started_at = _utc_now()
    started_clock = time.perf_counter()
    resolution = None
    model: RecurrentPPO | None = None
    vector: DummyVecEnv | None = None
    environment: BalancedWindowTrainingEnv | None = None
    callback: BalancedSectorProgressCallback | None = None
    optimizer_timesteps = 0
    actual_device: str | None = None
    training_diagnostics = None
    input_hashes_before = {
        symbol: _dataframe_digest(frame) for symbol, frame in train_data.items()
    }

    def finish(
        status: str,
        message: str,
        error: str | None = None,
    ) -> BalancedSectorTrainingResult:
        elapsed = max(0.0, time.perf_counter() - started_clock)
        expected_windows = {
            symbol: schedule.rounds for symbol in schedule.symbols
        }
        completed_windows = {
            symbol: int(
                (environment.completed_window_counts if environment else {}).get(
                    symbol, 0
                )
            )
            for symbol in schedule.symbols
        }
        expected_transitions = {
            symbol: schedule.expected_transitions_per_symbol
            for symbol in schedule.symbols
        }
        actual_transitions = {
            symbol: int(
                (environment.actual_transitions_by_symbol if environment else {}).get(
                    symbol, 0
                )
            )
            for symbol in schedule.symbols
        }
        actual_scheduled = sum(actual_transitions.values())
        contributions = {
            symbol: (
                100.0 * count / actual_scheduled if actual_scheduled else 0.0
            )
            for symbol, count in actual_transitions.items()
        }
        methodology = (
            callback.methodology_diagnostics
            if callback is not None and actual_scheduled
            else _empty_methodology_diagnostics()
        )
        collapse = dict(methodology.get("collapse_diagnostics", {}))
        warnings = list(collapse.get("warnings", ()))
        if status != "completed":
            warnings.append("methodology_run_not_completed")
        termination_reasons: Counter[str] = Counter()
        if environment is not None:
            for index in environment.completed_records:
                termination_reasons[
                    schedule.records[index].boundary_kind
                ] += 1
        reset_snapshots = tuple(
            environment.reset_snapshots if environment is not None else ()
        )
        portfolio_reset = bool(
            reset_snapshots
            and all(
                snapshot.get("episode_start") is True
                and snapshot.get("cash") == SingleSymbolEnvConfig().initial_cash
                and snapshot.get("shares_held") == 0
                and snapshot.get("average_entry_price") == 0.0
                and snapshot.get("current_position_value") == 0.0
                and snapshot.get("realized_profit_loss") == 0.0
                and snapshot.get("unrealized_profit_loss") == 0.0
                and snapshot.get("total_transaction_costs") == 0.0
                and snapshot.get("number_of_trades") == 0
                and snapshot.get("drawdown") == 0.0
                and snapshot.get("portfolio_value")
                == SingleSymbolEnvConfig().initial_cash
                and snapshot.get("peak_portfolio_value")
                == SingleSymbolEnvConfig().initial_cash
                for snapshot in reset_snapshots
            )
        )
        completed_total = sum(completed_windows.values())
        boundary_resets = callback.environment_episode_resets if callback else 0
        observed_episode_starts = (
            callback.policy_episode_start_flags if callback else []
        )
        expected_episode_starts = [
            index % schedule.window_transition_count == 0
            for index in range(actual_scheduled)
        ]
        directly_observed_episode_starts = bool(
            status == "completed"
            and observed_episode_starts == expected_episode_starts
            and sum(observed_episode_starts) == completed_total
        )
        recurrent_reset = bool(
            status == "completed"
            and callback is not None
            and callback.first_episode_start is True
            and boundary_resets == completed_total
            and directly_observed_episode_starts
            and all(record.recurrent_state_reset for record in schedule.records)
        )
        boundary_episode_start = directly_observed_episode_starts
        periodic = tuple(
            item.to_dict()
            for item in (callback.periodic_diagnostics if callback else ())
        )
        padding = max(0, optimizer_timesteps - schedule.expected_total_scheduled_transitions)
        config_payload = config.to_dict()
        fingerprint_identity: dict[str, object] = {
            "git_commit": _git_commit(Path(project_root)),
            "dependencies": _dependency_versions(),
            "sector_universe_hash": universe.universe_hash,
            "taxonomy_version": config.taxonomy_version,
            "trainer_version": config.trainer_version,
            "recurrent_contract_version": universe.recurrent_contract_version,
            "feature_version": universe.feature_version,
            "environment_version": universe.environment_version,
            "sampling_version": config.sampling_version,
            "reward": config.reward.to_metadata(),
            "action_validity_version": config.action_validity_version,
            "invalid_action_mode": config.invalid_action_mode,
            "invalid_action_mode_version": config.invalid_action_mode_version,
            "window_transition_count": config.window_transition_count,
            "balanced_rounds": config.balanced_rounds,
            "symbol_count": len(schedule.symbols),
            "expected_transitions_per_symbol": schedule.expected_transitions_per_symbol,
            "data_schedule_seed": config.data_schedule_seed,
            "model_seed": config.model_seed,
            "experiment_seed_set": list(config.experiment_seed_set),
            "device": config.ppo.device,
            "normalization_scope": config.normalization_scope,
            "ppo_lstm_config": config.ppo.to_dict(),
            "schedule_digest": schedule.schedule_digest,
            "authorized_experiment_spec_hash": (
                authorized_spec.get("experiment_spec_hash")
                if authorized_spec is not None
                else None
            ),
            "train_frame_hashes_before": input_hashes_before,
            "observed_policy_episode_start_count": sum(
                observed_episode_starts
            ),
        }
        fingerprint = {
            **fingerprint_identity,
            "run_fingerprint_hash": canonical_payload_hash(fingerprint_identity),
        }
        return BalancedSectorTrainingResult(
            sector_id=config.sector_id,
            sector_name=config.sector_name,
            trainer_version=config.trainer_version,
            algorithm=config.ppo.algorithm,
            policy=config.ppo.policy,
            taxonomy_version=config.taxonomy_version,
            sector_universe_hash=universe.universe_hash,
            recurrent_contract_version=universe.recurrent_contract_version,
            environment_version=universe.environment_version,
            feature_version=universe.feature_version,
            constituent_symbols=schedule.symbols,
            target_symbol=schedule.target_symbol,
            target_excluded_from_pretraining=(
                schedule.target_excluded_from_pretraining
            ),
            sampling_version=config.sampling_version,
            reward_version=config.reward.reward_version,
            action_validity_version=config.action_validity_version,
            invalid_action_mode=config.invalid_action_mode,
            invalid_action_mode_version=config.invalid_action_mode_version,
            normalization_scope=config.normalization_scope,
            observation_shape=universe.observation_shape,
            observation_features=universe.observation_features,
            symbol_identity_in_observation=config.observation_includes_symbol_identity,
            model_seed=config.model_seed,
            data_schedule_seed=config.data_schedule_seed,
            experiment_seed_set=config.experiment_seed_set,
            window_transition_count=schedule.window_transition_count,
            source_rows_per_window=schedule.source_rows_per_window,
            balanced_rounds=schedule.rounds,
            scheduled_window_count=len(schedule.records),
            schedule_digest=schedule.schedule_digest,
            expected_transitions_per_symbol=(
                schedule.expected_transitions_per_symbol
            ),
            expected_total_scheduled_transitions=(
                schedule.expected_total_scheduled_transitions
            ),
            actual_scheduled_transitions=actual_scheduled,
            optimizer_requested_timesteps=config.ppo.total_timesteps,
            optimizer_actual_timesteps=optimizer_timesteps,
            rollout_padding_timesteps=padding,
            duration_seconds=elapsed,
            requested_device=config.ppo.device,
            resolved_device=(resolution.resolved_device if resolution else None),
            device=actual_device,
            parameter_count=(
                count_model_parameters(model) if model is not None else 0
            ),
            expected_windows_by_symbol=expected_windows,
            completed_windows_by_symbol=completed_windows,
            expected_transitions_by_symbol=expected_transitions,
            actual_transitions_by_symbol=actual_transitions,
            contribution_percentages=contributions,
            coverage_statistics=tuple(
                item.to_dict() for item in schedule.symbol_statistics
            ),
            termination_reasons=dict(termination_reasons),
            reset_snapshots=reset_snapshots,
            passive_post_schedule_resets=int(
                getattr(environment, "passive_post_schedule_reset_count", 0)
                if environment is not None
                else 0
            ),
            first_episode_start=(callback.first_episode_start if callback else None),
            window_boundary_episode_start_verified=boundary_episode_start,
            portfolio_reset_verified=portfolio_reset,
            recurrent_reset_verified=recurrent_reset,
            rollout_boundaries_observed=(
                callback.rollout_boundaries_observed if callback else 0
            ),
            rollout_continuity_checks=(
                callback.rollout_continuity_checks if callback else 0
            ),
            rollout_continuity_verified=(
                callback.rollout_continuity_verified if callback else False
            ),
            environment_episode_resets=boundary_resets,
            reward_action_diagnostics=methodology,
            collapse_diagnostics=collapse,
            training_diagnostics=(
                training_diagnostics if status == "completed" else None
            ),
            periodic_training_diagnostics=periodic,
            reproducibility_fingerprint=fingerprint,
            config=config_payload,
            warnings=tuple(dict.fromkeys(warnings)),
            status=status,
            started_at=started_at,
            completed_at=_utc_now(),
            message=message,
            error=error,
            model=model if status == "completed" else None,
        )

    try:
        resolution = resolve_torch_device(config.ppo.device)
        _seed_everything(config.model_seed, resolved_device=resolution.resolved_device)
        environment_config = SingleSymbolEnvConfig(
            reward_version=config.reward.reward_version,
            action_validity_version=config.action_validity_version,
            invalid_action_mode=config.invalid_action_mode,
            portfolio_growth_reward_weight=config.reward.portfolio_growth_weight,
            transaction_cost_penalty_weight=config.reward.transaction_cost_weight,
            drawdown_penalty_weight=config.reward.drawdown_increment_weight,
            invalid_action_penalty=config.reward.invalid_action_penalty,
        )
        environment = BalancedWindowTrainingEnv(
            schedule,
            train_data,
            config=environment_config,
            cycle_schedule=False,
        )
        if environment.observation_space.shape != EXPECTED_OBSERVATION_SHAPE:
            raise BalancedSectorTrainerError("observation shape is incompatible")
        if tuple(environment.observation_feature_names) != (
            *DEFAULT_OBSERVATION_FEATURES,
            *DYNAMIC_PORTFOLIO_FEATURES,
        ):
            raise BalancedSectorTrainerError(
                "observation ordering or identity exclusion is incompatible"
            )
        environment.action_space.seed(config.model_seed)
        vector = DummyVecEnv([lambda: Monitor(environment)])
        vector.seed(config.model_seed)
        model = RecurrentPPO(
            config.ppo.policy,
            vector,
            verbose=0,
            **config.ppo.model_kwargs(
                resolved_device=resolution.resolved_device
            ),
        )
        actual_device = verify_sb3_model_device(model, resolution)
        callback = BalancedSectorProgressCallback(
            symbol=config.sector_name,
            requested_timesteps=config.ppo.total_timesteps,
            interval_steps=max(
                config.ppo.n_steps,
                config.ppo.total_timesteps // 10,
            ),
            diagnostic_interval_rollouts=max(
                1,
                (config.ppo.total_timesteps // config.ppo.n_steps) // 10,
            ),
            handler=progress_callback,
        )
        model.learn(
            total_timesteps=config.ppo.total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        synchronize_torch_device(resolution.resolved_device)
        actual_device = verify_sb3_model_device(model, resolution)
        optimizer_timesteps = int(model.num_timesteps)
        _assert_finite_training_state(model)
        training_diagnostics = callback.training_diagnostics
        if callback.cancel_requested:
            model = None
            return finish(
                "interrupted",
                "Balanced sector training stopped cooperatively; no model retained.",
            )
        input_hashes_after = {
            symbol: _dataframe_digest(frame) for symbol, frame in train_data.items()
        }
        if input_hashes_after != input_hashes_before:
            raise BalancedSectorTrainerError(
                "canonical TRAIN frames were mutated during balanced training"
            )
        actual_scheduled = sum(environment.actual_transitions_by_symbol.values())
        if actual_scheduled != schedule.expected_total_scheduled_transitions:
            raise BalancedSectorTrainerError(
                "environment did not consume the exact scheduled transitions"
            )
        if optimizer_timesteps != schedule.expected_total_scheduled_transitions:
            raise BalancedSectorTrainerError(
                "SB3 optimizer timesteps differ from scheduled transitions"
            )
        expected_by_symbol = schedule.expected_transitions_per_symbol
        if any(
            environment.actual_transitions_by_symbol[symbol]
            != expected_by_symbol
            for symbol in schedule.symbols
        ):
            raise BalancedSectorTrainerError("per-symbol exposure is unequal")
        if any(
            environment.completed_window_counts[symbol] != schedule.rounds
            for symbol in schedule.symbols
        ):
            raise BalancedSectorTrainerError("window completion counts differ")
        environment.assert_exact_schedule_completion()
        return finish(
            "completed",
            "Completed the exact TRAIN-only balanced-window methodology schedule.",
        )
    except KeyboardInterrupt:
        optimizer_timesteps = int(model.num_timesteps) if model is not None else 0
        model = None
        return finish(
            "interrupted",
            "Balanced sector training interrupted; no model retained.",
        )
    except Exception as exc:
        optimizer_timesteps = int(model.num_timesteps) if model is not None else 0
        model = None
        return finish(
            "failed",
            "Balanced sector training failed safely; no model retained.",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if vector is not None:
            vector.close()


__all__ = (
    "BalancedSectorTrainerError",
    "MAX_METHODOLOGY_SMOKE_ROUNDS",
    "MAX_METHODOLOGY_SMOKE_SYMBOLS",
    "MAX_METHODOLOGY_SMOKE_TRANSITIONS",
    "train_balanced_sector_recurrent_ppo",
)
