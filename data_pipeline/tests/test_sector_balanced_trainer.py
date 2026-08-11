"""Offline integration tests for the repaired 6E.2 sector trainer."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    SAVED_MODELS_DIR,
)
from feature_engineering.schemas import FEATURE_VERSION
from reinforcement_learning.environments.action_validity import (
    MASK_ACTION_VALIDITY_VERSION,
    MASK_MODE,
    PENALTY_ACTION_VALIDITY_VERSION,
)
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
)
from reinforcement_learning.sector_universe import (
    SECTOR_TAXONOMY_VERSION,
    SECTOR_UNIVERSE_SCHEMA_VERSION,
)
from reinforcement_learning.training.recurrent_config import RecurrentPPOConfig
from reinforcement_learning.training.sector_balanced_callbacks import (
    BalancedSectorProgressCallback,
)
from reinforcement_learning.training.sector_balanced_config import (
    BALANCED_SECTOR_CONFIG_VERSION,
    BALANCED_SECTOR_TRAINER_VERSION,
    PREDECLARED_RESEARCH_PURPOSE,
    BalancedSectorRecurrentPPOConfig,
)
from reinforcement_learning.training.sector_balanced_trainer import (
    BalancedSectorTrainerError,
    train_balanced_sector_recurrent_ppo,
)
from reinforcement_learning.training.sector_balanced_windows import (
    BalancedWindowTrainingEnv,
    PREDECLARED_MODEL_SEEDS,
    SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
    build_balanced_window_schedule,
)
from reinforcement_learning.training.sector_methodology_diagnostics import (
    action_pattern_digest,
    detect_action_collapse,
)
from reinforcement_learning.training.sector_recurrent_config import (
    COMMERCIAL_BANKS_SECTOR_ID,
    COMMERCIAL_BANKS_SECTOR_NAME,
)
from reinforcement_learning.training.sector_recurrent_trainer import (
    LoadedSectorTrainingUniverse,
)


UNIVERSE_HASH = "a" * 64
SMOKE_SYMBOLS = ("ABL", "MCB")
ALL_BANKS = (
    "ABL", "AKBL", "BAFL", "BAHL", "BIPL", "BML", "BOK", "BOP", "FABL",
    "HBL", "HMB", "JSBL", "MCB", "MEBL", "NBP", "SBL", "SCBPL", "SNBL", "UBL",
)


def _manifest(symbols: tuple[str, ...]) -> dict[str, object]:
    return {
        "artifact_schema_version": SECTOR_UNIVERSE_SCHEMA_VERSION,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "universe_hash": UNIVERSE_HASH,
        "sector": {
            "sector_id": COMMERCIAL_BANKS_SECTOR_ID,
            "sector_name": COMMERCIAL_BANKS_SECTOR_NAME,
        },
        "experiment_mode": {
            "pretraining_constituent_symbols": list(symbols),
        },
    }


def _frame(symbol: str, rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    open_price = 100.0 + index / 100.0
    values: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.bdate_range("2020-01-01", periods=rows),
        "open": open_price,
        "high": open_price + 1.0,
        "low": open_price - 1.0,
        "close": open_price + 0.25,
        "volume": 10_000.0 + index,
    }
    for offset, feature in enumerate(DEFAULT_OBSERVATION_FEATURES, start=10):
        values[feature] = (index - index.mean()) / offset
    return pd.DataFrame(values)


def _universe(
    symbols: tuple[str, ...],
    *,
    rows: int,
    frames: dict[str, pd.DataFrame] | None = None,
) -> LoadedSectorTrainingUniverse:
    source = frames or {symbol: _frame(symbol, rows) for symbol in symbols}
    manifest = _manifest(symbols)
    return LoadedSectorTrainingUniverse(
        manifest_path=Path("fixture-manifest.json"),
        manifest=manifest,
        universe_hash=UNIVERSE_HASH,
        symbols=symbols,
        train_data=source,
        train_rows={symbol: len(source[symbol]) for symbol in symbols},
        total_train_rows=sum(len(source[symbol]) for symbol in symbols),
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        observation_shape=(17,),
    )


def _small_config(
    *,
    mode: str = "penalty",
) -> BalancedSectorRecurrentPPOConfig:
    transitions = len(SMOKE_SYMBOLS) * 2 * 8
    return BalancedSectorRecurrentPPOConfig(
        sector_universe_hash=UNIVERSE_HASH,
        constituent_symbols=SMOKE_SYMBOLS,
        window_transition_count=8,
        balanced_rounds=2,
        invalid_action_mode=mode,
        invalid_action_mode_version=(
            PENALTY_ACTION_VALIDITY_VERSION
            if mode == "penalty"
            else MASK_ACTION_VALIDITY_VERSION
        ),
        ppo=RecurrentPPOConfig(
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            total_timesteps=transitions,
            seed=42,
            device="cpu",
            lstm_hidden_size=8,
            net_arch=(8,),
        ),
    )


def _path_snapshot(path: Path) -> object:
    source = Path(path)
    if source.is_file():
        return ("file", sha256_file(source))
    if source.is_dir():
        return (
            "directory",
            tuple(
                sorted(
                    (
                        item.relative_to(source).as_posix(),
                        sha256_file(item),
                    )
                    for item in source.rglob("*")
                    if item.is_file()
                )
            ),
        )
    return ("missing",)


@pytest.fixture(scope="module")
def balanced_smoke_result():
    source = {symbol: _frame(symbol, 20) for symbol in SMOKE_SYMBOLS}
    pristine = {symbol: frame.copy(deep=True) for symbol, frame in source.items()}
    universe = _universe(SMOKE_SYMBOLS, rows=20, frames=source)
    partition_trace: list[tuple[str, str]] = []
    loader_trace: list[object] = []

    def partition_loader(symbol: str, partition: str, **_: object) -> None:
        partition_trace.append((symbol, partition))
        if partition != "train":
            raise AssertionError("balanced optimizer requested non-TRAIN data")

    def universe_loader(*_: object, **kwargs: object):
        supplied = kwargs.get("partition_loader")
        loader_trace.append(supplied)
        assert supplied is partition_loader
        for symbol in SMOKE_SYMBOLS:
            supplied(symbol, "train")  # type: ignore[operator]
        return universe

    protected = {
        path: _path_snapshot(path)
        for path in (MODEL_REGISTRY_PATH, SAVED_MODELS_DIR, MODELS_DATA_DIR)
    }
    result = train_balanced_sector_recurrent_ppo(
        _small_config(),
        partition_loader=partition_loader,
        universe_loader=universe_loader,
        methodology_smoke=True,
    )
    for symbol in SMOKE_SYMBOLS:
        pd.testing.assert_frame_equal(source[symbol], pristine[symbol])
    assert protected == {path: _path_snapshot(path) for path in protected}
    return result, tuple(partition_trace), tuple(loader_trace)


def test_balanced_config_versions_seed_separation_and_exact_rollout_alignment() -> None:
    manifest = _manifest(("ABL", "MCB", "UBL"))
    config = BalancedSectorRecurrentPPOConfig.from_manifest(
        manifest,
        constituent_symbols=("MCB", "UBL"),
        rounds=2,
        window_transition_count=512,
        data_schedule_seed=9,
        model_seed=42,
        n_steps=128,
    )
    assert config.config_version == BALANCED_SECTOR_CONFIG_VERSION
    assert config.trainer_version == BALANCED_SECTOR_TRAINER_VERSION
    assert config.sampling_version == SECTOR_BALANCED_WINDOW_SAMPLING_VERSION
    assert config.constituent_symbols == ("MCB", "UBL")
    assert config.expected_transitions_per_symbol == 1_024
    assert config.expected_total_transitions == 2_048
    assert config.ppo.total_timesteps == 2_048
    assert config.experiment_seed_set == PREDECLARED_MODEL_SEEDS
    assert config.data_schedule_seed == 9
    assert config.model_seed == 42
    assert not config.observation_includes_symbol_identity

    other_model_seed = config.with_model_seed(43)
    assert other_model_seed.model_seed == 43
    assert other_model_seed.data_schedule_seed == config.data_schedule_seed
    assert other_model_seed.constituent_symbols == config.constituent_symbols

    with pytest.raises(ValueError, match="divide exactly"):
        replace(
            _small_config(),
            ppo=replace(_small_config().ppo, n_steps=6, batch_size=3),
        )
    with pytest.raises(ValueError, match="exact scheduled exposure"):
        replace(
            _small_config(),
            ppo=replace(_small_config().ppo, total_timesteps=24),
        )


def test_mask_mode_is_versioned_but_training_fails_closed() -> None:
    config = _small_config(mode=MASK_MODE)
    assert config.invalid_action_mode_version == MASK_ACTION_VALIDITY_VERSION
    universe = _universe(SMOKE_SYMBOLS, rows=20)
    result = train_balanced_sector_recurrent_ppo(
        config,
        universe_loader=lambda *_args, **_kwargs: universe,
        methodology_smoke=True,
    )
    assert result.status == "failed"
    assert result.model is None
    assert "unsupported_or_deferred" in str(result.error)
    assert result.actual_scheduled_transitions == 0


def test_full_predeclared_run_refuses_to_start_without_future_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(ALL_BANKS)
    config = BalancedSectorRecurrentPPOConfig.predeclared_research_from_manifest(
        manifest,
        model_seed=42,
    )
    assert config.execution_purpose == PREDECLARED_RESEARCH_PURPOSE
    assert config.expected_transitions_per_symbol == 10_240
    assert config.expected_total_transitions == 194_560
    universe = _universe(ALL_BANKS, rows=513)

    def forbidden_model(*_: object, **__: object) -> None:
        raise AssertionError("RecurrentPPO must not be constructed")

    monkeypatch.setattr(
        "reinforcement_learning.training.sector_balanced_trainer.RecurrentPPO",
        forbidden_model,
    )
    with pytest.raises(BalancedSectorTrainerError, match="explicit future authorization"):
        train_balanced_sector_recurrent_ppo(
            config,
            universe_loader=lambda *_args, **_kwargs: universe,
        )


def test_balanced_smoke_is_train_only_exact_and_does_not_mutate_artifacts(
    balanced_smoke_result,
) -> None:
    result, partition_trace, loader_trace = balanced_smoke_result
    assert result.succeeded, result.error
    assert loader_trace and all(item is loader_trace[0] for item in loader_trace)
    assert partition_trace == (("ABL", "train"), ("MCB", "train"))
    assert result.expected_total_scheduled_transitions == 32
    assert result.actual_scheduled_transitions == 32
    assert result.optimizer_requested_timesteps == 32
    assert result.optimizer_actual_timesteps == 32
    assert result.rollout_padding_timesteps == 0
    assert result.expected_transitions_by_symbol == {"ABL": 16, "MCB": 16}
    assert result.actual_transitions_by_symbol == {"ABL": 16, "MCB": 16}
    assert result.expected_windows_by_symbol == {"ABL": 2, "MCB": 2}
    assert result.completed_windows_by_symbol == {"ABL": 2, "MCB": 2}
    assert result.first_episode_start is True
    assert result.window_boundary_episode_start_verified
    assert result.recurrent_reset_verified
    assert result.portfolio_reset_verified
    assert result.rollout_boundaries_observed == 8
    assert result.training_diagnostics is not None
    assert result.training_diagnostics.timesteps == 32
    assert result.reward_action_diagnostics["reward_and_action"][
        "reward_distribution"
    ]["count"] == 32
    assert set(
        result.reward_action_diagnostics["reward_and_action"]["per_symbol"]
    ) == set(SMOKE_SYMBOLS)
    assert result.observation_shape == (17,)
    assert result.observation_features == DEFAULT_OBSERVATION_FEATURES
    assert not result.symbol_identity_in_observation
    assert result.normalization_scope == "symbol"
    assert result.reproducibility_fingerprint["model_seed"] == 42
    assert result.reproducibility_fingerprint["data_schedule_seed"] == 42
    assert result.model is not None


def test_periodic_ppo_diagnostics_are_whitelisted_and_deduplicated() -> None:
    callback = BalancedSectorProgressCallback(
        symbol="COMMERCIAL BANKS",
        requested_timesteps=128,
        interval_steps=64,
        diagnostic_interval_rollouts=1,
    )
    callback.model = SimpleNamespace(
        logger=SimpleNamespace(
            name_to_value={
                "train/n_updates": 2,
                "train/approx_kl": np.float32(0.01),
                "train/clip_fraction": 0.2,
                "train/entropy_loss": -1.0,
                "train/explained_variance": 0.4,
                "train/policy_gradient_loss": -0.02,
                "train/value_loss": 0.3,
                "train/learning_rate": 3e-4,
                "unsafe/unbounded_field": 999,
            }
        )
    )
    callback.num_timesteps = 64
    callback._capture_logger_diagnostics()
    callback._capture_logger_diagnostics()
    assert len(callback.periodic_diagnostics) == 1
    first = callback.periodic_diagnostics[0].to_dict()
    assert first == {
        "timesteps": 64,
        "updates": 2,
        "approximate_kl": pytest.approx(0.01),
        "clip_fraction": 0.2,
        "entropy_loss": -1.0,
        "explained_variance": 0.4,
        "policy_gradient_loss": -0.02,
        "value_loss": 0.3,
        "learning_rate": 3e-4,
    }
    callback.num_timesteps = 128
    callback._capture_logger_diagnostics()
    assert [item.timesteps for item in callback.periodic_diagnostics] == [64, 128]


def test_action_collapse_diagnostics_are_warning_only_and_thresholds_are_strict() -> None:
    shared = action_pattern_digest([2, 2, 2])
    collapsed = detect_action_collapse(
        selected_action_counts={"hold": 5, "buy": 5, "sell": 90},
        invalid_action_rate=0.90,
        per_symbol_exposure_percentages={"ABL": 1.0, "MCB": 2.0},
        per_symbol_trade_counts={"ABL": 0, "MCB": 0},
        per_symbol_action_digests={"ABL": shared, "MCB": shared},
    )
    assert collapsed["thresholds_are_warnings_not_selection_rules"] is True
    assert set(collapsed["warnings"]) == {
        "possible_action_collapse",
        "possible_invalid_action_attractor",
        "possible_cash_policy_collapse",
        "zero_trade_symbols_observed",
        "identical_policy_behavior_across_symbols",
    }

    boundary = detect_action_collapse(
        selected_action_counts={"hold": 10, "buy": 10, "sell": 80},
        invalid_action_rate=0.80,
        per_symbol_exposure_percentages={"ABL": 5.0},
        per_symbol_trade_counts={"ABL": 1},
        per_symbol_action_digests={"ABL": action_pattern_digest([0, 1, 2])},
    )
    assert boundary["warnings"] == []


def test_result_metadata_is_json_serializable_without_the_in_memory_model(
    balanced_smoke_result,
) -> None:
    result, _, _ = balanced_smoke_result
    payload = result.to_dict()
    assert "model" not in payload
    assert payload["status"] == "completed"
    assert payload["periodic_training_diagnostics"]
    json.dumps(payload, allow_nan=False)


class _RolloutCaptureCallback(BaseCallback):
    """Copy pre-optimizer rewards and raw environment diagnostics."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.raw_rewards: list[float] = []
        self.buffer_rewards: np.ndarray | None = None

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.raw_rewards.append(float(info["reward_breakdown"]["total_reward"]))
        return True

    def _on_rollout_end(self) -> None:
        self.buffer_rewards = self.model.rollout_buffer.rewards.copy()


def test_recurrent_ppo_bootstraps_artificial_window_terminal_value() -> None:
    """Exercise sb3-contrib's real timeout path, not a local approximation."""

    source = {"ABL": _frame("ABL", 20)}
    schedule = build_balanced_window_schedule(
        source,
        universe_hash=UNIVERSE_HASH,
        rounds=2,
        window_transition_count=8,
    )
    assert schedule.records[0].boundary_kind == "artificial_window_truncation"
    environment = BalancedWindowTrainingEnv(schedule, source)
    vector = DummyVecEnv([lambda: Monitor(environment)])
    try:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vector,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            gamma=0.99,
            seed=42,
            device="cpu",
            policy_kwargs={
                "lstm_hidden_size": 8,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
                "net_arch": {"pi": [8], "vf": [8]},
            },
            verbose=0,
        )

        def fixed_terminal_value(
            observation: torch.Tensor,
            lstm_states: tuple[torch.Tensor, torch.Tensor],
            episode_starts: torch.Tensor,
        ) -> torch.Tensor:
            del lstm_states, episode_starts
            return torch.full(
                (observation.shape[0],),
                2.0,
                dtype=torch.float32,
                device=observation.device,
            )

        model.policy.predict_values = fixed_terminal_value  # type: ignore[method-assign]
        capture = _RolloutCaptureCallback()
        model.learn(total_timesteps=8, callback=capture, progress_bar=False)
        assert capture.buffer_rewards is not None
        assert capture.raw_rewards[-1] == pytest.approx(0.0)
        assert capture.buffer_rewards[-1, 0] == pytest.approx(0.99 * 2.0)
        assert model.rollout_buffer.episode_starts[:, 0].tolist() == pytest.approx(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    finally:
        vector.close()


def test_recurrent_gae_does_not_propagate_across_window_episode_start() -> None:
    """The next window's return cannot alter the previous window's GAE."""

    def advantages(second_window_reward: float) -> np.ndarray:
        buffer = RecurrentRolloutBuffer(
            4,
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            spaces.Discrete(3),
            hidden_state_shape=(4, 1, 1, 2),
            device="cpu",
            gae_lambda=1.0,
            gamma=1.0,
            n_envs=1,
        )
        buffer.rewards[:, 0] = [1.0, 1.0, second_window_reward, second_window_reward]
        buffer.values[:, 0] = 0.0
        buffer.episode_starts[:, 0] = [1.0, 0.0, 1.0, 0.0]
        buffer.compute_returns_and_advantage(
            last_values=torch.zeros(1),
            dones=np.asarray([True]),
        )
        return buffer.advantages[:, 0].copy()

    low = advantages(0.0)
    high = advantages(100.0)
    assert low[:2].tolist() == pytest.approx([2.0, 1.0])
    assert high[:2].tolist() == pytest.approx([2.0, 1.0])
    assert low[2:].tolist() != pytest.approx(high[2:].tolist())
