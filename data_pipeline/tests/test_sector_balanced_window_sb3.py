"""Offline Gym/SB3 boundary tests for the balanced sector-window controller."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from reinforcement_learning.environments import SingleSymbolEnvConfig
from reinforcement_learning.training.sector_balanced_windows import (
    BalancedWindowScheduleExhausted,
    BalancedWindowTrainingEnv,
    build_balanced_window_schedule,
    materialize_scheduled_window,
)


FEATURES = ("simple_return", "rsi_14")
UNIVERSE_HASH = "a" * 64
WINDOW_TRANSITIONS = 512
WINDOW_SOURCE_ROWS = WINDOW_TRANSITIONS + 1


def _train_frame(symbol: str, *, rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    open_price = 100.0 + index / 100.0
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "open": open_price,
            "high": open_price + 2.0,
            "low": open_price - 1.0,
            "close": open_price + 1.0,
            "volume": 10_000.0 + index,
            "simple_return": index / 10_000.0,
            "rsi_14": -1.0 + (2.0 * index / max(1, rows - 1)),
        }
    )


def _config() -> SingleSymbolEnvConfig:
    return SingleSymbolEnvConfig(
        feature_columns=FEATURES,
        commission_rate=0.0,
        slippage_rate=0.0,
    )


def _schedule(frame: pd.DataFrame, *, rounds: int):
    return build_balanced_window_schedule(
        {"AAA": frame},
        universe_hash=UNIVERSE_HASH,
        rounds=rounds,
        window_transition_count=WINDOW_TRANSITIONS,
        data_schedule_seed=42,
    )


def _tiny_schedule(frame: pd.DataFrame, *, rounds: int):
    return build_balanced_window_schedule(
        {"AAA": frame},
        universe_hash=UNIVERSE_HASH,
        rounds=rounds,
        window_transition_count=4,
        data_schedule_seed=42,
    )


class _RolloutCapture(BaseCallback):
    """Copy rollout locals before SB3 mutates timeout rewards in-place."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.raw_rewards: list[float] = []
        self.infos: list[dict[str, object]] = []

    def _on_step(self) -> bool:
        self.raw_rewards.append(float(np.asarray(self.locals["rewards"])[0]))
        self.infos.append(dict(self.locals["infos"][0]))
        return True


def _recurrent_model(
    vector: DummyVecEnv,
    *,
    n_steps: int,
    gamma: float = 0.9,
) -> RecurrentPPO:
    return RecurrentPPO(
        "MlpLstmPolicy",
        vector,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        gamma=gamma,
        gae_lambda=1.0,
        policy_kwargs={
            "lstm_hidden_size": 4,
            "n_lstm_layers": 1,
            "net_arch": [4],
        },
        seed=42,
        device="cpu",
        verbose=0,
    )


def _setup_rollout(
    model: RecurrentPPO,
    callback: _RolloutCapture,
    *,
    total_timesteps: int,
) -> BaseCallback:
    _, initialized = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="balanced_window_contract_test",
        progress_bar=False,
    )
    return initialized


def _finish_window(
    environment: BalancedWindowTrainingEnv,
    *,
    first_action: int = 0,
) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
    result = None
    for transition in range(WINDOW_TRANSITIONS):
        result = environment.step(first_action if transition == 0 else 0)
        if transition < WINDOW_TRANSITIONS - 1:
            assert result[2:4] == (False, False)
    assert result is not None
    return result


def test_window_materialization_uses_513_rows_for_512_transitions() -> None:
    frame = _train_frame("AAA", rows=WINDOW_SOURCE_ROWS)
    schedule = _schedule(frame, rounds=1)
    record = schedule.records[0]

    window = materialize_scheduled_window(record, {"AAA": frame})

    assert len(window) == WINDOW_SOURCE_ROWS
    assert record.source_row_count == WINDOW_SOURCE_ROWS
    assert record.expected_transition_count == WINDOW_TRANSITIONS
    assert record.actual_transition_count == WINDOW_TRANSITIONS
    assert record.final_observation_date == window["date"].iloc[-2].date().isoformat()
    assert record.final_execution_date == window["date"].iloc[-1].date().isoformat()

    environment = BalancedWindowTrainingEnv(
        schedule,
        {"AAA": frame},
        config=_config(),
    )
    try:
        environment.reset(seed=42)
        _, _, terminated, truncated, info = _finish_window(environment)
        assert terminated is True
        assert truncated is False
        assert info["window_transition_number"] == WINDOW_TRANSITIONS
        assert environment.actual_transitions_by_symbol == {"AAA": WINDOW_TRANSITIONS}
    finally:
        environment.close()


def test_artificial_window_truncates_while_natural_train_end_terminates() -> None:
    long_frame = _train_frame("AAA", rows=2 * WINDOW_SOURCE_ROWS)
    artificial_schedule = _schedule(long_frame, rounds=2)
    artificial_record = artificial_schedule.records[0]
    assert artificial_record.boundary_kind == "artificial_window_truncation"

    artificial = BalancedWindowTrainingEnv(
        artificial_schedule,
        {"AAA": long_frame},
        config=_config(),
    )
    try:
        artificial.reset(seed=42)
        observation, _, terminated, truncated, info = _finish_window(artificial)
        assert terminated is False
        assert truncated is True
        assert info["TimeLimit.truncated"] is True
        np.testing.assert_array_equal(info["terminal_observation"], observation)
    finally:
        artificial.close()

    exact_frame = _train_frame("AAA", rows=WINDOW_SOURCE_ROWS)
    natural_schedule = _schedule(exact_frame, rounds=1)
    assert natural_schedule.records[0].boundary_kind == "natural_train_partition_end"
    natural = BalancedWindowTrainingEnv(
        natural_schedule,
        {"AAA": exact_frame},
        config=_config(),
    )
    try:
        natural.reset(seed=42)
        _, _, terminated, truncated, info = _finish_window(natural)
        assert terminated is True
        assert truncated is False
        assert info["TimeLimit.truncated"] is False
    finally:
        natural.close()


def test_dummy_vec_env_preserves_terminal_observation_and_auto_resets_portfolio() -> None:
    frame = _train_frame("AAA", rows=2 * WINDOW_SOURCE_ROWS)
    schedule = _schedule(frame, rounds=2)
    first_record, second_record = schedule.records
    assert first_record.boundary_kind == "artificial_window_truncation"
    environment = BalancedWindowTrainingEnv(
        schedule,
        {"AAA": frame},
        config=_config(),
    )
    vector = DummyVecEnv([lambda: Monitor(environment)])
    try:
        first_observation = vector.reset()[0].copy()
        final_observation = None
        final_info = None
        for transition in range(WINDOW_TRANSITIONS):
            action = np.asarray([1 if transition == 0 else 0])
            observation, _, done, infos = vector.step(action)
            if transition < WINDOW_TRANSITIONS - 1:
                assert not bool(done[0])
            else:
                assert bool(done[0])
                final_observation = observation[0].copy()
                final_info = infos[0]

        assert final_info is not None and final_observation is not None
        assert final_info["TimeLimit.truncated"] is True
        assert "terminal_observation" in final_info
        assert not np.array_equal(final_info["terminal_observation"], final_observation)
        assert final_info["balanced_window_schedule_index"] == first_record.schedule_index
        assert environment.current_record == second_record
        assert len(environment.reset_snapshots) == 2
        assert environment.completed_records == [first_record.schedule_index]
        assert environment.actual_transitions_by_symbol == {"AAA": WINDOW_TRANSITIONS}

        underlying = environment._environment
        assert underlying is not None
        assert underlying.cash == pytest.approx(_config().initial_cash)
        assert underlying.shares_held == 0
        assert underlying.realized_profit_loss == 0.0
        assert underlying.unrealized_profit_loss == 0.0
        assert underlying.total_transaction_costs == 0.0
        assert underlying.current_drawdown == 0.0
        assert underlying.total_portfolio_value == pytest.approx(_config().initial_cash)
        assert final_observation[-5:].tolist() == pytest.approx(
            [1.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert not np.array_equal(first_observation, final_info["terminal_observation"])
    finally:
        vector.close()


def test_nonterminal_rollout_boundary_does_not_reset_window_or_portfolio() -> None:
    frame = _train_frame("AAA", rows=2 * WINDOW_SOURCE_ROWS)
    schedule = _schedule(frame, rounds=2)
    environment = BalancedWindowTrainingEnv(
        schedule,
        {"AAA": frame},
        config=_config(),
    )
    try:
        environment.reset(seed=42)
        record = environment.current_record
        environment.step(1)
        shares_after_buy = environment._environment.shares_held  # type: ignore[union-attr]
        assert shares_after_buy > 0

        # 128 is the PPO rollout length used by this contract test, not an
        # environment boundary. No reset or episode_start may be invented here.
        for _ in range(127):
            _, _, terminated, truncated, _ = environment.step(0)
            assert not terminated and not truncated

        assert environment.current_record == record
        assert len(environment.reset_snapshots) == 1
        assert environment.completed_records == []
        assert environment._environment is not None
        assert environment._environment.shares_held == shares_after_buy
        assert environment.actual_transitions_by_symbol == {"AAA": 128}

        for _ in range(WINDOW_TRANSITIONS - 128):
            _, _, terminated, truncated, _ = environment.step(0)
        assert truncated and not terminated
        assert environment.actual_transitions_by_symbol == {
            "AAA": WINDOW_TRANSITIONS
        }
    finally:
        environment.close()


def test_bounded_schedule_exhaustion_is_explicit_and_does_not_cycle() -> None:
    frame = _train_frame("AAA", rows=WINDOW_SOURCE_ROWS)
    schedule = _schedule(frame, rounds=1)
    environment = BalancedWindowTrainingEnv(
        schedule,
        {"AAA": frame},
        config=_config(),
        cycle_schedule=False,
    )
    try:
        environment.reset(seed=42)
        _finish_window(environment)
        observation, info = environment.reset()
        assert observation.shape == environment.observation_space.shape
        assert info["schedule_complete"] is True
        assert info["passive_post_schedule_reset"] is True
        assert environment.passive_post_schedule_reset_count == 1
        with pytest.raises(BalancedWindowScheduleExhausted, match="schedule is complete"):
            environment.step(0)
        assert environment.actual_transitions_by_symbol == {
            "AAA": WINDOW_TRANSITIONS
        }
        assert environment.completed_records == [schedule.records[0].schedule_index]
        assert len(environment.reset_snapshots) == 1
    finally:
        environment.close()


def test_recurrent_ppo_natural_train_end_has_no_timeout_value_bootstrap() -> None:
    gamma = 0.9
    terminal_value = 2.5
    frame = _train_frame("AAA", rows=5)
    schedule = _tiny_schedule(frame, rounds=1)
    assert schedule.records[0].boundary_kind == "natural_train_partition_end"
    environment = BalancedWindowTrainingEnv(
        schedule,
        {"AAA": frame},
        config=_config(),
    )
    vector = DummyVecEnv([lambda: Monitor(environment)])
    model = _recurrent_model(vector, n_steps=4, gamma=gamma)
    callback = _RolloutCapture()
    initialized = _setup_rollout(model, callback, total_timesteps=4)
    value_episode_starts: list[bool] = []

    def constant_values(
        observation: torch.Tensor,
        lstm_states: object,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        del lstm_states
        value_episode_starts.append(bool(episode_starts.detach().cpu().item()))
        return torch.full(
            (observation.shape[0], 1),
            terminal_value,
            dtype=torch.float32,
            device=observation.device,
        )

    model.policy.predict_values = constant_values  # type: ignore[method-assign]
    try:
        assert model.collect_rollouts(
            vector,
            initialized,
            model.rollout_buffer,
            n_rollout_steps=4,
        )
        raw_final_reward = callback.raw_rewards[-1]
        buffered_final_reward = float(model.rollout_buffer.rewards[-1, 0])
        assert callback.infos[-1]["TimeLimit.truncated"] is False
        assert callback.infos[-1]["balanced_window_boundary_kind"] == (
            "natural_train_partition_end"
        )
        assert buffered_final_reward == pytest.approx(raw_final_reward)
        assert buffered_final_reward != pytest.approx(
            raw_final_reward + gamma * terminal_value
        )
        # Only the post-rollout value call occurs, on DummyVecEnv's passive
        # reset observation. There is no continuing terminal-state value call.
        assert value_episode_starts == [True]
        assert model._last_episode_starts.tolist() == [True]
        assert environment.completed_records == [schedule.records[0].schedule_index]
        assert environment.passive_post_schedule_reset_count == 1
        assert environment.actual_transitions_by_symbol == {"AAA": 4}
        assert model.rollout_buffer.episode_starts[:, 0].tolist() == pytest.approx(
            [1.0, 0.0, 0.0, 0.0]
        )
    finally:
        vector.close()
