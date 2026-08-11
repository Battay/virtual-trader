"""Offline tests for the predeclared fair sector window methodology."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    SingleSymbolEnvConfig,
)
from reinforcement_learning.training.sector_balanced_windows import (
    BalancedWindowSchedule,
    BalancedWindowScheduleError,
    BalancedWindowScheduleExhausted,
    BalancedWindowTrainingEnv,
    DEFAULT_BALANCED_ROUNDS,
    DEFAULT_WINDOW_SOURCE_ROWS,
    DEFAULT_WINDOW_TRANSITIONS,
    PREDECLARED_MODEL_SEEDS,
    PREDECLARED_TOTAL_SECTOR_TRANSITIONS,
    PREDECLARED_TRANSITIONS_PER_SYMBOL,
    SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
    build_balanced_window_schedule,
    build_predeclared_fair_experiment_spec,
    canonical_payload_hash,
    materialize_scheduled_window,
)


BANKS = (
    "ABL", "AKBL", "BAFL", "BAHL", "BIPL", "BML", "BOK", "BOP", "FABL",
    "HBL", "HMB", "JSBL", "MCB", "MEBL", "NBP", "SBL", "SCBPL", "SNBL", "UBL",
)
UNIVERSE_HASH = "a" * 64
REWARD_METADATA = {
    "reward_version": "sector_reward_v1",
    "reward_equation": "growth - costs - drawdown - invalid",
    "portfolio_growth_definition": "log portfolio ratio",
    "transaction_cost_treatment": "accounting plus configured reward weight",
    "drawdown_increment_definition": "positive increment only",
    "invalid_action_treatment": "no-op plus penalty",
    "portfolio_growth_weight": 1.0,
    "transaction_cost_weight": 0.0,
    "drawdown_increment_weight": 0.1,
    "invalid_action_penalty": 0.0001,
}
ACTION_METADATA = {
    "action_validity_version": "sector_action_validity_v1",
    "methodology_version": "sector_action_validity_penalty_v1",
    "invalid_action_mode": "penalty",
    "flat_valid_actions": (0, 1),
    "long_valid_actions": (0, 2),
    "masking_status": "unsupported_or_deferred",
}


def _frame(symbol: str, rows: int = 900) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    values: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.bdate_range("2018-01-01", periods=rows),
        "open": 100.0 + index / 100,
        "high": 101.0 + index / 100,
        "low": 99.0 + index / 100,
        "close": 100.5 + index / 100,
        "volume": 10_000.0 + index,
    }
    for feature_index, feature in enumerate(DEFAULT_OBSERVATION_FEATURES):
        values[feature] = index / (feature_index + 10.0)
    return pd.DataFrame(values)


def _frames(symbols: tuple[str, ...], rows: int = 900) -> dict[str, pd.DataFrame]:
    return {symbol: _frame(symbol, rows=rows) for symbol in symbols}


def _spec(schedule) -> dict[str, object]:
    return build_predeclared_fair_experiment_spec(
        schedule,
        taxonomy_version="psx_sector_taxonomy_v1",
        trainer_version="recurrent_ppo_sector_v1",
        recurrent_contract_version="rl_recurrent_partition_v1",
        feature_version="feature_v1",
        environment_version="single_symbol_env_v1",
        ppo_configuration={"n_steps": 512, "batch_size": 64, "gamma": 0.99},
        recurrent_architecture={"lstm_hidden_size": 64, "n_lstm_layers": 1},
        reward_configuration=REWARD_METADATA,
        action_validity_configuration=ACTION_METADATA,
        git_commit="b" * 40,
    )


def test_off_by_one_contract_requires_513_rows_for_512_transitions() -> None:
    schedule = build_balanced_window_schedule(
        {"MCB": _frame("MCB", DEFAULT_WINDOW_SOURCE_ROWS)},
        universe_hash=UNIVERSE_HASH,
        rounds=1,
    )
    record = schedule.records[0]
    window = materialize_scheduled_window(record, {"MCB": _frame("MCB", 513)})
    assert record.source_row_count == 513
    assert record.source_end_row - record.source_start_row == 512
    assert record.expected_transition_count == 512
    assert record.actual_transition_count == 512
    assert len(window) == 513
    assert record.start_date == window["date"].iloc[0].date().isoformat()
    assert record.final_observation_date == window["date"].iloc[-2].date().isoformat()
    assert record.final_execution_date == window["date"].iloc[-1].date().isoformat()
    assert record.terminal_observation_date == record.final_execution_date
    assert record.boundary_kind == "natural_train_partition_end"


def test_scheduler_rejects_512_source_rows_and_non_train_partition() -> None:
    with pytest.raises(BalancedWindowScheduleError, match="at least 513 source rows"):
        build_balanced_window_schedule(
            {"MCB": _frame("MCB", 512)}, universe_hash=UNIVERSE_HASH
        )
    with pytest.raises(BalancedWindowScheduleError, match="VALIDATION or TEST"):
        build_balanced_window_schedule(
            {"MCB": _frame("MCB")},
            universe_hash=UNIVERSE_HASH,
            partition="validation",
        )
    with pytest.raises(BalancedWindowScheduleError, match="VALIDATION or TEST"):
        build_balanced_window_schedule(
            {"MCB": _frame("MCB")}, universe_hash=UNIVERSE_HASH, partition="test"
        )


def test_full_schedule_has_every_symbol_once_per_round_and_exact_exposure() -> None:
    schedule = build_balanced_window_schedule(
        _frames(BANKS), universe_hash=UNIVERSE_HASH
    )
    assert schedule.sampling_version == SECTOR_BALANCED_WINDOW_SAMPLING_VERSION
    assert schedule.full_commercial_banks_research_contract
    schedule.assert_full_research_contract()
    assert len(schedule.records) == 19 * DEFAULT_BALANCED_ROUNDS
    assert schedule.expected_transitions_per_symbol == PREDECLARED_TRANSITIONS_PER_SYMBOL
    assert schedule.expected_total_scheduled_transitions == PREDECLARED_TOTAL_SECTOR_TRANSITIONS
    for round_number in range(1, 21):
        records = [item for item in schedule.records if item.round_number == round_number]
        assert len(records) == 19
        assert {item.symbol for item in records} == set(BANKS)
        assert len({item.symbol for item in records}) == 19
    for symbol in BANKS:
        records = [item for item in schedule.records if item.symbol == symbol]
        assert len(records) == 20
        assert sum(item.actual_transition_count for item in records) == 10_240
        assert schedule.statistics_by_symbol[symbol].scheduled_transitions == 10_240


def test_schedule_hash_and_locations_are_deterministic_but_data_seed_matters() -> None:
    train = _frames(("ABL", "MCB", "UBL"), rows=1_100)
    first = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=6, data_schedule_seed=7
    )
    repeated = build_balanced_window_schedule(
        dict(reversed(tuple(train.items()))),
        universe_hash=UNIVERSE_HASH,
        rounds=6,
        data_schedule_seed=7,
    )
    changed = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=6, data_schedule_seed=8
    )
    assert first.to_dict() == repeated.to_dict()
    assert first.schedule_digest == repeated.schedule_digest
    assert first.schedule_digest != changed.schedule_digest
    assert [item.source_start_row for item in first.records] != [
        item.source_start_row for item in changed.records
    ]


def test_chronology_spread_and_overlap_statistics_are_exact() -> None:
    schedule = build_balanced_window_schedule(
        {"MCB": _frame("MCB", 900)}, universe_hash=UNIVERSE_HASH
    )
    stats = schedule.statistics_by_symbol["MCB"]
    records = sorted(schedule.records, key=lambda item: item.window_number_for_symbol)
    starts = [item.source_start_row for item in records]
    usage = sum((list(item.transition_source_rows) for item in records), [])
    counts = pd.Series(usage).value_counts()
    assert starts == sorted(starts)
    assert stats.available_unique_train_transitions == 899
    assert stats.scheduled_transitions == 10_240
    assert stats.unique_transitions_used == counts.size
    assert stats.unused_unique_train_transitions == 899 - counts.size
    assert stats.repeated_transition_occurrences == 10_240 - counts.size
    assert stats.minimum_usage_count == int(counts.min())
    assert stats.maximum_usage_count == int(counts.max())
    assert stats.overlap_percentage == pytest.approx(
        100 * (10_240 - counts.size) / 10_240
    )
    assert stats.coverage_percentage == pytest.approx(100 * counts.size / 899)
    assert stats.overlapping_window_count > 0
    assert sum(item.overlapping_transition_count for item in records) == (
        stats.repeated_transition_occurrences
    )
    assert sum(item.new_unique_transition_count for item in records) == (
        stats.unique_transitions_used
    )
    assert records[0].overlaps_previous_windows is False
    assert records[-1].cumulative_unique_coverage_percentage == pytest.approx(
        stats.coverage_percentage
    )
    assert all(item.start_chronology_quartile in {"Q1", "Q2", "Q3", "Q4"} for item in records)
    assert stats.all_chronological_quartiles_represented
    assert [item.quartile for item in stats.chronological_quartiles] == [
        "Q1", "Q2", "Q3", "Q4"
    ]
    assert all(item.unique_transitions_used > 0 for item in stats.chronological_quartiles)


def test_reuse_is_reported_when_only_one_valid_window_exists() -> None:
    schedule = build_balanced_window_schedule(
        {"MCB": _frame("MCB", 513)}, universe_hash=UNIVERSE_HASH
    )
    stats = schedule.statistics_by_symbol["MCB"]
    assert stats.unique_window_count == 1
    assert stats.reused_window_count == 19
    assert stats.unique_transitions_used == 512
    assert stats.unused_unique_train_transitions == 0
    assert stats.repeated_transition_occurrences == 9_728
    assert stats.minimum_usage_count == 20
    assert stats.maximum_usage_count == 20
    assert stats.coverage_percentage == 100.0


def test_artificial_window_uses_truncation_and_sb3_terminal_observation() -> None:
    train = {"MCB": _frame("MCB", 900)}
    schedule = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=2
    )
    assert schedule.records[0].boundary_kind == "artificial_window_truncation"
    env = BalancedWindowTrainingEnv(schedule, train)
    vector = DummyVecEnv([lambda: env])
    try:
        vector.reset()
        done = np.asarray([False])
        infos = [{}]
        for _ in range(512):
            _, _, done, infos = vector.step(np.asarray([0]))
        assert bool(done[0])
        assert infos[0]["TimeLimit.truncated"] is True
        assert infos[0]["terminal_observation"].shape == env.observation_space.shape
        assert infos[0]["balanced_window_boundary_kind"] == "artificial_window_truncation"
        assert infos[0]["window_transition_number"] == 512
        # DummyVecEnv reset the next independent window after the truncation.
        assert len(env.reset_snapshots) == 2
        assert all(snapshot["episode_start"] for snapshot in env.reset_snapshots)
    finally:
        vector.close()


def test_natural_source_end_terminates_without_truncation() -> None:
    train = {"MCB": _frame("MCB", 513)}
    schedule = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=1
    )
    env = BalancedWindowTrainingEnv(schedule, train)
    try:
        env.reset()
        for _ in range(512):
            _, _, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert truncated is False
        assert info["TimeLimit.truncated"] is False
        assert info["balanced_window_boundary_kind"] == "natural_train_partition_end"
        env.assert_exact_schedule_completion()
        snapshot_count = len(env.reset_snapshots)
        observation, passive = env.reset()
        assert env.observation_space.contains(observation)
        assert passive["schedule_complete"] is True
        assert passive["passive_post_schedule_reset"] is True
        assert passive["episode_start"] is False
        assert env.passive_post_schedule_reset_count == 1
        assert len(env.reset_snapshots) == snapshot_count
        with pytest.raises(BalancedWindowScheduleExhausted, match="passive reset"):
            env.step(0)
    finally:
        env.close()


def test_dummy_vec_final_done_uses_passive_reset_without_extra_exposure() -> None:
    train = {"MCB": _frame("MCB", 900)}
    schedule = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=1
    )
    assert schedule.records[0].boundary_kind == "artificial_window_truncation"
    env = BalancedWindowTrainingEnv(schedule, train)
    vector = DummyVecEnv([lambda: env])
    try:
        vector.reset()
        for _ in range(512):
            _, _, done, infos = vector.step(np.asarray([0]))
        assert bool(done[0])
        assert infos[0]["TimeLimit.truncated"] is True
        assert env.passive_post_schedule_reset_count == 1
        assert env.current_record is None
        assert len(env.reset_snapshots) == 1
        assert env.completed_records == [schedule.records[0].schedule_index]
        assert env.actual_transitions_by_symbol == {"MCB": 512}
        env.assert_exact_schedule_completion()
        with pytest.raises(BalancedWindowScheduleExhausted, match="passive reset"):
            vector.step(np.asarray([0]))
    finally:
        vector.close()


def test_overlapping_windows_are_deep_copies_and_reset_portfolio_state() -> None:
    source = _frame("MCB", 700)
    pristine = source.copy(deep=True)
    train = {"MCB": source}
    schedule = build_balanced_window_schedule(
        train, universe_hash=UNIVERSE_HASH, rounds=2
    )
    first, second = schedule.records
    assert set(first.transition_source_rows).intersection(second.transition_source_rows)
    first_window = materialize_scheduled_window(first, train)
    second_window = materialize_scheduled_window(second, train)
    overlap_source_row = max(first.source_start_row, second.source_start_row)
    first_local = overlap_source_row - first.source_start_row
    second_local = overlap_source_row - second.source_start_row
    original_price = float(source["open"].iloc[overlap_source_row])
    assert first_window["open"].iloc[first_local] == original_price
    assert second_window["open"].iloc[second_local] == original_price
    first_window.loc[first_local, "open"] = -123.0
    assert second_window["open"].iloc[second_local] == original_price
    pd.testing.assert_frame_equal(source, pristine)

    env = BalancedWindowTrainingEnv(schedule, train)
    try:
        env.reset()
        for _ in range(512):
            _, _, terminated, truncated, _ = env.step(1 if _ == 0 else 0)
        assert terminated or truncated
        env.reset()
        assert len(env.reset_snapshots) == 2
        assert env.reset_snapshots[1]["cash"] == env.config.initial_cash
        assert env.reset_snapshots[1]["shares_held"] == 0
        assert env.reset_snapshots[1]["realized_profit_loss"] == 0.0
        assert env.reset_snapshots[1]["unrealized_profit_loss"] == 0.0
        assert env.reset_snapshots[1]["total_transaction_costs"] == 0.0
        assert env.reset_snapshots[1]["drawdown"] == 0.0
        assert env.reset_snapshots[1]["portfolio_value"] == env.config.initial_cash
        assert env.reset_snapshots[1]["peak_portfolio_value"] == env.config.initial_cash
        assert env.reset_snapshots[1]["recurrent_state_reset_required"] is True
        # Running steps inside a window cannot create an implicit reset.
        for _ in range(10):
            env.step(0)
        assert len(env.reset_snapshots) == 2
    finally:
        env.close()
    pd.testing.assert_frame_equal(source, pristine)


def test_leave_one_out_target_gets_zero_windows_and_zero_normalization_role() -> None:
    train = _frames(("ABL", "MCB", "UBL"))
    schedule = build_balanced_window_schedule(
        {"ABL": train["ABL"], "UBL": train["UBL"]},
        universe_hash=UNIVERSE_HASH,
        rounds=3,
        target_symbol="MCB",
    )
    assert schedule.target_symbol == "MCB"
    assert schedule.target_excluded_from_pretraining
    assert schedule.symbols == ("ABL", "UBL")
    assert schedule.normalization_contributors == ("ABL", "UBL")
    assert all(record.symbol != "MCB" for record in schedule.records)
    assert {record.symbol for record in schedule.records} == {"ABL", "UBL"}
    assert all(
        sum(
            record.actual_transition_count
            for record in schedule.records
            if record.symbol == symbol
        )
        == 3 * 512
        for symbol in schedule.symbols
    )
    with pytest.raises(BalancedWindowScheduleError, match="excluded before TRAIN"):
        build_balanced_window_schedule(
            train,
            universe_hash=UNIVERSE_HASH,
            target_symbol="MCB",
        )


def test_frozen_experiment_spec_is_json_safe_deterministic_and_seed_separated() -> None:
    schedule = build_balanced_window_schedule(
        _frames(BANKS), universe_hash=UNIVERSE_HASH
    )
    first = _spec(schedule)
    second = _spec(schedule)
    assert first == second
    assert canonical_payload_hash(
        {key: value for key, value in first.items() if key not in {
            "experiment_spec_hash", "reproducibility_fingerprint"
        }}
    ) == first["experiment_spec_hash"]
    assert first["seed_policy"]["experiment_model_seeds"] == list(
        PREDECLARED_MODEL_SEEDS
    )
    assert first["seed_policy"]["data_schedule_seed"] == 42
    assert first["seed_policy"]["window_locations_constant_across_model_seeds"] is True
    assert "seed" not in first["ppo_configuration"]
    assert first["ppo_configuration"]["seed_source"] == (
        "seed_policy.experiment_model_seeds"
    )
    assert first["execution_status"] == "specification_only_not_executed"
    assert first["full_three_seed_run_requires_future_explicit_authorization"] is True
    assert first["test_sealing_rule"] == "TEST_metadata_only_never_loaded_or_evaluated"
    assert first["expected_transitions_per_symbol"] == 10_240
    assert first["expected_total_scheduled_transitions"] == 194_560
    assert first["reproducibility_fingerprint"]["schedule_digest"] == schedule.schedule_digest


def test_experiment_spec_hash_changes_with_methodology_but_not_git_commit() -> None:
    schedule = build_balanced_window_schedule(
        _frames(BANKS), universe_hash=UNIVERSE_HASH
    )
    base = _spec(schedule)
    changed_reward = build_predeclared_fair_experiment_spec(
        schedule,
        taxonomy_version="psx_sector_taxonomy_v1",
        trainer_version="recurrent_ppo_sector_v1",
        recurrent_contract_version="rl_recurrent_partition_v1",
        feature_version="feature_v1",
        environment_version="single_symbol_env_v1",
        ppo_configuration={"n_steps": 512, "batch_size": 64, "gamma": 0.99},
        recurrent_architecture={"lstm_hidden_size": 64, "n_lstm_layers": 1},
        reward_configuration={**REWARD_METADATA, "reward_version": "sector_reward_v2"},
        action_validity_configuration=ACTION_METADATA,
        git_commit="c" * 40,
        reward_version="sector_reward_v2",
    )
    same_config_new_commit = build_predeclared_fair_experiment_spec(
        schedule,
        taxonomy_version="psx_sector_taxonomy_v1",
        trainer_version="recurrent_ppo_sector_v1",
        recurrent_contract_version="rl_recurrent_partition_v1",
        feature_version="feature_v1",
        environment_version="single_symbol_env_v1",
        ppo_configuration={"n_steps": 512, "batch_size": 64, "gamma": 0.99},
        recurrent_architecture={"lstm_hidden_size": 64, "n_lstm_layers": 1},
        reward_configuration=REWARD_METADATA,
        action_validity_configuration=ACTION_METADATA,
        git_commit="c" * 40,
    )
    assert base["experiment_spec_hash"] != changed_reward["experiment_spec_hash"]
    assert base["experiment_spec_hash"] == same_config_new_commit["experiment_spec_hash"]
    assert (
        base["reproducibility_fingerprint"]["git_commit"]
        != same_config_new_commit["reproducibility_fingerprint"]["git_commit"]
    )


def test_schedule_digest_invariant_rejects_tampering() -> None:
    schedule = build_balanced_window_schedule(
        _frames(("MCB", "UBL")), universe_hash=UNIVERSE_HASH, rounds=2
    )
    with pytest.raises(BalancedWindowScheduleError, match="digest is stale"):
        replace(
            schedule,
            records=(replace(schedule.records[0], start_date="1999-01-01"),)
            + schedule.records[1:],
        )


def test_checked_in_frozen_experiment_spec_is_portable_and_self_consistent() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "config"
        / "sector_recurrent_fair_experiment_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"experiment_spec_hash", "reproducibility_fingerprint"}
    }
    assert canonical_payload_hash(identity) == payload["experiment_spec_hash"]
    assert payload["execution_status"] == "specification_only_not_executed"
    assert payload["seed_policy"]["experiment_model_seeds"] == [42, 43, 44]
    assert payload["seed_policy"]["ppo_seed_source"] == "experiment_model_seeds"
    assert payload["expected_total_scheduled_transitions"] == 194_560
    assert (
        payload["test_sealing_rule"]
        == "TEST_metadata_only_never_loaded_or_evaluated"
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert "/Users/" not in serialized
    assert "generated_at" not in serialized
