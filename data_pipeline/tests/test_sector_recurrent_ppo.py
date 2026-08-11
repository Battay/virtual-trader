"""Offline safety coverage for Commercial Banks sector RecurrentPPO."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
    SAVED_MODELS_DIR,
)
from reinforcement_learning.environments import SingleSymbolEnvConfig
from reinforcement_learning.environments.config import DEFAULT_OBSERVATION_FEATURES
from reinforcement_learning.environments.sector_training_env import (
    EqualSymbolEpisodeSampler,
    SectorTrainingEnv,
)
from reinforcement_learning.evaluation.sector_recurrent_evaluator import (
    aggregate_sector_validation,
    evaluate_sector_recurrent_on_validation,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.model_management.sector_recurrent_persistence import (
    verify_temporary_sector_round_trip,
)
from reinforcement_learning.recurrent_data_contract import load_recurrent_partition
from reinforcement_learning.sector_universe import (
    build_leave_one_out_sector_manifest,
    deterministic_universe_hash,
)
from reinforcement_learning.training.recurrent_config import RecurrentPPOConfig
from reinforcement_learning.training.sector_recurrent_config import (
    SECTOR_RECURRENT_TRAINER_VERSION,
    SectorRecurrentPPOConfig,
)
from reinforcement_learning.training.sector_recurrent_trainer import (
    COMMERCIAL_BANKS_MANIFEST_PATH,
    SectorRecurrentTrainerError,
    load_sector_training_universe,
    train_sector_recurrent_ppo,
)


EXPECTED_BANKS = (
    "ABL", "AKBL", "BAFL", "BAHL", "BIPL", "BML", "BOK", "BOP", "FABL",
    "HBL", "HMB", "JSBL", "MCB", "MEBL", "NBP", "SBL", "SCBPL", "SNBL", "UBL",
)


def _small_config() -> SectorRecurrentPPOConfig:
    manifest = json.loads(COMMERCIAL_BANKS_MANIFEST_PATH.read_text(encoding="utf-8"))
    config = SectorRecurrentPPOConfig.from_manifest(manifest, total_timesteps=32)
    return replace(
        config,
        ppo=RecurrentPPOConfig(
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            total_timesteps=32,
            seed=42,
            device="cpu",
        ),
    )


@pytest.fixture(scope="module")
def sector_training():
    trace: list[tuple[str, str]] = []

    def traced(symbol: str, partition: str, **kwargs):
        trace.append((symbol, partition))
        if partition != "train":
            raise AssertionError("optimizer attempted a non-TRAIN partition")
        return load_recurrent_partition(symbol, partition, **kwargs)

    result = train_sector_recurrent_ppo(
        config=_small_config(),
        partition_loader=traced,
        smoke_test=True,
    )
    assert result.succeeded, result.error
    return result, tuple(trace)


@pytest.fixture(scope="module")
def sector_validation(sector_training):
    training, _ = sector_training
    trace: list[tuple[str, str]] = []

    def traced(symbol: str, partition: str, **kwargs):
        trace.append((symbol, partition))
        if partition != "validation":
            raise AssertionError("sector evaluator attempted a non-validation partition")
        return load_recurrent_partition(symbol, partition, **kwargs)

    result = evaluate_sector_recurrent_on_validation(
        training,
        partition_loader=traced,
    )
    return result, tuple(trace)


def _two_feature_frame(symbol: str, rows: int = 6) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2026-01-01", periods=rows),
            "open": 100 + index,
            "high": 102 + index,
            "low": 99 + index,
            "close": 101 + index,
            "volume": 1_000 + index,
            "simple_return": index / 100,
            "rsi_14": index / 10,
        }
    )


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_commercial_banks_manifest_audits_every_constituent() -> None:
    universe = load_sector_training_universe()
    assert universe.symbols == EXPECTED_BANKS
    assert universe.universe_hash == "589485c8adfe6170a6c2391687202ac3a287de9eb30737a2cb1f57a34f111e5b"
    assert universe.total_train_rows == 31_157
    assert len(universe.train_data) == 19
    assert set(universe.train_rows) == set(EXPECTED_BANKS)
    assert universe.observation_shape == (17,)
    assert universe.observation_features == DEFAULT_OBSERVATION_FEATURES
    for symbol, frame in universe.train_data.items():
        assert frame["symbol"].astype(str).eq(symbol).all()
        assert np.isfinite(frame.loc[:, DEFAULT_OBSERVATION_FEATURES].to_numpy(float)).all()
        assert (frame.loc[:, ["open", "high", "low", "close"]] > 0).all().all()


def test_manifest_incompatibility_fails_closed_without_dropping(tmp_path: Path) -> None:
    payload = json.loads(COMMERCIAL_BANKS_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["constituents"][3]["history_class"] = "COLD_START"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SectorRecurrentTrainerError, match="BAHL.*history class"):
        load_sector_training_universe(path)


def test_duplicate_constituent_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(COMMERCIAL_BANKS_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["constituents"][1] = payload["constituents"][0]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SectorRecurrentTrainerError, match="symbols are invalid"):
        load_sector_training_universe(path)


def test_sector_config_reuses_exact_6c_lstm_architecture() -> None:
    config = _small_config()
    assert config.trainer_version == SECTOR_RECURRENT_TRAINER_VERSION
    assert config.sampling_strategy == "equal_symbol_episode_sampling_v1"
    assert config.normalization_scope == "symbol"
    assert config.ppo.lstm_hidden_size == 64
    assert config.ppo.n_lstm_layers == 1
    assert config.ppo.shared_lstm is False
    assert config.ppo.enable_critic_lstm is True
    assert config.ppo.net_arch == (64,)
    assert config.ppo.activation_fn == "Tanh"
    assert config.ppo.ortho_init is True


def test_equal_symbol_sampler_is_deterministic_seeded_and_exhaustive() -> None:
    universe_hash = "a" * 64
    one = EqualSymbolEpisodeSampler(("A", "B", "C", "D"), seed=42, universe_hash=universe_hash)
    two = EqualSymbolEpisodeSampler(("A", "B", "C", "D"), seed=42, universe_hash=universe_hash)
    other = EqualSymbolEpisodeSampler(("A", "B", "C", "D"), seed=43, universe_hash=universe_hash)
    first = tuple(one.next_symbol() for _ in range(8))
    assert first == tuple(two.next_symbol() for _ in range(8))
    assert first != tuple(other.next_symbol() for _ in range(8))
    assert set(first[:4]) == {"A", "B", "C", "D"}
    assert set(first[4:]) == {"A", "B", "C", "D"}
    assert one.sequence_digest == two.sequence_digest


def test_sector_environment_preserves_rollout_continuity_then_resets_portfolio() -> None:
    config = SingleSymbolEnvConfig(feature_columns=("simple_return", "rsi_14"))
    env = SectorTrainingEnv(
        {"A": _two_feature_frame("A"), "B": _two_feature_frame("B")},
        universe_hash="b" * 64,
        seed=42,
        config=config,
    )
    try:
        _, first_info = env.reset(seed=42)
        first_symbol = first_info["sector_symbol"]
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(1)
            assert not terminated and not truncated
        assert len(env.reset_snapshots) == 1  # a rollout boundary cannot reset a symbol
        while not terminated:
            _, _, terminated, truncated, info = env.step(0)
            assert not truncated
        assert info["sector_episode_end_reason"] == "natural_train_partition_end"
        _, next_info = env.reset()
        assert next_info["episode_start"] is True
        assert next_info["sector_symbol"] != first_symbol
        snapshot = env.reset_snapshots[-1]
        assert snapshot["cash"] == config.initial_cash
        assert snapshot["shares_held"] == 0
        assert snapshot["realized_profit_loss"] == 0
        assert snapshot["drawdown"] == 0
    finally:
        env.close()


def test_training_uses_train_only_and_records_exposure(sector_training) -> None:
    result, trace = sector_training
    assert len(trace) == 19
    assert {partition for _, partition in trace} == {"train"}
    assert tuple(symbol for symbol, _ in trace) == EXPECTED_BANKS
    assert result.actual_timesteps == 32
    assert sum(result.timesteps_by_symbol.values()) == 32
    assert result.sampling_sequence
    assert len(result.sampling_sequence_digest) == 64
    assert result.total_episodes_started == 1
    assert result.total_episodes_completed == 0
    assert result.termination_reasons == {}
    assert result.observation_shape == (17,)
    assert "symbol" not in result.observation_features
    assert result.training_diagnostics is not None
    assert result.first_episode_start is True
    assert result.symbol_boundary_episode_start_verified
    assert result.portfolio_reset_verified
    assert result.rollout_continuity_verified


def test_per_symbol_scalers_and_train_sources_are_distinct() -> None:
    manifest = json.loads(COMMERCIAL_BANKS_MANIFEST_PATH.read_text(encoding="utf-8"))
    scaler_paths = [item["scaler_path"] for item in manifest["constituents"]]
    train_paths = [item["train_rl_path"] for item in manifest["constituents"]]
    assert len(set(scaler_paths)) == len(EXPECTED_BANKS)
    assert len(set(train_paths)) == len(EXPECTED_BANKS)
    assert all("/train_rl.csv" in path for path in train_paths)
    assert all("rl_observation_scaler.joblib" in path for path in scaler_paths)


def test_validation_is_separate_resets_state_and_preserves_model(
    sector_training, sector_validation
) -> None:
    training, _ = sector_training
    result, trace = sector_validation
    assert len(trace) == 19
    assert {partition for _, partition in trace} == {"validation"}
    assert len(result.symbol_results) == 19
    assert result.failures == {}
    assert result.model_parameters_unchanged
    assert training.model.num_timesteps == training.actual_timesteps
    assert all(item.first_episode_start for item in result.symbol_results)
    assert all(item.portfolio_reset_verified for item in result.symbol_results)
    assert all(item.model_parameters_unchanged for item in result.symbol_results)
    assert all(item.initial_capital == result.initial_capital_per_symbol for item in result.symbol_results)
    assert result.test_evaluated is False


def test_aggregate_metrics_are_deterministic_and_show_distribution(sector_validation) -> None:
    validation, _ = sector_validation
    first, collapse = aggregate_sector_validation(validation.symbol_results, {})
    second, repeated = aggregate_sector_validation(validation.symbol_results, {})
    assert first == second
    assert collapse == repeated
    distribution = first["ppo_total_return"]
    assert set(distribution) == {"mean", "median", "minimum", "maximum", "q25", "q75"}
    assert first["symbols_evaluated"] == 19
    assert (
        first["symbols_beating_buy_and_hold_return"]
        + first["symbols_losing_to_buy_and_hold_return"]
        + first["symbols_tied_buy_and_hold_return"]
        == 19
    )
    assert set(collapse["action_percentages"]) == {"Hold", "Buy", "Sell"}


def test_collapse_diagnostics_flag_sell_dominance_and_mostly_cash(
    sector_validation,
) -> None:
    validation, _ = sector_validation
    synthetic = []
    for index, item in enumerate(validation.symbol_results):
        metrics = dict(item.ppo.metrics)
        metrics["exposure_percentage"] = 0.0
        synthetic.append(
            replace(
                item,
                ppo=replace(item.ppo, metrics=metrics),
                action_counts={"Hold": 4, "Buy": 4, "Sell": 92},
                action_pattern_digest=f"unique-{index}",
            )
        )
    _, collapse = aggregate_sector_validation(synthetic, {})
    assert collapse["obvious_policy_collapse_flag"] is True
    assert any("at least 90%" in warning for warning in collapse["warnings"])
    assert any("median exposure" in warning for warning in collapse["warnings"])


def test_leave_one_out_excludes_target_from_training_and_scaler_identity() -> None:
    root = COMMERCIAL_BANKS_MANIFEST_PATH.parent.parent
    result = build_leave_one_out_sector_manifest(
        "MCB",
        standard_manifest_path=COMMERCIAL_BANKS_MANIFEST_PATH,
        current_verified_path=root / "current_verified_symbols.csv",
        generated_at="2030-01-01T00:00:00+00:00",
    )
    assert result["experiment_mode"]["target_symbol"] == "MCB"
    assert result["experiment_mode"]["target_excluded_from_pretraining"] is True
    assert "MCB" not in result["experiment_mode"]["pretraining_constituent_symbols"]
    assert "MCB" not in result["normalization"]["normalization_contributor_symbols"]
    assert result["normalization"]["target_contributes_to_pretraining_normalization"] is False
    assert result["universe_hash"] == deterministic_universe_hash(result["deterministic_identity"])
    assert result["universe_hash"] != json.loads(
        COMMERCIAL_BANKS_MANIFEST_PATH.read_text(encoding="utf-8")
    )["universe_hash"]


def test_temporary_sector_round_trip_never_touches_production(
    tmp_path: Path, sector_training, sector_validation
) -> None:
    training, _ = sector_training
    validation, _ = sector_validation
    registry_before = MODEL_REGISTRY_PATH.read_bytes()
    model_trees_before = {
        str(path): _tree(path) for path in (SAVED_MODELS_DIR, MODELS_DATA_DIR)
    }
    root = tmp_path / "sector_research_bundle"
    result = verify_temporary_sector_round_trip(
        training, validation, temporary_root=root
    )
    assert result.deterministic_action_match
    assert result.recurrent_state_match
    assert result.metadata_integrity_verified
    assert result.registry_touched is False
    assert MODEL_REGISTRY_PATH.read_bytes() == registry_before
    assert {
        str(path): _tree(path) for path in (SAVED_MODELS_DIR, MODELS_DATA_DIR)
    } == model_trees_before


def test_production_result_contains_reproducibility_fingerprint(sector_training) -> None:
    result, _ = sector_training
    fingerprint = result.reproducibility_fingerprint
    assert fingerprint["sector_universe_hash"] == result.sector_universe_hash
    assert fingerprint["trainer_version"] == SECTOR_RECURRENT_TRAINER_VERSION
    assert fingerprint["requested_timesteps"] == 32
    assert fingerprint["actual_timesteps"] == 32
    assert fingerprint["dependencies"]["sb3_contrib"] == "2.9.0"
    assert fingerprint["constituent_symbols"] == list(EXPECTED_BANKS)
    assert set(fingerprint["timesteps_by_symbol"]) == set(EXPECTED_BANKS)
