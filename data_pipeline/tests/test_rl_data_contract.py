"""Offline tests for real-price execution and scaled RL observations."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering.schemas import FEATURE_COLUMNS, FEATURE_VERSION
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.data_contract import (
    DEFAULT_OBSERVATION_FEATURES,
    EXECUTION_ACCOUNTING_COLUMNS,
    IDENTITY_TIME_COLUMNS,
    RLDataContractError,
    RL_PARTITION_SCHEMA_VERSION,
    load_rl_partition,
    scaled_observation_column,
)
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)


def _processed(rows: int = 20, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2025-01-01", periods=rows),
        "open": 100.0 + index,
        "high": 102.0 + index,
        "low": 99.0 + index,
        "close": 101.0 + index,
        "volume": 1_000.0 + 10 * index,
    }
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = (feature_index + 1) * 0.1 + index
    return pd.DataFrame(data)


def _persist(tmp_path: Path, data: pd.DataFrame | None = None):
    split = chronological_split(data if data is not None else _processed(), scope="symbol")
    result = persist_split_artifacts(split, tmp_path / "symbols" / "MCB")
    return split, result


def test_rl_artifacts_preserve_execution_and_align_scaled_observations(
    tmp_path: Path,
) -> None:
    split, result = _persist(tmp_path)
    assert result.metadata["rl_artifact_schema_version"] == RL_PARTITION_SCHEMA_VERSION
    assert result.rl_artifacts.contract["scaler_fit_partition"] == "train"
    assert tuple(result.rl_artifacts.contract["observation_features"]) == (
        DEFAULT_OBSERVATION_FEATURES
    )

    for name, source in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        artifact = pd.read_csv(
            result.rl_artifacts.partition_paths[name],
            dtype={"symbol": "string"},
            parse_dates=["date"],
        )
        assert len(artifact) == len(source)
        assert not artifact.duplicated(list(IDENTITY_TIME_COLUMNS)).any()
        pd.testing.assert_frame_equal(
            artifact.loc[:, list(IDENTITY_TIME_COLUMNS)].reset_index(drop=True),
            source.loc[:, list(IDENTITY_TIME_COLUMNS)].reset_index(drop=True),
            check_dtype=False,
        )
        for column in EXECUTION_ACCOUNTING_COLUMNS:
            np.testing.assert_array_equal(artifact[column], source[column])

    train = split.train
    validation_artifact = pd.read_csv(result.rl_artifacts.partition_paths["validation"])
    mean = train.loc[:, list(DEFAULT_OBSERVATION_FEATURES)].mean()
    scale = train.loc[:, list(DEFAULT_OBSERVATION_FEATURES)].std(ddof=0).replace(0, 1)
    expected = (
        split.validation.iloc[0][list(DEFAULT_OBSERVATION_FEATURES)] - mean
    ) / scale
    actual = validation_artifact.iloc[0][
        [scaled_observation_column(name) for name in DEFAULT_OBSERVATION_FEATURES]
    ]
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_loader_uses_scaled_observations_but_environment_accounts_in_real_prices(
    tmp_path: Path,
) -> None:
    split, result = _persist(tmp_path)
    loaded = load_rl_partition("MCB", "train", splits_dir=tmp_path)
    raw = split.train.reset_index(drop=True)
    artifact = pd.read_csv(result.rl_artifacts.partition_paths["train"])
    config = SingleSymbolEnvConfig(
        initial_cash=10_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        drawdown_penalty_weight=0.0,
    )
    env = SingleSymbolTradingEnv(loaded.data, config)
    observation, _ = env.reset(seed=42)

    expected_observation = artifact.iloc[0][
        [scaled_observation_column(name) for name in DEFAULT_OBSERVATION_FEATURES]
    ].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(
        observation[: len(DEFAULT_OBSERVATION_FEATURES)], expected_observation
    )

    _, _, _, _, info = env.step(1)
    raw_open = float(raw.iloc[1]["open"])
    raw_close = float(raw.iloc[1]["close"])
    execution_price = raw_open * (1 + config.slippage_rate)
    shares = math.floor(
        config.initial_cash / (execution_price * (1 + config.commission_rate))
    )
    commission = shares * execution_price * config.commission_rate
    slippage = shares * (execution_price - raw_open)
    expected_cash = config.initial_cash - shares * execution_price - commission

    assert info["execution_price"] == pytest.approx(execution_price)
    assert info["transaction_cost"] == pytest.approx(commission + slippage)
    assert info["portfolio_value"] == pytest.approx(expected_cash + shares * raw_close)
    assert loaded.data.iloc[1]["open"] == raw_open
    assert loaded.data.iloc[1]["close"] == raw_close


def test_validation_and_test_use_training_scaler_and_future_cannot_change_it(
    tmp_path: Path,
) -> None:
    base = _processed()
    split_a, result_a = _persist(tmp_path / "a", base)
    future_changed = base.copy()
    future_changed.loc[17:, list(DEFAULT_OBSERVATION_FEATURES)] += 1_000_000
    split_b, result_b = _persist(tmp_path / "b", future_changed)

    scaler_a = json.loads(
        result_a.rl_artifacts.scaler_path.with_suffix(".json").read_text()
    )
    scaler_b = json.loads(
        result_b.rl_artifacts.scaler_path.with_suffix(".json").read_text()
    )
    assert scaler_a["scaled_features"] == list(DEFAULT_OBSERVATION_FEATURES)
    assert scaler_a["training_mean"] == scaler_b["training_mean"]
    assert scaler_a["training_scale"] == scaler_b["training_scale"]
    assert split_a.metadata["training"] == split_b.metadata["training"]
    train_a = pd.read_csv(result_a.rl_artifacts.partition_paths["train"])
    train_b = pd.read_csv(result_b.rl_artifacts.partition_paths["train"])
    pd.testing.assert_frame_equal(train_a, train_b)


def test_stale_rl_artifact_version_fails_clearly(tmp_path: Path) -> None:
    _, result = _persist(tmp_path)
    contract_path = result.rl_artifacts.contract_path
    contract = json.loads(contract_path.read_text())
    contract["artifact_schema_version"] = "rl_partition_v0"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(RLDataContractError, match="Incompatible RL artifact schema"):
        load_rl_partition("MCB", "train", splits_dir=tmp_path)
