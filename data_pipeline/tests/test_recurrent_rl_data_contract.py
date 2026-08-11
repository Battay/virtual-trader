"""Offline regression tests for the recurrent RL metadata contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering.schemas import FEATURE_COLUMNS, FEATURE_VERSION
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.data_contract import (
    DEFAULT_OBSERVATION_FEATURES,
    EXECUTION_ACCOUNTING_COLUMNS,
    RL_PARTITION_SCHEMA_VERSION,
    load_rl_partition,
)
from reinforcement_learning.environments import (
    ENVIRONMENT_VERSION,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.history_policy import HistoryClass
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RECURRENT_EPISODE_BOUNDARIES_FILENAME,
    RECURRENT_EPISODE_SCHEMA_VERSION,
    RECURRENT_LOADABLE_PARTITIONS,
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    RecurrentDataContractError,
    build_recurrent_artifacts,
    load_recurrent_contract_metadata,
    load_recurrent_partition,
    persist_recurrent_contract,
    recurrent_eligibility,
    recurrent_episode_start_mask,
)


def _processed(rows: int = 180, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2020-01-01", periods=rows),
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


def _persist_source(
    tmp_path: Path,
    *,
    data: pd.DataFrame | None = None,
    symbol: str = "MCB",
):
    source = data if data is not None else _processed(symbol=symbol)
    split = chronological_split(source, scope="symbol")
    result = persist_split_artifacts(
        split,
        tmp_path / "symbols" / symbol,
    )
    return split, result


def _persist_recurrent(
    tmp_path: Path,
    *,
    symbol: str = "MCB",
    data: pd.DataFrame | None = None,
):
    split, source_result = _persist_source(
        tmp_path,
        data=data,
        symbol=symbol,
    )
    result = persist_recurrent_contract(
        symbol,
        company=f"{symbol} Limited",
        sector="Commercial Banks",
        sector_verified=True,
        usable_observations=len(split.train) + len(split.validation) + len(split.test),
        splits_dir=tmp_path,
        source_snapshot={"fixture": "offline"},
    )
    return split, source_result, result


def test_recurrent_contract_is_separate_and_preserves_mlp_contract(
    tmp_path: Path,
) -> None:
    split, source, recurrent = _persist_recurrent(tmp_path)

    contract = json.loads(recurrent.contract_path.read_text(encoding="utf-8"))
    boundaries = json.loads(recurrent.boundaries_path.read_text(encoding="utf-8"))
    assert contract["artifact_schema_version"] == RL_RECURRENT_PARTITION_SCHEMA_VERSION
    assert contract["source_rl_contract_version"] == RL_PARTITION_SCHEMA_VERSION
    assert boundaries["artifact_schema_version"] == RECURRENT_EPISODE_SCHEMA_VERSION
    assert recurrent.contract_path.parent.name == "recurrent"
    assert recurrent.boundaries_path.name == RECURRENT_EPISODE_BOUNDARIES_FILENAME
    assert not (recurrent.contract_path.parent / "train_recurrent.csv").exists()

    old_contract = json.loads(source.rl_artifacts.contract_path.read_text())
    assert old_contract["artifact_schema_version"] == RL_PARTITION_SCHEMA_VERSION
    old_loaded = load_rl_partition("MCB", "train", splits_dir=tmp_path)
    assert len(old_loaded.data) == len(split.train)


def test_recurrent_loader_preserves_observation_order_and_real_execution_ohlcv(
    tmp_path: Path,
) -> None:
    split, _, _ = _persist_recurrent(tmp_path)
    loaded = load_recurrent_partition("MCB", "train", splits_dir=tmp_path)

    assert loaded.metadata.observation_features == DEFAULT_OBSERVATION_FEATURES
    assert loaded.metadata.observation_shape == (17,)
    assert loaded.metadata.execution_columns == EXECUTION_ACCOUNTING_COLUMNS
    raw = split.train.reset_index(drop=True)
    for column in EXECUTION_ACCOUNTING_COLUMNS:
        np.testing.assert_array_equal(loaded.data[column], raw[column])
    assert not np.array_equal(
        loaded.data["simple_return"].to_numpy(),
        raw["simple_return"].to_numpy(),
    )


def test_scaler_provenance_and_sequence_defaults_are_explicit(tmp_path: Path) -> None:
    _, source, recurrent = _persist_recurrent(tmp_path)
    contract = json.loads(recurrent.contract_path.read_text(encoding="utf-8"))
    metadata = load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)

    assert contract["normalization"]["normalization_scope"] == "symbol"
    assert contract["normalization"]["fit_partition"] == "train"
    assert contract["normalization"]["source_scaler_path"] == (
        "../rl_observation_scaler.joblib"
    )
    assert contract["normalization"]["source_scaler_sha256"] == sha256_file(
        source.rl_artifacts.scaler_path
    )
    assert contract["normalization"][
        "source_scaler_metadata_sha256"
    ] == sha256_file(source.rl_artifacts.scaler_path.with_suffix(".json"))
    assert source.rl_artifacts.scaler_path.is_file()
    assert metadata.scaler_fit_partition == "train"
    assert metadata.normalization_scope == "symbol"
    assert metadata.sequence_length is None
    assert metadata.burn_in_length is None
    assert metadata.episode_length is None
    assert metadata.episode_strategy == "full_partition"
    assert metadata.minimum_sequence_rows == 2


def test_episode_start_mask_resets_at_first_symbol_partition_and_window_boundaries() -> None:
    mask = recurrent_episode_start_mask(
        ["MCB", "MCB", "MCB", "OGDC", "OGDC", "OGDC"],
        ["train", "train", "train", "train", "validation", "validation"],
        explicit_window_starts=[False, False, True, False, False, False],
    )

    np.testing.assert_array_equal(
        mask,
        np.asarray([True, False, True, True, True, False]),
    )


def test_episode_start_mask_has_no_internal_resets_for_continuous_episode() -> None:
    mask = recurrent_episode_start_mask(["MCB"] * 8, ["train"] * 8)
    np.testing.assert_array_equal(
        mask,
        np.asarray([True, False, False, False, False, False, False, False]),
    )


def test_train_and_validation_are_distinct_complete_episodes(tmp_path: Path) -> None:
    _, _, _ = _persist_recurrent(tmp_path)
    train = load_recurrent_partition("MCB", "train", splits_dir=tmp_path)
    validation = load_recurrent_partition("MCB", "validation", splits_dir=tmp_path)

    assert train.episode_start[0]
    assert validation.episode_start[0]
    assert train.episode_start.sum() == 1
    assert validation.episode_start.sum() == 1
    assert train.episode_boundaries[0].end == train.metadata.train.end
    assert validation.episode_boundaries[0].start == validation.metadata.validation.start
    assert pd.Timestamp(train.metadata.train.end) < pd.Timestamp(
        validation.metadata.validation.start
    )
    assert pd.Timestamp(validation.metadata.validation.end) < pd.Timestamp(
        validation.metadata.test.start
    )


@pytest.mark.parametrize(
    ("partition", "field", "replacement_from"),
    (
        ("validation", "start", ("train", "end")),
        ("test", "start", ("validation", "end")),
    ),
)
def test_recurrent_contract_rejects_partition_boundary_crossing(
    tmp_path: Path,
    partition: str,
    field: str,
    replacement_from: tuple[str, str],
) -> None:
    _, _, result = _persist_recurrent(tmp_path)
    contract = json.loads(result.contract_path.read_text(encoding="utf-8"))
    source_partition, source_field = replacement_from
    contract["partitions"][partition][field] = contract["partitions"][
        source_partition
    ][source_field]
    result.contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(RecurrentDataContractError):
        load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)


def test_test_frame_is_never_loaded_by_build_metadata_or_supported_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _persist_source(tmp_path)
    original_read_csv = pd.read_csv
    reads: list[str] = []

    def guarded_read_csv(path: object, *args: object, **kwargs: object):
        name = Path(path).name
        reads.append(name)
        if name in {"test.csv", "test_rl.csv"}:
            raise AssertionError("sealed TEST frame was read")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    persist_recurrent_contract(
        "MCB",
        company="MCB Limited",
        sector="Commercial Banks",
        sector_verified=True,
        usable_observations=180,
        splits_dir=tmp_path,
    )
    metadata = load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)
    loaded = load_recurrent_partition("MCB", "validation", splits_dir=tmp_path)

    assert metadata.test.rows > 0
    assert loaded.partition == "validation"
    assert "test.csv" not in reads
    assert "test_rl.csv" not in reads


def test_test_partition_is_rejected_before_canonical_loader_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _persist_recurrent(tmp_path)

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("canonical partition loader must not be reached")

    monkeypatch.setattr(
        "reinforcement_learning.recurrent_data_contract.load_rl_partition",
        forbidden_loader,
    )
    with pytest.raises(RecurrentDataContractError, match="TEST is sealed"):
        load_recurrent_partition("MCB", "test", splits_dir=tmp_path)
    assert RECURRENT_LOADABLE_PARTITIONS == ("train", "validation")


def test_future_validation_values_cannot_change_train_recurrent_sequence(
    tmp_path: Path,
) -> None:
    base = _processed()
    changed = base.copy(deep=True)
    changed.loc[140:, list(DEFAULT_OBSERVATION_FEATURES)] += 1_000_000

    _persist_recurrent(tmp_path / "a", data=base)
    _persist_recurrent(tmp_path / "b", data=changed)
    train_a = load_recurrent_partition("MCB", "train", splits_dir=tmp_path / "a")
    train_b = load_recurrent_partition("MCB", "train", splits_dir=tmp_path / "b")

    pd.testing.assert_frame_equal(train_a.data, train_b.data)
    np.testing.assert_array_equal(train_a.episode_start, train_b.episode_start)
    assert train_a.data["date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("rows", "history_class", "artifact", "independent", "transfer"),
    (
        (126, HistoryClass.MATURE, True, True, True),
        (100, HistoryClass.COLD_START, False, False, True),
        (99, HistoryClass.INSUFFICIENT, False, False, False),
    ),
)
def test_history_class_recurrent_eligibility(
    rows: int,
    history_class: HistoryClass,
    artifact: bool,
    independent: bool,
    transfer: bool,
) -> None:
    result = recurrent_eligibility(rows)
    assert result.history_class is history_class
    assert result.recurrent_artifact_eligible is artifact
    assert result.independent_recurrent_ready is independent
    assert result.transfer_fine_tune_eligible is transfer


@pytest.mark.parametrize("rows", (100, 99))
def test_non_mature_history_is_rejected_for_recurrent_artifact(
    tmp_path: Path,
    rows: int,
) -> None:
    _persist_source(tmp_path)
    with pytest.raises(RecurrentDataContractError):
        persist_recurrent_contract(
            "MCB",
            company="MCB Limited",
            sector="Commercial Banks",
            sector_verified=True,
            usable_observations=rows,
            splits_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("artifact_schema_version", "rl_recurrent_partition_v0", "contract version"),
        ("feature_version", "stale_features", "feature version is stale"),
        ("environment_version", "stale_environment", "environment version is stale"),
    ),
)
def test_stale_or_incompatible_recurrent_contract_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: str,
    error: str,
) -> None:
    _, _, result = _persist_recurrent(tmp_path)
    contract = json.loads(result.contract_path.read_text(encoding="utf-8"))
    contract[field] = replacement
    result.contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(RecurrentDataContractError, match=error):
        load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)


def test_stale_source_partition_or_episode_boundary_hash_is_rejected(
    tmp_path: Path,
) -> None:
    _, _, result = _persist_recurrent(tmp_path)
    train_path = tmp_path / "symbols" / "MCB" / "train_rl.csv"
    train_path.write_bytes(train_path.read_bytes() + b"\n")
    with pytest.raises(RecurrentDataContractError, match="source hash is stale"):
        load_recurrent_partition("MCB", "train", splits_dir=tmp_path)

    _persist_recurrent(tmp_path)
    result.boundaries_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RecurrentDataContractError, match="boundary artifact hash"):
        load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)


def test_build_summary_generates_only_valid_mature_symbols(tmp_path: Path) -> None:
    _persist_source(tmp_path, symbol="MCB")
    registry_snapshot = tmp_path / "company_registry.csv"
    registry_snapshot.write_text("symbol,sector\nMCB,Commercial Banks\n", encoding="utf-8")
    status = pd.DataFrame(
        {
            "symbol": pd.Series(["MCB", "MISSING", "COLD", "NEW"], dtype="string"),
            "company_name": ["MCB Limited", "Missing Limited", "Cold Ltd", "New Ltd"],
            "sector": ["Commercial Banks", "Oil & Gas", "Textile", "Technology"],
            "security_type": ["ordinary_equity"] * 4,
            "is_active": [True] * 4,
            "usable_rows": [180, 126, 100, 99],
        }
    )
    original = status.copy(deep=True)

    summary = build_recurrent_artifacts(
        status,
        splits_dir=tmp_path,
        sector_source_path=registry_snapshot,
    )

    assert summary.mature_symbols_inspected == 2
    assert summary.recurrent_compatible_symbols_generated == 1
    assert summary.cold_start_symbols == 1
    assert summary.insufficient_symbols == 1
    assert summary.failures == 1
    assert summary.artifact_files_written == 2
    assert [record.symbol for record in summary.records if record.generated] == ["MCB"]
    pd.testing.assert_frame_equal(status, original)


def test_recurrent_environment_api_remains_gymnasium_compatible(tmp_path: Path) -> None:
    _persist_recurrent(tmp_path)
    loaded = load_recurrent_partition("MCB", "train", splits_dir=tmp_path)
    source_before = loaded.data.copy(deep=True)
    env = SingleSymbolTradingEnv(loaded.data)

    observation_a, info_a = env.reset(seed=42)
    observation_b, info_b = env.reset(seed=42)
    np.testing.assert_array_equal(observation_a, observation_b)
    assert observation_a.dtype == np.float32
    assert observation_a.shape == (17,)
    assert info_a["environment_version"] == ENVIRONMENT_VERSION
    assert info_b["environment_version"] == ENVIRONMENT_VERSION
    transition = env.step(0)
    assert len(transition) == 5
    _, reward, terminated, truncated, _ = transition
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    history = env.get_history()
    assert not history.empty
    history.iloc[0, history.columns.get_loc("portfolio_value")] = -1
    assert env.get_history().iloc[0]["portfolio_value"] != -1
    env.close()
    pd.testing.assert_frame_equal(loaded.data, source_before)


def test_contract_metadata_is_complete_and_test_is_metadata_only(
    tmp_path: Path,
) -> None:
    split, _, result = _persist_recurrent(tmp_path)
    metadata = load_recurrent_contract_metadata("MCB", splits_dir=tmp_path)
    contract = json.loads(result.contract_path.read_text(encoding="utf-8"))

    assert metadata.recurrent_contract_version == RL_RECURRENT_PARTITION_SCHEMA_VERSION
    assert metadata.source_rl_contract_version == RL_PARTITION_SCHEMA_VERSION
    assert metadata.feature_version == FEATURE_VERSION
    assert metadata.environment_version == ENVIRONMENT_VERSION
    assert metadata.training_scope == "symbol"
    assert metadata.constituent_symbols == ("MCB",)
    assert metadata.universe_id == f"symbol:MCB:{metadata.train.end}"
    assert metadata.sector == "Commercial Banks"
    assert metadata.sector_verified
    assert metadata.history.history_class is HistoryClass.MATURE
    assert metadata.train.rows == len(split.train)
    assert metadata.validation.rows == len(split.validation)
    assert metadata.test.rows == len(split.test)
    assert metadata.test.sealed
    assert contract["test_sealing"]["evaluation_performed"] is False
    assert "source_rl_path" not in contract["partitions"]["test"]
