"""Offline safety tests for canonical TRAIN-only recurrent v2 artifacts."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.src.parquet_market_data import load_market_data
from reinforcement_learning.canonical_recurrent_artifacts import (
    CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION,
    CANONICAL_TRAIN_MINIMUM_USABLE_ROWS,
    RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION,
    build_canonical_recurrent_train_artifacts,
    load_canonical_recurrent_train_metadata,
    load_canonical_recurrent_train_partition,
    load_training_recurrent_contract_metadata,
)
from reinforcement_learning.environments import SingleSymbolTradingEnv
from reinforcement_learning.environments.config import DEFAULT_OBSERVATION_FEATURES
from reinforcement_learning.training.recurrent_config import RecurrentPPOConfig
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    build_training_run,
    discover_recurrent_training_universe,
)
from reinforcement_learning.training.recurrent_trainer import (
    train_recurrent_single_symbol,
)


MARKET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32()),
        pa.field("symbol", pa.string()),
        pa.field("ldcp", pa.float64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("change_percent", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)


def _market_rows(
    symbols: tuple[str, ...],
    *,
    periods: int = 260,
    future_outlier: bool = False,
) -> list[dict[str, object]]:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for index, market_date in enumerate(dates):
            close = 20.0 + symbol_index + index / 100.0
            if future_outlier and index == periods - 1:
                close = 20_000.0
            rows.append(
                {
                    "market_date": market_date.date(),
                    "symbol": symbol,
                    "ldcp": close - 0.05,
                    "open": close - 0.02,
                    "high": close + 0.10,
                    "low": close - 0.10,
                    "close": close,
                    "change": 0.05,
                    "change_percent": 0.25,
                    "volume": 10_000 + index,
                }
            )
    return rows


def _write_market(
    path: Path,
    symbols: tuple[str, ...],
    *,
    periods: int = 260,
    future_outlier: bool = False,
) -> Path:
    table = pa.Table.from_pandas(
        pd.DataFrame(
            _market_rows(
                symbols, periods=periods, future_outlier=future_outlier
            )
        ),
        schema=MARKET_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path)
    return path


def _identity(
    specifications: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol for symbol, _ in specifications],
            "company_name": [f"{symbol} Limited" for symbol, _ in specifications],
            "sector": ["TEST SECTOR"] * len(specifications),
            "security_type": [kind for _, kind in specifications],
            "source": ["https://dps.psx.com.pk/listings-table/main/nc"]
            * len(specifications),
            "snapshot_date": ["2026-08-02"] * len(specifications),
        }
    )


def _build(
    tmp_path: Path,
    *,
    symbols: tuple[str, ...] = ("ORDY",),
    specifications: tuple[tuple[str, str], ...] | None = None,
    periods: int = 260,
    artifacts_name: str = "artifacts",
    market_path: Path | None = None,
    market_loader=load_market_data,
):
    identity = _identity(
        specifications
        or tuple((symbol, "ordinary_equity") for symbol in symbols)
    )
    parquet = market_path or _write_market(
        tmp_path / f"{artifacts_name}.parquet", symbols, periods=periods
    )
    return build_canonical_recurrent_train_artifacts(
        symbols,
        identity=identity,
        identity_universe_hash="a" * 64,
        parquet_path=parquet,
        artifacts_dir=tmp_path / artifacts_name,
        market_loader=market_loader,
    )


def test_train_predicate_pushdown_scaler_fit_and_test_sealing(tmp_path: Path) -> None:
    parquet = _write_market(tmp_path / "market.parquet", ("ORDY",))
    loads: list[tuple[date, date]] = []

    def guarded_loader(path: Path, *, end_date: date, symbols: list[str]):
        frame = load_market_data(path, end_date=end_date, symbols=symbols)
        loads.append((end_date, max(frame["market_date"])))
        assert max(frame["market_date"]) <= end_date
        return frame

    summary = _build(
        tmp_path,
        market_path=parquet,
        artifacts_name="sealed",
        market_loader=guarded_loader,
    )
    result = summary.generated[0]
    contract = result.contract
    train = pd.read_csv(result.train_features_path)
    scaler = joblib.load(result.scaler_path)

    assert loads == [(loads[0][0], loads[0][0])]
    np.testing.assert_allclose(
        scaler.mean_, train.loc[:, DEFAULT_OBSERVATION_FEATURES].mean().to_numpy()
    )
    assert contract["normalization"]["fit_partition"] == "train"
    assert contract["partitions"]["validation"] == {
        "available": False,
        "frame_access": "not_built_not_loaded",
    }
    assert contract["partitions"]["test"]["frame_access"] == "sealed_not_built_not_loaded"
    assert not (result.artifact_directory / "validation.csv").exists()
    assert not (result.artifact_directory / "test.csv").exists()


def test_backward_features_ignore_post_train_value_change(tmp_path: Path) -> None:
    plain = _write_market(tmp_path / "plain.parquet", ("ORDY",))
    changed = _write_market(
        tmp_path / "changed.parquet", ("ORDY",), future_outlier=True
    )
    first = _build(
        tmp_path, market_path=plain, artifacts_name="plain_artifacts"
    ).generated[0]
    second = _build(
        tmp_path, market_path=changed, artifacts_name="changed_artifacts"
    ).generated[0]

    assert first.contract["partitions"]["train"]["feature_sha256"] == second.contract[
        "partitions"
    ]["train"]["feature_sha256"]
    assert first.contract["normalization"]["scaler_metadata_sha256"] == second.contract[
        "normalization"
    ]["scaler_metadata_sha256"]
    assert first.contract["source_snapshot"]["parquet_sha256"] != second.contract[
        "source_snapshot"
    ]["parquet_sha256"]


def test_derived_minimum_is_126_usable_train_rows(tmp_path: Path) -> None:
    exact = _build(
        tmp_path,
        periods=250,
        artifacts_name="exact",
    )
    short = _build(
        tmp_path,
        periods=249,
        artifacts_name="short",
    )

    assert CANONICAL_TRAIN_MINIMUM_USABLE_ROWS == 126
    assert exact.evidence[0].quality_usable_ohlcv_rows == 175
    assert exact.evidence[0].feature_warmup_loss == 49
    assert exact.evidence[0].final_usable_feature_rows == 126
    assert exact.recovered_symbols == ("ORDY",)
    assert short.evidence[0].final_usable_feature_rows == 125
    assert short.generated == ()


def test_ordinary_and_gem_allowed_but_other_instruments_fail_closed(
    tmp_path: Path,
) -> None:
    symbols = ("ETF", "GEM", "ORDY", "PREF", "RIGHT")
    summary = _build(
        tmp_path,
        symbols=symbols,
        specifications=(
            ("ETF", "etf"),
            ("GEM", "gem_equity"),
            ("ORDY", "ordinary_equity"),
            ("PREF", "preference_share"),
            ("RIGHT", "rights"),
        ),
    )

    assert summary.recovered_symbols == ("GEM", "ORDY")
    blocked = {row.symbol: row.blocker for row in summary.evidence if not row.recovery_valid}
    assert set(blocked) == {"ETF", "PREF", "RIGHT"}
    assert all("unsupported authoritative security_type" in value for value in blocked.values())


def test_version_isolation_deterministic_hashes_and_environment_compatibility(
    tmp_path: Path,
) -> None:
    parquet = _write_market(tmp_path / "market.parquet", ("ORDY",))
    v1 = tmp_path / "splits" / "symbols" / "ORDY" / "sentinel.txt"
    v1.parent.mkdir(parents=True)
    v1.write_text("v1-unchanged", encoding="utf-8")
    first = _build(
        tmp_path, market_path=parquet, artifacts_name="first"
    ).generated[0]
    second = _build(
        tmp_path, market_path=parquet, artifacts_name="second"
    ).generated[0]

    assert first.contract["artifact_schema_version"] == RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION
    assert first.contract["feature_version"] == CANONICAL_SYMBOL_FEATURE_SCHEMA_VERSION
    assert first.contract["deterministic_contract_identity"] == second.contract[
        "deterministic_contract_identity"
    ]
    assert first.contract["partitions"]["train"]["rl_sha256"] == second.contract[
        "partitions"
    ]["train"]["rl_sha256"]
    assert v1.read_text(encoding="utf-8") == "v1-unchanged"

    loaded = load_canonical_recurrent_train_partition(
        "ORDY", "train", artifacts_dir=tmp_path / "first"
    )
    env = SingleSymbolTradingEnv(loaded.data)
    try:
        observation, _ = env.reset(seed=42)
        assert observation.shape == (17,)
        assert env.observation_space.contains(observation)
    finally:
        env.close()
    assert loaded.episode_start[0]
    assert int(loaded.episode_start.sum()) == 1
    raw = pd.read_csv(first.train_features_path)
    for column in ("open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(loaded.data[column], raw[column])


def test_recovered_contract_enters_discovery_and_tiny_recurrent_smoke(
    tmp_path: Path,
) -> None:
    identity = _identity((("ORDY", "ordinary_equity"),))
    parquet = _write_market(tmp_path / "market.parquet", ("ORDY",))
    artifacts = tmp_path / "canonical"
    _build(
        tmp_path,
        market_path=parquet,
        artifacts_name="canonical",
    )

    def metadata_loader(symbol: str, **_: object):
        return load_training_recurrent_contract_metadata(
            symbol,
            splits_dir=tmp_path / "v1",
            canonical_artifacts_dir=artifacts,
        )

    discovery = discover_recurrent_training_universe(
        identity=identity,
        splits_dir=tmp_path / "v1",
        readiness_evidence_path=None,
        metadata_loader=metadata_loader,
    )

    assert discovery.identity_count == discovery.eligible_count == 1
    assert discovery.records.iloc[0]["category"] == ELIGIBLE_TRAINABLE
    assert discovery.records.iloc[0]["recurrent_contract_version"] == RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION
    assert bool(discovery.records.iloc[0]["validation_available"]) is False
    manifest, jobs = build_training_run(
        discovery,
        config=RecurrentPPOConfig(total_timesteps=8),
        validation_enabled=True,
        created_at="2026-08-27T00:00:00+00:00",
    )
    assert manifest.identity_count == len(jobs) == 1
    assert jobs[0].status == "QUEUED"
    assert jobs[0].validation_status == "not_available_train_only_contract"

    result = train_recurrent_single_symbol(
        "ORDY",
        config=RecurrentPPOConfig(
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            total_timesteps=8,
            device="cpu",
        ),
        splits_dir=tmp_path / "v1",
        canonical_artifacts_dir=artifacts,
        smoke_test=True,
    )

    assert result.succeeded, result.error
    assert result.recurrent_contract_version == RL_RECURRENT_TRAIN_PARTITION_SCHEMA_VERSION
    assert result.training_rows >= CANONICAL_TRAIN_MINIMUM_USABLE_ROWS
    assert result.observation_shape == (17,)
