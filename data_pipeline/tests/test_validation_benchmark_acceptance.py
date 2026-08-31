"""Offline tests for the frozen VALIDATION benchmark and acceptance policy."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.evaluation.validation_acceptance import (
    ACCEPTABLE_VALIDATION,
    INSUFFICIENT_VALIDATION_HISTORY,
    INVALID_VALIDATION,
    STRONG_VALIDATION,
    VALIDATION_ACCEPTANCE_FREEZE_SCHEMA,
    VALIDATION_ACCEPTANCE_POLICY_VERSION,
    WEAK_VALIDATION,
    ValidationAcceptanceError,
    apply_frozen_acceptance_policy,
    attach_validation_benchmarks,
    build_policy_freeze_payload,
    validate_policy_freeze,
)
from reinforcement_learning.evaluation.validation_benchmark import (
    VALIDATION_BUY_AND_HOLD_VERSION,
    ValidationBuyAndHoldResult,
    compute_validation_buy_and_hold,
)
from reinforcement_learning.recurrent_data_contract import (
    load_recurrent_partition,
    persist_recurrent_contract,
)
from reinforcement_learning.training.global_validation import (
    EXPORT_COLUMNS,
    VALID,
    validation_export_frame,
)


def _processed(rows: int = 180, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    opens = 100.0 + index
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2024-01-01", periods=rows),
        "open": opens,
        "high": opens + 2.0,
        "low": opens - 2.0,
        "close": opens + 1.0,
        "volume": 10_000.0 + index,
    }
    for position, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = index + position / 100.0
    return pd.DataFrame(data)


def _persist_recurrent(root: Path, symbol: str = "MCB") -> None:
    split = chronological_split(_processed(symbol=symbol), scope="symbol")
    persist_split_artifacts(split, root / "symbols" / symbol)
    persist_recurrent_contract(
        symbol,
        company=f"{symbol} Limited",
        sector="Commercial Banks",
        sector_verified=True,
        usable_observations=sum(len(value) for value in (split.train, split.validation, split.test)),
        splits_dir=root,
        source_snapshot={"fixture": "offline"},
    )


def _inventory_row(symbol: str = "AAA", *, rows: int = 150) -> dict[str, object]:
    return {
        "symbol": symbol,
        "run_type": "SELECTED",
        "run_id": "run-selected",
        "attempt": 1,
        "model_sha256": "m" * 64,
        "validation_artifact_sha256": "v" * 64,
        "validation_artifact_status": VALID,
        "validation_start": "2024-01-01",
        "validation_end": "2024-06-30",
        "validation_rows": rows,
        "validation_total_return": 0.20,
        "validation_sharpe": 1.0,
        "validation_sortino": 1.2,
        "validation_max_drawdown": 0.10,
        "validation_volatility": 0.20,
        "partition_contract_version": "rl_partition_v1",
        "recurrent_contract_version": "rl_recurrent_partition_v1",
        "feature_version": "feature-v1",
        "environment_version": "single_symbol_env_v1",
        "test_partition_loaded": False,
    }


def _benchmark_result(symbol: str = "AAA") -> ValidationBuyAndHoldResult:
    return ValidationBuyAndHoldResult(
        symbol=symbol,
        benchmark_contract_version=VALIDATION_BUY_AND_HOLD_VERSION,
        validation_start="2024-01-01",
        validation_end="2024-06-30",
        validation_rows=150,
        validation_transition_count=149,
        validation_membership_sha256="a" * 64,
        source_validation_artifact_path="symbols/AAA/validation_rl.csv",
        source_validation_artifact_sha256="b" * 64,
        recurrent_contract_version="rl_recurrent_partition_v1",
        environment_version="single_symbol_env_v1",
        feature_version="feature-v1",
        deterministic_seed=42,
        initial_cash=1_000_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        total_return=0.10,
        annualized_return=0.20,
        sharpe_ratio=0.50,
        sortino_ratio=0.70,
        maximum_drawdown=0.15,
        annualized_volatility=0.18,
        final_portfolio_value=1_100_000.0,
        number_of_trades=1,
        total_transaction_costs=1_500.0,
        metric_warnings=(),
        test_partition_loaded=False,
    )


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_buy_and_hold_uses_exact_validation_membership_and_is_read_only(
    tmp_path: Path,
) -> None:
    _persist_recurrent(tmp_path)
    expected = load_recurrent_partition("MCB", "validation", splits_dir=tmp_path)
    before = _hash_tree(tmp_path)
    calls: list[str] = []

    def loader(symbol: str, partition: str):
        calls.append(partition)
        assert partition == "validation"
        return load_recurrent_partition(symbol, partition, splits_dir=tmp_path)

    result = compute_validation_buy_and_hold("MCB", partition_loader=loader)

    assert calls == ["validation"]
    assert result.validation_rows == len(expected.data)
    assert result.validation_start == expected.metadata.validation.start
    assert result.validation_end == expected.metadata.validation.end
    assert result.validation_transition_count == len(expected.data) - 1
    assert result.number_of_trades == 1
    assert result.final_portfolio_value == pytest.approx(
        1_000_000.0 * (1.0 + float(result.total_return))
    )
    assert result.test_partition_loaded is False
    assert _hash_tree(tmp_path) == before


def test_benchmark_rejects_non_validation_loader(tmp_path: Path) -> None:
    _persist_recurrent(tmp_path)

    def loader(symbol: str, partition: str):
        del partition
        loaded = load_recurrent_partition(symbol, "train", splits_dir=tmp_path)
        return replace(loaded, partition="test")

    with pytest.raises(RuntimeError, match="non-VALIDATION"):
        compute_validation_buy_and_hold("MCB", partition_loader=loader)


def test_relative_metrics_and_policy_categories_are_explicit() -> None:
    base = pd.DataFrame([_inventory_row()])
    attached = attach_validation_benchmarks(base, benchmark_runner=_benchmark_result)
    row = attached.iloc[0]
    assert row["excess_return"] == pytest.approx(0.10)
    assert row["sharpe_delta"] == pytest.approx(0.50)
    assert row["sortino_delta"] == pytest.approx(0.50)
    assert row["drawdown_improvement"] == pytest.approx(0.05)
    assert row["volatility_difference"] == pytest.approx(0.02)

    acceptable = dict(attached.iloc[0])
    acceptable.update(excess_return=-0.05, sharpe_delta=-0.10)
    weak = dict(attached.iloc[0])
    weak.update(validation_total_return=-0.01, validation_sharpe=-0.1)
    insufficient = dict(attached.iloc[0])
    insufficient.update(symbol="AATM", validation_rows=73)
    invalid = dict(attached.iloc[0])
    invalid.update(symbol="BAD", benchmark_status="INVALID")
    classified = apply_frozen_acceptance_policy(
        pd.DataFrame([dict(attached.iloc[0]), acceptable, weak, insufficient, invalid])
    )
    assert classified["acceptance_classification"].tolist() == [
        STRONG_VALIDATION,
        ACCEPTABLE_VALIDATION,
        WEAK_VALIDATION,
        INSUFFICIENT_VALIDATION_HISTORY,
        INVALID_VALIDATION,
    ]
    aatm = classified.loc[classified["symbol"].eq("AATM")].iloc[0]
    assert aatm["validation_sufficiency"] == "INSUFFICIENT"
    assert "73 rows" in aatm["acceptance_reasons"]


def test_policy_is_deterministic_and_freeze_schema_validates() -> None:
    rows = []
    for index in range(16):
        symbol = f"S{index:02d}"
        source = pd.DataFrame([_inventory_row(symbol)])
        result = replace(_benchmark_result(symbol), source_validation_artifact_sha256=f"{index:064x}")
        rows.append(attach_validation_benchmarks(source, benchmark_runner=lambda _, value=result: value).iloc[0])
    inventory = apply_frozen_acceptance_policy(pd.DataFrame(rows))
    first = build_policy_freeze_payload(inventory, frozen_at="2026-08-31T00:00:00Z")
    second = build_policy_freeze_payload(
        inventory.sample(frac=1.0, random_state=7),
        frozen_at="2026-08-31T00:00:00Z",
    )

    assert first == second
    assert first["artifact_schema_version"] == VALIDATION_ACCEPTANCE_FREEZE_SCHEMA
    assert first["policy_version"] == VALIDATION_ACCEPTANCE_POLICY_VERSION
    assert first["model_inventory"]["count"] == 16
    assert first["test_status_at_freeze"] == "SEALED"
    assert first["test_observations_accessed"] is False
    validate_policy_freeze(first, inventory)
    tampered = dict(first)
    tampered["thresholds"] = dict(first["thresholds"], maximum_rl_drawdown=0.99)
    with pytest.raises(ValidationAcceptanceError, match="thresholds"):
        validate_policy_freeze(tampered, inventory)


def test_test_evidence_columns_and_loaded_flag_fail_closed() -> None:
    inventory = pd.DataFrame([_inventory_row()])
    inventory["test_metrics"] = [{"return": 99.0}]
    with pytest.raises(ValidationAcceptanceError, match="TEST evidence"):
        attach_validation_benchmarks(inventory, benchmark_runner=_benchmark_result)

    loaded = pd.DataFrame([_inventory_row()])
    loaded["test_partition_loaded"] = True
    with pytest.raises(ValidationAcceptanceError, match="TEST access"):
        attach_validation_benchmarks(loaded, benchmark_runner=_benchmark_result)


def test_extended_export_has_benchmark_policy_provenance_and_sealed_test() -> None:
    classified = apply_frozen_acceptance_policy(
        attach_validation_benchmarks(
            pd.DataFrame([_inventory_row("001")]),
            benchmark_runner=lambda _: _benchmark_result("001"),
        )
    )
    export = validation_export_frame(classified)
    required = {
        "benchmark_total_return",
        "benchmark_sharpe",
        "benchmark_sortino",
        "benchmark_max_drawdown",
        "excess_return",
        "sharpe_delta",
        "sortino_delta",
        "drawdown_improvement",
        "validation_sufficiency",
        "acceptance_classification",
        "acceptance_policy_version",
        "benchmark_contract_version",
        "run_id",
        "model_sha256",
        "TEST_status",
    }
    assert required.issubset(EXPORT_COLUMNS)
    assert export.loc[0, "symbol"] == "001"
    assert export.loc[0, "TEST_status"] == "SEALED"
    assert bool(export.loc[0, "test_partition_loaded"]) is False


def test_modules_have_no_training_promotion_or_test_loader_path() -> None:
    root = Path(__file__).resolve().parents[2]
    benchmark_source = (
        root / "reinforcement_learning/evaluation/validation_benchmark.py"
    ).read_text(encoding="utf-8")
    policy_source = (
        root / "reinforcement_learning/evaluation/validation_acceptance.py"
    ).read_text(encoding="utf-8")
    combined = benchmark_source + policy_source

    assert "train_recurrent_single_symbol" not in combined
    assert ".learn(" not in combined
    assert "register_model" not in combined
    assert "promote_model" not in combined
    assert 'partition_loader(symbol_text, "validation")' in benchmark_source
    assert 'partition_loader(symbol_text, "test")' not in benchmark_source
