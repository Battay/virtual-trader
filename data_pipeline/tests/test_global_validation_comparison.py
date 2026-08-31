"""Offline tests for persisted recurrent VALIDATION comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reinforcement_learning.training.global_validation import (
    EXPORT_COLUMNS,
    GlobalValidationError,
    INVALID,
    MISSING,
    VALID,
    build_global_validation_inventory,
    build_sector_validation_summary,
    load_persisted_validation_returns,
    summarize_validation_inventory,
    validation_export_csv,
    validation_export_frame,
)


def _source_row(root: Path, symbol: str, *, run_type: str, sector: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "company_name": f"{symbol} Limited",
        "sector": sector,
        "run_type": run_type,
        "run_id": f"run-{run_type.lower()}",
        "attempt": 1,
        "algorithm": "RecurrentPPO",
        "policy": "MlpLstmPolicy",
        "trainer_version": "recurrent_ppo_single_symbol_v1",
        "artifact_verification": "verified",
        "validation_status": "completed",
        "validation_start": "2024-01-01",
        "validation_end": "2024-06-30",
        "validation_rows": 150,
        "partition_contract_version": "rl_partition_v1",
        "recurrent_contract_version": "rl_recurrent_partition_v1",
        "feature_version": "feature-v1",
        "environment_version": "single_symbol_env_v1",
        "split_policy_version": "split-v1",
        "scaler_fit_partition": "train",
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "train_rows": 700,
        "model_path": f"models/{symbol}/attempt_000/model.zip",
        "model_sha256": symbol.lower().ljust(64, "0")[:64],
        "validation_metrics_reference": f"validation/{symbol}/attempt_000.json",
        "source_contract_sha256": "c" * 64,
        "hyperparameters_hash": "h" * 64,
        "seed": 42,
        "run_directory": str(root),
    }


def _payload(
    symbol: str,
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    drawdown: float,
    benchmark: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": symbol,
        "evaluation_partition": "validation",
        "validation_start": "2024-01-01",
        "validation_end": "2024-06-30",
        "validation_rows": 150,
        "recurrent_contract_version": "rl_recurrent_partition_v1",
        "feature_version": "feature-v1",
        "environment_version": "single_symbol_env_v1",
        "model_parameters_unchanged": True,
        "parameter_hash_before": "p" * 64,
        "parameter_hash_after": "p" * 64,
        "model_timesteps_before": 100_352,
        "model_timesteps_after": 100_352,
        "strategy_result": {
            "strategy": "RecurrentPPO",
            "metrics": {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "maximum_drawdown": drawdown,
                "annualized_volatility": 0.20,
                "number_of_trades": 7,
                "completed_trade_win_rate": 0.6,
                "final_portfolio_value": 1_000_000 * (1 + total_return),
                "daily_returns": [0.0, total_return],
                "metric_warnings": [],
            },
        },
    }
    if benchmark is not None:
        payload["buy_and_hold"] = {
            "strategy": "Buy & Hold",
            "metrics": {"total_return": benchmark},
        }
        payload["candidate_decision"] = {
            "status": "validation_pass",
            "passed": True,
        }
    return payload


def _write_artifacts(root: Path, row: dict[str, object], payload: dict[str, object]) -> None:
    validation = root / str(row["validation_metrics_reference"])
    validation.parent.mkdir(parents=True, exist_ok=True)
    validation.write_text(json.dumps(payload), encoding="utf-8")
    log = root / "logs" / str(row["symbol"]) / "attempt_000.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"training_diagnostics": {"approximate_kl": 0.01}}),
        encoding="utf-8",
    )


def test_inventory_extracts_all_cross_run_models_and_metrics(tmp_path: Path) -> None:
    rows = [
        _source_row(tmp_path, "BBB", run_type="SELECTED", sector="BANKS"),
        _source_row(tmp_path, "AAA", run_type="FULL_PRODUCTION", sector="BANKS"),
        _source_row(tmp_path, "CCC", run_type="SELECTED", sector="CEMENT"),
    ]
    values = {
        "AAA": (0.10, 1.0, 1.2, 0.10, 0.08),
        "BBB": (0.05, 0.5, 0.8, 0.20, None),
        "CCC": (-0.02, -0.1, -0.2, 0.30, 0.01),
    }
    for row in rows:
        _write_artifacts(tmp_path, row, _payload(row["symbol"], total_return=values[row["symbol"]][0], sharpe=values[row["symbol"]][1], sortino=values[row["symbol"]][2], drawdown=values[row["symbol"]][3], benchmark=values[row["symbol"]][4]))

    inventory = build_global_validation_inventory(verified_inventory=pd.DataFrame(rows))

    assert set(inventory["symbol"]) == {"AAA", "BBB", "CCC"}
    assert inventory["validation_artifact_status"].eq(VALID).all()
    aaa = inventory.set_index("symbol").loc["AAA"]
    assert aaa["validation_total_return"] == 0.10
    assert aaa["benchmark_total_return"] == 0.08
    assert aaa["excess_return"] == pytest.approx(0.02)
    assert aaa["validation_sharpe"] == 1.0
    assert aaa["validation_sortino"] == 1.2
    assert aaa["validation_max_drawdown"] == 0.10
    assert aaa["validation_volatility"] == 0.20
    assert aaa["trade_count"] == 7
    assert aaa["win_rate"] == 0.6
    assert aaa["final_portfolio_value"] == 1_100_000
    assert aaa["acceptance_passed"] is True
    assert bool(aaa["training_diagnostics_available"]) is True


def test_all_sixteen_verified_models_remain_in_cross_run_validation_inventory(
    tmp_path: Path,
) -> None:
    symbols = (
        "786", "AABS", "AATM", "ABL", "ENGROH", "FFC", "HUBC", "LUCK",
        "MARI", "MCB", "OGDC", "PSO", "SYS", "TRG", "UBL", "UNITY",
    )
    rows = []
    for index, symbol in enumerate(symbols):
        row = _source_row(
            tmp_path,
            symbol,
            run_type="FULL_PRODUCTION" if index < 4 else "SELECTED",
            sector="BANKS" if index % 2 else "ENERGY",
        )
        rows.append(row)
        _write_artifacts(
            tmp_path,
            row,
            _payload(
                symbol,
                total_return=index / 100,
                sharpe=index / 10,
                sortino=index / 8,
                drawdown=0.05 + index / 100,
            ),
        )

    inventory = build_global_validation_inventory(
        verified_inventory=pd.DataFrame(reversed(rows))
    )

    assert len(inventory) == 16
    assert set(inventory["symbol"]) == set(symbols)
    assert set(inventory["run_type"]) == {"FULL_PRODUCTION", "SELECTED"}
    assert inventory["validation_artifact_status"].eq(VALID).all()


def test_missing_and_invalid_artifacts_are_reported_without_recompute(tmp_path: Path) -> None:
    valid = _source_row(tmp_path, "AAA", run_type="SELECTED", sector="BANKS")
    missing = _source_row(tmp_path, "BBB", run_type="SELECTED", sector="BANKS")
    invalid = _source_row(tmp_path, "CCC", run_type="SELECTED", sector="CEMENT")
    _write_artifacts(tmp_path, valid, _payload("AAA", total_return=0.1, sharpe=1, sortino=1, drawdown=0.1))
    bad = _payload("CCC", total_return=0.1, sharpe=1, sortino=1, drawdown=0.1)
    bad["test_metrics"] = {"total_return": 99.0}
    _write_artifacts(tmp_path, invalid, bad)

    inventory = build_global_validation_inventory(
        verified_inventory=pd.DataFrame([valid, missing, invalid])
    ).set_index("symbol")

    assert inventory.loc["AAA", "validation_artifact_status"] == VALID
    assert inventory.loc["BBB", "validation_artifact_status"] == MISSING
    assert "missing" in inventory.loc["BBB", "validation_error"]
    assert inventory.loc["CCC", "validation_artifact_status"] == INVALID
    assert "TEST" in inventory.loc["CCC", "validation_error"]


def test_rankings_sector_summary_and_aggregates_are_deterministic(tmp_path: Path) -> None:
    rows = [
        _source_row(tmp_path, "AAA", run_type="FULL_PRODUCTION", sector="BANKS"),
        _source_row(tmp_path, "BBB", run_type="SELECTED", sector="BANKS"),
        _source_row(tmp_path, "CCC", run_type="SELECTED", sector="CEMENT"),
    ]
    metrics = {
        "AAA": (0.10, 1.0, 1.4, 0.10, 0.05),
        "BBB": (0.20, 0.5, 0.7, 0.20, 0.25),
        "CCC": (-0.10, -0.2, -0.4, 0.30, -0.20),
    }
    for row in rows:
        values = metrics[row["symbol"]]
        _write_artifacts(tmp_path, row, _payload(row["symbol"], total_return=values[0], sharpe=values[1], sortino=values[2], drawdown=values[3], benchmark=values[4]))

    first = build_global_validation_inventory(verified_inventory=pd.DataFrame(rows))
    second = build_global_validation_inventory(
        verified_inventory=pd.DataFrame(list(reversed(rows)))
    )
    rank_fields = ["symbol", "return_rank", "sharpe_rank", "sortino_rank", "drawdown_rank", "excess_return_rank"]
    pd.testing.assert_frame_equal(first[rank_fields], second[rank_fields])
    ranks = first.set_index("symbol")
    assert ranks.loc["BBB", "return_rank"] == 1
    assert ranks.loc["AAA", "sharpe_rank"] == 1
    assert ranks.loc["AAA", "drawdown_rank"] == 1

    summary = summarize_validation_inventory(first)
    assert summary.verified_models == 3
    assert summary.validation_complete == 3
    assert summary.positive_validation_return == 2
    assert summary.positive_excess_return == 2
    assert summary.median_return == 0.10
    sectors = build_sector_validation_summary(first).set_index("sector")
    assert sectors.loc["BANKS", "model_count"] == 2
    assert sectors.loc["BANKS", "positive_return_count"] == 2
    assert sectors.loc["BANKS", "median_validation_return"] == pytest.approx(0.15)


def test_export_is_deterministic_and_explicitly_test_sealed(tmp_path: Path) -> None:
    row = _source_row(tmp_path, "001", run_type="SELECTED", sector="BANKS")
    _write_artifacts(tmp_path, row, _payload("001", total_return=0.1, sharpe=1, sortino=1, drawdown=0.1))
    inventory = build_global_validation_inventory(verified_inventory=pd.DataFrame([row]))

    export = validation_export_frame(inventory)
    assert tuple(export.columns) == EXPORT_COLUMNS
    assert export.loc[0, "symbol"] == "001"
    assert export.loc[0, "TEST_status"] == "SEALED"
    assert bool(export.loc[0, "test_partition_loaded"]) is False
    assert validation_export_csv(inventory) == validation_export_csv(inventory)
    assert b"daily_returns" not in validation_export_csv(inventory)


def test_loader_reads_json_only_and_persisted_validation_returns(tmp_path: Path, monkeypatch) -> None:
    row = _source_row(tmp_path, "AAA", run_type="SELECTED", sector="BANKS")
    _write_artifacts(tmp_path, row, _payload("AAA", total_return=0.1, sharpe=1, sortino=1, drawdown=0.1))
    read_paths: list[Path] = []

    def loader(path: Path) -> dict[str, object]:
        read_paths.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("partition frame opened")))
    inventory = build_global_validation_inventory(
        verified_inventory=pd.DataFrame([row]), json_loader=loader
    )
    curve = load_persisted_validation_returns(
        tmp_path,
        row["validation_metrics_reference"],
        json_loader=loader,
    )

    assert inventory.loc[0, "validation_artifact_status"] == VALID
    assert not curve.empty
    assert curve.iloc[-1]["cumulative_return"] == pytest.approx(0.1)
    assert all(path.suffix == ".json" for path in read_paths)
    assert not any(path.name in {"test.csv", "test_rl.csv"} for path in read_paths)


def test_comparison_rejects_source_inventory_that_reports_test_access(
    tmp_path: Path,
) -> None:
    row = _source_row(tmp_path, "AAA", run_type="SELECTED", sector="BANKS")
    row["test_partition_loaded"] = True
    _write_artifacts(
        tmp_path,
        row,
        _payload("AAA", total_return=0.1, sharpe=1, sortino=1, drawdown=0.1),
    )

    with pytest.raises(GlobalValidationError, match="TEST partition access"):
        build_global_validation_inventory(verified_inventory=pd.DataFrame([row]))


def test_comparison_module_has_no_training_evaluation_or_promotion_path() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "reinforcement_learning"
        / "training"
        / "global_validation.py"
    ).read_text(encoding="utf-8")

    assert "evaluate_recurrent_on_validation" not in source
    assert "load_recurrent_partition" not in source
    assert "train_recurrent_single_symbol" not in source
    assert "register_model" not in source
    assert "promote" not in source.casefold()
