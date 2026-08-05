"""Offline tests for exact symbol AI dataset-readiness reporting."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.dataset_builder import build_symbol_datasets
from feature_engineering.readiness import (
    additional_required_rows,
    build_training_readiness_report,
    summarize_symbol_build_readiness,
)
from reinforcement_learning.model_management.registry import empty_model_registry
from reinforcement_learning.model_management.status import build_model_readiness_table


def _market(symbol: str, rows: int) -> pd.DataFrame:
    close = pd.Series(np.arange(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2025-01-01", periods=rows),
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
        }
    )


def _registry(symbol: str, security_type: str = "ordinary_equity") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "official_status": "listed",
                "lifecycle_status": "listed_recently_traded",
                "security_type": security_type,
            }
        ]
    )


def test_insufficient_history_and_additional_rows_are_exact(tmp_path: Path) -> None:
    report = build_training_readiness_report(
        _market("MCB", 60),
        _registry("MCB"),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )
    row = report.iloc[0]

    assert row["raw_trading_rows"] == 60
    assert row["warmup_rows_removed"] == 49
    assert row["usable_feature_rows"] == 11
    assert row["additional_rows_required"] == 241
    assert row["train_rows"] == 0
    assert row["readiness_status"] == "Insufficient History"
    assert additional_required_rows(252, 252) == 0


def test_ready_symbol_reports_chronological_partition_rows(tmp_path: Path) -> None:
    market = _market("MCB", 310)
    registry = _registry("MCB")
    build_symbol_datasets(
        market_data=market,
        registry=registry,
        include_market_context=False,
        minimum_usable_rows=252,
        output_dir=tmp_path,
    )

    report = build_training_readiness_report(
        market,
        registry,
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )
    row = report.iloc[0]

    assert row["usable_feature_rows"] == 261
    assert row["additional_rows_required"] == 0
    assert (row["train_rows"], row["validation_rows"], row["test_rows"]) == (
        182,
        39,
        40,
    )
    assert row["readiness_status"] == "Ready"


def test_missing_processed_unsupported_and_quality_statuses(tmp_path: Path) -> None:
    missing = build_training_readiness_report(
        _market("MCB", 310),
        _registry("MCB"),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )
    unsupported = build_training_readiness_report(
        _market("ETF", 310),
        _registry("ETF", "etf"),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )
    invalid_market = _market("BAD", 310)
    invalid_market.loc[1, "date"] = invalid_market.loc[0, "date"]
    invalid = build_training_readiness_report(
        invalid_market,
        _registry("BAD"),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )

    assert missing.iloc[0]["readiness_status"] == "Missing Processed Dataset"
    assert unsupported.iloc[0]["readiness_status"] == "Unsupported Security Type"
    assert invalid.iloc[0]["readiness_status"] == "Data Quality Issue"


def test_notebook_summary_treats_skipped_symbol_as_readiness_result(
    tmp_path: Path,
) -> None:
    market = _market("MCB", 33)
    registry = _registry("MCB")
    metrics = build_symbol_datasets(
        market_data=market,
        registry=registry,
        include_market_context=False,
        minimum_usable_rows=252,
        output_dir=tmp_path,
    )
    summary = summarize_symbol_build_readiness(
        symbol="MCB",
        raw_history=market,
        metrics=metrics,
        minimum_usable_rows=252,
        processed_path=tmp_path / "MCB.csv",
    )

    assert not summary.is_training_ready
    assert summary.raw_history_rows == 33
    assert summary.warmup_rows_removed == 33
    assert summary.usable_rows == 0
    assert summary.additional_usable_rows_required == 252
    assert not summary.processed_path.exists()


def test_model_status_table_uses_missing_processed_readiness(tmp_path: Path) -> None:
    status = build_model_readiness_table(
        _market("MCB", 310),
        _registry("MCB"),
        empty_model_registry(),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )

    assert status.iloc[0]["readiness_status"] == "Missing Processed Dataset"
    assert status.iloc[0]["training_status"] == "missing_processed_dataset"
    assert not status.iloc[0]["eligible"]
