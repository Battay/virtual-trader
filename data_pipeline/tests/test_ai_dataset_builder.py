"""Offline tests for atomic symbol and master AI dataset construction."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.dataset_builder import (
    build_master_ai_dataset,
    build_symbol_datasets,
    validate_ai_dataset,
)
from feature_engineering.schemas import AI_DATASET_COLUMNS, FEATURE_VERSION
from feature_engineering.storage import atomic_write_dataframe


def _market(symbol: str, rows: int = 70, base: float = 100.0) -> pd.DataFrame:
    close = pd.Series(np.arange(base, base + rows), dtype=float)
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


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "786",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "official_status": "listed",
                "lifecycle_status": "listed_recently_traded",
                "security_type": "ordinary_equity",
            },
            {
                "symbol": "HIST",
                "officially_listed": False,
                "activity_status": "not_recently_traded",
                "official_status": "historical",
                "lifecycle_status": "historical_only",
                "security_type": "ordinary_equity",
            },
            {
                "symbol": "UNKNOWN",
                "officially_listed": False,
                "activity_status": "not_recently_traded",
                "official_status": "historical",
                "lifecycle_status": "historical_only",
                "security_type": "unknown",
            },
        ]
    )


def _all_market() -> pd.DataFrame:
    return pd.concat(
        [
            _market("786", base=100),
            _market("HIST", base=200),
            _market("UNKNOWN", base=300),
        ],
        ignore_index=True,
    )


def test_symbol_dataset_generation_uses_shared_features_and_eligibility(
    tmp_path: Path,
) -> None:
    metrics = build_symbol_datasets(
        market_data=_all_market(),
        registry=_registry(),
        minimum_usable_rows=10,
        output_dir=tmp_path,
    )

    output = pd.read_csv(tmp_path / "786.csv", dtype={"symbol": "string"})
    assert metrics.output_rows == 21
    assert metrics.unique_symbols == 1
    assert metrics.warmup_rows_removed == 147
    assert metrics.symbols_skipped == ("HIST", "UNKNOWN")
    assert tuple(output.columns) == AI_DATASET_COLUMNS
    assert output["symbol"].unique().tolist() == ["786"]
    assert output["feature_version"].unique().tolist() == [FEATURE_VERSION]
    assert not output.isna().any(axis=None)


def test_master_dataset_includes_active_and_historical_but_excludes_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "psx_ai_master.csv"

    metrics = build_master_ai_dataset(
        market_data=_all_market(),
        registry=_registry(),
        output_path=path,
    )

    output = pd.read_csv(path, dtype={"symbol": "string"})
    assert metrics.output_rows == 42
    assert metrics.unique_symbols == 2
    assert set(output["symbol"]) == {"786", "HIST"}
    assert set(output["lifecycle_status"]) == {
        "listed_recently_traded",
        "historical_only",
    }
    assert output.loc[output["symbol"] == "786", "is_active"].all()
    assert not output.loc[output["symbol"] == "HIST", "is_active"].any()


def test_insufficient_history_is_skipped_without_fake_symbol_files(
    tmp_path: Path,
) -> None:
    metrics = build_symbol_datasets(
        market_data=_market("786", rows=60),
        registry=_registry().iloc[[0]],
        minimum_usable_rows=20,
        output_dir=tmp_path,
    )

    assert metrics.output_rows == 0
    assert metrics.symbols_skipped == ("786",)
    assert not (tmp_path / "786.csv").exists()


def test_atomic_dataframe_write_preserves_existing_file_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "processed.csv"
    destination.write_text("original\n", encoding="utf-8")

    def fail_to_csv(*args, **kwargs) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    try:
        atomic_write_dataframe(pd.DataFrame({"value": [1]}), destination)
    except OSError:
        pass

    assert destination.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_ai_dataset_validation_handles_empty_and_malformed_files(tmp_path: Path) -> None:
    valid_empty = tmp_path / "empty.csv"
    malformed = tmp_path / "malformed.csv"
    pd.DataFrame(columns=AI_DATASET_COLUMNS).to_csv(valid_empty, index=False)
    pd.DataFrame({"symbol": ["MCB"]}).to_csv(malformed, index=False)

    empty_result = validate_ai_dataset(valid_empty)
    malformed_result = validate_ai_dataset(malformed)

    assert empty_result.valid
    assert empty_result.rows == 0
    assert not malformed_result.valid
    assert "Missing columns" in malformed_result.errors[0]
