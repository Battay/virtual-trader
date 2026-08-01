"""Offline tests for chronological splits and training-only scaling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.preprocessing import fit_training_scaler
from feature_engineering.schemas import AI_DATASET_COLUMNS, FEATURE_COLUMNS, FEATURE_VERSION
from feature_engineering.splitting import chronological_split, persist_split_artifacts


def _processed(symbols: tuple[str, ...], dates: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, trading_date in enumerate(pd.date_range("2025-01-01", periods=dates)):
            row: dict[str, object] = {
                "symbol": symbol,
                "date": trading_date,
                "is_active": symbol_index == 0,
                "official_status": "listed" if symbol_index == 0 else "historical",
                "lifecycle_status": (
                    "listed_recently_traded" if symbol_index == 0 else "historical_only"
                ),
                "security_type": "ordinary_equity",
                "feature_version": FEATURE_VERSION,
            }
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = float(date_index + feature_index + symbol_index)
            rows.append(row)
    return pd.DataFrame(rows).loc[:, AI_DATASET_COLUMNS]


def _date_sets(split) -> tuple[set[object], set[object], set[object]]:
    return tuple(
        set(partition["date"])
        for partition in (split.train, split.validation, split.test)
    )


def test_symbol_split_is_chronological_with_no_date_overlap() -> None:
    split = chronological_split(_processed(("MCB",), dates=20), scope="symbol")
    train_dates, validation_dates, test_dates = _date_sets(split)

    assert len(train_dates) == 14
    assert len(validation_dates) == 3
    assert len(test_dates) == 3
    assert max(train_dates) < min(validation_dates) < min(test_dates)
    assert train_dates.isdisjoint(validation_dates | test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_master_split_uses_global_boundaries_for_every_symbol() -> None:
    split = chronological_split(
        _processed(("MCB", "HIST"), dates=20),
        scope="master",
    )
    train_dates, validation_dates, test_dates = _date_sets(split)

    assert len(split.train) == 28
    assert len(split.validation) == 6
    assert len(split.test) == 6
    assert train_dates.isdisjoint(validation_dates | test_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert split.metadata["training"]["symbols"] == ["HIST", "MCB"]


def test_scaler_is_fit_only_on_training_rows_and_identity_is_not_scaled() -> None:
    train = pd.DataFrame({"symbol": ["MCB", "MCB"], "date": [1, 2], "open": [0.0, 2.0]})
    validation = pd.DataFrame({"symbol": ["MCB"], "date": [3], "open": [100.0]})
    test = pd.DataFrame({"symbol": ["MCB"], "date": [4], "open": [200.0]})

    result = fit_training_scaler(
        train,
        validation,
        test,
        feature_columns=("open",),
    )

    assert result.scaler.mean_.tolist() == [1.0]
    assert result.validation["open"].iloc[0] == 99.0
    assert result.test["open"].iloc[0] == 199.0
    assert result.validation["symbol"].tolist() == ["MCB"]
    assert result.validation["date"].tolist() == [3]


def test_split_artifacts_include_atomic_metadata_and_scaler(tmp_path: Path) -> None:
    split = chronological_split(_processed(("MCB",), dates=20), scope="symbol")

    result = persist_split_artifacts(split, tmp_path / "MCB")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.scaler_path.is_file()
    assert (result.output_dir / "train.csv").is_file()
    assert (result.output_dir / "validation_scaled.csv").is_file()
    assert metadata["training"]["end"] < metadata["validation"]["start"]
    assert metadata["validation"]["end"] < metadata["testing"]["start"]
    assert metadata["scaled_features"] == list(FEATURE_COLUMNS)
