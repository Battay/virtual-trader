"""Offline tests for model lifecycle, staleness, and symbol selection."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.presentation import format_model_registry_for_display
from feature_engineering.schemas import FEATURE_VERSION
from reinforcement_learning.model_management.registry import (
    MODEL_REGISTRY_COLUMNS,
    ModelRegistryError,
    append_model_version,
    create_model_record,
    empty_model_registry,
    initialize_model_registry,
    load_model_registry,
)
from reinforcement_learning.model_management.selection import (
    bulk_select_symbols,
    filter_symbol_status,
    merge_symbol_selections,
    normalize_symbol_selection,
    select_all_active_eligible,
    select_needing_retraining,
    select_never_trained,
    select_newly_added_eligible,
    select_visible_symbols,
    selected_symbols_from_editor,
    symbol_selection_counts,
    update_visible_symbol_selection,
)
from reinforcement_learning.model_management.status import (
    build_symbol_status_table,
    complete_history_training_metadata,
    count_new_trading_dates,
)


def _market(symbol: str, rows: int = 60) -> pd.DataFrame:
    close = pd.Series(np.arange(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2026-01-01", periods=rows),
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
                "company_name": "786 Investments Limited",
                "sector": "Investment",
                "security_type": "ordinary_equity",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "is_new_listing": True,
            },
            {
                "symbol": "OLD",
                "company_name": "Old Limited",
                "sector": "Other",
                "security_type": "ordinary_equity",
                "officially_listed": False,
                "activity_status": "not_recently_traded",
                "is_new_listing": False,
            },
        ]
    )


def test_model_registry_creation_and_versioning_preserve_history(tmp_path: Path) -> None:
    path = tmp_path / "model_registry.csv"
    initialize_model_registry(path)
    registry = load_model_registry(path)
    first = create_model_record(
        registry=registry,
        model_scope="symbol",
        symbol="786",
        feature_version=FEATURE_VERSION,
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    registry = append_model_version(first, path)
    second = create_model_record(
        registry=registry,
        model_scope="symbol",
        symbol="786",
        feature_version=FEATURE_VERSION,
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    updated = append_model_version(second, path)

    assert updated["model_version"].tolist() == [1, 2]
    assert updated["model_id"].tolist() == [
        "ppo-symbol-786-v0001",
        "ppo-symbol-786-v0002",
    ]
    assert len(load_model_registry(path)) == 2


def test_malformed_model_registry_fails_safely(tmp_path: Path) -> None:
    path = tmp_path / "model_registry.csv"
    pd.DataFrame({"model_id": ["bad"]}).to_csv(path, index=False)
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="missing columns"):
        load_model_registry(path)

    assert path.read_text(encoding="utf-8") == original


def test_never_trained_and_outdated_status_use_distinct_trading_dates() -> None:
    market = pd.concat([_market("786"), _market("OLD")], ignore_index=True)
    record = create_model_record(
        registry=empty_model_registry(),
        model_scope="symbol",
        symbol="786",
        feature_version=FEATURE_VERSION,
        values={
            "model_status": "trained",
            "training_status": "up_to_date",
            "training_data_start": "2026-01-01",
            "training_data_end": "2026-02-25",
            "last_trained_at": "2026-02-25T10:00:00+00:00",
        },
    )
    models = pd.DataFrame([record], columns=MODEL_REGISTRY_COLUMNS)

    status = build_symbol_status_table(
        market,
        _registry(),
        models,
        minimum_usable_rows=1,
    )
    row = status.iloc[0]

    assert row["symbol"] == "786"
    assert row["is_newly_added"]
    assert row["eligible"]
    assert row["training_status"] == "retraining_recommended"
    assert row["new_data_days"] == 4
    assert count_new_trading_dates(_market("786"), "2026-02-25") == 4


def test_sufficient_new_active_symbol_without_model_is_never_trained() -> None:
    status = build_symbol_status_table(
        _market("786"),
        _registry(),
        empty_model_registry(),
        minimum_usable_rows=1,
    )

    assert status["training_status"].tolist() == ["never_trained"]


def test_registry_metadata_separates_complete_history_from_partitions() -> None:
    history = _market("786", rows=10)
    metadata = complete_history_training_metadata(
        history,
        {
            "training": {
                "start": "2026-01-01",
                "end": "2026-01-07",
                "rows": 7,
            },
            "validation": {"start": "2026-01-08", "end": "2026-01-08", "rows": 1},
            "testing": {"start": "2026-01-09", "end": "2026-01-10", "rows": 2},
        },
    )

    assert metadata["complete_available_history_start"] == "2026-01-01"
    assert metadata["complete_available_history_end"] == "2026-01-10"
    assert metadata["training_data_start"] == "2026-01-01"
    assert metadata["training_data_end"] == "2026-01-07"
    assert metadata["validation_data_start"] == "2026-01-08"
    assert metadata["test_data_start"] == "2026-01-09"
    assert metadata["dataset_latest_date"] == "2026-01-10"
    assert metadata["training_rows"] == 7


def test_multiple_and_bulk_symbol_selection_helpers() -> None:
    status = pd.DataFrame(
        [
            {
                "symbol": "786",
                "company_name": "786 Investments",
                "sector": "Investment",
                "security_type": "ordinary_equity",
                "training_status": "never_trained",
                "is_active": True,
                "is_newly_added": True,
                "eligible": True,
                "needs_retraining": False,
            },
            {
                "symbol": "MCB",
                "company_name": "MCB Bank",
                "sector": "Banks",
                "security_type": "ordinary_equity",
                "training_status": "retraining_recommended",
                "is_active": True,
                "is_newly_added": False,
                "eligible": True,
                "needs_retraining": True,
            },
            {
                "symbol": "ETF",
                "company_name": "Example ETF",
                "sector": "ETF",
                "security_type": "etf",
                "training_status": "unsupported_security_type",
                "is_active": True,
                "is_newly_added": False,
                "eligible": False,
                "needs_retraining": False,
            },
        ]
    )

    filtered = filter_symbol_status(status, search="bank", sectors={"Banks"})

    assert select_visible_symbols(filtered) == ("MCB",)
    assert select_all_active_eligible(status) == ("786", "MCB")
    assert select_never_trained(status) == ("786",)
    assert select_needing_retraining(status) == ("MCB",)
    assert select_newly_added_eligible(status) == ("786",)
    assert merge_symbol_selections(("MCB",), ("786", "MCB")) == ("786", "MCB")


def test_editor_checkbox_selection_adds_and_unchecking_removes_symbol() -> None:
    checked = pd.DataFrame(
        {"selected": [True, False], "symbol": ["MCB", "OGDC"]}
    )
    unchecked = checked.assign(selected=[False, False])

    assert selected_symbols_from_editor(checked) == ("MCB",)
    assert selected_symbols_from_editor(unchecked) == ()


def test_editor_selection_preserves_numeric_symbol_as_string_and_deduplicates() -> None:
    edited = pd.DataFrame(
        {"selected": [True, True, True], "symbol": [786, "786", "MCB"]}
    )

    assert selected_symbols_from_editor(edited) == ("786", "MCB")


def test_bulk_select_all_visible_and_clear_selection() -> None:
    all_symbols = ("786", "MCB", "OGDC")

    selected = bulk_select_symbols(
        ("MCB", "OGDC"),
        all_symbols=all_symbols,
        current_selection=("786",),
    )

    assert selected == all_symbols
    assert normalize_symbol_selection((), allowed_symbols=all_symbols) == ()


def test_editor_selection_survives_filtering_and_reports_both_counts() -> None:
    all_symbols = ("786", "MCB", "OGDC")
    updated = update_visible_symbol_selection(
        ("786", "MCB"),
        ("MCB",),
        ("MCB",),
        all_symbols=all_symbols,
    )

    assert updated == ("786", "MCB")
    assert symbol_selection_counts(updated, ("MCB", "OGDC")) == (1, 2)


def test_multiselect_and_editor_values_share_one_canonical_selection() -> None:
    all_symbols = ("786", "MCB", "OGDC")
    multiselect_selection = normalize_symbol_selection(
        ("OGDC", "786", "OGDC"),
        allowed_symbols=all_symbols,
    )
    edited = pd.DataFrame(
        {"selected": [True, False], "symbol": ["786", "MCB"]}
    )
    synchronized = update_visible_symbol_selection(
        multiselect_selection,
        edited["symbol"],
        selected_symbols_from_editor(edited),
        all_symbols=all_symbols,
    )

    assert multiselect_selection == ("786", "OGDC")
    assert synchronized == ("786", "OGDC")


def test_model_registry_display_uses_readable_statuses() -> None:
    record = create_model_record(
        registry=empty_model_registry(),
        model_scope="master",
        feature_version=FEATURE_VERSION,
    )
    display = format_model_registry_for_display(
        pd.DataFrame([record], columns=MODEL_REGISTRY_COLUMNS)
    )
    text = " ".join(display.astype(str).to_numpy().ravel())

    assert "Not Trained" in text
    assert "not_trained" not in text
    assert "Master" in text
