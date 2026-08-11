"""Static safety tests for Training & Models page ownership."""

from pathlib import Path

from data_pipeline.src.data_products import rebuild_data_products
from feature_engineering.dataset_builder import (
    build_master_ai_dataset,
    build_symbol_datasets,
    validate_ai_dataset,
)
from feature_engineering.splitting import create_master_split, create_symbol_split


PAGE_PATH = Path(__file__).resolve().parents[2] / "app_pages" / "6_Training_and_Models.py"


def _source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


def test_training_page_has_one_fixed_current_mlp_readiness_authority() -> None:
    source = _source()

    assert "AI_MINIMUM_USABLE_ROWS" in source
    assert "Minimum usable rows for symbol eligibility" not in source
    assert "training_minimum_history" not in source
    assert "history_counts = future_history_class_counts(status_table)" in source
    assert "do not rewrite this gate" in source


def test_training_page_contains_no_dataset_build_or_split_controls() -> None:
    source = _source()
    forbidden = (
        "Build/Refresh AI Datasets",
        "Validate AI Datasets",
        "Create/Refresh Chronological Splits",
        "Prepare Selected Datasets",
        "Prepare Master Dataset",
        "Train Selected Models",
        "Train Master Model",
        "build_symbol_datasets",
        "build_master_ai_dataset",
        "validate_ai_dataset",
        "create_symbol_split",
        "create_master_split",
    )

    assert not [value for value in forbidden if value in source]


def test_dataset_operation_backends_remain_available_for_future_owner() -> None:
    assert callable(rebuild_data_products)
    assert callable(build_symbol_datasets)
    assert callable(build_master_ai_dataset)
    assert callable(validate_ai_dataset)
    assert callable(create_symbol_split)
    assert callable(create_master_split)


def test_page_actions_remain_explicit_and_test_has_no_execution_route() -> None:
    source = _source()

    assert "if train_clicked:" in source
    assert "if validation_clicked:" in source
    assert '"Save candidate"' in source
    assert 'load_rl_partition(\n            symbol,\n            "train"' in source
    assert 'load_rl_partition(\n            symbol,\n            "test"' not in source
    assert "test_rl.csv" not in source
    assert "FINAL TEST SET: SEALED" in source
