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


def test_training_page_uses_frozen_recurrent_production_authority() -> None:
    source = _source()

    assert "production_plan()" in source
    assert "prepare_production_run()" in source
    assert "launch_production_controller(snapshot.store)" in source
    assert "TRAINABLE_MEMBERS_OF_FROZEN_RESEARCH_UNIVERSE_V1" not in source
    assert "total_timesteps =" not in source
    assert "RecurrentPPOConfig(" not in source
    assert "prepare_selected_run(confirmed_symbols)" in source
    assert "build_global_model_coverage()" in source
    assert "SELECTED" in source


def test_training_page_has_native_top_spacing_and_unambiguous_frozen_labels() -> None:
    source = _source()

    assert source.index('st.space("medium")') < source.index(
        'st.title("Training & Models")'
    )
    required_labels = (
        "Frozen research universe",
        "Frozen snapshot date",
        "Research identities",
        "Frozen universe version",
        "Frozen universe hash",
        "Execution-training policy",
        "Trainable agents",
        "Trainable symbol hash",
        "Underlying identity contract",
    )
    assert not [label for label in required_labels if label not in source]
    assert "The current operational identity universe is outside this run." in source


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


def test_page_actions_are_explicit_detached_and_test_has_no_execution_route() -> None:
    source = _source()

    assert '"Confirm start/continue"' in source
    assert '"Confirm stop after current jobs"' in source
    assert '"Confirm active interruption"' in source
    assert "I confirm this exact SELECTED membership" in source
    assert "Retrain selected trained symbols" in source
    assert "run_training_action" not in source
    assert "train_recurrent_single_symbol" not in source
    assert "load_rl_partition" not in source
    assert "test_rl.csv" not in source
    assert "TEST remains sealed" in source
