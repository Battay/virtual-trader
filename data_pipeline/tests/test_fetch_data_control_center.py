"""Streamlit acceptance coverage for the Fetch Data control center."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from data_pipeline.src.data_completeness import (
    CsvDateRecord,
    CsvInventorySummary,
    DataCompletenessInventory,
    DateClassification,
    MasterArtifactStatus,
    MasterParityStatus,
    ParquetDateRecord,
    ParquetDateState,
    ParquetInventorySummary,
)
from data_pipeline.src.maintenance_history import MaintenanceHistory


PAGE_PATH = Path(__file__).resolve().parents[2] / "app_pages/1_Fetch_Data.py"


def _inventory(tmp_path: Path) -> DataCompletenessInventory:
    missing = CsvDateRecord(
        date(2024, 1, 8),
        DateClassification.MISSING,
        "MISSING",
        "No retained HTML",
        0,
        "not_attempted",
        None,
        True,
        "Missing source",
    )
    anomaly = CsvDateRecord(
        date(2024, 1, 9),
        DateClassification.SOURCE_ANOMALY,
        "MISSING",
        "1 retained HTML response(s)",
        1,
        "source_anomaly",
        "2024-01-09T00:00:00+00:00",
        False,
        "Review only",
    )
    missing_partition = ParquetDateRecord(
        date(2024, 1, 2),
        ParquetDateState.MISSING,
        None,
        None,
        1,
        "expected",
        None,
        "Missing partition",
    )
    master = MasterArtifactStatus(
        tmp_path / "master.csv",
        True,
        date(2024, 1, 2),
        1,
        1,
        1,
        "PASS",
        "hash",
    )
    parquet = MasterArtifactStatus(
        tmp_path / "market.parquet",
        True,
        date(2024, 1, 2),
        1,
        1,
        1,
        "PASS",
        "hash",
    )
    return DataCompletenessInventory(
        generated_at="2024-01-10T00:00:00+00:00",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 10),
        csv_records=(missing, anomaly),
        csv_summary=CsvInventorySummary(
            date(2024, 1, 2), date(2024, 1, 2), 1, 1, 0, 0, 1, 0, 0
        ),
        parquet_records=(missing_partition,),
        parquet_summary=ParquetInventorySummary(
            date(2024, 1, 2), date(2024, 1, 2), 0, 1, 0, 0, 0
        ),
        master_csv=master,
        master_parquet=parquet,
        master_parity=MasterParityStatus(True, True, "PASS", "PASS"),
        pending_source_dates=(),
        canonical_dates_with_noncurrent_daily=(date(2024, 1, 2),),
        history=MaintenanceHistory(),
    )


def test_control_center_has_no_default_bulk_selection_and_requires_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "data_pipeline.src.data_completeness.build_data_completeness_inventory",
        lambda: _inventory(tmp_path),
    )
    app = AppTest.from_file(str(PAGE_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Fetch Data"
    labels = [item.label for item in app.button]
    assert "Select visible actionable" in labels
    assert "Fetch selected dates" in labels
    assert "Repair selected Parquet dates" in labels
    fetch_button = next(item for item in app.button if item.label == "Fetch selected dates")
    assert fetch_button.disabled
    assert tuple(app.session_state["fetch_control_csv_selected_dates"]) == ()
    assert tuple(app.session_state["fetch_control_parquet_selected_dates"]) == ()
    assert len(app.dataframe) == 2


def test_select_visible_adds_only_actionable_source_date(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "data_pipeline.src.data_completeness.build_data_completeness_inventory",
        lambda: _inventory(tmp_path),
    )
    app = AppTest.from_file(str(PAGE_PATH), default_timeout=30).run()
    button = next(item for item in app.button if item.label == "Select visible actionable")
    app = button.click().run()

    assert not app.exception
    assert tuple(app.session_state["fetch_control_csv_selected_dates"]) == (
        date(2024, 1, 8),
    )
    fetch_button = next(item for item in app.button if item.label == "Fetch selected dates")
    assert fetch_button.disabled


def test_page_uses_read_only_row_selection_and_preserves_manual_fetch() -> None:
    source = PAGE_PATH.read_text(encoding="utf-8")
    assert "st.dataframe(" in source
    assert 'selection_mode="multi-row"' in source
    assert "st.data_editor(" not in source
    assert "Existing manual fetch controls" in source
    assert "collect_single_date" in source
    assert "collect_date_range" in source
    assert "I confirm this will make bounded PSX network requests" in source
