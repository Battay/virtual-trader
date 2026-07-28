"""Offline tests for Streamlit CSV preview integration."""

from pathlib import Path

import pandas as pd

from dashboard.data_loader import load_csv_preview


def test_preview_loads_only_returned_csv_paths(tmp_path: Path) -> None:
    first_path = tmp_path / "market_2026-07-24.csv"
    second_path = tmp_path / "market_2026-07-27.csv"
    excluded_path = tmp_path / "market_2026-07-28.csv"
    pd.DataFrame([{"symbol": "OGDC", "date": "2026-07-24"}]).to_csv(
        first_path, index=False
    )
    pd.DataFrame([{"symbol": "PPL", "date": "2026-07-27"}]).to_csv(
        second_path, index=False
    )
    pd.DataFrame([{"symbol": "EXCLUDED", "date": "2026-07-28"}]).to_csv(
        excluded_path, index=False
    )

    preview, errors = load_csv_preview((first_path, second_path))

    assert errors == ()
    assert preview is not None
    assert preview["symbol"].tolist() == ["OGDC", "PPL"]
    assert "EXCLUDED" not in preview["symbol"].tolist()


def test_preview_reports_a_missing_returned_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "market_2026-07-24.csv"

    preview, errors = load_csv_preview((missing_path,))

    assert preview is None
    assert errors == (f"Returned CSV path does not exist: {missing_path}",)
