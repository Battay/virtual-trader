"""Lightweight acceptance guards for the registered Streamlit application."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app.py"


def _registered_page_paths() -> tuple[Path, ...]:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    pages: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "st"
            and function.attr == "Page"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        pages.append(PROJECT_ROOT / node.args[0].value)
    return tuple(pages)


def test_every_streamlit_page_is_registered_once_and_compiles() -> None:
    registered = _registered_page_paths()
    discovered = tuple(sorted((PROJECT_ROOT / "app_pages").glob("*.py")))

    assert registered
    assert len(registered) == len(set(registered))
    assert set(registered) == set(discovered)
    for page in registered:
        source = page.read_text(encoding="utf-8")
        compile(source, str(page), "exec")


def test_pages_have_no_runtime_dependency_on_sibling_data_producer() -> None:
    application_sources = (APP_PATH, *_registered_page_paths())

    for path in application_sources:
        source = path.read_text(encoding="utf-8")
        assert "psx-data-sync" not in source
        assert "/Users/" not in source


def test_data_maintenance_buttons_call_canonical_native_workflows() -> None:
    automation_source = (
        PROJECT_ROOT / "app_pages" / "4_Automation.py"
    ).read_text(encoding="utf-8")
    backfill_source = (
        PROJECT_ROOT / "app_pages" / "7_Historical_Backfill.py"
    ).read_text(encoding="utf-8")

    assert "Rebuild canonical market artifacts" in automation_source
    assert "rebuild_canonical_market_artifacts" in automation_source
    assert "build_master_dataset" not in automation_source
    assert "reconcile_native_source_csvs" in backfill_source

