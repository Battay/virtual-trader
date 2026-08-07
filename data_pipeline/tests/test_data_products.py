"""Offline tests for ordered post-backfill data-product rebuilding."""

from types import SimpleNamespace

import pandas as pd

from data_pipeline.src.data_products import SplitRefreshResult, rebuild_data_products


def test_post_backfill_products_run_in_required_order() -> None:
    calls: list[str] = []

    def master_builder():
        calls.append("master")
        return SimpleNamespace(total_rows=1_000, unique_symbols=20, errors=())

    def registry_builder():
        calls.append("registry")
        return object()

    def master_ai_builder():
        calls.append("master_ai")
        return SimpleNamespace(output_rows=800, invalid_ohlc_rows_removed=17)

    def symbol_ai_builder(*, minimum_usable_rows: int):
        calls.append("symbol_ai")
        assert minimum_usable_rows == 252
        return SimpleNamespace(unique_symbols=12)

    def split_refresher():
        calls.append("splits")
        return SplitRefreshResult(13, ())

    def readiness_builder(minimum_usable_rows: int):
        calls.append("readiness")
        return pd.DataFrame(
            {"readiness_status": ["Ready", "Ready", "Insufficient History"]}
        )

    result = rebuild_data_products(
        raw_dates_added=10,
        master_builder=master_builder,
        registry_builder=registry_builder,
        master_ai_builder=master_ai_builder,
        symbol_ai_builder=symbol_ai_builder,
        split_refresher=split_refresher,
        readiness_builder=readiness_builder,
    )

    assert calls == [
        "master",
        "registry",
        "master_ai",
        "symbol_ai",
        "splits",
        "readiness",
    ]
    assert result.raw_dates_added == 10
    assert result.master_rows == 1_000
    assert result.master_symbols == 20
    assert result.processed_rows == 800
    assert result.processed_symbols == 12
    assert result.symbols_ready_for_training == 2
    assert result.insufficient_history_symbols == 1
    assert result.split_sets == 13
    assert result.invalid_ohlc_rows_removed == 17
    assert result.errors == ()


def test_post_backfill_failure_is_reported_and_later_stages_continue() -> None:
    calls: list[str] = []

    def fail_master():
        calls.append("master")
        raise OSError("disk unavailable")

    result = rebuild_data_products(
        master_builder=fail_master,
        registry_builder=lambda: calls.append("registry"),
        master_ai_builder=lambda: SimpleNamespace(output_rows=0),
        symbol_ai_builder=lambda **kwargs: SimpleNamespace(unique_symbols=0),
        split_refresher=lambda: SplitRefreshResult(0, ()),
        readiness_builder=lambda minimum: pd.DataFrame(
            {"readiness_status": ["Insufficient History"]}
        ),
    )

    assert calls == ["master", "registry"]
    assert "Master dataset rebuild failed: disk unavailable" in result.errors
    assert result.insufficient_history_symbols == 1
