"""Offline tests for shared dashboard dataset loading and filtering."""

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from dashboard.data_loader import (
    filter_history_period,
    filter_market_data,
    filter_security_history,
    filter_stock_history_period,
    history_csv_bytes,
    load_dashboard_dataset,
    load_market_dataset,
    paginate_dataframe,
    resolve_pagination,
    sort_history_newest_first,
)


def _row(symbol: str, trading_date: str, close: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": trading_date,
        "ldcp": close - 1,
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "change": 1.0,
        "change_percent": 1.0,
        "volume": 100,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_empty_csv_directory_returns_an_empty_dataset(tmp_path: Path) -> None:
    result = load_market_dataset(tmp_path)

    assert result.file_count == 0
    assert result.csv_paths == ()
    assert result.errors == ()
    assert result.data.empty


def test_combines_multiple_csv_files(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "market_2026-07-27.csv",
        [_row("PPL", "2026-07-27", 180.0)],
    )
    _write_csv(
        tmp_path / "market_2026-07-24.csv",
        [_row("OGDC", "2026-07-24", 220.0)],
    )

    result = load_market_dataset(tmp_path)

    assert result.file_count == 2
    assert len(result.data) == 2
    assert result.data["symbol"].tolist() == ["OGDC", "PPL"]
    assert pd.api.types.is_datetime64_any_dtype(result.data["date"])


def test_duplicate_rows_are_not_removed(tmp_path: Path) -> None:
    duplicate = _row("OGDC", "2026-07-24", 220.0)
    _write_csv(tmp_path / "market_2026-07-24.csv", [duplicate])
    _write_csv(tmp_path / "market_2026-07-27.csv", [duplicate])

    result = load_market_dataset(tmp_path)

    assert len(result.data) == 2
    assert result.data.iloc[0].to_dict() == result.data.iloc[1].to_dict()


def test_filters_by_symbol_without_losing_numeric_looking_symbols() -> None:
    data = pd.DataFrame(
        [
            _row("786", "2026-07-24", 21.0),
            _row("OGDC", "2026-07-24", 220.0),
        ]
    )

    filtered = filter_market_data(data, symbol="786")

    assert filtered["symbol"].tolist() == ["786"]


def test_filters_dates_inclusively() -> None:
    data = pd.DataFrame(
        [
            _row("OGDC", "2026-07-24", 220.0),
            _row("OGDC", "2026-07-25", 221.0),
            _row("OGDC", "2026-07-27", 222.0),
        ]
    )

    filtered = filter_market_data(
        data,
        start_date=date(2026, 7, 24),
        end_date=date(2026, 7, 25),
    )

    assert filtered["date"].dt.date.tolist() == [
        date(2026, 7, 24),
        date(2026, 7, 25),
    ]


def test_rejects_a_reversed_date_filter() -> None:
    with pytest.raises(ValueError, match="end date cannot be earlier"):
        filter_market_data(
            pd.DataFrame(),
            start_date=date(2026, 7, 27),
            end_date=date(2026, 7, 24),
        )


def test_filter_results_are_sorted_chronologically() -> None:
    data = pd.DataFrame(
        [
            _row("OGDC", "2026-07-27", 222.0),
            _row("OGDC", "2026-07-24", 220.0),
            _row("OGDC", "2026-07-25", 221.0),
        ]
    )

    filtered = filter_market_data(data, symbol="OGDC")

    assert filtered["date"].dt.date.tolist() == [
        date(2026, 7, 24),
        date(2026, 7, 25),
        date(2026, 7, 27),
    ]


def test_dashboard_prefers_master_dataset_when_it_exists(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(
        raw_dir / "market_2026-07-24.csv",
        [_row("RAW", "2026-07-24", 100.0)],
    )
    master_path = tmp_path / "master" / "psx_master.csv"
    master_path.parent.mkdir()
    _write_csv(master_path, [_row("MASTER", "2026-07-25", 200.0)])

    result = load_dashboard_dataset(
        master_csv_path=master_path,
        raw_csv_dir=raw_dir,
    )

    assert result.source == "master"
    assert result.data["symbol"].tolist() == ["MASTER"]
    assert result.file_count == 1
    assert "Using persistent master dataset" in result.message


def test_dashboard_falls_back_to_raw_with_a_clear_message(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(
        raw_dir / "market_2026-07-24.csv",
        [_row("OGDC", "2026-07-24", 220.0)],
    )

    result = load_dashboard_dataset(
        master_csv_path=tmp_path / "missing-master.csv",
        raw_csv_dir=raw_dir,
    )

    assert result.source == "raw"
    assert result.data["symbol"].tolist() == ["OGDC"]
    assert "has not been built" in result.message


def test_stock_history_period_is_relative_to_latest_date_and_does_not_mutate() -> None:
    data = pd.DataFrame(
        [
            _row("OGDC", "2025-07-30", 100.0),
            _row("OGDC", "2026-01-29", 110.0),
            _row("OGDC", "2026-01-30", 120.0),
            _row("OGDC", "2026-07-30", 130.0),
        ]
    )
    original = data.copy(deep=True)

    six_months = filter_stock_history_period(data, "6M")
    all_history = filter_stock_history_period(data, "All")

    assert six_months["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-30",
        "2026-07-30",
    ]
    assert len(all_history) == 4
    pd.testing.assert_frame_equal(data, original)


def test_stock_history_period_rejects_an_unknown_option() -> None:
    with pytest.raises(ValueError, match="Unsupported stock history period"):
        filter_stock_history_period(pd.DataFrame(), "2W")


def test_security_history_excludes_every_other_company_and_is_newest_first() -> None:
    data = pd.DataFrame(
        [
            _row("MCB", "2026-07-28", 300.0),
            _row("OGDC", "2026-07-29", 220.0),
            _row("MCB", "2026-07-30", 302.0),
            _row("PPL", "2026-07-30", 180.0),
            _row("MCB", "2026-07-29", 301.0),
        ]
    )
    original = data.copy(deep=True)

    selected = filter_security_history(data, "MCB")
    newest_first = sort_history_newest_first(selected)

    assert selected["symbol"].unique().tolist() == ["MCB"]
    assert newest_first["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-30",
        "2026-07-29",
        "2026-07-28",
    ]
    pd.testing.assert_frame_equal(data, original)


def test_history_period_applies_an_explicit_reference_date_inclusively() -> None:
    data = pd.DataFrame(
        [
            _row("MCB", "2026-05-30", 300.0),
            _row("MCB", "2026-06-30", 301.0),
            _row("MCB", "2026-07-30", 302.0),
            _row("MCB", "2026-07-31", 303.0),
        ]
    )

    filtered = filter_history_period(
        data,
        "1M",
        reference_date=date(2026, 7, 30),
    )

    assert filtered["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-30",
        "2026-07-30",
    ]


def test_pagination_returns_the_requested_rows_and_final_partial_page() -> None:
    data = pd.DataFrame({"row": range(1, 121)})

    second_page = paginate_dataframe(data, page_number=2, rows_per_page=50)
    final_page = paginate_dataframe(data, page_number=3, rows_per_page=50)

    assert second_page["row"].tolist() == list(range(51, 101))
    assert final_page["row"].tolist() == list(range(101, 121))


def test_pagination_clamps_invalid_page_numbers_safely() -> None:
    data = pd.DataFrame({"row": range(1, 61)})

    below_range = paginate_dataframe(data, page_number=0, rows_per_page=25)
    above_range = paginate_dataframe(data, page_number=99, rows_per_page=25)
    non_integer = paginate_dataframe(
        data,
        page_number=None,  # type: ignore[arg-type]
        rows_per_page=25,
    )

    assert below_range["row"].tolist() == list(range(1, 26))
    assert above_range["row"].tolist() == list(range(51, 61))
    assert non_integer["row"].tolist() == list(range(1, 26))


def test_changing_rows_per_page_clamps_a_now_invalid_page() -> None:
    total_rows = 80

    before_change = resolve_pagination(total_rows, page_number=4, rows_per_page=25)
    after_change = resolve_pagination(total_rows, page_number=4, rows_per_page=50)

    assert before_change.page_number == 4
    assert before_change.start_row == 76
    assert before_change.end_row == 80
    assert after_change.page_number == 2
    assert after_change.start_row == 51
    assert after_change.end_row == 80


def test_history_csv_export_contains_all_filtered_rows_not_only_visible_page() -> None:
    rows = [
        _row("MCB", (date(2026, 1, 1) + timedelta(days=index)).isoformat(), 300.0)
        for index in range(80)
    ]
    rows.extend(
        _row("OGDC", (date(2026, 1, 1) + timedelta(days=index)).isoformat(), 200.0)
        for index in range(10)
    )
    source = pd.DataFrame(rows)
    selected = filter_security_history(source, "MCB")
    original = selected.copy(deep=True)
    visible_page = paginate_dataframe(selected, page_number=1, rows_per_page=25)

    exported = pd.read_csv(BytesIO(history_csv_bytes(selected)))

    assert len(visible_page) == 25
    assert len(exported) == 80
    assert exported["symbol"].unique().tolist() == ["MCB"]
    pd.testing.assert_frame_equal(selected, original)


def test_company_selection_is_unchanged_as_other_history_grows() -> None:
    target_rows = [
        _row("MCB", "2026-07-28", 300.0),
        _row("MCB", "2026-07-29", 301.0),
        _row("MCB", "2026-07-30", 302.0),
    ]
    small_dataset = pd.DataFrame(
        [*target_rows, _row("OGDC", "2026-07-30", 220.0)]
    )
    large_dataset = pd.DataFrame(
        [
            *target_rows,
            *(
                _row(
                    f"OTHER{index % 100}",
                    (date(2000, 1, 1) + timedelta(days=index)).isoformat(),
                    float(index),
                )
                for index in range(10_000)
            ),
        ]
    )

    small_selection = filter_security_history(small_dataset, "MCB")
    large_selection = filter_security_history(large_dataset, "MCB")

    pd.testing.assert_frame_equal(small_selection, large_selection)
