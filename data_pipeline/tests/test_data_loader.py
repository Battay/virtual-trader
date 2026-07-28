"""Offline tests for shared dashboard dataset loading and filtering."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from dashboard.data_loader import filter_market_data, load_market_dataset


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
