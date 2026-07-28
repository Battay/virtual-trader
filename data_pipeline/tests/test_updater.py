"""Offline tests for incremental daily-date discovery and collection."""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data_pipeline.src.main import DateProcessingResult
from data_pipeline.src.updater import (
    BootstrapStartDateRequired,
    determine_missing_dates,
    discover_available_raw_dates,
    run_incremental_update,
)


def _row(trading_date: date) -> dict[str, object]:
    return {
        "symbol": "OGDC",
        "date": trading_date.isoformat(),
        "ldcp": 219.0,
        "open": 220.0,
        "high": 225.0,
        "low": 218.0,
        "close": 224.0,
        "change": 5.0,
        "change_percent": 2.28,
        "volume": 1000,
    }


def _write_daily_csv(directory: Path, trading_date: date) -> Path:
    path = directory / f"market_{trading_date.isoformat()}.csv"
    pd.DataFrame([_row(trading_date)]).to_csv(path, index=False)
    return path


class StubClient:
    """A no-network client used only to satisfy the processing protocol."""

    def fetch_market_by_date(self, trading_date: date) -> str:
        raise AssertionError("The fake date processor should handle this request")


def test_discovers_only_valid_available_raw_dates(tmp_path: Path) -> None:
    _write_daily_csv(tmp_path, date(2026, 7, 1))
    pd.DataFrame(columns=_row(date(2026, 7, 2))).to_csv(
        tmp_path / "market_2026-07-02.csv",
        index=False,
    )
    (tmp_path / "notes.csv").write_text("not,a,daily,file\n", encoding="utf-8")

    assert discover_available_raw_dates(tmp_path) == (date(2026, 7, 1),)


def test_identifies_missing_calendar_dates_inclusively() -> None:
    missing = determine_missing_dates(
        {date(2026, 7, 1), date(2026, 7, 3)},
        date(2026, 7, 4),
        date(2026, 7, 1),
    )

    assert missing == (date(2026, 7, 2), date(2026, 7, 4))


def test_empty_store_requires_an_explicit_bootstrap_date() -> None:
    with pytest.raises(BootstrapStartDateRequired, match="bootstrap start date"):
        determine_missing_dates((), date(2026, 7, 3))

    assert determine_missing_dates(
        (),
        date(2026, 7, 3),
        date(2026, 7, 1),
    ) == (date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3))


def test_incremental_update_does_not_refetch_existing_csv_dates(
    tmp_path: Path,
) -> None:
    _write_daily_csv(tmp_path, date(2026, 7, 1))
    processed: list[date] = []

    def fake_processor(
        trading_date: date,
        client: Any,
    ) -> DateProcessingResult:
        processed.append(trading_date)
        return DateProcessingResult(
            trading_date=trading_date,
            status="successful",
            parsed_rows=1,
            valid_rows=1,
            rejected_rows=0,
            output_path=tmp_path / f"market_{trading_date.isoformat()}.csv",
        )

    result = run_incremental_update(
        date(2026, 7, 3),
        bootstrap_start_date=date(2026, 7, 1),
        csv_dir=tmp_path,
        client=StubClient(),
        date_processor=fake_processor,
    )

    assert processed == [date(2026, 7, 2), date(2026, 7, 3)]
    assert result.missing_dates == (date(2026, 7, 2), date(2026, 7, 3))
    assert result.available_dates_before == (date(2026, 7, 1),)


def test_a_skipped_date_remains_eligible_for_a_later_retry(tmp_path: Path) -> None:
    processed: list[date] = []

    def skipped_processor(
        trading_date: date,
        client: Any,
    ) -> DateProcessingResult:
        processed.append(trading_date)
        return DateProcessingResult(
            trading_date=trading_date,
            status="skipped",
            parsed_rows=0,
            valid_rows=0,
            rejected_rows=0,
            output_path=None,
        )

    for _ in range(2):
        result = run_incremental_update(
            date(2026, 7, 4),
            bootstrap_start_date=date(2026, 7, 4),
            csv_dir=tmp_path,
            client=StubClient(),
            date_processor=skipped_processor,
        )
        assert result.skipped_dates == (date(2026, 7, 4),)

    assert processed == [date(2026, 7, 4), date(2026, 7, 4)]

