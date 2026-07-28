"""Offline tests for one-date and date-range CLI orchestration."""

from datetime import date
import logging
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.src import main as main_module
from data_pipeline.src.main import (
    DateProcessingResult,
    iter_calendar_dates,
    process_date,
    run_date_range,
)


class StubClient:
    """Return configured HTML without making network requests."""

    def __init__(self, html: str = "<table></table>") -> None:
        self.html = html

    def fetch_market_by_date(self, trading_date: date) -> str:
        return self.html


def _successful_result(trading_date: date) -> DateProcessingResult:
    return DateProcessingResult(
        trading_date=trading_date,
        status="successful",
        parsed_rows=1,
        valid_rows=1,
        rejected_rows=0,
        output_path=Path(f"market_{trading_date.isoformat()}.csv"),
    )


def test_iter_calendar_dates_is_inclusive() -> None:
    assert list(iter_calendar_dates(date(2026, 7, 30), date(2026, 8, 2))) == [
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 1),
        date(2026, 8, 2),
    ]


def test_reversed_range_is_rejected_by_iterator_and_cli() -> None:
    with pytest.raises(ValueError, match="end date cannot be earlier"):
        list(iter_calendar_dates(date(2026, 7, 2), date(2026, 7, 1)))

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(
            ["--start-date", "2026-07-02", "--end-date", "2026-07-01"]
        )

    assert exc_info.value.code == 2


def test_empty_trading_date_is_skipped_without_valid_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    rejected_dir = tmp_path / "rejected"
    monkeypatch.setattr(main_module, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(main_module, "REJECTED_DATA_DIR", rejected_dir)

    result = process_date(date(2026, 7, 4), StubClient())

    assert result.status == "skipped"
    assert (raw_dir / "market_2026-07-04.html").exists()
    assert not (raw_dir / "market_2026-07-04.csv").exists()


def test_range_logs_each_skipped_date_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(main_module, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(main_module, "REJECTED_DATA_DIR", tmp_path / "rejected")

    with caplog.at_level(logging.INFO, logger=main_module.LOGGER.name):
        summary = run_date_range(
            date(2026, 7, 25),
            date(2026, 7, 25),
            client=StubClient(),
        )

    skip_messages = [
        record.getMessage()
        for record in caplog.records
        if "2026-07-25" in record.getMessage()
        and "equity rows" in record.getMessage()
    ]
    assert skip_messages == [
        "Skipping 2026-07-25: response contains no equity rows"
    ]
    assert summary.skipped_dates == 1


def test_failed_date_does_not_stop_later_dates() -> None:
    processed: list[date] = []

    def fake_processor(
        trading_date: date, client: Any
    ) -> DateProcessingResult:
        processed.append(trading_date)
        if trading_date == date(2026, 7, 2):
            raise OSError("temporary file error")
        return _successful_result(trading_date)

    summary = run_date_range(
        date(2026, 7, 1),
        date(2026, 7, 3),
        client=StubClient(),
        date_processor=fake_processor,
    )

    assert processed == [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]
    assert summary.total_dates == 3
    assert summary.successful_dates == 2
    assert summary.skipped_dates == 0
    assert summary.failed_dates == (
        (date(2026, 7, 2), "OSError: temporary file error"),
    )


def test_existing_single_date_cli_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_dates: list[date] = []

    def fake_run(trading_date: date) -> DateProcessingResult:
        requested_dates.append(trading_date)
        return _successful_result(trading_date)

    monkeypatch.setattr(main_module, "run", fake_run)

    assert main_module.main(["--date", "2026-07-27"]) == 0
    assert requested_dates == [date(2026, 7, 27)]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--start-date", "2026-07-01"],
        ["--end-date", "2026-07-31"],
        [
            "--date",
            "2026-07-27",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-31",
        ],
    ],
)
def test_cli_rejects_invalid_date_option_combinations(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main_module.main(arguments)

    assert exc_info.value.code == 2
