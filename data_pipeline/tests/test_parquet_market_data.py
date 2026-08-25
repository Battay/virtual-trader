"""Offline tests for the consolidated PSX Parquet read/audit boundary."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.config import PROJECT_ROOT, PSX_MARKET_PARQUET_ENV_VAR
from data_pipeline.src.parquet_market_data import (
    MarketParquetNotFoundError,
    MarketParquetSchemaError,
    audit_market_parquet,
    inspect_market_parquet_schema,
    load_market_calendar,
    load_market_data,
    load_market_date_range,
    main,
    resolve_market_parquet_path,
)


MARKET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32()),
        pa.field("symbol", pa.string()),
        pa.field("ldcp", pa.float64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("change_percent", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)


def _valid_rows() -> list[dict[str, object]]:
    return [
        {
            "market_date": date(2024, 1, 3),
            "symbol": "BBB",
            "ldcp": 19.0,
            "open": 20.0,
            "high": 22.0,
            "low": 19.5,
            "close": 21.0,
            "change": 2.0,
            "change_percent": 10.526,
            "volume": 200,
        },
        {
            "market_date": date(2024, 1, 2),
            "symbol": "BBB",
            "ldcp": 18.0,
            "open": 18.5,
            "high": 20.0,
            "low": 18.0,
            "close": 19.0,
            "change": 1.0,
            "change_percent": 5.556,
            "volume": 150,
        },
        {
            "market_date": date(2024, 1, 3),
            "symbol": "AAA",
            "ldcp": 10.5,
            "open": 10.7,
            "high": 11.2,
            "low": 10.4,
            "close": 11.0,
            "change": 0.5,
            "change_percent": 4.762,
            "volume": 100,
        },
        {
            "market_date": date(2024, 1, 4),
            "symbol": "AAA",
            "ldcp": 11.0,
            "open": 11.1,
            "high": 11.5,
            "low": 10.8,
            "close": 11.3,
            "change": 0.3,
            "change_percent": 2.727,
            "volume": 125,
        },
    ]


def _write_market(path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, schema=MARKET_SCHEMA, preserve_index=False)
    pq.write_table(table, path, row_group_size=2)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_valid_parquet_load_is_deterministically_ordered(tmp_path: Path) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    frame = load_market_data(path)

    assert list(zip(frame["market_date"], frame["symbol"], strict=True)) == [
        (date(2024, 1, 2), "BBB"),
        (date(2024, 1, 3), "AAA"),
        (date(2024, 1, 3), "BBB"),
        (date(2024, 1, 4), "AAA"),
    ]
    assert tuple(frame.columns) == tuple(field.name for field in MARKET_SCHEMA)
    assert inspect_market_parquet_schema(path).valid is True


def test_missing_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(MarketParquetNotFoundError, match="does not exist"):
        load_market_data(missing)
    with pytest.raises(MarketParquetNotFoundError, match="does not exist"):
        audit_market_parquet(missing)


def test_missing_required_column_is_reported_and_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing_close.parquet"
    table = pa.Table.from_pandas(pd.DataFrame(_valid_rows()), preserve_index=False)
    pq.write_table(table.drop(["close"]), path)

    validation = inspect_market_parquet_schema(path)
    audit = audit_market_parquet(path)

    assert validation.valid is False
    assert any("close" in error for error in validation.errors)
    assert audit.schema_valid is False
    assert audit.null_counts["close"] is None
    with pytest.raises(MarketParquetSchemaError, match="close"):
        load_market_data(path)


def test_inclusive_date_filter_uses_requested_bounds(tmp_path: Path) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    frame = load_market_date_range("2024-01-03", "2024-01-03", path=path)

    assert frame["market_date"].tolist() == [date(2024, 1, 3)] * 2
    assert frame["symbol"].tolist() == ["AAA", "BBB"]
    with pytest.raises(ValueError, match="cannot be after"):
        load_market_date_range("2024-01-04", "2024-01-03", path=path)


def test_market_calendar_reads_only_distinct_sorted_dates(tmp_path: Path) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    calendar = load_market_calendar(
        path, start_date="2024-01-03", end_date="2024-01-04"
    )

    assert calendar.tolist() == list(
        pd.to_datetime(["2024-01-03", "2024-01-04"])
    )


def test_symbol_filter_is_exact_and_order_independent(tmp_path: Path) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    frame = load_market_data(path, symbols=["BBB", "BBB"])

    assert set(frame["symbol"]) == {"BBB"}
    assert frame["market_date"].tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    with pytest.raises(TypeError, match="sequence"):
        load_market_data(path, symbols="BBB")  # type: ignore[arg-type]


def test_audit_detects_duplicates_nulls_and_numeric_quality_issues(
    tmp_path: Path,
) -> None:
    rows = _valid_rows()
    rows.append(dict(rows[0]))
    rows[1]["ldcp"] = None
    rows[2].update(
        {
            "open": 12.0,
            "high": 10.0,
            "low": 11.0,
            "close": 0.0,
            "volume": -5,
        }
    )
    path = _write_market(tmp_path / "quality.parquet", rows)

    audit = audit_market_parquet(path)

    assert audit.schema_valid is True
    assert audit.row_count == 5
    assert audit.unique_symbol_count == 2
    assert audit.unique_market_date_count == 3
    assert audit.earliest_market_date == "2024-01-02"
    assert audit.latest_market_date == "2024-01-04"
    assert audit.duplicate_market_date_symbol_count == 1
    assert audit.null_counts["ldcp"] == 1
    assert audit.total_required_null_count == 1
    assert audit.negative_volume_count == 1
    assert audit.non_positive_close_count == 1
    assert audit.zero_open_rows == 0
    assert audit.zero_high_rows == 0
    assert audit.zero_low_rows == 0
    assert audit.rows_with_any_zero_ohl == 0
    assert audit.invalid_ohlc_relationship_counts == {
        "high_below_low": 1,
        "high_below_open": 1,
        "high_below_close": 0,
        "low_above_open": 0,
        "low_above_close": 0,
        "invalid_ohlc_rows": 1,
    }


def test_zero_ohl_values_are_availability_not_relationship_failures(
    tmp_path: Path,
) -> None:
    rows = _valid_rows()
    rows[0]["high"] = 0.0
    rows[1]["low"] = 0.0
    rows[2]["open"] = 0.0
    path = _write_market(tmp_path / "zero_ohl.parquet", rows)

    audit = audit_market_parquet(path)

    assert audit.zero_open_rows == 1
    assert audit.zero_high_rows == 1
    assert audit.zero_low_rows == 1
    assert audit.rows_with_any_zero_ohl == 3
    assert audit.non_positive_close_count == 0
    assert audit.invalid_ohlc_relationship_counts == {
        "high_below_low": 0,
        "high_below_open": 0,
        "high_below_close": 0,
        "low_above_open": 0,
        "low_above_close": 0,
        "invalid_ohlc_rows": 0,
    }


def test_genuine_positive_ohlc_inconsistencies_remain_detected(
    tmp_path: Path,
) -> None:
    rows = _valid_rows()[:2]
    rows[0].update({"open": 12.0, "high": 10.0, "low": 11.0, "close": 12.5})
    rows[1].update({"open": 10.0, "high": 12.0, "low": 11.0, "close": 10.5})
    path = _write_market(tmp_path / "positive_inconsistencies.parquet", rows)

    audit = audit_market_parquet(path)

    assert audit.rows_with_any_zero_ohl == 0
    assert audit.invalid_ohlc_relationship_counts == {
        "high_below_low": 1,
        "high_below_open": 1,
        "high_below_close": 1,
        "low_above_open": 1,
        "low_above_close": 1,
        "invalid_ohlc_rows": 2,
    }


def test_reads_and_audit_leave_source_parquet_byte_identical(tmp_path: Path) -> None:
    path = _write_market(tmp_path / "immutable.parquet", _valid_rows())
    before = _sha256(path)

    load_market_data(path, start_date="2024-01-02", symbols=["AAA"])
    audit_market_parquet(path)

    assert path.is_file()
    assert _sha256(path) == before


def test_environment_path_is_portable_and_project_relative_when_needed() -> None:
    resolved = resolve_market_parquet_path(
        environ={PSX_MARKET_PARQUET_ENV_VAR: "external/market.parquet"}
    )

    assert resolved == (PROJECT_ROOT / "external/market.parquet").resolve()
    assert "/Users/" not in str(resolve_market_parquet_path.__doc__)


def test_cli_audit_prints_concise_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    exit_code = main(["--audit", "--path", str(path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Rows: 4" in output
    assert "Schema valid: True" in output
    assert "Required-column nulls: 0" in output


def test_cli_supports_short_date_range_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_market(tmp_path / "market.parquet", _valid_rows())

    exit_code = main(
        [
            "--path",
            str(path),
            "--start",
            "2024-01-03",
            "--end",
            "2024-01-03",
            "--symbol",
            "AAA",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Rows: 1" in output
    assert "Date range: 2024-01-03 to 2024-01-03" in output
