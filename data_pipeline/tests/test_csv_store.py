"""Offline tests for deterministic master-dataset rebuilding."""

import os
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.src.csv_store import build_master_dataset


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


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_master_rebuild_is_deterministic_and_sorted(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_path = tmp_path / "master" / "psx_master.csv"
    _write(
        raw_dir / "market_2026-07-02.csv",
        [_row("PPL", "2026-07-02", 180.0)],
    )
    _write(
        raw_dir / "market_2026-07-01.csv",
        [
            _row("UBL", "2026-07-01", 300.0),
            _row("OGDC", "2026-07-01", 220.0),
        ],
    )

    first = build_master_dataset(raw_csv_dir=raw_dir, output_path=output_path)
    first_bytes = output_path.read_bytes()
    second = build_master_dataset(raw_csv_dir=raw_dir, output_path=output_path)

    master = pd.read_csv(output_path)
    assert output_path.read_bytes() == first_bytes
    assert master[["date", "symbol"]].values.tolist() == [
        ["2026-07-01", "OGDC"],
        ["2026-07-01", "UBL"],
        ["2026-07-02", "PPL"],
    ]
    assert first == second
    assert first.total_rows == 3
    assert first.unique_symbols == 3
    assert first.duplicate_count == 0


def test_exact_duplicate_business_keys_are_removed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    duplicate = _row("OGDC", "2026-07-01", 220.0)
    _write(raw_dir / "market_2026-07-01.csv", [duplicate])
    _write(raw_dir / "market_2026-07-02.csv", [duplicate])

    result = build_master_dataset(
        raw_csv_dir=raw_dir,
        output_path=tmp_path / "master.csv",
    )

    assert result.total_rows == 1
    assert result.duplicate_count == 1


def test_conflicting_duplicate_keeps_newest_raw_file_version(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    older_path = raw_dir / "market_2026-07-01.csv"
    newer_path = raw_dir / "market_2026-07-02.csv"
    _write(
        older_path,
        [_row("OGDC", "2026-07-01", 220.0)],
    )
    _write(
        newer_path,
        [_row("OGDC", "2026-07-01", 225.0)],
    )
    os.utime(older_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer_path, ns=(2_000_000_000, 2_000_000_000))
    output_path = tmp_path / "master.csv"

    with caplog.at_level("WARNING"):
        result = build_master_dataset(
            raw_csv_dir=raw_dir,
            output_path=output_path,
        )

    assert pd.read_csv(output_path).iloc[0]["close"] == 225.0
    assert result.duplicate_count == 1
    assert "Conflicting rows for OGDC on 2026-07-01" in caplog.text
