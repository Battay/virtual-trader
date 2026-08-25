"""Offline tests for the descriptive consolidated-Parquet universe audit."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.universe_audit import (
    UNIVERSE_AUDIT_COLUMNS,
    UniverseAuditError,
    build_symbol_universe_audit,
    run_universe_audit,
    summarize_symbol_universe,
    write_universe_audit_csv,
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


def _market_rows() -> list[dict[str, object]]:
    def row(
        symbol: str,
        market_date: date,
        *,
        volume: int,
        open_: float = 10.0,
        high: float = 11.0,
        low: float = 9.0,
        close: float = 10.5,
    ) -> dict[str, object]:
        return {
            "market_date": market_date,
            "symbol": symbol,
            "ldcp": 10.0,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "change": close - 10.0,
            "change_percent": (close - 10.0) * 10,
            "volume": volume,
        }

    return [
        row("BBB", date(2024, 1, 4), volume=300),
        row("AAA", date(2024, 1, 2), volume=10, open_=0.0),
        row("CCC", date(2024, 1, 3), volume=50),
        row("AAA", date(2024, 1, 1), volume=0, close=0.0),
        row("BBB", date(2024, 1, 1), volume=100, high=0.0, low=0.0),
        row("AAA", date(2024, 1, 4), volume=20),
    ]


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(_market_rows())


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "BBB", "DDD"], dtype="string"),
            "company_name": pd.Series(
                ["Alpha Limited", "Beta Limited", "Delta Limited"], dtype="string"
            ),
            "sector": pd.Series(["Commercial Banks", pd.NA, "Other"], dtype="string"),
        }
    )


def _write_market(path: Path) -> Path:
    table = pa.Table.from_pandas(
        pd.DataFrame(_market_rows()), schema=MARKET_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def _write_registry(path: Path) -> Path:
    _metadata().to_csv(path, index=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_symbol_dates_observations_coverage_volume_and_ratios() -> None:
    universe = build_symbol_universe_audit(_market_frame())

    aaa = universe.set_index("symbol").loc["AAA"]
    assert aaa["first_market_date"] == date(2024, 1, 1)
    assert aaa["last_market_date"] == date(2024, 1, 4)
    assert aaa["observation_count"] == 3
    assert aaa["calendar_span_days"] == 3
    assert aaa["unique_market_dates"] == 3
    assert aaa["coverage_ratio"] == pytest.approx(0.75)
    assert aaa["average_volume"] == pytest.approx(10.0)
    assert aaa["median_volume"] == pytest.approx(10.0)
    assert aaa["zero_volume_count"] == 1
    assert aaa["zero_volume_ratio"] == pytest.approx(1 / 3)
    assert aaa["zero_open_count"] == 1
    assert aaa["zero_high_count"] == 0
    assert aaa["zero_low_count"] == 0
    assert aaa["rows_with_any_zero_ohl"] == 1
    assert aaa["zero_ohl_ratio"] == pytest.approx(1 / 3)
    assert aaa["positive_close_count"] == 2
    assert aaa["positive_close_ratio"] == pytest.approx(2 / 3)


def test_zero_ohl_counts_each_row_once_and_output_is_symbol_sorted() -> None:
    market = _market_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)

    universe = build_symbol_universe_audit(market)

    assert universe["symbol"].tolist() == ["AAA", "BBB", "CCC"]
    bbb = universe.set_index("symbol").loc["BBB"]
    assert bbb["zero_high_count"] == 1
    assert bbb["zero_low_count"] == 1
    assert bbb["rows_with_any_zero_ohl"] == 1
    assert bbb["zero_ohl_ratio"] == pytest.approx(0.5)


def test_metadata_join_preserves_unknowns_without_inventing_values() -> None:
    market = _market_frame()
    metadata = _metadata()
    market_before = market.copy(deep=True)
    metadata_before = metadata.copy(deep=True)

    universe = build_symbol_universe_audit(market, metadata=metadata).set_index("symbol")

    assert universe.loc["AAA", "company_name"] == "Alpha Limited"
    assert universe.loc["AAA", "sector"] == "Commercial Banks"
    assert bool(universe.loc["AAA", "metadata_matched"]) is True
    assert universe.loc["BBB", "company_name"] == "Beta Limited"
    assert pd.isna(universe.loc["BBB", "sector"])
    assert bool(universe.loc["BBB", "metadata_matched"]) is True
    assert pd.isna(universe.loc["CCC", "company_name"])
    assert pd.isna(universe.loc["CCC", "sector"])
    assert bool(universe.loc["CCC", "metadata_matched"]) is False
    pd.testing.assert_frame_equal(market, market_before)
    pd.testing.assert_frame_equal(metadata, metadata_before)


def test_duplicate_metadata_symbols_fail_before_join() -> None:
    duplicated = pd.concat([_metadata(), _metadata().iloc[[0]]], ignore_index=True)

    with pytest.raises(UniverseAuditError, match="unique"):
        build_symbol_universe_audit(_market_frame(), metadata=duplicated)


def test_dataset_summary_quantiles_metadata_and_rankings() -> None:
    universe = build_symbol_universe_audit(_market_frame(), metadata=_metadata())

    summary = summarize_symbol_universe(universe)

    assert summary.total_historical_symbols == 3
    assert summary.global_unique_market_dates == 4
    assert summary.global_first_market_date == "2024-01-01"
    assert summary.global_last_market_date == "2024-01-04"
    assert summary.history_length_counts == {
        "at_least_1_year": 0,
        "at_least_2_years": 0,
        "at_least_3_years": 0,
        "at_least_5_years": 0,
    }
    assert summary.observation_count_quantiles["p50"] == pytest.approx(2.0)
    assert summary.coverage_ratio_quantiles["p50"] == pytest.approx(0.5)
    assert summary.median_volume_quantiles["p50"] == pytest.approx(50.0)
    assert summary.zero_ohl_ratio_quantiles["p50"] == pytest.approx(1 / 3)
    assert summary.symbols_with_sector_metadata == 1
    assert summary.symbols_without_sector_metadata == 2
    assert summary.symbols_matched_to_registry == 2
    assert summary.symbols_unmatched_to_registry == 1
    assert [row["symbol"] for row in summary.top_symbols_by_observation_count] == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert [
        row["symbol"] for row in summary.bottom_nontrivial_symbols_by_observation_count
    ] == ["BBB", "AAA"]


def test_real_boundary_run_on_temporary_files_leaves_parquet_unchanged(
    tmp_path: Path,
) -> None:
    parquet = _write_market(tmp_path / "market.parquet")
    registry = _write_registry(tmp_path / "company_registry.csv")
    before = _sha256(parquet)

    result = run_universe_audit(parquet_path=parquet, registry_path=registry)

    assert tuple(result.symbols.columns) == UNIVERSE_AUDIT_COLUMNS
    assert result.summary.total_historical_symbols == 3
    assert _sha256(parquet) == before


def test_deterministic_csv_requires_explicit_overwrite(tmp_path: Path) -> None:
    universe = build_symbol_universe_audit(_market_frame(), metadata=_metadata())
    output = tmp_path / "universe.csv"

    written = write_universe_audit_csv(universe, output)
    first_bytes = written.read_bytes()
    with pytest.raises(UniverseAuditError, match="--overwrite"):
        write_universe_audit_csv(universe, output)
    write_universe_audit_csv(universe, output, overwrite=True)

    assert output.read_bytes() == first_bytes

