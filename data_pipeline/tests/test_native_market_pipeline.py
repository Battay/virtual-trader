"""Offline regression tests for virtual-trader's native market pipeline."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data_pipeline.src import parquet_market_data
from data_pipeline.src.native_market_pipeline import (
    BUSINESS_KEY,
    CANONICAL_MARKET_COLUMNS,
    NATIVE_MARKET_SCHEMA_VERSION,
    SECTOR_CONTEXT,
    DuplicateMarketRecordError,
    NativeMarketPaths,
    NativeMarketPipelineError,
    canonical_content_hash,
    compare_parquet_core,
    enrich_current_sector,
    full_rebuild,
    incremental_update,
    normalize_market_records,
    rebuild_generated_artifacts,
    sha256_file,
)
from data_pipeline.src.parquet_market_data import load_market_data


def _row(day: str, symbol: str, close: float, volume: int = 100) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": day,
        "ldcp": close - 1.0,
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "change": 1.0,
        "change_percent": 1.0,
        "volume": volume,
    }


def _write_source(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_listings(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "symbol": "AAA",
            "company_name": "Alpha Limited",
            "security_type": "ordinary_equity",
            "sector": "COMMERCIAL BANKS",
            "board": "Main",
            "listing_segment": "normal_counter",
            "clearing_type": "NC",
            "listed_in": "ALLSHR",
            "shares": 1,
            "free_float": 1,
            "officially_listed": True,
            "official_status": "listed",
            "non_compliance_reason": "",
            "source": "https://dps.psx.com.pk/listings-table/main/nc",
            "listing_refreshed_at": "2026-08-02T00:00:00+05:00",
            "snapshot_date": "2026-08-02",
        },
        {
            "symbol": "FUND",
            "company_name": "A Fund",
            # The listing source can label funds with an ordinary-equity-like
            # type; authoritative sector evidence still prevents promotion to
            # the frozen common-equity identity policy.
            "security_type": "ordinary_equity",
            "sector": "CLOSED-END FUNDS",
            "board": "Main",
            "listing_segment": "normal_counter",
            "clearing_type": "NC",
            "listed_in": "ALLSHR",
            "shares": 1,
            "free_float": 1,
            "officially_listed": True,
            "official_status": "listed",
            "non_compliance_reason": "",
            "source": "https://dps.psx.com.pk/listings-table/main/nc",
            "listing_refreshed_at": "2026-08-02T00:00:00+05:00",
            "snapshot_date": "2026-08-02",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _paths(root: Path) -> NativeMarketPaths:
    return NativeMarketPaths(
        master_csv=root / "master/psx_master.csv",
        symbol_csv_dir=root / "processed/market_symbols",
        daily_parquet_dir=root / "parquet/daily",
        consolidated_parquet=root / "parquet/market.parquet",
        state_json=root / "metadata/native_market_pipeline_state.json",
    )


def _fixture_sources(root: Path) -> tuple[Path, Path]:
    first = _write_source(
        root / "market_2024-01-02.csv",
        [_row("2024-01-02", "BBB", 20.0), _row("2024-01-02", "AAA", 10.0)],
    )
    second = _write_source(
        root / "market_2024-01-03.csv",
        [_row("2024-01-03", "AAA", 11.0), _row("2024-01-03", "FUND", 30.0)],
    )
    return first, second


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalization_enforces_canonical_schema_and_order() -> None:
    raw = pd.DataFrame(
        [_row("2024-01-02", " bbb ", 20.0), _row("2024-01-01", "aaa", 10.0)]
    )

    result = normalize_market_records(raw)

    assert tuple(result.columns) == CANONICAL_MARKET_COLUMNS[:10]
    assert list(zip(result["market_date"], result["symbol"], strict=True)) == [
        (pd.Timestamp("2024-01-01"), "AAA"),
        (pd.Timestamp("2024-01-02"), "BBB"),
    ]
    assert str(result["volume"].dtype) == "int64"
    assert str(result["close"].dtype) == "float64"


def test_sector_enrichment_is_authoritative_current_context_only(
    tmp_path: Path,
) -> None:
    listings = pd.read_csv(_write_listings(tmp_path / "listings.csv"))
    normalized = normalize_market_records(
        pd.DataFrame(
            [
                _row("2024-01-02", "AAA", 10.0),
                _row("2024-01-02", "FUND", 20.0),
                _row("2024-01-02", "OLD", 30.0),
            ]
        )
    )

    enriched, matched = enrich_current_sector(normalized, listings)

    assert matched == 1
    assert enriched.loc[enriched["symbol"] == "AAA", "sector_current"].item() == "COMMERCIAL BANKS"
    assert enriched.loc[enriched["symbol"] == "AAA", "sector_snapshot_date"].item() == "2026-08-02"
    assert pd.isna(enriched.loc[enriched["symbol"] == "FUND", "sector_current"].item())
    assert pd.isna(enriched.loc[enriched["symbol"] == "OLD", "sector_current"].item())
    assert "not_historical" in SECTOR_CONTEXT


def test_full_rebuild_generates_all_artifacts_without_symbol_loss(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    _fixture_sources(sources)
    listings = _write_listings(tmp_path / "listings.csv")
    targets = _paths(tmp_path / "outputs")

    result = full_rebuild(
        source_csv_dir=sources, listings_path=listings, paths=targets
    )

    assert result.schema_version == NATIVE_MARKET_SCHEMA_VERSION
    assert result.consolidated_rows == 4
    assert result.symbol_count == 3
    assert result.duplicate_count == 0
    assert targets.master_csv.is_file()
    assert sorted(path.name for path in targets.symbol_csv_dir.glob("*.csv")) == [
        "AAA.csv",
        "BBB.csv",
        "FUND.csv",
    ]
    assert len(tuple(targets.daily_parquet_dir.glob("*.parquet"))) == 2
    assert targets.consolidated_parquet.is_file()
    master = pd.read_csv(targets.master_csv, dtype={"symbol": "string"})
    assert tuple(master.columns) == CANONICAL_MARKET_COLUMNS
    assert list(zip(master["market_date"], master["symbol"], strict=True)) == [
        ("2024-01-02", "AAA"),
        ("2024-01-02", "BBB"),
        ("2024-01-03", "AAA"),
        ("2024-01-03", "FUND"),
    ]
    metadata = pq.ParquetFile(targets.consolidated_parquet).schema_arrow.metadata
    assert metadata[b"schema_version"].decode() == NATIVE_MARKET_SCHEMA_VERSION
    assert metadata[b"sector_context"].decode() == SECTOR_CONTEXT
    csv_round_trip = pd.read_csv(
        targets.master_csv, dtype={"symbol": "string"}, float_precision="round_trip"
    )
    csv_round_trip["market_date"] = pd.to_datetime(csv_round_trip["market_date"])
    for column in ("sector_current", "sector_source", "sector_snapshot_date"):
        csv_round_trip[column] = csv_round_trip[column].astype("string")
    parquet_round_trip = pq.read_table(targets.consolidated_parquet).to_pandas()
    parquet_round_trip["market_date"] = pd.to_datetime(
        parquet_round_trip["market_date"]
    )
    parquet_round_trip["symbol"] = parquet_round_trip["symbol"].astype("string")
    for column in ("sector_current", "sector_source", "sector_snapshot_date"):
        parquet_round_trip[column] = parquet_round_trip[column].astype("string")
    assert canonical_content_hash(csv_round_trip) == canonical_content_hash(
        parquet_round_trip
    )


def test_duplicate_source_keys_are_rejected(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    row = _row("2024-01-02", "AAA", 10.0)
    _write_source(sources / "market_2024-01-02.csv", [row, dict(row)])

    with pytest.raises(DuplicateMarketRecordError, match="Duplicate"):
        full_rebuild(
            source_csv_dir=sources,
            listings_path=_write_listings(tmp_path / "listings.csv"),
            paths=_paths(tmp_path / "outputs"),
        )


def test_incremental_is_idempotent_and_rejects_conflicting_replacement(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    first = _write_source(
        first_dir / "market_2024-01-02.csv", [_row("2024-01-02", "AAA", 10.0)]
    )
    listings = _write_listings(tmp_path / "listings.csv")
    targets = _paths(tmp_path / "outputs")
    full_rebuild(source_csv_dir=first_dir, listings_path=listings, paths=targets)
    before = (_sha(targets.master_csv), _sha(targets.consolidated_parquet))

    noop = incremental_update([first], listings_path=listings, paths=targets)

    assert noop.idempotent_noop is True
    assert before == (_sha(targets.master_csv), _sha(targets.consolidated_parquet))
    conflicting = _write_source(
        tmp_path / "replacement" / "market_2024-01-02.csv",
        [_row("2024-01-02", "AAA", 99.0)],
    )
    with pytest.raises(DuplicateMarketRecordError, match="conflicts|replacement"):
        incremental_update([conflicting], listings_path=listings, paths=targets)


def test_full_and_incremental_complete_source_set_are_equivalent(
    tmp_path: Path,
) -> None:
    all_sources = tmp_path / "all"
    first, second = _fixture_sources(all_sources)
    first_only = tmp_path / "first_only"
    first_copy = _write_source(first_only / first.name, pd.read_csv(first).to_dict("records"))
    listings = _write_listings(tmp_path / "listings.csv")
    full_paths = _paths(tmp_path / "full")
    incremental_paths = _paths(tmp_path / "incremental")

    full = full_rebuild(
        source_csv_dir=all_sources, listings_path=listings, paths=full_paths
    )
    full_rebuild(
        source_csv_dir=first_only, listings_path=listings, paths=incremental_paths
    )
    incremental = incremental_update(
        [second], listings_path=listings, paths=incremental_paths
    )

    assert full.content_hash == incremental.content_hash
    assert full.source_set_hash == incremental.source_set_hash
    assert incremental.rows_added == 2
    assert incremental.daily_parquets_written == 1
    assert incremental.symbol_csvs_written == 2
    assert compare_parquet_core(
        full_paths.consolidated_parquet, incremental_paths.consolidated_parquet
    ).passed
    full_frame = pq.read_table(full_paths.consolidated_parquet).to_pandas()
    incremental_frame = pq.read_table(incremental_paths.consolidated_parquet).to_pandas()
    pd.testing.assert_frame_equal(full_frame, incremental_frame)
    assert canonical_content_hash(full_frame) == canonical_content_hash(incremental_frame)
    assert first_copy.read_bytes() == first.read_bytes()


def test_atomic_promotion_failure_restores_every_existing_output(
    tmp_path: Path,
) -> None:
    initial_sources = tmp_path / "initial"
    _write_source(
        initial_sources / "market_2024-01-02.csv",
        [_row("2024-01-02", "AAA", 10.0)],
    )
    listings = _write_listings(tmp_path / "listings.csv")
    targets = _paths(tmp_path / "outputs")
    full_rebuild(source_csv_dir=initial_sources, listings_path=listings, paths=targets)
    tracked = {
        "master": _sha(targets.master_csv),
        "parquet": _sha(targets.consolidated_parquet),
        "state": _sha(targets.state_json),
        "symbol": _sha(targets.symbol_csv_dir / "AAA.csv"),
    }
    new_source = _write_source(
        tmp_path / "new" / "market_2024-01-03.csv",
        [_row("2024-01-03", "AAA", 11.0)],
    )
    calls = 0

    def fail_midway(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated promotion failure")

    with pytest.raises(RuntimeError, match="simulated"):
        incremental_update(
            [new_source],
            listings_path=listings,
            paths=targets,
            before_promote=fail_midway,
        )

    assert tracked == {
        "master": _sha(targets.master_csv),
        "parquet": _sha(targets.consolidated_parquet),
        "state": _sha(targets.state_json),
        "symbol": _sha(targets.symbol_csv_dir / "AAA.csv"),
    }


def test_source_files_remain_unchanged(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    source, _ = _fixture_sources(sources)
    before = (sha256_file(source), source.stat().st_mtime_ns)

    full_rebuild(
        source_csv_dir=sources,
        listings_path=_write_listings(tmp_path / "listings.csv"),
        paths=_paths(tmp_path / "outputs"),
    )

    assert (sha256_file(source), source.stat().st_mtime_ns) == before


def test_explicit_artifact_rebuild_preserves_source_identity(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    _fixture_sources(sources)
    listings = _write_listings(tmp_path / "listings.csv")
    targets = _paths(tmp_path / "outputs")
    initial = full_rebuild(
        source_csv_dir=sources,
        listings_path=listings,
        paths=targets,
    )
    source_bytes = {
        path.name: path.read_bytes() for path in sorted(sources.glob("*.csv"))
    }

    rebuilt = rebuild_generated_artifacts(
        listings_path=listings,
        paths=targets,
    )

    assert rebuilt.operation == "artifact_rebuild"
    assert rebuilt.source_set_hash == initial.source_set_hash
    assert rebuilt.content_hash == initial.content_hash
    assert rebuilt.rows_added == 0
    assert rebuilt.daily_parquets_written == 2
    assert rebuilt.symbol_csvs_written == 3
    assert source_bytes == {
        path.name: path.read_bytes() for path in sorted(sources.glob("*.csv"))
    }


def test_full_rebuild_refuses_to_regress_existing_source_set(tmp_path: Path) -> None:
    complete = tmp_path / "complete"
    first, _ = _fixture_sources(complete)
    listings = _write_listings(tmp_path / "listings.csv")
    targets = _paths(tmp_path / "outputs")
    full_rebuild(source_csv_dir=complete, listings_path=listings, paths=targets)
    before = (_sha(targets.master_csv), _sha(targets.consolidated_parquet))
    incomplete = tmp_path / "incomplete"
    _write_source(incomplete / first.name, pd.read_csv(first).to_dict("records"))

    with pytest.raises(NativeMarketPipelineError, match="would remove"):
        full_rebuild(
            source_csv_dir=incomplete, listings_path=listings, paths=targets
        )

    assert before == (_sha(targets.master_csv), _sha(targets.consolidated_parquet))


def test_reader_can_load_sector_enriched_native_parquet(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    _fixture_sources(sources)
    targets = _paths(tmp_path / "outputs")
    full_rebuild(
        source_csv_dir=sources,
        listings_path=_write_listings(tmp_path / "listings.csv"),
        paths=targets,
    )

    loaded = load_market_data(targets.consolidated_parquet, symbols=["AAA"])

    assert loaded["symbol"].tolist() == ["AAA", "AAA"]
    assert loaded["sector_current"].tolist() == ["COMMERCIAL BANKS"] * 2


def test_reader_resolution_prefers_local_and_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local/market.parquet"
    override = tmp_path / "override/market.parquet"
    monkeypatch.setattr(parquet_market_data, "DEFAULT_PSX_MARKET_PARQUET_PATH", local)

    assert parquet_market_data.resolve_market_parquet_path(environ={}) == local.resolve()
    assert parquet_market_data.resolve_market_parquet_path(
        environ={"PSX_MARKET_PARQUET_PATH": str(override)}
    ) == override.resolve()


def test_invalid_numeric_source_fails_without_output(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    bad = _row("2024-01-02", "AAA", 10.0)
    bad["volume"] = -1
    _write_source(sources / "market_2024-01-02.csv", [bad])
    targets = _paths(tmp_path / "outputs")

    with pytest.raises(NativeMarketPipelineError, match="negative"):
        full_rebuild(
            source_csv_dir=sources,
            listings_path=_write_listings(tmp_path / "listings.csv"),
            paths=targets,
        )

    assert not targets.consolidated_parquet.exists()
    failure = json.loads(
        targets.state_json.with_name("native_market_pipeline_last_failure.json").read_text()
    )
    assert failure["status"] == "failed"
    assert "negative" in failure["failure_reason"]
