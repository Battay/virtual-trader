"""Offline tests for the full-identity recurrent trainability gap audit."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.parquet_market_data import load_market_data
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    INSUFFICIENT_DATA,
    MISSING_REQUIRED_ARTIFACTS,
    UNSUPPORTED,
    RecurrentUniverseDiscovery,
)
from reinforcement_learning.training.recurrent_trainability_audit import (
    COLD_START,
    DATA_LIMITED,
    FEATURE_BUILD_GAP,
    LEGACY_PIPELINE_ONLY,
    NO_RECENT_TRADING_ACTIVITY,
    PIPELINE_LIMITED,
    SYMBOL_ALIAS_GAP,
    audit_recurrent_trainability_gaps,
    write_gap_audit_csv,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_market(path: Path, symbols: tuple[str, ...]) -> Path:
    dates = pd.bdate_range("2024-01-02", periods=220)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for row_index, market_date in enumerate(dates):
            close = 10.0 + symbol_index + row_index / 100.0
            rows.append(
                {
                    "market_date": market_date.date(),
                    "symbol": symbol,
                    "ldcp": close - 0.05,
                    "open": close - 0.02,
                    "high": close + 0.10,
                    "low": close - 0.10,
                    "close": close,
                    "change": 0.05,
                    "change_percent": 0.5,
                    "volume": 1_000 + row_index,
                }
            )
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows), schema=MARKET_SCHEMA), path
    )
    return path


def _discovery() -> RecurrentUniverseDiscovery:
    records = pd.DataFrame(
        [
            {
                "symbol": "READY",
                "company_name": "Ready Limited",
                "security_type": "ordinary_equity",
                "category": ELIGIBLE_TRAINABLE,
                "reason": "canonical_mature_recurrent_contract",
                "compatibility_error": "",
            },
            {
                "symbol": "COLD",
                "company_name": "Cold Limited",
                "security_type": "ordinary_equity",
                "category": INSUFFICIENT_DATA,
                "reason": "cold_start_not_independent_training",
                "compatibility_error": "",
            },
            {
                "symbol": "GEM",
                "company_name": "Gem Limited",
                "security_type": "gem_equity",
                "category": UNSUPPORTED,
                "reason": "unsupported_security_type:gem_equity",
                "compatibility_error": "",
            },
            {
                "symbol": "MISS",
                "company_name": "Missing Limited",
                "security_type": "ordinary_equity",
                "category": MISSING_REQUIRED_ARTIFACTS,
                "reason": "missing_or_incompatible_recurrent_contract",
                "compatibility_error": "missing recurrent contract",
            },
            {
                "symbol": "STALE",
                "company_name": "Stale Limited",
                "security_type": "ordinary_equity",
                "category": UNSUPPORTED,
                "reason": "not_active_recently_traded",
                "compatibility_error": "",
            },
        ]
    )
    return RecurrentUniverseDiscovery(
        records=records,
        universe_version="fixture_universe_v1",
        universe_hash="a" * 64,
        identity_count=5,
        category_counts={
            ELIGIBLE_TRAINABLE: 1,
            INSUFFICIENT_DATA: 1,
            MISSING_REQUIRED_ARTIFACTS: 1,
            UNSUPPORTED: 2,
        },
        source_inventory_hash="b" * 64,
    )


def _readiness(path: Path) -> Path:
    pd.DataFrame(
        {
            "symbol": ["COLD", "GEM", "MISS", "STALE"],
            "usable_observations": [105, 200, 200, 0],
            "history_class": ["COLD_START", "NOT_APPLICABLE", "MATURE", "NOT_APPLICABLE"],
        }
    ).to_csv(path, index=False)
    return path


def test_gap_audit_is_precise_deterministic_and_accounts_for_every_identity(
    tmp_path: Path,
) -> None:
    symbols = ("COLD", "GEM", "MISS", "STALE")
    parquet = _write_market(tmp_path / "market.parquet", symbols)
    readiness = _readiness(tmp_path / "readiness.csv")

    first = audit_recurrent_trainability_gaps(
        discovery=_discovery(),
        parquet_path=parquet,
        readiness_evidence_path=readiness,
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
    )
    second = audit_recurrent_trainability_gaps(
        discovery=_discovery(),
        parquet_path=parquet,
        readiness_evidence_path=readiness,
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
    )

    assert first.identity_count == first.final_accounting_count == 5
    assert first.non_trainable_count == 4
    assert first.records["symbol"].tolist() == sorted(symbols)
    assert first.records.equals(second.records)
    assert first.precise_category_counts == {
        COLD_START: 1,
        LEGACY_PIPELINE_ONLY: 1,
        FEATURE_BUILD_GAP: 1,
        NO_RECENT_TRADING_ACTIVITY: 1,
    }
    assert first.limitation_counts == {DATA_LIMITED: 2, PIPELINE_LIMITED: 2}
    assert "UNSUPPORTED" not in set(first.records["precise_category"])


def test_full_scale_accounting_keeps_all_508_identity_records(tmp_path: Path) -> None:
    symbols = [f"S{index:03d}" for index in range(508)]
    records = pd.DataFrame(
        {
            "symbol": symbols,
            "company_name": [f"Company {symbol}" for symbol in symbols],
            "security_type": ["ordinary_equity"] * 508,
            "category": [ELIGIBLE_TRAINABLE] * 432
            + [MISSING_REQUIRED_ARTIFACTS] * 76,
            "reason": ["canonical_mature_recurrent_contract"] * 432
            + ["missing_or_incompatible_recurrent_contract"] * 76,
            "compatibility_error": [""] * 508,
        }
    )
    discovery = RecurrentUniverseDiscovery(
        records=records,
        universe_version="fixture_508_v1",
        universe_hash="c" * 64,
        identity_count=508,
        category_counts={ELIGIBLE_TRAINABLE: 432, MISSING_REQUIRED_ARTIFACTS: 76},
        source_inventory_hash="d" * 64,
    )
    parquet = _write_market(tmp_path / "market.parquet", ("NOT_A_MEMBER",))

    audit = audit_recurrent_trainability_gaps(
        discovery=discovery,
        parquet_path=parquet,
        readiness_evidence_path=None,
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
        market_loader=lambda *_args, **_kwargs: pytest.fail(
            "absent symbols must not load market values"
        ),
    )

    assert audit.identity_count == audit.final_accounting_count == 508
    assert audit.trainable_count == 432
    assert audit.non_trainable_count == 76
    assert audit.precise_category_counts == {SYMBOL_ALIAS_GAP: 76}


def test_train_values_never_cross_metadata_derived_cutoff_and_source_is_unchanged(
    tmp_path: Path,
) -> None:
    symbols = ("COLD", "GEM", "MISS", "STALE")
    parquet = _write_market(tmp_path / "market.parquet", symbols)
    readiness = _readiness(tmp_path / "readiness.csv")
    before = _sha256(parquet)
    loads: list[tuple[str, date, date]] = []

    def sealed_loader(path: Path, *, end_date: date, symbols: list[str]):
        frame = load_market_data(path, end_date=end_date, symbols=symbols)
        maximum = max(frame["market_date"])
        loads.append((symbols[0], end_date, maximum))
        assert maximum <= end_date
        return frame

    audit = audit_recurrent_trainability_gaps(
        discovery=_discovery(),
        parquet_path=parquet,
        readiness_evidence_path=readiness,
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
        market_loader=sealed_loader,
    )

    assert len(loads) == 4
    assert all(end == maximum for _, end, maximum in loads)
    assert set(audit.records["train_raw_date_count"]) == {154}
    assert set(audit.records["train_row_count"]) == {105}
    assert _sha256(parquet) == before


def test_missing_symbol_is_identity_limited_and_never_value_loaded(
    tmp_path: Path,
) -> None:
    parquet = _write_market(tmp_path / "market.parquet", ("OTHER",))
    discovery = _discovery()
    discovery = RecurrentUniverseDiscovery(
        records=discovery.records.loc[discovery.records["symbol"] == "MISS"].reset_index(drop=True),
        universe_version=discovery.universe_version,
        universe_hash=discovery.universe_hash,
        identity_count=1,
        category_counts={MISSING_REQUIRED_ARTIFACTS: 1},
        source_inventory_hash=discovery.source_inventory_hash,
    )
    readiness = _readiness(tmp_path / "readiness.csv")

    audit = audit_recurrent_trainability_gaps(
        discovery=discovery,
        parquet_path=parquet,
        readiness_evidence_path=readiness,
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
        market_loader=lambda *_args, **_kwargs: pytest.fail(
            "market values loaded for absent identity"
        ),
    )

    row = audit.records.iloc[0]
    assert row["precise_category"] == SYMBOL_ALIAS_GAP
    assert row["limitation_type"] == "IDENTITY_LIMITED"
    assert bool(row["canonical_parquet_symbol_present"]) is False


def test_missing_contract_and_legacy_type_are_pipeline_not_data_limited(
    tmp_path: Path,
) -> None:
    symbols = ("COLD", "GEM", "MISS", "STALE")
    parquet = _write_market(tmp_path / "market.parquet", symbols)
    audit = audit_recurrent_trainability_gaps(
        discovery=_discovery(),
        parquet_path=parquet,
        readiness_evidence_path=_readiness(tmp_path / "readiness.csv"),
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
    )
    by_symbol = audit.records.set_index("symbol")

    assert by_symbol.loc["MISS", "precise_category"] == FEATURE_BUILD_GAP
    assert by_symbol.loc["GEM", "precise_category"] == LEGACY_PIPELINE_ONLY
    assert by_symbol.loc["MISS", "limitation_type"] == PIPELINE_LIMITED
    assert by_symbol.loc["GEM", "limitation_type"] == PIPELINE_LIMITED
    assert bool(by_symbol.loc["MISS", "canonical_parquet_sufficient"]) is True
    assert bool(by_symbol.loc["GEM", "canonical_parquet_sufficient"]) is True


def test_audit_csv_refuses_silent_overwrite(tmp_path: Path) -> None:
    symbols = ("COLD", "GEM", "MISS", "STALE")
    audit = audit_recurrent_trainability_gaps(
        discovery=_discovery(),
        parquet_path=_write_market(tmp_path / "market.parquet", symbols),
        readiness_evidence_path=_readiness(tmp_path / "readiness.csv"),
        processed_symbols_dir=tmp_path / "symbols",
        splits_dir=tmp_path / "splits",
    )
    output = tmp_path / "audit.csv"

    write_gap_audit_csv(audit, output)

    assert pd.read_csv(output, dtype={"symbol": "string"})["symbol"].tolist() == sorted(symbols)
    with pytest.raises(FileExistsError, match="already exists"):
        write_gap_audit_csv(audit, output)
