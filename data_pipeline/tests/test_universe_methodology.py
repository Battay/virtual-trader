"""Offline tests for equity-universe methodology diagnostics."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.src.instrument_audit import COMMON_EQUITY, UNKNOWN
from data_pipeline.src.universe_methodology import (
    LEXICAL_PREFIX_SIGNAL_ONLY,
    LIKELY_OTHER_INSTRUMENT,
    NO_LOCAL_ALIAS_EVIDENCE,
    SPARSE_OR_UNCERTAIN,
    STRONG_HISTORICAL_EQUITY_CANDIDATE,
    UNRESOLVED,
    add_active_span_coverage,
    add_alias_diagnostics,
    build_sensitivity_table,
    classify_unknown_diagnostics,
    detect_unknown_methodology_pattern_signals,
    run_universe_methodology_audit,
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
OFFICIAL_SOURCE = "https://dps.psx.com.pk/listings-table/main/nc"
REGISTRY_COLUMNS = (
    "symbol",
    "company_name",
    "security_type",
    "sector",
    "officially_listed",
    "official_status",
    "lifecycle_status",
    "source",
    "previous_symbol",
    "successor_symbol",
    "corporate_action_type",
)


def _coverage_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["FULL", "NEW", "SPARSE"],
            "first_market_date": [
                date(2024, 1, 1),
                date(2024, 1, 5),
                date(2024, 1, 1),
            ],
            "last_market_date": [
                date(2024, 1, 10),
                date(2024, 1, 10),
                date(2024, 1, 10),
            ],
            "unique_market_dates": [4, 2, 2],
        }
    )


def test_active_span_coverage_uses_actual_market_dates_and_protects_new_listing() -> None:
    # There are four actual market dates, not ten calendar days.
    global_dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-10"]
    )

    result = add_active_span_coverage(_coverage_universe(), global_dates).set_index(
        "symbol"
    )

    assert result.loc["FULL", "active_span_market_date_count"] == 4
    assert result.loc["FULL", "active_span_coverage"] == pytest.approx(1.0)
    assert result.loc["NEW", "active_span_market_date_count"] == 2
    assert result.loc["NEW", "active_span_coverage"] == pytest.approx(1.0)
    assert result.loc["SPARSE", "active_span_market_date_count"] == 4
    assert result.loc["SPARSE", "active_span_coverage"] == pytest.approx(0.5)


def _diagnostic_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "observation_count": 100,
        "active_span_coverage": 0.8,
        "zero_ohl_ratio": 0.0,
        "positive_close_ratio": 1.0,
        "pattern_signals": "plain_symbol",
        "methodology_pattern_signals": "",
        "registry_security_type": "unknown",
        "successor_symbol": pd.NA,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_unknown_diagnostic_groups_are_deterministic_and_non_authoritative() -> None:
    current = _diagnostic_frame(
        [
            {
                "symbol": "CURRA",
                "observation_count": 1_000,
                "active_span_coverage": 0.90,
                "zero_ohl_ratio": 0.05,
            },
            {
                "symbol": "CURRB",
                "observation_count": 800,
                "active_span_coverage": 0.80,
                "zero_ohl_ratio": 0.10,
            },
            {
                "symbol": "CURRC",
                "observation_count": 600,
                "active_span_coverage": 0.70,
                "zero_ohl_ratio": 0.15,
            },
            {
                "symbol": "CURRD",
                "observation_count": 400,
                "active_span_coverage": 0.60,
                "zero_ohl_ratio": 0.20,
            },
        ]
    )
    unknowns = _diagnostic_frame(
        [
            {
                "symbol": "STRONG",
                "observation_count": 900,
                "active_span_coverage": 0.90,
                "zero_ohl_ratio": 0.05,
            },
            {
                "symbol": "SPARSE",
                "observation_count": 10,
                "active_span_coverage": 0.10,
            },
            {
                "symbol": "OTHER",
                "methodology_pattern_signals": "government_gis_identifier",
            },
            {
                "symbol": "REVIEW",
                "observation_count": 700,
                "active_span_coverage": 0.75,
                "zero_ohl_ratio": 0.50,
            },
        ]
    )

    first = classify_unknown_diagnostics(unknowns, current).set_index("symbol")
    second = classify_unknown_diagnostics(
        unknowns.sample(frac=1.0, random_state=7), current
    ).set_index("symbol")

    assert first.loc["STRONG", "diagnostic_group"] == (
        STRONG_HISTORICAL_EQUITY_CANDIDATE
    )
    assert first.loc["SPARSE", "diagnostic_group"] == SPARSE_OR_UNCERTAIN
    assert first.loc["OTHER", "diagnostic_group"] == LIKELY_OTHER_INSTRUMENT
    assert first.loc["REVIEW", "diagnostic_group"] == UNRESOLVED
    pd.testing.assert_series_equal(
        first["diagnostic_group"], second["diagnostic_group"]
    )


def test_alias_diagnostics_report_signals_without_asserting_aliases() -> None:
    unknowns = _diagnostic_frame(
        [{"symbol": "OLD"}, {"symbol": "NOMATCH"}]
    )

    result = add_alias_diagnostics(unknowns, ("OLDNEW", "CURRENT")).set_index(
        "symbol"
    )

    assert result.loc["OLD", "alias_evidence_status"] == LEXICAL_PREFIX_SIGNAL_ONLY
    assert result.loc["OLD", "current_symbol_lexical_candidates"] == "OLDNEW"
    assert result.loc["NOMATCH", "alias_evidence_status"] == NO_LOCAL_ALIAS_EVIDENCE


def test_unknown_methodology_patterns_remain_explicit_diagnostic_signals() -> None:
    assert detect_unknown_methodology_pattern_signals("P01GIS090525") == (
        "government_gis_identifier",
    )
    assert detect_unknown_methodology_pattern_signals("DAWHSC2") == (
        "corporate_action_sc_suffix",
    )
    assert detect_unknown_methodology_pattern_signals("SMBLCPSB") == (
        "preference_cps_suffix",
    )
    assert detect_unknown_methodology_pattern_signals("FFLNV") == (
        "non_voting_nv_suffix",
    )
    assert detect_unknown_methodology_pattern_signals("ENGRO") == ()


def test_sensitivity_table_is_deterministic_and_reconciled() -> None:
    current = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "calendar_span_days": [400, 800, 2_000],
            "active_span_coverage": [0.60, 0.80, 0.95],
            "median_volume": [100.0, 1_000.0, 10_000.0],
        }
    )

    first = build_sensitivity_table(current)
    second = build_sensitivity_table(
        current.sample(frac=1.0, random_state=11).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 64
    row = first.loc[
        (first["history_window"] == "2y")
        & (first["minimum_active_span_coverage"] == 0.70)
        & (first["liquidity_cutoff"] == "current_common_p50")
    ].iloc[0]
    assert row["minimum_median_volume"] == pytest.approx(1_000.0)
    assert row["symbol_count"] == 2


def _market_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = [date(2024, 1, 1), date(2024, 1, 3), date(2024, 1, 8)]
    for symbol, selected_dates in (
        ("CURR", dates),
        ("OLD", dates[:2]),
    ):
        for index, market_date in enumerate(selected_dates):
            rows.append(
                {
                    "market_date": market_date,
                    "symbol": symbol,
                    "ldcp": 10.0,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "change": 0.5,
                    "change_percent": 5.0,
                    "volume": 100 + index,
                }
            )
    return rows


def _write_market(path: Path) -> Path:
    table = pa.Table.from_pandas(
        pd.DataFrame(_market_rows()), schema=MARKET_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path)
    return path


def _write_registry(path: Path) -> Path:
    rows = [
        {
            "symbol": "CURR",
            "company_name": "Current Limited",
            "security_type": "ordinary_equity",
            "sector": "ENGINEERING",
            "officially_listed": True,
            "official_status": "listed",
            "lifecycle_status": "listed_recently_traded",
            "source": OFFICIAL_SOURCE,
            "previous_symbol": pd.NA,
            "successor_symbol": pd.NA,
            "corporate_action_type": pd.NA,
        },
        {
            "symbol": "OLD",
            "company_name": pd.NA,
            "security_type": "unknown",
            "sector": pd.NA,
            "officially_listed": False,
            "official_status": "historical",
            "lifecycle_status": "historical_only",
            "source": "master_market_history",
            "previous_symbol": pd.NA,
            "successor_symbol": pd.NA,
            "corporate_action_type": pd.NA,
        },
    ]
    pd.DataFrame(rows, columns=REGISTRY_COLUMNS).to_csv(path, index=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_boundary_methodology_run_leaves_source_parquet_unchanged(
    tmp_path: Path,
) -> None:
    parquet = _write_market(tmp_path / "market.parquet")
    registry = _write_registry(tmp_path / "registry.csv")
    before = _sha256(parquet)

    result = run_universe_methodology_audit(
        parquet_path=parquet,
        registry_path=registry,
    )

    assert result.summary.current_common_equity_count == 1
    assert result.summary.unknown_count == 1
    assert result.summary.current_common_first_date_quantiles["min"] == "2024-01-01"
    assert result.current_common.iloc[0]["instrument_category"] == COMMON_EQUITY
    assert result.unknowns.iloc[0]["instrument_category"] == UNKNOWN
    assert _sha256(parquet) == before
