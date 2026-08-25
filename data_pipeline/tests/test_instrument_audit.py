"""Offline tests for authoritative-first historical instrument diagnosis."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.src.instrument_audit import (
    COMMON_EQUITY,
    DEBT_OR_TFC,
    ETF_OR_FUND,
    GOVERNMENT_SECURITY,
    INSUFFICIENT_LOCAL_EVIDENCE,
    OFFICIAL_LISTING_DERIVED,
    OTHER_IDENTIFIED_INSTRUMENT,
    PATTERN_SIGNAL_ONLY,
    RIGHTS_OR_VARIANT,
    UNKNOWN,
    classify_instrument_universe,
    detect_symbol_pattern_signals,
    run_instrument_audit,
    summarize_instrument_audit,
)
from data_pipeline.src.universe_audit import build_symbol_universe_audit


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


def _symbols() -> tuple[str, ...]:
    return (
        "BANKR2",
        "RIGHT1",
        "ETFONE",
        "PREF",
        "REITX",
        "CLOSEFUND",
        "MODX",
        "ABC-ODL",
        "ALACR3",
        "AIRLINK-CMAY",
        "BAFLTFC7",
        "P03PIB050824",
        "PK03TB200423",
        "HISTETF",
        "OLDCO",
    )


def _market_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, symbol in enumerate(reversed(_symbols())):
        rows.append(
            {
                "market_date": date(2024, 1, 1),
                "symbol": symbol,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100 + index,
            }
        )
        if symbol == "OLDCO":
            rows.append(
                {
                    "market_date": date(2024, 1, 2),
                    "symbol": symbol,
                    "open": 10.5,
                    "high": 11.5,
                    "low": 10.0,
                    "close": 11.0,
                    "volume": 200,
                }
            )
    return pd.DataFrame(rows)


def _registry() -> pd.DataFrame:
    rows = [
        ("BANKR2", "Bank R2", "ordinary_equity", "COMMERCIAL BANKS"),
        ("RIGHT1", "Company Right", "right", "ENGINEERING"),
        ("ETFONE", "Index ETF", "etf", "EXCHANGE TRADED FUNDS"),
        ("PREF", "Preference Share", "preference_share", "CHEMICAL"),
        ("REITX", "Residence REIT", "other", "REAL ESTATE INVESTMENT TRUST"),
        (
            "CLOSEFUND",
            "Closed-end Fund",
            "ordinary_equity",
            "CLOSE - END MUTUAL FUND",
        ),
        ("MODX", "Example Modaraba", "ordinary_equity", "MODARABAS"),
    ]
    records: list[dict[str, object]] = []
    for symbol, company, security_type, sector in rows:
        records.append(
            {
                "symbol": symbol,
                "company_name": company,
                "security_type": security_type,
                "sector": sector,
                "officially_listed": True,
                "official_status": "listed",
                "lifecycle_status": "listed_recently_traded",
                "source": OFFICIAL_SOURCE,
                "previous_symbol": pd.NA,
                "successor_symbol": pd.NA,
                "corporate_action_type": pd.NA,
            }
        )
    for symbol in set(_symbols()).difference({row[0] for row in rows}):
        records.append(
            {
                "symbol": symbol,
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
            }
        )
    return pd.DataFrame(records, columns=REGISTRY_COLUMNS)


def _classified() -> pd.DataFrame:
    universe = build_symbol_universe_audit(_market_frame())
    return classify_instrument_universe(universe, registry=_registry())


def test_official_metadata_classification_precedes_pattern_signals() -> None:
    classified = _classified().set_index("symbol")

    # BANKR2 has a rights-looking suffix, but official listing evidence says
    # ordinary equity; the ticker signal must not override it.
    assert classified.loc["BANKR2", "instrument_category"] == COMMON_EQUITY
    assert (
        classified.loc["BANKR2", "classification_basis"]
        == OFFICIAL_LISTING_DERIVED
    )
    assert "rights_or_entitlement_suffix" in classified.loc[
        "BANKR2", "pattern_signals"
    ]
    assert classified.loc["RIGHT1", "instrument_category"] == RIGHTS_OR_VARIANT
    assert classified.loc["ETFONE", "instrument_category"] == ETF_OR_FUND
    assert (
        classified.loc["PREF", "instrument_category"]
        == OTHER_IDENTIFIED_INSTRUMENT
    )
    assert classified.loc["REITX", "instrument_category"] == ETF_OR_FUND
    assert classified.loc["CLOSEFUND", "instrument_category"] == ETF_OR_FUND
    assert (
        classified.loc["MODX", "instrument_category"]
        == OTHER_IDENTIFIED_INSTRUMENT
    )


def test_pattern_only_variants_debt_government_and_fund_are_explicit() -> None:
    classified = _classified().set_index("symbol")

    expected = {
        "ABC-ODL": RIGHTS_OR_VARIANT,
        "ALACR3": RIGHTS_OR_VARIANT,
        "AIRLINK-CMAY": RIGHTS_OR_VARIANT,
        "BAFLTFC7": DEBT_OR_TFC,
        "P03PIB050824": GOVERNMENT_SECURITY,
        "PK03TB200423": GOVERNMENT_SECURITY,
        "HISTETF": ETF_OR_FUND,
    }
    for symbol, category in expected.items():
        assert classified.loc[symbol, "instrument_category"] == category
        assert classified.loc[symbol, "classification_basis"] == PATTERN_SIGNAL_ONLY
        assert bool(classified.loc[symbol, "authoritative_metadata_available"]) is False


def test_unknown_stays_unknown_and_is_only_a_historical_equity_candidate_signal() -> None:
    classified = _classified().set_index("symbol")

    assert classified.loc["OLDCO", "instrument_category"] == UNKNOWN
    assert classified.loc["OLDCO", "classification_basis"] == INSUFFICIENT_LOCAL_EVIDENCE
    assert classified.loc["OLDCO", "pattern_signals"] == "plain_symbol"
    assert bool(classified.loc["OLDCO", "historical_equity_candidate_signal"]) is True
    assert pd.isna(classified.loc["OLDCO", "sector"])


def test_pattern_signal_detection_is_deterministic_and_representative() -> None:
    assert detect_symbol_pattern_signals("ABC-ODL") == ("odl_suffix",)
    assert detect_symbol_pattern_signals("AIRLINK-CMAY") == (
        "corporate_action_cmonth",
    )
    assert detect_symbol_pattern_signals("BAFLTFC7") == ("tfc_or_debt_identifier",)
    assert detect_symbol_pattern_signals("P03PIB050824") == ("pib_identifier",)
    assert detect_symbol_pattern_signals("PIBTL") == ("plain_symbol",)
    assert detect_symbol_pattern_signals("PK03TB200423") == (
        "treasury_bill_identifier",
    )
    assert detect_symbol_pattern_signals("OLDCO") == ("plain_symbol",)


def test_classification_and_summary_are_symbol_sorted_and_reconciled() -> None:
    classified = _classified()
    summary = summarize_instrument_audit(classified)

    assert classified["symbol"].tolist() == sorted(_symbols())
    assert sum(summary.category_counts.values()) == len(_symbols())
    assert summary.registry_matched_symbols == len(_symbols())
    assert summary.official_listing_backed_symbols == 7
    assert summary.history_only_registry_symbols == len(_symbols()) - 7
    assert summary.sector_tagged_count == 7
    assert summary.sector_tagged_non_common_equity_count == 6
    assert summary.no_sector_unknown_count == 1
    assert summary.historical_equity_candidate_signal_count == 1


def _write_real_contract_parquet(path: Path) -> Path:
    rows = []
    for raw in _market_frame().to_dict(orient="records"):
        close = float(raw["close"])
        rows.append(
            {
                **raw,
                "ldcp": close - 0.5,
                "change": 0.5,
                "change_percent": 5.0,
            }
        )
    table = pa.Table.from_pandas(
        pd.DataFrame(rows), schema=MARKET_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_run_uses_read_only_boundary_and_leaves_source_parquet_unchanged(
    tmp_path: Path,
) -> None:
    parquet = _write_real_contract_parquet(tmp_path / "market.parquet")
    registry_path = tmp_path / "company_registry.csv"
    _registry().to_csv(registry_path, index=False)
    before = _sha256(parquet)

    result = run_instrument_audit(
        parquet_path=parquet,
        registry_path=registry_path,
    )

    assert result.summary.total_historical_symbols == len(_symbols())
    assert _sha256(parquet) == before
