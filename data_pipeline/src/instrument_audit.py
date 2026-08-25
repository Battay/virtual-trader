"""Authoritative-first audit of historical PSX instrument identifiers.

Current official-listing metadata is used before ticker-pattern diagnostics.
Patterns never override official-listing-backed evidence and remain explicitly
labelled as signals rather than authoritative historical classifications.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd

from .config import COMPANY_REGISTRY_PATH
from .parquet_market_data import MarketParquetError
from .universe_audit import (
    QUANTILE_LABELS,
    QUANTILE_PROBABILITIES,
    UNIVERSE_AUDIT_COLUMNS,
    UniverseAuditError,
    run_universe_audit,
)


COMMON_EQUITY = "COMMON_EQUITY"
RIGHTS_OR_VARIANT = "RIGHTS_OR_VARIANT"
DEBT_OR_TFC = "DEBT_OR_TFC"
GOVERNMENT_SECURITY = "GOVERNMENT_SECURITY"
ETF_OR_FUND = "ETF_OR_FUND"
OTHER_IDENTIFIED_INSTRUMENT = "OTHER_IDENTIFIED_INSTRUMENT"
UNKNOWN = "UNKNOWN"
INSTRUMENT_CATEGORIES = (
    COMMON_EQUITY,
    RIGHTS_OR_VARIANT,
    DEBT_OR_TFC,
    GOVERNMENT_SECURITY,
    ETF_OR_FUND,
    OTHER_IDENTIFIED_INSTRUMENT,
    UNKNOWN,
)

OFFICIAL_LISTING_DERIVED = "OFFICIAL_LISTING_DERIVED"
PATTERN_SIGNAL_ONLY = "PATTERN_SIGNAL_ONLY"
INSUFFICIENT_LOCAL_EVIDENCE = "INSUFFICIENT_LOCAL_EVIDENCE"
OFFICIAL_LISTING_SOURCE_PREFIX = "https://dps.psx.com.pk/listings-table/"

PATTERN_ODL = "odl_suffix"
PATTERN_RIGHT_NUMBER = "rights_or_entitlement_suffix"
PATTERN_CORPORATE_MONTH = "corporate_action_cmonth"
PATTERN_MONTH_CONTRACT = "month_contract_suffix"
PATTERN_TFC_DEBT = "tfc_or_debt_identifier"
PATTERN_PIB = "pib_identifier"
PATTERN_TREASURY_BILL = "treasury_bill_identifier"
PATTERN_ETF_FUND = "etf_or_fund_identifier"
PATTERN_PLAIN = "plain_symbol"
PATTERN_SIGNAL_NAMES = (
    PATTERN_ODL,
    PATTERN_RIGHT_NUMBER,
    PATTERN_CORPORATE_MONTH,
    PATTERN_MONTH_CONTRACT,
    PATTERN_TFC_DEBT,
    PATTERN_PIB,
    PATTERN_TREASURY_BILL,
    PATTERN_ETF_FUND,
    PATTERN_PLAIN,
)

MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
ODL_PATTERN = re.compile(r"-ODL$")
RIGHT_NUMBER_PATTERN = re.compile(r"(?:R|R\d+)$")
CORPORATE_MONTH_PATTERN = re.compile(rf"-C(?:{MONTHS})[A-Z]?$")
MONTH_CONTRACT_PATTERN = re.compile(rf"-(?:{MONTHS})[A-Z]?$")
TFC_DEBT_PATTERN = re.compile(r"(?:TFC\d*|SUKUK|BOND|DEBT)")
PIB_PATTERN = re.compile(r"^P\d{2}PIB\d+$")
TREASURY_BILL_PATTERN = re.compile(r"^PK\d{2}TB\d+")
ETF_FUND_PATTERN = re.compile(r"(?:ETF|FUND)")
PLAIN_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")

INSTRUMENT_AUDIT_COLUMNS = (
    "symbol",
    "instrument_category",
    "classification_basis",
    "authoritative_metadata_available",
    "pattern_signals",
    "historical_equity_candidate_signal",
    "company_name",
    "sector",
    "registry_matched",
    "registry_security_type",
    "officially_listed",
    "official_status",
    "lifecycle_status",
    "registry_source",
    "previous_symbol",
    "successor_symbol",
    "corporate_action_type",
    "first_market_date",
    "last_market_date",
    "observation_count",
    "calendar_span_days",
    "unique_market_dates",
    "coverage_ratio",
)


class InstrumentAuditError(RuntimeError):
    """Raised when classification evidence cannot be reconciled safely."""


@dataclass(frozen=True)
class InstrumentAuditSummary:
    total_historical_symbols: int
    registry_matched_symbols: int
    registry_unmatched_symbols: int
    official_listing_backed_symbols: int
    history_only_registry_symbols: int
    category_counts: Mapping[str, int]
    classification_basis_counts: Mapping[str, int]
    pattern_counts: Mapping[str, int]
    pattern_examples: Mapping[str, tuple[str, ...]]
    category_examples: Mapping[str, tuple[str, ...]]
    sector_tagged_count: int
    no_sector_count: int
    distinct_sector_count: int
    sector_symbol_counts: Mapping[str, int]
    sector_tagged_category_counts: Mapping[str, int]
    sector_tagged_non_common_equity_count: int
    sector_tagged_first_market_date: str | None
    sector_tagged_last_market_date: str | None
    sector_tagged_observation_quantiles: Mapping[str, float | None]
    no_sector_category_counts: Mapping[str, int]
    no_sector_unknown_count: int
    no_sector_unknown_examples: tuple[str, ...]
    historical_equity_candidate_signal_count: int
    historical_equity_candidate_examples: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "category_counts",
            "classification_basis_counts",
            "pattern_counts",
            "sector_symbol_counts",
            "sector_tagged_category_counts",
            "sector_tagged_observation_quantiles",
            "no_sector_category_counts",
        ):
            payload[name] = dict(getattr(self, name))
        payload["pattern_examples"] = {
            key: list(value) for key, value in self.pattern_examples.items()
        }
        payload["category_examples"] = {
            key: list(value) for key, value in self.category_examples.items()
        }
        payload["no_sector_unknown_examples"] = list(
            self.no_sector_unknown_examples
        )
        payload["historical_equity_candidate_examples"] = list(
            self.historical_equity_candidate_examples
        )
        return payload


@dataclass(frozen=True)
class InstrumentAuditResult:
    instruments: pd.DataFrame = field(repr=False, compare=False)
    summary: InstrumentAuditSummary
    parquet_path: Path
    registry_path: Path | None


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def load_registry_evidence(
    path: str | os.PathLike[str] | None = COMPANY_REGISTRY_PATH,
) -> pd.DataFrame | None:
    """Load the existing registry without treating history-only rows as official."""

    if path is None:
        return None
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        return None
    required = (
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
    try:
        registry = pd.read_csv(resolved, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise InstrumentAuditError(
            f"Could not read company registry {resolved}: {exc}"
        ) from exc
    missing = [column for column in required if column not in registry]
    if missing:
        raise InstrumentAuditError(
            "Company registry is missing classification fields: " + ", ".join(missing)
        )
    evidence = registry.loc[:, required].copy()
    evidence["symbol"] = evidence["symbol"].astype("string").str.strip()
    if evidence["symbol"].isna().any() or (evidence["symbol"] == "").any():
        raise InstrumentAuditError("Company registry contains a blank symbol")
    if evidence["symbol"].duplicated().any():
        duplicates = sorted(
            evidence.loc[evidence["symbol"].duplicated(False), "symbol"]
            .astype(str)
            .unique()
        )
        raise InstrumentAuditError(
            "Company registry contains duplicate symbols: " + ", ".join(duplicates)
        )
    return evidence.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def detect_symbol_pattern_signals(symbol: str) -> tuple[str, ...]:
    """Return deterministic non-authoritative ticker diagnostics."""

    normalized = str(symbol).strip().upper()
    signals: list[str] = []
    if ODL_PATTERN.search(normalized):
        signals.append(PATTERN_ODL)
    corporate_month = bool(CORPORATE_MONTH_PATTERN.search(normalized))
    month_contract = bool(MONTH_CONTRACT_PATTERN.search(normalized))
    if corporate_month:
        signals.append(PATTERN_CORPORATE_MONTH)
    if month_contract:
        signals.append(PATTERN_MONTH_CONTRACT)
    # APR/MAR-style month variants also end in R; do not double-report them as
    # potential rights suffixes.  This signal remains diagnostic because a
    # bare trailing R is not authoritative evidence by itself.
    if not corporate_month and not month_contract and RIGHT_NUMBER_PATTERN.search(
        normalized
    ):
        signals.append(PATTERN_RIGHT_NUMBER)
    checks = (
        (PATTERN_TFC_DEBT, TFC_DEBT_PATTERN),
        (PATTERN_PIB, PIB_PATTERN),
        (PATTERN_TREASURY_BILL, TREASURY_BILL_PATTERN),
        (PATTERN_ETF_FUND, ETF_FUND_PATTERN),
    )
    for name, pattern in checks:
        if pattern.search(normalized):
            signals.append(name)
    if not signals and PLAIN_SYMBOL_PATTERN.fullmatch(normalized):
        signals.append(PATTERN_PLAIN)
    return tuple(signals)


def _official_listing_evidence(record: Mapping[str, object]) -> bool:
    return _bool_value(record.get("officially_listed")) and str(
        record.get("source", "")
    ).startswith(OFFICIAL_LISTING_SOURCE_PREFIX)


def _category_from_official_metadata(record: Mapping[str, object]) -> str:
    security_type = str(record.get("security_type", "")).strip().lower()
    sector = str(
        record.get("registry_sector")
        if not pd.isna(record.get("registry_sector"))
        else record.get("sector", "")
    ).strip().upper()
    # The official sector is more specific than the registry's deliberately
    # broad listing-type inference for funds and Modarabas.  In particular,
    # the current listing table labels closed-end funds and Modarabas with an
    # ordinary-equity-like security type even though neither is a common
    # company equity for this research classification.
    if (
        "FUND" in sector
        or "REAL ESTATE INVESTMENT TRUST" in sector
        or "REIT" in sector
    ):
        return ETF_OR_FUND
    if sector == "MODARABAS":
        return OTHER_IDENTIFIED_INSTRUMENT
    if security_type in {"ordinary_equity", "gem_equity"}:
        return COMMON_EQUITY
    if security_type == "right":
        return RIGHTS_OR_VARIANT
    if security_type == "etf":
        return ETF_OR_FUND
    if security_type == "preference_share":
        return OTHER_IDENTIFIED_INSTRUMENT
    if security_type == "other":
        return OTHER_IDENTIFIED_INSTRUMENT
    return UNKNOWN


def _category_from_pattern_signals(signals: Sequence[str]) -> str:
    signal_set = set(signals)
    if signal_set.intersection({PATTERN_PIB, PATTERN_TREASURY_BILL}):
        return GOVERNMENT_SECURITY
    if PATTERN_TFC_DEBT in signal_set:
        return DEBT_OR_TFC
    if signal_set.intersection(
        {
            PATTERN_ODL,
            PATTERN_RIGHT_NUMBER,
            PATTERN_CORPORATE_MONTH,
            PATTERN_MONTH_CONTRACT,
        }
    ):
        return RIGHTS_OR_VARIANT
    if PATTERN_ETF_FUND in signal_set:
        return ETF_OR_FUND
    return UNKNOWN


def classify_instrument_universe(
    universe: pd.DataFrame,
    *,
    registry: pd.DataFrame | None,
) -> pd.DataFrame:
    """Classify every historical identifier without dropping unknowns."""

    required_universe = {
        "symbol",
        "first_market_date",
        "last_market_date",
        "observation_count",
        "calendar_span_days",
        "unique_market_dates",
        "coverage_ratio",
    }
    missing = sorted(required_universe.difference(universe.columns))
    if missing:
        raise InstrumentAuditError(
            "Universe audit is missing classification columns: " + ", ".join(missing)
        )
    base = universe.loc[:, list(UNIVERSE_AUDIT_COLUMNS)].copy(deep=True)
    base["symbol"] = base["symbol"].astype("string").str.strip()
    if base["symbol"].isna().any() or base["symbol"].duplicated().any():
        raise InstrumentAuditError("Universe symbols must be non-null and unique")

    if registry is None:
        joined = base.copy()
        for column in (
            "registry_company_name",
            "registry_sector",
            "security_type",
            "officially_listed",
            "official_status",
            "lifecycle_status",
            "source",
            "previous_symbol",
            "successor_symbol",
            "corporate_action_type",
        ):
            joined[column] = pd.NA
        joined["registry_matched"] = False
    else:
        evidence = registry.copy(deep=True)
        evidence["symbol"] = evidence["symbol"].astype("string").str.strip()
        if evidence["symbol"].isna().any() or evidence["symbol"].duplicated().any():
            raise InstrumentAuditError("Registry evidence symbols must be unique")
        evidence = evidence.rename(
            columns={
                "company_name": "registry_company_name",
                "sector": "registry_sector",
            }
        )
        evidence["registry_matched"] = True
        joined = base.merge(
            evidence,
            on="symbol",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        joined["registry_matched"] = (
            joined["registry_matched"].fillna(False).astype(bool)
        )

    rows: list[dict[str, object]] = []
    records = joined.sort_values("symbol", kind="mergesort").to_dict(
        orient="records"
    )
    for raw in records:
        signals = detect_symbol_pattern_signals(str(raw["symbol"]))
        official = bool(raw["registry_matched"]) and _official_listing_evidence(raw)
        if official:
            category = _category_from_official_metadata(raw)
            basis = OFFICIAL_LISTING_DERIVED
        else:
            category = _category_from_pattern_signals(signals)
            basis = (
                PATTERN_SIGNAL_ONLY
                if category != UNKNOWN
                else INSUFFICIENT_LOCAL_EVIDENCE
            )
        company_name = _optional_text(raw.get("registry_company_name")) or _optional_text(
            raw.get("company_name")
        )
        sector = _optional_text(raw.get("registry_sector")) or _optional_text(
            raw.get("sector")
        )
        historical_equity_candidate = bool(
            category == UNKNOWN
            and sector is None
            and PATTERN_PLAIN in signals
            and not official
        )
        rows.append(
            {
                "symbol": str(raw["symbol"]),
                "instrument_category": category,
                "classification_basis": basis,
                "authoritative_metadata_available": official,
                "pattern_signals": "|".join(signals),
                "historical_equity_candidate_signal": historical_equity_candidate,
                "company_name": company_name,
                "sector": sector,
                "registry_matched": bool(raw["registry_matched"]),
                "registry_security_type": _optional_text(raw.get("security_type")),
                "officially_listed": (
                    _bool_value(raw.get("officially_listed"))
                    if bool(raw["registry_matched"])
                    else False
                ),
                "official_status": _optional_text(raw.get("official_status")),
                "lifecycle_status": _optional_text(raw.get("lifecycle_status")),
                "registry_source": _optional_text(raw.get("source")),
                "previous_symbol": _optional_text(raw.get("previous_symbol")),
                "successor_symbol": _optional_text(raw.get("successor_symbol")),
                "corporate_action_type": _optional_text(
                    raw.get("corporate_action_type")
                ),
                "first_market_date": raw["first_market_date"],
                "last_market_date": raw["last_market_date"],
                "observation_count": int(raw["observation_count"]),
                "calendar_span_days": int(raw["calendar_span_days"]),
                "unique_market_dates": int(raw["unique_market_dates"]),
                "coverage_ratio": float(raw["coverage_ratio"]),
            }
        )
    result = pd.DataFrame(rows, columns=INSTRUMENT_AUDIT_COLUMNS)
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {label: None for label in QUANTILE_LABELS}
    measured = numeric.quantile(QUANTILE_PROBABILITIES, interpolation="linear")
    return {
        label: float(measured.iloc[index])
        for index, label in enumerate(QUANTILE_LABELS)
    }


def _ordered_counts(values: pd.Series, keys: Sequence[str]) -> dict[str, int]:
    counts = values.value_counts(dropna=False)
    return {key: int(counts.get(key, 0)) for key in keys}


def summarize_instrument_audit(
    instruments: pd.DataFrame,
) -> InstrumentAuditSummary:
    """Build deterministic evidence/category/sector diagnostics."""

    missing = [column for column in INSTRUMENT_AUDIT_COLUMNS if column not in instruments]
    if missing:
        raise InstrumentAuditError(
            "Instrument audit is missing summary columns: " + ", ".join(missing)
        )
    ordered = instruments.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    sector_values = ordered["sector"].astype("string").str.strip()
    sector_mask = sector_values.notna() & sector_values.ne("")
    sector_group = ordered.loc[sector_mask]
    no_sector_group = ordered.loc[~sector_mask]
    category_counts = _ordered_counts(ordered["instrument_category"], INSTRUMENT_CATEGORIES)
    basis_keys = (
        OFFICIAL_LISTING_DERIVED,
        PATTERN_SIGNAL_ONLY,
        INSUFFICIENT_LOCAL_EVIDENCE,
    )
    pattern_counts = {
        signal: int(
            ordered["pattern_signals"]
            .fillna("")
            .astype(str)
            .str.split("|")
            .map(lambda values: signal in values)
            .sum()
        )
        for signal in PATTERN_SIGNAL_NAMES
    }
    pattern_examples = {
        signal: tuple(
            ordered.loc[
                ordered["pattern_signals"]
                .fillna("")
                .astype(str)
                .str.split("|")
                .map(lambda values: signal in values),
                "symbol",
            ]
            .astype(str)
            .head(10)
        )
        for signal in PATTERN_SIGNAL_NAMES
    }
    category_examples = {
        category: tuple(
            ordered.loc[ordered["instrument_category"] == category, "symbol"]
            .astype(str)
            .head(12)
        )
        for category in INSTRUMENT_CATEGORIES
    }
    sector_counts_series = sector_group["sector"].value_counts()
    sector_counts = {
        str(sector): int(count)
        for sector, count in sorted(
            sector_counts_series.items(), key=lambda item: (-item[1], str(item[0]))
        )
    }
    candidate_mask = ordered["historical_equity_candidate_signal"].astype(bool)
    no_sector_unknown = no_sector_group.loc[
        no_sector_group["instrument_category"] == UNKNOWN
    ]
    return InstrumentAuditSummary(
        total_historical_symbols=len(ordered),
        registry_matched_symbols=int(ordered["registry_matched"].astype(bool).sum()),
        registry_unmatched_symbols=int((~ordered["registry_matched"].astype(bool)).sum()),
        official_listing_backed_symbols=int(
            ordered["authoritative_metadata_available"].astype(bool).sum()
        ),
        history_only_registry_symbols=int(
            (
                ordered["registry_matched"].astype(bool)
                & ~ordered["authoritative_metadata_available"].astype(bool)
            ).sum()
        ),
        category_counts=category_counts,
        classification_basis_counts=_ordered_counts(
            ordered["classification_basis"], basis_keys
        ),
        pattern_counts=pattern_counts,
        pattern_examples=pattern_examples,
        category_examples=category_examples,
        sector_tagged_count=len(sector_group),
        no_sector_count=len(no_sector_group),
        distinct_sector_count=int(sector_group["sector"].nunique()),
        sector_symbol_counts=sector_counts,
        sector_tagged_category_counts=_ordered_counts(
            sector_group["instrument_category"], INSTRUMENT_CATEGORIES
        ),
        sector_tagged_non_common_equity_count=int(
            (sector_group["instrument_category"] != COMMON_EQUITY).sum()
        ),
        sector_tagged_first_market_date=(
            pd.to_datetime(sector_group["first_market_date"]).min().date().isoformat()
            if not sector_group.empty
            else None
        ),
        sector_tagged_last_market_date=(
            pd.to_datetime(sector_group["last_market_date"]).max().date().isoformat()
            if not sector_group.empty
            else None
        ),
        sector_tagged_observation_quantiles=_quantiles(
            sector_group["observation_count"]
        ),
        no_sector_category_counts=_ordered_counts(
            no_sector_group["instrument_category"], INSTRUMENT_CATEGORIES
        ),
        no_sector_unknown_count=len(no_sector_unknown),
        no_sector_unknown_examples=tuple(
            no_sector_unknown["symbol"].astype(str).head(30)
        ),
        historical_equity_candidate_signal_count=int(candidate_mask.sum()),
        historical_equity_candidate_examples=tuple(
            ordered.loc[candidate_mask]
            .sort_values(
                ["observation_count", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            )["symbol"]
            .astype(str)
            .head(30)
        ),
    )


def run_instrument_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] | None = COMPANY_REGISTRY_PATH,
) -> InstrumentAuditResult:
    """Run the complete read-only historical-instrument diagnosis."""

    universe_result = run_universe_audit(
        parquet_path=parquet_path,
        registry_path=registry_path,
    )
    registry = load_registry_evidence(registry_path)
    instruments = classify_instrument_universe(
        universe_result.symbols,
        registry=registry,
    )
    return InstrumentAuditResult(
        instruments=instruments,
        summary=summarize_instrument_audit(instruments),
        parquet_path=universe_result.parquet_path,
        registry_path=universe_result.registry_path,
    )


def write_instrument_audit_csv(
    instruments: pd.DataFrame,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    source_parquet_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write deterministic diagnostics only after explicit output request."""

    output = Path(output_path).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".csv":
        raise InstrumentAuditError("Instrument audit output must use a .csv suffix")
    if source_parquet_path is not None and output == Path(source_parquet_path).resolve(
        strict=False
    ):
        raise InstrumentAuditError("Audit output cannot target the source Parquet file")
    if not output.parent.is_dir():
        raise InstrumentAuditError(f"Output directory does not exist: {output.parent}")
    if output.exists() and not overwrite:
        raise InstrumentAuditError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )
    ordered = instruments.loc[:, INSTRUMENT_AUDIT_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    )
    ordered.to_csv(
        output,
        index=False,
        mode="w" if overwrite else "x",
        lineterminator="\n",
        na_rep="",
        float_format="%.12g",
        date_format="%Y-%m-%d",
    )
    return output


def _print_mapping(label: str, values: Mapping[str, object]) -> None:
    print(f"{label}: {json.dumps(dict(values), sort_keys=False)}")


def _print_summary(result: InstrumentAuditResult) -> None:
    summary = result.summary
    print(f"Parquet: {result.parquet_path}")
    print(f"Registry: {result.registry_path or 'not available'}")
    print(f"Historical symbol strings: {summary.total_historical_symbols:,}")
    print(f"Registry matches: {summary.registry_matched_symbols:,}")
    print(f"Registry unmatched: {summary.registry_unmatched_symbols:,}")
    print(f"Official-listing-backed rows: {summary.official_listing_backed_symbols:,}")
    print(f"History-only registry matches: {summary.history_only_registry_symbols:,}")
    _print_mapping("Category counts", summary.category_counts)
    _print_mapping("Classification-basis counts", summary.classification_basis_counts)
    _print_mapping("Pattern counts", summary.pattern_counts)
    print(f"Sector-tagged symbols: {summary.sector_tagged_count:,}")
    print(f"No-sector symbols: {summary.no_sector_count:,}")
    print(f"Distinct sectors: {summary.distinct_sector_count:,}")
    _print_mapping("Sector symbol counts", summary.sector_symbol_counts)
    _print_mapping("Sector-tagged category counts", summary.sector_tagged_category_counts)
    print(
        "Sector-tagged non-common-equity instruments: "
        f"{summary.sector_tagged_non_common_equity_count:,}"
    )
    print(
        "Sector-tagged date range: "
        f"{summary.sector_tagged_first_market_date} to "
        f"{summary.sector_tagged_last_market_date}"
    )
    _print_mapping(
        "Sector-tagged observation quantiles",
        summary.sector_tagged_observation_quantiles,
    )
    _print_mapping("No-sector category counts", summary.no_sector_category_counts)
    print(f"No-sector UNKNOWN: {summary.no_sector_unknown_count:,}")
    print(
        "Plausible unverified historical-equity signals: "
        f"{summary.historical_equity_candidate_signal_count:,}"
    )
    for category in INSTRUMENT_CATEGORIES:
        print(
            f"Examples {category}: "
            + ", ".join(summary.category_examples[category])
        )
    for pattern in PATTERN_SIGNAL_NAMES:
        print(
            f"Examples pattern {pattern}: "
            + ", ".join(summary.pattern_examples[pattern])
        )
    print(
        "Examples no-sector UNKNOWN: "
        + ", ".join(summary.no_sector_unknown_examples)
    )
    print(
        "Examples plausible unverified historical equities: "
        + ", ".join(summary.historical_equity_candidate_examples)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit historical PSX instrument classifications without filtering."
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry", default=str(COMPANY_REGISTRY_PATH)
    )
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.overwrite and not args.output_csv:
        parser.error("--overwrite requires --output-csv")
    try:
        result = run_instrument_audit(
            parquet_path=args.path,
            registry_path=None if args.no_registry else args.company_registry,
        )
        _print_summary(result)
        if args.output_csv:
            output = write_instrument_audit_csv(
                result.instruments,
                args.output_csv,
                overwrite=args.overwrite,
                source_parquet_path=result.parquet_path,
            )
            print(f"Audit CSV: {output}")
        return 0
    except (
        InstrumentAuditError,
        MarketParquetError,
        UniverseAuditError,
        ValueError,
        TypeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())


__all__ = (
    "COMMON_EQUITY",
    "DEBT_OR_TFC",
    "ETF_OR_FUND",
    "GOVERNMENT_SECURITY",
    "INSTRUMENT_AUDIT_COLUMNS",
    "INSTRUMENT_CATEGORIES",
    "INSUFFICIENT_LOCAL_EVIDENCE",
    "InstrumentAuditError",
    "InstrumentAuditResult",
    "InstrumentAuditSummary",
    "OFFICIAL_LISTING_DERIVED",
    "OTHER_IDENTIFIED_INSTRUMENT",
    "PATTERN_SIGNAL_NAMES",
    "PATTERN_SIGNAL_ONLY",
    "RIGHTS_OR_VARIANT",
    "UNKNOWN",
    "classify_instrument_universe",
    "detect_symbol_pattern_signals",
    "load_registry_evidence",
    "main",
    "run_instrument_audit",
    "summarize_instrument_audit",
    "write_instrument_audit_csv",
)
