"""Deterministic current authoritative common-equity identity universe.

Identity membership is intentionally separate from later clustering
eligibility.  No history, liquidity, or data-quality threshold is applied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from .config import (
    COMPANY_REGISTRY_PATH,
    CURRENT_COMMON_EQUITY_UNIVERSE_PATH,
    CURRENT_LISTINGS_PATH,
    PROJECT_ROOT,
)
from .instrument_audit import (
    COMMON_EQUITY,
    OFFICIAL_LISTING_DERIVED,
    InstrumentAuditError,
    classify_instrument_universe,
    load_registry_evidence,
)
from .official_listings import ListingsUnavailableError, load_listing_snapshot
from .parquet_market_data import (
    MarketParquetError,
    load_market_data,
    resolve_market_parquet_path,
)
from .universe_audit import (
    QUANTILE_LABELS,
    QUANTILE_PROBABILITIES,
    UniverseAuditError,
    build_symbol_universe_audit,
    load_company_metadata,
)
from .universe_methodology import (
    UniverseMethodologyError,
    add_active_span_coverage,
)


EQUITY_UNIVERSE_VERSION = "current_common_equity_universe_v1"
IDENTITY_POLICY = "CURRENT_AUTHORITATIVE_COMMON_EQUITY"
CLASSIFICATION_POLICY_VERSION = "instrument_audit_authoritative_first_v1"
RESEARCH_LIMITATION = (
    "This universe uses current authoritative common-equity membership and "
    "therefore contains survivorship bias when historical observations are used. "
    "It is not a historical-dynamic or survivorship-free PSX universe."
)
ALLOWED_COMMON_EQUITY_SECURITY_TYPES = frozenset(
    {"ordinary_equity", "gem_equity"}
)

EQUITY_UNIVERSE_COLUMNS = (
    "symbol",
    "company_name",
    "sector",
    "board",
    "listing_segment",
    "security_type",
    "authoritative_source",
    "classification_basis",
    "instrument_category",
    "listing_snapshot_date",
    "first_market_date",
    "last_market_date",
    "observation_count",
    "active_span_coverage",
    "median_volume",
    "zero_volume_ratio",
    "zero_ohl_ratio",
)


class EquityUniverseError(RuntimeError):
    """Raised when the authoritative identity universe cannot be frozen safely."""


@dataclass(frozen=True)
class EquityUniverseProvenance:
    universe_version: str
    identity_policy: str
    classification_policy_version: str
    classification_basis: str
    instrument_category: str
    universe_count: int
    universe_hash: str
    listing_snapshot_date: str
    listing_refreshed_at: str
    authoritative_sources: tuple[str, ...]
    dataset_first_market_date: str
    dataset_last_market_date: str
    source_paths: Mapping[str, str]
    source_sha256: Mapping[str, str]
    identity_payload: Mapping[str, object]
    research_limitation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EquityUniverseSummary:
    universe_count: int
    universe_hash: str
    sector_count: int
    sector_symbol_counts: Mapping[str, int]
    security_type_counts: Mapping[str, int]
    dataset_first_market_date: str
    dataset_last_market_date: str
    observation_count_quantiles: Mapping[str, float | None]
    listing_snapshot_date: str
    research_limitation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EquityUniverseResult:
    records: pd.DataFrame = field(repr=False, compare=False)
    provenance: EquityUniverseProvenance
    summary: EquityUniverseSummary
    parquet_path: Path
    registry_path: Path
    listing_snapshot_path: Path


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_universe_identity(identity_payload: Mapping[str, object]) -> str:
    """Hash canonical stable identity content without paths or timestamps."""

    return _canonical_hash(dict(identity_payload))


def _single_snapshot_value(listings: pd.DataFrame, column: str) -> str:
    values = sorted(
        {
            str(value).strip()
            for value in listings[column].dropna()
            if str(value).strip()
        }
    )
    if len(values) != 1:
        raise EquityUniverseError(
            f"Official listings must contain exactly one {column}; found {values}"
        )
    return values[0]


def _validate_listing_evidence(listings: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol",
        "company_name",
        "security_type",
        "sector",
        "board",
        "listing_segment",
        "officially_listed",
        "source",
        "listing_refreshed_at",
        "snapshot_date",
    }
    missing = sorted(required.difference(listings.columns))
    if missing:
        raise EquityUniverseError(
            "Official listings are missing identity fields: " + ", ".join(missing)
        )
    evidence = listings.loc[:, sorted(required)].copy(deep=True)
    evidence["symbol"] = evidence["symbol"].astype("string").str.strip().str.upper()
    if evidence["symbol"].isna().any() or (evidence["symbol"] == "").any():
        raise EquityUniverseError("Official listings contain a blank symbol")
    if evidence["symbol"].duplicated().any():
        raise EquityUniverseError("Official listings contain duplicate symbols")
    listed = evidence["officially_listed"]
    if not pd.api.types.is_bool_dtype(listed):
        listed = listed.astype("string").str.strip().str.lower().isin({"true", "1"})
    if not listed.all():
        raise EquityUniverseError(
            "Current official listing snapshot contains a non-listed record"
        )
    evidence["officially_listed"] = listed.astype(bool)
    for column in (
        "company_name",
        "security_type",
        "sector",
        "board",
        "listing_segment",
        "source",
        "listing_refreshed_at",
        "snapshot_date",
    ):
        evidence[column] = evidence[column].astype("string").str.strip()
    if (evidence["source"] == "").any():
        raise EquityUniverseError("Official listings contain a blank source")
    return evidence.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def build_current_common_equity_universe(
    market: pd.DataFrame,
    *,
    registry: pd.DataFrame,
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Build identity records without applying history or liquidity eligibility."""

    metadata = registry.loc[:, ["symbol", "company_name", "sector"]].copy(deep=True)
    universe = build_symbol_universe_audit(market, metadata=metadata)
    classifications = classify_instrument_universe(universe, registry=registry)
    selected = classifications.loc[
        (classifications["instrument_category"] == COMMON_EQUITY)
        & classifications["authoritative_metadata_available"].astype(bool)
        & (classifications["classification_basis"] == OFFICIAL_LISTING_DERIVED)
    ].copy()
    if selected.empty:
        raise EquityUniverseError(
            "No authoritative current common equities were derived from local evidence"
        )
    if selected["symbol"].duplicated().any():
        raise EquityUniverseError("Classification produced duplicate common equities")

    audited = add_active_span_coverage(universe, market["market_date"])
    quality_columns = (
        "symbol",
        "first_market_date",
        "last_market_date",
        "observation_count",
        "active_span_coverage",
        "median_volume",
        "zero_volume_ratio",
        "zero_ohl_ratio",
    )
    records = selected.loc[
        :, ["symbol", "classification_basis", "instrument_category"]
    ].merge(
        audited.loc[:, quality_columns],
        on="symbol",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    listing_evidence = _validate_listing_evidence(listings).rename(
        columns={
            "source": "authoritative_source",
            "snapshot_date": "listing_snapshot_date",
        }
    )
    listing_columns = (
        "symbol",
        "company_name",
        "sector",
        "board",
        "listing_segment",
        "security_type",
        "authoritative_source",
        "listing_snapshot_date",
    )
    records = records.merge(
        listing_evidence.loc[:, listing_columns],
        on="symbol",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    missing_listing_symbols = sorted(
        records.loc[records["_merge"] != "both", "symbol"].astype(str)
    )
    if missing_listing_symbols:
        raise EquityUniverseError(
            "Authoritative common equities missing from the listing snapshot: "
            + ", ".join(missing_listing_symbols)
        )
    records = records.drop(columns="_merge")
    security_types = set(records["security_type"].astype(str))
    if not security_types.issubset(ALLOWED_COMMON_EQUITY_SECURITY_TYPES):
        unexpected = sorted(security_types.difference(ALLOWED_COMMON_EQUITY_SECURITY_TYPES))
        raise EquityUniverseError(
            "Common-equity classification conflicts with listing security types: "
            + ", ".join(unexpected)
        )
    if records.loc[:, EQUITY_UNIVERSE_COLUMNS].isna().any().any():
        missing_fields = sorted(
            column
            for column in EQUITY_UNIVERSE_COLUMNS
            if records[column].isna().any()
        )
        raise EquityUniverseError(
            "Authoritative common-equity records have missing fields: "
            + ", ".join(missing_fields)
        )
    return records.loc[:, EQUITY_UNIVERSE_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)


def build_identity_payload(records: pd.DataFrame) -> dict[str, object]:
    """Return stable membership and classification evidence for hashing."""

    missing = [column for column in EQUITY_UNIVERSE_COLUMNS if column not in records]
    if missing:
        raise EquityUniverseError(
            "Equity universe is missing identity fields: " + ", ".join(missing)
        )
    if records["symbol"].duplicated().any():
        raise EquityUniverseError("Equity universe contains duplicate symbols")
    members = []
    identity_columns = (
        "symbol",
        "instrument_category",
        "classification_basis",
        "security_type",
        "sector",
        "authoritative_source",
    )
    ordered = records.sort_values("symbol", kind="mergesort")
    for row in ordered.loc[:, identity_columns].to_dict(orient="records"):
        members.append({key: str(value) for key, value in row.items()})
    snapshot_dates = sorted(set(ordered["listing_snapshot_date"].astype(str)))
    if len(snapshot_dates) != 1:
        raise EquityUniverseError(
            "Equity universe records do not share one listing snapshot date"
        )
    return {
        "universe_version": EQUITY_UNIVERSE_VERSION,
        "identity_policy": IDENTITY_POLICY,
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
        "listing_snapshot_date": snapshot_dates[0],
        "members": members,
    }


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {label: None for label in QUANTILE_LABELS}
    measured = numeric.quantile(QUANTILE_PROBABILITIES, interpolation="linear")
    return {
        label: float(measured.iloc[index])
        for index, label in enumerate(QUANTILE_LABELS)
    }


def _sector_counts(records: pd.DataFrame) -> dict[str, int]:
    counts = records["sector"].astype(str).value_counts()
    return {
        sector: int(count)
        for sector, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    }


def _security_type_counts(records: pd.DataFrame) -> dict[str, int]:
    counts = records["security_type"].astype(str).value_counts()
    return {key: int(counts[key]) for key in sorted(counts.index)}


def run_equity_universe(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] = COMPANY_REGISTRY_PATH,
    listing_snapshot_path: str | os.PathLike[str] = CURRENT_LISTINGS_PATH,
) -> EquityUniverseResult:
    """Load local evidence and derive the frozen current identity universe."""

    resolved_parquet = resolve_market_parquet_path(parquet_path)
    resolved_registry = Path(registry_path).expanduser().resolve(strict=False)
    resolved_listings = Path(listing_snapshot_path).expanduser().resolve(strict=False)
    if not resolved_registry.is_file():
        raise EquityUniverseError(f"Company registry does not exist: {resolved_registry}")
    if not resolved_listings.is_file():
        raise EquityUniverseError(
            f"Official listing snapshot does not exist: {resolved_listings}"
        )
    market = load_market_data(resolved_parquet)
    registry = load_registry_evidence(resolved_registry)
    if registry is None:  # pragma: no cover - file existence checked above
        raise EquityUniverseError("Company registry evidence is unavailable")
    listings = load_listing_snapshot(resolved_listings)
    records = build_current_common_equity_universe(
        market,
        registry=registry,
        listings=listings,
    )
    identity_payload = build_identity_payload(records)
    universe_hash = deterministic_universe_identity(identity_payload)
    snapshot_date = _single_snapshot_value(listings, "snapshot_date")
    listing_refreshed_at = _single_snapshot_value(listings, "listing_refreshed_at")
    sources = tuple(sorted(set(records["authoritative_source"].astype(str))))
    market_dates = pd.to_datetime(market["market_date"], errors="raise")
    dataset_first = market_dates.min().date().isoformat()
    dataset_last = market_dates.max().date().isoformat()
    provenance = EquityUniverseProvenance(
        universe_version=EQUITY_UNIVERSE_VERSION,
        identity_policy=IDENTITY_POLICY,
        classification_policy_version=CLASSIFICATION_POLICY_VERSION,
        classification_basis=OFFICIAL_LISTING_DERIVED,
        instrument_category=COMMON_EQUITY,
        universe_count=len(records),
        universe_hash=universe_hash,
        listing_snapshot_date=snapshot_date,
        listing_refreshed_at=listing_refreshed_at,
        authoritative_sources=sources,
        dataset_first_market_date=dataset_first,
        dataset_last_market_date=dataset_last,
        source_paths={
            "market_parquet": _portable_path(resolved_parquet),
            "company_registry": _portable_path(resolved_registry),
            "official_listings": _portable_path(resolved_listings),
        },
        source_sha256={
            "market_parquet": _sha256_file(resolved_parquet),
            "company_registry": _sha256_file(resolved_registry),
            "official_listings": _sha256_file(resolved_listings),
        },
        identity_payload=identity_payload,
        research_limitation=RESEARCH_LIMITATION,
    )
    summary = EquityUniverseSummary(
        universe_count=len(records),
        universe_hash=universe_hash,
        sector_count=int(records["sector"].nunique()),
        sector_symbol_counts=_sector_counts(records),
        security_type_counts=_security_type_counts(records),
        dataset_first_market_date=dataset_first,
        dataset_last_market_date=dataset_last,
        observation_count_quantiles=_quantiles(records["observation_count"]),
        listing_snapshot_date=snapshot_date,
        research_limitation=RESEARCH_LIMITATION,
    )
    return EquityUniverseResult(
        records=records,
        provenance=provenance,
        summary=summary,
        parquet_path=resolved_parquet,
        registry_path=resolved_registry,
        listing_snapshot_path=resolved_listings,
    )


def _sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".provenance.json")


def write_equity_universe_artifacts(
    result: EquityUniverseResult,
    output_csv: str | os.PathLike[str] = CURRENT_COMMON_EQUITY_UNIVERSE_PATH,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write deterministic CSV/JSON artifacts only after an explicit request."""

    csv_path = Path(output_csv).expanduser().resolve(strict=False)
    if csv_path.suffix.lower() != ".csv":
        raise EquityUniverseError("Equity universe output must use a .csv suffix")
    json_path = _sidecar_path(csv_path)
    protected_sources = {
        result.parquet_path.resolve(),
        result.registry_path.resolve(),
        result.listing_snapshot_path.resolve(),
    }
    if csv_path in protected_sources or json_path in protected_sources:
        raise EquityUniverseError("Universe outputs cannot replace source data")
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not overwrite:
        raise EquityUniverseError(
            "Universe artifact exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_descriptor, csv_temporary_name = tempfile.mkstemp(
        prefix=f".{csv_path.stem}.", suffix=".csv.tmp", dir=csv_path.parent
    )
    json_descriptor, json_temporary_name = tempfile.mkstemp(
        prefix=f".{json_path.stem}.", suffix=".json.tmp", dir=json_path.parent
    )
    os.close(csv_descriptor)
    os.close(json_descriptor)
    csv_temporary = Path(csv_temporary_name)
    json_temporary = Path(json_temporary_name)
    try:
        result.records.loc[:, EQUITY_UNIVERSE_COLUMNS].to_csv(
            csv_temporary,
            index=False,
            lineterminator="\n",
            na_rep="",
            float_format="%.12g",
            date_format="%Y-%m-%d",
        )
        json_temporary.write_text(
            json.dumps(result.provenance.to_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if not overwrite:
            raced = [path for path in (csv_path, json_path) if path.exists()]
            if raced:
                raise EquityUniverseError(
                    "Universe artifact appeared during write; refusing replacement: "
                    + ", ".join(str(path) for path in raced)
                )
        os.replace(csv_temporary, csv_path)
        os.replace(json_temporary, json_path)
    finally:
        csv_temporary.unlink(missing_ok=True)
        json_temporary.unlink(missing_ok=True)
    return csv_path, json_path


def _print_summary(result: EquityUniverseResult) -> None:
    summary = result.summary
    provenance = result.provenance
    print(f"Universe version: {provenance.universe_version}")
    print(f"Identity policy: {provenance.identity_policy}")
    print(f"Derived common-equity count: {summary.universe_count:,}")
    print(f"Universe identity: {summary.universe_hash}")
    print(f"Listing snapshot date: {summary.listing_snapshot_date}")
    print(f"Listing refreshed at: {provenance.listing_refreshed_at}")
    print("Authoritative sources: " + ", ".join(provenance.authoritative_sources))
    print(
        f"Dataset date range: {summary.dataset_first_market_date} to "
        f"{summary.dataset_last_market_date}"
    )
    print(f"Sectors represented: {summary.sector_count}")
    print(
        "Sector counts: "
        + json.dumps(dict(summary.sector_symbol_counts), sort_keys=False)
    )
    print(
        "Security-type counts: "
        + json.dumps(dict(summary.security_type_counts), sort_keys=True)
    )
    print(
        "Observation-count quantiles: "
        + json.dumps(dict(summary.observation_count_quantiles), sort_keys=False)
    )
    print(f"Research limitation: {summary.research_limitation}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen current authoritative common-equity universe."
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry",
        default=str(COMPANY_REGISTRY_PATH),
        help="Existing company registry CSV.",
    )
    parser.add_argument(
        "--listing-snapshot",
        default=str(CURRENT_LISTINGS_PATH),
        help="Existing official PSX listing snapshot CSV.",
    )
    parser.add_argument(
        "--output-csv",
        nargs="?",
        const=str(CURRENT_COMMON_EQUITY_UNIVERSE_PATH),
        help="Explicitly write CSV and provenance JSON; optional path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing explicitly requested universe artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.overwrite and not args.output_csv:
        parser.error("--overwrite requires --output-csv")
    try:
        result = run_equity_universe(
            parquet_path=args.path,
            registry_path=args.company_registry,
            listing_snapshot_path=args.listing_snapshot,
        )
        _print_summary(result)
        if args.output_csv:
            csv_path, json_path = write_equity_universe_artifacts(
                result,
                args.output_csv,
                overwrite=args.overwrite,
            )
            print(f"Universe CSV: {csv_path}")
            print(f"Provenance JSON: {json_path}")
        return 0
    except (
        EquityUniverseError,
        InstrumentAuditError,
        ListingsUnavailableError,
        MarketParquetError,
        UniverseAuditError,
        UniverseMethodologyError,
        ValueError,
        TypeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())


__all__ = (
    "ALLOWED_COMMON_EQUITY_SECURITY_TYPES",
    "CLASSIFICATION_POLICY_VERSION",
    "EQUITY_UNIVERSE_COLUMNS",
    "EQUITY_UNIVERSE_VERSION",
    "EquityUniverseError",
    "EquityUniverseProvenance",
    "EquityUniverseResult",
    "EquityUniverseSummary",
    "IDENTITY_POLICY",
    "RESEARCH_LIMITATION",
    "build_current_common_equity_universe",
    "build_identity_payload",
    "deterministic_universe_identity",
    "main",
    "run_equity_universe",
    "write_equity_universe_artifacts",
)
