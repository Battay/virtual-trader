"""Read-only equity-universe methodology and survivorship-bias audit.

This module deliberately produces diagnostics, not an eligibility decision.
Current official common equities are the reference population; historical
UNKNOWN identifiers remain unverified and are never promoted by this audit.
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
from .instrument_audit import (
    COMMON_EQUITY,
    UNKNOWN,
    InstrumentAuditError,
    classify_instrument_universe,
    load_registry_evidence,
)
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


METHODOLOGY_VERSION = "equity_universe_methodology_audit_v1"

CURRENT_UNIVERSE = "CURRENT_UNIVERSE"
HISTORICAL_DYNAMIC_UNIVERSE = "HISTORICAL_DYNAMIC_UNIVERSE"
HYBRID_RESEARCH_UNIVERSE = "HYBRID_RESEARCH_UNIVERSE"

STRONG_HISTORICAL_EQUITY_CANDIDATE = (
    "STRONG_HISTORICAL_EQUITY_CANDIDATE"
)
SPARSE_OR_UNCERTAIN = "SPARSE_OR_UNCERTAIN"
LIKELY_OTHER_INSTRUMENT = "LIKELY_OTHER_INSTRUMENT"
UNRESOLVED = "UNRESOLVED"
UNKNOWN_DIAGNOSTIC_GROUPS = (
    STRONG_HISTORICAL_EQUITY_CANDIDATE,
    SPARSE_OR_UNCERTAIN,
    LIKELY_OTHER_INSTRUMENT,
    UNRESOLVED,
)

NO_LOCAL_ALIAS_EVIDENCE = "NO_LOCAL_ALIAS_EVIDENCE"
LEXICAL_PREFIX_SIGNAL_ONLY = "LEXICAL_PREFIX_SIGNAL_ONLY"
REGISTRY_SUCCESSOR_LINK = "REGISTRY_SUCCESSOR_LINK"

UNKNOWN_PATTERN_GOVERNMENT_GIS = "government_gis_identifier"
UNKNOWN_PATTERN_CORPORATE_ACTION_SC = "corporate_action_sc_suffix"
UNKNOWN_PATTERN_PREFERENCE_CPS = "preference_cps_suffix"
UNKNOWN_PATTERN_NON_VOTING = "non_voting_nv_suffix"
UNKNOWN_GOVERNMENT_GIS_PATTERN = re.compile(r"^P\d{2}GIS\d+$")
UNKNOWN_CORPORATE_ACTION_SC_PATTERN = re.compile(r"SC\d*$")
UNKNOWN_PREFERENCE_CPS_PATTERN = re.compile(r"CPS[A-Z0-9]*$")
UNKNOWN_NON_VOTING_PATTERN = re.compile(r"NV$")

HISTORY_WINDOWS_DAYS = {
    "1y": 365,
    "2y": 730,
    "3y": 1_095,
    "5y": 1_825,
}
ACTIVE_SPAN_COVERAGE_LEVELS = (0.50, 0.70, 0.80, 0.90)
LIQUIDITY_QUANTILES = {
    "none": None,
    "current_common_p10": 0.10,
    "current_common_p25": 0.25,
    "current_common_p50": 0.50,
}

QUALITY_QUANTILE_METRICS = (
    "observation_count",
    "calendar_span_days",
    "coverage_ratio",
    "active_span_coverage",
    "median_volume",
    "average_volume",
    "zero_volume_ratio",
    "zero_ohl_ratio",
)

METHODOLOGY_AUDIT_COLUMNS = (
    "symbol",
    "instrument_category",
    "classification_basis",
    "authoritative_metadata_available",
    "diagnostic_group",
    "diagnostic_reason",
    "pattern_signals",
    "methodology_pattern_signals",
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
    "alias_evidence_status",
    "current_symbol_lexical_candidates",
    "first_market_date",
    "last_market_date",
    "observation_count",
    "calendar_span_days",
    "unique_market_dates",
    "global_market_date_count",
    "active_span_market_date_count",
    "coverage_ratio",
    "active_span_coverage",
    "average_volume",
    "median_volume",
    "zero_volume_count",
    "zero_volume_ratio",
    "rows_with_any_zero_ohl",
    "zero_ohl_ratio",
    "positive_close_count",
    "positive_close_ratio",
)

SENSITIVITY_COLUMNS = (
    "population",
    "history_window",
    "minimum_calendar_span_days",
    "minimum_active_span_coverage",
    "liquidity_cutoff",
    "minimum_median_volume",
    "symbol_count",
)


class UniverseMethodologyError(RuntimeError):
    """Raised when methodology diagnostics cannot be produced safely."""


@dataclass(frozen=True)
class UniverseStrategyAssessment:
    name: str
    definition: str
    benefits: tuple[str, ...]
    weaknesses: tuple[str, ...]
    survivorship_bias: str
    implementation_complexity: str
    architecture_suitability: str


STRATEGY_ASSESSMENTS = (
    UniverseStrategyAssessment(
        name=CURRENT_UNIVERSE,
        definition="Current official-listing-derived common equities only.",
        benefits=(
            "Highest locally supported instrument identity confidence.",
            "Simple reproducible anchor for clustering and later RL experiments.",
        ),
        weaknesses=(
            "Omits delisted, renamed, and merged historical companies.",
            "Current membership does not prove membership at an earlier cutoff.",
        ),
        survivorship_bias=(
            "High if used to make historical market-wide claims; failed or exited "
            "companies are systematically absent."
        ),
        implementation_complexity="Low",
        architecture_suitability=(
            "Useful as a clean baseline, but not sufficient for unbiased historical "
            "cluster or transfer claims."
        ),
    ),
    UniverseStrategyAssessment(
        name=HISTORICAL_DYNAMIC_UNIVERSE,
        definition=(
            "Point-in-time common equities with verified listing, delisting, rename, "
            "merger, and identity-effective intervals."
        ),
        benefits=(
            "Best control of survivorship and future-membership leakage.",
            "Supports historically faithful clustering cohorts.",
        ),
        weaknesses=(
            "Required point-in-time identity evidence is absent locally for UNKNOWNs.",
            "Incorrect alias stitching could duplicate or fabricate company histories.",
        ),
        survivorship_bias="Lowest when supported by authoritative point-in-time evidence.",
        implementation_complexity="High; currently blocked by metadata provenance.",
        architecture_suitability=(
            "Scientifically strongest long-term design for dynamic clusters and shared "
            "recurrent policies."
        ),
    ),
    UniverseStrategyAssessment(
        name=HYBRID_RESEARCH_UNIVERSE,
        definition=(
            "Current authoritative common-equity anchor plus separately evidenced "
            "historical equities and aliases; unverified UNKNOWNs remain excluded."
        ),
        benefits=(
            "Preserves a reproducible high-confidence anchor.",
            "Allows survivorship bias to decline as historical evidence is recovered.",
            "Keeps evidence tier and identity interval explicit per constituent.",
        ),
        weaknesses=(
            "Initially retains some survivorship bias because no UNKNOWN is promoted.",
            "Requires careful alias and effective-date review before each recovery.",
        ),
        survivorship_bias=(
            "Lower than current-only after verified historical recovery, but residual "
            "bias must be measured and disclosed."
        ),
        implementation_complexity="Moderate and incremental",
        architecture_suitability=(
            "Best current foundation: evidence tiers can feed point-in-time clustering, "
            "symbol-isolated episodes, and later robustness comparisons."
        ),
    ),
)


@dataclass(frozen=True)
class UniverseMethodologySummary:
    methodology_version: str
    global_market_date_count: int
    current_common_equity_count: int
    unknown_count: int
    current_common_history_window_counts: Mapping[str, int]
    current_common_quantiles: Mapping[str, Mapping[str, float | None]]
    current_common_first_date_quantiles: Mapping[str, str | None]
    current_common_last_date_quantiles: Mapping[str, str | None]
    unknown_quantiles: Mapping[str, Mapping[str, float | None]]
    strong_candidate_quantiles: Mapping[str, Mapping[str, float | None]]
    unknown_diagnostic_group_counts: Mapping[str, int]
    unknown_diagnostic_group_examples: Mapping[str, tuple[str, ...]]
    unknown_alias_evidence_counts: Mapping[str, int]
    unknown_alias_signal_examples: tuple[Mapping[str, str], ...]
    liquidity_cutoffs: Mapping[str, float | None]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["unknown_diagnostic_group_examples"] = {
            key: list(value)
            for key, value in self.unknown_diagnostic_group_examples.items()
        }
        payload["unknown_alias_signal_examples"] = [
            dict(row) for row in self.unknown_alias_signal_examples
        ]
        return payload


@dataclass(frozen=True)
class UniverseMethodologyResult:
    symbols: pd.DataFrame = field(repr=False, compare=False)
    current_common: pd.DataFrame = field(repr=False, compare=False)
    unknowns: pd.DataFrame = field(repr=False, compare=False)
    sensitivity: pd.DataFrame = field(repr=False, compare=False)
    summary: UniverseMethodologySummary
    parquet_path: Path
    registry_path: Path | None


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {label: None for label in QUANTILE_LABELS}
    measured = numeric.quantile(QUANTILE_PROBABILITIES, interpolation="linear")
    return {
        label: float(measured.iloc[index])
        for index, label in enumerate(QUANTILE_LABELS)
    }


def _date_quantiles(values: pd.Series) -> dict[str, str | None]:
    dates = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    if dates.empty:
        return {label: None for label in QUANTILE_LABELS}
    numeric = pd.Series(
        dates.astype("datetime64[ns]").astype("int64"),
        dtype="float64",
    )
    measured = numeric.quantile(
        QUANTILE_PROBABILITIES,
        interpolation="linear",
    )
    return {
        label: pd.Timestamp(round(float(measured.iloc[index]))).date().isoformat()
        for index, label in enumerate(QUANTILE_LABELS)
    }


def add_active_span_coverage(
    universe: pd.DataFrame,
    global_market_dates: Sequence[object] | pd.Series | pd.Index,
) -> pd.DataFrame:
    """Add coverage using actual global trading dates within each symbol span."""

    required = {
        "first_market_date",
        "last_market_date",
        "unique_market_dates",
    }
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise UniverseMethodologyError(
            "Universe is missing active-span columns: " + ", ".join(missing)
        )
    dates = pd.DatetimeIndex(pd.to_datetime(global_market_dates, errors="coerce"))
    if dates.isna().any():
        raise UniverseMethodologyError("Global market date set contains invalid dates")
    dates = dates.unique().sort_values()
    if dates.empty:
        raise UniverseMethodologyError("Global market date set is empty")

    enriched = universe.copy(deep=True)
    first_dates = pd.to_datetime(enriched["first_market_date"], errors="coerce")
    last_dates = pd.to_datetime(enriched["last_market_date"], errors="coerce")
    if first_dates.isna().any() or last_dates.isna().any():
        raise UniverseMethodologyError("Universe contains invalid first/last dates")
    if (first_dates > last_dates).any():
        raise UniverseMethodologyError("Universe contains first dates after last dates")

    denominators = [
        int(dates.searchsorted(last, side="right") - dates.searchsorted(first, side="left"))
        for first, last in zip(first_dates, last_dates, strict=True)
    ]
    enriched["global_market_date_count"] = int(len(dates))
    enriched["active_span_market_date_count"] = pd.Series(
        denominators,
        index=enriched.index,
        dtype="int64",
    )
    if (enriched["active_span_market_date_count"] <= 0).any():
        raise UniverseMethodologyError(
            "A symbol span contains no dates from the global market calendar"
        )
    unique_dates = pd.to_numeric(enriched["unique_market_dates"], errors="coerce")
    if unique_dates.isna().any():
        raise UniverseMethodologyError("Universe contains invalid unique date counts")
    if (unique_dates > enriched["active_span_market_date_count"]).any():
        raise UniverseMethodologyError(
            "A symbol has more observed dates than its active-span denominator"
        )
    enriched["active_span_coverage"] = (
        unique_dates / enriched["active_span_market_date_count"]
    ).astype("float64")
    return enriched


def _lexical_current_symbol_candidates(
    symbol: str,
    current_symbols: Sequence[str],
) -> tuple[str, ...]:
    """Return narrow ticker-prefix signals; these are never treated as aliases."""

    normalized = str(symbol).strip().upper()
    matches = {
        candidate
        for candidate in current_symbols
        if candidate != normalized
        and min(len(normalized), len(candidate)) >= 3
        and abs(len(normalized) - len(candidate)) <= 3
        and (normalized.startswith(candidate) or candidate.startswith(normalized))
    }
    return tuple(sorted(matches))


def add_alias_diagnostics(
    unknowns: pd.DataFrame,
    current_symbols: Sequence[str],
) -> pd.DataFrame:
    """Expose registry links and lexical signals without asserting identity."""

    result = unknowns.copy(deep=True)
    candidates: list[str] = []
    statuses: list[str] = []
    current_set = set(str(symbol) for symbol in current_symbols)
    for row in result.to_dict(orient="records"):
        registry_successor = row.get("successor_symbol")
        successor = "" if pd.isna(registry_successor) else str(registry_successor).strip()
        lexical = _lexical_current_symbol_candidates(
            str(row["symbol"]),
            tuple(sorted(current_set)),
        )
        if successor and successor in current_set:
            status = REGISTRY_SUCCESSOR_LINK
            lexical = tuple(sorted(set(lexical).union({successor})))
        elif lexical:
            status = LEXICAL_PREFIX_SIGNAL_ONLY
        else:
            status = NO_LOCAL_ALIAS_EVIDENCE
        statuses.append(status)
        candidates.append("|".join(lexical))
    result["alias_evidence_status"] = statuses
    result["current_symbol_lexical_candidates"] = candidates
    return result


def detect_unknown_methodology_pattern_signals(symbol: str) -> tuple[str, ...]:
    """Return narrow extra diagnostics that do not alter 7B.1 classification."""

    normalized = str(symbol).strip().upper()
    checks = (
        (UNKNOWN_PATTERN_GOVERNMENT_GIS, UNKNOWN_GOVERNMENT_GIS_PATTERN),
        (
            UNKNOWN_PATTERN_CORPORATE_ACTION_SC,
            UNKNOWN_CORPORATE_ACTION_SC_PATTERN,
        ),
        (UNKNOWN_PATTERN_PREFERENCE_CPS, UNKNOWN_PREFERENCE_CPS_PATTERN),
        (UNKNOWN_PATTERN_NON_VOTING, UNKNOWN_NON_VOTING_PATTERN),
    )
    return tuple(name for name, pattern in checks if pattern.search(normalized))


def _reference_cutoffs(current_common: pd.DataFrame) -> dict[str, float]:
    if current_common.empty:
        raise UniverseMethodologyError(
            "Current common-equity reference population is empty"
        )
    return {
        "observation_p10": float(current_common["observation_count"].quantile(0.10)),
        "observation_p25": float(current_common["observation_count"].quantile(0.25)),
        "active_coverage_p10": float(
            current_common["active_span_coverage"].quantile(0.10)
        ),
        "active_coverage_p25": float(
            current_common["active_span_coverage"].quantile(0.25)
        ),
        "zero_ohl_p75": float(current_common["zero_ohl_ratio"].quantile(0.75)),
    }


def classify_unknown_diagnostics(
    unknowns: pd.DataFrame,
    current_common: pd.DataFrame,
) -> pd.DataFrame:
    """Assign non-authoritative, data-derived diagnostic confidence groups."""

    required = {
        "symbol",
        "observation_count",
        "active_span_coverage",
        "zero_ohl_ratio",
        "positive_close_ratio",
        "pattern_signals",
        "methodology_pattern_signals",
        "registry_security_type",
    }
    missing = sorted(required.difference(unknowns.columns))
    if missing:
        raise UniverseMethodologyError(
            "UNKNOWN diagnostics are missing columns: " + ", ".join(missing)
        )
    cutoffs = _reference_cutoffs(current_common)
    result = unknowns.copy(deep=True)
    groups: list[str] = []
    reasons: list[str] = []
    identified_non_equity_types = {"right", "etf", "preference_share", "other"}

    for row in result.sort_values("symbol", kind="mergesort").to_dict(
        orient="records"
    ):
        pattern_signals = str(row.get("pattern_signals") or "")
        methodology_signals = str(row.get("methodology_pattern_signals") or "")
        registry_type = str(row.get("registry_security_type") or "").lower()
        other_signal = (
            registry_type in identified_non_equity_types
            or methodology_signals != ""
        )
        strong = (
            pattern_signals == "plain_symbol"
            and float(row["observation_count"]) >= cutoffs["observation_p25"]
            and float(row["active_span_coverage"])
            >= cutoffs["active_coverage_p25"]
            and float(row["zero_ohl_ratio"]) <= cutoffs["zero_ohl_p75"]
            and float(row["positive_close_ratio"]) == 1.0
        )
        sparse = (
            float(row["observation_count"]) < cutoffs["observation_p10"]
            or float(row["active_span_coverage"])
            < cutoffs["active_coverage_p10"]
        )
        if other_signal:
            group = LIKELY_OTHER_INSTRUMENT
            reason = (
                "Non-equity registry type or methodology-level government/variant "
                "ticker signal; still not an authoritative historical classification."
            )
        elif strong:
            group = STRONG_HISTORICAL_EQUITY_CANDIDATE
            reason = (
                "Plain identifier meets current-common p25 history/active-coverage "
                "benchmarks, current-common p75 zero-OHL bound, and positive-close policy."
            )
        elif sparse:
            group = SPARSE_OR_UNCERTAIN
            reason = (
                "Falls below the current-common p10 observation or active-span "
                "coverage benchmark."
            )
        else:
            group = UNRESOLVED
            reason = (
                "Equity-like plain identifier, but local continuity/quality evidence "
                "does not reach the strong diagnostic band."
            )
        groups.append(group)
        reasons.append(reason)

    ordered = result.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    ordered["diagnostic_group"] = groups
    ordered["diagnostic_reason"] = reasons
    return ordered


def build_sensitivity_table(current_common: pd.DataFrame) -> pd.DataFrame:
    """Count the current authoritative anchor under predeclared cutoff grids."""

    required = {
        "calendar_span_days",
        "active_span_coverage",
        "median_volume",
    }
    missing = sorted(required.difference(current_common.columns))
    if missing:
        raise UniverseMethodologyError(
            "Sensitivity population is missing columns: " + ", ".join(missing)
        )
    if current_common.empty:
        raise UniverseMethodologyError("Sensitivity population is empty")
    median_volume = pd.to_numeric(current_common["median_volume"], errors="coerce")
    cutoffs = {
        label: (
            None if quantile is None else float(median_volume.quantile(quantile))
        )
        for label, quantile in LIQUIDITY_QUANTILES.items()
    }
    rows: list[dict[str, object]] = []
    for history_label, days in HISTORY_WINDOWS_DAYS.items():
        for active_coverage in ACTIVE_SPAN_COVERAGE_LEVELS:
            for liquidity_label, minimum_volume in cutoffs.items():
                mask = (
                    pd.to_numeric(
                        current_common["calendar_span_days"], errors="coerce"
                    )
                    >= days
                ) & (
                    pd.to_numeric(
                        current_common["active_span_coverage"], errors="coerce"
                    )
                    >= active_coverage
                )
                if minimum_volume is not None:
                    mask &= median_volume >= minimum_volume
                rows.append(
                    {
                        "population": "CURRENT_OFFICIAL_COMMON_EQUITY_ANCHOR",
                        "history_window": history_label,
                        "minimum_calendar_span_days": days,
                        "minimum_active_span_coverage": active_coverage,
                        "liquidity_cutoff": liquidity_label,
                        "minimum_median_volume": minimum_volume,
                        "symbol_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)


def _quality_quantiles(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {metric: _quantiles(frame[metric]) for metric in QUALITY_QUANTILE_METRICS}


def _ordered_counts(values: pd.Series, keys: Sequence[str]) -> dict[str, int]:
    counts = values.value_counts()
    return {key: int(counts.get(key, 0)) for key in keys}


def summarize_universe_methodology(
    current_common: pd.DataFrame,
    unknowns: pd.DataFrame,
    *,
    global_market_date_count: int,
) -> UniverseMethodologySummary:
    """Summarize facts without turning any diagnostic band into eligibility."""

    history_counts = {
        label: int((current_common["calendar_span_days"] >= days).sum())
        for label, days in HISTORY_WINDOWS_DAYS.items()
    }
    liquidity_cutoffs = {
        label: (
            None
            if quantile is None
            else float(current_common["median_volume"].quantile(quantile))
        )
        for label, quantile in LIQUIDITY_QUANTILES.items()
    }
    group_examples = {
        group: tuple(
            unknowns.loc[unknowns["diagnostic_group"] == group]
            .sort_values(
                ["observation_count", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            )["symbol"]
            .astype(str)
            .head(15)
        )
        for group in UNKNOWN_DIAGNOSTIC_GROUPS
    }
    alias_keys = (
        REGISTRY_SUCCESSOR_LINK,
        LEXICAL_PREFIX_SIGNAL_ONLY,
        NO_LOCAL_ALIAS_EVIDENCE,
    )
    alias_signals = unknowns.loc[
        unknowns["alias_evidence_status"] != NO_LOCAL_ALIAS_EVIDENCE,
        ["symbol", "alias_evidence_status", "current_symbol_lexical_candidates"],
    ].sort_values("symbol", kind="mergesort")
    strong_candidates = unknowns.loc[
        unknowns["diagnostic_group"] == STRONG_HISTORICAL_EQUITY_CANDIDATE
    ]
    return UniverseMethodologySummary(
        methodology_version=METHODOLOGY_VERSION,
        global_market_date_count=int(global_market_date_count),
        current_common_equity_count=len(current_common),
        unknown_count=len(unknowns),
        current_common_history_window_counts=history_counts,
        current_common_quantiles=_quality_quantiles(current_common),
        current_common_first_date_quantiles=_date_quantiles(
            current_common["first_market_date"]
        ),
        current_common_last_date_quantiles=_date_quantiles(
            current_common["last_market_date"]
        ),
        unknown_quantiles=_quality_quantiles(unknowns),
        strong_candidate_quantiles=_quality_quantiles(strong_candidates),
        unknown_diagnostic_group_counts=_ordered_counts(
            unknowns["diagnostic_group"], UNKNOWN_DIAGNOSTIC_GROUPS
        ),
        unknown_diagnostic_group_examples=group_examples,
        unknown_alias_evidence_counts=_ordered_counts(
            unknowns["alias_evidence_status"], alias_keys
        ),
        unknown_alias_signal_examples=tuple(alias_signals.to_dict(orient="records")),
        liquidity_cutoffs=liquidity_cutoffs,
    )


def build_methodology_universe(
    market: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None,
    registry: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full, current-common, UNKNOWN, and sensitivity diagnostics."""

    universe = build_symbol_universe_audit(market, metadata=metadata)
    classifications = classify_instrument_universe(universe, registry=registry)
    classification_columns = [
        column
        for column in classifications.columns
        if column
        not in {
            "company_name",
            "sector",
            "first_market_date",
            "last_market_date",
            "observation_count",
            "calendar_span_days",
            "unique_market_dates",
            "coverage_ratio",
        }
    ]
    combined = universe.merge(
        classifications.loc[:, classification_columns],
        on="symbol",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    combined = add_active_span_coverage(combined, market["market_date"])
    current_common = combined.loc[
        (combined["instrument_category"] == COMMON_EQUITY)
        & combined["authoritative_metadata_available"].astype(bool)
    ].copy()
    unknowns = combined.loc[combined["instrument_category"] == UNKNOWN].copy()
    current_symbols = tuple(sorted(current_common["symbol"].astype(str)))
    unknowns["methodology_pattern_signals"] = unknowns["symbol"].map(
        lambda symbol: "|".join(
            detect_unknown_methodology_pattern_signals(str(symbol))
        )
    )
    unknowns = add_alias_diagnostics(unknowns, current_symbols)
    unknowns = classify_unknown_diagnostics(unknowns, current_common)

    combined["diagnostic_group"] = pd.NA
    combined["diagnostic_reason"] = pd.NA
    combined["methodology_pattern_signals"] = pd.NA
    combined["alias_evidence_status"] = pd.NA
    combined["current_symbol_lexical_candidates"] = pd.NA
    unknown_updates = unknowns.set_index("symbol")
    for column in (
        "diagnostic_group",
        "diagnostic_reason",
        "methodology_pattern_signals",
        "alias_evidence_status",
        "current_symbol_lexical_candidates",
    ):
        combined[column] = combined["symbol"].map(unknown_updates[column])
    combined = combined.loc[:, METHODOLOGY_AUDIT_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
    current_common = combined.loc[
        combined["instrument_category"] == COMMON_EQUITY
    ].reset_index(drop=True)
    unknowns = combined.loc[
        combined["instrument_category"] == UNKNOWN
    ].reset_index(drop=True)
    sensitivity = build_sensitivity_table(current_common)
    return combined, current_common, unknowns, sensitivity


def run_universe_methodology_audit(
    *,
    parquet_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] | None = COMPANY_REGISTRY_PATH,
) -> UniverseMethodologyResult:
    """Run the complete methodology audit through the read-only boundary."""

    resolved_parquet = resolve_market_parquet_path(parquet_path)
    market = load_market_data(resolved_parquet)
    metadata = load_company_metadata(registry_path)
    registry = load_registry_evidence(registry_path)
    symbols, current_common, unknowns, sensitivity = build_methodology_universe(
        market,
        metadata=metadata,
        registry=registry,
    )
    summary = summarize_universe_methodology(
        current_common,
        unknowns,
        global_market_date_count=int(pd.Series(market["market_date"]).nunique()),
    )
    resolved_registry = (
        Path(registry_path).expanduser().resolve(strict=False)
        if registry_path is not None and Path(registry_path).expanduser().is_file()
        else None
    )
    return UniverseMethodologyResult(
        symbols=symbols,
        current_common=current_common,
        unknowns=unknowns,
        sensitivity=sensitivity,
        summary=summary,
        parquet_path=resolved_parquet,
        registry_path=resolved_registry,
    )


def _print_quantile_group(
    label: str,
    quantiles: Mapping[str, Mapping[str, float | None]],
) -> None:
    print(label + ":")
    for metric, values in quantiles.items():
        print(f"  {metric}: {json.dumps(dict(values), sort_keys=False)}")


def _print_strategies() -> None:
    print("Candidate strategy comparison:")
    for strategy in STRATEGY_ASSESSMENTS:
        print(f"  {strategy.name}: {strategy.definition}")
        print(f"    Benefits: {'; '.join(strategy.benefits)}")
        print(f"    Weaknesses: {'; '.join(strategy.weaknesses)}")
        print(f"    Survivorship bias: {strategy.survivorship_bias}")
        print(f"    Complexity: {strategy.implementation_complexity}")
        print(f"    Architecture: {strategy.architecture_suitability}")


def _print_sensitivity(sensitivity: pd.DataFrame) -> None:
    print("Sensitivity tables (rows=history, columns=active-span coverage):")
    for liquidity_label in LIQUIDITY_QUANTILES:
        subset = sensitivity.loc[sensitivity["liquidity_cutoff"] == liquidity_label]
        cutoff = subset["minimum_median_volume"].iloc[0]
        cutoff_text = "none" if pd.isna(cutoff) else str(float(cutoff))
        print(
            f"  Liquidity {liquidity_label} "
            f"(minimum median volume={cutoff_text}):"
        )
        pivot = subset.pivot(
            index="history_window",
            columns="minimum_active_span_coverage",
            values="symbol_count",
        ).reindex(index=HISTORY_WINDOWS_DAYS, columns=ACTIVE_SPAN_COVERAGE_LEVELS)
        for history_label, row in pivot.iterrows():
            counts = ", ".join(
                f"{coverage:.0%}={int(row[coverage])}"
                for coverage in ACTIVE_SPAN_COVERAGE_LEVELS
            )
            print(f"    {history_label}: {counts}")


def _print_summary(result: UniverseMethodologyResult) -> None:
    summary = result.summary
    print(f"Methodology version: {summary.methodology_version}")
    print(f"Parquet: {result.parquet_path}")
    print(f"Registry: {result.registry_path or 'not available'}")
    print(f"Global market dates: {summary.global_market_date_count:,}")
    print(f"Current common equities: {summary.current_common_equity_count:,}")
    print(f"UNKNOWN historical identifiers: {summary.unknown_count:,}")
    print(
        "Current-common history windows: "
        + json.dumps(dict(summary.current_common_history_window_counts))
    )
    _print_quantile_group(
        "Current-common quality quantiles", summary.current_common_quantiles
    )
    print(
        "Current-common first-date quantiles: "
        + json.dumps(dict(summary.current_common_first_date_quantiles))
    )
    print(
        "Current-common last-date quantiles: "
        + json.dumps(dict(summary.current_common_last_date_quantiles))
    )
    _print_quantile_group("UNKNOWN quality quantiles", summary.unknown_quantiles)
    _print_quantile_group(
        "Strong UNKNOWN-candidate quality quantiles",
        summary.strong_candidate_quantiles,
    )
    print(
        "UNKNOWN diagnostic groups: "
        + json.dumps(dict(summary.unknown_diagnostic_group_counts))
    )
    for group, examples in summary.unknown_diagnostic_group_examples.items():
        print(f"  Examples {group}: {', '.join(examples)}")
    print(
        "UNKNOWN alias evidence: "
        + json.dumps(dict(summary.unknown_alias_evidence_counts))
    )
    for row in summary.unknown_alias_signal_examples:
        print(
            "  Alias signal "
            f"{row['symbol']} -> {row['current_symbol_lexical_candidates']} "
            f"({row['alias_evidence_status']})"
        )
    print("Liquidity cutoffs: " + json.dumps(dict(summary.liquidity_cutoffs)))
    _print_strategies()
    _print_sensitivity(result.sensitivity)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit equity-universe methodology without selecting a final universe."
        )
    )
    parser.add_argument("--path", help="Override consolidated market Parquet path.")
    parser.add_argument(
        "--company-registry",
        default=str(COMPANY_REGISTRY_PATH),
        help="Existing local company registry CSV.",
    )
    parser.add_argument("--no-registry", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_universe_methodology_audit(
            parquet_path=args.path,
            registry_path=None if args.no_registry else args.company_registry,
        )
        _print_summary(result)
        return 0
    except (
        InstrumentAuditError,
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
    "ACTIVE_SPAN_COVERAGE_LEVELS",
    "CURRENT_UNIVERSE",
    "HISTORICAL_DYNAMIC_UNIVERSE",
    "HISTORY_WINDOWS_DAYS",
    "HYBRID_RESEARCH_UNIVERSE",
    "LIKELY_OTHER_INSTRUMENT",
    "METHODOLOGY_AUDIT_COLUMNS",
    "METHODOLOGY_VERSION",
    "NO_LOCAL_ALIAS_EVIDENCE",
    "SENSITIVITY_COLUMNS",
    "SPARSE_OR_UNCERTAIN",
    "STRATEGY_ASSESSMENTS",
    "STRONG_HISTORICAL_EQUITY_CANDIDATE",
    "UNKNOWN_DIAGNOSTIC_GROUPS",
    "UNRESOLVED",
    "UniverseMethodologyError",
    "UniverseMethodologyResult",
    "UniverseMethodologySummary",
    "UniverseStrategyAssessment",
    "add_active_span_coverage",
    "add_alias_diagnostics",
    "build_methodology_universe",
    "build_sensitivity_table",
    "classify_unknown_diagnostics",
    "detect_unknown_methodology_pattern_signals",
    "main",
    "run_universe_methodology_audit",
    "summarize_universe_methodology",
)
