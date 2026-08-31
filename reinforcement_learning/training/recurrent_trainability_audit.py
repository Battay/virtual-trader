"""TRAIN-value-sealed audit of recurrent trainability gaps.

The audit accounts for every ineligible member produced by the recurrent
orchestrator.  Complete symbol/date identity metadata may be inspected to set a
chronological boundary, but market values are predicate-loaded only through the
derived TRAIN cutoff.  The module never loads VALIDATION or TEST values and
never creates training artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from data_pipeline.src.config import (
    PROCESSED_SPLITS_DIR,
    PROCESSED_SYMBOLS_DIR,
)
from data_pipeline.src.parquet_market_data import (
    load_market_calendar,
    load_market_data,
    load_symbol_market_date_inventory,
    resolve_market_parquet_path,
)
from feature_engineering.indicators import calculate_features
from feature_engineering.preprocessing import filter_ai_quality_rows
from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.storage import safe_path_component
from reinforcement_learning.history_policy import (
    COLD_START_MINIMUM_USABLE_OBSERVATIONS,
    MATURE_MINIMUM_USABLE_OBSERVATIONS,
)

from .recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    DEFAULT_READINESS_EVIDENCE_PATH,
    RecurrentUniverseDiscovery,
    discover_recurrent_training_universe,
)


TRAINABILITY_GAP_AUDIT_VERSION = "recurrent_trainability_gap_audit_v1"
# Eligibility-audit cutoff for safely inspecting candidate TRAIN values. This
# raw-date prefix does not redefine the persisted post-feature rl_partition_v1
# boundaries used by trained models.
TRAIN_PREFIX_POLICY = "symbol_raw_market_dates_first_70_percent_v1"

INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
COLD_START = "COLD_START"
MISSING_RECURRENT_CONTRACT = "MISSING_RECURRENT_CONTRACT"
INCOMPATIBLE_RECURRENT_CONTRACT = "INCOMPATIBLE_RECURRENT_CONTRACT"
LEGACY_PIPELINE_ONLY = "LEGACY_PIPELINE_ONLY"
FEATURE_BUILD_GAP = "FEATURE_BUILD_GAP"
DATA_ALIGNMENT_GAP = "DATA_ALIGNMENT_GAP"
SYMBOL_ALIAS_GAP = "SYMBOL_ALIAS_GAP"
NO_RECENT_TRADING_ACTIVITY = "NO_RECENT_TRADING_ACTIVITY"
OTHER_EXPLICIT_REASON = "OTHER_EXPLICIT_REASON"

DATA_LIMITED = "DATA_LIMITED"
PIPELINE_LIMITED = "PIPELINE_LIMITED"
CONTRACT_LIMITED = "CONTRACT_LIMITED"
IDENTITY_LIMITED = "IDENTITY_LIMITED"

AUDIT_COLUMNS = (
    "symbol",
    "company_name",
    "security_type",
    "current_category",
    "current_reason",
    "precise_category",
    "exact_blocking_reason",
    "train_row_count",
    "train_raw_date_count",
    "train_start",
    "train_end",
    "active_span_coverage",
    "current_recurrent_contract_present",
    "current_feature_artifact_present",
    "canonical_parquet_symbol_present",
    "canonical_parquet_sufficient",
    "limitation_type",
    "safely_recoverable_now",
    "recommended_action",
    "full_observation_count",
    "first_market_date",
    "last_market_date",
    "existing_usable_observations",
    "required_mature_observations",
    "observation_deficit",
    "train_boundary_policy",
)


class RecurrentTrainabilityAuditError(RuntimeError):
    """Raised when the gap audit cannot preserve complete accounting."""


@dataclass(frozen=True)
class RecurrentTrainabilityGapAudit:
    records: pd.DataFrame
    identity_count: int
    trainable_count: int
    non_trainable_count: int
    before_category_counts: Mapping[str, int]
    precise_category_counts: Mapping[str, int]
    limitation_counts: Mapping[str, int]
    universe_hash: str
    parquet_path: Path

    @property
    def final_accounting_count(self) -> int:
        return self.trainable_count + self.non_trainable_count


def _read_readiness(path: Path | None) -> dict[str, Mapping[str, object]]:
    if path is None or not Path(path).is_file():
        return {}
    frame = pd.read_csv(path, dtype={"symbol": "string"})
    if "symbol" not in frame or frame["symbol"].duplicated().any():
        raise RecurrentTrainabilityAuditError(
            "readiness evidence must contain unique symbol rows"
        )
    return {
        str(row["symbol"]).strip(): row
        for row in frame.to_dict(orient="records")
        if str(row.get("symbol", "")).strip()
    }


def _nonnegative_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _training_prefix(symbol_dates: pd.Series) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(
        pd.to_datetime(symbol_dates, errors="coerce").dropna().unique()
    ).sort_values()
    if len(dates) < 3:
        return dates[:0]
    train_count = max(1, int(len(dates) * 0.70))
    if train_count + max(1, int(len(dates) * 0.15)) >= len(dates):
        train_count = len(dates) - 2
    return dates[:train_count]


def _train_feature_evidence(market: pd.DataFrame) -> tuple[int, str, str]:
    if market.empty:
        return 0, "", ""
    raw = market.rename(columns={"market_date": "date"})[
        ["symbol", "date", "open", "high", "low", "close", "volume"]
    ]
    quality = filter_ai_quality_rows(raw)
    if quality.data.empty:
        return 0, "", ""
    featured = calculate_features(quality.data)
    usable = featured.loc[
        ~featured["is_warmup"].astype(bool)
        & ~featured.loc[:, FEATURE_COLUMNS].isna().any(axis=1)
    ]
    dates = pd.to_datetime(usable.get("date"), errors="coerce").dropna()
    return (
        len(usable),
        dates.min().date().isoformat() if not dates.empty else "",
        dates.max().date().isoformat() if not dates.empty else "",
    )


def _classify_gap(
    row: Mapping[str, object],
    *,
    evidence_usable: int,
    train_usable: int,
    last_market_date: str,
    parquet_present: bool,
) -> tuple[str, str, str, bool, str]:
    """Return precise category, reason, limitation, recoverable-now, action."""

    current_category = str(row.get("category", "")).strip()
    current_reason = str(row.get("reason", "")).strip()
    usable = max(evidence_usable, train_usable)

    if not parquet_present:
        return (
            SYMBOL_ALIAS_GAP,
            "No exact symbol identity row exists in the canonical Parquet inventory.",
            IDENTITY_LIMITED,
            False,
            "Investigate authoritative symbol/alias effective dates; do not guess a mapping.",
        )
    if current_reason == "cold_start_not_independent_training":
        return (
            COLD_START,
            f"{evidence_usable} usable observations is Cold Start ("
            f"{COLD_START_MINIMUM_USABLE_OBSERVATIONS}-"
            f"{MATURE_MINIMUM_USABLE_OBSERVATIONS - 1}); independent recurrent "
            f"training requires at least {MATURE_MINIMUM_USABLE_OBSERVATIONS}.",
            DATA_LIMITED,
            False,
            "Keep independent training disabled; use only a separately approved transfer route after more real history.",
        )
    if current_reason in {"insufficient_usable_history", "insufficient_history"}:
        return (
            INSUFFICIENT_HISTORY,
            f"{evidence_usable} usable observations is below the approved Mature "
            f"minimum of {MATURE_MINIMUM_USABLE_OBSERVATIONS}.",
            DATA_LIMITED,
            False,
            "Wait for real observations and rerun the canonical feature pipeline; do not lower the threshold for coverage counts.",
        )
    if current_reason.startswith("insufficient_canonical_train_history_v2:"):
        return (
            INSUFFICIENT_HISTORY,
            "The canonical TRAIN-only v2 feature path is mechanically compatible "
            f"but does not meet the approved Mature minimum: {current_reason}.",
            DATA_LIMITED,
            False,
            "Wait for additional real TRAIN-eligible observations; do not use sealed later values or lower the v2 threshold.",
        )
    if current_reason.startswith("unsupported_security_type:gem_equity"):
        return (
            LEGACY_PIPELINE_ONLY,
            "The identity is an authoritative GEM common equity, but the legacy "
            "symbol feature builder accepts only security_type=ordinary_equity.",
            PIPELINE_LIMITED,
            False,
            "Add a separately reviewed TRAIN-only GEM-compatible artifact path using unchanged features and execution semantics.",
        )
    if current_reason == "not_active_recently_traded":
        suffix = f"; last canonical observation is {last_market_date}" if last_market_date else ""
        return (
            NO_RECENT_TRADING_ACTIVITY,
            "The current official identity lacks recent trading evidence required "
            f"by the independent-training pipeline{suffix}.",
            DATA_LIMITED,
            False,
            "Keep an explicit ineligible job; reconsider only when authoritative current trading observations resume.",
        )
    if current_category == "missing_required_artifacts":
        return (
            FEATURE_BUILD_GAP,
            f"History evidence is Mature ({evidence_usable} usable observations), "
            "but the processed symbol/base RL split and recurrent contract were not "
            "built because the legacy feature path uses a 252-row gate.",
            PIPELINE_LIMITED,
            False,
            "Implement a canonical TRAIN-only artifact builder in a dedicated milestone; do not open sealed later values here.",
        )
    if current_category in {
        "incompatible_contract",
        "incompatible_feature_contract",
    }:
        return (
            INCOMPATIBLE_RECURRENT_CONTRACT,
            str(row.get("compatibility_error", "")).strip()
            or "Existing recurrent metadata is incompatible with the active contract.",
            CONTRACT_LIMITED,
            False,
            "Rebuild only through the canonical versioned artifact pipeline after reviewing the incompatibility.",
        )
    return (
        OTHER_EXPLICIT_REASON,
        current_reason or "An explicit supported trainability rule was not identified.",
        CONTRACT_LIMITED,
        False,
        "Review the recorded compatibility error before changing eligibility.",
    )


def audit_recurrent_trainability_gaps(
    *,
    discovery: RecurrentUniverseDiscovery | None = None,
    parquet_path: str | Path | None = None,
    readiness_evidence_path: Path | None = DEFAULT_READINESS_EVIDENCE_PATH,
    processed_symbols_dir: Path = PROCESSED_SYMBOLS_DIR,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    inventory_loader: Callable[..., pd.DataFrame] = load_symbol_market_date_inventory,
    calendar_loader: Callable[..., pd.DatetimeIndex] = load_market_calendar,
    market_loader: Callable[..., pd.DataFrame] = load_market_data,
) -> RecurrentTrainabilityGapAudit:
    """Audit all non-trainable identities without loading later market values."""

    discovered = discovery or discover_recurrent_training_universe(
        splits_dir=splits_dir,
        readiness_evidence_path=readiness_evidence_path,
    )
    if len(discovered.records) != discovered.identity_count:
        raise RecurrentTrainabilityAuditError(
            "discovery record count does not match frozen identity count"
        )
    gaps = discovered.records.loc[
        discovered.records["category"] != ELIGIBLE_TRAINABLE
    ].copy()
    symbols = tuple(sorted(gaps["symbol"].astype(str)))
    resolved = resolve_market_parquet_path(parquet_path)
    inventory = inventory_loader(resolved, symbols=symbols)
    required_inventory = {"market_date", "symbol"}
    if not required_inventory.issubset(inventory.columns):
        raise RecurrentTrainabilityAuditError(
            "symbol/date inventory is missing required identity columns"
        )
    inventory = inventory.copy()
    inventory["symbol"] = inventory["symbol"].astype("string").str.strip()
    inventory["market_date"] = pd.to_datetime(
        inventory["market_date"], errors="coerce"
    )
    if inventory["market_date"].isna().any():
        raise RecurrentTrainabilityAuditError("inventory contains invalid market dates")
    calendar = pd.DatetimeIndex(calendar_loader(resolved)).sort_values().unique()
    evidence = _read_readiness(readiness_evidence_path)

    records: list[dict[str, object]] = []
    for row in gaps.sort_values("symbol", kind="mergesort").to_dict(orient="records"):
        symbol = str(row["symbol"])
        dates = pd.DatetimeIndex(
            inventory.loc[inventory["symbol"] == symbol, "market_date"].unique()
        ).sort_values()
        train_dates = _training_prefix(pd.Series(dates))
        market = pd.DataFrame()
        if len(train_dates):
            market = market_loader(
                resolved,
                end_date=train_dates[-1].date(),
                symbols=[symbol],
            )
            loaded_dates = pd.to_datetime(market["market_date"], errors="coerce")
            if loaded_dates.isna().any() or (loaded_dates > train_dates[-1]).any():
                raise RecurrentTrainabilityAuditError(
                    f"{symbol}: market loader crossed the declared TRAIN cutoff"
                )
        train_usable, train_start, train_end = _train_feature_evidence(market)
        if len(train_dates):
            active_calendar = calendar[
                (calendar >= dates[0]) & (calendar <= train_dates[-1])
            ]
            active_coverage = (
                len(train_dates) / len(active_calendar) if len(active_calendar) else 0.0
            )
        else:
            active_coverage = 0.0
        ready = evidence.get(symbol, {})
        evidence_usable = _nonnegative_int(ready.get("usable_observations"))
        parquet_present = bool(len(dates))
        precise, exact, limitation, recoverable, action = _classify_gap(
            row,
            evidence_usable=evidence_usable,
            train_usable=train_usable,
            last_market_date=dates[-1].date().isoformat() if len(dates) else "",
            parquet_present=parquet_present,
        )
        usable_for_policy = max(evidence_usable, train_usable)
        component = safe_path_component(symbol)
        recurrent_path = (
            Path(splits_dir)
            / "symbols"
            / component
            / "recurrent"
            / "recurrent_contract.json"
        )
        feature_path = Path(processed_symbols_dir) / f"{component}.csv"
        records.append(
            {
                "symbol": symbol,
                "company_name": str(row.get("company_name", "")),
                "security_type": str(row.get("security_type", "")),
                "current_category": str(row.get("category", "")),
                "current_reason": str(row.get("reason", "")),
                "precise_category": precise,
                "exact_blocking_reason": exact,
                "train_row_count": train_usable,
                "train_raw_date_count": len(train_dates),
                "train_start": train_start,
                "train_end": train_end,
                "active_span_coverage": round(float(active_coverage), 6),
                "current_recurrent_contract_present": recurrent_path.is_file(),
                "current_feature_artifact_present": feature_path.is_file(),
                "canonical_parquet_symbol_present": parquet_present,
                "canonical_parquet_sufficient": (
                    usable_for_policy >= MATURE_MINIMUM_USABLE_OBSERVATIONS
                ),
                "limitation_type": limitation,
                "safely_recoverable_now": recoverable,
                "recommended_action": action,
                "full_observation_count": len(dates),
                "first_market_date": dates[0].date().isoformat() if len(dates) else "",
                "last_market_date": dates[-1].date().isoformat() if len(dates) else "",
                "existing_usable_observations": evidence_usable,
                "required_mature_observations": MATURE_MINIMUM_USABLE_OBSERVATIONS,
                "observation_deficit": max(
                    0, MATURE_MINIMUM_USABLE_OBSERVATIONS - usable_for_policy
                ),
                "train_boundary_policy": TRAIN_PREFIX_POLICY,
            }
        )
    result = pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS).sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
    expected_gaps = discovered.identity_count - discovered.eligible_count
    if len(result) != expected_gaps or result["symbol"].duplicated().any():
        raise RecurrentTrainabilityAuditError(
            "gap audit did not account for every non-trainable identity exactly once"
        )
    if (result["precise_category"] == "UNSUPPORTED").any():
        raise RecurrentTrainabilityAuditError(
            "gap audit retained an imprecise unsupported category"
        )
    return RecurrentTrainabilityGapAudit(
        records=result,
        identity_count=discovered.identity_count,
        trainable_count=discovered.eligible_count,
        non_trainable_count=len(result),
        before_category_counts=dict(discovered.category_counts),
        precise_category_counts=result["precise_category"].value_counts().to_dict(),
        limitation_counts=result["limitation_type"].value_counts().to_dict(),
        universe_hash=discovered.universe_hash,
        parquet_path=resolved,
    )


def write_gap_audit_csv(
    audit: RecurrentTrainabilityGapAudit,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write deterministic diagnostics; refuse accidental replacement."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Audit output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    audit.records.to_csv(target, index=False, lineterminator="\n")
    return target


def _print_summary(audit: RecurrentTrainabilityGapAudit) -> None:
    print(f"Audit version: {TRAINABILITY_GAP_AUDIT_VERSION}")
    print(f"Universe hash: {audit.universe_hash}")
    print(f"Identity count: {audit.identity_count}")
    print(f"Trainable: {audit.trainable_count}")
    print(f"Non-trainable audited: {audit.non_trainable_count}")
    print("Precise categories:")
    for name, count in sorted(audit.precise_category_counts.items()):
        print(f"  {name}: {count}")
    print("Limitation types:")
    for name, count in sorted(audit.limitation_counts.items()):
        print(f"  {name}: {count}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit recurrent trainability gaps")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    audit = audit_recurrent_trainability_gaps()
    _print_summary(audit)
    if args.output_csv:
        print(f"CSV: {write_gap_audit_csv(audit, args.output_csv, overwrite=args.overwrite)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
