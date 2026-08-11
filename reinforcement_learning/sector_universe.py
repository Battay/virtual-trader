"""Research-safe sector universes for future recurrent pretraining.

This module builds metadata and TRAIN-only episode references.  It does not
train a model, load a validation or TEST frame, or alter symbol artifacts.
Current PSX sector labels are kept separate from unverified historical sector
membership so the generated universes cannot be presented as historical
taxonomy reconstructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping, Sequence

import pandas as pd

from data_pipeline.src.config import (
    COMPANY_REGISTRY_PATH,
    CURRENT_LISTINGS_PATH,
    MASTER_CSV_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_MASTER_PATH,
    PROCESSED_SPLITS_DIR,
    PROJECT_ROOT,
)
from data_pipeline.src.official_listings import write_dataframe_atomically
from feature_engineering.readiness import build_training_readiness_report
from feature_engineering.schemas import FEATURE_VERSION
from feature_engineering.storage import atomic_write_json, safe_path_component
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.history_policy import HistoryClass, classify_usable_history
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    RecurrentContractMetadata,
    RecurrentDataContractError,
    load_recurrent_contract_metadata,
)


SECTOR_TAXONOMY_VERSION = "psx_sector_taxonomy_v1"
SECTOR_UNIVERSE_SCHEMA_VERSION = "sector_universe_v1"
SECTOR_EPISODE_INDEX_SCHEMA_VERSION = "sector_train_episode_index_v1"
SECTOR_NORMALIZATION_POLICY_VERSION = "per_symbol_train_scaler_v1"
SECTOR_SAMPLING_POLICY_VERSION = "equal_symbol_episode_sampling_v1"
STANDARD_SECTOR_PRETRAINING = "standard_sector_pretraining"
LEAVE_ONE_SYMBOL_OUT = "leave_one_symbol_out_transfer"
SECTOR_UNIVERSES_DIR = PROCESSED_DATA_DIR / "sector_universes"

READY_FOR_SECTOR_RESEARCH = "READY_FOR_SECTOR_RESEARCH"
LIMITED = "LIMITED"
INSUFFICIENT = "INSUFFICIENT"
UNVERIFIED_HISTORY = "UNVERIFIED_HISTORY"

# Five constituents leaves at least four independent peers in a future
# leave-one-symbol-out experiment.  A one-year median TRAIN history and 5,120
# total rows prevent a single long symbol from creating a misleading READY
# label.  The concentration limit is evaluated on proportional row sampling;
# equal-symbol sampling remains the recommended initial baseline.
MINIMUM_READY_MATURE_SYMBOLS = 5
MINIMUM_READY_TOTAL_TRAIN_ROWS = 5_120
MINIMUM_READY_MEDIAN_TRAIN_ROWS = 252
MAXIMUM_READY_PROPORTIONAL_CONCENTRATION = 0.40
MINIMUM_LIMITED_MATURE_SYMBOLS = 3
MINIMUM_LIMITED_TOTAL_TRAIN_ROWS = 1_000


class SectorUniverseError(ValueError):
    """Raised when sector evidence or a pooled universe is unsafe."""


@dataclass(frozen=True)
class SectorTaxonomyMatch:
    raw_value: str
    sector_id: str
    display_name: str
    recognized: bool
    provenance: str


@dataclass(frozen=True)
class SectorUniverseBuildResult:
    output_directory: Path
    taxonomy_path: Path
    current_verified_path: Path
    historical_audit_path: Path
    summary_path: Path
    manifest_paths: tuple[Path, ...]
    episode_index_paths: tuple[Path, ...]
    taxonomy_version: str
    cohort_cutoff: str
    listing_snapshot_date: str
    verified_sector_symbols: int
    historical_only_total: int
    historical_non_equity_contract_like: int
    historical_ordinary_equity_candidates: int
    historical_unknown: int
    sector_count: int
    recurrent_compatible_symbols: int
    source_registry_sha256: str


_CANONICAL_SECTORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("apparel", "Apparel", ("APPAREL",)),
    ("automobile_assembler", "Automobile Assembler", ("AUTOMOBILE ASSEMBLER",)),
    (
        "automobile_parts_accessories",
        "Automobile Parts & Accessories",
        ("AUTOMOBILE PARTS & ACCESSORIES", "AUTOMOBILE PARTS AND ACCESSORIES"),
    ),
    (
        "cable_electrical_goods",
        "Cable & Electrical Goods",
        ("CABLE & ELECTRICAL GOODS", "CABLE AND ELECTRICAL GOODS"),
    ),
    ("cement", "Cement", ("CEMENT",)),
    ("chemical", "Chemical", ("CHEMICAL",)),
    (
        "closed_end_mutual_funds",
        "Closed-End Mutual Funds",
        ("CLOSE - END MUTUAL FUND", "CLOSE END MUTUAL FUND", "CLOSED END MUTUAL FUNDS"),
    ),
    ("commercial_banks", "Commercial Banks", ("COMMERCIAL BANKS",)),
    ("engineering", "Engineering", ("ENGINEERING",)),
    ("exchange_traded_funds", "Exchange Traded Funds", ("EXCHANGE TRADED FUNDS",)),
    ("fertilizer", "Fertilizer", ("FERTILIZER",)),
    (
        "food_personal_care_products",
        "Food & Personal Care Products",
        ("FOOD & PERSONAL CARE PRODUCTS", "FOOD AND PERSONAL CARE PRODUCTS"),
    ),
    ("glass_ceramics", "Glass & Ceramics", ("GLASS & CERAMICS", "GLASS AND CERAMICS")),
    ("insurance", "Insurance", ("INSURANCE",)),
    (
        "investment_banks_companies_securities",
        "Investment Banks / Investment Companies / Securities Companies",
        (
            "INV. BANKS / INV. COS. / SECURITIES COS.",
            "INV BANKS / INV COS / SECURITIES COS",
            "INVESTMENT BANKS / INVESTMENT COMPANIES / SECURITIES COMPANIES",
        ),
    ),
    ("jute", "Jute", ("JUTE",)),
    ("leasing_companies", "Leasing Companies", ("LEASING COMPANIES",)),
    (
        "leather_tanneries",
        "Leather & Tanneries",
        ("LEATHER & TANNERIES", "LEATHER AND TANNERIES"),
    ),
    ("miscellaneous", "Miscellaneous", ("MISCELLANEOUS",)),
    ("modarabas", "Modarabas", ("MODARABAS",)),
    (
        "oil_gas_exploration_companies",
        "Oil & Gas Exploration Companies",
        ("OIL & GAS EXPLORATION COMPANIES", "OIL AND GAS EXPLORATION COMPANIES"),
    ),
    (
        "oil_gas_marketing_companies",
        "Oil & Gas Marketing Companies",
        ("OIL & GAS MARKETING COMPANIES", "OIL AND GAS MARKETING COMPANIES"),
    ),
    (
        "paper_board_packaging",
        "Paper, Board & Packaging",
        ("PAPER, BOARD & PACKAGING", "PAPER BOARD AND PACKAGING"),
    ),
    ("pharmaceuticals", "Pharmaceuticals", ("PHARMACEUTICALS",)),
    (
        "power_generation_distribution",
        "Power Generation & Distribution",
        ("POWER GENERATION & DISTRIBUTION", "POWER GENERATION AND DISTRIBUTION"),
    ),
    ("property", "Property", ("PROPERTY",)),
    (
        "real_estate_investment_trust",
        "Real Estate Investment Trust",
        ("REAL ESTATE INVESTMENT TRUST", "REAL ESTATE INVESTMENT TRUSTS", "REIT"),
    ),
    ("refinery", "Refinery", ("REFINERY",)),
    (
        "sugar_allied_industries",
        "Sugar & Allied Industries",
        ("SUGAR & ALLIED INDUSTRIES", "SUGAR AND ALLIED INDUSTRIES"),
    ),
    (
        "synthetic_rayon",
        "Synthetic & Rayon",
        ("SYNTHETIC & RAYON", "SYNTHETIC AND RAYON"),
    ),
    (
        "technology_communication",
        "Technology & Communication",
        ("TECHNOLOGY & COMMUNICATION", "TECHNOLOGY AND COMMUNICATION"),
    ),
    ("textile_composite", "Textile Composite", ("TEXTILE COMPOSITE",)),
    ("textile_spinning", "Textile Spinning", ("TEXTILE SPINNING",)),
    ("textile_weaving", "Textile Weaving", ("TEXTILE WEAVING",)),
    ("tobacco", "Tobacco", ("TOBACCO",)),
    ("transport", "Transport", ("TRANSPORT",)),
    (
        "vanaspati_allied_industries",
        "Vanaspati & Allied Industries",
        ("VANASPATI & ALLIED INDUSTRIES", "VANASPATI AND ALLIED INDUSTRIES"),
    ),
    ("woollen", "Woollen", ("WOOLLEN",)),
)


def _normalized_sector_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SECTOR_ALIASES: dict[str, tuple[str, str]] = {}
for _sector_id, _display_name, _aliases in _CANONICAL_SECTORS:
    for _alias in _aliases:
        _key = _normalized_sector_key(_alias)
        if _key in _SECTOR_ALIASES and _SECTOR_ALIASES[_key][0] != _sector_id:
            raise RuntimeError(f"ambiguous sector taxonomy alias: {_alias}")
        _SECTOR_ALIASES[_key] = (_sector_id, _display_name)


def normalize_sector(value: object) -> SectorTaxonomyMatch:
    """Normalize only explicit aliases; unknown values are never guessed."""

    raw = "" if value is None or pd.isna(value) else str(value).strip()
    match = _SECTOR_ALIASES.get(_normalized_sector_key(raw))
    if match is None:
        return SectorTaxonomyMatch(
            raw_value=raw,
            sector_id="unknown",
            display_name="Unknown / Unverified",
            recognized=False,
            provenance="no_explicit_psx_sector_alias_match",
        )
    return SectorTaxonomyMatch(
        raw_value=raw,
        sector_id=match[0],
        display_name=match[1],
        recognized=True,
        provenance=f"explicit_alias:{SECTOR_TAXONOMY_VERSION}",
    )


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_universe_hash(identity: Mapping[str, object]) -> str:
    """Hash canonical identity content, excluding timestamps and machine paths."""

    return _canonical_hash(dict(identity))


def taxonomy_payload(*, generated_at: str) -> dict[str, object]:
    entries = [
        {
            "sector_id": sector_id,
            "display_name": display,
            "raw_aliases": list(aliases),
            "normalization_provenance": "explicit_local_psx_label_mapping",
        }
        for sector_id, display, aliases in _CANONICAL_SECTORS
    ]
    identity = {
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "unknown_policy": "fail_to_unknown_without_guessing",
        "entries": entries,
    }
    return {
        **identity,
        "taxonomy_hash": _canonical_hash(identity),
        "generated_at": generated_at,
    }


_MONTH_CONTRACT = re.compile(
    r"-(?:C)?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]?$"
)
_ODD_LOT = re.compile(r"-ODL$")
_RIGHT_OR_ENTITLEMENT = re.compile(r"(?:R|R[0-9]+)$")
_DEBT_OR_OTHER = re.compile(r"(?:TFC|SUKUK|PREF|PFC)[A-Z0-9-]*$")


def classify_historical_instruments(registry: pd.DataFrame) -> pd.DataFrame:
    """Classify historical records conservatively without sector inference."""

    required = {
        "symbol",
        "lifecycle_status",
        "security_type",
        "company_name",
        "sector",
        "first_seen_date",
        "last_seen_date",
        "trading_days",
    }
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise SectorUniverseError(
            "registry is missing historical audit columns: " + ", ".join(missing)
        )
    historical = registry.loc[
        registry["lifecycle_status"].astype("string").eq("historical_only")
    ].copy(deep=True)
    records: list[dict[str, object]] = []
    for row in historical.sort_values("symbol", kind="stable").itertuples(index=False):
        symbol = str(row.symbol).strip()
        if _MONTH_CONTRACT.search(symbol):
            instrument_class = "historical_month_coded_contract"
            group = "non_equity_or_contract_like"
            evidence = "month-coded contract symbol pattern"
        elif _ODD_LOT.search(symbol):
            instrument_class = "historical_odd_lot_segment_security"
            group = "non_equity_or_contract_like"
            evidence = "ODL segment suffix"
        elif _RIGHT_OR_ENTITLEMENT.search(symbol):
            instrument_class = "historical_right_or_security_entitlement"
            group = "non_equity_or_contract_like"
            evidence = "rights/entitlement suffix pattern"
        elif _DEBT_OR_OTHER.search(symbol):
            instrument_class = "historical_debt_or_other_instrument"
            group = "non_equity_or_contract_like"
            evidence = "debt/preference instrument marker"
        else:
            instrument_class = "possible_equity_base_symbol_unverified"
            group = "unknown_requires_investigation"
            evidence = "bare historical symbol; no local company/type evidence"
        records.append(
            {
                "symbol": symbol,
                "saved_security_type": str(row.security_type),
                "historical_instrument_class": instrument_class,
                "historical_audit_group": group,
                "ordinary_equity_verified": False,
                "sector_membership_meaningful": group != "non_equity_or_contract_like",
                "company_name_available": bool(
                    "" if pd.isna(row.company_name) else str(row.company_name).strip()
                ),
                "sector_available": bool(
                    "" if pd.isna(row.sector) else str(row.sector).strip()
                ),
                "evidence_basis": evidence,
                "classification_confidence": (
                    "pattern_supported_exclusion"
                    if group == "non_equity_or_contract_like"
                    else "unknown_requires_authoritative_verification"
                ),
                "first_seen_date": row.first_seen_date,
                "last_seen_date": row.last_seen_date,
                "trading_days": int(row.trading_days),
                "recommended_action": (
                    "exclude_from_equity_sector_reconstruction"
                    if group == "non_equity_or_contract_like"
                    else "authoritative_manual_instrument_and_sector_investigation"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def _clean_text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _iso_date(value: object, *, label: str) -> str:
    if isinstance(value, date):
        text = value.isoformat()
    else:
        text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise SectorUniverseError(f"{label} must use YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise SectorUniverseError(f"{label} must use YYYY-MM-DD")
    return text


def _project_relative(path: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SectorUniverseError(
            f"artifact path must be inside the project: {path}"
        ) from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise SectorUniverseError("artifact path is not portable")
    return relative.as_posix()


def _portable_error_message(error: object, *, project_root: Path) -> str:
    """Remove developer-machine prefixes while preserving actionable context."""

    message = str(error)
    root = str(Path(project_root).resolve())
    return message.replace(root + "/", "")


def _current_security_category(security_type: object, sector_id: str) -> str:
    value = str(security_type).strip().lower()
    if value == "other" and sector_id == "real_estate_investment_trust":
        return "reit"
    return {
        "ordinary_equity": "ordinary_equity",
        "gem_equity": "gem_equity",
        "preference_share": "preference_share",
        "etf": "etf",
        "right": "rights_security_entitlement",
    }.get(value, "unsupported_or_unknown")


def _contract_sources(
    symbol: str,
    metadata: RecurrentContractMetadata,
    *,
    project_root: Path,
) -> dict[str, object]:
    symbol_directory = metadata.contract_path.parent.parent
    train_path = symbol_directory / "train_rl.csv"
    scaler_path = symbol_directory / "rl_observation_scaler.joblib"
    scaler_metadata_path = symbol_directory / "rl_observation_scaler.json"
    required = (train_path, scaler_path, scaler_metadata_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SectorUniverseError(
            f"{symbol} is missing canonical TRAIN/scaler artifacts: "
            + ", ".join(missing)
        )
    return {
        "recurrent_contract_path": _project_relative(
            metadata.contract_path, project_root=project_root
        ),
        "recurrent_contract_sha256": sha256_file(metadata.contract_path),
        "train_rl_path": _project_relative(train_path, project_root=project_root),
        "train_rl_sha256": sha256_file(train_path),
        "scaler_path": _project_relative(scaler_path, project_root=project_root),
        "scaler_sha256": sha256_file(scaler_path),
        "scaler_metadata_path": _project_relative(
            scaler_metadata_path, project_root=project_root
        ),
        "scaler_metadata_sha256": sha256_file(scaler_metadata_path),
    }


def build_current_verified_universe(
    registry: pd.DataFrame,
    readiness: pd.DataFrame,
    *,
    cohort_cutoff: str,
    listing_snapshot_date: str,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    metadata_loader: Callable[..., RecurrentContractMetadata] = (
        load_recurrent_contract_metadata
    ),
) -> pd.DataFrame:
    """Build current official-sector evidence and TRAIN eligibility metadata."""

    cutoff_text = _iso_date(cohort_cutoff, label="cohort cutoff")
    snapshot_text = _iso_date(listing_snapshot_date, label="listing snapshot date")
    required_registry = {
        "symbol",
        "company_name",
        "security_type",
        "sector",
        "officially_listed",
        "official_status",
        "activity_status",
        "lifecycle_status",
        "first_seen_date",
        "last_seen_date",
        "source",
    }
    missing = sorted(required_registry.difference(registry.columns))
    if missing:
        raise SectorUniverseError(
            "registry is missing current-universe columns: " + ", ".join(missing)
        )
    required_readiness = {
        "symbol",
        "usable_feature_rows",
        "first_usable_date",
        "last_usable_date",
        "readiness_status",
    }
    missing_readiness = sorted(required_readiness.difference(readiness.columns))
    if missing_readiness:
        raise SectorUniverseError(
            "readiness is missing columns: " + ", ".join(missing_readiness)
        )
    current = registry.loc[registry["officially_listed"].map(_as_bool)].copy(deep=True)
    current["symbol"] = current["symbol"].astype("string").str.strip()
    if current["symbol"].eq("").any() or current["symbol"].duplicated().any():
        raise SectorUniverseError("current registry symbols must be unique and non-empty")
    readiness_values = readiness.loc[:, list(required_readiness)].copy(deep=True)
    readiness_values["symbol"] = readiness_values["symbol"].astype("string").str.strip()
    current = current.merge(
        readiness_values,
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    records: list[dict[str, object]] = []
    for row in current.sort_values("symbol", kind="stable").to_dict(orient="records"):
        symbol = str(row["symbol"])
        taxonomy = normalize_sector(row.get("sector"))
        security_type = str(row.get("security_type", "unknown")).strip().lower()
        active = (
            _as_bool(row.get("officially_listed"))
            and str(row.get("activity_status", "")) == "recently_traded"
        )
        raw_usable = row.get("usable_feature_rows")
        usable = 0 if raw_usable is None or pd.isna(raw_usable) else int(raw_usable)
        history_class = "NOT_APPLICABLE"
        if active and security_type == "ordinary_equity":
            history_class = classify_usable_history(usable).history_class.value
        first_observed = _clean_text(row.get("first_seen_date"))
        first_allowed = bool(
            first_observed and first_observed <= cutoff_text
        )
        source = str(row.get("source", ""))
        sector_verified_current = (
            taxonomy.recognized and source.startswith("https://dps.psx.com.pk/")
        )
        sector_verified_at_cutoff = (
            sector_verified_current and snapshot_text == cutoff_text
        )
        metadata: RecurrentContractMetadata | None = None
        compatibility_error = ""
        sources: dict[str, object] = {}
        if active and security_type == "ordinary_equity" and history_class == "MATURE":
            try:
                metadata = metadata_loader(symbol, splits_dir=Path(splits_dir))
                if normalize_sector(metadata.sector).sector_id != taxonomy.sector_id:
                    raise SectorUniverseError(
                        "recurrent/current canonical sectors differ"
                    )
                if metadata.recurrent_contract_version != RL_RECURRENT_PARTITION_SCHEMA_VERSION:
                    raise SectorUniverseError("recurrent contract version is stale")
                if metadata.feature_version != FEATURE_VERSION:
                    raise SectorUniverseError("feature version is stale")
                if metadata.environment_version != ENVIRONMENT_VERSION:
                    raise SectorUniverseError("environment version is stale")
                if metadata.observation_features != DEFAULT_OBSERVATION_FEATURES:
                    raise SectorUniverseError("observation feature order differs")
                if metadata.observation_shape != (
                    len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
                ):
                    raise SectorUniverseError("observation shape differs")
                if metadata.normalization_scope != "symbol":
                    raise SectorUniverseError("normalization scope differs")
                if metadata.scaler_fit_partition != "train":
                    raise SectorUniverseError("scaler was not fitted on TRAIN")
                sources = _contract_sources(
                    symbol, metadata, project_root=Path(project_root)
                )
            except (RecurrentDataContractError, SectorUniverseError, OSError, ValueError) as exc:
                compatibility_error = _portable_error_message(
                    exc, project_root=Path(project_root)
                )
                metadata = None
        recurrent_compatible = metadata is not None
        train_rows = metadata.train.rows if metadata else 0
        train_start = metadata.train.start if metadata else ""
        last_train = metadata.train.end if metadata else ""
        train_within_cutoff = bool(metadata and last_train <= cutoff_text)
        rows_at_cutoff = train_rows if train_within_cutoff else 0

        exclusion_reason = ""
        if security_type != "ordinary_equity":
            exclusion_reason = f"unsupported_security_type:{security_type}"
        elif not active:
            exclusion_reason = "not_active_recently_traded"
        elif not taxonomy.recognized or not sector_verified_current:
            exclusion_reason = "current_sector_not_verified"
        elif not first_allowed:
            exclusion_reason = "first_observation_after_cohort_cutoff"
        elif history_class == "COLD_START":
            exclusion_reason = "cold_start_not_independent_pretraining"
        elif history_class == "INSUFFICIENT":
            exclusion_reason = "insufficient_history"
        elif not recurrent_compatible:
            exclusion_reason = "missing_or_incompatible_recurrent_contract"
        elif not train_within_cutoff:
            exclusion_reason = "train_partition_extends_beyond_cohort_cutoff"
        eligible = exclusion_reason == ""
        records.append(
            {
                "symbol": symbol,
                "company_name": str(row.get("company_name", "")),
                "raw_sector": taxonomy.raw_value,
                "sector_id": taxonomy.sector_id,
                "sector_name": taxonomy.display_name,
                "sector_normalization_provenance": taxonomy.provenance,
                "security_type": security_type,
                "research_security_category": _current_security_category(
                    security_type, taxonomy.sector_id
                ),
                "official_status": str(row.get("official_status", "")),
                "lifecycle_status": str(row.get("lifecycle_status", "")),
                "currently_listed": True,
                "active_recently_traded": active,
                "sector_verified_current": sector_verified_current,
                "sector_verified_at_cutoff": sector_verified_at_cutoff,
                "historical_sector_membership_verified": False,
                "historical_sector_membership_unknown": True,
                "current_sector_only": True,
                "sector_changed_over_time": "unknown_no_local_evidence",
                "sector_source": source,
                "listing_snapshot_date": snapshot_text,
                "cohort_cutoff": cutoff_text,
                "first_observed_date": first_observed,
                "latest_observed_date": _clean_text(row.get("last_seen_date")),
                "usable_observations": usable,
                "history_class": history_class,
                "first_usable_date": _clean_text(row.get("first_usable_date")),
                "latest_usable_date": _clean_text(row.get("last_usable_date")),
                "source_readiness_status": _clean_text(row.get("readiness_status")),
                "recurrent_compatible": recurrent_compatible,
                "feature_version": metadata.feature_version if metadata else "",
                "recurrent_contract_version": (
                    metadata.recurrent_contract_version if metadata else ""
                ),
                "environment_version": metadata.environment_version if metadata else "",
                "observation_features": (
                    json.dumps(list(metadata.observation_features), separators=(",", ":"))
                    if metadata
                    else ""
                ),
                "observation_shape": (
                    json.dumps(list(metadata.observation_shape), separators=(",", ":"))
                    if metadata
                    else ""
                ),
                "execution_semantics": (
                    "single_symbol_env_v1_real_ohlcv_next_open_execution"
                    if metadata
                    else ""
                ),
                "normalization_scope": metadata.normalization_scope if metadata else "",
                "scaler_fit_partition": metadata.scaler_fit_partition if metadata else "",
                "train_rows": train_rows,
                "train_start": train_start,
                "last_train_date": last_train,
                "rows_available_at_cutoff": rows_at_cutoff,
                "eligible_at_cutoff": eligible,
                "exclusion_reason": exclusion_reason,
                "compatibility_error": compatibility_error,
                **sources,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "symbol", kind="stable"
    ).reset_index(drop=True)


def _tuple_from_json(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise SectorUniverseError("invalid JSON-encoded compatibility field") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise SectorUniverseError("compatibility field must be a string list")
    return tuple(parsed)


def _homogeneous_constituents(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "feature_version": FEATURE_VERSION,
            "recurrent_contract_version": RL_RECURRENT_PARTITION_SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "observation_features": list(DEFAULT_OBSERVATION_FEATURES),
            "observation_shape": [
                len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES)
            ],
            "execution_semantics": "single_symbol_env_v1_real_ohlcv_next_open_execution",
            "source_normalization_scope": "symbol",
            "scaler_fit_partition": "train",
        }
    fields = (
        "feature_version",
        "recurrent_contract_version",
        "environment_version",
        "observation_features",
        "observation_shape",
        "execution_semantics",
        "normalization_scope",
        "scaler_fit_partition",
    )
    values: dict[str, object] = {}
    for field in fields:
        unique = tuple(sorted(set(str(value) for value in rows[field])))
        if len(unique) != 1:
            raise SectorUniverseError(
                f"sector constituent contracts are heterogeneous for {field}"
            )
        values[field] = unique[0]
    if values["feature_version"] != FEATURE_VERSION:
        raise SectorUniverseError("sector feature version is incompatible")
    if values["recurrent_contract_version"] != RL_RECURRENT_PARTITION_SCHEMA_VERSION:
        raise SectorUniverseError("sector recurrent contract version is incompatible")
    if values["environment_version"] != ENVIRONMENT_VERSION:
        raise SectorUniverseError("sector environment version is incompatible")
    if _tuple_from_json(values["observation_features"]) != DEFAULT_OBSERVATION_FEATURES:
        raise SectorUniverseError("sector observation ordering is incompatible")
    expected_shape = (
        len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
    )
    try:
        shape = tuple(int(item) for item in json.loads(str(values["observation_shape"])))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SectorUniverseError("sector observation shape is invalid") from exc
    if shape != expected_shape:
        raise SectorUniverseError("sector observation shape is incompatible")
    if values["normalization_scope"] != "symbol" or values["scaler_fit_partition"] != "train":
        raise SectorUniverseError("sector normalization provenance is incompatible")
    return {
        "feature_version": values["feature_version"],
        "recurrent_contract_version": values["recurrent_contract_version"],
        "environment_version": values["environment_version"],
        "observation_features": list(DEFAULT_OBSERVATION_FEATURES),
        "observation_shape": list(expected_shape),
        "execution_semantics": values["execution_semantics"],
        "source_normalization_scope": values["normalization_scope"],
        "scaler_fit_partition": values["scaler_fit_partition"],
    }


def build_sector_manifest(
    sector_rows: pd.DataFrame,
    *,
    sector_id: str,
    sector_name: str,
    cohort_cutoff: str,
    listing_snapshot_date: str,
    source_registry_sha256: str,
    source_registry_path: str = "data/master/company_registry.csv",
    source_listing_path: str = "data/metadata/listings/current_listings.csv",
    source_listing_sha256: str = "",
    generated_at: str,
    git_commit: str | None,
    mode: str = STANDARD_SECTOR_PRETRAINING,
    target_symbol: str | None = None,
) -> dict[str, object]:
    """Build a deterministic standard or target-excluded sector manifest."""

    if mode not in {STANDARD_SECTOR_PRETRAINING, LEAVE_ONE_SYMBOL_OUT}:
        raise SectorUniverseError("unsupported sector pretraining mode")
    cutoff_text = _iso_date(cohort_cutoff, label="cohort cutoff")
    snapshot_text = _iso_date(listing_snapshot_date, label="listing snapshot date")
    target = str(target_symbol).strip() if target_symbol is not None else ""
    if mode == LEAVE_ONE_SYMBOL_OUT and not target:
        raise SectorUniverseError("leave-one-symbol-out mode requires target_symbol")
    if mode == STANDARD_SECTOR_PRETRAINING and target:
        raise SectorUniverseError("standard sector mode cannot declare a target")
    rows = sector_rows.loc[sector_rows["sector_id"].eq(sector_id)].copy(deep=True)
    if rows.empty:
        raise SectorUniverseError(f"sector has no current verified rows: {sector_id}")
    if rows["symbol"].duplicated().any():
        raise SectorUniverseError("sector rows contain duplicate symbols")
    if mode == LEAVE_ONE_SYMBOL_OUT and target not in set(rows["symbol"].astype(str)):
        raise SectorUniverseError("target symbol is not a verified member of the sector")

    eligible = rows.loc[rows["eligible_at_cutoff"].map(_as_bool)].copy()
    if mode == LEAVE_ONE_SYMBOL_OUT:
        eligible = eligible.loc[eligible["symbol"].astype(str) != target]
    eligible = eligible.sort_values("symbol", kind="stable").reset_index(drop=True)
    compatibility = _homogeneous_constituents(eligible)
    constituent_symbols = tuple(eligible["symbol"].astype(str))
    if len(set(constituent_symbols)) != len(constituent_symbols):
        raise SectorUniverseError("pretraining constituents are duplicated")
    if target and target in constituent_symbols:
        raise SectorUniverseError("target leaked into its own pretraining universe")

    total_train_rows = int(eligible["rows_available_at_cutoff"].sum())
    constituent_records: list[dict[str, object]] = []
    for row in eligible.to_dict(orient="records"):
        constituent_records.append(
            {
                "symbol": str(row["symbol"]),
                "company_name": str(row["company_name"]),
                "security_type": str(row["security_type"]),
                "history_class": str(row["history_class"]),
                "first_observed_date": str(row["first_observed_date"]),
                "first_usable_date": str(row["first_usable_date"]),
                "train_start": str(row["train_start"]),
                "last_train_date": str(row["last_train_date"]),
                "rows_available_at_cutoff": int(row["rows_available_at_cutoff"]),
                "eligible_at_cutoff": True,
                "sector_verified_current": bool(row["sector_verified_current"]),
                "sector_verified_at_cutoff": bool(row["sector_verified_at_cutoff"]),
                "historical_sector_membership_verified": False,
                "historical_sector_membership_unknown": True,
                "recurrent_contract_path": str(row["recurrent_contract_path"]),
                "recurrent_contract_sha256": str(row["recurrent_contract_sha256"]),
                "train_rl_path": str(row["train_rl_path"]),
                "train_rl_sha256": str(row["train_rl_sha256"]),
                "scaler_path": str(row["scaler_path"]),
                "scaler_sha256": str(row["scaler_sha256"]),
                "scaler_metadata_path": str(row["scaler_metadata_path"]),
                "scaler_metadata_sha256": str(row["scaler_metadata_sha256"]),
            }
        )

    excluded: list[dict[str, object]] = []
    for row in rows.sort_values("symbol", kind="stable").to_dict(orient="records"):
        symbol = str(row["symbol"])
        reason = str(row.get("exclusion_reason", ""))
        if mode == LEAVE_ONE_SYMBOL_OUT and symbol == target:
            reason = "target_excluded_leave_one_symbol_out"
        elif symbol in constituent_symbols:
            continue
        excluded.append(
            {
                "symbol": symbol,
                "security_type": str(row.get("security_type", "unknown")),
                "history_class": str(row.get("history_class", "NOT_APPLICABLE")),
                "reason": reason or "not_approved_for_pretraining",
            }
        )

    normalization_contributors = list(constituent_symbols)
    if target and target in normalization_contributors:
        raise SectorUniverseError("target leaked into normalization contributors")
    identity: dict[str, object] = {
        "artifact_schema_version": SECTOR_UNIVERSE_SCHEMA_VERSION,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "sector_id": sector_id,
        "sector_name": sector_name,
        "cohort_cutoff": cutoff_text,
        "mode": mode,
        "target_symbol": target or None,
        "target_excluded_from_pretraining": mode == LEAVE_ONE_SYMBOL_OUT,
        "pretraining_constituent_symbols": list(constituent_symbols),
        "exclusions": excluded,
        "feature_version": compatibility["feature_version"],
        "recurrent_contract_version": compatibility["recurrent_contract_version"],
        "environment_version": compatibility["environment_version"],
        "observation_features": compatibility["observation_features"],
        "observation_shape": compatibility["observation_shape"],
        "execution_semantics": compatibility["execution_semantics"],
        "normalization_policy": SECTOR_NORMALIZATION_POLICY_VERSION,
        "normalization_contributor_symbols": normalization_contributors,
        "sampling_strategy": SECTOR_SAMPLING_POLICY_VERSION,
        "source_registry_sha256": source_registry_sha256,
    }
    universe_hash = deterministic_universe_hash(identity)
    return {
        "artifact_schema_version": SECTOR_UNIVERSE_SCHEMA_VERSION,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "universe_hash": universe_hash,
        "pretraining_universe_hash": universe_hash,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "sector": {
            "sector_id": sector_id,
            "sector_name": sector_name,
            "raw_sector_values": sorted(set(rows["raw_sector"].astype(str))),
        },
        "cohort": {
            "cohort_cutoff": cutoff_text,
            "listing_snapshot_date": snapshot_text,
            "membership_basis": "current_official_sector_snapshot_only",
            "validation_performance_used_for_membership": False,
            "test_data_used_for_membership": False,
        },
        "experiment_mode": {
            "mode": mode,
            "target_symbol": target or None,
            "target_excluded_from_pretraining": mode == LEAVE_ONE_SYMBOL_OUT,
            "pretraining_constituent_symbols": list(constituent_symbols),
        },
        "constituent_count": len(constituent_records),
        "mature_count": len(constituent_records),
        "cold_start_count": int(rows["history_class"].eq("COLD_START").sum()),
        "total_train_observations": total_train_rows,
        "constituents": constituent_records,
        "excluded_symbols": excluded,
        "compatibility": compatibility,
        "normalization": {
            "policy_version": SECTOR_NORMALIZATION_POLICY_VERSION,
            "implemented_scope": "per_symbol_train_fitted_scalers",
            "fit_partition": "train",
            "approved_cohort_only": True,
            "normalization_contributor_symbols": normalization_contributors,
            "target_contributes_to_pretraining_normalization": False if target else None,
            "sector_wide_scaler_status": "not_fitted_in_6d",
        },
        "sampling": {
            "policy_version": SECTOR_SAMPLING_POLICY_VERSION,
            "recommended_initial_strategy": "equal_symbol_episode_sampling",
            "advanced_prioritized_sampling": False,
        },
        "reset_semantics": {
            "each_symbol_is_separate_episode": True,
            "episode_start_true_at_symbol_boundary": True,
            "environment_reset": True,
            "cash_reset": True,
            "holdings_reset": True,
            "realized_profit_loss_reset": True,
            "portfolio_peak_and_drawdown_reset": True,
            "recurrent_hidden_state_reset": True,
            "state_carry_between_symbols": False,
        },
        "historical_membership": {
            "sector_verified_current": True,
            "sector_verified_at_cutoff": snapshot_text == cutoff_text,
            "historical_sector_membership_verified": False,
            "historical_sector_membership_unknown": True,
            "sector_changed_over_time": "unknown_no_local_evidence",
            "claim_limit": "current-sector grouping; not historical membership proof",
        },
        "source_provenance": {
            "company_registry_path": source_registry_path,
            "company_registry_sha256": source_registry_sha256,
            "official_listing_snapshot_path": source_listing_path,
            "official_listing_snapshot_sha256": source_listing_sha256,
            "paths_are_project_relative": True,
        },
        "data_access": {
            "training_partition": "train_only",
            "validation_frames_referenced": False,
            "test_frame_access": "prohibited",
            "test_values_in_manifest": False,
        },
        "deterministic_identity": identity,
    }


def build_train_episode_index(manifest: Mapping[str, object]) -> dict[str, object]:
    """Create isolated TRAIN episode references without concatenating frames."""

    if manifest.get("artifact_schema_version") != SECTOR_UNIVERSE_SCHEMA_VERSION:
        raise SectorUniverseError("sector manifest version is incompatible")
    constituents = manifest.get("constituents")
    if not isinstance(constituents, Sequence) or isinstance(constituents, (str, bytes)):
        raise SectorUniverseError("sector manifest constituents must be a sequence")
    symbols = [str(item.get("symbol", "")) for item in constituents]
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise SectorUniverseError("episode symbols must be unique and non-empty")
    total_rows = sum(int(item["rows_available_at_cutoff"]) for item in constituents)
    count = len(symbols)
    episodes: list[dict[str, object]] = []
    for sequence, item in enumerate(constituents, start=1):
        rows = int(item["rows_available_at_cutoff"])
        if rows < 2:
            raise SectorUniverseError("every TRAIN episode needs at least two rows")
        path = str(item["train_rl_path"])
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise SectorUniverseError("episode source paths must be portable")
        episodes.append(
            {
                "episode_sequence": sequence,
                "episode_id": f"{manifest['sector']['sector_id']}:{item['symbol']}:train",
                "symbol": item["symbol"],
                "partition": "train",
                "source_train_rl_path": path,
                "source_train_rl_sha256": item["train_rl_sha256"],
                "rows": rows,
                "start": item["train_start"],
                "end": item["last_train_date"],
                "episode_start": True,
                "reset_before_episode": {
                    "environment": True,
                    "cash": True,
                    "holdings": True,
                    "realized_profit_loss": True,
                    "portfolio_peak_and_drawdown": True,
                    "recurrent_hidden_state": True,
                },
                "proportional_row_sampling_share": (
                    rows / total_rows if total_rows else 0.0
                ),
                "equal_symbol_episode_sampling_share": 1.0 / count if count else 0.0,
            }
        )
    return {
        "artifact_schema_version": SECTOR_EPISODE_INDEX_SCHEMA_VERSION,
        "sector_universe_version": SECTOR_UNIVERSE_SCHEMA_VERSION,
        "universe_hash": manifest["universe_hash"],
        "sector_id": manifest["sector"]["sector_id"],
        "partition": "train",
        "episode_count": len(episodes),
        "total_train_rows": total_rows,
        "sampling_strategy": SECTOR_SAMPLING_POLICY_VERSION,
        "symbol_transition_requires_full_reset": True,
        "validation_references_included": False,
        "test_references_included": False,
        "episodes": episodes,
    }


def build_leave_one_out_sector_manifest(
    target_symbol: str,
    *,
    standard_manifest_path: Path,
    current_verified_path: Path,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Derive a deterministic target-excluded universe without loading prices."""

    try:
        standard = json.loads(Path(standard_manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SectorUniverseError(f"Could not load standard sector manifest: {exc}") from exc
    if not isinstance(standard, Mapping):
        raise SectorUniverseError("standard sector manifest must be an object")
    identity = standard.get("deterministic_identity")
    if (
        standard.get("artifact_schema_version") != SECTOR_UNIVERSE_SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or deterministic_universe_hash(identity) != standard.get("universe_hash")
    ):
        raise SectorUniverseError("standard sector manifest is stale or incompatible")
    experiment = standard.get("experiment_mode")
    if not isinstance(experiment, Mapping) or experiment.get("mode") != STANDARD_SECTOR_PRETRAINING:
        raise SectorUniverseError("leave-one-out must derive from a standard universe")
    try:
        current = pd.read_csv(current_verified_path, dtype={"symbol": "string"})
    except (OSError, pd.errors.ParserError) as exc:
        raise SectorUniverseError(f"Could not load current sector evidence: {exc}") from exc
    sector = standard.get("sector")
    cohort = standard.get("cohort")
    source = standard.get("source_provenance")
    if not all(isinstance(value, Mapping) for value in (sector, cohort, source)):
        raise SectorUniverseError("standard sector provenance is incomplete")
    return build_sector_manifest(
        current,
        sector_id=str(sector["sector_id"]),
        sector_name=str(sector["sector_name"]),
        cohort_cutoff=str(cohort["cohort_cutoff"]),
        listing_snapshot_date=str(cohort["listing_snapshot_date"]),
        source_registry_sha256=str(source["company_registry_sha256"]),
        source_registry_path=str(source["company_registry_path"]),
        source_listing_path=str(source["official_listing_snapshot_path"]),
        source_listing_sha256=str(source["official_listing_snapshot_sha256"]),
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        git_commit=(str(standard["git_commit"]) if standard.get("git_commit") else None),
        mode=LEAVE_ONE_SYMBOL_OUT,
        target_symbol=str(target_symbol).strip(),
    )


def build_sector_statistics(current: pd.DataFrame) -> pd.DataFrame:
    """Summarize diversity, depth, compatibility, and concentration by sector."""

    records: list[dict[str, object]] = []
    recognized = current.loc[current["sector_id"].ne("unknown")]
    for (sector_id, sector_name), rows in recognized.groupby(
        ["sector_id", "sector_name"], sort=True
    ):
        approved = rows.loc[rows["eligible_at_cutoff"].map(_as_bool)].copy()
        train = pd.to_numeric(approved["rows_available_at_cutoff"], errors="coerce").fillna(0)
        total = int(train.sum())
        concentration = float(train.max() / total) if total else 0.0
        compatible = int(rows["recurrent_compatible"].map(_as_bool).sum())
        verified_equity = int(rows["security_type"].eq("ordinary_equity").sum())
        mature = int(rows["history_class"].eq("MATURE").sum())
        cold = int(rows["history_class"].eq("COLD_START").sum())
        insufficient = int(rows["history_class"].eq("INSUFFICIENT").sum())
        median = float(train.median()) if not train.empty else 0.0
        homogeneous = bool(
            approved.empty
            or (
                approved["feature_version"].nunique() == 1
                and approved["recurrent_contract_version"].nunique() == 1
                and approved["environment_version"].nunique() == 1
                and approved["observation_features"].nunique() == 1
                and approved["normalization_scope"].nunique() == 1
            )
        )
        if int(rows["sector_verified_current"].map(_as_bool).sum()) == 0:
            readiness = UNVERIFIED_HISTORY
        elif (
            compatible >= MINIMUM_READY_MATURE_SYMBOLS
            and total >= MINIMUM_READY_TOTAL_TRAIN_ROWS
            and median >= MINIMUM_READY_MEDIAN_TRAIN_ROWS
            and concentration <= MAXIMUM_READY_PROPORTIONAL_CONCENTRATION
            and homogeneous
        ):
            readiness = READY_FOR_SECTOR_RESEARCH
        elif (
            compatible >= MINIMUM_LIMITED_MATURE_SYMBOLS
            and total >= MINIMUM_LIMITED_TOTAL_TRAIN_ROWS
            and homogeneous
        ):
            readiness = LIMITED
        else:
            readiness = INSUFFICIENT
        starts = approved["train_start"].astype("string")
        ends = approved["last_train_date"].astype("string")
        records.append(
            {
                "sector_id": sector_id,
                "sector_name": sector_name,
                "verified_symbols": len(rows),
                "verified_equity_symbols": verified_equity,
                "mature_symbols": mature,
                "cold_start_symbols": cold,
                "insufficient_symbols": insufficient,
                "recurrent_compatible_symbols": compatible,
                "approved_constituent_symbols": len(approved),
                "total_train_rows": total,
                "minimum_train_rows": int(train.min()) if not train.empty else 0,
                "first_quartile_train_rows": float(train.quantile(0.25)) if not train.empty else 0.0,
                "median_train_rows": median,
                "third_quartile_train_rows": float(train.quantile(0.75)) if not train.empty else 0.0,
                "maximum_train_rows": int(train.max()) if not train.empty else 0,
                "earliest_train_start": starts.min() if not starts.empty else "",
                "latest_train_end": ends.max() if not ends.empty else "",
                "maximum_proportional_symbol_share": concentration,
                "contracts_homogeneous": homogeneous,
                "verified_sector_evidence": bool(
                    rows["sector_verified_current"].map(_as_bool).all()
                ),
                "historical_membership_status": "LIMITED_CURRENT_SECTOR_ONLY",
                "research_readiness_status": readiness,
            }
        )
    order = {
        READY_FOR_SECTOR_RESEARCH: 0,
        LIMITED: 1,
        INSUFFICIENT: 2,
        UNVERIFIED_HISTORY: 3,
    }
    result = pd.DataFrame.from_records(records)
    result["_order"] = result["research_readiness_status"].map(order)
    return result.sort_values(
        ["_order", "recurrent_compatible_symbols", "total_train_rows", "sector_name"],
        ascending=[True, False, False, True],
        kind="stable",
    ).drop(columns="_order").reset_index(drop=True)


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def generate_local_sector_universes(
    *,
    output_directory: Path = SECTOR_UNIVERSES_DIR,
    registry_path: Path = COMPANY_REGISTRY_PATH,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    master_path: Path = MASTER_CSV_PATH,
    processed_master_path: Path = PROCESSED_MASTER_PATH,
    splits_dir: Path = PROCESSED_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    generated_at: str | None = None,
) -> SectorUniverseBuildResult:
    """Generate portable local sector research metadata without model training."""

    root = Path(project_root).resolve()
    registry_file = Path(registry_path)
    listings_file = Path(listings_path)
    master_file = Path(master_path)
    processed_master_file = Path(processed_master_path)
    for label, path in (
        ("company registry", registry_file),
        ("official listing snapshot", listings_file),
        ("master market data", master_file),
        ("processed master data", processed_master_file),
    ):
        if not path.is_file():
            raise SectorUniverseError(f"{label} is missing: {path}")
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    registry = pd.read_csv(registry_file, dtype={"symbol": "string"})
    listings = pd.read_csv(listings_file, dtype={"symbol": "string"})
    master = pd.read_csv(master_file, dtype={"symbol": "string"})
    # Read the processed file to establish local feature-version provenance; its
    # rows are not used to choose sector membership or validation performance.
    processed_versions = pd.read_csv(
        processed_master_file,
        usecols=["feature_version"],
    )["feature_version"].dropna().astype(str).unique()
    if tuple(processed_versions) != (FEATURE_VERSION,):
        raise SectorUniverseError("processed master feature version is incompatible")
    listing_dates = sorted(set(listings["snapshot_date"].dropna().astype(str)))
    if len(listing_dates) != 1:
        raise SectorUniverseError("listing snapshot must contain one snapshot date")
    listing_snapshot_date = _iso_date(
        listing_dates[0], label="listing snapshot date"
    )
    master_dates = pd.to_datetime(master["date"], errors="coerce")
    if master_dates.isna().any() or master_dates.empty:
        raise SectorUniverseError("master market data contains invalid dates")
    cohort_cutoff = master_dates.max().date().isoformat()
    readiness = build_training_readiness_report(master, registry)
    current = build_current_verified_universe(
        registry,
        readiness,
        cohort_cutoff=cohort_cutoff,
        listing_snapshot_date=listing_snapshot_date,
        splits_dir=Path(splits_dir),
        project_root=root,
    )
    historical = classify_historical_instruments(registry)
    statistics = build_sector_statistics(current)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_path = output / "sector_taxonomy.json"
    current_path = output / "current_verified_symbols.csv"
    historical_path = output / "historical_instrument_audit.csv"
    summary_path = output / "sector_universe_summary.csv"
    atomic_write_json(taxonomy_payload(generated_at=timestamp), taxonomy_path)
    write_dataframe_atomically(current, current_path)
    write_dataframe_atomically(historical, historical_path)
    write_dataframe_atomically(statistics, summary_path)

    registry_hash = sha256_file(registry_file)
    listing_hash = sha256_file(listings_file)
    manifest_paths: list[Path] = []
    episode_paths: list[Path] = []
    commit = _git_commit(root)
    for row in statistics.itertuples(index=False):
        sector_rows = current.loc[current["sector_id"].eq(row.sector_id)]
        manifest = build_sector_manifest(
            sector_rows,
            sector_id=row.sector_id,
            sector_name=row.sector_name,
            cohort_cutoff=cohort_cutoff,
            listing_snapshot_date=listing_snapshot_date,
            source_registry_sha256=registry_hash,
            source_registry_path=_project_relative(registry_file, project_root=root),
            source_listing_path=_project_relative(listings_file, project_root=root),
            source_listing_sha256=listing_hash,
            generated_at=timestamp,
            git_commit=commit,
        )
        episodes = build_train_episode_index(manifest)
        sector_directory = output / row.sector_id
        manifest_path = sector_directory / "universe_manifest.json"
        episode_path = sector_directory / "train_episode_index.json"
        atomic_write_json(manifest, manifest_path)
        atomic_write_json(episodes, episode_path)
        manifest_paths.append(manifest_path)
        episode_paths.append(episode_path)

    historical_groups = historical["historical_audit_group"].value_counts()
    return SectorUniverseBuildResult(
        output_directory=output,
        taxonomy_path=taxonomy_path,
        current_verified_path=current_path,
        historical_audit_path=historical_path,
        summary_path=summary_path,
        manifest_paths=tuple(manifest_paths),
        episode_index_paths=tuple(episode_paths),
        taxonomy_version=SECTOR_TAXONOMY_VERSION,
        cohort_cutoff=cohort_cutoff,
        listing_snapshot_date=listing_snapshot_date,
        verified_sector_symbols=len(current),
        historical_only_total=len(historical),
        historical_non_equity_contract_like=int(
            historical_groups.get("non_equity_or_contract_like", 0)
        ),
        historical_ordinary_equity_candidates=0,
        historical_unknown=int(
            historical_groups.get("unknown_requires_investigation", 0)
        ),
        sector_count=len(statistics),
        recurrent_compatible_symbols=int(
            current["recurrent_compatible"].map(_as_bool).sum()
        ),
        source_registry_sha256=registry_hash,
    )


__all__ = (
    "LEAVE_ONE_SYMBOL_OUT",
    "LIMITED",
    "READY_FOR_SECTOR_RESEARCH",
    "SECTOR_EPISODE_INDEX_SCHEMA_VERSION",
    "SECTOR_NORMALIZATION_POLICY_VERSION",
    "SECTOR_SAMPLING_POLICY_VERSION",
    "SECTOR_TAXONOMY_VERSION",
    "SECTOR_UNIVERSE_SCHEMA_VERSION",
    "STANDARD_SECTOR_PRETRAINING",
    "SectorTaxonomyMatch",
    "SectorUniverseBuildResult",
    "SectorUniverseError",
    "build_current_verified_universe",
    "build_leave_one_out_sector_manifest",
    "build_sector_manifest",
    "build_sector_statistics",
    "build_train_episode_index",
    "classify_historical_instruments",
    "deterministic_universe_hash",
    "generate_local_sector_universes",
    "normalize_sector",
    "taxonomy_payload",
)
