"""Leakage-safe market and macro primitives for the Phase-2 Lead Agent.

The module implements the data boundary only.  It deliberately does not create
an RL environment, train a model, open a Phase-2 TEST observation frame, or
download network data.  Authoritative source acquisition is a separate,
provenance-preserving operation whose outputs must satisfy this contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT
from .phase1_closure import FINAL_PHASE1_DECISION, load_phase1_closure


PHASE2_DATA_CONTRACT_VERSION = "lead_agent_market_macro_contract_v1"
PHASE2_MACRO_SCHEMA_VERSION = "lead_agent_macro_observation_v1"
PHASE2_SPLIT_VERSION = "lead_agent_common_calendar_split_v1"
PHASE2_MARKET_FEATURE_VERSION = "lead_agent_market_features_v1"
PHASE2_MACRO_FEATURE_VERSION = "lead_agent_macro_features_v1"
PHASE2_SCALER_VERSION = "lead_agent_train_standard_scaler_v1"
PHASE2_DATA_CONTRACT_DECISION = "BLOCKED_PHASE2_DATA_CONTRACT"
PHASE2_DATA_CONTRACT_ARTIFACT = (
    PROJECT_ROOT / "docs" / "config" / "phase2_lead_agent_data_contract_v1.json"
)

DECISION_TIMEZONE = "Asia/Karachi"
DECISION_CUTOFF_POLICY = "start_of_decision_session_calendar_day"
REQUIRED_INDEX_CODES = ("KSE100", "KSE30", "KMI30", "ALLSHR")
POLICY_RATE_SERIES = "sbp_policy_target_rate"
CPI_YOY_SERIES = "pbs_national_cpi_yoy"
USD_PKR_SERIES = "sbp_usd_pkr_m2m"
REQUIRED_MACRO_SERIES = (
    POLICY_RATE_SERIES,
    CPI_YOY_SERIES,
    USD_PKR_SERIES,
)

MACRO_REQUIRED_COLUMNS = (
    "schema_version",
    "series",
    "reference_date",
    "value",
    "unit",
    "native_frequency",
    "release_date",
    "release_timestamp",
    "effective_available_timestamp",
    "availability_method",
    "vintage_id",
    "source",
    "source_version",
    "retrieved_at",
    "provenance_hash",
    "point_in_time_safe",
)

MARKET_FEATURE_COLUMNS = (
    "kse100_return_1d_lag1",
    "kse100_return_5d_lag1",
    "kse100_return_21d_lag1",
    "kse100_volatility_20d_lag1",
    "kse100_drawdown_expanding_lag1",
    "kse100_sma20_distance_lag1",
    "kse100_sma50_distance_lag1",
    "kse100_volume_ratio20_lag1",
    "cross_index_return_dispersion_1d_lag1",
)

MACRO_FEATURE_COLUMNS = (
    "sbp_policy_rate",
    "sbp_policy_rate_change",
    "sbp_policy_rate_age_days",
    "pbs_cpi_yoy",
    "pbs_cpi_yoy_change",
    "pbs_cpi_release_age_days",
    "sbp_usd_pkr_m2m",
    "sbp_usd_pkr_return_1d",
    "sbp_usd_pkr_volatility_20d",
    "sbp_usdpkr_release_age_days",
)

LEAD_AGENT_FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS
_HASH_EXCLUDED_FIELDS = frozenset({"generated_at", "contract_evidence_hash"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class Phase2DataContractError(RuntimeError):
    """Raised when Phase-2 input evidence is absent, malformed, or unsafe."""


@dataclass(frozen=True)
class TrainValidationScalingResult:
    """TRAIN-fitted scaler and transformed non-TEST partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    scaler: StandardScaler
    metadata: dict[str, Any]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def deterministic_hash(value: object) -> str:
    """Return a stable SHA-256 identity for canonical JSON-compatible data."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def contract_evidence_hash(payload: Mapping[str, Any]) -> str:
    """Hash nonvolatile contract evidence while excluding archival time."""

    evidence = {
        key: value
        for key, value in payload.items()
        if key not in _HASH_EXCLUDED_FIELDS
    }
    return deterministic_hash(evidence)


def _json_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number == 0.0:
            return 0.0
        return number
    if pd.isna(value):
        return None
    return str(value)


def deterministic_frame_hash(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    """Hash logical tabular content independently of row and column layout."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise Phase2DataContractError(
            "Cannot hash frame missing columns: " + ", ".join(missing)
        )
    ordered = frame.loc[:, list(columns)].sort_values(
        list(sort_by), kind="stable", na_position="first"
    )
    rows = [
        [_json_scalar(value) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    return deterministic_hash({"columns": list(columns), "rows": rows})


def build_raw_evidence_manifest(
    source_file: str | Path,
    *,
    source_url: str,
    source_identifier: str,
    retrieved_at: str | datetime,
    media_type: str,
    parse_status: str = "not_parsed",
    source_version: str = "not_stated",
) -> dict[str, Any]:
    """Describe an already-downloaded source file without altering it."""

    path = Path(source_file)
    if not path.is_file():
        raise Phase2DataContractError(f"Raw macro evidence is missing: {path}")
    if not source_url.startswith("https://"):
        raise Phase2DataContractError("Raw macro evidence requires an HTTPS source")
    timestamp = pd.to_datetime(retrieved_at, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise Phase2DataContractError("Raw macro retrieval timestamp is invalid")
    content = path.read_bytes()
    return {
        "manifest_version": "macro_raw_evidence_manifest_v1",
        "source_identifier": str(source_identifier),
        "source_url": source_url,
        "retrieved_at": timestamp.isoformat(),
        "filename": path.name,
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": str(media_type),
        "parse_status": str(parse_status),
        "source_version": str(source_version),
    }


def _normalise_boolean(series: pd.Series, *, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    text = series.astype("string").str.lower()
    if not text.isin(("true", "false")).all():
        raise Phase2DataContractError(f"Macro {field} must be boolean")
    return text.eq("true")


def validate_macro_observations(
    macro: pd.DataFrame,
    *,
    required_series: Iterable[str] = REQUIRED_MACRO_SERIES,
    require_point_in_time_safe: bool = True,
) -> pd.DataFrame:
    """Validate and normalize canonical macro vintages without mutating input."""

    if not isinstance(macro, pd.DataFrame):
        raise Phase2DataContractError("Macro observations must be a dataframe")
    missing = sorted(set(MACRO_REQUIRED_COLUMNS).difference(macro.columns))
    if missing:
        raise Phase2DataContractError(
            "Macro observations are missing required columns: " + ", ".join(missing)
        )
    data = macro.loc[:, list(MACRO_REQUIRED_COLUMNS)].copy(deep=True)
    if data.empty:
        raise Phase2DataContractError("Macro observations are empty")

    for column in (
        "schema_version",
        "series",
        "unit",
        "native_frequency",
        "availability_method",
        "vintage_id",
        "source",
        "source_version",
        "provenance_hash",
    ):
        data[column] = data[column].astype("string").str.strip()
        if data[column].isna().any() or data[column].eq("").any():
            raise Phase2DataContractError(f"Macro {column} contains empty values")
    if not data["schema_version"].eq(PHASE2_MACRO_SCHEMA_VERSION).all():
        raise Phase2DataContractError("Macro schema version is incompatible")

    data["reference_date"] = pd.to_datetime(
        data["reference_date"], errors="coerce"
    ).dt.normalize()
    data["release_date"] = pd.to_datetime(
        data["release_date"], errors="coerce"
    ).dt.normalize()
    for column in (
        "release_timestamp",
        "effective_available_timestamp",
        "retrieved_at",
    ):
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True)
    if data[["reference_date", "release_date"]].isna().any(axis=None):
        raise Phase2DataContractError("Macro reference/release date is invalid")
    if data[["effective_available_timestamp", "retrieved_at"]].isna().any(
        axis=None
    ):
        raise Phase2DataContractError(
            "Macro effective/retrieval timestamp is required and must include a timezone"
        )

    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data["value"].isna().any() or not np.isfinite(data["value"]).all():
        raise Phase2DataContractError("Macro values must be finite numbers")
    data["point_in_time_safe"] = _normalise_boolean(
        data["point_in_time_safe"], field="point_in_time_safe"
    )
    if require_point_in_time_safe and not data["point_in_time_safe"].all():
        raise Phase2DataContractError("Unsafe macro vintages cannot enter Phase-2")

    required = {str(value) for value in required_series}
    present = set(data["series"].astype(str))
    absent = sorted(required.difference(present))
    if absent:
        raise Phase2DataContractError(
            "Required macro series are missing: " + ", ".join(absent)
        )
    unexpected = sorted(present.difference(REQUIRED_MACRO_SERIES))
    if unexpected:
        raise Phase2DataContractError(
            "Unsupported macro series are present: " + ", ".join(unexpected)
        )
    if not data["provenance_hash"].map(
        lambda value: bool(_SHA256_PATTERN.fullmatch(str(value)))
    ).all():
        raise Phase2DataContractError("Macro provenance hash is malformed")

    release_known = data["release_timestamp"].notna()
    if (
        data.loc[release_known, "effective_available_timestamp"]
        < data.loc[release_known, "release_timestamp"]
    ).any():
        raise Phase2DataContractError(
            "Macro availability cannot precede its release timestamp"
        )
    if (
        data["effective_available_timestamp"].dt.date
        < data["release_date"].dt.date
    ).any():
        raise Phase2DataContractError(
            "Macro availability cannot precede its release date"
        )
    if (
        data.loc[release_known, "retrieved_at"]
        < data.loc[release_known, "release_timestamp"]
    ).any():
        raise Phase2DataContractError(
            "Macro evidence retrieval cannot predate a known release timestamp"
        )

    policy = data["series"].eq(POLICY_RATE_SERIES)
    cpi = data["series"].eq(CPI_YOY_SERIES)
    fx = data["series"].eq(USD_PKR_SERIES)
    if data.loc[policy, "value"].lt(0).any():
        raise Phase2DataContractError("SBP policy rate cannot be negative")
    if data.loc[fx, "value"].le(0).any():
        raise Phase2DataContractError("USD/PKR rate must be positive")
    if (data.loc[cpi, "release_date"] <= data.loc[cpi, "reference_date"]).any():
        raise Phase2DataContractError(
            "CPI release must follow the represented reference month"
        )
    if (data.loc[fx, "release_date"] < data.loc[fx, "reference_date"]).any():
        raise Phase2DataContractError("FX release cannot precede its reference date")
    if (
        data.loc[policy, "effective_available_timestamp"].dt.date
        < data.loc[policy, "reference_date"].dt.date
    ).any():
        raise Phase2DataContractError(
            "A policy setting cannot be used before its official effective date"
        )

    duplicate_key = [
        "series",
        "reference_date",
        "effective_available_timestamp",
        "vintage_id",
    ]
    if data.duplicated(duplicate_key, keep=False).any():
        raise Phase2DataContractError("Macro canonical vintage keys are duplicated")

    return data.sort_values(
        ["effective_available_timestamp", "series", "reference_date", "vintage_id"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_index_history(indices: pd.DataFrame) -> pd.DataFrame:
    required = {"index_code", "date", "value", "volume"}
    missing = sorted(required.difference(indices.columns))
    if missing:
        raise Phase2DataContractError(
            "Official index history is missing columns: " + ", ".join(missing)
        )
    data = indices.loc[:, ["index_code", "date", "value", "volume"]].copy(
        deep=True
    )
    data["index_code"] = data["index_code"].astype("string").str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    if data[["index_code", "date", "value", "volume"]].isna().any(axis=None):
        raise Phase2DataContractError("Official index history has invalid values")
    if not np.isfinite(data[["value", "volume"]].to_numpy()).all():
        raise Phase2DataContractError("Official index history has non-finite values")
    if data["value"].le(0).any() or data["volume"].lt(0).any():
        raise Phase2DataContractError("Official index levels/volumes are invalid")
    codes = set(data["index_code"].astype(str))
    absent = sorted(set(REQUIRED_INDEX_CODES).difference(codes))
    if absent:
        raise Phase2DataContractError(
            "Required official indices are missing: " + ", ".join(absent)
        )
    if data.duplicated(["index_code", "date"], keep=False).any():
        raise Phase2DataContractError("Official index keys are duplicated")
    return data.sort_values(["date", "index_code"], kind="stable").reset_index(
        drop=True
    )


def decision_cutoffs(decision_dates: Iterable[object]) -> pd.DatetimeIndex:
    """Return conservative start-of-session decision cutoffs in UTC."""

    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates), errors="coerce"))
    if dates.isna().any():
        raise Phase2DataContractError("Decision calendar contains invalid dates")
    dates = dates.normalize()
    if dates.tz is None:
        dates = dates.tz_localize(DECISION_TIMEZONE)
    else:
        dates = dates.tz_convert(DECISION_TIMEZONE).normalize()
    return dates.tz_convert("UTC")


def build_causal_market_features(
    indices: pd.DataFrame,
    *,
    include_realized_target: bool = False,
) -> pd.DataFrame:
    """Derive level-based market features with an explicit one-session lag."""

    source = _validate_index_history(indices)
    values = source.pivot(index="date", columns="index_code", values="value")
    volumes = source.pivot(index="date", columns="index_code", values="volume")
    values = values.reindex(columns=list(REQUIRED_INDEX_CODES)).dropna(how="any")
    volumes = volumes.reindex(index=values.index, columns=list(REQUIRED_INDEX_CODES))
    if values.empty:
        raise Phase2DataContractError("Official indices have no common calendar")

    one_day = values.pct_change(fill_method=None)
    kse100 = values["KSE100"]
    kse100_one_day = one_day["KSE100"]
    output = pd.DataFrame(index=values.index)
    output["decision_date"] = values.index
    output["decision_cutoff"] = decision_cutoffs(values.index)
    output["kse100_return_1d_lag1"] = kse100_one_day.shift(1)
    output["kse100_return_5d_lag1"] = kse100.pct_change(
        5, fill_method=None
    ).shift(1)
    output["kse100_return_21d_lag1"] = kse100.pct_change(
        21, fill_method=None
    ).shift(1)
    output["kse100_volatility_20d_lag1"] = kse100_one_day.rolling(
        20, min_periods=20
    ).std(ddof=0).shift(1)
    output["kse100_drawdown_expanding_lag1"] = (
        kse100.div(kse100.cummax()).sub(1.0).shift(1)
    )
    output["kse100_sma20_distance_lag1"] = (
        kse100.div(kse100.rolling(20, min_periods=20).mean()).sub(1.0).shift(1)
    )
    output["kse100_sma50_distance_lag1"] = (
        kse100.div(kse100.rolling(50, min_periods=50).mean()).sub(1.0).shift(1)
    )
    output["kse100_volume_ratio20_lag1"] = (
        volumes["KSE100"]
        .div(volumes["KSE100"].rolling(20, min_periods=20).mean())
        .sub(1.0)
        .shift(1)
    )
    output["cross_index_return_dispersion_1d_lag1"] = one_day.std(
        axis=1, ddof=0
    ).shift(1)
    if include_realized_target:
        output["target_kse100_close_t_to_t_plus_1"] = kse100.pct_change(
            fill_method=None
        ).shift(-1)
        output["target_end_date"] = pd.Series(
            values.index, index=values.index
        ).shift(-1)
    return output.reset_index(drop=True)


def _macro_state_timeline(series: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct latest-reference vintages as they became knowable."""

    ordered = series.sort_values(
        ["effective_available_timestamp", "reference_date", "vintage_id"],
        kind="stable",
    )
    known: dict[pd.Timestamp, pd.Series] = {}
    records: list[dict[str, Any]] = []
    last_state_identity: tuple[object, ...] | None = None
    for _, releases in ordered.groupby(
        "effective_available_timestamp", sort=True
    ):
        for _, row in releases.iterrows():
            known[pd.Timestamp(row["reference_date"])] = row
        latest_reference = max(known)
        latest = known[latest_reference]
        state_identity = (
            latest_reference,
            float(latest["value"]),
            str(latest["vintage_id"]),
            str(latest["provenance_hash"]),
            pd.Timestamp(latest["effective_available_timestamp"]),
        )
        # A revision to an older period is valid evidence but does not make the
        # latest current-period state younger or create a synthetic FX return.
        if state_identity == last_state_identity:
            continue
        records.append(
            {
                "effective_available_timestamp": pd.Timestamp(
                    latest["effective_available_timestamp"]
                ),
                "reference_date": latest_reference,
                "value": float(latest["value"]),
                "release_date": pd.Timestamp(latest["release_date"]),
                "provenance_hash": str(latest["provenance_hash"]),
            }
        )
        last_state_identity = state_identity
    return pd.DataFrame.from_records(records)


def align_release_aware_macro(
    decisions: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Backward as-of join of latest safely released macro observations."""

    required_decisions = {"decision_date", "decision_cutoff"}
    missing = sorted(required_decisions.difference(decisions.columns))
    if missing:
        raise Phase2DataContractError(
            "Decision frame is missing columns: " + ", ".join(missing)
        )
    left = decisions.copy(deep=True)
    left["decision_date"] = pd.to_datetime(
        left["decision_date"], errors="coerce"
    ).dt.normalize()
    left["decision_cutoff"] = pd.to_datetime(
        left["decision_cutoff"], errors="coerce", utc=True
    )
    if left[["decision_date", "decision_cutoff"]].isna().any(axis=None):
        raise Phase2DataContractError("Decision frame contains invalid timestamps")
    if left["decision_date"].duplicated().any():
        raise Phase2DataContractError("Decision dates must be unique")
    left = left.sort_values("decision_cutoff", kind="stable").reset_index(drop=True)
    canonical = validate_macro_observations(macro)

    mappings = {
        POLICY_RATE_SERIES: (
            "sbp_policy_rate",
            "sbp_policy_rate_change",
            "sbp_policy_rate_age_days",
        ),
        CPI_YOY_SERIES: (
            "pbs_cpi_yoy",
            "pbs_cpi_yoy_change",
            "pbs_cpi_release_age_days",
        ),
        USD_PKR_SERIES: (
            "sbp_usd_pkr_m2m",
            "sbp_usd_pkr_return_1d",
            "sbp_usdpkr_release_age_days",
        ),
    }
    result = left
    for series_name, (level_name, change_name, age_name) in mappings.items():
        timeline = _macro_state_timeline(
            canonical.loc[canonical["series"].eq(series_name)]
        )
        timeline[level_name] = timeline["value"]
        timeline[change_name] = timeline[level_name].diff()
        if series_name == USD_PKR_SERIES:
            timeline[change_name] = timeline[level_name].pct_change(fill_method=None)
            timeline["sbp_usd_pkr_volatility_20d"] = timeline[
                change_name
            ].rolling(20, min_periods=20).std(ddof=0)
        timeline = timeline.rename(
            columns={
                "effective_available_timestamp": f"{series_name}_available_at",
                "reference_date": f"{series_name}_reference_date",
                "provenance_hash": f"{series_name}_provenance_hash",
            }
        )
        available_at = f"{series_name}_available_at"
        keep = [
            available_at,
            f"{series_name}_reference_date",
            f"{series_name}_provenance_hash",
            level_name,
            change_name,
        ]
        if series_name == USD_PKR_SERIES:
            keep.append("sbp_usd_pkr_volatility_20d")
        result = pd.merge_asof(
            result.sort_values("decision_cutoff"),
            timeline.loc[:, keep].sort_values(available_at),
            left_on="decision_cutoff",
            right_on=available_at,
            direction="backward",
            allow_exact_matches=True,
        )
        result[age_name] = (
            result["decision_cutoff"] - result[available_at]
        ).dt.total_seconds().div(86_400.0)
        known = result[available_at].notna()
        if (
            result.loc[known, available_at]
            > result.loc[known, "decision_cutoff"]
        ).any():
            raise Phase2DataContractError("Future macro release entered an as-of join")
    return result.sort_values("decision_date", kind="stable").reset_index(drop=True)


def build_market_macro_candidates(
    indices: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    include_realized_target: bool = False,
) -> pd.DataFrame:
    """Build complete causal candidate rows; incomplete early rows are excluded."""

    market = build_causal_market_features(
        indices, include_realized_target=include_realized_target
    )
    combined = align_release_aware_macro(market, macro)
    required = list(LEAD_AGENT_FEATURE_COLUMNS)
    if include_realized_target:
        required.extend(("target_kse100_close_t_to_t_plus_1", "target_end_date"))
    result = combined.dropna(subset=required).copy()
    if result.empty:
        raise Phase2DataContractError(
            "No complete common market/macro decision dates are available"
        )
    if not np.isfinite(result.loc[:, LEAD_AGENT_FEATURE_COLUMNS].to_numpy()).all():
        raise Phase2DataContractError("Phase-2 features contain non-finite values")
    return result.sort_values("decision_date", kind="stable").reset_index(drop=True)


def _date_hash(dates: Sequence[pd.Timestamp]) -> str:
    return deterministic_hash([pd.Timestamp(value).date().isoformat() for value in dates])


def freeze_common_calendar_split(
    decision_dates: Iterable[object],
) -> dict[str, Any]:
    """Freeze 70/15/15 common-calendar metadata without loading observations."""

    dates = pd.DatetimeIndex(
        pd.to_datetime(list(decision_dates), errors="coerce")
    ).normalize()
    if dates.isna().any() or len(dates) < 7:
        raise Phase2DataContractError(
            "Common calendar needs at least seven valid decision dates"
        )
    if dates.has_duplicates:
        raise Phase2DataContractError("Common decision calendar contains duplicates")
    if not dates.is_monotonic_increasing:
        raise Phase2DataContractError("Common decision calendar is not monotonic")
    total = len(dates)
    train_count = math.floor(total * 0.70)
    validation_count = math.floor(total * 0.15)
    test_count = total - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise Phase2DataContractError("Common split produced an empty partition")
    partitions = {
        "train": dates[:train_count],
        "validation": dates[train_count : train_count + validation_count],
        "test": dates[train_count + validation_count :],
    }
    metadata: dict[str, Any] = {
        "split_version": PHASE2_SPLIT_VERSION,
        "calendar_scope": "COMMON_MARKET_MACRO_CALENDAR",
        "construction": "first_floor_70_next_floor_15_remainder_15",
        "total_decision_dates": total,
        "calendar_first_date": dates[0].date().isoformat(),
        "calendar_last_date": dates[-1].date().isoformat(),
        "calendar_hash": _date_hash(list(dates)),
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "test_status": "SEALED",
        "test_observations_loaded": False,
    }
    for name, partition_dates in partitions.items():
        metadata[name] = {
            "row_count": len(partition_dates),
            "first_date": partition_dates[0].date().isoformat(),
            "last_date": partition_dates[-1].date().isoformat(),
            "date_hash": _date_hash(list(partition_dates)),
        }
    metadata["split_hash"] = deterministic_hash(metadata)
    return metadata


def select_train_validation_rows(
    non_test_rows: pd.DataFrame,
    split: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select TRAIN/VALIDATION while failing if a TEST row was supplied."""

    if "decision_date" not in non_test_rows.columns:
        raise Phase2DataContractError("Decision rows lack decision_date")
    data = non_test_rows.copy(deep=True)
    data["decision_date"] = pd.to_datetime(
        data["decision_date"], errors="coerce"
    ).dt.normalize()
    if data["decision_date"].isna().any():
        raise Phase2DataContractError("Decision rows contain invalid dates")
    test_first = pd.Timestamp(split["test"]["first_date"])
    if data["decision_date"].ge(test_first).any():
        raise Phase2DataContractError(
            "TEST observations were supplied to the TRAIN/VALIDATION selector"
        )
    train_last = pd.Timestamp(split["train"]["last_date"])
    validation_first = pd.Timestamp(split["validation"]["first_date"])
    validation_last = pd.Timestamp(split["validation"]["last_date"])
    train = data.loc[data["decision_date"].le(train_last)].copy()
    validation = data.loc[
        data["decision_date"].between(validation_first, validation_last)
    ].copy()
    if len(train) != int(split["train"]["row_count"]):
        raise Phase2DataContractError("TRAIN row count does not match split metadata")
    if len(validation) != int(split["validation"]["row_count"]):
        raise Phase2DataContractError(
            "VALIDATION row count does not match split metadata"
        )
    if set(train["decision_date"]).intersection(validation["decision_date"]):
        raise Phase2DataContractError("TRAIN and VALIDATION dates overlap")
    return (
        train.sort_values("decision_date", kind="stable").reset_index(drop=True),
        validation.sort_values("decision_date", kind="stable").reset_index(drop=True),
    )


def fit_train_only_scaler(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = LEAD_AGENT_FEATURE_COLUMNS,
) -> TrainValidationScalingResult:
    """Fit StandardScaler on TRAIN and transform VALIDATION only."""

    columns = tuple(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise Phase2DataContractError("Scaler feature ordering is empty or duplicated")
    for name, frame in (("TRAIN", train), ("VALIDATION", validation)):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise Phase2DataContractError(
                f"{name} is missing scaler features: " + ", ".join(missing)
            )
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if values.isna().any(axis=None) or not np.isfinite(values.to_numpy()).all():
            raise Phase2DataContractError(f"{name} scaler values are invalid")
    if train.empty or validation.empty:
        raise Phase2DataContractError("TRAIN and VALIDATION must both be non-empty")

    train_copy = train.copy(deep=True)
    validation_copy = validation.copy(deep=True)
    scaler = StandardScaler()
    train_values = train_copy.loc[:, columns].to_numpy(dtype=np.float64)
    validation_values = validation_copy.loc[:, columns].to_numpy(dtype=np.float64)
    train_copy.loc[:, columns] = scaler.fit_transform(train_values).astype(np.float32)
    validation_copy.loc[:, columns] = scaler.transform(validation_values).astype(
        np.float32
    )
    training_dates = (
        pd.to_datetime(train["decision_date"], errors="coerce").dt.date.astype(str).tolist()
        if "decision_date" in train.columns
        else list(range(len(train)))
    )
    identity = {
        "scaler_version": PHASE2_SCALER_VERSION,
        "implementation": "sklearn.preprocessing.StandardScaler",
        "sklearn_version": sklearn.__version__,
        "fit_partition": "TRAIN_ONLY",
        "validation_fit_rows": 0,
        "test_rows_loaded": False,
        "feature_columns": list(columns),
        "training_row_count": len(train),
        "training_date_hash": deterministic_hash(training_dates),
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "variance": [float(value) for value in scaler.var_],
    }
    metadata = {**identity, "scaler_hash": deterministic_hash(identity)}
    return TrainValidationScalingResult(
        train=train_copy,
        validation=validation_copy,
        scaler=scaler,
        metadata=metadata,
    )


def audit_macro_coverage(macro: pd.DataFrame) -> dict[str, Any]:
    """Return non-mutating coverage and provenance diagnostics by series."""

    canonical = validate_macro_observations(macro)
    records: dict[str, Any] = {}
    for name, group in canonical.groupby("series", sort=True):
        records[str(name)] = {
            "rows": len(group),
            "unique_reference_dates": int(group["reference_date"].nunique()),
            "reference_first": group["reference_date"].min().date().isoformat(),
            "reference_last": group["reference_date"].max().date().isoformat(),
            "release_first": group["release_date"].min().date().isoformat(),
            "release_last": group["release_date"].max().date().isoformat(),
            "available_first": group["effective_available_timestamp"]
            .min()
            .isoformat(),
            "available_last": group["effective_available_timestamp"]
            .max()
            .isoformat(),
            "point_in_time_safe_rows": int(group["point_in_time_safe"].sum()),
            "provenance_files": int(group["provenance_hash"].nunique()),
        }
    return {
        "schema_version": PHASE2_MACRO_SCHEMA_VERSION,
        "total_rows": len(canonical),
        "series": records,
        "canonical_hash": deterministic_frame_hash(
            canonical,
            columns=MACRO_REQUIRED_COLUMNS,
            sort_by=(
                "effective_available_timestamp",
                "series",
                "reference_date",
                "vintage_id",
            ),
        ),
    }


def validate_phase2_data_contract(payload: Mapping[str, Any]) -> None:
    """Fail closed on fabricated readiness, leakage, or provenance drift."""

    expected = {
        "artifact_version": PHASE2_DATA_CONTRACT_VERSION,
        "phase": "PHASE_2",
        "milestone": "P2.2",
        "decision": PHASE2_DATA_CONTRACT_DECISION,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise Phase2DataContractError(f"Phase-2 data contract {field} changed")
    phase1 = payload.get("phase1_inherited_constraint")
    if not isinstance(phase1, Mapping) or phase1.get("decision") != FINAL_PHASE1_DECISION:
        raise Phase2DataContractError("Phase-1 rejection constraint is missing")
    frozen_phase1 = load_phase1_closure()
    if phase1.get("evidence_hash") != frozen_phase1.get("decision_evidence_hash"):
        raise Phase2DataContractError("Phase-1 evidence identity changed")

    sources = payload.get("authoritative_sources")
    if not isinstance(sources, list) or {
        item.get("series") for item in sources if isinstance(item, Mapping)
    } != set(REQUIRED_MACRO_SERIES):
        raise Phase2DataContractError("Authoritative macro source inventory is incomplete")
    if any(item.get("authority") != "FIRST_PARTY" for item in sources):
        raise Phase2DataContractError("A non-authoritative macro source entered P2.2")

    timing = payload.get("point_in_time_contract")
    if not isinstance(timing, Mapping):
        raise Phase2DataContractError("Point-in-time contract is missing")
    if timing.get("decision_cutoff_policy") != DECISION_CUTOFF_POLICY:
        raise Phase2DataContractError("Decision cutoff policy changed")
    if timing.get("market_information_latest_allowed") != "t_minus_1_session":
        raise Phase2DataContractError("Same-session market data is not allowed")
    if timing.get("macro_join") != "backward_asof_effective_availability":
        raise Phase2DataContractError("Macro release-aware join is missing")
    if timing.get("future_interpolation") is not False:
        raise Phase2DataContractError("Future interpolation was enabled")
    if timing.get("revision_backfill") is not False:
        raise Phase2DataContractError("Revision backfill was enabled")

    macro = payload.get("canonical_macro_dataset")
    split = payload.get("common_calendar_split")
    scaler = payload.get("normalization")
    if not all(isinstance(item, Mapping) for item in (macro, split, scaler)):
        raise Phase2DataContractError("P2.2 dataset/split/scaler status is missing")
    if macro.get("created") is not False or macro.get("status") != "BLOCKED":
        raise Phase2DataContractError("Missing macro evidence was presented as canonical")
    if split.get("frozen") is not False or split.get("test_observations_loaded") is not False:
        raise Phase2DataContractError("Unverified common split was frozen or TEST opened")
    if scaler.get("fitted") is not False or scaler.get("fit_partition") != "TRAIN_ONLY":
        raise Phase2DataContractError("Unverified scaler was fitted or policy changed")

    safety = payload.get("safety")
    if not isinstance(safety, Mapping):
        raise Phase2DataContractError("P2.2 safety declaration is missing")
    required_false = (
        "test_observations_loaded",
        "lead_agent_environment_implemented",
        "rl_training_performed",
        "phase3_agents_modified",
        "source_data_modified",
        "model_artifacts_modified",
    )
    if any(safety.get(field) is not False for field in required_false):
        raise Phase2DataContractError("P2.2 safety declaration is not fail-closed")
    if safety.get("test_status") != "SEALED":
        raise Phase2DataContractError("Phase-2 TEST is not sealed")

    serialized = _canonical_json(payload)
    if "/Users/" in serialized or "\\\\Users\\" in serialized:
        raise Phase2DataContractError("P2.2 artifact contains a machine-specific path")
    recorded = payload.get("contract_evidence_hash")
    if not isinstance(recorded, str) or not _SHA256_PATTERN.fullmatch(recorded):
        raise Phase2DataContractError("P2.2 contract evidence hash is malformed")
    if recorded != contract_evidence_hash(payload):
        raise Phase2DataContractError("P2.2 contract evidence hash verification failed")


def load_phase2_data_contract(
    path: str | Path = PHASE2_DATA_CONTRACT_ARTIFACT,
) -> dict[str, Any]:
    """Load and verify the tracked P2.2 contract decision artifact."""

    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2DataContractError(
            f"Phase-2 data contract artifact is unavailable: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase2DataContractError("Phase-2 data contract must be an object")
    validate_phase2_data_contract(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Phase-2 market+macro data-contract decision."
    )
    parser.add_argument("--artifact", default=str(PHASE2_DATA_CONTRACT_ARTIFACT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = load_phase2_data_contract(args.artifact)
    except Phase2DataContractError as exc:
        print(f"BLOCKED_PHASE2_DATA_CONTRACT: {exc}")
        return 2
    print(
        f"{payload['decision']} ({payload['contract_evidence_hash']}); "
        f"reason={payload['blocking_reason']}"
    )
    return 2 if payload["decision"] == "BLOCKED_PHASE2_DATA_CONTRACT" else 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = (
    "CPI_YOY_SERIES",
    "DECISION_CUTOFF_POLICY",
    "LEAD_AGENT_FEATURE_COLUMNS",
    "MACRO_FEATURE_COLUMNS",
    "MACRO_REQUIRED_COLUMNS",
    "MARKET_FEATURE_COLUMNS",
    "PHASE2_DATA_CONTRACT_ARTIFACT",
    "PHASE2_DATA_CONTRACT_DECISION",
    "PHASE2_DATA_CONTRACT_VERSION",
    "PHASE2_MACRO_SCHEMA_VERSION",
    "PHASE2_SPLIT_VERSION",
    "POLICY_RATE_SERIES",
    "Phase2DataContractError",
    "TrainValidationScalingResult",
    "USD_PKR_SERIES",
    "align_release_aware_macro",
    "audit_macro_coverage",
    "build_causal_market_features",
    "build_market_macro_candidates",
    "build_raw_evidence_manifest",
    "contract_evidence_hash",
    "deterministic_frame_hash",
    "deterministic_hash",
    "decision_cutoffs",
    "fit_train_only_scaler",
    "freeze_common_calendar_split",
    "load_phase2_data_contract",
    "main",
    "select_train_validation_rows",
    "validate_macro_observations",
    "validate_phase2_data_contract",
)
