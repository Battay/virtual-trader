"""Unified read-only inventories and selective market-data maintenance controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .automation import (
    AutomationProgress,
    NativeSourceReconciliationResult,
    SourceDateDisposition,
    UpdateLock,
    karachi_today,
    load_automation_config,
    reconcile_native_source_csvs,
)
from .backfill import (
    BackfillDateResult,
    collect_date_with_backfill_retries,
    load_backfill_state,
    source_health_pause_required,
)
from .config import (
    AUTOMATION_CONFIG_PATH,
    AUTOMATION_LOCK_PATH,
    BACKFILL_STATE_PATH,
    DAILY_MARKET_PARQUET_DIR,
    DATA_MAINTENANCE_HISTORY_PATH,
    LOCAL_PSX_MARKET_PARQUET_PATH,
    MASTER_CSV_PATH,
    NATIVE_MARKET_PIPELINE_STATE_PATH,
    RAW_CSV_DIR,
    RAW_HTML_DIR,
    TRADING_DATE_EVIDENCE_PATH,
)
from .maintenance_history import (
    MaintenanceDateResult,
    MaintenanceHistory,
    append_maintenance_operation,
    load_maintenance_history,
    new_operation,
)
from .main import OUTPUT_FIELDS
from .native_market_pipeline import (
    BUSINESS_KEY,
    CANONICAL_ARROW_SCHEMA,
    CANONICAL_MARKET_COLUMNS,
    NativeMarketPaths,
    DailyParquetRepairResult,
    canonical_content_hash,
    canonical_content_hashes_by_date,
    repair_daily_parquet_partitions,
    sha256_file,
)
from .updater import valid_daily_csv_row_count


TRADING_DATE_EVIDENCE_VERSION = "trading_date_evidence_v1"
CSV_NAME_PATTERN = re.compile(r"market_(\d{4}-\d{2}-\d{2})\.csv")
PARQUET_NAME_PATTERN = re.compile(r"market_(\d{4}-\d{2}-\d{2})\.parquet")
HTML_DATE_PATTERN = re.compile(r"market_(\d{4}-\d{2}-\d{2})(?:_|\.)")


class DateClassification(str, Enum):
    CURRENT = "CURRENT"
    MISSING = "MISSING"
    LIKELY_NON_TRADING = "LIKELY_NON_TRADING"
    CONFIRMED_NON_TRADING = "CONFIRMED_NON_TRADING"
    SOURCE_ANOMALY = "SOURCE_ANOMALY"
    NOT_FINAL = "NOT_FINAL"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    INVALID_SOURCE = "INVALID_SOURCE"
    WEEKEND = "WEEKEND"


class ParquetDateState(str, Enum):
    CURRENT = "CURRENT"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    ORPHAN = "ORPHAN"


ACTIONABLE_FETCH_CLASSIFICATIONS = frozenset(
    {
        DateClassification.MISSING,
        DateClassification.NOT_FINAL,
        DateClassification.FAILED_RETRYABLE,
    }
)
REPAIRABLE_PARQUET_STATES = frozenset(
    {ParquetDateState.MISSING, ParquetDateState.STALE, ParquetDateState.CORRUPT}
)


class DataCompletenessError(RuntimeError):
    """Raised when an inventory or selective operation cannot proceed safely."""


@dataclass(frozen=True)
class TradingDateEvidence:
    trading_date: date
    classification: DateClassification
    reason: str
    evidence: str
    recorded_at: str | None = None


@dataclass(frozen=True)
class TradingDateEvidenceLedger:
    version: str
    records: tuple[TradingDateEvidence, ...]
    source_snapshot: Mapping[str, object]
    error: str | None = None

    @property
    def by_date(self) -> dict[date, TradingDateEvidence]:
        return {item.trading_date: item for item in self.records}


@dataclass(frozen=True)
class CsvDateRecord:
    trading_date: date
    classification: DateClassification
    csv_status: str
    raw_evidence_status: str
    attempts: int
    last_result: str
    last_checked: str | None
    actionable: bool
    reason: str
    csv_path: Path | None = None


@dataclass(frozen=True)
class CsvInventorySummary:
    earliest_valid_date: date | None
    latest_valid_date: date | None
    valid_accepted_dates: int
    actionable_missing_dates: int
    likely_non_trading_candidates: int
    confirmed_non_trading_dates: int
    source_anomalies: int
    retryable_not_final_dates: int
    invalid_corrupt_csv_count: int


@dataclass(frozen=True)
class ParquetDateRecord:
    trading_date: date
    state: ParquetDateState
    path: Path | None
    row_count: int | None
    expected_rows: int | None
    canonical_hash: str | None
    partition_hash: str | None
    reason: str

    @property
    def repairable(self) -> bool:
        return self.state in REPAIRABLE_PARQUET_STATES


@dataclass(frozen=True)
class ParquetInventorySummary:
    earliest_date: date | None
    latest_date: date | None
    current: int
    missing: int
    stale: int
    corrupt: int
    orphan: int


@dataclass(frozen=True)
class MasterArtifactStatus:
    path: Path
    exists: bool
    latest_date: date | None
    row_count: int | None
    date_count: int | None
    symbol_count: int | None
    integrity_status: str
    content_hash: str | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class MasterParityStatus:
    key_parity: bool
    logical_content_parity: bool
    source_set_hash_status: str
    content_hash_status: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataCompletenessInventory:
    generated_at: str
    start_date: date | None
    end_date: date
    csv_records: tuple[CsvDateRecord, ...]
    csv_summary: CsvInventorySummary
    parquet_records: tuple[ParquetDateRecord, ...]
    parquet_summary: ParquetInventorySummary
    master_csv: MasterArtifactStatus
    master_parquet: MasterArtifactStatus
    master_parity: MasterParityStatus
    pending_source_dates: tuple[date, ...]
    canonical_dates_with_noncurrent_daily: tuple[date, ...]
    history: MaintenanceHistory
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedFetchResult:
    requested_dates: tuple[date, ...]
    executed_dates: tuple[date, ...]
    skipped_dates: tuple[date, ...]
    outcomes: tuple[BackfillDateResult, ...]
    reconciliation: NativeSourceReconciliationResult | None
    circuit_breaker_triggered: bool
    status: str
    errors: tuple[str, ...] = ()


def _parse_date_list(payload: object, name: str) -> tuple[date, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return tuple(sorted({date.fromisoformat(str(value)) for value in payload}))


def load_trading_date_evidence(
    path: Path = TRADING_DATE_EVIDENCE_PATH,
) -> TradingDateEvidenceLedger:
    """Load the versioned classification ledger without inventing evidence."""

    ledger_path = Path(path)
    if not ledger_path.is_file():
        return TradingDateEvidenceLedger(
            version=TRADING_DATE_EVIDENCE_VERSION,
            records=(),
            source_snapshot={},
            error=f"Trading-date evidence ledger does not exist: {ledger_path}",
        )
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("trading-date evidence ledger must be an object")
        version = str(payload.get("version") or "")
        if version != TRADING_DATE_EVIDENCE_VERSION:
            raise ValueError(f"unsupported trading-date evidence version: {version}")
        source_snapshot = payload.get("source_snapshot", {})
        if not isinstance(source_snapshot, dict):
            raise ValueError("source_snapshot must be an object")
        records: list[TradingDateEvidence] = []
        grouped = {
            "confirmed_non_trading_dates": DateClassification.CONFIRMED_NON_TRADING,
            "likely_non_trading_dates": DateClassification.LIKELY_NON_TRADING,
            "failed_retryable_dates": DateClassification.FAILED_RETRYABLE,
            "source_anomaly_dates": DateClassification.SOURCE_ANOMALY,
            "not_final_dates": DateClassification.NOT_FINAL,
        }
        reasons = payload.get("date_details", {})
        if not isinstance(reasons, dict):
            raise ValueError("date_details must be an object")
        for field, classification in grouped.items():
            for trading_date in _parse_date_list(payload.get(field, []), field):
                detail = reasons.get(trading_date.isoformat(), {})
                if not isinstance(detail, dict):
                    raise ValueError("date detail must be an object")
                records.append(
                    TradingDateEvidence(
                        trading_date=trading_date,
                        classification=classification,
                        reason=str(
                            detail.get("reason")
                            or classification.value.replace("_", " ").title()
                        ),
                        evidence=str(detail.get("evidence") or "Versioned evidence ledger"),
                        recorded_at=(
                            str(detail["recorded_at"])
                            if detail.get("recorded_at")
                            else None
                        ),
                    )
                )
        dates = [item.trading_date for item in records]
        if len(dates) != len(set(dates)):
            raise ValueError("trading-date evidence contains duplicate classifications")
        return TradingDateEvidenceLedger(
            version=version,
            records=tuple(sorted(records, key=lambda item: item.trading_date)),
            source_snapshot=dict(source_snapshot),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return TradingDateEvidenceLedger(
            version=TRADING_DATE_EVIDENCE_VERSION,
            records=(),
            source_snapshot={},
            error=f"Could not load trading-date evidence: {exc}",
        )


def _disposition_evidence(
    dispositions: Sequence[SourceDateDisposition],
) -> dict[date, TradingDateEvidence]:
    mapping = {
        "confirmed_non_trading": DateClassification.CONFIRMED_NON_TRADING,
        "likely_non_trading": DateClassification.LIKELY_NON_TRADING,
        "source_anomaly": DateClassification.SOURCE_ANOMALY,
        "not_final": DateClassification.NOT_FINAL,
        "failed_retryable": DateClassification.FAILED_RETRYABLE,
    }
    return {
        item.trading_date: TradingDateEvidence(
            trading_date=item.trading_date,
            classification=mapping[item.classification],
            reason=item.reason,
            evidence=item.evidence,
            recorded_at=item.recorded_at,
        )
        for item in dispositions
        if item.classification in mapping
    }


def _date_from_name(path: Path, pattern: re.Pattern[str]) -> date | None:
    match = pattern.fullmatch(path.name)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _raw_html_inventory(raw_html_dir: Path) -> dict[date, tuple[Path, ...]]:
    values: dict[date, list[Path]] = {}
    if not Path(raw_html_dir).is_dir():
        return {}
    for path in sorted(Path(raw_html_dir).glob("market_*.html")):
        match = HTML_DATE_PATTERN.match(path.name)
        if match is None:
            continue
        try:
            trading_date = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        values.setdefault(trading_date, []).append(path)
    return {key: tuple(items) for key, items in values.items()}


def _last_completed_source_date(today: date) -> date:
    return today - timedelta(days=1)


def _csv_inventory(
    *,
    raw_csv_dir: Path,
    raw_html_dir: Path,
    ledger: TradingDateEvidenceLedger,
    backfill_state_path: Path,
    automation_config_path: Path,
    end_date: date,
    today: date,
) -> tuple[tuple[CsvDateRecord, ...], CsvInventorySummary, tuple[str, ...]]:
    csv_paths: dict[date, Path] = {}
    invalid_paths: dict[date, Path] = {}
    for path in sorted(Path(raw_csv_dir).glob("market_*.csv")):
        trading_date = _date_from_name(path, CSV_NAME_PATTERN)
        if trading_date is None:
            continue
        if valid_daily_csv_row_count(path, trading_date) is None:
            invalid_paths[trading_date] = path
        else:
            csv_paths[trading_date] = path
    html_by_date = _raw_html_inventory(raw_html_dir)
    backfill = load_backfill_state(backfill_state_path)
    automation = load_automation_config(automation_config_path)
    evidence = ledger.by_date
    evidence.update(_disposition_evidence(automation.source_date_dispositions))
    attempts_by_date = dict(backfill.attempt_records) if backfill is not None else {}
    temp_reasons = dict(backfill.temporary_skips) if backfill is not None else {}
    failed_reasons = dict(backfill.failed_dates) if backfill is not None else {}
    start_candidates = [*csv_paths, *invalid_paths, *evidence]
    if backfill is not None:
        start_candidates.append(backfill.requested_start_date)
    start_date = min(start_candidates) if start_candidates else None
    if start_date is None:
        return (
            (),
            CsvInventorySummary(None, None, 0, 0, 0, 0, 0, 0, len(invalid_paths)),
            tuple(filter(None, (ledger.error,))),
        )

    records: list[CsvDateRecord] = []
    current = start_date
    while current <= end_date:
        csv_path = csv_paths.get(current)
        invalid_path = invalid_paths.get(current)
        date_evidence = evidence.get(current)
        attempts = attempts_by_date.get(current, ())
        html_paths = html_by_date.get(current, ())
        last_result = ""
        last_checked: str | None = None
        reason = ""
        if csv_path is not None:
            classification = DateClassification.CURRENT
            csv_status = "VALID"
            reason = (
                "Valid legacy weekend source artifact; not fetch-actionable"
                if current.weekday() >= 5
                else "Valid daily CSV exists"
            )
            last_result = "successful"
            last_checked = datetime.fromtimestamp(
                csv_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        elif invalid_path is not None:
            classification = DateClassification.INVALID_SOURCE
            csv_status = "INVALID"
            reason = "A date-named CSV exists but fails the canonical source validator"
        elif current.weekday() >= 5:
            classification = DateClassification.WEEKEND
            csv_status = "NOT_EXPECTED"
            reason = "Weekend date"
        elif date_evidence is not None:
            classification = date_evidence.classification
            csv_status = "MISSING"
            reason = date_evidence.reason
            last_checked = date_evidence.recorded_at
            last_result = classification.value.lower()
        elif current in failed_reasons:
            classification = DateClassification.FAILED_RETRYABLE
            csv_status = "MISSING"
            reason = failed_reasons[current]
            last_result = "failed"
            last_checked = backfill.updated_at if backfill is not None else None
        elif current in temp_reasons:
            classification = (
                DateClassification.NOT_FINAL
                if "not final" in temp_reasons[current].lower()
                else DateClassification.FAILED_RETRYABLE
            )
            csv_status = "MISSING"
            reason = temp_reasons[current]
            last_result = "temporary_unavailable"
            last_checked = backfill.updated_at if backfill is not None else None
        else:
            classification = DateClassification.MISSING
            csv_status = "MISSING"
            reason = "No valid daily CSV or non-trading evidence exists"
        actionable = (
            classification in ACTIONABLE_FETCH_CLASSIFICATIONS
            and current < today
            and current.weekday() < 5
        )
        if classification == DateClassification.NOT_FINAL and current >= today:
            actionable = False
        if attempts:
            last_result = attempts[-1].result
        records.append(
            CsvDateRecord(
                trading_date=current,
                classification=classification,
                csv_status=csv_status,
                raw_evidence_status=(
                    f"{len(html_paths)} retained HTML response(s)"
                    if html_paths
                    else "No retained HTML"
                ),
                attempts=max(len(attempts), len(html_paths)),
                last_result=last_result or "not_attempted",
                last_checked=last_checked,
                actionable=actionable,
                reason=reason,
                csv_path=csv_path or invalid_path,
            )
        )
        current += timedelta(days=1)

    weekday_valid = [value for value in csv_paths if value.weekday() < 5]
    summary = CsvInventorySummary(
        earliest_valid_date=min(csv_paths) if csv_paths else None,
        latest_valid_date=max(csv_paths) if csv_paths else None,
        valid_accepted_dates=len(weekday_valid),
        actionable_missing_dates=sum(item.actionable for item in records),
        likely_non_trading_candidates=sum(
            item.classification == DateClassification.LIKELY_NON_TRADING
            for item in records
        ),
        confirmed_non_trading_dates=sum(
            item.classification == DateClassification.CONFIRMED_NON_TRADING
            for item in records
        ),
        source_anomalies=sum(
            item.classification == DateClassification.SOURCE_ANOMALY
            for item in records
        ),
        retryable_not_final_dates=sum(
            item.classification
            in {DateClassification.NOT_FINAL, DateClassification.FAILED_RETRYABLE}
            and item.csv_status == "MISSING"
            for item in records
        ),
        invalid_corrupt_csv_count=len(invalid_paths),
    )
    return tuple(records), summary, tuple(filter(None, (ledger.error,)))


def _canonical_frame(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(CANONICAL_MARKET_COLUMNS))
    frame = table.to_pandas()
    frame["market_date"] = pd.to_datetime(frame["market_date"])
    frame["symbol"] = frame["symbol"].astype("string")
    for column in ("sector_current", "sector_source", "sector_snapshot_date"):
        frame[column] = frame[column].astype("string")
    return frame.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)


def _read_daily_partition(path: Path) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    if parquet_file.schema_arrow.remove_metadata() != CANONICAL_ARROW_SCHEMA:
        raise DataCompletenessError("schema does not match canonical daily contract")
    frame = pq.read_table(path).to_pandas()
    frame["market_date"] = pd.to_datetime(frame["market_date"])
    frame["symbol"] = frame["symbol"].astype("string")
    for column in ("sector_current", "sector_source", "sector_snapshot_date"):
        frame[column] = frame[column].astype("string")
    if frame.empty:
        raise DataCompletenessError("partition is empty")
    if frame.duplicated(list(BUSINESS_KEY)).any():
        raise DataCompletenessError("partition contains duplicate keys")
    ordered = frame.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(ordered):
        raise DataCompletenessError("partition is not deterministically ordered")
    return frame.loc[:, list(CANONICAL_MARKET_COLUMNS)]


def _partition_sector_snapshot_date(
    parquet_file: pq.ParquetFile,
) -> str | None:
    """Read one uniform sector snapshot value from Parquet footer statistics."""

    try:
        column_index = parquet_file.schema_arrow.names.index("sector_snapshot_date")
        values: set[str] = set()
        for index in range(parquet_file.metadata.num_row_groups):
            statistics = parquet_file.metadata.row_group(index).column(
                column_index
            ).statistics
            if (
                statistics is None
                or not statistics.has_min_max
                or statistics.min != statistics.max
            ):
                return None
            values.add(str(statistics.min))
        return next(iter(values)) if len(values) == 1 else None
    except (AttributeError, ValueError):
        return None


def _matches_except_sector_snapshot_date(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> bool:
    """Compare stable market and sector assignment semantics.

    ``sector_snapshot_date`` records when current listing context was attached;
    it is not an effective-dated market observation. A newer listing refresh
    must not make otherwise identical historical market partitions repairable.
    """

    if canonical_content_hash(actual, core_only=True) != canonical_content_hash(
        expected, core_only=True
    ):
        return False
    for column in ("sector_current", "sector_source"):
        equal = (
            actual[column].eq(expected[column])
            | (actual[column].isna() & expected[column].isna())
        ).fillna(False)
        if not bool(equal.all()):
            return False
    return True


def _embedded_hash_matches_prior_sector_snapshot(
    parquet_file: pq.ParquetFile,
    expected: pd.DataFrame,
    embedded_hash: str,
) -> bool:
    """Use verified footer provenance to avoid reading unchanged data pages."""

    prior_snapshot = _partition_sector_snapshot_date(parquet_file)
    if not prior_snapshot or not embedded_hash:
        return False
    expected_at_prior_snapshot = expected.copy()
    available = expected_at_prior_snapshot["sector_snapshot_date"].notna()
    expected_at_prior_snapshot.loc[available, "sector_snapshot_date"] = prior_snapshot
    return canonical_content_hash(expected_at_prior_snapshot) == embedded_hash


def _classify_parquet_date(
    trading_date: date,
    expected: pd.DataFrame | None,
    path: Path | None,
    expected_hash: str | None,
) -> ParquetDateRecord:
    if expected is None:
        return ParquetDateRecord(
            trading_date,
            ParquetDateState.ORPHAN,
            path,
            None,
            None,
            None,
            None,
            "Daily Parquet exists without a canonical market-date slice",
        )
    if path is None:
        return ParquetDateRecord(
            trading_date,
            ParquetDateState.MISSING,
            None,
            None,
            len(expected),
            expected_hash,
            None,
            "Canonical date exists but daily Parquet is missing",
        )
    try:
        parquet_file = pq.ParquetFile(path)
        if parquet_file.schema_arrow.remove_metadata() != CANONICAL_ARROW_SCHEMA:
            raise DataCompletenessError("schema does not match canonical daily contract")
        metadata = parquet_file.schema_arrow.metadata or {}
        embedded_hash = metadata.get(b"canonical_content_hash", b"").decode()
        if embedded_hash == expected_hash and parquet_file.metadata.num_rows == len(expected):
            return ParquetDateRecord(
                trading_date,
                ParquetDateState.CURRENT,
                path,
                parquet_file.metadata.num_rows,
                len(expected),
                expected_hash,
                embedded_hash,
                "Readable canonical schema and persisted logical hash match",
            )
        if (
            parquet_file.metadata.num_rows == len(expected)
            and _embedded_hash_matches_prior_sector_snapshot(
                parquet_file,
                expected,
                embedded_hash,
            )
        ):
            return ParquetDateRecord(
                trading_date,
                ParquetDateState.CURRENT,
                path,
                parquet_file.metadata.num_rows,
                len(expected),
                expected_hash,
                embedded_hash,
                "Market values and sector assignments match; only the current-sector "
                "snapshot provenance is older",
            )
        # Only suspicious partitions pay the full data-page read cost. The
        # native writer embeds the canonical slice hash after validation;
        # matching schema/footer/hash is the persisted verified fast path.
        actual = _read_daily_partition(path)
        actual_dates = set(actual["market_date"].dt.date.unique())
        if actual_dates != {trading_date}:
            raise DataCompletenessError("partition contains the wrong market date")
        actual_hash = canonical_content_hash(actual)
        state = (
            ParquetDateState.CURRENT
            if actual_hash == expected_hash
            or _matches_except_sector_snapshot_date(actual, expected)
            else ParquetDateState.STALE
        )
        reason = (
            (
                "Readable partition matches the canonical date slice"
                if actual_hash == expected_hash
                else "Market values and sector assignments match; only the "
                "current-sector snapshot provenance is older"
            )
            if state == ParquetDateState.CURRENT
            else "Readable partition differs from the canonical date slice"
        )
        return ParquetDateRecord(
            trading_date,
            state,
            path,
            len(actual),
            len(expected),
            expected_hash,
            actual_hash,
            reason,
        )
    except (OSError, ValueError, KeyError, pa.ArrowException, DataCompletenessError) as exc:
        return ParquetDateRecord(
            trading_date,
            ParquetDateState.CORRUPT,
            path,
            None,
            len(expected),
            expected_hash,
            None,
            f"Unreadable or structurally invalid partition: {exc}",
        )


def _parquet_inventory(
    canonical: pd.DataFrame,
    *,
    daily_dir: Path,
) -> tuple[tuple[ParquetDateRecord, ...], ParquetInventorySummary]:
    grouped = {
        pd.Timestamp(key).date(): group.reset_index(drop=True)
        for key, group in canonical.groupby("market_date", sort=True, observed=True)
    }
    expected_hashes = canonical_content_hashes_by_date(canonical)
    paths: dict[date, Path] = {}
    for path in sorted(Path(daily_dir).glob("market_*.parquet")):
        trading_date = _date_from_name(path, PARQUET_NAME_PATTERN)
        if trading_date is not None:
            paths[trading_date] = path
    all_dates = tuple(sorted(set(grouped).union(paths)))
    with ThreadPoolExecutor(max_workers=min(16, max(1, len(all_dates)))) as executor:
        records = list(
            executor.map(
                lambda value: _classify_parquet_date(
                    value,
                    grouped.get(value),
                    paths.get(value),
                    expected_hashes.get(value),
                ),
                all_dates,
            )
        )
    canonical_dates = sorted(grouped)
    summary = ParquetInventorySummary(
        earliest_date=canonical_dates[0] if canonical_dates else None,
        latest_date=canonical_dates[-1] if canonical_dates else None,
        current=sum(item.state == ParquetDateState.CURRENT for item in records),
        missing=sum(item.state == ParquetDateState.MISSING for item in records),
        stale=sum(item.state == ParquetDateState.STALE for item in records),
        corrupt=sum(item.state == ParquetDateState.CORRUPT for item in records),
        orphan=sum(item.state == ParquetDateState.ORPHAN for item in records),
    )
    return tuple(records), summary


def _master_statuses(
    *,
    master_csv_path: Path,
    consolidated_path: Path,
    state_path: Path,
    canonical: pd.DataFrame,
) -> tuple[MasterArtifactStatus, MasterArtifactStatus, MasterParityStatus]:
    state: dict[str, object] = {}
    state_errors: list[str] = []
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("state is not an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        state_errors.append(f"Native pipeline state is invalid: {exc}")
    parquet_hash = canonical_content_hash(canonical)
    canonical_dates = pd.to_datetime(canonical["market_date"])
    parquet_errors: list[str] = []
    metadata = pq.ParquetFile(consolidated_path).schema_arrow.metadata or {}
    if canonical.duplicated(list(BUSINESS_KEY)).any():
        parquet_errors.append("Consolidated Parquet contains duplicate keys")
    if not canonical.reset_index(drop=True).equals(
        canonical.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)
    ):
        parquet_errors.append("Consolidated Parquet is not deterministically ordered")
    if state and parquet_hash != str(state.get("canonical_content_hash") or ""):
        parquet_errors.append("Consolidated logical content hash differs from state")
    state_sha = str(state.get("consolidated_sha256") or "")
    physical_hash = sha256_file(consolidated_path)
    if state_sha and physical_hash != state_sha:
        parquet_errors.append("Consolidated physical hash differs from state")
    parquet_status = MasterArtifactStatus(
        path=Path(consolidated_path),
        exists=True,
        latest_date=(canonical_dates.max().date() if not canonical.empty else None),
        row_count=len(canonical),
        date_count=int(canonical["market_date"].nunique()),
        symbol_count=int(canonical["symbol"].nunique()),
        integrity_status="PASS" if not parquet_errors else "FAIL",
        content_hash=parquet_hash,
        errors=tuple(parquet_errors),
    )

    csv_errors: list[str] = []
    csv_frame: pd.DataFrame | None = None
    if not Path(master_csv_path).is_file():
        csv_errors.append("Canonical master CSV does not exist")
    else:
        try:
            csv_frame = pd.read_csv(
                master_csv_path,
                dtype={"symbol": "string"},
                float_precision="round_trip",
            )
            if tuple(csv_frame.columns) != CANONICAL_MARKET_COLUMNS:
                raise DataCompletenessError("canonical master CSV schema is invalid")
            csv_frame["market_date"] = pd.to_datetime(csv_frame["market_date"])
            for column in ("sector_current", "sector_source", "sector_snapshot_date"):
                csv_frame[column] = csv_frame[column].astype("string")
            if csv_frame.duplicated(list(BUSINESS_KEY)).any():
                csv_errors.append("Canonical master CSV contains duplicate keys")
            if not csv_frame.reset_index(drop=True).equals(
                csv_frame.sort_values(list(BUSINESS_KEY), kind="mergesort").reset_index(drop=True)
            ):
                csv_errors.append("Canonical master CSV is not deterministically ordered")
        except (OSError, UnicodeError, ValueError, KeyError, pd.errors.ParserError, DataCompletenessError) as exc:
            csv_errors.append(f"Could not validate canonical master CSV: {exc}")
            csv_frame = None
    csv_hash = canonical_content_hash(csv_frame) if csv_frame is not None else None
    if state and csv_hash is not None and csv_hash != str(state.get("canonical_content_hash") or ""):
        csv_errors.append("Canonical master CSV logical hash differs from state")
    csv_dates = (
        pd.to_datetime(csv_frame["market_date"]) if csv_frame is not None else pd.Series(dtype="datetime64[ns]")
    )
    csv_status = MasterArtifactStatus(
        path=Path(master_csv_path),
        exists=Path(master_csv_path).is_file(),
        latest_date=(csv_dates.max().date() if not csv_dates.empty else None),
        row_count=(len(csv_frame) if csv_frame is not None else None),
        date_count=(int(csv_frame["market_date"].nunique()) if csv_frame is not None else None),
        symbol_count=(int(csv_frame["symbol"].nunique()) if csv_frame is not None else None),
        integrity_status="PASS" if not csv_errors else "FAIL",
        content_hash=csv_hash,
        errors=tuple(csv_errors),
    )
    key_parity = bool(
        csv_frame is not None
        and len(csv_frame) == len(canonical)
        and pd.MultiIndex.from_frame(csv_frame.loc[:, list(BUSINESS_KEY)]).equals(
            pd.MultiIndex.from_frame(canonical.loc[:, list(BUSINESS_KEY)])
        )
    )
    logical_parity = bool(csv_hash is not None and csv_hash == parquet_hash)
    source_hash_status = (
        "PASS"
        if state
        and metadata.get(b"source_set_hash", b"").decode()
        == str(state.get("source_set_hash") or "")
        else "FAIL"
    )
    content_hash_status = (
        "PASS"
        if state
        and csv_hash == parquet_hash == str(state.get("canonical_content_hash") or "")
        else "FAIL"
    )
    parity_errors = [*state_errors]
    if not key_parity:
        parity_errors.append("Canonical master keys differ")
    if not logical_parity:
        parity_errors.append("Canonical master logical values differ")
    return (
        csv_status,
        parquet_status,
        MasterParityStatus(
            key_parity,
            logical_parity,
            source_hash_status,
            content_hash_status,
            tuple(parity_errors),
        ),
    )


def build_data_completeness_inventory(
    *,
    raw_csv_dir: Path = RAW_CSV_DIR,
    raw_html_dir: Path = RAW_HTML_DIR,
    daily_parquet_dir: Path = DAILY_MARKET_PARQUET_DIR,
    master_csv_path: Path = MASTER_CSV_PATH,
    consolidated_path: Path = LOCAL_PSX_MARKET_PARQUET_PATH,
    native_state_path: Path = NATIVE_MARKET_PIPELINE_STATE_PATH,
    backfill_state_path: Path = BACKFILL_STATE_PATH,
    automation_config_path: Path = AUTOMATION_CONFIG_PATH,
    evidence_path: Path = TRADING_DATE_EVIDENCE_PATH,
    history_path: Path = DATA_MAINTENANCE_HISTORY_PATH,
    end_date: date | None = None,
    now: datetime | None = None,
) -> DataCompletenessInventory:
    """Build the complete read-only data-maintenance inventory."""

    today = karachi_today(now)
    inventory_end = end_date or _last_completed_source_date(today)
    ledger = load_trading_date_evidence(evidence_path)
    csv_records, csv_summary, warnings = _csv_inventory(
        raw_csv_dir=raw_csv_dir,
        raw_html_dir=raw_html_dir,
        ledger=ledger,
        backfill_state_path=backfill_state_path,
        automation_config_path=automation_config_path,
        end_date=inventory_end,
        today=today,
    )
    if not Path(consolidated_path).is_file():
        raise DataCompletenessError(
            f"Canonical consolidated Parquet does not exist: {consolidated_path}"
        )
    canonical = _canonical_frame(consolidated_path)
    parquet_records, parquet_summary = _parquet_inventory(
        canonical, daily_dir=daily_parquet_dir
    )
    csv_master, parquet_master, parity = _master_statuses(
        master_csv_path=master_csv_path,
        consolidated_path=consolidated_path,
        state_path=native_state_path,
        canonical=canonical,
    )
    canonical_dates = set(pd.to_datetime(canonical["market_date"]).dt.date.unique())
    valid_source_dates = {
        item.trading_date
        for item in csv_records
        if item.csv_status == "VALID" and item.trading_date.weekday() < 5
    }
    pending = tuple(sorted(valid_source_dates.difference(canonical_dates)))
    noncurrent = tuple(
        item.trading_date
        for item in parquet_records
        if item.state != ParquetDateState.CURRENT
        and item.trading_date in canonical_dates
    )
    history = load_maintenance_history(history_path)
    all_warnings = list(warnings)
    if history.error:
        all_warnings.append(history.error)
    return DataCompletenessInventory(
        generated_at=(now or datetime.now(timezone.utc)).isoformat(),
        start_date=(csv_records[0].trading_date if csv_records else None),
        end_date=inventory_end,
        csv_records=csv_records,
        csv_summary=csv_summary,
        parquet_records=parquet_records,
        parquet_summary=parquet_summary,
        master_csv=csv_master,
        master_parquet=parquet_master,
        master_parity=parity,
        pending_source_dates=pending,
        canonical_dates_with_noncurrent_daily=noncurrent,
        history=history,
        warnings=tuple(all_warnings),
    )


def update_visible_selection(
    current: Sequence[date],
    visible: Sequence[date],
    selected_visible: Sequence[date],
) -> tuple[date, ...]:
    """Replace only visible membership while preserving hidden selections."""

    visible_set = set(visible)
    return tuple(
        sorted((set(current).difference(visible_set)).union(selected_visible))
    )


def select_visible_actionable(
    current: Sequence[date], records: Sequence[CsvDateRecord]
) -> tuple[date, ...]:
    return tuple(
        sorted(set(current).union(item.trading_date for item in records if item.actionable))
    )


def clear_visible_selection(
    current: Sequence[date], visible: Sequence[date]
) -> tuple[date, ...]:
    return tuple(sorted(set(current).difference(visible)))


def validate_fetch_selection(
    selected_dates: Sequence[date], records: Sequence[CsvDateRecord]
) -> tuple[CsvDateRecord, ...]:
    """Fail closed unless every selected date is currently fetch-actionable."""

    selected = tuple(sorted(set(selected_dates)))
    by_date = {item.trading_date: item for item in records}
    missing = [value for value in selected if value not in by_date]
    blocked = [value for value in selected if value in by_date and not by_date[value].actionable]
    if not selected:
        raise DataCompletenessError("No source dates were selected")
    if missing or blocked:
        values = [*missing, *blocked]
        raise DataCompletenessError(
            "Selected source date is not currently actionable: "
            + ", ".join(value.isoformat() for value in values)
        )
    return tuple(by_date[value] for value in selected)


def fetch_selected_dates(
    selected_dates: Sequence[date],
    records: Sequence[CsvDateRecord],
    *,
    collector: Callable[[date], BackfillDateResult] = collect_date_with_backfill_retries,
    reconciler: Callable[..., NativeSourceReconciliationResult] = reconcile_native_source_csvs,
    history_path: Path = DATA_MAINTENANCE_HISTORY_PATH,
    progress_callback: Callable[[AutomationProgress], None] | None = None,
    circuit_window: int = 10,
    circuit_unhealthy_ratio: float = 0.70,
) -> SelectedFetchResult:
    """Fetch exact validated membership and reconcile only successful sources."""

    selected_records = validate_fetch_selection(selected_dates, records)
    history = load_maintenance_history(history_path)
    if history.error is not None:
        raise DataCompletenessError(history.error)
    outcomes: list[BackfillDateResult] = []
    executed: list[date] = []
    circuit_triggered = False
    errors: list[str] = []
    for item in selected_records:
        executed.append(item.trading_date)
        try:
            outcome = collector(item.trading_date)
        except Exception as exc:  # noqa: BLE001 - isolate one selected date.
            outcome = BackfillDateResult(
                item.trading_date,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        outcomes.append(outcome)
        if source_health_pause_required(
            outcomes,
            window_size=circuit_window,
            unhealthy_ratio=circuit_unhealthy_ratio,
        ):
            circuit_triggered = True
            break
    skipped = tuple(
        item.trading_date for item in selected_records if item.trading_date not in set(executed)
    )
    paths = tuple(
        Path(item.output_path)
        for item in outcomes
        if item.status in {"successful", "already_downloaded"}
        and item.output_path is not None
    )
    reconciliation: NativeSourceReconciliationResult | None = None
    if paths:
        try:
            reconciliation = reconciler(
                paths,
                reference_date=max(executed),
                progress_callback=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001 - record failed promotion truthfully.
            errors.append(f"Native reconciliation failed: {type(exc).__name__}: {exc}")
    if errors:
        status = "FAILED"
    elif circuit_triggered:
        status = "PAUSED"
    elif outcomes and all(
        item.status in {"failed", "temporary_unavailable"} for item in outcomes
    ):
        status = "FAILED"
    elif any(item.status in {"failed", "temporary_unavailable"} for item in outcomes):
        status = "PARTIAL"
    else:
        status = "COMPLETED"
    native = reconciliation.native if reconciliation is not None else None
    operation = new_operation(
        "FETCH_SELECTED",
        requested_dates=[item.trading_date.isoformat() for item in selected_records],
        executed_dates=[value.isoformat() for value in executed],
        skipped_dates=[value.isoformat() for value in skipped],
        per_date_results=[
            MaintenanceDateResult(
                trading_date=item.trading_date.isoformat(),
                status=item.status,
                message=item.reason,
                input_classification=next(
                    record.classification.value
                    for record in selected_records
                    if record.trading_date == item.trading_date
                ),
                attempts=item.attempt_count,
                output_path=(str(item.output_path) if item.output_path else None),
            )
            for item in outcomes
        ],
        artifact_status={
            "canonical_master_csv": "UPDATED" if native and native.rows_added else "CURRENT",
            "consolidated_parquet": "UPDATED" if native and native.rows_added else "CURRENT",
            "daily_partitions_affected": native.daily_parquets_written if native else 0,
            "symbol_artifacts_affected": native.symbol_csvs_written if native else 0,
            "logical_parity": "PASS" if native is not None or not paths else "FAIL",
        },
        errors=errors,
        master_latest_date=(native.latest_date if native else None),
        source_set_hash=(native.source_set_hash if native else None),
        content_hash=(native.content_hash if native else None),
    )
    append_maintenance_operation(operation, history_path)
    return SelectedFetchResult(
        requested_dates=tuple(item.trading_date for item in selected_records),
        executed_dates=tuple(executed),
        skipped_dates=skipped,
        outcomes=tuple(outcomes),
        reconciliation=reconciliation,
        circuit_breaker_triggered=circuit_triggered,
        status=status,
        errors=tuple(errors),
    )


def repair_selected_parquet_dates(
    selected_dates: Sequence[date],
    records: Sequence[ParquetDateRecord],
    *,
    repairer: Callable[..., DailyParquetRepairResult] = repair_daily_parquet_partitions,
    paths: NativeMarketPaths = NativeMarketPaths(),
    lock_path: Path = AUTOMATION_LOCK_PATH,
    history_path: Path = DATA_MAINTENANCE_HISTORY_PATH,
) -> DailyParquetRepairResult:
    """Lock, validate, repair, and persist an exact daily-partition selection."""

    selected = tuple(sorted(set(selected_dates)))
    by_date = {item.trading_date: item for item in records}
    invalid = [
        value
        for value in selected
        if value not in by_date or not by_date[value].repairable
    ]
    if not selected:
        raise DataCompletenessError("No daily Parquet dates were selected")
    if invalid:
        raise DataCompletenessError(
            "Selected Parquet date is not repairable: "
            + ", ".join(value.isoformat() for value in invalid)
        )
    history = load_maintenance_history(history_path)
    if history.error is not None:
        raise DataCompletenessError(history.error)
    lock = UpdateLock(lock_path)
    lock.acquire()
    try:
        result = repairer(selected, paths=paths)
    finally:
        lock.release()
    operation = new_operation(
        "PARQUET_REPAIR_SELECTED",
        requested_dates=[value.isoformat() for value in selected],
        executed_dates=result.repaired_dates,
        skipped_dates=result.already_current_dates,
        per_date_results=[
            MaintenanceDateResult(
                trading_date=value.isoformat(),
                status="REPAIRED" if value.isoformat() in result.repaired_dates else "CURRENT",
                input_classification=by_date[value].state.value,
            )
            for value in selected
        ],
        artifact_status={
            "canonical_master_csv": result.master_csv_status,
            "consolidated_parquet": result.consolidated_parquet_status,
            "daily_partitions_affected": result.daily_parquets_written,
            "symbol_artifacts_affected": 0,
            "logical_parity": "PASS" if result.logical_parity else "FAIL",
        },
        master_latest_date=result.latest_date,
        source_set_hash=result.source_set_hash,
        content_hash=result.content_hash,
    )
    append_maintenance_operation(operation, history_path)
    return result


def reconcile_pending_source_dates(
    pending_dates: Sequence[date],
    *,
    raw_csv_dir: Path = RAW_CSV_DIR,
    reconciler: Callable[..., NativeSourceReconciliationResult] = reconcile_native_source_csvs,
    history_path: Path = DATA_MAINTENANCE_HISTORY_PATH,
    progress_callback: Callable[[AutomationProgress], None] | None = None,
) -> NativeSourceReconciliationResult:
    """Reconcile exact already-valid local sources without any HTTP request."""

    selected = tuple(sorted(set(pending_dates)))
    if not selected:
        raise DataCompletenessError("No valid source dates await incorporation")
    history = load_maintenance_history(history_path)
    if history.error is not None:
        raise DataCompletenessError(history.error)
    paths = tuple(
        Path(raw_csv_dir) / f"market_{value.isoformat()}.csv" for value in selected
    )
    invalid = [
        value
        for value, path in zip(selected, paths, strict=True)
        if valid_daily_csv_row_count(path, value) is None
    ]
    if invalid:
        raise DataCompletenessError(
            "Pending source failed validation: "
            + ", ".join(value.isoformat() for value in invalid)
        )
    result = reconciler(
        paths,
        reference_date=max(selected),
        progress_callback=progress_callback,
    )
    native = result.native
    append_maintenance_operation(
        new_operation(
            "MASTER_RECONCILE",
            requested_dates=[value.isoformat() for value in selected],
            executed_dates=[value.isoformat() for value in selected],
            per_date_results=[
                MaintenanceDateResult(
                    value.isoformat(),
                    "reconciled",
                    input_classification="CURRENT_SOURCE_PENDING",
                    output_path=str(path),
                )
                for value, path in zip(selected, paths, strict=True)
            ],
            artifact_status={
                "canonical_master_csv": "UPDATED" if native.rows_added else "CURRENT",
                "consolidated_parquet": "UPDATED" if native.rows_added else "CURRENT",
                "daily_partitions_affected": native.daily_parquets_written,
                "symbol_artifacts_affected": native.symbol_csvs_written,
                "logical_parity": "PASS",
            },
            master_latest_date=native.latest_date,
            source_set_hash=native.source_set_hash,
            content_hash=native.content_hash,
        ),
        history_path,
    )
    return result
