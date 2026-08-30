"""Automation configuration, locking, and standalone update orchestration."""

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal, Sequence
from zoneinfo import ZoneInfo

from .config import (
    AUTOMATION_CONFIG_PATH,
    AUTOMATION_LOCK_PATH,
    AUTO_UPDATE_LOG_PATH,
    BACKFILL_STATE_PATH,
    CURRENT_LISTINGS_PATH,
    DATA_MAINTENANCE_HISTORY_PATH,
    NATIVE_MARKET_PIPELINE_STATE_PATH,
    LOCAL_PSX_MARKET_PARQUET_PATH,
    PROJECT_TIMEZONE,
    RAW_CSV_DIR,
)
from .backfill import load_backfill_state
from .company_registry import RegistryBuildResult, build_company_registry
from .csv_store import MasterBuildResult, build_master_dataset
from .official_listings import ListingsRefreshResult, refresh_official_listings
from .native_market_pipeline import (
    NativeMarketBuildResult,
    incremental_update as run_native_incremental_update,
    rebuild_generated_artifacts,
)
from .maintenance_history import (
    MaintenanceDateResult,
    MaintenanceOperation,
    append_maintenance_operation,
    new_operation,
)
from .updater import (
    IncrementalUpdateResult,
    SourceEvidenceInventory,
    discover_source_evidence,
    run_incremental_update,
)
from market_intelligence.refresh_indices import IndexRefreshResult, refresh_indices
from feature_engineering.dataset_builder import build_master_ai_dataset, build_symbol_datasets


LOGGER = logging.getLogger(__name__)
KARACHI_TIMEZONE = PROJECT_TIMEZONE
SCHEDULED_TIME = "17:15"
DEFAULT_LOCK_STALE_AFTER = timedelta(hours=6)
STALE_RECOVERY_MESSAGE = (
    "Recovered an abandoned automation run with no live lock owner"
)
RunStatus = Literal[
    "disabled",
    "success",
    "partial_success",
    "failed",
    "no_update_needed",
    "already_running",
]
Updater = Callable[..., IncrementalUpdateResult]
MasterBuilder = Callable[..., MasterBuildResult]
ListingRefresher = Callable[..., ListingsRefreshResult]
RegistryBuilder = Callable[..., RegistryBuildResult]
IndexRefresher = Callable[..., IndexRefreshResult]
NativeUpdater = Callable[..., NativeMarketBuildResult]
NativeRebuilder = Callable[..., NativeMarketBuildResult]
EvidenceDiscoverer = Callable[..., SourceEvidenceInventory]
ProgressCallback = Callable[["AutomationProgress"], None]


def _resolved_history_path(
    explicit: Path | None,
    *,
    sibling_of: Path,
) -> Path:
    if explicit is not None:
        return Path(explicit)
    if Path(sibling_of) == AUTOMATION_CONFIG_PATH or Path(sibling_of) == AUTOMATION_LOCK_PATH:
        return DATA_MAINTENANCE_HISTORY_PATH
    return Path(sibling_of).parent / DATA_MAINTENANCE_HISTORY_PATH.name


def _append_history_safely(
    operation: MaintenanceOperation,
    path: Path,
) -> None:
    try:
        append_maintenance_operation(operation, path)
    except Exception:
        LOGGER.exception("Could not append market-data maintenance history")


@dataclass(frozen=True)
class AutomationProgress:
    """Truthful stage event shared by CLI, scheduler logging, and Streamlit."""

    stage: str
    message: str


@dataclass(frozen=True)
class AutomationRunAudit:
    """Durable details for the latest finalized orchestration run."""

    started_at: str
    finished_at: str | None
    target_end_date: str
    missing_dates_discovered: tuple[str, ...] = ()
    dates_attempted: tuple[str, ...] = ()
    dates_downloaded: tuple[str, ...] = ()
    dates_skipped: tuple[str, ...] = ()
    dates_failed: tuple[tuple[str, str], ...] = ()
    native_update_status: str = "not_started"
    native_rows_added: int = 0
    native_rows_replaced: int = 0
    daily_parquets_created: int = 0
    consolidated_parquet_latest_date: str | None = None
    ai_rebuild_status: str = "not_requested"
    source_evidence_inconsistencies: tuple[str, ...] = ()
    stale_run_recovered: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["dates_failed"] = {
            value: reason for value, reason in self.dates_failed
        }
        return values

    @classmethod
    def from_dict(cls, values: object) -> "AutomationRunAudit | None":
        if not isinstance(values, dict):
            return None

        def strings(name: str) -> tuple[str, ...]:
            raw = values.get(name, [])
            return tuple(str(value) for value in raw) if isinstance(raw, list) else ()

        failed_raw = values.get("dates_failed", {})
        failed = (
            tuple(sorted((str(key), str(value)) for key, value in failed_raw.items()))
            if isinstance(failed_raw, dict)
            else ()
        )
        return cls(
            started_at=str(values.get("started_at") or ""),
            finished_at=(str(values["finished_at"]) if values.get("finished_at") else None),
            target_end_date=str(values.get("target_end_date") or ""),
            missing_dates_discovered=strings("missing_dates_discovered"),
            dates_attempted=strings("dates_attempted"),
            dates_downloaded=strings("dates_downloaded"),
            dates_skipped=strings("dates_skipped"),
            dates_failed=failed,
            native_update_status=str(values.get("native_update_status") or "not_started"),
            native_rows_added=int(values.get("native_rows_added", 0)),
            native_rows_replaced=int(values.get("native_rows_replaced", 0)),
            daily_parquets_created=int(values.get("daily_parquets_created", 0)),
            consolidated_parquet_latest_date=(
                str(values["consolidated_parquet_latest_date"])
                if values.get("consolidated_parquet_latest_date")
                else None
            ),
            ai_rebuild_status=str(values.get("ai_rebuild_status") or "not_requested"),
            source_evidence_inconsistencies=strings("source_evidence_inconsistencies"),
            stale_run_recovered=bool(values.get("stale_run_recovered", False)),
            failure_reason=(str(values["failure_reason"]) if values.get("failure_reason") else None),
        )


@dataclass(frozen=True)
class SourceDateDisposition:
    """Auditable decision controlling automatic retries for one source date."""

    trading_date: date
    classification: str
    reason: str
    evidence: str
    retry_automatically: bool
    recorded_at: str

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["trading_date"] = self.trading_date.isoformat()
        return values

    @classmethod
    def from_dict(cls, values: object) -> "SourceDateDisposition":
        if not isinstance(values, dict):
            raise ValueError("source-date disposition must be an object")
        return cls(
            trading_date=date.fromisoformat(str(values["trading_date"])),
            classification=str(values["classification"]),
            reason=str(values["reason"]),
            evidence=str(values["evidence"]),
            retry_automatically=bool(values["retry_automatically"]),
            recorded_at=str(values["recorded_at"]),
        )


@dataclass(frozen=True)
class AutomationConfig:
    """Persisted, non-secret settings and execution status."""

    enabled: bool = False
    timezone: str = KARACHI_TIMEZONE
    scheduled_time: str = SCHEDULED_TIME
    bootstrap_start_date: date | None = None
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_status: str = "never_run"
    last_message: str = "Automation has not run yet"
    rebuild_ai_datasets: bool = False
    deferred_empty_dates: tuple[date, ...] = ()
    source_date_dispositions: tuple[SourceDateDisposition, ...] = ()
    last_run: AutomationRunAudit | None = None

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["bootstrap_start_date"] = (
            self.bootstrap_start_date.isoformat()
            if self.bootstrap_start_date is not None
            else None
        )
        values["deferred_empty_dates"] = [
            value.isoformat() for value in self.deferred_empty_dates
        ]
        values["source_date_dispositions"] = [
            value.to_dict() for value in self.source_date_dispositions
        ]
        values["last_run"] = self.last_run.to_dict() if self.last_run else None
        return values


@dataclass(frozen=True)
class AutomationRunResult:
    """Outcome returned by manual and scheduled automation runs."""

    status: RunStatus
    message: str
    exit_code: int
    update_result: IncrementalUpdateResult | None = None
    master_result: MasterBuildResult | None = None
    listings_result: ListingsRefreshResult | None = None
    registry_result: RegistryBuildResult | None = None
    index_result: IndexRefreshResult | None = None
    market_update_succeeded: bool = False
    master_rebuild_succeeded: bool = False
    listing_refresh_succeeded: bool = False
    registry_rebuild_succeeded: bool = False
    cached_listings_used: bool = False
    cached_indices_used: bool = False
    index_refresh_succeeded: bool = False
    ai_rebuild_succeeded: bool = False
    native_result: NativeMarketBuildResult | None = None
    native_update_succeeded: bool = False
    audit: AutomationRunAudit | None = None


@dataclass(frozen=True)
class NativeSourceReconciliationResult:
    """Native artifacts and metadata refreshed from validated source CSVs."""

    native: NativeMarketBuildResult
    listings: ListingsRefreshResult
    registry: RegistryBuildResult


class UpdateAlreadyRunning(RuntimeError):
    """Raised when a non-stale automation lock already exists."""


def karachi_now(now: datetime | None = None) -> datetime:
    """Return an aware datetime converted to the official project timezone."""
    timezone = ZoneInfo(KARACHI_TIMEZONE)
    if now is None:
        return datetime.now(timezone)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone)


def karachi_today(now: datetime | None = None) -> date:
    """Return the current date in Asia/Karachi."""
    return karachi_now(now).date()


def _config_from_dict(values: object) -> AutomationConfig:
    if not isinstance(values, dict):
        raise ValueError("automation configuration must be a JSON object")

    timezone = values.get("timezone", KARACHI_TIMEZONE)
    scheduled_time = values.get("scheduled_time", SCHEDULED_TIME)
    if timezone != KARACHI_TIMEZONE:
        raise ValueError(f"timezone must be {KARACHI_TIMEZONE}")
    if scheduled_time != SCHEDULED_TIME:
        raise ValueError(f"scheduled_time must be {SCHEDULED_TIME}")

    bootstrap_value = values.get("bootstrap_start_date")
    bootstrap_start_date = (
        date.fromisoformat(bootstrap_value)
        if isinstance(bootstrap_value, str) and bootstrap_value
        else None
    )
    enabled = values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    rebuild_ai = values.get("rebuild_ai_datasets", False)
    if not isinstance(rebuild_ai, bool):
        raise ValueError("rebuild_ai_datasets must be a boolean")
    deferred_raw = values.get("deferred_empty_dates", [])
    if not isinstance(deferred_raw, list):
        raise ValueError("deferred_empty_dates must be a list")
    deferred_dates = tuple(sorted(date.fromisoformat(str(value)) for value in deferred_raw))
    dispositions_raw = values.get("source_date_dispositions", [])
    if not isinstance(dispositions_raw, list):
        raise ValueError("source_date_dispositions must be a list")
    dispositions = tuple(
        sorted(
            (SourceDateDisposition.from_dict(value) for value in dispositions_raw),
            key=lambda value: value.trading_date,
        )
    )

    def optional_string(field: str) -> str | None:
        value = values.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        return value

    return AutomationConfig(
        enabled=enabled,
        timezone=KARACHI_TIMEZONE,
        scheduled_time=SCHEDULED_TIME,
        bootstrap_start_date=bootstrap_start_date,
        last_attempt_at=optional_string("last_attempt_at"),
        last_success_at=optional_string("last_success_at"),
        last_status=optional_string("last_status") or "never_run",
        last_message=optional_string("last_message") or "Automation has not run yet",
        rebuild_ai_datasets=rebuild_ai,
        deferred_empty_dates=deferred_dates,
        source_date_dispositions=dispositions,
        last_run=AutomationRunAudit.from_dict(values.get("last_run")),
    )


def load_automation_config(
    path: Path = AUTOMATION_CONFIG_PATH,
) -> AutomationConfig:
    """Load persisted automation state, falling back safely when malformed."""
    config_path = Path(path)
    if not config_path.exists():
        return AutomationConfig()
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"))
        return _config_from_dict(values)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "Using safe automation defaults; could not load %s: %s",
            path,
            exc,
        )
        return AutomationConfig(
            last_status="configuration_error",
            last_message=f"Could not load automation configuration: {exc}",
        )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_automation_config(
    config: AutomationConfig,
    path: Path = AUTOMATION_CONFIG_PATH,
) -> None:
    """Persist automation state atomically as formatted JSON."""
    if config.timezone != KARACHI_TIMEZONE or config.scheduled_time != SCHEDULED_TIME:
        raise ValueError(
            "automation timezone or schedule does not match project policy"
        )
    content = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    _atomic_write_text(Path(path), content)


class UpdateLock:
    """Exclusive lock-file context manager with stale-lock recovery."""

    def __init__(
        self,
        path: Path = AUTOMATION_LOCK_PATH,
        *,
        stale_after: timedelta = DEFAULT_LOCK_STALE_AFTER,
    ) -> None:
        self.path = Path(path)
        self.stale_after = stale_after
        self.acquired = False
        self._payload: bytes | None = None
        self.recovered_stale = False

    def _owner_pid(self) -> int | None:
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            pid = values.get("pid") if isinstance(values, dict) else None
            return int(pid) if pid is not None else None
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None

    def _owner_is_dead(self) -> bool:
        pid = self._owner_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _is_stale(self, now: datetime) -> bool:
        try:
            modified = datetime.fromtimestamp(
                self.path.stat().st_mtime,
                tz=now.tzinfo,
            )
        except FileNotFoundError:
            return False
        return self._owner_is_dead() or now - modified > self.stale_after

    def acquire(self, now: datetime | None = None) -> None:
        current_time = karachi_now(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if self._is_stale(current_time):
                    LOGGER.warning("Removing stale automation lock: %s", self.path)
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    self.recovered_stale = True
                    continue
                raise UpdateAlreadyRunning(
                    f"Another automation run holds lock {self.path}"
                )

            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "created_at": current_time.isoformat(timespec="seconds"),
                }
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            self.acquired = True
            self._payload = payload
            return

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self.path.read_bytes() == self._payload:
                self.path.unlink()
        except (FileNotFoundError, OSError):
            pass
        self.acquired = False
        self._payload = None

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def recover_stale_automation_state(
    *,
    config_path: Path = AUTOMATION_CONFIG_PATH,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    now: datetime | None = None,
) -> AutomationConfig:
    """Finalize an abandoned running state when no live lock owner remains."""

    config = load_automation_config(config_path)
    if config.last_status != "running":
        return config
    current_time = karachi_now(now)
    lock = UpdateLock(lock_path)
    lock_exists = Path(lock_path).exists()
    stale = not lock_exists or lock._is_stale(current_time)
    if not stale:
        return config
    if lock_exists:
        try:
            Path(lock_path).unlink()
        except FileNotFoundError:
            pass
    message = STALE_RECOVERY_MESSAGE
    prior_audit = config.last_run
    recovered_audit = (
        replace(
            prior_audit,
            finished_at=current_time.isoformat(timespec="seconds"),
            stale_run_recovered=True,
            failure_reason=message,
        )
        if prior_audit is not None
        else None
    )
    recovered = replace(
        config,
        last_status="failed",
        last_message=message,
        last_run=recovered_audit,
    )
    save_automation_config(recovered, config_path)
    return recovered


def configure_auto_update_logging(path: Path = AUTO_UPDATE_LOG_PATH) -> None:
    """Attach one file handler for standalone automation diagnostics."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = log_path.resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(
            handler.baseFilename
        ).resolve() == resolved:
            return

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _emit_progress(
    callback: ProgressCallback | None, stage: str, message: str
) -> None:
    LOGGER.info("Automation stage %s: %s", stage, message)
    if callback is not None:
        callback(AutomationProgress(stage=stage, message=message))


def _known_non_request_dates(
    *,
    backfill_state_path: Path,
    deferred_dates: Sequence[date],
    source_dispositions: Sequence[SourceDateDisposition] = (),
) -> tuple[date, ...]:
    excluded = set(deferred_dates)
    excluded.update(
        value.trading_date
        for value in source_dispositions
        if not value.retry_automatically
    )
    state = load_backfill_state(backfill_state_path)
    if state is not None:
        excluded.update(state.non_trading_dates)
        excluded.update(value for value, _ in state.temporary_skips)
        excluded.update(value for value, _ in state.failed_dates)
    return tuple(sorted(excluded))


def _reconcile_native_sources_unlocked(
    source_csvs: Sequence[Path],
    *,
    reference_date: date,
    refreshed_at: datetime,
    listing_refresher: ListingRefresher,
    native_updater: NativeUpdater,
    registry_builder: RegistryBuilder,
    progress_callback: ProgressCallback | None,
) -> NativeSourceReconciliationResult:
    """Apply validated source CSVs to every native artifact without fetching."""

    paths = tuple(Path(path) for path in source_csvs)
    if not paths:
        raise ValueError("At least one validated source CSV is required")
    _emit_progress(
        progress_callback,
        "refreshing_listings",
        "Refreshing authoritative listing and sector metadata",
    )
    listings = listing_refresher(refreshed_at=refreshed_at)
    _emit_progress(
        progress_callback,
        "updating_native_market",
        f"Updating native artifacts from {len(paths):,} source CSV file(s)",
    )
    native = native_updater(
        paths,
        listings_path=listings.current_snapshot_path,
        progress_callback=(
            lambda stage, message: _emit_progress(
                progress_callback, stage, message
            )
        ),
    )
    registry = registry_builder(
        listing_data=listings.data,
        reference_date=reference_date,
        registry_updated_at=refreshed_at,
        cached_listings_used=listings.used_cache,
    )
    return NativeSourceReconciliationResult(native, listings, registry)


def reconcile_native_source_csvs(
    source_csvs: Sequence[Path],
    *,
    reference_date: date,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    listing_refresher: ListingRefresher = refresh_official_listings,
    native_updater: NativeUpdater = run_native_incremental_update,
    registry_builder: RegistryBuilder = build_company_registry,
    progress_callback: ProgressCallback | None = None,
    now: datetime | None = None,
) -> NativeSourceReconciliationResult:
    """Public lock-protected native ingestion for backfill/local recovery."""

    refreshed_at = karachi_now(now)
    lock = UpdateLock(Path(lock_path))
    lock.acquire(refreshed_at)
    try:
        return _reconcile_native_sources_unlocked(
            source_csvs,
            reference_date=reference_date,
            refreshed_at=refreshed_at,
            listing_refresher=listing_refresher,
            native_updater=native_updater,
            registry_builder=registry_builder,
            progress_callback=progress_callback,
        )
    finally:
        lock.release()


def rebuild_canonical_market_artifacts(
    *,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    listings_path: Path = CURRENT_LISTINGS_PATH,
    native_rebuilder: NativeRebuilder = rebuild_generated_artifacts,
    progress_callback: ProgressCallback | None = None,
    now: datetime | None = None,
    history_path: Path | None = None,
) -> NativeMarketBuildResult:
    """Explicitly regenerate the canonical native artifacts without fetching."""

    current_time = karachi_now(now)
    lock = UpdateLock(Path(lock_path))
    lock.acquire(current_time)
    try:
        _emit_progress(
            progress_callback,
            "validating_native_source_state",
            "Validating native Parquet and source-manifest provenance",
        )
        result = native_rebuilder(
            listings_path=Path(listings_path),
            progress_callback=(
                lambda stage, message: _emit_progress(
                    progress_callback, stage, message
                )
            ),
        )
        _append_history_safely(
            new_operation(
                "FULL_REBUILD",
                executed_dates=result.source_dates,
                artifact_status={
                    "canonical_master_csv": "UPDATED",
                    "consolidated_parquet": "UPDATED",
                    "daily_partitions_affected": result.daily_parquets_written,
                    "symbol_artifacts_affected": result.symbol_csvs_written,
                    "logical_parity": "PASS",
                },
                master_latest_date=result.latest_date,
                source_set_hash=result.source_set_hash,
                content_hash=result.content_hash,
            ),
            _resolved_history_path(history_path, sibling_of=Path(lock_path)),
        )
        return result
    finally:
        lock.release()


def run_update_orchestration(
    config: AutomationConfig,
    *,
    end_date: date,
    config_path: Path,
    lock_path: Path,
    updater: Updater,
    index_refresher: IndexRefresher,
    master_builder: MasterBuilder,
    listing_refresher: ListingRefresher,
    registry_builder: RegistryBuilder,
    now: datetime | None,
    native_updater: NativeUpdater = run_native_incremental_update,
    evidence_discoverer: EvidenceDiscoverer = discover_source_evidence,
    symbol_ai_builder: Callable[..., object] = build_symbol_datasets,
    master_ai_builder: Callable[..., object] = build_master_ai_dataset,
    progress_callback: ProgressCallback | None = None,
    raw_csv_dir: Path = RAW_CSV_DIR,
    native_state_path: Path = NATIVE_MARKET_PIPELINE_STATE_PATH,
    native_parquet_path: Path = LOCAL_PSX_MARKET_PARQUET_PATH,
    backfill_state_path: Path = BACKFILL_STATE_PATH,
    history_path: Path | None = None,
) -> AutomationRunResult:
    """Run the one canonical manual/scheduled/CLI market update workflow."""

    attempt_time = karachi_now(now)
    lock = UpdateLock(lock_path)
    try:
        lock.acquire(attempt_time)
    except UpdateAlreadyRunning as exc:
        return AutomationRunResult(
            status="already_running",
            message=str(exc),
            exit_code=0,
        )

    initial_audit = AutomationRunAudit(
        started_at=attempt_time.isoformat(timespec="seconds"),
        finished_at=None,
        target_end_date=end_date.isoformat(),
    )
    attempted_config = replace(
        config,
        last_attempt_at=attempt_time.isoformat(timespec="seconds"),
        last_status="running",
        last_message=f"Updating through {end_date.isoformat()}",
        last_run=initial_audit,
    )
    update_result: IncrementalUpdateResult | None = None
    master_result: MasterBuildResult | None = None
    listings_result: ListingsRefreshResult | None = None
    registry_result: RegistryBuildResult | None = None
    index_result: IndexRefreshResult | None = None
    native_result: NativeMarketBuildResult | None = None
    inventory: SourceEvidenceInventory | None = None
    ai_status = "not_requested"
    stale_recovered = (
        lock.recovered_stale
        or config.last_status == "running"
        or (
            config.last_status == "failed"
            and config.last_message == STALE_RECOVERY_MESSAGE
            and config.last_run is not None
            and config.last_run.stale_run_recovered
        )
    )
    try:
        save_automation_config(attempted_config, config_path)
        _emit_progress(
            progress_callback,
            "checking_stored_dates",
            "Checking local CSV evidence and the trusted native source manifest",
        )
        inventory = evidence_discoverer(
            csv_dir=raw_csv_dir,
            native_state_path=native_state_path,
            parquet_path=native_parquet_path,
        )
        excluded_dates = _known_non_request_dates(
            backfill_state_path=backfill_state_path,
            deferred_dates=config.deferred_empty_dates,
            source_dispositions=config.source_date_dispositions,
        )
        update_result = updater(
            end_date=end_date,
            bootstrap_start_date=config.bootstrap_start_date,
            csv_dir=raw_csv_dir,
            available_source_dates=inventory.accepted_source_dates,
            local_source_dates=inventory.local_csv_dates,
            external_source_dates=inventory.external_manifest_dates,
            excluded_dates=excluded_dates,
            source_evidence_inconsistencies=inventory.inconsistencies,
            progress_callback=(
                lambda stage, message: _emit_progress(
                    progress_callback, stage, message
                )
            ),
        )
        manifest_dates = set(inventory.native_manifest_dates)
        pending_local_dates = (
            set(inventory.local_csv_dates)
            .intersection(inventory.accepted_source_dates)
            .difference(manifest_dates)
            .difference(excluded_dates)
        )
        pending_paths = [
            Path(raw_csv_dir) / f"market_{value.isoformat()}.csv"
            for value in sorted(pending_local_dates)
        ]
        native_inputs = tuple(
            dict.fromkeys([*pending_paths, *update_result.output_csv_paths])
        )

        if not update_result.missing_dates and not native_inputs:
            status: RunStatus = "no_update_needed"
            message = "No update needed; all request dates have accepted source evidence or a recorded non-request disposition"
        else:
            if native_inputs:
                reconciled = _reconcile_native_sources_unlocked(
                    native_inputs,
                    reference_date=end_date,
                    refreshed_at=attempt_time,
                    listing_refresher=listing_refresher,
                    native_updater=native_updater,
                    registry_builder=registry_builder,
                    progress_callback=progress_callback,
                )
                listings_result = reconciled.listings
                native_result = reconciled.native
                registry_result = reconciled.registry

                if config.rebuild_ai_datasets:
                    ai_status = "running"
                    _emit_progress(
                        progress_callback,
                        "rebuilding_ai_datasets",
                        "Rebuilding legacy feature/AI products after native success",
                    )
                    index_result = index_refresher()
                    if not index_result.has_usable_data:
                        raise RuntimeError(
                            "Official index refresh failed and no cached index data is available"
                        )
                    master_result = master_builder()
                    symbol_ai_builder()
                    master_ai_builder()
                    ai_status = "success"

            if update_result.failed_dates and (
                update_result.successful_dates or native_result is not None
            ):
                status = "partial_success"
            elif update_result.failed_dates:
                status = "failed"
            elif update_result.skipped_dates:
                status = "partial_success"
            else:
                status = "success"
            message = (
                f"Update {status.replace('_', ' ')}: "
                f"{len(update_result.successful_dates)} downloaded, "
                f"{len(update_result.skipped_dates)} skipped, "
                f"{len(update_result.failed_dates)} failed; "
                f"native rows added={native_result.rows_added if native_result else 0}"
            )

        finish_time = karachi_now(now)
        final_audit = replace(
            initial_audit,
            finished_at=finish_time.isoformat(timespec="seconds"),
            missing_dates_discovered=tuple(
                value.isoformat() for value in update_result.missing_dates
            ),
            dates_attempted=tuple(
                value.isoformat() for value in update_result.missing_dates
            ),
            dates_downloaded=tuple(
                value.isoformat() for value in update_result.successful_dates
            ),
            dates_skipped=tuple(
                value.isoformat() for value in update_result.skipped_dates
            ),
            dates_failed=tuple(
                (value.isoformat(), reason)
                for value, reason in update_result.failed_dates
            ),
            native_update_status=(
                "success"
                if native_result is not None
                else ("not_needed" if not native_inputs else "not_started")
            ),
            native_rows_added=native_result.rows_added if native_result else 0,
            native_rows_replaced=native_result.rows_replaced if native_result else 0,
            daily_parquets_created=(
                native_result.daily_parquets_written if native_result else 0
            ),
            consolidated_parquet_latest_date=(
                native_result.latest_date if native_result else None
            ),
            ai_rebuild_status=ai_status,
            source_evidence_inconsistencies=inventory.inconsistencies,
            stale_run_recovered=stale_recovered,
        )
        deferred = tuple(
            sorted(set(config.deferred_empty_dates).union(update_result.skipped_dates))
        )
        final_config = replace(
            attempted_config,
            last_success_at=(
                finish_time.isoformat(timespec="seconds")
                if status in {"success", "no_update_needed"}
                else config.last_success_at
            ),
            last_status=status,
            last_message=message,
            deferred_empty_dates=deferred,
            last_run=final_audit,
        )
        save_automation_config(final_config, config_path)
        _append_history_safely(
            new_operation(
                "AUTOMATION_UPDATE",
                requested_dates=final_audit.missing_dates_discovered,
                executed_dates=final_audit.dates_attempted,
                skipped_dates=final_audit.dates_skipped,
                per_date_results=(
                    *(
                        MaintenanceDateResult(value, "downloaded")
                        for value in final_audit.dates_downloaded
                    ),
                    *(
                        MaintenanceDateResult(value, "skipped")
                        for value in final_audit.dates_skipped
                    ),
                    *(
                        MaintenanceDateResult(value, "failed", message=reason)
                        for value, reason in final_audit.dates_failed
                    ),
                ),
                artifact_status={
                    "canonical_master_csv": (
                        "UPDATED" if native_result is not None else "CURRENT"
                    ),
                    "consolidated_parquet": (
                        "UPDATED" if native_result is not None else "CURRENT"
                    ),
                    "daily_partitions_affected": final_audit.daily_parquets_created,
                    "symbol_artifacts_affected": (
                        native_result.symbol_csvs_written if native_result else 0
                    ),
                    "logical_parity": "PASS",
                },
                master_latest_date=final_audit.consolidated_parquet_latest_date,
                source_set_hash=(native_result.source_set_hash if native_result else None),
                content_hash=(native_result.content_hash if native_result else None),
            ),
            _resolved_history_path(history_path, sibling_of=Path(config_path)),
        )
        _emit_progress(progress_callback, "complete", message)
        return AutomationRunResult(
            status=status,
            message=message,
            exit_code=0 if status in {"success", "partial_success", "no_update_needed"} else 1,
            update_result=update_result,
            master_result=master_result,
            listings_result=listings_result,
            registry_result=registry_result,
            index_result=index_result,
            market_update_succeeded=not bool(update_result.failed_dates),
            master_rebuild_succeeded=master_result is not None,
            listing_refresh_succeeded=listings_result is not None,
            registry_rebuild_succeeded=registry_result is not None,
            cached_listings_used=(listings_result.used_cache if listings_result else False),
            cached_indices_used=(index_result.cached_data_used if index_result else False),
            index_refresh_succeeded=(bool(index_result) and not bool(index_result.failed_indices)),
            ai_rebuild_succeeded=ai_status == "success",
            native_result=native_result,
            native_update_succeeded=native_result is not None or not native_inputs,
            audit=final_audit,
        )
    except Exception as exc:
        message = f"Automation failed: {type(exc).__name__}: {exc}"
        finish_time = karachi_now(now)
        failed_audit = replace(
            initial_audit,
            finished_at=finish_time.isoformat(timespec="seconds"),
            missing_dates_discovered=(
                tuple(value.isoformat() for value in update_result.missing_dates)
                if update_result is not None
                else ()
            ),
            dates_attempted=(
                tuple(value.isoformat() for value in update_result.missing_dates)
                if update_result is not None
                else ()
            ),
            dates_downloaded=(
                tuple(value.isoformat() for value in update_result.successful_dates)
                if update_result is not None
                else ()
            ),
            dates_skipped=(
                tuple(value.isoformat() for value in update_result.skipped_dates)
                if update_result is not None
                else ()
            ),
            dates_failed=(
                tuple((value.isoformat(), reason) for value, reason in update_result.failed_dates)
                if update_result is not None
                else ()
            ),
            native_update_status=("success" if native_result is not None else "failed"),
            native_rows_added=native_result.rows_added if native_result else 0,
            daily_parquets_created=(native_result.daily_parquets_written if native_result else 0),
            consolidated_parquet_latest_date=(native_result.latest_date if native_result else None),
            ai_rebuild_status=("failed" if ai_status == "running" else ai_status),
            source_evidence_inconsistencies=(inventory.inconsistencies if inventory else ()),
            stale_run_recovered=stale_recovered,
            failure_reason=message,
        )
        failed_config = replace(
            attempted_config,
            last_status="failed",
            last_message=message,
            last_run=failed_audit,
        )
        try:
            save_automation_config(failed_config, config_path)
        except OSError:
            LOGGER.exception("Could not persist automation failure status")
        LOGGER.exception(message)
        _append_history_safely(
            new_operation(
                "AUTOMATION_UPDATE",
                requested_dates=failed_audit.missing_dates_discovered,
                executed_dates=failed_audit.dates_attempted,
                skipped_dates=failed_audit.dates_skipped,
                per_date_results=tuple(
                    MaintenanceDateResult(value, "failed", message=reason)
                    for value, reason in failed_audit.dates_failed
                ),
                artifact_status={
                    "canonical_master_csv": "FAILED",
                    "consolidated_parquet": (
                        "UPDATED" if native_result is not None else "FAILED"
                    ),
                    "daily_partitions_affected": failed_audit.daily_parquets_created,
                    "symbol_artifacts_affected": (
                        native_result.symbol_csvs_written if native_result else 0
                    ),
                    "logical_parity": "FAIL",
                },
                errors=(message,),
                master_latest_date=failed_audit.consolidated_parquet_latest_date,
                source_set_hash=(native_result.source_set_hash if native_result else None),
                content_hash=(native_result.content_hash if native_result else None),
            ),
            _resolved_history_path(history_path, sibling_of=Path(config_path)),
        )
        return AutomationRunResult(
            status="failed",
            message=message,
            exit_code=1,
            update_result=update_result,
            master_result=master_result,
            listings_result=listings_result,
            registry_result=registry_result,
            index_result=index_result,
            market_update_succeeded=(
                update_result is not None and not update_result.failed_dates
            ),
            master_rebuild_succeeded=master_result is not None,
            listing_refresh_succeeded=listings_result is not None,
            registry_rebuild_succeeded=registry_result is not None,
            cached_listings_used=(
                listings_result.used_cache if listings_result is not None else False
            ),
            cached_indices_used=(index_result.cached_data_used if index_result else False),
            index_refresh_succeeded=(bool(index_result) and not bool(index_result.failed_indices)),
            native_result=native_result,
            native_update_succeeded=native_result is not None,
            audit=failed_audit,
        )
    finally:
        lock.release()


def run_scheduled_update(
    *,
    config_path: Path = AUTOMATION_CONFIG_PATH,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    updater: Updater = run_incremental_update,
    index_refresher: IndexRefresher = refresh_indices,
    master_builder: MasterBuilder = build_master_dataset,
    listing_refresher: ListingRefresher = refresh_official_listings,
    registry_builder: RegistryBuilder = build_company_registry,
    native_updater: NativeUpdater = run_native_incremental_update,
    evidence_discoverer: EvidenceDiscoverer = discover_source_evidence,
    symbol_ai_builder: Callable[..., object] = build_symbol_datasets,
    master_ai_builder: Callable[..., object] = build_master_ai_dataset,
    progress_callback: ProgressCallback | None = None,
    raw_csv_dir: Path = RAW_CSV_DIR,
    native_state_path: Path = NATIVE_MARKET_PIPELINE_STATE_PATH,
    native_parquet_path: Path = LOCAL_PSX_MARKET_PARQUET_PATH,
    backfill_state_path: Path = BACKFILL_STATE_PATH,
    history_path: Path | None = None,
    now: datetime | None = None,
) -> AutomationRunResult:
    """Run the configured scheduled update, or exit cleanly when disabled."""
    config = recover_stale_automation_state(
        config_path=Path(config_path), lock_path=Path(lock_path), now=now
    )
    if not config.enabled:
        return AutomationRunResult(
            status="disabled",
            message="Automatic daily fetching is disabled",
            exit_code=0,
        )
    return run_update_orchestration(
        config,
        end_date=karachi_today(now),
        config_path=Path(config_path),
        lock_path=Path(lock_path),
        updater=updater,
        index_refresher=index_refresher,
        master_builder=master_builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
        native_updater=native_updater,
        evidence_discoverer=evidence_discoverer,
        symbol_ai_builder=symbol_ai_builder,
        master_ai_builder=master_ai_builder,
        progress_callback=progress_callback,
        raw_csv_dir=raw_csv_dir,
        native_state_path=native_state_path,
        native_parquet_path=native_parquet_path,
        backfill_state_path=backfill_state_path,
        history_path=history_path,
        now=now,
    )


def run_manual_update(
    *,
    end_date: date | None = None,
    bootstrap_start_date: date | None = None,
    config_path: Path = AUTOMATION_CONFIG_PATH,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    updater: Updater = run_incremental_update,
    index_refresher: IndexRefresher = refresh_indices,
    master_builder: MasterBuilder = build_master_dataset,
    listing_refresher: ListingRefresher = refresh_official_listings,
    registry_builder: RegistryBuilder = build_company_registry,
    native_updater: NativeUpdater = run_native_incremental_update,
    evidence_discoverer: EvidenceDiscoverer = discover_source_evidence,
    symbol_ai_builder: Callable[..., object] = build_symbol_datasets,
    master_ai_builder: Callable[..., object] = build_master_ai_dataset,
    progress_callback: ProgressCallback | None = None,
    raw_csv_dir: Path = RAW_CSV_DIR,
    native_state_path: Path = NATIVE_MARKET_PIPELINE_STATE_PATH,
    native_parquet_path: Path = LOCAL_PSX_MARKET_PARQUET_PATH,
    backfill_state_path: Path = BACKFILL_STATE_PATH,
    history_path: Path | None = None,
    now: datetime | None = None,
) -> AutomationRunResult:
    """Run one explicit update cycle regardless of the enabled setting."""
    config = recover_stale_automation_state(
        config_path=Path(config_path), lock_path=Path(lock_path), now=now
    )
    if bootstrap_start_date is not None:
        config = replace(config, bootstrap_start_date=bootstrap_start_date)
    requested_end_date = end_date if end_date is not None else karachi_today(now)
    return run_update_orchestration(
        config,
        end_date=requested_end_date,
        config_path=Path(config_path),
        lock_path=Path(lock_path),
        updater=updater,
        index_refresher=index_refresher,
        master_builder=master_builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
        native_updater=native_updater,
        evidence_discoverer=evidence_discoverer,
        symbol_ai_builder=symbol_ai_builder,
        master_ai_builder=master_ai_builder,
        progress_callback=progress_callback,
        raw_csv_dir=raw_csv_dir,
        native_state_path=native_state_path,
        native_parquet_path=native_parquet_path,
        backfill_state_path=backfill_state_path,
        history_path=history_path,
        now=now,
    )
