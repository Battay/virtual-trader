"""Automation configuration, locking, and standalone update orchestration."""

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from .config import (
    AUTOMATION_CONFIG_PATH,
    AUTOMATION_LOCK_PATH,
    AUTO_UPDATE_LOG_PATH,
    PROJECT_TIMEZONE,
)
from .company_registry import RegistryBuildResult, build_company_registry
from .csv_store import MasterBuildResult, build_master_dataset
from .official_listings import ListingsRefreshResult, refresh_official_listings
from .updater import IncrementalUpdateResult, run_incremental_update


LOGGER = logging.getLogger(__name__)
KARACHI_TIMEZONE = PROJECT_TIMEZONE
SCHEDULED_TIME = "17:15"
DEFAULT_LOCK_STALE_AFTER = timedelta(hours=6)
RunStatus = Literal["disabled", "success", "failed", "already_running"]
Updater = Callable[..., IncrementalUpdateResult]
MasterBuilder = Callable[..., MasterBuildResult]
ListingRefresher = Callable[..., ListingsRefreshResult]
RegistryBuilder = Callable[..., RegistryBuildResult]


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

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["bootstrap_start_date"] = (
            self.bootstrap_start_date.isoformat()
            if self.bootstrap_start_date is not None
            else None
        )
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
    market_update_succeeded: bool = False
    master_rebuild_succeeded: bool = False
    listing_refresh_succeeded: bool = False
    registry_rebuild_succeeded: bool = False
    cached_listings_used: bool = False


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

    def _is_stale(self, now: datetime) -> bool:
        try:
            modified = datetime.fromtimestamp(
                self.path.stat().st_mtime,
                tz=now.tzinfo,
            )
        except FileNotFoundError:
            return False
        return now - modified > self.stale_after

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


def _run_enabled_update(
    config: AutomationConfig,
    *,
    end_date: date,
    config_path: Path,
    lock_path: Path,
    updater: Updater,
    master_builder: MasterBuilder,
    listing_refresher: ListingRefresher,
    registry_builder: RegistryBuilder,
    now: datetime | None,
) -> AutomationRunResult:
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

    attempted_config = replace(
        config,
        last_attempt_at=attempt_time.isoformat(timespec="seconds"),
        last_status="running",
        last_message=f"Updating through {end_date.isoformat()}",
    )
    update_result: IncrementalUpdateResult | None = None
    master_result: MasterBuildResult | None = None
    listings_result: ListingsRefreshResult | None = None
    registry_result: RegistryBuildResult | None = None
    try:
        save_automation_config(attempted_config, config_path)
        update_result = updater(
            end_date=end_date,
            bootstrap_start_date=config.bootstrap_start_date,
        )
        master_result = master_builder()

        if update_result.failed_dates:
            failed_text = ", ".join(
                f"{failed_date.isoformat()}: {reason}"
                for failed_date, reason in update_result.failed_dates
            )
            message = f"Update completed with failed dates: {failed_text}"
            final_config = replace(
                attempted_config,
                last_status="failed",
                last_message=message,
            )
            save_automation_config(final_config, config_path)
            LOGGER.error(message)
            return AutomationRunResult(
                status="failed",
                message=message,
                exit_code=1,
                update_result=update_result,
                master_result=master_result,
                market_update_succeeded=False,
                master_rebuild_succeeded=True,
            )

        listings_result = listing_refresher(refreshed_at=attempt_time)
        registry_result = registry_builder(
            listing_data=listings_result.data,
            reference_date=end_date,
            registry_updated_at=attempt_time,
            cached_listings_used=listings_result.used_cache,
        )
        listing_mode = "cached listings" if listings_result.used_cache else "live listings"
        message = (
            f"Update succeeded: {len(update_result.successful_dates)} successful, "
            f"{len(update_result.skipped_dates)} skipped; "
            f"master has {master_result.total_rows} rows; {listing_mode}; "
            f"registry has {registry_result.total_registry_symbols} symbols"
        )
        final_config = replace(
            attempted_config,
            last_success_at=attempt_time.isoformat(timespec="seconds"),
            last_status="success",
            last_message=message,
        )
        save_automation_config(final_config, config_path)
        LOGGER.info(message)
        return AutomationRunResult(
            status="success",
            message=message,
            exit_code=0,
            update_result=update_result,
            master_result=master_result,
            listings_result=listings_result,
            registry_result=registry_result,
            market_update_succeeded=True,
            master_rebuild_succeeded=True,
            listing_refresh_succeeded=True,
            registry_rebuild_succeeded=True,
            cached_listings_used=listings_result.used_cache,
        )
    except Exception as exc:
        message = f"Automation failed: {type(exc).__name__}: {exc}"
        failed_config = replace(
            attempted_config,
            last_status="failed",
            last_message=message,
        )
        try:
            save_automation_config(failed_config, config_path)
        except OSError:
            LOGGER.exception("Could not persist automation failure status")
        LOGGER.exception(message)
        return AutomationRunResult(
            status="failed",
            message=message,
            exit_code=1,
            update_result=update_result,
            master_result=master_result,
            listings_result=listings_result,
            registry_result=registry_result,
            market_update_succeeded=(
                update_result is not None and not update_result.failed_dates
            ),
            master_rebuild_succeeded=master_result is not None,
            listing_refresh_succeeded=listings_result is not None,
            registry_rebuild_succeeded=registry_result is not None,
            cached_listings_used=(
                listings_result.used_cache if listings_result is not None else False
            ),
        )
    finally:
        lock.release()


def run_scheduled_update(
    *,
    config_path: Path = AUTOMATION_CONFIG_PATH,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    updater: Updater = run_incremental_update,
    master_builder: MasterBuilder = build_master_dataset,
    listing_refresher: ListingRefresher = refresh_official_listings,
    registry_builder: RegistryBuilder = build_company_registry,
    now: datetime | None = None,
) -> AutomationRunResult:
    """Run the configured scheduled update, or exit cleanly when disabled."""
    config = load_automation_config(config_path)
    if not config.enabled:
        return AutomationRunResult(
            status="disabled",
            message="Automatic daily fetching is disabled",
            exit_code=0,
        )
    return _run_enabled_update(
        config,
        end_date=karachi_today(now),
        config_path=Path(config_path),
        lock_path=Path(lock_path),
        updater=updater,
        master_builder=master_builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
        now=now,
    )


def run_manual_update(
    *,
    end_date: date | None = None,
    bootstrap_start_date: date | None = None,
    config_path: Path = AUTOMATION_CONFIG_PATH,
    lock_path: Path = AUTOMATION_LOCK_PATH,
    updater: Updater = run_incremental_update,
    master_builder: MasterBuilder = build_master_dataset,
    listing_refresher: ListingRefresher = refresh_official_listings,
    registry_builder: RegistryBuilder = build_company_registry,
    now: datetime | None = None,
) -> AutomationRunResult:
    """Run one explicit update cycle regardless of the enabled setting."""
    config = load_automation_config(config_path)
    if bootstrap_start_date is not None:
        config = replace(config, bootstrap_start_date=bootstrap_start_date)
    requested_end_date = end_date if end_date is not None else karachi_today(now)
    return _run_enabled_update(
        config,
        end_date=requested_end_date,
        config_path=Path(config_path),
        lock_path=Path(lock_path),
        updater=updater,
        master_builder=master_builder,
        listing_refresher=listing_refresher,
        registry_builder=registry_builder,
        now=now,
    )
