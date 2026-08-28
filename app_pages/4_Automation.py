"""Configure and operate the standalone PSX data-management workflow."""

from dataclasses import replace
from datetime import date
import platform

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_csv_preview
from dashboard.presentation import (
    format_date,
    format_datetime,
    humanize_enum,
)
from data_pipeline.src.automation import (
    AutomationProgress,
    AutomationRunResult,
    rebuild_canonical_market_artifacts,
    recover_stale_automation_state,
    run_manual_update,
    save_automation_config,
)
from data_pipeline.src.config import MASTER_CSV_PATH
from data_pipeline.src.launchd import (
    get_launch_agent_status,
    install_launch_agent,
    trigger_launch_agent,
    uninstall_launch_agent,
)
from data_pipeline.src.updater import discover_available_raw_dates
from data_pipeline.src.market_schema import MarketSchemaError, with_legacy_date_alias


def _display_value(value: object | None) -> str:
    return format_datetime(value, fallback="Never")


def _latest_master_date() -> tuple[date | None, tuple[str, ...]]:
    if not MASTER_CSV_PATH.is_file():
        return None, ()
    data, errors = load_csv_preview((MASTER_CSV_PATH,))
    if data is None or data.empty:
        return None, errors
    try:
        data = with_legacy_date_alias(data)
    except MarketSchemaError as exc:
        return None, (*errors, f"Master dataset date schema is invalid: {exc}")
    if "date" not in data:
        return None, (*errors, f"Master dataset has no date column: {MASTER_CSV_PATH}")
    parsed_dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    latest_date = parsed_dates.max().date() if not parsed_dates.empty else None
    return latest_date, errors


def _show_update_result(result: AutomationRunResult) -> None:
    if result.status in {"success", "no_update_needed"}:
        st.success(result.message)
    elif result.status in {"already_running", "partial_success"}:
        st.warning(result.message)
    else:
        st.error(result.message)

    update = result.update_result
    if update is None:
        return
    metrics = st.columns(4)
    metrics[0].metric("Dates checked", update.total_processed, border=True)
    metrics[1].metric("Successful", len(update.successful_dates), border=True)
    metrics[2].metric("Skipped", len(update.skipped_dates), border=True)
    metrics[3].metric("Failed", len(update.failed_dates), border=True)
    if update.skipped_dates:
        st.write(
            "Skipped (eligible for retry): "
            + ", ".join(format_date(value) for value in update.skipped_dates)
        )
    if update.failed_dates:
        st.write("Failed dates:")
        for failed_date, reason in update.failed_dates:
            st.write(f"- {format_date(failed_date)}: {reason}")
    if result.status == "no_update_needed":
        st.write(
            "Update stages: market=current, native=current, downstream rebuilds="
            "not required"
        )
    else:
        st.write(
            "Update stages: "
            f"market={'complete' if result.market_update_succeeded else 'incomplete'}, "
            f"native={'complete' if result.native_update_succeeded else 'incomplete'}, "
            f"master={'complete' if result.master_rebuild_succeeded else 'not requested'}, "
            f"listings={'complete' if result.listing_refresh_succeeded else 'not requested'}, "
            f"registry={'complete' if result.registry_rebuild_succeeded else 'not requested'}"
        )
    if result.cached_listings_used:
        st.warning(
            "The registry was rebuilt with cached official listings because the "
            "live PSX listing source was unavailable."
        )
    if result.native_result is not None:
        native_metrics = st.columns(4)
        native_metrics[0].metric(
            "Native rows added", result.native_result.rows_added, border=True
        )
        native_metrics[1].metric(
            "Rows replaced", result.native_result.rows_replaced, border=True
        )
        native_metrics[2].metric(
            "Daily Parquets written",
            result.native_result.daily_parquets_written,
            border=True,
        )
        native_metrics[3].metric(
            "Native latest date",
            result.native_result.latest_date or "No data",
            border=True,
        )
    if result.audit is not None:
        for warning in result.audit.source_evidence_inconsistencies:
            st.warning(warning)


def _show_master_result(result) -> None:
    st.success(f"Canonical market artifacts rebuilt at {result.paths.master_csv}")
    metrics = st.columns(4)
    metrics[0].metric("Rows", result.master_rows, border=True)
    metrics[1].metric("Symbols", result.symbol_count, border=True)
    metrics[2].metric("Daily Parquets", result.daily_parquets_written, border=True)
    metrics[3].metric("Duplicate keys", result.duplicate_count, border=True)


st.title("Automation")
st.caption("Maintain missing daily files and the persistent master dataset.")
flash_message = st.session_state.pop("automation_flash_message", None)
if flash_message is not None:
    st.success(flash_message)

config = recover_stale_automation_state()
available_dates = discover_available_raw_dates()
latest_raw_date = available_dates[-1] if available_dates else None
latest_master_date, master_load_errors = _latest_master_date()
launch_status = get_launch_agent_status()
for error in master_load_errors:
    st.warning(error)

status_metrics = st.columns(4)
status_metrics[0].metric(
    "Automation configuration",
    "Enabled" if config.enabled else "Disabled",
    border=True,
)
status_metrics[1].metric("Schedule", "5:15 PM", border=True)
status_metrics[2].metric("Timezone", config.timezone, border=True)
status_metrics[3].metric(
    "LaunchAgent",
    "Loaded" if launch_status.loaded else "Not loaded",
    border=True,
)

data_metrics = st.columns(3)
data_metrics[0].metric(
    "Bootstrap start date",
    format_date(config.bootstrap_start_date)
    if config.bootstrap_start_date
    else "Not set",
    border=True,
)
data_metrics[1].metric(
    "Latest raw date",
    format_date(latest_raw_date, fallback="No data"),
    border=True,
)
data_metrics[2].metric(
    "Latest master date",
    format_date(latest_master_date, fallback="No data"),
    border=True,
)

st.write(f"Last attempt: {_display_value(config.last_attempt_at)}")
st.write(f"Last successful run: {_display_value(config.last_success_at)}")
st.write(f"Last status: {humanize_enum(config.last_status)}")
st.write(f"Last message: {config.last_message}")
if config.source_date_dispositions:
    with st.expander("Source-date dispositions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Date": value.trading_date.isoformat(),
                        "Classification": humanize_enum(value.classification),
                        "Reason": value.reason,
                        "Automatic retry": value.retry_automatically,
                        "Evidence": value.evidence,
                    }
                    for value in config.source_date_dispositions
                ]
            ),
            hide_index=True,
            width="stretch",
        )

st.info(
    "The Mac must be awake at 5:15 PM Pakistan time. Saving an enabled "
    "configuration does not install the macOS LaunchAgent; installation is a "
    "separate explicit action below."
)
st.caption(f"LaunchAgent path: {launch_status.plist_path}")
st.caption(launch_status.detail)

st.subheader("Automation settings")
with st.form("automation_settings"):
    enabled = st.checkbox(
        "Enable automatic daily fetch",
        value=config.enabled,
    )
    rebuild_ai_datasets = st.checkbox(
        "Rebuild AI datasets after a successful update",
        value=config.rebuild_ai_datasets,
        help="This prepares features only; it does not train or retrain models.",
    )
    bootstrap_start_date = st.date_input(
        "Bootstrap start date",
        value=config.bootstrap_start_date,
        help=(
            "Required before the first incremental update when no valid daily "
            "CSV exists. The updater will never guess an unlimited start date."
        ),
    )
    save_settings = st.form_submit_button(
        "Save automation settings",
        type="primary",
        icon=":material/save:",
    )

if save_settings:
    if enabled and bootstrap_start_date is None and not available_dates:
        st.error(
            "Set a bootstrap start date before enabling automation because no "
            "valid daily CSV exists."
        )
    else:
        try:
            updated_config = replace(
                config,
                enabled=enabled,
                bootstrap_start_date=bootstrap_start_date,
                rebuild_ai_datasets=rebuild_ai_datasets,
            )
            save_automation_config(updated_config)
            st.session_state["automation_flash_message"] = (
                "Automation settings saved."
            )
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Automation settings could not be saved: {exc}")

st.subheader("Data maintenance")
action_columns = st.columns(2)
if action_columns[0].button(
    "Fetch missing data now",
    icon=":material/sync:",
    width="stretch",
):
    try:
        with st.status(
            "Checking stored dates...", expanded=True, state="running"
        ) as update_status:
            def show_progress(event: AutomationProgress) -> None:
                update_status.update(label=event.message)
                update_status.write(f"**{humanize_enum(event.stage)}:** {event.message}")

            update_result = run_manual_update(progress_callback=show_progress)
            final_state = "error" if update_result.status == "failed" else "complete"
            update_status.update(
                label=update_result.message,
                state=final_state,
                expanded=update_result.status in {"failed", "partial_success"},
            )
        _show_update_result(update_result)
    except Exception as exc:
        st.error(f"Incremental update could not be completed: {exc}")

if action_columns[1].button(
    "Rebuild canonical market artifacts",
    icon=":material/build:",
    width="stretch",
):
    try:
        with st.status(
            "Validating native source state...", expanded=True
        ) as rebuild_status:
            def show_rebuild_progress(event: AutomationProgress) -> None:
                rebuild_status.update(label=event.message)
                rebuild_status.write(
                    f"**{humanize_enum(event.stage)}:** {event.message}"
                )

            master_result = rebuild_canonical_market_artifacts(
                progress_callback=show_rebuild_progress
            )
            rebuild_status.update(
                label="Canonical market artifacts rebuilt",
                state="complete",
                expanded=False,
            )
        _show_master_result(master_result)
    except Exception as exc:
        st.error(f"Canonical market rebuild could not be completed: {exc}")

st.subheader("macOS scheduler")
is_macos = platform.system() == "Darwin"
if not is_macos:
    st.warning("LaunchAgent controls are available only when running on macOS.")

scheduler_columns = st.columns(3)
if scheduler_columns[0].button(
    "Install scheduler",
    disabled=not is_macos,
    width="stretch",
):
    try:
        installed_status = install_launch_agent()
        st.session_state["automation_flash_message"] = (
            f"LaunchAgent installed at {installed_status.plist_path}"
        )
        st.rerun()
    except (OSError, RuntimeError) as exc:
        st.error(f"LaunchAgent installation failed: {exc}")

if scheduler_columns[1].button(
    "Trigger scheduler now",
    disabled=not is_macos or not launch_status.loaded,
    width="stretch",
):
    try:
        trigger_launch_agent()
        st.success("The installed LaunchAgent was triggered.")
    except (OSError, RuntimeError) as exc:
        st.error(f"LaunchAgent trigger failed: {exc}")

if scheduler_columns[2].button(
    "Uninstall scheduler",
    disabled=not is_macos or not launch_status.installed,
    width="stretch",
):
    try:
        removed_status = uninstall_launch_agent()
        st.session_state["automation_flash_message"] = (
            f"LaunchAgent removed from {removed_status.plist_path}"
        )
        st.rerun()
    except (OSError, RuntimeError) as exc:
        st.error(f"LaunchAgent removal failed: {exc}")
