"""Production control center for frozen-universe recurrent PPO training."""

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.presentation import format_integer, safe_display_value
from reinforcement_learning.training.job_state import FAILED, INTERRUPTED
from reinforcement_learning.training.production_control import (
    PRODUCTION_RUN_KIND,
    ProductionControlError,
    bounded_log_tail,
    default_run_selection,
    latest_job_diagnostics,
    launch_production_controller,
    list_run_catalog,
    load_run_snapshot,
    prepare_production_run,
    production_plan,
    recent_orchestration_events,
    registry_view,
    request_interrupt,
    request_stop_after_current,
    requeue_jobs,
)
from reinforcement_learning.training.model_details import (
    ModelDetailsAuditError,
    build_global_verified_model_inventory,
    research_partition_policy,
)
from reinforcement_learning.training.recurrent_orchestrator import TrainingRunStore
from reinforcement_learning.training.selective_training import (
    GLOBAL_COVERAGE_STATUSES,
    SELECTED_RUN_KIND,
    TRAINED,
    UNTRAINED,
    SelectiveTrainingError,
    build_global_model_coverage,
    canonical_symbol_selection,
    clear_visible_symbols,
    filter_symbol_coverage,
    load_selected_run_metadata,
    prepare_selected_run,
    reconcile_visible_symbol_selection,
    select_visible_symbols,
    selected_membership_hash,
)


SELECTOR_STATE_KEY = "selective_training_symbols"
SELECTOR_EDITOR_KEY = "selective_symbol_checkbox_editor"
SELECTOR_COLUMNS = (
    "selected",
    "symbol",
    "company_name",
    "sector",
    "coverage_status",
    "latest_progress_percent",
    "model_status",
    "validation_status",
    "latest_run_kind",
    "latest_run_id",
    "latest_attempt",
)
SELECTOR_READ_ONLY_COLUMNS = tuple(
    column for column in SELECTOR_COLUMNS if column != "selected"
)
TRAINING_DIAGNOSTIC_FIELDS = (
    ("explained_variance", "explained_variance"),
    ("approx_kl", "approximate_kl"),
    ("clip_fraction", "clip_fraction"),
    ("entropy_loss", "entropy_loss"),
    ("value_loss", "value_loss"),
    ("policy_gradient_loss", "policy_gradient_loss"),
)


def _duration(seconds: object) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else f"{minutes:d}m {secs:02d}s"


def _status_callout(status: str, message: str) -> None:
    if status == "RUNNING":
        st.info(message, icon=":material/progress_activity:")
    elif status == "COMPLETED":
        st.success(message, icon=":material/check_circle:")
    elif status in {
        "STOPPED_AFTER_CURRENT",
        "STOPPING_AFTER_CURRENT",
        "PAUSED",
        "INTERRUPTED",
    }:
        st.warning(message, icon=":material/warning:")
    elif status in {"BLOCKED", "FAILED"}:
        st.error(message, icon=":material/error:")
    else:
        st.caption(message)


def _filtered_jobs(
    jobs: pd.DataFrame,
    *,
    statuses: list[str],
    sectors: list[str],
    eligibility: str,
    search: str,
) -> pd.DataFrame:
    filtered = jobs.copy(deep=True)
    if statuses:
        filtered = filtered.loc[filtered["state"].isin(statuses)]
    if sectors:
        filtered = filtered.loc[filtered["sector"].isin(sectors)]
    if eligibility != "All":
        filtered = filtered.loc[filtered["eligibility"].eq(eligibility.lower())]
    query = search.strip()
    if query:
        matched = filtered["symbol"].astype("string").str.contains(
            query, case=False, regex=False, na=False
        ) | filtered["company_name"].astype("string").str.contains(
            query, case=False, regex=False, na=False
        )
        filtered = filtered.loc[matched]
    return filtered.reset_index(drop=True)


def _reset_selector_editor() -> None:
    st.session_state.pop(SELECTOR_EDITOR_KEY, None)


def _set_visible_selection(
    visible_symbols: tuple[str, ...], eligible_symbols: tuple[str, ...]
) -> None:
    st.session_state[SELECTOR_STATE_KEY] = list(
        select_visible_symbols(
            st.session_state.get(SELECTOR_STATE_KEY, ()),
            visible_symbols=visible_symbols,
            eligible_symbols=eligible_symbols,
        )
    )
    _reset_selector_editor()


def _clear_visible_selection(
    visible_symbols: tuple[str, ...], eligible_symbols: tuple[str, ...]
) -> None:
    st.session_state[SELECTOR_STATE_KEY] = list(
        clear_visible_symbols(
            st.session_state.get(SELECTOR_STATE_KEY, ()),
            visible_symbols=visible_symbols,
            eligible_symbols=eligible_symbols,
        )
    )
    _reset_selector_editor()


def _clear_all_selection() -> None:
    st.session_state[SELECTOR_STATE_KEY] = []
    _reset_selector_editor()


st.space("medium")
st.title("Training & Models")
st.caption(
    "Production control center for the frozen single-symbol RecurrentPPO run. "
    "This page reads persistent state and issues explicit commands; browser "
    "reruns never own worker lifetimes."
)

plan = production_plan()
catalog = list_run_catalog()

st.subheader("System readiness")
with st.container(horizontal=True):
    st.metric("Research identities", format_integer(plan.identity_count), border=True)
    st.metric("Trainable agents", format_integer(plan.trainable_count), border=True)
    st.metric("Excluded", format_integer(plan.excluded_count), border=True)
    st.metric("TEST", plan.test_status, border=True)
with st.container(horizontal=True):
    st.metric("Algorithm", plan.algorithm, border=True)
    st.metric("Policy", plan.policy, border=True)
    st.metric("Budget / symbol", f"{plan.requested_timesteps:,}", border=True)
    st.metric("Execution", "CPU · 4 × 2 threads", border=True)
st.caption(f"Training policy: `{plan.execution_training_policy}`")
st.caption(
    f"Trainable symbol hash: `{plan.trainable_symbol_hash[:12]}…{plan.trainable_symbol_hash[-8:]}`"
)
with st.expander("Full immutable hashes", icon=":material/fingerprint:"):
    st.code(
        f"universe_hash={plan.universe_hash}\n"
        f"trainable_symbol_hash={plan.trainable_symbol_hash}",
        language="text",
    )

st.subheader("Production training plan")
with st.container(border=True):
    st.caption(
        "Training scope is the frozen 508-identity research snapshot dated "
        "2026-08-02. The current operational identity universe is outside this run."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Frozen field": [
                    "Frozen research universe",
                    "Frozen snapshot date",
                    "Research identities",
                    "Frozen universe version",
                    "Frozen universe hash",
                    "Execution-training policy",
                    "Trainable agents",
                    "Trainable symbol hash",
                    "Underlying identity contract",
                    "Algorithm / policy",
                    "Seed",
                    "Timesteps",
                    "Recurrent config",
                    "Environment",
                    "Trainer",
                    "Validation",
                    "TEST",
                ],
                "Value": [
                    str(plan.identity_policy),
                    str(plan.identity_snapshot),
                    f"{plan.identity_count:,}",
                    str(plan.frozen_universe_version),
                    str(plan.universe_hash),
                    str(plan.execution_training_policy),
                    f"{plan.trainable_count:,}",
                    str(plan.trainable_symbol_hash),
                    str(plan.universe_version),
                    f"{plan.algorithm} / {plan.policy}",
                    str(plan.seed),
                    f"{plan.requested_timesteps:,}",
                    str(plan.recurrent_config_version),
                    str(plan.environment_version),
                    str(plan.trainer_version),
                    "after TRAIN",
                    str(plan.test_status),
                ],
            }
        ),
        hide_index=True,
    )
    st.caption(
        "These values are immutable and have no editing controls. The qualified "
        f"Apple M2 throughput was approximately {plan.qualified_agents_per_hour:.2f} "
        "agents/hour; this is an estimate, not a completion guarantee."
    )

st.subheader("Model coverage and selective training")
st.caption(
    "Coverage is reconstructed from valid persisted recurrent jobs, hash-verified "
    "models, and VALIDATION artifacts across all run history. Registry rows alone "
    "do not establish TRAINED status."
)
try:
    coverage, coverage_summary = build_global_model_coverage()
except (OSError, ValueError, RuntimeError) as exc:
    coverage = pd.DataFrame()
    coverage_summary = None
    st.error(f"Global model coverage failed closed: {type(exc).__name__}: {exc}")

if coverage_summary is not None:
    with st.container(horizontal=True):
        st.metric("Eligible symbols", coverage_summary.eligible, border=True)
        st.metric("Trained", coverage_summary.trained, border=True)
        st.metric("Untrained", coverage_summary.untrained, border=True)
        st.metric("Currently training", coverage_summary.training, border=True)
    with st.container(horizontal=True):
        st.metric("Validating", coverage_summary.validating, border=True)
        st.metric("Failed", coverage_summary.failed, border=True)
        st.metric("Interrupted", coverage_summary.interrupted, border=True)

    with st.container(border=True):
        st.markdown("**Choose exact frozen-eligible symbols**")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            coverage_statuses = st.multiselect(
                "Coverage status",
                list(GLOBAL_COVERAGE_STATUSES),
                key="selective_coverage_status_filter",
            )
            coverage_sectors = st.multiselect(
                "Sector",
                sorted(coverage["sector"].dropna().astype(str).unique()),
                key="selective_sector_filter",
            )
            coverage_search = st.text_input(
                "Search symbol/company", key="selective_symbol_search"
            )
        filtered_coverage = filter_symbol_coverage(
            coverage,
            statuses=coverage_statuses,
            sectors=coverage_sectors,
            search=coverage_search,
        )
        eligible_symbols = tuple(coverage["symbol"].astype(str))
        visible_symbols = tuple(filtered_coverage["symbol"].astype(str))
        selected_symbols = canonical_symbol_selection(
            st.session_state.get(SELECTOR_STATE_KEY, ()),
            eligible_symbols=eligible_symbols,
        )
        st.session_state[SELECTOR_STATE_KEY] = list(selected_symbols)
        shown_coverage = filtered_coverage.copy(deep=True)
        shown_coverage.insert(
            0, "selected", shown_coverage["symbol"].isin(selected_symbols)
        )
        editor_input = shown_coverage.loc[:, list(SELECTOR_COLUMNS)].set_index(
            "symbol", drop=False
        )
        edited_coverage = st.data_editor(
            editor_input,
            hide_index=True,
            num_rows="fixed",
            disabled=SELECTOR_READ_ONLY_COLUMNS,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Select this eligible symbol for a SELECTED training run.",
                ),
                "symbol": st.column_config.TextColumn("Symbol", pinned=True),
                "latest_progress_percent": st.column_config.ProgressColumn(
                    "Latest progress", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
            key=SELECTOR_EDITOR_KEY,
        )
        checked_visible = tuple(
            edited_coverage.loc[
                edited_coverage["selected"].astype(bool), "symbol"
            ].astype(str)
        )
        selected_symbols = reconcile_visible_symbol_selection(
            selected_symbols,
            visible_symbols=visible_symbols,
            checked_visible_symbols=checked_visible,
            eligible_symbols=eligible_symbols,
        )
        st.session_state[SELECTOR_STATE_KEY] = list(selected_symbols)
        selected_in_view = len(set(selected_symbols).intersection(visible_symbols))
        with st.container(horizontal=True):
            st.button(
                f"Select visible ({len(visible_symbols)})",
                icon=":material/select_all:",
                on_click=_set_visible_selection,
                args=(visible_symbols, eligible_symbols),
                disabled=not visible_symbols,
            )
            st.button(
                "Clear visible",
                icon=":material/deselect:",
                on_click=_clear_visible_selection,
                args=(visible_symbols, eligible_symbols),
                disabled=selected_in_view == 0,
            )
            st.button(
                "Clear all selection",
                icon=":material/delete_sweep:",
                on_click=_clear_all_selection,
                disabled=not selected_symbols,
            )
            st.caption(
                f"Selected: {len(selected_symbols)} total · "
                f"Visible: {len(filtered_coverage)} · "
                f"Selected in current view: {selected_in_view}"
            )

        confirmed_symbols = tuple(sorted(selected_symbols))
        selected_rows = coverage.loc[coverage["symbol"].isin(confirmed_symbols)]
        trained_selected = tuple(
            sorted(selected_rows.loc[selected_rows["trained"], "symbol"].astype(str))
        )
        default_members = tuple(
            symbol for symbol in confirmed_symbols if symbol not in trained_selected
        )
        st.markdown("**Train selected symbols**")
        if confirmed_symbols:
            st.write(
                f"Requested: **{len(confirmed_symbols)}** · default new-training "
                f"membership: **{len(default_members)}** · already TRAINED and skipped: "
                f"**{len(trained_selected)}**"
            )
            if default_members:
                membership_hash = selected_membership_hash(default_members)
                st.caption(
                    f"Effective membership hash: `{membership_hash}`"
                )
                with st.expander(
                    f"Exact effective membership ({len(default_members)})",
                    icon=":material/list:",
                ):
                    st.code("\n".join(default_members), language="text")
            st.caption(
                "100,000 timesteps each · RecurrentPPO / MlpLstmPolicy · CPU · "
                "4 workers / 2 threads each · VALIDATION after TRAIN · TEST sealed"
            )
        else:
            st.caption("No symbols selected. Selection never defaults to all 435.")
        selective_confirmed = st.checkbox(
            "I confirm this exact SELECTED membership and training contract.",
            key="confirm_selected_training",
        )
        if st.button(
            "Train selected symbols",
            type="primary",
            icon=":material/play_arrow:",
            disabled=(not default_members or not selective_confirmed),
        ):
            try:
                store, metadata, _ = prepare_selected_run(confirmed_symbols)
                launch_production_controller(store)
                st.session_state["training_control_run_id"] = metadata.run_id
                st.toast(
                    f"SELECTED run launched for {len(metadata.selected_symbols)} symbols.",
                    icon=":material/check_circle:",
                )
                st.rerun()
            except (OSError, ValueError, ProductionControlError, SelectiveTrainingError) as exc:
                st.error(f"Selected launch failed safely: {type(exc).__name__}: {exc}")
        with st.expander(
            "Explicit retraining of already TRAINED symbols",
            icon=":material/replay:",
        ):
            st.warning(
                "Retraining creates a new isolated attempt/version. It never "
                "overwrites the previously verified model."
            )
            if trained_selected:
                retrain_hash = selected_membership_hash(trained_selected)
                st.caption(
                    f"Retraining membership: {len(trained_selected)} · hash "
                    f"`{retrain_hash}` · 100,000 timesteps each · CPU 4 × 2 threads"
                )
                st.code("\n".join(trained_selected), language="text")
            retrain_confirmed = st.checkbox(
                "I explicitly authorize a new attempt for the selected TRAINED symbols.",
                key="confirm_selected_retraining",
            )
            if st.button(
                "Retrain selected trained symbols",
                icon=":material/restart_alt:",
                disabled=(not trained_selected or not retrain_confirmed),
            ):
                try:
                    store, metadata, _ = prepare_selected_run(
                        trained_selected, retrain_trained=True
                    )
                    launch_production_controller(store)
                    st.session_state["training_control_run_id"] = metadata.run_id
                    st.toast(
                        f"New retraining attempt launched for {len(metadata.selected_symbols)} symbols.",
                        icon=":material/check_circle:",
                    )
                    st.rerun()
                except (
                    OSError,
                    ValueError,
                    ProductionControlError,
                    SelectiveTrainingError,
                ) as exc:
                    st.error(f"Retraining failed safely: {type(exc).__name__}: {exc}")

st.subheader("Run selection and controls")
if catalog:
    options = {entry.run_id: entry for entry in catalog}
    selected_default = default_run_selection(
        catalog, st.session_state.get("training_control_run_id")
    )
    if selected_default is not None and (
        st.session_state.get("training_control_run_id") != selected_default
    ):
        st.session_state["training_control_run_id"] = selected_default
    selected_run_id = st.selectbox(
        "Run",
        list(options),
        format_func=lambda run_id: (
            f"[{options[run_id].run_kind}] {run_id} — {options[run_id].status}"
            + (
                f" · {options[run_id].selected_count} selected"
                if options[run_id].selected_count is not None
                else ""
            )
        ),
        key="training_control_run_id",
        persist_state="session",
    )
else:
    options = {}
    selected_run_id = None
    st.caption("No recurrent run has been prepared yet.")

snapshot = None
if selected_run_id:
    try:
        snapshot = load_run_snapshot(options[selected_run_id].run_directory)
    except (OSError, ValueError, ProductionControlError) as exc:
        st.error(f"Could not load selected run safely: {type(exc).__name__}: {exc}")

with st.container(border=True):
    if snapshot is None:
        st.markdown("**Pre-run state**")
        st.write(
            "Preparing creates 508 persistent job records: 435 eligible and 73 "
            "explicitly ineligible. It does not start training."
        )
        prepare_confirmed = st.checkbox(
            "I confirm this will create the immutable production run only.",
            key="confirm_prepare_production_run",
        )
        if st.button(
            "Prepare production run",
            type="primary",
            icon=":material/inventory_2:",
            disabled=not prepare_confirmed,
        ):
            try:
                store, created = prepare_production_run()
                st.session_state["training_control_run_id"] = store.read_manifest().run_id
                st.toast(
                    "Production run prepared." if created else "Compatible run already exists.",
                    icon=":material/check_circle:",
                )
                st.rerun()
            except (OSError, ValueError, ProductionControlError) as exc:
                st.error(f"Preparation failed safely: {type(exc).__name__}: {exc}")
    else:
        st.markdown(
            f"**{snapshot.run_kind}** · `{snapshot.manifest.run_id}` · "
            f"status `{snapshot.progress.system_status}`"
        )
        if snapshot.run_kind not in {PRODUCTION_RUN_KIND, SELECTED_RUN_KIND}:
            st.warning(
                "Benchmark, smoke, and legacy runs are read-only here and cannot "
                "be resumed as the production run."
            )
        else:
            if snapshot.run_kind == SELECTED_RUN_KIND:
                selected_metadata = load_selected_run_metadata(
                    snapshot.store.run_directory
                )
                st.write(
                    f"Start confirmation: {len(selected_metadata.selected_symbols)} "
                    "selected agents · 100,000 timesteps each · 4 workers · "
                    "2 CPU threads/worker · CPU · validation enabled · TEST sealed."
                )
                st.caption(
                    f"Immutable selected hash: `{selected_metadata.selected_symbol_hash}`"
                )
            else:
                st.write(
                    "Start confirmation: 435 eligible agents · 100,000 timesteps each · "
                    "4 workers · 2 CPU threads/worker · CPU · validation enabled · TEST sealed."
                )
            with st.container(horizontal=True, vertical_alignment="bottom"):
                if st.button(
                    "Refresh status", icon=":material/refresh:", key="refresh_training_status"
                ):
                    st.rerun()
                start_confirmed = st.checkbox(
                    "Confirm start/continue",
                    key="confirm_start_production_run",
                    disabled=snapshot.controller.alive,
                )
                if st.button(
                    "Start / continue run",
                    type="primary",
                    icon=":material/play_arrow:",
                    disabled=(
                        snapshot.controller.alive
                        or snapshot.progress.queued == 0
                        or not start_confirmed
                    ),
                ):
                    try:
                        launch_production_controller(snapshot.store)
                        st.toast("Detached controller started.", icon=":material/check_circle:")
                        st.rerun()
                    except (OSError, ValueError, ProductionControlError) as exc:
                        st.error(f"Start failed safely: {type(exc).__name__}: {exc}")
            if snapshot.controller.alive:
                with st.container(horizontal=True, vertical_alignment="bottom"):
                    stop_confirmed = st.checkbox(
                        "Confirm stop after current jobs", key="confirm_stop_after_current"
                    )
                    if st.button(
                        "Stop after current jobs",
                        icon=":material/pause:",
                        disabled=not stop_confirmed,
                    ):
                        try:
                            request_stop_after_current(snapshot.store)
                            st.rerun()
                        except ProductionControlError as exc:
                            st.error(str(exc))
                    interrupt_confirmed = st.checkbox(
                        "Confirm active interruption", key="confirm_interrupt_run"
                    )
                    if st.button(
                        "Interrupt active run",
                        icon=":material/stop_circle:",
                        disabled=not interrupt_confirmed,
                    ):
                        try:
                            request_interrupt(snapshot.store)
                            st.rerun()
                        except (OSError, ProductionControlError) as exc:
                            st.error(str(exc))
                st.caption(
                    "Stop-after-current launches no new jobs. Interrupt sends SIGINT; "
                    "active jobs become INTERRUPTED and can only restart from zero."
                )

if snapshot is not None:
    progress = snapshot.progress
    selected_metadata = (
        load_selected_run_metadata(snapshot.store.run_directory)
        if snapshot.run_kind == SELECTED_RUN_KIND
        else None
    )
    if selected_metadata is not None and progress.system_status == "COMPLETED":
        validations_completed = int(
            snapshot.jobs["validation_status"].eq("completed").sum()
        )
        with st.container(border=True):
            st.markdown("**Selected run completed**")
            with st.container(horizontal=True):
                st.metric("Run type", SELECTED_RUN_KIND, border=True)
                st.metric(
                    "Completed",
                    f"{progress.completed:,} / {progress.eligible:,}",
                    border=True,
                )
                st.metric("Failed", progress.failed, border=True)
                st.metric("Interrupted", progress.interrupted, border=True)
            with st.container(horizontal=True):
                st.metric(
                    "Validation completed",
                    f"{validations_completed:,} / {progress.eligible:,}",
                    border=True,
                )
                st.metric("Elapsed", _duration(progress.elapsed_seconds), border=True)
                st.metric("Final status", progress.system_status, border=True)
            st.caption(
                f"Membership hash: `{selected_metadata.selected_symbol_hash}`"
            )
    st.subheader("Overall progress")
    _status_callout(progress.system_status, f"Execution status: {progress.system_status}")
    if (
        progress.completed_training_timesteps > 0
        or progress.active
        or progress.failed
        or progress.interrupted
    ):
        st.progress(
            progress.progress_percent / 100.0,
            text=(
                f"{progress.completed_training_timesteps:,} / "
                f"{progress.requested_training_timesteps:,} persisted training "
                f"timesteps · {progress.progress_percent:.1f}%"
            ),
        )
    else:
        st.caption("Prepared and not running; no progress indicator is implied.")
    with st.container(horizontal=True):
        st.metric(
            "Agents completed",
            f"{progress.completed:,} / {progress.eligible:,}",
            border=True,
        )
        st.metric("Active", progress.active, border=True)
        st.metric("Queued", progress.queued, border=True)
        st.metric("Validating", progress.validating, border=True)
    with st.container(horizontal=True):
        st.metric("Failed", progress.failed, border=True)
        st.metric("Interrupted", progress.interrupted, border=True)
        st.metric("Ineligible", progress.ineligible, border=True)
        st.metric("Elapsed", _duration(progress.elapsed_seconds), border=True)
    historical_throughput = (
        f"{progress.agents_per_hour:.2f} agents/hour"
        if progress.agents_per_hour is not None
        else "unavailable until at least two completions span one minute"
    )
    st.caption(f"Historical observed throughput: {historical_throughput}")
    remaining_agents = max(0, progress.eligible - progress.completed)
    if progress.system_status == "RUNNING":
        if progress.estimated_remaining_seconds is not None:
            st.caption(
                "Active estimated remaining time: "
                + _duration(progress.estimated_remaining_seconds)
            )
        else:
            st.caption("Active ETA unavailable: insufficient throughput evidence.")
    elif progress.system_status in {"STOPPED_AFTER_CURRENT", "PAUSED"}:
        st.caption(
            f"Remaining work: {remaining_agents:,} agents · "
            "ETA unavailable while run is stopped."
        )
    if progress.system_status == "COMPLETED":
        st.success(
            "TRAIN and VALIDATION complete. TEST remains sealed.",
            icon=":material/verified:",
        )

    if selected_metadata is not None:
        st.caption(
            f"SELECTED membership: {len(selected_metadata.selected_symbols)} · "
            f"hash `{selected_metadata.selected_symbol_hash}` · "
            f"attempt version {selected_metadata.attempt_version}"
        )

    st.subheader("Active jobs")
    active_jobs = snapshot.jobs.loc[
        snapshot.jobs["state"].isin({"TRAINING", "VALIDATING"})
    ].copy()
    if active_jobs.empty:
        st.caption("No active TRAINING or VALIDATING jobs.")
    else:
        for active in active_jobs.itertuples(index=False):
            state_label = (
                "Training complete — validating"
                if active.state == "VALIDATING"
                else "Training"
            )
            with st.container(border=True, gap="small"):
                st.markdown(
                    f"**{active.symbol}** · "
                    f"{safe_display_value(active.company_name)} · "
                    f"{safe_display_value(active.sector)}"
                )
                st.caption(state_label)
                st.progress(
                    float(active.progress_percent) / 100.0,
                    text=(
                        f"{int(active.actual_timesteps):,} / "
                        f"{int(active.requested_timesteps):,} timesteps · "
                        f"{float(active.progress_percent):.1f}%"
                    ),
                )
                st.caption(
                    f"Elapsed {_duration(active.runtime_seconds)} · "
                    f"worker slot {safe_display_value(active.worker_slot)} · "
                    f"PID {safe_display_value(active.worker_pid)} · "
                    f"{safe_display_value(active.effective_device)} · "
                    f"{safe_display_value(active.cpu_threads)} CPU threads"
                )
                with st.expander(
                    f"{active.symbol} advanced diagnostics",
                    icon=":material/query_stats:",
                ):
                    diagnostics = latest_job_diagnostics(
                        snapshot.store, active.symbol
                    )
                    if diagnostics:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "Metric": [
                                        label for label, _ in TRAINING_DIAGNOSTIC_FIELDS
                                    ],
                                    "Value": [
                                        diagnostics.get(source_key)
                                        for _, source_key in TRAINING_DIAGNOSTIC_FIELDS
                                    ],
                                }
                            ),
                            hide_index=True,
                            key=f"active_diagnostics_{active.symbol}",
                        )
                    else:
                        st.caption("No retained diagnostics yet.")

    st.subheader("All jobs")
    with st.form("training_job_filters", border=False):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            selected_statuses = st.multiselect(
                "Status", sorted(snapshot.jobs["state"].dropna().unique()),
                key="training_job_status_filter",
            )
            selected_sectors = st.multiselect(
                "Sector", sorted(snapshot.jobs["sector"].dropna().astype(str).unique()),
                key="training_job_sector_filter",
            )
            selected_eligibility = st.segmented_control(
                "Eligibility", ["All", "Eligible", "Ineligible"], default="All",
                key="training_job_eligibility_filter",
            )
            search = st.text_input("Search symbol/company", key="training_job_search")
            st.form_submit_button("Apply filters", icon=":material/filter_alt:")
    filtered = _filtered_jobs(
        snapshot.jobs,
        statuses=selected_statuses,
        sectors=selected_sectors,
        eligibility=str(selected_eligibility or "All"),
        search=search,
    )
    st.dataframe(
        filtered.loc[
            :,
            [
                "symbol", "company_name", "sector", "eligibility", "state",
                "exclusion_reason", "requested_timesteps", "actual_timesteps",
                "progress_percent", "validation_status", "runtime_seconds",
                "started_at", "completed_at", "attempts", "last_error",
                "model_artifact_status",
            ],
        ],
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", pinned=True),
            "progress_percent": st.column_config.ProgressColumn(
                "Progress", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
        key="training_all_jobs",
    )

    st.subheader("Failures and retries")
    failures = snapshot.jobs.loc[
        snapshot.jobs["state"].isin({FAILED, INTERRUPTED})
    ].copy()
    if failures.empty:
        st.caption("No FAILED or INTERRUPTED jobs.")
    else:
        st.dataframe(
            failures.loc[
                :,
                [
                    "symbol", "state", "error_type", "last_error", "attempts",
                    "actual_timesteps", "requested_timesteps", "progress_percent",
                    "updated_at",
                ],
            ],
            hide_index=True,
            column_config={
                "progress_percent": st.column_config.ProgressColumn(
                    "Last persisted progress",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            },
        )
        retry_symbols = st.multiselect(
            "Selected jobs to restart from zero", failures["symbol"].tolist(),
            key="training_retry_symbols",
        )
        retry_confirmed = st.checkbox(
            "I understand these jobs restart from timestep zero.",
            key="confirm_retry_jobs",
        )
        with st.container(horizontal=True):
            if st.button(
                "Retry selected", icon=":material/replay:",
                disabled=snapshot.controller.alive or not retry_confirmed or not retry_symbols,
            ):
                states = frozenset(
                    snapshot.jobs.loc[
                        snapshot.jobs["symbol"].isin(retry_symbols), "state"
                    ]
                )
                try:
                    requeue_jobs(snapshot.store, statuses=states, symbols=retry_symbols)
                    st.rerun()
                except ProductionControlError as exc:
                    st.error(str(exc))
            if st.button(
                "Retry all failed", icon=":material/restart_alt:",
                disabled=snapshot.controller.alive or not retry_confirmed or progress.failed == 0,
            ):
                try:
                    requeue_jobs(snapshot.store, statuses=frozenset({FAILED}))
                    st.rerun()
                except ProductionControlError as exc:
                    st.error(str(exc))
            if st.button(
                "Requeue interrupted", icon=":material/resume:",
                disabled=(
                    snapshot.controller.alive or not retry_confirmed
                    or progress.interrupted == 0
                ),
            ):
                try:
                    requeue_jobs(snapshot.store, statuses=frozenset({INTERRUPTED}))
                    st.rerun()
                except ProductionControlError as exc:
                    st.error(str(exc))

    st.subheader("Recurrent model inventory")
    try:
        models = registry_view()
    except ProductionControlError as exc:
        st.error(f"Model registry could not be verified: {exc}")
        models = pd.DataFrame(columns=["symbol", "model_family"])
    registry_recurrent_count = (
        int(models["model_family"].eq("RECURRENT").sum())
        if "model_family" in models.columns
        else 0
    )
    with st.container(horizontal=True):
        st.metric(
            "Verified recurrent models",
            coverage_summary.trained if coverage_summary is not None else "Unavailable",
            border=True,
        )
        st.metric(
            "Registry-promoted models", registry_recurrent_count, border=True
        )
    st.caption(
        "Verified models are integrity-checked run-isolated artifacts. The model "
        "registry is an optional promotion layer and is not required for those "
        "artifacts to exist."
    )
    if models.empty:
        st.caption(
            "The optional promoted model registry contains no rows. Verified "
            "run-isolated recurrent models remain available in their training runs."
        )
    else:
        family = st.segmented_control(
            "Model family", ["All", "RECURRENT", "LEGACY"], default="All"
        )
        shown_models = (
            models if family == "All" else models.loc[models["model_family"].eq(family)]
        )
        st.dataframe(shown_models, hide_index=True, key="training_model_registry")

    st.subheader("Global verified model details")
    st.caption(
        "This inventory spans every valid recurrent run, independent of the run "
        "selected above. Only hash-verified models with compatible persisted "
        "training and validation metadata are included."
    )
    try:
        verified_inventory = build_global_verified_model_inventory(coverage=coverage)
    except (OSError, ValueError, RuntimeError, ModelDetailsAuditError) as exc:
        verified_inventory = pd.DataFrame()
        st.error(
            "Global verified model details failed closed: "
            f"{type(exc).__name__}: {exc}"
        )

    if verified_inventory.empty:
        st.caption("No globally verified recurrent model details are available.")
    else:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            detail_run_types = st.multiselect(
                "Training run type",
                sorted(verified_inventory["run_type"].unique()),
                key="training_detail_run_type_filter",
            )
            detail_sectors = st.multiselect(
                "Model sector",
                sorted(verified_inventory["sector"].dropna().astype(str).unique()),
                key="training_detail_sector_filter",
            )
            detail_search = st.text_input(
                "Search verified model", key="training_detail_search"
            )
        filtered_inventory = verified_inventory.copy(deep=True)
        if detail_run_types:
            filtered_inventory = filtered_inventory.loc[
                filtered_inventory["run_type"].isin(detail_run_types)
            ]
        if detail_sectors:
            filtered_inventory = filtered_inventory.loc[
                filtered_inventory["sector"].isin(detail_sectors)
            ]
        if detail_search.strip():
            query = detail_search.strip().casefold()
            filtered_inventory = filtered_inventory.loc[
                filtered_inventory["symbol"].str.casefold().str.contains(
                    query, regex=False
                )
                | filtered_inventory["company_name"].fillna("").str.casefold().str.contains(
                    query, regex=False
                )
            ]
        st.caption(
            f"Verified models: {len(verified_inventory):,} · "
            f"visible after filters: {len(filtered_inventory):,}"
        )
        if filtered_inventory.empty:
            st.info("No verified models match these filters.")
        else:
            selected_symbol = st.selectbox(
                "Verified model symbol",
                filtered_inventory["symbol"].tolist(),
                key="training_detail_symbol",
                persist_state="session",
            )
            detail = filtered_inventory.loc[
                filtered_inventory["symbol"].eq(selected_symbol)
            ].iloc[0]
            policy = research_partition_policy()
            with st.container(border=True):
                st.markdown("**Research partition policy**")
                st.write(
                    {
                        "policy_version": policy["version"],
                        "TRAIN": policy["train"],
                        "VALIDATION": policy["validation"],
                        "TEST": policy["test"],
                        "normalization": policy["normalization"],
                    }
                )
                st.caption(
                    "The policy is per symbol, not one global calendar cutoff. "
                    "Feature warm-up, listing age, suspensions, and missing market "
                    "dates can therefore produce different model-observed dates."
                )
            with st.container(border=True):
                with st.container(horizontal=True):
                    st.metric("Symbol", selected_symbol, border=True)
                    st.metric(
                        "Company", safe_display_value(detail["company_name"]), border=True
                    )
                    st.metric(
                        "Sector", safe_display_value(detail["sector"]), border=True
                    )
                    st.metric("Run type", detail["run_type"], border=True)
                with st.container(horizontal=True):
                    st.metric("Training", detail["training_status"], border=True)
                    st.metric("Validation", detail["validation_status"], border=True)
                    st.metric(
                        "Artifact", detail["artifact_verification"], border=True
                    )
                    st.metric("Attempt", int(detail["attempt"]), border=True)
                st.write(
                    {
                        "run_id": detail["run_id"],
                        "algorithm": detail["algorithm"],
                        "policy": detail["policy"],
                        "device": detail["effective_device"],
                        "runtime": _duration(detail["runtime_seconds"]),
                        "actual_timesteps": int(detail["actual_timesteps"]),
                    }
                )
                st.markdown("**Model-specific partition ranges**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Range": [
                                "Current raw availability (date column only)",
                                "Usable post-feature history",
                                "Model-observed TRAIN",
                                "Model-observed VALIDATION",
                                "SEALED TEST boundary metadata",
                            ],
                            "First date": [
                                detail["raw_available_start"],
                                detail["usable_feature_start"],
                                detail["train_start"],
                                detail["validation_start"],
                                detail["test_start"],
                            ],
                            "Last date": [
                                detail["raw_available_end"],
                                detail["usable_feature_end"],
                                detail["train_end"],
                                detail["validation_end"],
                                detail["test_end"],
                            ],
                            "Rows/dates": [
                                int(detail["raw_available_rows"]),
                                int(detail["usable_feature_rows"]),
                                int(detail["train_rows"]),
                                int(detail["validation_rows"]),
                                int(detail["test_rows"]),
                            ],
                        }
                    ),
                    hide_index=True,
                    key="training_model_partition_ranges",
                )
                st.caption(
                    "Model-specific ranges are persisted contract metadata. TEST "
                    "observations and returns are not opened by this view."
                )
                model_store = TrainingRunStore(Path(str(detail["run_directory"])))
                diagnostics = latest_job_diagnostics(model_store, selected_symbol)
                if diagnostics:
                    st.markdown("**Training diagnostics**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Metric": [
                                    label for label, _ in TRAINING_DIAGNOSTIC_FIELDS
                                ],
                                "Value": [
                                    diagnostics.get(source_key)
                                    for _, source_key in TRAINING_DIAGNOSTIC_FIELDS
                                ],
                            }
                        ),
                        hide_index=True,
                        key="training_model_diagnostics",
                    )
                st.markdown("**Validation and artifacts**")
                st.write(
                    {
                        "validation_status": detail["validation_status"],
                        "validation_partition": detail["validation_partition"],
                        "validation_artifact": detail[
                            "validation_metrics_reference"
                        ],
                        "model_path": detail["model_path"],
                        "artifact_verification": detail["artifact_verification"],
                        "registry_entry": (
                            "present"
                            if "symbol" in models.columns
                            and not models.loc[
                                models["symbol"].eq(selected_symbol)
                            ].empty
                            else "absent"
                        ),
                    }
                )
                st.caption(
                    "No TRAIN, VALIDATION, or TEST dataframe is opened by this "
                    "detail inventory. Only persisted manifests, hashes, and the "
                    "raw market-date column are inspected."
                )

    st.subheader("Logs and advanced state")
    with st.expander("Latest orchestration events", icon=":material/history:"):
        events = recent_orchestration_events(snapshot.store)
        if events.empty:
            st.caption("No persisted orchestration events.")
        else:
            st.dataframe(events, hide_index=True, key="training_recent_events")
    with st.expander("Bounded controller log", icon=":material/description:"):
        log_path = snapshot.store.run_directory / "logs" / "production_controller.log"
        st.code(bounded_log_tail(log_path) or "No controller log output yet.", language="text")
    with st.expander("Controller and manifest provenance", icon=":material/info:"):
        st.json(
            {
                "controller": {
                    "normalized_execution_status": progress.system_status,
                    "persisted_controller_state": snapshot.controller.state,
                    "pid": snapshot.controller.pid,
                    "alive": snapshot.controller.alive,
                    "started_at": snapshot.controller.started_at,
                    "updated_at": snapshot.controller.updated_at,
                    "message": snapshot.controller.message,
                },
                "run": {
                    "run_id": snapshot.manifest.run_id,
                    "identity_policy": snapshot.manifest.identity_policy,
                    "identity_snapshot": snapshot.manifest.identity_snapshot,
                    "universe_hash": snapshot.manifest.universe_hash,
                    "trainable_symbol_hash": snapshot.manifest.trainable_symbol_hash,
                    "TEST_loaded": snapshot.manifest.test_partition_loaded,
                },
            }
        )

if catalog:
    st.subheader("Run history")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "run_type": entry.run_kind,
                    "run_id": entry.run_id,
                    "status": entry.status,
                    "created_at": entry.created_at,
                    "identity_count": entry.identity_count,
                    "eligible_count": entry.eligible_count,
                    "selected_count": entry.selected_count,
                    "selected_symbol_hash": entry.selected_symbol_hash,
                }
                for entry in catalog
            ]
        ),
        hide_index=True,
        key="training_run_history",
    )

st.caption(
    "Training controls never evaluate TEST, mutate the frozen plan, or promote "
    "models automatically. Closing this browser does not stop a detached run."
)
