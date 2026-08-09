"""Prepare datasets and run an explicit, leakage-safe single-symbol PPO workflow."""

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_dashboard_dataset
from dashboard.ppo_workflow import (
    CPU_MPS_SPEED_MULTIPLIER,
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    DEVICE_OPTIONS,
    DEVICE_RECOMMENDATION,
    TIMESTEP_PRESETS,
    build_ready_symbol_catalog,
    build_workflow_identity,
    claim_workflow_job,
    initialize_workflow_session,
    mark_candidate_persisted,
    persistence_availability,
    pilot_readiness_table,
    preview_candidate_version,
    registry_history_table,
    release_workflow_job,
    reset_workflow_results,
    run_persistence_action,
    run_training_action,
    run_validation_action,
    selected_symbol_summary,
    sync_workflow_identity,
    training_availability,
    validation_availability,
    validation_chart_frames,
    validation_metrics_table,
)
from dashboard.presentation import (
    enum_display_options,
    format_date,
    format_datetime,
    format_integer,
    format_symbol_company,
    safe_display_value,
    selected_option_values,
    status_label,
)
from dashboard.registry_loader import load_company_registry
from data_pipeline.src.config import (
    AI_MINIMUM_USABLE_ROWS,
    PROCESSED_MASTER_PATH,
    PROCESSED_SPLITS_DIR,
    PROCESSED_SYMBOLS_DIR,
)
from feature_engineering.dataset_builder import (
    build_master_ai_dataset,
    build_symbol_datasets,
    validate_ai_dataset,
)
from feature_engineering.schemas import FEATURE_VERSION
from feature_engineering.splitting import create_master_split, create_symbol_split
from reinforcement_learning.data_contract import (
    RL_PARTITION_SCHEMA_VERSION,
    load_rl_partition,
)
from reinforcement_learning.model_management.registry import (
    MODEL_REGISTRY_SCHEMA_VERSION,
    ModelRegistryError,
    empty_model_registry,
    load_model_registry,
)
from reinforcement_learning.model_management.persistence import (
    PPOPersistenceError,
    RegistryCommitPendingError,
)
from reinforcement_learning.model_management.selection import (
    bulk_select_symbols,
    filter_symbol_status,
    normalize_symbol_selection,
    select_all_active_eligible,
    select_needing_retraining,
    select_never_trained,
    select_newly_added_eligible,
    select_visible_symbols,
    selected_symbols_from_editor,
    symbol_selection_counts,
    update_visible_symbol_selection,
)
from reinforcement_learning.model_management.status import (
    build_model_readiness_table,
    master_model_status,
)
from reinforcement_learning.environments import (
    ENVIRONMENT_VERSION,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.environments.validation import validate_environment
from reinforcement_learning.training.config import PPO_CONFIG_VERSION, PPOConfig
from reinforcement_learning.training.devices import TorchDeviceError, resolve_torch_device


SELECTION_KEY = "training_selected_symbols"
SELECTION_WIDGET_KEY = "training_selected_symbols_widget"
EDITOR_REVISION_KEY = "training_symbol_editor_revision"


def _write_selected_symbols(
    symbols: tuple[str, ...],
    all_symbols: tuple[str, ...],
) -> None:
    """Update canonical and multiselect state before widgets are rendered."""
    normalized = normalize_symbol_selection(symbols, allowed_symbols=all_symbols)
    st.session_state[SELECTION_KEY] = list(normalized)
    st.session_state[SELECTION_WIDGET_KEY] = list(normalized)
    st.session_state[EDITOR_REVISION_KEY] = (
        int(st.session_state.get(EDITOR_REVISION_KEY, 0)) + 1
    )


def _add_selected_symbols(
    symbols: tuple[str, ...],
    all_symbols: tuple[str, ...],
) -> None:
    """Add a bulk group without losing selections hidden by current filters."""
    selected = bulk_select_symbols(
        symbols,
        all_symbols=all_symbols,
        current_selection=st.session_state.get(SELECTION_KEY, ()),
    )
    _write_selected_symbols(selected, all_symbols)


def _sync_multiselect_selection(all_symbols: tuple[str, ...]) -> None:
    """Make the multiselect value the canonical symbol selection."""
    selected = normalize_symbol_selection(
        st.session_state.get(SELECTION_WIDGET_KEY, ()),
        allowed_symbols=all_symbols,
    )
    st.session_state[SELECTION_KEY] = list(selected)
    st.session_state[EDITOR_REVISION_KEY] = (
        int(st.session_state.get(EDITOR_REVISION_KEY, 0)) + 1
    )


def _load_processed_master() -> tuple[pd.DataFrame, str | None]:
    path = Path(PROCESSED_MASTER_PATH)
    if not path.is_file():
        return pd.DataFrame(), None
    try:
        return pd.read_csv(path, dtype={"symbol": "string"}), None
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return pd.DataFrame(), f"Could not read processed master dataset: {exc}"


def _processed_symbol_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Return a bounded cache key that changes when processed files change."""
    values: list[tuple[str, int, int]] = []
    for path in sorted(Path(PROCESSED_SYMBOLS_DIR).glob("*.csv")):
        try:
            details = path.stat()
        except OSError:
            continue
        values.append((path.name, details.st_mtime_ns, details.st_size))
    return tuple(values)


def _rl_contract_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Invalidate cached readiness when contract/scaler metadata changes."""
    values: list[tuple[str, int, int]] = []
    root = Path(PROCESSED_SPLITS_DIR) / "symbols"
    metadata_names = (
        "rl_contract.json",
        "metadata.json",
        "rl_observation_scaler.json",
        "rl_observation_scaler.joblib",
    )
    partition_names = tuple(
        filename
        for partition in ("train", "validation", "test")
        for filename in (f"{partition}.csv", f"{partition}_rl.csv")
    )
    for name in (*metadata_names, *partition_names):
        for path in sorted(root.glob(f"*/{name}")):
            try:
                details = path.stat()
            except OSError:
                continue
            values.append(
                (str(path.relative_to(root)), details.st_mtime_ns, details.st_size)
            )
    return tuple(values)


@st.cache_data(max_entries=8, show_spinner=False)
def _validate_rl_environment(
    symbol: str,
    contract_sha256: str,
    scaler_sha256: str,
    scaler_metadata_sha256: str,
    train_validation_fingerprint: str,
    splits_dir_text: str,
) -> dict[str, object]:
    """Validate one selected canonical TRAIN environment on explicit request."""
    del (
        contract_sha256,
        scaler_sha256,
        scaler_metadata_sha256,
        train_validation_fingerprint,
    )
    try:
        loaded = load_rl_partition(
            symbol,
            "train",
            splits_dir=Path(splits_dir_text),
        )
        environment = SingleSymbolTradingEnv(loaded.data)
        try:
            result = validate_environment(environment)
        finally:
            environment.close()
        return {
            "status": "Environment Ready" if result.valid else "Validation Failed",
            "message": (
                f"{symbol} canonical TRAIN environment is ready"
                if result.valid
                else "; ".join(result.errors)
            ),
            "shape": result.observation_shape,
        }
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        return {
            "status": "Validation Failed",
            "message": str(exc),
            "shape": None,
        }


@st.cache_data(ttl="5m", max_entries=8, show_spinner=False)
def _cached_model_readiness(
    market: pd.DataFrame,
    registry: pd.DataFrame,
    models: pd.DataFrame,
    minimum_usable_rows: int,
    processed_fingerprint: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    """Cache expensive feature readiness while retaining file invalidation."""
    del processed_fingerprint
    return build_model_readiness_table(
        market,
        registry,
        models,
        minimum_usable_rows=minimum_usable_rows,
    )


@st.cache_data(ttl="5m", max_entries=8, show_spinner=False)
def _cached_rl_ready_catalog(
    status: pd.DataFrame,
    contract_fingerprint: tuple[tuple[str, int, int], ...],
):
    """Cache metadata-only contract validation with file invalidation."""
    del contract_fingerprint
    return build_ready_symbol_catalog(status, splits_dir=Path(PROCESSED_SPLITS_DIR))


st.title("Training & Model Management")
st.caption(
    "Prepare leakage-safe AI datasets, train one in-memory PPO candidate, compare "
    "it on VALIDATION, and explicitly save validation-passing candidates."
)

market_result = load_dashboard_dataset()
registry_result = load_company_registry()
if market_result.data.empty:
    st.info("No master market history is available. Fetch and rebuild data first.")
    st.stop()
if not registry_result.available:
    for error in registry_result.errors:
        st.error(error)
    st.info("Build the Company Registry before preparing model datasets.")
    st.stop()

try:
    model_registry = load_model_registry()
except ModelRegistryError as exc:
    st.error(str(exc))
    model_registry = empty_model_registry()

processed_master, processed_error = _load_processed_master()
if processed_error:
    st.warning(processed_error)
raw_dates = pd.to_datetime(market_result.data["date"], errors="coerce").dropna()
processed_dates = (
    pd.to_datetime(processed_master["date"], errors="coerce").dropna()
    if "date" in processed_master
    else pd.Series(dtype="datetime64[ns]")
)
st.session_state.setdefault("training_minimum_history", AI_MINIMUM_USABLE_ROWS)
minimum_history = int(st.session_state["training_minimum_history"])
status_table = _cached_model_readiness(
    market_result.data,
    registry_result.data,
    model_registry,
    minimum_history,
    _processed_symbol_fingerprint(),
)

rl_catalog = _cached_rl_ready_catalog(status_table, _rl_contract_fingerprint())
workflow_state = initialize_workflow_session(st.session_state)
ready_symbols = rl_catalog.ready_symbols
insufficient_history_count = int(
    status_table["readiness_status"].eq("Insufficient History").sum()
)

st.subheader("A. RL readiness")
with st.container(horizontal=True):
    st.metric("Environment", ENVIRONMENT_VERSION, border=True)
    st.metric("RL contract", RL_PARTITION_SCHEMA_VERSION, border=True)
    st.metric("PPO trainer", PPO_CONFIG_VERSION, border=True)
    st.metric("Training-ready", format_integer(len(ready_symbols)), border=True)
    st.metric(
        "Insufficient history",
        format_integer(insufficient_history_count),
        border=True,
    )
    st.metric("Registered models", format_integer(len(model_registry)), border=True)
if ready_symbols:
    st.success(
        f"Single-symbol PPO research is ready for {len(ready_symbols):,} securities. "
        f"The registry contains {len(model_registry):,} persisted model record(s)."
    )
else:
    st.error(
        "No symbol has both current training readiness and a compatible RL contract."
    )
st.info(
    f"{DEVICE_RECOMMENDATION}. In the controlled MCB 5,120-step benchmark, "
    f"CPU was approximately {CPU_MPS_SPEED_MULTIPLIER:.2f}x faster than MPS."
)
with st.expander(
    "Pilot universe for Milestone 5B-6",
    icon=":material/groups:",
):
    st.caption("Read-only preparation. This page does not offer Train All.")
    st.dataframe(
        pilot_readiness_table(
            ready_symbols,
            rejected_reasons=rl_catalog.rejected_reasons,
        ),
        hide_index=True,
        width="stretch",
    )

selected_summary = None
workflow_identity = None
device_resolution = None
device_error = None

if ready_symbols:
    company_names = {
        str(row["symbol"]): row.get("company_name", "")
        for _, row in status_table.iterrows()
    }
    default_symbol = "OGDC" if "OGDC" in ready_symbols else ready_symbols[0]
    previous_symbol = st.session_state.get("ppo_training_symbol")
    if previous_symbol not in ready_symbols:
        if previous_symbol:
            reason = rl_catalog.rejected_reasons.get(
                str(previous_symbol),
                "The symbol is no longer in the current ready universe.",
            )
            st.warning(
                f"{previous_symbol} is no longer selectable: {reason} "
                f"Selection was reset to {default_symbol}."
            )
        st.session_state["ppo_training_symbol"] = default_symbol
    st.session_state.setdefault("ppo_training_timesteps", TIMESTEP_PRESETS[0])
    st.session_state.setdefault("ppo_training_seed", DEFAULT_SEED)
    st.session_state.setdefault("ppo_training_device", DEFAULT_DEVICE)
    controls_disabled = workflow_state.get("job_phase") != "idle"

    st.subheader("B. Training configuration")
    with st.container(border=True):
        selected_symbol = st.selectbox(
            "Training-ready symbol",
            ready_symbols,
            format_func=lambda symbol: format_symbol_company(
                symbol,
                company_names.get(symbol),
            ),
            key="ppo_training_symbol",
            disabled=controls_disabled,
            persist_state="session",
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            requested_timesteps = st.segmented_control(
                "Timesteps",
                TIMESTEP_PRESETS,
                format_func=lambda value: f"{int(value):,}",
                key="ppo_training_timesteps",
                required=True,
                disabled=controls_disabled,
                persist_state="session",
            )
            requested_seed = st.number_input(
                "Deterministic seed",
                min_value=0,
                step=1,
                key="ppo_training_seed",
                disabled=controls_disabled,
                persist_state="session",
            )
            requested_device = st.segmented_control(
                "Requested device",
                DEVICE_OPTIONS,
                format_func=lambda value: {
                    "cpu": "CPU · Recommended",
                    "mps": "MPS",
                    "auto": "AUTO",
                }[value],
                key="ppo_training_device",
                required=True,
                disabled=controls_disabled,
                persist_state="session",
            )

        selected_summary = selected_symbol_summary(rl_catalog, selected_symbol)
        workflow_identity = build_workflow_identity(
            selected_summary,
            int(requested_timesteps),
            int(requested_seed),
            str(requested_device),
        )
        sync_workflow_identity(st.session_state, workflow_identity)
        workflow_state = initialize_workflow_session(st.session_state)
        try:
            device_resolution = resolve_torch_device(
                workflow_identity.requested_device
            )
        except TorchDeviceError as exc:
            device_error = str(exc)
        if device_error:
            st.error(f"Requested device is unavailable: {device_error}")
        elif device_resolution is not None:
            st.caption(
                f"Requested device: {device_resolution.requested_device.upper()} · "
                f"Preflight resolution: {device_resolution.resolved_device.upper()}. "
                "The trainer will verify the actual model and policy device."
            )

        with st.expander(
            "Read-only PPO defaults",
            icon=":material/tune:",
        ):
            ppo_defaults = PPOConfig().to_dict()
            default_labels = {
                "policy": "Policy",
                "learning_rate": "Learning rate",
                "n_steps": "Rollout steps",
                "batch_size": "Batch size",
                "n_epochs": "Epochs",
                "gamma": "Gamma",
                "gae_lambda": "GAE lambda",
                "clip_range": "Clip range",
                "ent_coef": "Entropy coefficient",
                "vf_coef": "Value-function coefficient",
                "max_grad_norm": "Maximum gradient norm",
            }
            st.dataframe(
                pd.DataFrame(
                    {
                        "Parameter": list(default_labels.values()),
                        "Value": [
                            safe_display_value(ppo_defaults[key])
                            for key in default_labels
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption("Fixed for 5B-5. No tuning or Optuna controls are exposed.")

    st.subheader("C. Selected symbol data summary")
    with st.container(border=True):
        partition_summary = pd.DataFrame(
            {
                "Partition": ["TRAIN", "VALIDATION", "TEST · SEALED"],
                "Rows": [
                    selected_summary.train_rows,
                    selected_summary.validation_rows,
                    selected_summary.test_rows,
                ],
                "Start": [
                    format_date(selected_summary.train_start),
                    format_date(selected_summary.validation_start),
                    format_date(selected_summary.test_start),
                ],
                "End": [
                    format_date(selected_summary.train_end),
                    format_date(selected_summary.validation_end),
                    format_date(selected_summary.test_end),
                ],
                "Permitted use": [
                    "PPO learning only",
                    "Candidate evaluation only",
                    "Metadata display only",
                ],
            }
        )
        st.dataframe(partition_summary, hide_index=True, width="stretch")
        st.success(
            "FINAL TEST SET: SEALED — row count and dates above come from "
            "rl_contract.json metadata. The TEST frame is not loaded."
        )
        with st.container(horizontal=True):
            st.metric("Observation shape", str(selected_summary.observation_shape), border=True)
            st.metric("Feature version", selected_summary.feature_version, border=True)
            st.metric("RL contract", selected_summary.rl_contract_version, border=True)
            st.metric("Environment", selected_summary.environment_version, border=True)

        validation_key = (
            workflow_identity.symbol,
            workflow_identity.contract_sha256,
            workflow_identity.observation_scaler_sha256,
            workflow_identity.observation_scaler_metadata_sha256,
            workflow_identity.train_validation_artifact_fingerprint,
        )
        environment_validation = st.session_state.get(
            "ppo_selected_environment_validation"
        )
        if st.button(
            "Validate selected TRAIN environment",
            icon=":material/fact_check:",
            key="validate_selected_ppo_environment",
            disabled=controls_disabled,
        ):
            result = _validate_rl_environment(
                workflow_identity.symbol,
                workflow_identity.contract_sha256,
                workflow_identity.observation_scaler_sha256,
                workflow_identity.observation_scaler_metadata_sha256,
                workflow_identity.train_validation_artifact_fingerprint,
                str(PROCESSED_SPLITS_DIR),
            )
            st.session_state["ppo_selected_environment_validation"] = {
                "key": validation_key,
                "result": result,
            }
            environment_validation = st.session_state[
                "ppo_selected_environment_validation"
            ]
        if (
            isinstance(environment_validation, dict)
            and environment_validation.get("key") == validation_key
        ):
            result = environment_validation["result"]
            if result["status"] == "Environment Ready":
                st.success(
                    f"{result['message']} · Observation shape: {result['shape']}"
                )
            else:
                st.error(result["message"])

    st.subheader("D. Train PPO candidate")
    with st.container(border=True):
        st.markdown(
            f"**{workflow_identity.symbol}** · TRAIN "
            f"{format_date(selected_summary.train_start)} to "
            f"{format_date(selected_summary.train_end)} · "
            f"{selected_summary.train_rows:,} rows · "
            f"{workflow_identity.requested_timesteps:,} requested timesteps · "
            f"seed {workflow_identity.seed} · "
            f"{workflow_identity.requested_device.upper()}"
        )
        train_gate = training_availability(st.session_state, workflow_identity)
        train_disabled = not train_gate.allowed or device_error is not None
        train_clicked = st.button(
            "Train PPO candidate",
            type="primary",
            icon=":material/model_training:",
            key="train_single_ppo_candidate",
            disabled=train_disabled,
        )
        st.caption(
            "Training is synchronous and session-scoped. Streamlit cannot guarantee "
            "interactive cancellation after the page process is blocked, so no "
            "misleading cancel control is shown."
        )
        if not train_gate.allowed:
            st.caption(train_gate.reason)

        if train_clicked:
            claim = claim_workflow_job(
                st.session_state,
                workflow_identity,
                "training",
            )
            if not claim.allowed:
                st.error(claim.reason)
            else:
                workflow_state["training_error"] = None
                progress_bar = st.progress(0, text="Preparing canonical TRAIN data…")

                def _show_training_progress(event) -> bool:
                    percent = max(0, min(100, int(round(event.progress_percent))))
                    progress_bar.progress(
                        percent,
                        text=(
                            f"{event.symbol}: {event.current_timesteps:,} / "
                            f"{event.requested_timesteps:,} timesteps · {event.phase}"
                        ),
                    )
                    workflow_state["progress"] = event
                    return True

                try:
                    with st.status(
                        "Training one in-memory PPO candidate…",
                        expanded=True,
                    ) as training_status:
                        training_result = run_training_action(
                            workflow_identity,
                            progress_callback=_show_training_progress,
                        )
                        workflow_state["training_result"] = training_result
                        workflow_state["validation_result"] = None
                        workflow_state["persisted_bundle"] = None
                        workflow_state["persisted_candidate_key"] = None
                        if training_result.status == "completed":
                            training_status.update(
                                label="PPO training completed in memory",
                                state="complete",
                            )
                        elif training_result.status == "interrupted":
                            training_status.update(
                                label="PPO training was interrupted",
                                state="error",
                            )
                        else:
                            training_status.update(
                                label="PPO training failed safely",
                                state="error",
                            )
                except Exception as exc:
                    workflow_state["training_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    release_workflow_job(st.session_state)

        training_result = workflow_state.get("training_result")
        if workflow_state.get("training_error"):
            st.error(f"Training error: {workflow_state['training_error']}")
        if training_result is not None:
            if training_result.status == "completed":
                st.success(
                    "Training completed. This is an in-memory research candidate, "
                    "not a profitability claim or production model."
                )
            elif training_result.status == "interrupted":
                st.warning(training_result.message)
            else:
                st.error(
                    f"{training_result.message} "
                    f"{training_result.error or ''}".strip()
                )
            with st.container(horizontal=True):
                st.metric("Status", status_label(training_result.status), border=True)
                st.metric(
                    "Timesteps",
                    f"{training_result.actual_timesteps:,} actual / "
                    f"{training_result.requested_timesteps:,} requested",
                    border=True,
                )
                st.metric(
                    "Duration",
                    f"{training_result.duration_seconds:,.2f} seconds",
                    border=True,
                )
                st.metric("TRAIN rows", format_integer(training_result.training_rows), border=True)
            st.caption(
                f"TRAIN {format_date(training_result.training_start)} to "
                f"{format_date(training_result.training_end)} · seed "
                f"{training_result.seed} · requested device "
                f"{training_result.requested_device.upper()} · resolved "
                f"{safe_display_value(training_result.resolved_device).upper()} · "
                f"actual {safe_display_value(training_result.device).upper()} · "
                f"environment {training_result.environment_version} · feature "
                f"{training_result.feature_version} · contract "
                f"{training_result.rl_contract_version}"
            )

    st.subheader("E. Validation results")
    with st.container(border=True):
        validation_gate = validation_availability(
            st.session_state,
            workflow_identity,
        )
        validation_clicked = st.button(
            "Evaluate on validation",
            icon=":material/analytics:",
            key="evaluate_ppo_on_validation",
            disabled=not validation_gate.allowed,
        )
        if not validation_gate.allowed:
            st.caption(validation_gate.reason)
        if validation_clicked:
            claim = claim_workflow_job(
                st.session_state,
                workflow_identity,
                "validating",
            )
            if not claim.allowed:
                st.error(claim.reason)
            else:
                workflow_state["validation_error"] = None
                try:
                    with st.status(
                        "Comparing PPO and baselines on VALIDATION…",
                        expanded=True,
                    ) as validation_status:
                        comparison = run_validation_action(
                            workflow_state["training_result"],
                            workflow_identity,
                        )
                        workflow_state["validation_result"] = comparison
                        validation_status.update(
                            label="VALIDATION comparison completed",
                            state="complete",
                        )
                except Exception as exc:
                    workflow_state["validation_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    release_workflow_job(st.session_state)

        comparison = workflow_state.get("validation_result")
        if workflow_state.get("validation_error"):
            st.error(f"Validation error: {workflow_state['validation_error']}")
        if comparison is not None:
            st.caption(
                f"VALIDATION only · {comparison.validation_rows:,} rows · "
                f"{format_date(comparison.validation_start)} to "
                f"{format_date(comparison.validation_end)} · deterministic PPO · "
                "Random seed 42"
            )
            st.dataframe(
                validation_metrics_table(comparison),
                hide_index=True,
                width="stretch",
            )
            portfolio_history, drawdown_history = validation_chart_frames(comparison)
            st.markdown("**Validation portfolio value**")
            st.line_chart(
                portfolio_history,
                x="Date",
                y="Portfolio Value",
                color="Strategy",
            )
            st.markdown("**Validation drawdown**")
            st.line_chart(
                drawdown_history,
                x="Date",
                y="Drawdown",
                color="Strategy",
            )
            decision = comparison.candidate_decision
            decision_message = (
                f"{status_label(decision.status)} · "
                f"criteria {decision.criteria_version}"
            )
            if decision.status == "validation_pass":
                st.success(decision_message)
            elif decision.status == "validation_fail":
                st.warning(decision_message)
            elif decision.status == "insufficient_validation_data":
                st.info(decision_message)
            else:
                st.error(decision_message)
            st.markdown("**Decision reasons**")
            for reason in decision.reasons:
                st.markdown(f"- {reason}")
            if comparison.warnings:
                with st.expander("Validation metric warnings"):
                    for warning in comparison.warnings:
                        st.markdown(f"- {warning}")
            st.info(
                "A validation pass does not equal production promotion. A validation "
                "failure is an analytical decision, not evidence that the pipeline is broken."
            )

    st.subheader("F. Candidate persistence")
    with st.container(border=True):
        comparison = workflow_state.get("validation_result")
        persisted_bundle = workflow_state.get("persisted_bundle")
        if persisted_bundle is not None:
            st.success(
                f"Candidate already saved as {persisted_bundle.model_id} "
                f"(version {persisted_bundle.model_version})."
            )
            st.caption(
                f"Validation: {status_label(persisted_bundle.validation_status)} · "
                f"lifecycle: {status_label(persisted_bundle.model_status)} · "
                f"promotion: {status_label(persisted_bundle.promotion_status)}"
            )
        elif comparison is None:
            st.info("Complete validation before candidate persistence is considered.")
        elif comparison.candidate_decision.status != "validation_pass":
            st.info(
                "Production candidate saving is unavailable because this result did "
                f"not receive validation_pass ({comparison.candidate_decision.status})."
            )
        else:
            try:
                preview = preview_candidate_version(
                    model_registry,
                    workflow_identity.symbol,
                )
                st.write(
                    f"Expected next identity: **{preview.model_id}** · version "
                    f"**{preview.model_version}**. Final allocation occurs atomically "
                    "when Save candidate is pressed."
                )
            except (ValueError, PPOPersistenceError) as exc:
                preview = None
                st.error(f"Candidate version preview failed: {exc}")
            persistence_gate = persistence_availability(
                st.session_state,
                workflow_identity,
            )
            if st.button(
                "Save candidate",
                type="primary",
                icon=":material/save:",
                key="persist_ppo_candidate",
                disabled=not persistence_gate.allowed or preview is None,
            ):
                claim = claim_workflow_job(
                    st.session_state,
                    workflow_identity,
                    "persisting",
                )
                if not claim.allowed:
                    st.error(claim.reason)
                else:
                    workflow_state["persistence_error"] = None
                    try:
                        bundle = run_persistence_action(
                            workflow_state["training_result"],
                            comparison,
                            workflow_identity,
                        )
                        mark_candidate_persisted(
                            st.session_state,
                            workflow_identity,
                            comparison,
                            bundle,
                        )
                    except RegistryCommitPendingError as exc:
                        workflow_state["persistence_error"] = str(exc)
                    except (PPOPersistenceError, OSError, ValueError) as exc:
                        workflow_state["persistence_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    finally:
                        release_workflow_job(st.session_state)
                    if workflow_state.get("persisted_bundle") is not None:
                        st.rerun()
            if workflow_state.get("persistence_error"):
                st.error(
                    "Candidate persistence did not complete safely: "
                    f"{workflow_state['persistence_error']}"
                )
        st.caption(
            "No Promote to Production action is available. A saved model remains a "
            "candidate until a later, separately authorized promotion workflow."
        )

    if any(
        workflow_state.get(key) is not None
        for key in ("training_result", "validation_result", "persisted_bundle")
    ):
        if st.button(
            "Discard current in-memory workflow",
            icon=":material/delete_sweep:",
            key="discard_current_ppo_workflow",
        ):
            reset_workflow_results(st.session_state)
            st.rerun()

st.subheader("G. Model registry / history")
model_history = registry_history_table(model_registry)
st.caption(
    f"Registry schema: {MODEL_REGISTRY_SCHEMA_VERSION} · "
    "candidate history only; production promotion is not enabled."
)
if model_history.empty:
    st.info("No PPO model versions are registered yet.")
else:
    st.dataframe(model_history, hide_index=True, width="stretch")

st.divider()
st.header("Dataset preparation and broader model readiness")
st.caption(
    "The existing dataset, selection, and master-model preparation tools remain "
    "available below. Bulk and master PPO training stay disabled."
)

st.subheader("Dataset readiness")
minimum_history = int(
    st.number_input(
        "Minimum usable rows for symbol eligibility",
        min_value=1,
        step=1,
        help=(
            "The initial 252-row gate approximates one trading year after indicator "
            "warm-up. It is configurable and does not guarantee PPO sufficiency."
        ),
        key="training_minimum_history",
    )
)
split_metadata_paths = tuple(Path(PROCESSED_SPLITS_DIR).glob("**/metadata.json"))
with st.container(horizontal=True):
    st.metric(
        "Master latest date",
        format_date(raw_dates.max() if not raw_dates.empty else None),
        border=True,
    )
    st.metric(
        "Processed latest date",
        format_date(processed_dates.max() if not processed_dates.empty else None),
        border=True,
    )
    st.metric(
        "Total symbols",
        format_integer(market_result.data["symbol"].nunique()),
        border=True,
    )
    st.metric("Active symbols", format_integer(len(status_table)), border=True)
    st.metric(
        "Eligible symbol models",
        format_integer(status_table["eligible"].sum()),
        border=True,
    )
    st.metric(
        "Insufficient history",
        format_integer(
            (status_table["readiness_status"] == "Insufficient History").sum()
        ),
        border=True,
    )
st.caption(
    f"Feature version: {FEATURE_VERSION} · Splits: "
    f"{'Available' if split_metadata_paths else 'Not created'}"
)

with st.container(horizontal=True):
    if st.button(
        "Build/Refresh AI Datasets",
        icon=":material/build:",
        width="stretch",
    ):
        with st.spinner("Calculating isolated symbol features..."):
            try:
                symbol_metrics = build_symbol_datasets(
                    minimum_usable_rows=int(minimum_history)
                )
                master_metrics = build_master_ai_dataset()
                _cached_model_readiness.clear()
                st.success(
                    "AI datasets refreshed: "
                    f"{symbol_metrics.output_rows:,} symbol rows and "
                    f"{master_metrics.output_rows:,} master rows."
                )
            except (OSError, ValueError) as exc:
                st.error(f"AI dataset build failed: {exc}")
    if st.button(
        "Validate AI Datasets",
        icon=":material/fact_check:",
        width="stretch",
    ):
        paths = [Path(PROCESSED_MASTER_PATH), *sorted(Path(PROCESSED_SYMBOLS_DIR).glob("*.csv"))]
        existing = [path for path in paths if path.is_file()]
        if not existing:
            st.info("No processed AI datasets exist yet.")
        else:
            invalid = [path for path in existing if not validate_ai_dataset(path).valid]
            if invalid:
                st.error(f"Validation failed for {len(invalid)} dataset(s).")
            else:
                st.success(f"Validated {len(existing)} processed dataset(s).")
    if st.button(
        "Create/Refresh Chronological Splits",
        icon=":material/call_split:",
        width="stretch",
    ):
        with st.spinner("Creating chronological splits and training-only scalers..."):
            try:
                created = 0
                if Path(PROCESSED_MASTER_PATH).is_file():
                    create_master_split()
                    created += 1
                for path in sorted(Path(PROCESSED_SYMBOLS_DIR).glob("*.csv")):
                    create_symbol_split(path.stem)
                    created += 1
                if created:
                    _cached_model_readiness.clear()
                    st.success(f"Created or refreshed {created} split set(s).")
                else:
                    st.info("Build processed datasets before creating splits.")
            except (OSError, ValueError) as exc:
                st.error(f"Split creation failed: {exc}")

st.subheader("Symbol model management")
training_options = enum_display_options(status_table["training_status"])
security_options = enum_display_options(status_table["security_type"])
sector_options = tuple(
    sorted(
        {
            str(value).strip()
            for value in status_table["sector"].dropna()
            if str(value).strip()
        },
        key=str.casefold,
    )
)
with st.expander("Symbol filters", icon=":material/filter_list:"):
    symbol_search = st.text_input(
        "Search by symbol or company name",
        key="training_symbol_search",
    )
    selected_training_statuses = st.multiselect(
        "Training status",
        training_options,
        format_func=lambda option: option.label,
        key="training_status_filters",
    )
    selected_sectors = st.multiselect(
        "Sector",
        sector_options,
        key="training_sector_filters",
    )
    selected_security_types = st.multiselect(
        "Security type",
        security_options,
        format_func=lambda option: option.label,
        key="training_security_filters",
    )
    newly_added_only = st.toggle(
        "Newly added only",
        key="training_newly_added_only",
    )

visible_status = filter_symbol_status(
    status_table,
    search=symbol_search,
    training_statuses=selected_option_values(selected_training_statuses),
    sectors=selected_sectors,
    security_types=selected_option_values(selected_security_types),
    newly_added_only=newly_added_only,
)
all_symbols = tuple(dict.fromkeys(status_table["symbol"].astype(str)))
current_selection = normalize_symbol_selection(
    st.session_state.get(SELECTION_KEY, ()),
    allowed_symbols=all_symbols,
)
st.session_state[SELECTION_KEY] = list(current_selection)
widget_selection = normalize_symbol_selection(
    st.session_state.get(SELECTION_WIDGET_KEY, current_selection),
    allowed_symbols=all_symbols,
)
if widget_selection != current_selection or SELECTION_WIDGET_KEY not in st.session_state:
    st.session_state[SELECTION_WIDGET_KEY] = list(current_selection)
st.session_state.setdefault(EDITOR_REVISION_KEY, 0)
with st.container(horizontal=True):
    st.button(
        "Select All Visible",
        on_click=_add_selected_symbols,
        args=(select_visible_symbols(visible_status), all_symbols),
        key="training_select_all_visible",
    )
    st.button(
        "Clear Selection",
        on_click=_write_selected_symbols,
        args=((), all_symbols),
        key="training_clear_selection",
    )
    st.button(
        "Select Never Trained",
        on_click=_add_selected_symbols,
        args=(select_never_trained(status_table), all_symbols),
        key="training_select_never_trained",
    )
    st.button(
        "Select Needs Retraining",
        on_click=_add_selected_symbols,
        args=(select_needing_retraining(status_table), all_symbols),
        key="training_select_needs_retraining",
    )
    st.button(
        "Select Newly Added",
        on_click=_add_selected_symbols,
        args=(select_newly_added_eligible(status_table), all_symbols),
        key="training_select_newly_added",
    )
    st.button(
        "Select All Active Eligible",
        on_click=_add_selected_symbols,
        args=(select_all_active_eligible(status_table), all_symbols),
        key="training_select_all_active_eligible",
    )

st.multiselect(
    "Selected securities",
    all_symbols,
    format_func=lambda symbol: format_symbol_company(
        symbol,
        status_table.loc[
            status_table["symbol"].astype("string") == symbol,
            "company_name",
        ].iloc[0],
    ),
    key=SELECTION_WIDGET_KEY,
    on_change=_sync_multiselect_selection,
    args=(all_symbols,),
    placeholder="Search and select one or more securities",
)
current_selection = normalize_symbol_selection(
    st.session_state.get(SELECTION_KEY, ()),
    allowed_symbols=all_symbols,
)
selected_set = set(current_selection)
symbol_display = pd.DataFrame(
    {
        "selected": visible_status["symbol"].astype(str).isin(selected_set),
        "symbol": visible_status["symbol"].astype(str),
        "Company Name": visible_status["company_name"].map(safe_display_value),
        "Sector": visible_status["sector"].map(safe_display_value),
        "Data Start": visible_status["data_start"].map(format_date),
        "Data End": visible_status["data_end"].map(format_date),
        "Raw Rows": visible_status["raw_trading_rows"].map(format_integer),
        "Invalid OHLC Rows Removed": visible_status[
            "invalid_ohlc_rows_removed"
        ].map(format_integer),
        "Valid Rows Before Features": visible_status[
            "usable_pre_feature_rows"
        ].map(format_integer),
        "Quality Retention": visible_status["quality_retention_percent"].map(
            lambda value: f"{float(value):.2f}%" if pd.notna(value) else "—"
        ),
        "Quality Note": visible_status["quality_removal_reason"].map(
            lambda value: "Invalid OHLC rows removed" if value else "No removals"
        ),
        "Warm-up Rows": visible_status["warmup_rows_removed"].map(format_integer),
        "Rows After Warm-up": visible_status["post_warmup_rows"].map(format_integer),
        "Usable Rows": visible_status["usable_rows"].map(format_integer),
        "First Usable Date": visible_status["first_usable_date"].map(format_date),
        "Last Usable Date": visible_status["last_usable_date"].map(format_date),
        "Additional Rows": visible_status["additional_rows_required"].map(
            format_integer
        ),
        "Train Rows": visible_status["train_rows"].map(format_integer),
        "Validation Rows": visible_status["validation_rows"].map(format_integer),
        "Test Rows": visible_status["test_rows"].map(format_integer),
        "Dataset Readiness": visible_status["readiness_status"],
        "Model Version": visible_status["model_version"].map(format_integer),
        "Last Trained": visible_status["last_trained_at"].map(format_datetime),
        "Training Data Start": visible_status["training_data_start"].map(format_date),
        "Training Data End": visible_status["training_data_end"].map(format_date),
        "New Trading Days": visible_status["new_data_days"].map(format_integer),
        "Training Status": visible_status["training_status"].map(status_label),
    }
)
symbol_display = symbol_display.set_index("symbol", drop=False)
symbol_display.index.name = "symbol_identity"
visible_selected_count, total_selected_count = symbol_selection_counts(
    current_selection,
    visible_status["symbol"].astype(str),
)
st.caption(
    f"Showing {len(visible_status):,} active securities · "
    f"{visible_selected_count:,} visible selected · "
    f"{total_selected_count:,} total selected"
)
selection_noun = "security" if total_selected_count == 1 else "securities"
st.write(f"{total_selected_count:,} {selection_noun} selected")
edited_table = st.data_editor(
    symbol_display,
    width="stretch",
    hide_index=True,
    height=500,
    num_rows="fixed",
    disabled=[column for column in symbol_display.columns if column != "selected"],
    column_config={
        "selected": st.column_config.CheckboxColumn(
            "Select",
            help="Select this security for dataset preparation or future training.",
        ),
        "symbol": st.column_config.TextColumn("Symbol", pinned=True),
    },
    key=f"training_symbol_editor_{st.session_state[EDITOR_REVISION_KEY]}",
)
edited_selection = selected_symbols_from_editor(edited_table)
updated_selection = update_visible_symbol_selection(
    current_selection,
    visible_status["symbol"].astype(str),
    edited_selection,
    all_symbols=all_symbols,
)
if updated_selection != current_selection:
    st.session_state[SELECTION_KEY] = list(updated_selection)
    st.session_state[EDITOR_REVISION_KEY] += 1
    st.rerun()

selected_symbols = list(
    normalize_symbol_selection(
        st.session_state.get(SELECTION_KEY, ()),
        allowed_symbols=all_symbols,
    )
)
with st.container(horizontal=True):
    if st.button(
        "Prepare Selected Datasets",
        disabled=not selected_symbols,
        icon=":material/dataset:",
    ):
        with st.spinner("Preparing complete selected-symbol histories..."):
            try:
                metrics = build_symbol_datasets(
                    symbols=selected_symbols,
                    minimum_usable_rows=int(minimum_history),
                )
                _cached_model_readiness.clear()
                if metrics.output_paths:
                    st.success(f"Prepared {len(metrics.output_paths)} symbol dataset(s).")
                else:
                    st.warning(
                        "No selected symbol met the current history and quality gates."
                    )
            except (OSError, ValueError) as exc:
                st.error(f"Selected dataset preparation failed: {exc}")
    st.button(
        "Train Selected Models",
        disabled=True,
        help="Bulk PPO training is intentionally deferred to Milestone 5B-6.",
        icon=":material/model_training:",
    )
st.info(
    "Bulk training remains disabled. Use the explicit one-symbol PPO workflow "
    "above for a single research candidate."
)

st.subheader("Master model")
master_status = master_model_status(processed_master, model_registry)
st.caption("Universe: supported active + inactive + historical securities.")
with st.container(horizontal=True):
    st.metric("Model status", status_label(master_status["model_status"]), border=True)
    st.metric("Dataset start", format_date(master_status["dataset_start"]), border=True)
    st.metric("Dataset end", format_date(master_status["dataset_end"]), border=True)
    st.metric("Dataset rows", format_integer(master_status["dataset_rows"]), border=True)
    st.metric("Symbols", format_integer(master_status["symbols"]), border=True)
    st.metric("New trading days", format_integer(master_status["new_data_days"]), border=True)
st.caption(
    f"Feature version: {master_status['feature_version']} · Last trained: "
    f"{format_datetime(master_status['last_trained_at'])} · Training range: "
    f"{format_date(master_status['training_data_start'])} to "
    f"{format_date(master_status['training_data_end'])} · "
    f"{status_label(master_status['training_status'])}"
)
with st.container(horizontal=True):
    if st.button("Prepare Master Dataset", icon=":material/dataset:"):
        with st.spinner("Preparing the all-lifecycle master dataset..."):
            try:
                metrics = build_master_ai_dataset()
                _cached_model_readiness.clear()
                st.success(f"Prepared {metrics.output_rows:,} master AI rows.")
            except (OSError, ValueError) as exc:
                st.error(f"Master dataset preparation failed: {exc}")
    st.button(
        "Train Master Model",
        disabled=True,
        help="A universal/master PPO remains outside Milestone 5B-5.",
        icon=":material/model_training:",
    )

st.subheader("Why this preparation matters")
with st.container(border=True):
    st.markdown(
        """
- **Symbol models** will learn independently from one active ordinary equity's
  complete history; the **master model** will retain symbol identity across
  supported active, inactive, and historical securities.
- Chronological train/validation/test boundaries prevent later market
  observations from leaking into earlier decisions. Scalers are fitted on
  training rows only.
- Retraining uses the complete updated history rather than only incremental
  rows, so earlier market context is preserved.
- A model becomes outdated when newly fetched trading dates occur after its
  recorded complete-history training cutoff. Newly eligible symbols appear as
  **Never Trained**.
        """
    )
