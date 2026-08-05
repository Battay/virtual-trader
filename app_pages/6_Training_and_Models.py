"""Manage AI datasets and future PPO model readiness without training models."""

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data_loader import load_dashboard_dataset
from dashboard.presentation import (
    enum_display_options,
    format_date,
    format_datetime,
    format_integer,
    format_model_registry_for_display,
    format_symbol_company,
    safe_display_value,
    selected_option_values,
    status_label,
)
from dashboard.registry_loader import load_company_registry
from data_pipeline.src.config import (
    AI_MINIMUM_USABLE_ROWS,
    MODEL_REGISTRY_PATH,
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
from reinforcement_learning.model_management.registry import (
    ModelRegistryError,
    latest_model_versions,
    load_model_registry,
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


@st.cache_data(max_entries=8, show_spinner=False)
def _validate_rl_environment(
    path_text: str,
    modified_ns: int,
    size: int,
) -> dict[str, object]:
    """Validate one unchanged local symbol dataset only on explicit request."""
    del modified_ns, size
    try:
        data = pd.read_csv(Path(path_text), dtype={"symbol": "string"})
        result = validate_environment(SingleSymbolTradingEnv(data))
        return {
            "status": "Environment Ready" if result.valid else "Validation Failed",
            "message": (
                "Environment v1 Ready"
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


st.title("Training & Model Management")
st.caption(
    "Prepare leakage-safe AI datasets and track future PPO model readiness. "
    "Historical readiness is evaluated, but no model training is performed."
)

st.subheader("RL environment readiness")
st.session_state.setdefault(
    "rl_environment_validation",
    {
        "status": "Not Implemented",
        "message": "Validate a processed single-symbol dataset to confirm readiness.",
        "shape": None,
    },
)
rl_validation = st.session_state["rl_environment_validation"]
with st.container(horizontal=True):
    st.metric("RL Environment", rl_validation["status"], border=True)
    st.metric("Environment Version", ENVIRONMENT_VERSION, border=True)
if st.button(
    "Validate RL Environment",
    icon=":material/fact_check:",
    key="validate_rl_environment",
):
    candidates = sorted(Path(PROCESSED_SYMBOLS_DIR).glob("*.csv"))
    if not candidates:
        st.session_state["rl_environment_validation"] = {
            "status": "Not Implemented",
            "message": "No processed single-symbol dataset is available.",
            "shape": None,
        }
    else:
        candidate = candidates[0]
        details = candidate.stat()
        st.session_state["rl_environment_validation"] = _validate_rl_environment(
            str(candidate),
            details.st_mtime_ns,
            details.st_size,
        )
    st.rerun()
if rl_validation["status"] == "Environment Ready":
    st.success(
        f"{rl_validation['message']} · Observation shape: "
        f"{rl_validation['shape']}"
    )
elif rl_validation["status"] == "Validation Failed":
    st.error(rl_validation["message"])
else:
    st.info(rl_validation["message"])
st.caption("PPO training remains disabled and arrives in Milestone 5B.")

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
    model_registry = pd.DataFrame()

processed_master, processed_error = _load_processed_master()
if processed_error:
    st.warning(processed_error)
raw_dates = pd.to_datetime(market_result.data["date"], errors="coerce").dropna()
processed_dates = (
    pd.to_datetime(processed_master["date"], errors="coerce").dropna()
    if "date" in processed_master
    else pd.Series(dtype="datetime64[ns]")
)
minimum_history = st.number_input(
    "Minimum usable rows for symbol eligibility",
    min_value=1,
    value=AI_MINIMUM_USABLE_ROWS,
    step=1,
    help=(
        "The initial 252-row gate approximates one trading year after indicator "
        "warm-up. It is configurable and does not guarantee PPO sufficiency."
    ),
    key="training_minimum_history",
)
status_table = _cached_model_readiness(
    market_result.data,
    registry_result.data,
    model_registry,
    int(minimum_history),
    _processed_symbol_fingerprint(),
)

st.subheader("Dataset readiness")
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
        "Warm-up Rows": visible_status["warmup_rows_removed"].map(format_integer),
        "Usable Rows": visible_status["usable_rows"].map(format_integer),
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
        help="PPO training arrives in Milestone 5B.",
        icon=":material/model_training:",
    )
st.info("Environment v1 is implemented; PPO training remains disabled until Milestone 5B.")

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
        help="PPO environment and trainer are outside the historical-backfill milestone.",
        icon=":material/model_training:",
    )

st.subheader("Model registry")
latest_models = latest_model_versions(model_registry)
if latest_models.empty:
    st.info(f"No model versions are registered yet. Registry path: {MODEL_REGISTRY_PATH}")
else:
    st.dataframe(
        format_model_registry_for_display(latest_models),
        width="stretch",
        hide_index=True,
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
