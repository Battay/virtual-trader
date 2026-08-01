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
    filter_symbol_status,
    select_all_active_eligible,
    select_needing_retraining,
    select_never_trained,
    select_newly_added_eligible,
    select_visible_symbols,
)
from reinforcement_learning.model_management.status import (
    build_model_readiness_table,
    master_model_status,
)


def _set_selected_symbols(symbols: tuple[str, ...]) -> None:
    """Replace the page-local multi-symbol selection deterministically."""
    st.session_state["training_selected_symbols"] = list(symbols)


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
all_symbols = tuple(status_table["symbol"].astype("string"))
current_selection = [
    symbol
    for symbol in st.session_state.get("training_selected_symbols", [])
    if symbol in all_symbols
]
st.session_state["training_selected_symbols"] = current_selection
with st.container(horizontal=True):
    st.button(
        "Select All Visible",
        on_click=_set_selected_symbols,
        args=(select_visible_symbols(visible_status),),
    )
    st.button("Clear Selection", on_click=_set_selected_symbols, args=((),))
    st.button(
        "Select Never Trained",
        on_click=_set_selected_symbols,
        args=(select_never_trained(status_table),),
    )
    st.button(
        "Select Needs Retraining",
        on_click=_set_selected_symbols,
        args=(select_needing_retraining(status_table),),
    )
    st.button(
        "Select Newly Added",
        on_click=_set_selected_symbols,
        args=(select_newly_added_eligible(status_table),),
    )
    st.button(
        "Select All Active Eligible",
        on_click=_set_selected_symbols,
        args=(select_all_active_eligible(status_table),),
    )

selected_symbols = st.multiselect(
    "Selected securities",
    all_symbols,
    format_func=lambda symbol: format_symbol_company(
        symbol,
        status_table.loc[
            status_table["symbol"].astype("string") == symbol,
            "company_name",
        ].iloc[0],
    ),
    key="training_selected_symbols",
    placeholder="Search and select one or more securities",
)
selected_set = set(selected_symbols)
symbol_display = pd.DataFrame(
    {
        "Select": visible_status["symbol"].isin(selected_set),
        "Symbol": visible_status["symbol"],
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
st.caption(
    f"Showing {len(visible_status):,} active securities · "
    f"{len(selected_symbols):,} selected"
)
st.dataframe(symbol_display, width="stretch", hide_index=True, height=500)
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
        help="PPO environment and trainer are outside the historical-backfill milestone.",
        icon=":material/model_training:",
    )
st.info("PPO environment and trainer remain intentionally unimplemented.")

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
