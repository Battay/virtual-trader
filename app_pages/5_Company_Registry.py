"""Explore official PSX status alongside locally observed trading activity."""

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.data_loader import filter_market_data, load_dashboard_dataset
from dashboard.presentation import (
    MISSING_VALUE,
    enum_display_options,
    format_date,
    format_datetime,
    format_integer,
    format_price,
    format_source,
    safe_display_value,
    selected_option_values,
    status_badge_color,
    status_label,
)
from dashboard.registry_loader import (
    filter_company_registry,
    load_company_registry,
    summarize_registry_for_display,
)
from data_pipeline.src.company_registry import refresh_and_build_registry


REGISTRY_FILTER_KEYS = (
    "registry_search",
    "registry_lifecycle_statuses",
    "registry_official_statuses",
    "registry_activity_statuses",
    "registry_security_types",
    "registry_sectors",
    "registry_boards",
    "registry_listing_segments",
    "registry_first_seen_range",
    "registry_last_seen_range",
    "registry_minimum_trading_days",
)


def _clear_registry_filters() -> None:
    """Remove Company Registry filter widget state before the next rerun."""
    for key in REGISTRY_FILTER_KEYS:
        st.session_state.pop(key, None)


def _values(data: pd.DataFrame, column: str) -> list[str]:
    values = data[column].dropna().astype("string").str.strip()
    return sorted(str(value) for value in values.loc[values != ""].unique())


def _date_bounds(data: pd.DataFrame, column: str) -> tuple[date | None, date | None]:
    values = pd.to_datetime(data[column], errors="coerce").dropna()
    if values.empty:
        return None, None
    return values.min().date(), values.max().date()


def _selected_range(value: object) -> tuple[date | None, date | None]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return None, None


def _display_registry_table(registry: pd.DataFrame) -> pd.DataFrame:
    """Return a readable table without changing the filtered raw registry."""
    return pd.DataFrame(
        {
            "Symbol": registry["symbol"].map(safe_display_value),
            "Company Name": registry["company_name"].map(safe_display_value),
            "Security Type": registry["security_type"].map(status_label),
            "Sector": registry["sector"].map(safe_display_value),
            "Board": registry["board"].map(safe_display_value),
            "Official Status": registry["official_status"].map(status_label),
            "Trading Activity": registry["activity_status"].map(status_label),
            "Lifecycle Status": registry["lifecycle_status"].map(status_label),
            "First Seen": registry["first_seen_date"].map(format_date),
            "Last Seen": registry["last_seen_date"].map(format_date),
            "Trading Days": registry["trading_days"].map(format_integer),
            "Days Since Last Trade": registry["days_since_last_seen"].map(
                format_integer
            ),
        }
    )


def _nonempty_detail(label: str, value: object) -> tuple[str, str] | None:
    display_value = safe_display_value(value)
    return None if display_value == MISSING_VALUE else (label, display_value)


def _show_detail(container: object, label: str, value: str) -> None:
    container.markdown(f"**{label}**")
    container.write(value)


st.title("Company Registry")
st.caption(
    "Official PSX listing information combined with locally observed trading history."
)
st.write(
    "Listing status comes from official PSX information. Trading activity is "
    "calculated from locally stored market data, so a security is not considered "
    "delisted simply because it has not traded recently."
)
with st.expander(
    "How statuses are calculated",
    icon=":material/info:",
):
    st.write(
        "Official status is sourced from the current PSX listing tables or an "
        "evidence-based override. Trading activity compares the last locally "
        "observed trading date with the configured recent-trading window. "
        "Lifecycle status combines those two independent signals."
    )
    st.caption(
        "First Seen is the earliest date in the local master dataset; it is not "
        "an official PSX listing date."
    )

flash_message = st.session_state.pop("registry_flash_message", None)
if flash_message:
    st.success(flash_message)

if st.button(
    "Refresh Listings & Rebuild Registry",
    icon=":material/refresh:",
    type="secondary",
):
    try:
        with st.spinner("Refreshing official listings and rebuilding the registry..."):
            refresh_result = refresh_and_build_registry()
        mode = (
            "cached official listings"
            if refresh_result.listings.used_cache
            else "live official listings"
        )
        st.session_state["registry_flash_message"] = (
            f"Registry rebuilt with {mode}: "
            f"{refresh_result.registry.total_registry_symbols:,} securities."
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Registry refresh could not be completed: {exc}")

load_result = load_company_registry()
for error in load_result.errors:
    st.error(error)
if not load_result.available:
    st.warning(
        "No valid Company Registry is available. Refresh it after building the "
        "master market dataset."
    )
    st.stop()

registry = load_result.data
cached_listings_used = bool(registry["cached_listings_used"].any())
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(
        "Official listings refreshed: "
        f"{format_datetime(registry['listing_refreshed_at'].iloc[0])}"
    )
    st.caption(
        f"Registry built: {format_datetime(registry['registry_updated_at'].iloc[0])}"
    )
    st.badge(
        "Cached official listings" if cached_listings_used else "Live official listings",
        color="yellow" if cached_listings_used else "green",
        icon=(
            ":material/database:"
            if cached_listings_used
            else ":material/cloud_done:"
        ),
    )
if cached_listings_used:
    st.warning(
        "The live PSX listing source was unavailable when this registry was built. "
        "The newest valid cached official snapshot is being shown."
    )

metrics = summarize_registry_for_display(registry)
with st.container(horizontal=True):
    st.metric(
        "Total Securities",
        format_integer(metrics.total_securities),
        border=True,
    )
    st.metric(
        "Currently Listed",
        format_integer(metrics.currently_listed),
        help="Securities present in the current official PSX listing snapshot.",
        border=True,
    )
    st.metric(
        "Listed & Recently Traded",
        format_integer(metrics.listed_and_recently_traded),
        help=(
            "Currently listed securities observed in local market history within "
            "the configured recent-trading window."
        ),
        border=True,
    )
    st.metric(
        "Historical Only",
        format_integer(metrics.historical_only),
        help=(
            "Symbols found in local market history but absent from the current "
            "official listing snapshot. This does not prove delisting."
        ),
        border=True,
    )
with st.container(horizontal=True):
    st.metric(
        "Listed, Not Recently Traded",
        format_integer(metrics.listed_not_recently_traded),
        border=True,
    )
    st.metric(
        "New Listings",
        format_integer(metrics.new_listings),
        help=(
            "Currently listed securities first observed within the configured "
            "local-history window. First observed is not an official listing date."
        ),
        border=True,
    )
    st.metric("Suspended", format_integer(metrics.suspended), border=True)
    st.metric("Non-Compliant", format_integer(metrics.non_compliant), border=True)
    st.metric(
        "Officially Delisted",
        format_integer(metrics.officially_delisted),
        border=True,
    )
    st.metric("Unknown", format_integer(metrics.unknown), border=True)
st.caption(
    "Metrics describe different registry facets and are not intended to add up. "
    "Listed & Recently Traded excludes historical-only symbols."
)

first_min, first_max = _date_bounds(registry, "first_seen_date")
last_min, last_max = _date_bounds(registry, "last_seen_date")
with st.sidebar:
    st.subheader("Registry filters")
    st.button(
        "Clear All Filters",
        icon=":material/filter_alt_off:",
        on_click=_clear_registry_filters,
        width="stretch",
    )
    with st.expander("Search", icon=":material/search:", expanded=True):
        search = st.text_input(
            "Symbol or company name",
            placeholder="Search securities",
            key="registry_search",
        )
    with st.expander("Status", icon=":material/fact_check:"):
        lifecycle_options = enum_display_options(registry["lifecycle_status"])
        official_options = enum_display_options(registry["official_status"])
        activity_options = enum_display_options(registry["activity_status"])
        selected_lifecycles = st.multiselect(
            "Lifecycle Status",
            lifecycle_options,
            format_func=lambda option: option.label,
            key="registry_lifecycle_statuses",
        )
        selected_official_statuses = st.multiselect(
            "Official Status",
            official_options,
            format_func=lambda option: option.label,
            key="registry_official_statuses",
        )
        selected_activity_statuses = st.multiselect(
            "Trading Activity",
            activity_options,
            format_func=lambda option: option.label,
            key="registry_activity_statuses",
        )
    with st.expander("Classification", icon=":material/category:"):
        security_type_options = enum_display_options(registry["security_type"])
        board_options = enum_display_options(registry["board"])
        segment_options = enum_display_options(registry["listing_segment"])
        selected_security_types = st.multiselect(
            "Security Type",
            security_type_options,
            format_func=lambda option: option.label,
            key="registry_security_types",
        )
        selected_sectors = st.multiselect(
            "Sector",
            _values(registry, "sector"),
            key="registry_sectors",
        )
        selected_boards = st.multiselect(
            "Board",
            board_options,
            format_func=lambda option: option.label,
            key="registry_boards",
        )
        selected_listing_segments = st.multiselect(
            "Listing Segment",
            segment_options,
            format_func=lambda option: option.label,
            key="registry_listing_segments",
        )
    with st.expander("Trading History", icon=":material/history:"):
        first_seen_range = st.date_input(
            "First Seen range",
            value=[],
            min_value=first_min,
            max_value=first_max,
            key="registry_first_seen_range",
        )
        last_seen_range = st.date_input(
            "Last Seen range",
            value=[],
            min_value=last_min,
            max_value=last_max,
            key="registry_last_seen_range",
        )
        minimum_trading_days = st.number_input(
            "Minimum trading days",
            min_value=0,
            value=0,
            step=1,
            help="Use zero to include securities with no local trading history.",
            key="registry_minimum_trading_days",
        )

first_start, first_end = _selected_range(first_seen_range)
last_start, last_end = _selected_range(last_seen_range)
filtered = filter_company_registry(
    registry,
    lifecycle_statuses=selected_option_values(selected_lifecycles),
    official_statuses=selected_option_values(selected_official_statuses),
    activity_statuses=selected_option_values(selected_activity_statuses),
    security_types=selected_option_values(selected_security_types),
    sectors=selected_sectors,
    boards=selected_option_values(selected_boards),
    listing_segments=selected_option_values(selected_listing_segments),
    search=search,
    first_seen_start=first_start,
    first_seen_end=first_end,
    last_seen_start=last_start,
    last_seen_end=last_end,
    minimum_trading_days=(
        int(minimum_trading_days) if minimum_trading_days else None
    ),
)

st.subheader("Securities")
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(f"{len(filtered):,} of {len(registry):,} registry rows")
    st.download_button(
        "Download Filtered Registry",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="psx_company_registry_filtered.csv",
        mime="text/csv",
        icon=":material/download:",
        type="secondary",
        on_click="ignore",
    )

table_data = _display_registry_table(filtered)
selection = st.dataframe(
    table_data,
    hide_index=True,
    width="stretch",
    key="company_registry_table",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Symbol": st.column_config.TextColumn(pinned=True),
        "Company Name": st.column_config.TextColumn(pinned=True),
    },
)

if selection.selection.rows and selection.selection.rows[0] < len(filtered):
    selected_row = filtered.iloc[selection.selection.rows[0]]
    symbol = safe_display_value(selected_row["symbol"])
    selected_name = safe_display_value(
        selected_row["company_name"],
        fallback="",
    )
    st.subheader(selected_name or symbol)
    if selected_name:
        st.caption(symbol)
    with st.container(horizontal=True):
        for status_field in (
            "lifecycle_status",
            "official_status",
            "activity_status",
        ):
            status = selected_row[status_field]
            st.badge(
                status_label(status),
                color=status_badge_color(status),
            )
        st.badge(status_label(selected_row["security_type"]), color="gray")

    market_result = load_dashboard_dataset()
    history = filter_market_data(market_result.data, symbol=symbol)
    latest_close = (
        format_price(history.iloc[-1]["close"])
        if not history.empty and "close" in history
        else MISSING_VALUE
    )

    detail_items: list[tuple[str, str]] = [
        ("Symbol", symbol),
        ("Official Status", status_label(selected_row["official_status"])),
        ("Trading Activity", status_label(selected_row["activity_status"])),
        ("Lifecycle Status", status_label(selected_row["lifecycle_status"])),
        ("Security Type", status_label(selected_row["security_type"])),
        ("First Seen", format_date(selected_row["first_seen_date"])),
        ("Last Seen", format_date(selected_row["last_seen_date"])),
        ("Trading Days", format_integer(selected_row["trading_days"])),
        ("Latest Close", latest_close),
    ]
    for optional_detail in (
        _nonempty_detail("Company Name", selected_row["company_name"]),
        _nonempty_detail("Sector", selected_row["sector"]),
        _nonempty_detail("Board", selected_row["board"]),
        _nonempty_detail(
            "Listing Segment",
            status_label(selected_row["listing_segment"]),
        ),
        _nonempty_detail("Previous Symbol", selected_row["previous_symbol"]),
        _nonempty_detail("Successor Symbol", selected_row["successor_symbol"]),
        _nonempty_detail(
            "Corporate Action",
            status_label(selected_row["corporate_action_type"]),
        ),
        _nonempty_detail("Notes", selected_row["notes"]),
    ):
        if optional_detail is not None:
            detail_items.append(optional_detail)

    with st.container(border=True):
        left_details, right_details = st.columns(2)
        for index, (label, value) in enumerate(detail_items):
            _show_detail(
                left_details if index % 2 == 0 else right_details,
                label,
                value,
            )

    with st.expander("Additional Listing Metadata"):
        metadata_items = [
            _nonempty_detail("Clearing Type", selected_row["clearing_type"]),
            _nonempty_detail("Listed In", selected_row["listed_in"]),
            _nonempty_detail("Shares", format_integer(selected_row["shares"])),
            _nonempty_detail(
                "Free Float",
                format_integer(selected_row["free_float"]),
            ),
            _nonempty_detail("Source", format_source(selected_row["source"])),
        ]
        metadata = [item for item in metadata_items if item is not None]
        if metadata:
            st.dataframe(
                pd.DataFrame(metadata, columns=("Field", "Value")),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No additional listing metadata is available.")
