"""Browse the local PSX dataset company first, then inspect one history."""

from collections.abc import Iterable

import pandas as pd
import streamlit as st

from dashboard.data_loader import (
    STOCK_HISTORY_PERIODS,
    build_company_summary,
    filter_history_period,
    filter_security_history,
    history_csv_bytes,
    load_dashboard_dataset,
    paginate_dataframe,
    resolve_pagination,
    sort_history_newest_first,
    summarize_dataset,
)
from dashboard.presentation import (
    company_summary_csv_bytes,
    enum_display_options,
    format_company_summary_for_display,
    format_date,
    format_decimal,
    format_integer,
    format_percentage,
    format_price,
    format_volume,
    safe_display_value,
    selected_option_values,
    status_badge_color,
    status_label,
)
from dashboard.registry_loader import (
    filter_company_registry,
    load_company_registry,
)


HISTORY_ROWS_PER_PAGE = (25, 50, 100, 250)


def _reset_history_page() -> None:
    """Return selected-company history to its first page."""
    st.session_state.pop("dataset_history_page", None)


def _plain_options(values: Iterable[object]) -> tuple[str, ...]:
    """Return sorted non-empty text options without changing their values."""
    options = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    return tuple(sorted(options, key=str.casefold))


st.title("Dataset Explorer")
st.caption(
    "Browse one row per security, then inspect only the selected company's "
    "trading history."
)

load_result = load_dashboard_dataset()
st.caption(load_result.message)
for error in load_result.errors:
    st.warning(error)

if load_result.data.empty:
    st.info("No market rows are available. Use the Fetch Data page first.")
    st.stop()

dataset_summary = summarize_dataset(load_result)
with st.container(horizontal=True):
    st.metric(
        "Securities",
        format_integer(dataset_summary.unique_symbols),
        border=True,
    )
    st.metric(
        "Trading dates",
        format_integer(dataset_summary.trading_dates),
        border=True,
    )
    st.metric("Market rows", format_integer(dataset_summary.total_rows), border=True)
    st.metric("CSV files", format_integer(dataset_summary.csv_files), border=True)
st.caption(
    "Trading-date coverage: "
    f"{format_date(dataset_summary.earliest_date)} to "
    f"{format_date(dataset_summary.latest_date)}"
)

market_data = load_result.data
registry_result = load_company_registry()
for error in registry_result.errors:
    st.warning(error)
if not registry_result.available:
    st.caption(
        "Company Registry metadata is unavailable; market-derived company history "
        "remains browsable."
    )

company_summary = build_company_summary(
    market_data,
    registry_result.data if registry_result.available else None,
)
total_companies = len(company_summary)

with st.expander("Company filters", icon=":material/filter_list:"):
    search = st.text_input(
        "Search by symbol or company name",
        key="dataset_company_search",
    )
    with st.container(horizontal=True):
        currently_listed_only = st.toggle(
            "Currently listed only",
            key="dataset_currently_listed_only",
        )
        ordinary_equities_only = st.toggle(
            "Ordinary equities only",
            key="dataset_ordinary_equities_only",
        )

    st.markdown("**Status**")
    official_options = enum_display_options(company_summary["official_status"])
    activity_options = enum_display_options(company_summary["activity_status"])
    lifecycle_options = enum_display_options(company_summary["lifecycle_status"])
    with st.container(horizontal=True):
        selected_official_statuses = st.multiselect(
            "Official status",
            official_options,
            format_func=lambda option: option.label,
            key="dataset_official_statuses",
        )
        selected_activity_statuses = st.multiselect(
            "Trading activity",
            activity_options,
            format_func=lambda option: option.label,
            key="dataset_activity_statuses",
        )
        selected_lifecycle_statuses = st.multiselect(
            "Lifecycle status",
            lifecycle_options,
            format_func=lambda option: option.label,
            key="dataset_lifecycle_statuses",
        )

    st.markdown("**Classification**")
    security_type_options = enum_display_options(company_summary["security_type"])
    board_options = enum_display_options(company_summary["board"])
    segment_options = enum_display_options(company_summary["listing_segment"])
    with st.container(horizontal=True):
        selected_security_types = st.multiselect(
            "Security type",
            security_type_options,
            format_func=lambda option: option.label,
            key="dataset_security_types",
        )
        selected_sectors = st.multiselect(
            "Sector",
            _plain_options(company_summary["sector"]),
            key="dataset_sectors",
        )
        selected_boards = st.multiselect(
            "Board",
            board_options,
            format_func=lambda option: option.label,
            key="dataset_boards",
        )
        selected_segments = st.multiselect(
            "Listing segment",
            segment_options,
            format_func=lambda option: option.label,
            key="dataset_listing_segments",
        )

    minimum_trading_days = st.number_input(
        "Minimum trading days",
        min_value=0,
        value=0,
        step=1,
        key="dataset_minimum_trading_days",
    )
    first_seen_start = first_seen_end = None
    last_seen_start = last_seen_end = None
    with st.container(horizontal=True):
        filter_first_seen = st.toggle(
            "Filter first-seen dates",
            key="dataset_filter_first_seen",
        )
        filter_last_seen = st.toggle(
            "Filter last-seen dates",
            key="dataset_filter_last_seen",
        )
    if filter_first_seen:
        first_seen_dates = pd.to_datetime(
            company_summary["first_seen_date"],
            errors="coerce",
        ).dropna()
        if not first_seen_dates.empty:
            first_seen_range = st.date_input(
                "First-seen range",
                value=(first_seen_dates.min().date(), first_seen_dates.max().date()),
                key="dataset_first_seen_range",
            )
            if len(first_seen_range) == 2:
                first_seen_start, first_seen_end = first_seen_range
    if filter_last_seen:
        last_seen_dates = pd.to_datetime(
            company_summary["last_seen_date"],
            errors="coerce",
        ).dropna()
        if not last_seen_dates.empty:
            last_seen_range = st.date_input(
                "Last-seen range",
                value=(last_seen_dates.min().date(), last_seen_dates.max().date()),
                key="dataset_last_seen_range",
            )
            if len(last_seen_range) == 2:
                last_seen_start, last_seen_end = last_seen_range

filtered_companies = filter_company_registry(
    company_summary,
    search=search,
    officially_listed_only=currently_listed_only,
    ordinary_equities_only=ordinary_equities_only,
    official_statuses=selected_option_values(selected_official_statuses),
    activity_statuses=selected_option_values(selected_activity_statuses),
    lifecycle_statuses=selected_option_values(selected_lifecycle_statuses),
    security_types=selected_option_values(selected_security_types),
    sectors=selected_sectors,
    boards=selected_option_values(selected_boards),
    listing_segments=selected_option_values(selected_segments),
    minimum_trading_days=(
        int(minimum_trading_days) if minimum_trading_days else None
    ),
    first_seen_start=first_seen_start,
    first_seen_end=first_seen_end,
    last_seen_start=last_seen_start,
    last_seen_end=last_seen_end,
)

st.subheader("Companies and securities")
st.caption(
    f"Showing {len(filtered_companies):,} of {total_companies:,} securities. "
    "Select one row to inspect its history."
)
if filtered_companies.empty:
    st.info("No securities match the selected company filters.")
    st.stop()

company_display = format_company_summary_for_display(filtered_companies)
company_table = company_display.copy()
company_table["First Seen"] = company_table["First Seen"].map(format_date)
company_table["Last Seen"] = company_table["Last Seen"].map(format_date)
company_table["Trading Days"] = company_table["Trading Days"].map(format_integer)
company_table["Latest Close"] = company_table["Latest Close"].map(format_price)
selection = st.dataframe(
    company_table,
    width="stretch",
    height=500,
    hide_index=True,
    key="dataset_company_table",
    on_select=_reset_history_page,
    selection_mode="single-row-required",
    selection_default={"selection": {"rows": [0]}},
    placeholder="—",
)
st.download_button(
    "Download filtered company list",
    data=company_summary_csv_bytes(filtered_companies),
    file_name="psx_filtered_companies.csv",
    mime="text/csv",
    icon=":material/download:",
    on_click="ignore",
)

selected_rows = list(selection.selection.rows)
selected_index = selected_rows[0] if selected_rows else 0
if selected_index < 0 or selected_index >= len(filtered_companies):
    selected_index = 0
selected_company = filtered_companies.iloc[selected_index]
selected_symbol = str(selected_company["symbol"])

selected_history = filter_security_history(market_data, selected_symbol)
for numeric_column in (
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
):
    selected_history[numeric_column] = pd.to_numeric(
        selected_history[numeric_column],
        errors="coerce",
    )

st.divider()
st.subheader("Selected security")
selected_name = safe_display_value(selected_company.get("company_name"), fallback="")
st.header(selected_name or selected_symbol)
selected_sector = safe_display_value(selected_company.get("sector"), fallback="")
st.caption(
    " · ".join(
        part for part in (selected_symbol, selected_sector) if part
    )
)
with st.container(horizontal=True):
    lifecycle_status = selected_company.get("lifecycle_status")
    official_status = selected_company.get("official_status")
    st.badge(
        status_label(lifecycle_status),
        color=status_badge_color(lifecycle_status),
    )
    st.badge(
        f"Official: {status_label(official_status)}",
        color=status_badge_color(official_status),
    )

selected_period = st.segmented_control(
    "History period",
    STOCK_HISTORY_PERIODS,
    default="6M",
    key="dataset_history_period",
    on_change=_reset_history_page,
)
period_label = selected_period or "6M"
displayed_history = filter_history_period(selected_history, period_label)
latest_row = displayed_history.iloc[-1]
period_high = displayed_history["high"].max()
period_low = displayed_history["low"].min()
trading_days = displayed_history["date"].nunique()
total_volume = displayed_history["volume"].sum(min_count=1)

with st.container(horizontal=True):
    st.metric(
        "Latest trading date",
        format_date(latest_row.get("date")),
        border=True,
    )
    st.metric(
        "Latest close",
        format_price(latest_row.get("close")),
        border=True,
    )
    st.metric("Period high", format_price(period_high), border=True)
    st.metric("Period low", format_price(period_low), border=True)
    st.metric("Total trading days", format_integer(trading_days), border=True)
    st.metric(
        "Total volume",
        format_volume(total_volume),
        help=f"Full value: {format_integer(total_volume)} shares.",
        border=True,
    )

st.subheader("Price history")
chart_data = displayed_history.dropna(subset=["date", "close"]).rename(
    columns={"date": "Trading Date", "close": "Close (PKR)"}
)
if chart_data.empty:
    st.info("No close-price values are available for the selected period.")
else:
    st.line_chart(
        chart_data,
        x="Trading Date",
        y="Close (PKR)",
        x_label="Trading Date",
        y_label="Close Price (PKR)",
    )

st.subheader("Day-by-day data")
rows_per_page = st.selectbox(
    "Rows per page",
    HISTORY_ROWS_PER_PAGE,
    index=1,
    key="dataset_history_rows_per_page",
    on_change=_reset_history_page,
)
newest_first = sort_history_newest_first(displayed_history)
pagination = resolve_pagination(
    len(newest_first),
    st.session_state.get("dataset_history_page", 1),
    rows_per_page,
)
if st.session_state.get("dataset_history_page", 1) != pagination.page_number:
    st.session_state.pop("dataset_history_page", None)
page_number = st.pagination(
    pagination.total_pages,
    key="dataset_history_page",
    width="stretch",
)
pagination = resolve_pagination(len(newest_first), page_number, rows_per_page)
visible_history = paginate_dataframe(newest_first, page_number, rows_per_page)
st.caption(
    f"Showing {pagination.start_row:,}–{pagination.end_row:,} of "
    f"{pagination.total_rows:,} trading records for {selected_symbol}"
)
history_table = pd.DataFrame(
    {
        "Trading Date": visible_history["date"].map(format_date),
        "Open": visible_history["open"].map(
            lambda value: format_price(value, include_currency=False)
        ),
        "High": visible_history["high"].map(
            lambda value: format_price(value, include_currency=False)
        ),
        "Low": visible_history["low"].map(
            lambda value: format_price(value, include_currency=False)
        ),
        "Close": visible_history["close"].map(
            lambda value: format_price(value, include_currency=False)
        ),
        "Change": visible_history["change"].map(
            lambda value: format_decimal(value, show_sign=True)
        ),
        "Change %": visible_history["change_percent"].map(format_percentage),
        "Volume": visible_history["volume"].map(format_integer),
    }
)
st.dataframe(history_table, width="stretch", hide_index=True)
st.download_button(
    "Download selected-company history",
    data=history_csv_bytes(newest_first),
    file_name=f"{selected_symbol}_{period_label}_history.csv",
    mime="text/csv",
    icon=":material/download:",
    on_click="ignore",
)
