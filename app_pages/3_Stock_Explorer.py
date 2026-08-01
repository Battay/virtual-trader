"""Present a clear stock/security profile from local PSX market history."""

import pandas as pd
import streamlit as st

from dashboard.data_loader import (
    STOCK_HISTORY_PERIODS,
    filter_history_period,
    filter_security_history,
    history_csv_bytes,
    load_dashboard_dataset,
    paginate_dataframe,
    resolve_pagination,
    sort_history_newest_first,
)
from dashboard.presentation import (
    MISSING_VALUE,
    enum_display_options,
    format_date,
    format_decimal,
    format_integer,
    format_percentage,
    format_price,
    format_symbol_company,
    format_volume,
    safe_display_value,
    selected_option_values,
    status_badge_color,
    status_label,
)
from dashboard.registry_loader import (
    load_company_registry,
    restrict_market_data_by_registry,
)


STOCK_FILTER_DEFAULTS = {
    "stock_currently_listed_only": False,
    "stock_ordinary_equities_only": False,
    "stock_lifecycle_statuses": [],
    "stock_security_types": [],
}
HISTORY_ROWS_PER_PAGE = (25, 50, 100, 250)


def _clear_stock_filters() -> None:
    """Reset only Stock Explorer filter widget state."""
    for key, value in STOCK_FILTER_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def _reset_history_page() -> None:
    """Return the history table to its first page after a filter change."""
    st.session_state.pop("stock_history_page", None)


def _registry_row(registry: pd.DataFrame, symbol: str) -> pd.Series | None:
    rows = registry.loc[registry["symbol"].astype("string") == symbol]
    return None if rows.empty else rows.iloc[0]


def _selector_label(
    symbol: str,
    registry: pd.DataFrame,
) -> str:
    row = _registry_row(registry, symbol)
    company_name = "" if row is None else row.get("company_name", "")
    return format_symbol_company(symbol, company_name)


def _show_detail(container: object, label: str, value: object) -> None:
    container.markdown(f"**{label}**")
    container.write(safe_display_value(value))


st.title("Stock Explorer")
st.caption(
    "Review a security profile, price movement, and locally available trading history."
)

load_result = load_dashboard_dataset()
st.caption(load_result.message)
for error in load_result.errors:
    st.warning(error)

if load_result.data.empty:
    st.info("No local market data is available. Use the Fetch Data page first.")
    st.stop()

market_data = load_result.data
registry_result = load_company_registry()
registry = registry_result.data
if registry_result.available:
    lifecycle_options = enum_display_options(registry["lifecycle_status"])
    security_type_options = enum_display_options(registry["security_type"])
    with st.sidebar:
        with st.expander(
            "Security filters",
            icon=":material/filter_list:",
            expanded=False,
        ):
            currently_listed_only = st.toggle(
                "Currently listed only",
                key="stock_currently_listed_only",
            )
            ordinary_equities_only = st.toggle(
                "Ordinary equities only",
                key="stock_ordinary_equities_only",
            )
            selected_lifecycle_options = st.multiselect(
                "Lifecycle status",
                lifecycle_options,
                format_func=lambda option: option.label,
                key="stock_lifecycle_statuses",
            )
            selected_security_type_options = st.multiselect(
                "Security type",
                security_type_options,
                format_func=lambda option: option.label,
                key="stock_security_types",
            )
            st.button(
                "Clear filters",
                icon=":material/filter_alt_off:",
                on_click=_clear_stock_filters,
                width="stretch",
            )

    market_data = restrict_market_data_by_registry(
        market_data,
        registry,
        officially_listed_only=currently_listed_only,
        ordinary_equities_only=ordinary_equities_only,
        lifecycle_statuses=selected_option_values(selected_lifecycle_options),
        security_types=selected_option_values(selected_security_type_options),
    )
elif registry_result.errors:
    for error in registry_result.errors:
        st.warning(error)
else:
    st.caption("Build the Company Registry to show security names and statuses.")

symbols = sorted(
    str(symbol)
    for symbol in market_data["symbol"].dropna().astype("string").unique()
)
if not symbols:
    st.info("No securities match the selected filters.")
    st.stop()

selected_symbol = st.selectbox(
    "Security",
    symbols,
    format_func=lambda symbol: _selector_label(symbol, registry),
    placeholder="Search by security name or symbol",
    key="stock_selected_symbol",
    on_change=_reset_history_page,
)
stock_history = filter_security_history(market_data, selected_symbol)
if stock_history.empty:
    st.info(f"No local trading rows are available for {selected_symbol}.")
    st.stop()

stock_history = stock_history.copy()
for numeric_column in (
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
):
    stock_history[numeric_column] = pd.to_numeric(
        stock_history[numeric_column],
        errors="coerce",
    )

profile = _registry_row(registry, selected_symbol) if registry_result.available else None
company_name = (
    safe_display_value(profile.get("company_name"), fallback="")
    if profile is not None
    else ""
)
st.header(company_name or selected_symbol)
identity_parts = [selected_symbol]
if profile is not None:
    sector = safe_display_value(profile.get("sector"), fallback="")
    if sector:
        identity_parts.append(sector)
st.caption(" · ".join(identity_parts))

if profile is not None:
    with st.container(horizontal=True):
        lifecycle_status = profile.get("lifecycle_status")
        official_status = profile.get("official_status")
        security_type = profile.get("security_type")
        st.badge(
            status_label(lifecycle_status),
            color=status_badge_color(lifecycle_status),
        )
        st.badge(
            f"Official: {status_label(official_status)}",
            color=status_badge_color(official_status),
        )
        st.badge(status_label(security_type), color="gray")

latest_row = stock_history.iloc[-1]
latest_date = latest_row["date"]
metric_period = st.session_state.get("stock_history_period", "6M")
if metric_period not in STOCK_HISTORY_PERIODS:
    metric_period = "6M"
metric_history = filter_history_period(stock_history, metric_period)
period_high = metric_history["high"].max()
period_low = metric_history["low"].min()
latest_change = latest_row.get("change")
latest_change_percent = latest_row.get("change_percent")

st.caption(f"Latest trading date: {format_date(latest_date)}")
with st.container(horizontal=True):
    st.metric(
        "Latest Close",
        format_price(latest_row.get("close")),
        border=True,
    )
    st.metric(
        "Daily Change",
        format_price(latest_change, show_sign=True),
        delta=(
            None
            if format_percentage(latest_change_percent) == MISSING_VALUE
            else format_percentage(latest_change_percent)
        ),
        delta_description="vs. previous close",
        border=True,
    )
    st.metric(
        "Latest Open",
        format_price(latest_row.get("open")),
        border=True,
    )
with st.container(horizontal=True):
    st.metric(
        f"{metric_period} High",
        format_price(period_high),
        border=True,
    )
    st.metric(
        f"{metric_period} Low",
        format_price(period_low),
        border=True,
    )
    st.metric(
        "Latest Volume",
        format_volume(latest_row.get("volume")),
        help=(
            "Latest session volume. Full value: "
            f"{format_integer(latest_row.get('volume'))} shares."
        ),
        border=True,
    )

st.subheader("Price History")
selected_period = st.segmented_control(
    "History period",
    STOCK_HISTORY_PERIODS,
    default="6M",
    key="stock_history_period",
    on_change=_reset_history_page,
)
displayed_history = filter_history_period(
    stock_history,
    selected_period or "6M",
)
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
    latest_chart_row = chart_data.iloc[-1]
    st.caption(
        "Latest observation: "
        f"{format_date(latest_chart_row['Trading Date'])} · "
        f"{format_price(latest_chart_row['Close (PKR)'])}"
    )

st.subheader("Security Information")
if profile is None:
    st.caption("Registry metadata is unavailable for this security.")
else:
    details = (
        ("Symbol", profile.get("symbol")),
        ("Company Name", profile.get("company_name")),
        ("Sector", profile.get("sector")),
        ("Board", profile.get("board")),
        ("Listing Segment", status_label(profile.get("listing_segment"))),
        ("Security Type", status_label(profile.get("security_type"))),
        ("Official Status", status_label(profile.get("official_status"))),
        ("Trading Activity", status_label(profile.get("activity_status"))),
        ("Lifecycle Status", status_label(profile.get("lifecycle_status"))),
        ("First Seen", format_date(profile.get("first_seen_date"))),
        ("Last Seen", format_date(profile.get("last_seen_date"))),
        ("Trading Days", format_integer(profile.get("trading_days"))),
        (
            "Days Since Last Trade",
            format_integer(profile.get("days_since_last_seen")),
        ),
    )
    with st.container(border=True):
        left_details, right_details = st.columns(2)
        for index, (label, value) in enumerate(details):
            _show_detail(
                left_details if index % 2 == 0 else right_details,
                label,
                value,
            )

st.subheader("Historical Prices")
rows_per_page = st.selectbox(
    "Rows per page",
    HISTORY_ROWS_PER_PAGE,
    index=1,
    key="stock_history_rows_per_page",
    on_change=_reset_history_page,
)
newest_first = sort_history_newest_first(displayed_history)
pagination = resolve_pagination(
    len(newest_first),
    st.session_state.get("stock_history_page", 1),
    rows_per_page,
)
if st.session_state.get("stock_history_page", 1) != pagination.page_number:
    st.session_state.pop("stock_history_page", None)
page_number = st.pagination(
    pagination.total_pages,
    key="stock_history_page",
    width="stretch",
)
pagination = resolve_pagination(len(newest_first), page_number, rows_per_page)
visible_history = paginate_dataframe(newest_first, page_number, rows_per_page)
st.caption(
    f"Showing {pagination.start_row:,}–{pagination.end_row:,} of "
    f"{pagination.total_rows:,} trading records for {selected_symbol} · "
    f"{selected_period or '6M'} period · newest first"
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
    "Download Historical Prices",
    data=history_csv_bytes(newest_first),
    file_name=f"{selected_symbol}_{selected_period or '6M'}_history.csv",
    mime="text/csv",
    icon=":material/download:",
    on_click="ignore",
)
