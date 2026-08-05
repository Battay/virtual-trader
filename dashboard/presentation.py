"""Human-readable, defensive presentation helpers for the Streamlit dashboard."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from zoneinfo import ZoneInfo

import pandas as pd

from data_pipeline.src.config import PROJECT_TIMEZONE


MISSING_VALUE = "—"
ENUM_LABELS = {
    "listed": "Listed",
    "suspended": "Suspended",
    "non_compliant": "Non-Compliant",
    "delisted": "Delisted",
    "historical": "Historical",
    "unknown": "Unknown",
    "recently_traded": "Recently Traded",
    "not_recently_traded": "Not Recently Traded",
    "never_seen_in_market_history": "No Local Trading History",
    "listed_recently_traded": "Listed · Recently Traded",
    "listed_not_recently_traded": "Listed · Not Recently Traded",
    "newly_listed": "Newly Listed",
    "historical_only": "Historical Only",
    "officially_delisted": "Officially Delisted",
    "ordinary_equity": "Ordinary Equity",
    "preference_share": "Preference Share",
    "gem_equity": "GEM Equity",
    "etf": "ETF",
    "right": "Right",
    "other": "Other",
    "normal_counter": "Normal Counter",
    "non_compliant_segment": "Non-Compliant Segment",
    "main": "Main",
    "gem": "GEM",
    "never": "Never",
    "success": "Success",
    "failed": "Failed",
    "disabled": "Disabled",
    "already_running": "Already Running",
    "running": "Running",
    "not_trained": "Not Trained",
    "training": "Training",
    "trained": "Trained",
    "archived": "Archived",
    "never_trained": "Never Trained",
    "up_to_date": "Up to Date",
    "retraining_recommended": "Retraining Recommended",
    "insufficient_history": "Insufficient History",
    "data_quality_issue": "Data Quality Issue",
    "unsupported_security_type": "Unsupported Security Type",
    "missing_processed_dataset": "Missing Processed Dataset",
    "training_failed": "Training Failed",
    "symbol": "Symbol",
    "master": "Master",
}
STATUS_BADGE_COLORS = {
    "listed": "green",
    "listed_recently_traded": "green",
    "recently_traded": "green",
    "newly_listed": "blue",
    "listed_not_recently_traded": "yellow",
    "not_recently_traded": "yellow",
    "never_seen_in_market_history": "gray",
    "historical": "gray",
    "historical_only": "gray",
    "suspended": "orange",
    "non_compliant": "orange",
    "officially_delisted": "red",
    "delisted": "red",
    "unknown": "gray",
}
COLUMN_LABELS = {
    "symbol": "Symbol",
    "date": "Trading Date",
    "ldcp": "LDCP",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "change": "Change",
    "change_percent": "Change %",
    "volume": "Volume",
}
COMPANY_SUMMARY_DISPLAY_COLUMNS = (
    "Symbol",
    "Company Name",
    "Security Type",
    "Sector",
    "Official Status",
    "Trading Activity",
    "Lifecycle Status",
    "First Seen",
    "Last Seen",
    "Trading Days",
    "Latest Close",
)


@dataclass(frozen=True)
class DisplayOption:
    """Backend option value paired with its human-readable widget label."""

    value: str
    label: str


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "nat", "none", "null"}
    return False


def safe_display_value(value: object, fallback: str = MISSING_VALUE) -> str:
    """Return a clean scalar string without leaking pandas missing markers."""
    if _is_missing(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def humanize_enum(value: object) -> str:
    """Convert a backend enum to an explicit or fallback readable label."""
    if _is_missing(value):
        return MISSING_VALUE
    normalized = str(value).strip().lower()
    if normalized in ENUM_LABELS:
        return ENUM_LABELS[normalized]
    words = re.sub(r"[_-]+", " ", normalized).split()
    return " ".join(word.capitalize() for word in words) or MISSING_VALUE


def status_label(value: object) -> str:
    """Return the dashboard label for a status or classification value."""
    return humanize_enum(value)


def status_badge_color(value: object) -> str:
    """Return a native Streamlit badge color for a backend status value."""
    if _is_missing(value):
        return "gray"
    return STATUS_BADGE_COLORS.get(str(value).strip().lower(), "gray")


def column_label(value: object) -> str:
    """Return a readable table-column label with common financial abbreviations."""
    if _is_missing(value):
        return MISSING_VALUE
    normalized = str(value).strip().lower()
    return COLUMN_LABELS.get(normalized, humanize_enum(normalized))


def enum_display_options(values: Iterable[object]) -> tuple[DisplayOption, ...]:
    """Return unique widget options with labels while retaining backend values."""
    normalized_values = {
        str(value).strip()
        for value in values
        if not _is_missing(value) and str(value).strip()
    }
    options = (
        DisplayOption(value=value, label=humanize_enum(value))
        for value in normalized_values
    )
    return tuple(sorted(options, key=lambda option: option.label.casefold()))


def selected_option_values(
    options: Sequence[DisplayOption] | None,
) -> tuple[str, ...]:
    """Extract backend values from selected presentation options."""
    if not options:
        return ()
    return tuple(option.value for option in options)


def format_date(value: object, fallback: str = MISSING_VALUE) -> str:
    """Format a date-like value as ``30 Jul 2026``."""
    if _is_missing(value):
        return fallback
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return fallback
    if pd.isna(timestamp):
        return fallback
    return timestamp.strftime("%d %b %Y")


def format_datetime(value: object, fallback: str = MISSING_VALUE) -> str:
    """Format a timestamp in Pakistan time with an explicit PKT suffix."""
    if _is_missing(value):
        return fallback
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return fallback
    if pd.isna(timestamp):
        return fallback
    timezone = ZoneInfo(PROJECT_TIMEZONE)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    hour = timestamp.strftime("%I").lstrip("0") or "0"
    return (
        f"{timestamp.strftime('%d %b %Y')}, "
        f"{hour}:{timestamp.strftime('%M %p')} PKT"
    )


def _number(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_integer(value: object, fallback: str = MISSING_VALUE) -> str:
    """Format an integer-like value with thousands separators."""
    number = _number(value)
    return fallback if number is None else f"{int(round(number)):,}"


def format_decimal(
    value: object,
    *,
    precision: int = 2,
    show_sign: bool = False,
    fallback: str = MISSING_VALUE,
) -> str:
    """Format a finite decimal at a predictable precision."""
    if precision < 0:
        raise ValueError("precision cannot be negative")
    number = _number(value)
    if number is None:
        return fallback
    sign = "+" if show_sign and number > 0 else ""
    return f"{sign}{number:,.{precision}f}"


def format_price(
    value: object,
    *,
    include_currency: bool = True,
    show_sign: bool = False,
    fallback: str = MISSING_VALUE,
) -> str:
    """Format a price with sensible PSX precision and optional PKR context."""
    formatted = format_decimal(
        value,
        precision=2,
        show_sign=show_sign,
        fallback=fallback,
    )
    if formatted == fallback:
        return fallback
    if not include_currency:
        return formatted
    return f"PKR {formatted}"


def format_source(value: object, fallback: str = MISSING_VALUE) -> str:
    """Humanize local source tags while preserving URLs verbatim."""
    source = safe_display_value(value, fallback=fallback)
    if source == fallback:
        return fallback
    parts = (part.strip() for part in source.split(";"))
    return "; ".join(
        part if "://" in part else humanize_enum(part)
        for part in parts
        if part
    )


def format_volume(value: object, fallback: str = MISSING_VALUE) -> str:
    """Abbreviate large share volumes using K, M, B, or T notation."""
    number = _number(value)
    if number is None:
        return fallback
    absolute = abs(number)
    for threshold, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if absolute >= threshold:
            return f"{number / threshold:.1f}{suffix}"
    return f"{int(round(number)):,}"


def format_percentage(
    value: object,
    *,
    show_sign: bool = True,
    fallback: str = MISSING_VALUE,
) -> str:
    """Format a percentage value that is already expressed in percent units."""
    formatted = format_decimal(
        value,
        precision=2,
        show_sign=show_sign,
        fallback=fallback,
    )
    return fallback if formatted == fallback else f"{formatted}%"


def format_directional_percentage(
    value: object,
    *,
    fallback: str = MISSING_VALUE,
) -> str:
    """Format a percentage with a non-colour directional marker and sign."""
    number = _number(value)
    if number is None:
        return fallback
    if number > 0:
        marker = "▲"
    elif number < 0:
        marker = "▼"
    else:
        marker = "●"
    return f"{marker} {format_percentage(number, fallback=fallback)}"


def format_symbol_company(symbol: object, company_name: object) -> str:
    """Build a searchable security selector label without fabricating a name."""
    symbol_text = safe_display_value(symbol, fallback="")
    company_text = safe_display_value(company_name, fallback="")
    if company_text and symbol_text:
        return f"{company_text} ({symbol_text})"
    return company_text or symbol_text or MISSING_VALUE


def format_company_summary_for_display(data: pd.DataFrame) -> pd.DataFrame:
    """Return a readable company-list copy without exposing backend enums."""

    def values(column: str) -> pd.Series:
        if column in data:
            return data[column]
        return pd.Series(pd.NA, index=data.index, dtype="object")

    display_data = pd.DataFrame(
        {
            "Symbol": values("symbol").map(
                lambda value: safe_display_value(value)
            ),
            "Company Name": values("company_name").map(
                lambda value: safe_display_value(value)
            ),
            "Security Type": values("security_type").map(status_label),
            "Sector": values("sector").map(
                lambda value: safe_display_value(value)
            ),
            "Official Status": values("official_status").map(status_label),
            "Trading Activity": values("activity_status").map(status_label),
            "Lifecycle Status": values("lifecycle_status").map(status_label),
            "First Seen": pd.to_datetime(
                values("first_seen_date"),
                errors="coerce",
            ),
            "Last Seen": pd.to_datetime(
                values("last_seen_date"),
                errors="coerce",
            ),
            "Trading Days": pd.to_numeric(
                values("trading_days"),
                errors="coerce",
            ).astype("Int64"),
            "Latest Close": pd.to_numeric(
                values("latest_close"),
                errors="coerce",
            ),
        }
    )
    return display_data.loc[:, COMPANY_SUMMARY_DISPLAY_COLUMNS].reset_index(drop=True)


def company_summary_csv_bytes(data: pd.DataFrame) -> bytes:
    """Export every company-summary row using readable display labels."""
    display_data = format_company_summary_for_display(data)
    return display_data.to_csv(
        index=False,
        date_format="%Y-%m-%d",
    ).encode("utf-8")


def format_model_registry_for_display(data: pd.DataFrame) -> pd.DataFrame:
    """Return readable model-registry columns without changing stored enums."""
    if data.empty:
        return pd.DataFrame(
            columns=(
                "Model scope",
                "Symbol",
                "Version",
                "Model status",
                "Last trained",
                "Training start",
                "Training end",
                "New trading days",
                "Retraining recommendation",
            )
        )
    display = data.copy()
    return pd.DataFrame(
        {
            "Model scope": display["model_scope"].map(status_label),
            "Symbol": display["symbol"].map(
                lambda value: safe_display_value(value, fallback="Master model")
            ),
            "Version": display["model_version"].map(format_integer),
            "Model status": display["model_status"].map(status_label),
            "Last trained": display["last_trained_at"].map(format_datetime),
            "Training start": display["training_data_start"].map(format_date),
            "Training end": display["training_data_end"].map(format_date),
            "New trading days": display["new_data_days"].map(format_integer),
            "Retraining recommendation": display["needs_retraining"].map(
                lambda value: (
                    "Recommended"
                    if value is True
                    or str(value).strip().lower() in {"true", "1"}
                    else "Not needed"
                )
            ),
        }
    )
