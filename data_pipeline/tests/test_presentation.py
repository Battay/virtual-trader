"""Offline tests for shared dashboard presentation helpers."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from dashboard.presentation import (
    MISSING_VALUE,
    column_label,
    enum_display_options,
    format_date,
    format_datetime,
    format_decimal,
    format_integer,
    format_percentage,
    format_price,
    format_source,
    format_symbol_company,
    format_volume,
    humanize_enum,
    safe_display_value,
    selected_option_values,
    status_label,
)


@pytest.mark.parametrize(
    ("backend_value", "expected"),
    [
        ("historical_only", "Historical Only"),
        ("listed_recently_traded", "Listed · Recently Traded"),
        ("listed_not_recently_traded", "Listed · Not Recently Traded"),
        ("never_seen_in_market_history", "No Local Trading History"),
        ("non_compliant", "Non-Compliant"),
        ("ordinary_equity", "Ordinary Equity"),
        ("preference_share", "Preference Share"),
        ("missing_processed_dataset", "Missing Processed Dataset"),
    ],
)
def test_explicit_enum_labels(backend_value: str, expected: str) -> None:
    assert humanize_enum(backend_value) == expected
    assert status_label(backend_value) == expected


def test_fallback_snake_case_humanization() -> None:
    assert humanize_enum("awaiting_official_review") == "Awaiting Official Review"
    assert column_label("change_percent") == "Change %"


@pytest.mark.parametrize("value", [None, float("nan"), pd.NaT, "", "nan", "NaT"])
def test_missing_values_never_leak_python_or_pandas_markers(value: object) -> None:
    assert safe_display_value(value) == MISSING_VALUE
    assert format_date(value) == MISSING_VALUE
    assert format_price(value) == MISSING_VALUE


def test_date_and_datetime_formatting_in_pakistan_time() -> None:
    assert format_date(date(2026, 7, 30)) == "30 Jul 2026"
    assert format_date("malformed") == MISSING_VALUE
    utc_value = datetime(2026, 7, 30, 12, 13, tzinfo=timezone.utc)
    assert format_datetime(utc_value) == "30 Jul 2026, 5:13 PM PKT"
    assert format_datetime("malformed") == MISSING_VALUE


def test_numeric_price_volume_and_percentage_formatting() -> None:
    assert format_integer(1_234_567) == "1,234,567"
    assert format_decimal(12.3456) == "12.35"
    assert format_price(1_234.5) == "PKR 1,234.50"
    assert format_price(-2.5, show_sign=True) == "PKR -2.50"
    assert format_volume(80_400_000) == "80.4M"
    assert format_volume(925_300) == "925.3K"
    assert format_percentage(1.234) == "+1.23%"


def test_source_formatting_preserves_urls_and_humanizes_local_tags() -> None:
    assert format_source("master_market_history") == "Master Market History"
    assert format_source("https://dps.psx.com.pk/listings") == (
        "https://dps.psx.com.pk/listings"
    )
    assert format_source("official_source; manual_override") == (
        "Official Source; Manual Override"
    )


def test_company_symbol_selector_label_uses_available_context() -> None:
    assert format_symbol_company("MCB", "MCB Bank Limited") == "MCB Bank Limited (MCB)"
    assert format_symbol_company("786", "") == "786"
    assert format_symbol_company("", "Example Security") == "Example Security"


def test_filter_options_keep_backend_values_but_never_expose_snake_case_labels() -> None:
    options = enum_display_options(
        ["historical_only", "ordinary_equity", "historical_only"]
    )

    assert selected_option_values(options) == ("historical_only", "ordinary_equity")
    assert [option.label for option in options] == [
        "Historical Only",
        "Ordinary Equity",
    ]
    assert all("_" not in option.label for option in options)
