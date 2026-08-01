"""Offline tests for company-first Dataset Explorer helpers."""

from io import BytesIO

import pandas as pd

from dashboard.data_loader import (
    build_company_summary,
    filter_security_history,
    history_csv_bytes,
    paginate_dataframe,
    sort_history_newest_first,
)
from dashboard.presentation import (
    company_summary_csv_bytes,
    format_company_summary_for_display,
)
from dashboard.registry_loader import filter_company_registry


def _market_row(
    symbol: str,
    trading_date: str,
    close: float,
    volume: int = 100,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": trading_date,
        "ldcp": close - 1,
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "change": 1.0,
        "change_percent": 1.0,
        "volume": volume,
    }


def _registry_row(
    symbol: str,
    company_name: str,
    *,
    officially_listed: bool,
    official_status: str,
    activity_status: str,
    lifecycle_status: str,
    security_type: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "company_name": company_name,
        "security_type": security_type,
        "sector": "Commercial Banks",
        "official_status": official_status,
        "activity_status": activity_status,
        "lifecycle_status": lifecycle_status,
        "officially_listed": officially_listed,
        "board": "main",
        "listing_segment": "normal_counter",
    }


def _sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.DataFrame(
        [
            _market_row("MCB", "2026-07-30", 305.0, 300),
            _market_row("786", "2026-07-29", 21.0, 100),
            _market_row("MCB", "2026-07-28", 300.0, 200),
            _market_row("OGDC", "2026-07-29", 220.0, 400),
            _market_row("786", "2026-07-30", 22.0, 150),
        ]
    )
    registry = pd.DataFrame(
        [
            _registry_row(
                "MCB",
                "MCB Bank Limited",
                officially_listed=True,
                official_status="listed",
                activity_status="recently_traded",
                lifecycle_status="listed_recently_traded",
                security_type="ordinary_equity",
            ),
            _registry_row(
                "786",
                "786 Investments Limited",
                officially_listed=True,
                official_status="listed",
                activity_status="not_recently_traded",
                lifecycle_status="listed_not_recently_traded",
                security_type="ordinary_equity",
            ),
            _registry_row(
                "OGDC",
                "Oil & Gas Development Company",
                officially_listed=False,
                official_status="historical",
                activity_status="recently_traded",
                lifecycle_status="historical_only",
                security_type="ordinary_equity",
            ),
        ]
    )
    return market, registry


def test_company_summary_has_one_row_per_symbol_and_correct_history_metrics() -> None:
    market, registry = _sample_data()
    original_market = market.copy(deep=True)
    original_registry = registry.copy(deep=True)

    summary = build_company_summary(market, registry)
    mcb = summary.loc[summary["symbol"] == "MCB"].iloc[0]

    assert summary["symbol"].tolist() == ["786", "MCB", "OGDC"]
    assert summary["symbol"].is_unique
    assert mcb["first_seen_date"] == pd.Timestamp("2026-07-28")
    assert mcb["last_seen_date"] == pd.Timestamp("2026-07-30")
    assert mcb["trading_days"] == 2
    assert mcb["latest_close"] == 305.0
    pd.testing.assert_frame_equal(market, original_market)
    pd.testing.assert_frame_equal(registry, original_registry)


def test_702_historical_symbols_produce_702_company_rows() -> None:
    market = pd.DataFrame(
        [
            _market_row(str(index), "2026-07-30", float(index))
            for index in range(702)
        ]
    )

    summary = build_company_summary(market)

    assert len(summary) == 702
    assert summary["symbol"].nunique() == 702
    assert pd.api.types.is_string_dtype(summary["symbol"])


def test_company_filters_reduce_the_company_list_before_selection() -> None:
    market, registry = _sample_data()
    summary = build_company_summary(market, registry)

    filtered = filter_company_registry(
        summary,
        search="bank",
        officially_listed_only=True,
        lifecycle_statuses={"listed_recently_traded"},
        minimum_trading_days=2,
    )

    assert filtered["symbol"].tolist() == ["MCB"]


def test_numeric_looking_symbols_remain_strings_in_company_summary() -> None:
    market, registry = _sample_data()

    summary = build_company_summary(market, registry)

    numeric_symbol = summary.loc[summary["symbol"] == "786", "symbol"].iloc[0]
    assert isinstance(numeric_symbol, str)
    assert numeric_symbol == "786"


def test_selected_company_rows_are_isolated_newest_first_and_paginated() -> None:
    market, _ = _sample_data()

    selected = filter_security_history(market, "MCB")
    newest_first = sort_history_newest_first(selected)
    first_page = paginate_dataframe(newest_first, page_number=1, rows_per_page=1)

    assert selected["symbol"].unique().tolist() == ["MCB"]
    assert newest_first["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-30",
        "2026-07-28",
    ]
    assert first_page["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-30"
    ]


def test_selected_company_export_contains_complete_history_not_visible_page() -> None:
    market, _ = _sample_data()
    selected = sort_history_newest_first(filter_security_history(market, "MCB"))
    visible_page = paginate_dataframe(selected, page_number=1, rows_per_page=1)

    exported = pd.read_csv(BytesIO(history_csv_bytes(selected)))

    assert len(visible_page) == 1
    assert len(exported) == 2
    assert exported["symbol"].unique().tolist() == ["MCB"]


def test_filtered_company_export_has_one_readable_row_per_remaining_symbol() -> None:
    market, registry = _sample_data()
    summary = build_company_summary(market, registry)
    filtered = filter_company_registry(summary, officially_listed_only=True)
    original = filtered.copy(deep=True)

    exported = pd.read_csv(
        BytesIO(company_summary_csv_bytes(filtered)),
        dtype={"Symbol": "string"},
    )

    assert exported["Symbol"].tolist() == ["786", "MCB"]
    assert exported["Symbol"].is_unique
    assert exported["Security Type"].tolist() == [
        "Ordinary Equity",
        "Ordinary Equity",
    ]
    assert exported["Lifecycle Status"].tolist() == [
        "Listed · Not Recently Traded",
        "Listed · Recently Traded",
    ]
    pd.testing.assert_frame_equal(filtered, original)


def test_company_display_contains_no_raw_snake_case_labels() -> None:
    market, registry = _sample_data()
    display = format_company_summary_for_display(
        build_company_summary(market, registry)
    )
    displayed_text = " ".join(
        display.astype("string").fillna("").to_numpy().ravel().tolist()
    )

    assert "historical_only" not in displayed_text
    assert "listed_recently_traded" not in displayed_text
    assert "ordinary_equity" not in displayed_text
    assert "Historical Only" in displayed_text
    assert "Listed · Recently Traded" in displayed_text
    assert "Ordinary Equity" in displayed_text
