"""Offline tests for shared Streamlit registry filtering helpers."""

from datetime import date

import pandas as pd

from dashboard.registry_loader import (
    filter_company_registry,
    restrict_market_data_by_registry,
    summarize_registry_for_display,
)


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "786",
                "company_name": "786 Investments Limited",
                "officially_listed": True,
                "official_status": "listed",
                "activity_status": "recently_traded",
                "lifecycle_status": "listed_recently_traded",
                "security_type": "ordinary_equity",
                "sector": "Investment",
                "board": "Main",
                "listing_segment": "normal_counter",
                "first_seen_date": "2026-07-01",
                "last_seen_date": "2026-07-29",
                "trading_days": 20,
                "is_new_listing": True,
            },
            {
                "symbol": "ACIETF",
                "company_name": "Alfalah Consumer Index ETF",
                "officially_listed": True,
                "official_status": "listed",
                "activity_status": "not_recently_traded",
                "lifecycle_status": "listed_not_recently_traded",
                "security_type": "etf",
                "sector": "ETF",
                "board": "Main",
                "listing_segment": "normal_counter",
                "first_seen_date": "2026-01-01",
                "last_seen_date": "2026-05-01",
                "trading_days": 10,
                "is_new_listing": False,
            },
            {
                "symbol": "FUT-JUL",
                "company_name": "",
                "officially_listed": False,
                "official_status": "historical",
                "activity_status": "recently_traded",
                "lifecycle_status": "historical_only",
                "security_type": "other",
                "sector": "",
                "board": "",
                "listing_segment": "",
                "first_seen_date": "2026-07-20",
                "last_seen_date": "2026-07-29",
                "trading_days": 5,
                "is_new_listing": False,
            },
        ]
    )


def test_filters_registry_by_search_status_type_and_dates() -> None:
    filtered = filter_company_registry(
        _registry(),
        search="investments",
        lifecycle_statuses={"listed_recently_traded"},
        security_types={"ordinary_equity"},
        first_seen_start=date(2026, 7, 1),
        first_seen_end=date(2026, 7, 31),
        last_seen_start=date(2026, 7, 1),
        last_seen_end=date(2026, 7, 31),
    )

    assert filtered["symbol"].tolist() == ["786"]


def test_registry_filter_preserves_numeric_looking_symbol() -> None:
    filtered = filter_company_registry(_registry(), search="786")

    assert filtered["symbol"].tolist() == ["786"]


def test_market_registry_filter_is_optional_and_keeps_historical_access() -> None:
    market = pd.DataFrame(
        [
            {"symbol": "786", "date": "2026-07-29"},
            {"symbol": "ACIETF", "date": "2026-07-29"},
            {"symbol": "FUT-JUL", "date": "2026-07-29"},
        ]
    )

    unfiltered = restrict_market_data_by_registry(market, _registry())
    ordinary = restrict_market_data_by_registry(
        market,
        _registry(),
        officially_listed_only=True,
        ordinary_equities_only=True,
    )

    assert unfiltered["symbol"].tolist() == ["786", "ACIETF", "FUT-JUL"]
    assert ordinary["symbol"].tolist() == ["786"]


def test_registry_filter_applies_minimum_trading_days() -> None:
    filtered = filter_company_registry(_registry(), minimum_trading_days=10)

    assert filtered["symbol"].tolist() == ["786", "ACIETF"]


def test_registry_display_metrics_separate_listed_recent_from_historical_recent() -> None:
    metrics = summarize_registry_for_display(_registry())

    assert metrics.total_securities == 3
    assert metrics.currently_listed == 2
    assert metrics.listed_and_recently_traded == 1
    assert metrics.historical_only == 1
    assert metrics.listed_not_recently_traded == 1
    assert metrics.new_listings == 1
