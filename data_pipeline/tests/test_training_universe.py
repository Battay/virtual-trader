"""Tests for future-model training-universe symbol selection."""

import pandas as pd

from data_pipeline.src.training_universe import select_training_universe


def test_selects_only_eligible_symbols_without_modifying_registry() -> None:
    registry = pd.DataFrame(
        [
            {
                "symbol": "786",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "security_type": "ordinary_equity",
                "trading_days": 40,
            },
            {
                "symbol": "ACIETF",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "security_type": "etf",
                "trading_days": 40,
            },
            {
                "symbol": "OLD",
                "officially_listed": False,
                "activity_status": "not_recently_traded",
                "security_type": "ordinary_equity",
                "trading_days": 100,
            },
            {
                "symbol": "SHORT",
                "officially_listed": True,
                "activity_status": "recently_traded",
                "security_type": "ordinary_equity",
                "trading_days": 5,
            },
        ]
    )
    original = registry.copy(deep=True)

    symbols = select_training_universe(registry, minimum_trading_days=30)

    assert symbols == ("786",)
    pd.testing.assert_frame_equal(registry, original)


def test_training_universe_filters_are_configurable() -> None:
    registry = pd.DataFrame(
        [
            {
                "symbol": "ETF",
                "officially_listed": True,
                "activity_status": "not_recently_traded",
                "security_type": "etf",
                "trading_days": 2,
            },
            {
                "symbol": "HIST",
                "officially_listed": False,
                "activity_status": "recently_traded",
                "security_type": "unknown",
                "trading_days": 3,
            },
        ]
    )

    symbols = select_training_universe(
        registry,
        officially_listed_only=False,
        recently_traded_only=False,
        security_types=None,
    )

    assert symbols == ("ETF", "HIST")
