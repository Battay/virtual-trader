"""Select registry symbols eligible for future model training."""

from collections.abc import Collection

import pandas as pd


def select_training_universe(
    registry: pd.DataFrame,
    *,
    officially_listed_only: bool = True,
    recently_traded_only: bool = True,
    security_types: Collection[str] | None = frozenset({"ordinary_equity"}),
    minimum_trading_days: int | None = None,
) -> tuple[str, ...]:
    """Return deterministic eligible symbols without modifying market data."""
    required = {"symbol", "officially_listed", "activity_status", "security_type"}
    if minimum_trading_days is not None:
        if minimum_trading_days < 0:
            raise ValueError("minimum_trading_days cannot be negative")
        required.add("trading_days")
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(
            f"Registry is missing required columns: {', '.join(missing)}"
        )

    eligible = registry.copy()
    if officially_listed_only:
        listed = eligible["officially_listed"]
        if not pd.api.types.is_bool_dtype(listed):
            listed = listed.astype("string").str.lower().isin({"true", "1"})
        eligible = eligible.loc[listed]
    if recently_traded_only:
        eligible = eligible.loc[
            eligible["activity_status"].astype("string") == "recently_traded"
        ]
    if security_types is not None:
        eligible = eligible.loc[
            eligible["security_type"].astype("string").isin(set(security_types))
        ]
    if minimum_trading_days is not None:
        trading_days = pd.to_numeric(eligible["trading_days"], errors="coerce")
        eligible = eligible.loc[trading_days >= minimum_trading_days]

    symbols = (
        eligible["symbol"]
        .astype("string")
        .str.strip()
        .dropna()
    )
    symbols = symbols.loc[symbols != ""]
    return tuple(sorted(set(str(symbol) for symbol in symbols)))
