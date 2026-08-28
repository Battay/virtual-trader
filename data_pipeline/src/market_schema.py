"""Canonical market-schema compatibility helpers.

The native on-disk contract uses ``market_date``.  Older analytical code uses
``date`` internally.  Compatibility is applied only in memory so the canonical
master is never rewritten into the legacy schema.
"""

from __future__ import annotations

import pandas as pd


class MarketSchemaError(ValueError):
    """Raised when date columns are absent or disagree."""


def with_legacy_date_alias(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose canonical date is available as legacy ``date``.

    If both columns exist, they must describe the same dates.  The canonical
    ``market_date`` column is retained when already present so sector/native
    provenance remains inspectable by modern consumers.
    """

    if "date" not in data.columns and "market_date" not in data.columns:
        raise MarketSchemaError("Market data has neither market_date nor date")
    normalized = data.copy()
    if "market_date" in normalized.columns:
        canonical = pd.to_datetime(normalized["market_date"], errors="coerce")
        if canonical.isna().any():
            raise MarketSchemaError("market_date contains invalid values")
        if "date" in normalized.columns:
            legacy = pd.to_datetime(normalized["date"], errors="coerce")
            if legacy.isna().any() or not canonical.equals(legacy):
                raise MarketSchemaError("market_date and date disagree")
        else:
            normalized["date"] = canonical
    return normalized
