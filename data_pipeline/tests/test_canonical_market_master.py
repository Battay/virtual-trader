"""Canonical market-master contract and compatibility tests."""

import pandas as pd
import pytest

from data_pipeline.src.config import (
    LEGACY_MARKET_COMPAT_PATH,
    MASTER_CSV_PATH,
    NATIVE_MARKET_MASTER_PATH,
)
from data_pipeline.src.market_schema import MarketSchemaError, with_legacy_date_alias
from data_pipeline.src.native_market_pipeline import CANONICAL_MARKET_COLUMNS


def test_native_and_user_facing_master_are_one_path() -> None:
    assert NATIVE_MARKET_MASTER_PATH == MASTER_CSV_PATH
    assert LEGACY_MARKET_COMPAT_PATH != MASTER_CSV_PATH
    assert MASTER_CSV_PATH.name == "psx_master.csv"


def test_canonical_market_date_is_aliased_only_in_memory() -> None:
    source = pd.DataFrame(
        {
            "market_date": ["2026-08-27"],
            "symbol": ["MCB"],
        }
    )

    normalized = with_legacy_date_alias(source)

    assert "date" not in source.columns
    assert normalized["date"].dt.date.astype(str).tolist() == ["2026-08-27"]
    assert normalized["market_date"].tolist() == ["2026-08-27"]


def test_conflicting_date_alias_fails_closed() -> None:
    source = pd.DataFrame(
        {
            "market_date": ["2026-08-27"],
            "date": ["2026-08-26"],
        }
    )

    with pytest.raises(MarketSchemaError, match="disagree"):
        with_legacy_date_alias(source)


def test_canonical_columns_include_sector_provenance() -> None:
    assert CANONICAL_MARKET_COLUMNS[-3:] == (
        "sector_current",
        "sector_source",
        "sector_snapshot_date",
    )
