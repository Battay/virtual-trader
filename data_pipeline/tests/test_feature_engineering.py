"""Deterministic offline tests for causal, symbol-isolated features."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering.indicators import (
    FeatureCalculationError,
    calculate_features,
)
from feature_engineering.schemas import FEATURE_COLUMNS, FEATURE_VERSION


def _history(symbol: str, rows: int, start: float = 1.0) -> pd.DataFrame:
    close = pd.Series(np.arange(start, start + rows), dtype=float)
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2025-01-01", periods=rows),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100,
        }
    )


def test_feature_calculations_are_isolated_by_symbol() -> None:
    mcb = _history("MCB", 60, 10.0)
    ogdc = _history("OGDC", 60, 1_000.0)

    combined = calculate_features(pd.concat([ogdc, mcb], ignore_index=True))
    mcb_combined = combined.loc[combined["symbol"] == "MCB"].reset_index(drop=True)
    mcb_alone = calculate_features(mcb)

    pd.testing.assert_frame_equal(
        mcb_combined.loc[:, FEATURE_COLUMNS],
        mcb_alone.loc[:, FEATURE_COLUMNS],
    )
    assert pd.isna(mcb_combined.loc[0, "simple_return"])
    assert pd.isna(combined.loc[combined["symbol"] == "OGDC", "simple_return"].iloc[0])


def test_adding_future_rows_does_not_change_past_features() -> None:
    complete = _history("MCB", 80, 20.0)

    prefix_features = calculate_features(complete.iloc[:60])
    complete_features = calculate_features(complete).iloc[:60].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        prefix_features.loc[:, FEATURE_COLUMNS],
        complete_features.loc[:, FEATURE_COLUMNS],
    )


def test_rsi_macd_bollinger_atr_obv_and_warmup_on_deterministic_fixture() -> None:
    featured = calculate_features(_history("MCB", 60))
    latest = featured.iloc[-1]

    assert latest["rsi_14"] == pytest.approx(100.0)
    assert latest["macd"] == pytest.approx(6.8669642870)
    assert latest["macd_signal"] == pytest.approx(6.8053137565)
    assert latest["macd_histogram"] == pytest.approx(
        latest["macd"] - latest["macd_signal"]
    )
    assert latest["bollinger_middle"] == pytest.approx(50.5)
    assert latest["bollinger_upper"] == pytest.approx(62.0325625947)
    assert latest["bollinger_lower"] == pytest.approx(38.9674374053)
    assert latest["atr_14"] == pytest.approx(2.0)
    assert latest["obv"] == pytest.approx(5_900.0)
    assert featured["is_warmup"].sum() == 49
    assert featured.loc[:18, "sma_20"].isna().all()
    assert featured.loc[:48, "sma_50"].isna().all()
    assert not featured.loc[49:, "sma_50"].isna().any()


def test_numeric_looking_symbols_remain_strings_and_version_is_deterministic() -> None:
    featured = calculate_features(_history("786", 60))

    assert featured["symbol"].dtype.name == "string"
    assert featured["symbol"].unique().tolist() == ["786"]
    assert featured["feature_version"].unique().tolist() == [FEATURE_VERSION]
    assert FEATURE_VERSION.startswith("psx-3a-")


def test_empty_data_is_supported_but_missing_required_columns_fail_clearly() -> None:
    empty = _history("MCB", 0)
    result = calculate_features(empty)

    assert result.empty
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        calculate_features(pd.DataFrame({"symbol": ["MCB"]}))
