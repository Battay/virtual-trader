"""Offline regression tests for row-level AI OHLC quality handling."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.dataset_builder import (
    build_master_ai_dataset,
    build_symbol_datasets,
)
from feature_engineering.indicators import calculate_features
from feature_engineering.preprocessing import (
    fatal_quality_errors_by_symbol,
    filter_ai_quality_rows,
)
from feature_engineering.readiness import build_training_readiness_report


def _market(rows: int = 310, symbol: str = "MCB") -> pd.DataFrame:
    close = pd.Series(np.arange(100.0, 100.0 + rows))
    return pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * rows, dtype="string"),
            "date": pd.date_range("2024-01-01", periods=rows),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000,
        }
    )


def _registry(symbol: str = "MCB") -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "symbol": symbol,
            "company_name": f"{symbol} Limited",
            "officially_listed": True,
            "activity_status": "recently_traded",
            "official_status": "listed",
            "lifecycle_status": "listed_recently_traded",
            "security_type": "ordinary_equity",
        }]
    )


def test_invalid_ohlc_rows_are_removed_before_features_without_imputation() -> None:
    source = _market(80)
    original = source.copy(deep=True)
    source.loc[10, "open"] = 0
    source.loc[20, "high"] = 0
    source.loc[30, "low"] = 0
    source.loc[40, "close"] = 0

    quality = filter_ai_quality_rows(source)
    expected = original.drop(index=[10, 20, 30, 40]).reset_index(drop=True)

    assert quality.metadata.iloc[0]["invalid_ohlc_rows_removed"] == 4
    assert quality.metadata.iloc[0]["usable_pre_feature_rows"] == 76
    assert quality.metadata.iloc[0]["quality_removal_reason"] == "invalid_ohlc_row"
    assert quality.data["date"].is_monotonic_increasing
    assert (quality.data[["open", "high", "low", "close"]] > 0).all(axis=None)
    pd.testing.assert_frame_equal(
        quality.data.reset_index(drop=True), expected, check_dtype=False
    )
    pd.testing.assert_frame_equal(source, original.assign(
        open=lambda frame: frame["open"].mask(frame.index == 10, 0),
        high=lambda frame: frame["high"].mask(frame.index == 20, 0),
        low=lambda frame: frame["low"].mask(frame.index == 30, 0),
        close=lambda frame: frame["close"].mask(frame.index == 40, 0),
    ))


def test_one_invalid_row_does_not_exclude_valid_symbol_and_builders_agree(
    tmp_path: Path,
) -> None:
    source = _market(310)
    source.loc[25, "open"] = 0
    symbol_dir = tmp_path / "symbols"
    master_path = tmp_path / "master.csv"

    symbol_metrics = build_symbol_datasets(
        market_data=source,
        registry=_registry(),
        minimum_usable_rows=252,
        include_market_context=False,
        output_dir=symbol_dir,
    )
    master_metrics = build_master_ai_dataset(
        market_data=source,
        registry=_registry(),
        include_market_context=False,
        output_path=master_path,
    )
    symbol_data = pd.read_csv(symbol_dir / "MCB.csv")
    master_data = pd.read_csv(master_path)

    assert symbol_metrics.invalid_ohlc_rows_removed == 1
    assert master_metrics.invalid_ohlc_rows_removed == 1
    assert symbol_metrics.unique_symbols == 1
    assert len(symbol_data) == 260
    assert not symbol_data["date"].eq(source.loc[25, "date"].date().isoformat()).any()
    assert not symbol_data.duplicated(["symbol", "date"]).any()
    pd.testing.assert_frame_equal(symbol_data, master_data, check_dtype=False)


def test_minimum_history_uses_rows_remaining_after_quality_filter(tmp_path: Path) -> None:
    source = _market(301)
    source.loc[100, "low"] = 0
    metrics = build_symbol_datasets(
        market_data=source,
        registry=_registry(),
        minimum_usable_rows=252,
        include_market_context=False,
        output_dir=tmp_path,
    )
    report = build_training_readiness_report(
        source,
        _registry(),
        minimum_usable_rows=252,
        processed_symbols_dir=tmp_path,
    )
    row = report.iloc[0]

    assert metrics.unique_symbols == 0
    assert not (tmp_path / "MCB.csv").exists()
    assert row["raw_trading_rows"] == 301
    assert row["invalid_ohlc_rows_removed"] == 1
    assert row["usable_pre_feature_rows"] == 300
    assert row["post_warmup_rows"] == 251
    assert row["usable_feature_rows"] == 251
    assert row["readiness_status"] == "Insufficient History"


def test_negative_price_is_fatal_and_never_enters_ai_features(tmp_path: Path) -> None:
    source = _market(310)
    source.loc[100, "high"] = -1

    assert fatal_quality_errors_by_symbol(source)["MCB"] == ("price is negative",)
    metrics = build_symbol_datasets(
        market_data=source,
        registry=_registry(),
        minimum_usable_rows=1,
        include_market_context=False,
        output_dir=tmp_path,
    )

    assert metrics.unique_symbols == 0
    assert metrics.invalid_ohlc_rows_removed == 1
    assert not (tmp_path / "MCB.csv").exists()


def test_filtered_indicator_pipeline_remains_causal() -> None:
    complete = _market(90)
    complete.loc[15, "open"] = 0
    prefix = complete.iloc[:70].copy()

    prefix_features = calculate_features(filter_ai_quality_rows(prefix).data)
    complete_features = calculate_features(filter_ai_quality_rows(complete).data)
    past = complete_features.loc[
        complete_features["date"] <= prefix["date"].max()
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(prefix_features, past)
