"""Backward-looking, symbol-isolated PSX feature calculations."""

import numpy as np
import pandas as pd

from .schemas import (
    FEATURE_VERSION,
    FEATURE_WARMUP_ROWS,
    RAW_OHLCV_COLUMNS,
    RAW_REQUIRED_COLUMNS,
)


class FeatureCalculationError(ValueError):
    """Raised when market data cannot be safely feature engineered."""


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator != 0)
    return numerator / valid_denominator


def _relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    average_loss = loss.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    relative_strength = _safe_ratio(average_gain, average_loss)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    contribution = direction * volume
    if not contribution.empty and pd.notna(close.iloc[0]) and pd.notna(volume.iloc[0]):
        contribution.iloc[0] = 0.0
    return contribution.cumsum(skipna=False)


def calculate_symbol_features(symbol_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal features for one symbol in chronological order."""
    missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(symbol_data.columns))
    if missing:
        raise FeatureCalculationError(
            f"Market data is missing required columns: {', '.join(missing)}"
        )

    data = symbol_data.copy()
    data["symbol"] = data["symbol"].astype("string").str.strip()
    symbols = data["symbol"].dropna().unique()
    if len(symbols) > 1:
        raise FeatureCalculationError(
            "calculate_symbol_features accepts exactly one symbol"
        )
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    for column in RAW_OHLCV_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    close = data["close"]
    previous_close = close.shift(1)
    data["simple_return"] = _safe_ratio(close, previous_close) - 1
    positive_ratio = _safe_ratio(close.where(close > 0), previous_close.where(previous_close > 0))
    data["log_return"] = np.log(positive_ratio)
    data["high_low_range"] = data["high"] - data["low"]
    data["open_close_return"] = _safe_ratio(
        close - data["open"],
        data["open"],
    )
    data["rolling_volatility_20"] = data["simple_return"].rolling(
        window=20,
        min_periods=20,
    ).std()

    data["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    data["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    data["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    data["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    data["rsi_14"] = _relative_strength_index(close)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()
    data["macd_histogram"] = data["macd"] - data["macd_signal"]

    bollinger_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    data["bollinger_middle"] = data["sma_20"]
    data["bollinger_upper"] = data["bollinger_middle"] + 2 * bollinger_std
    data["bollinger_lower"] = data["bollinger_middle"] - 2 * bollinger_std

    true_range = pd.concat(
        (
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1, skipna=True)
    data["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    data["obv"] = _on_balance_volume(close, data["volume"])
    data["volume_ma_20"] = data["volume"].rolling(
        window=20,
        min_periods=20,
    ).mean()
    data["is_warmup"] = data.index < FEATURE_WARMUP_ROWS
    data["feature_version"] = FEATURE_VERSION
    return data


def calculate_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate features independently per symbol and return symbol/date order."""
    missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(market_data.columns))
    if missing:
        raise FeatureCalculationError(
            f"Market data is missing required columns: {', '.join(missing)}"
        )
    if market_data.empty:
        return calculate_symbol_features(market_data)

    data = market_data.copy()
    data["symbol"] = data["symbol"].astype("string").str.strip()
    calculated = [
        calculate_symbol_features(group)
        for _, group in data.groupby("symbol", sort=True, dropna=False)
    ]
    return pd.concat(calculated, ignore_index=True).sort_values(
        ["symbol", "date"],
        kind="stable",
    ).reset_index(drop=True)
