"""Canonical, leakage-safe period analysis for official PSX index series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


INDEX_PERIOD_VERSION = "index_period_v1"
INDEX_PERIOD_OPTIONS = ("1M", "3M", "6M", "1Y", "Maximum")
_INDEX_PERIOD_OFFSETS: Mapping[str, pd.DateOffset] = MappingProxyType(
    {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
    }
)


@dataclass(frozen=True)
class IndexPeriodMetadata:
    """Immutable identity and actual coverage of one requested index window."""

    contract_version: str
    index_code: str
    requested_period: str
    actual_start_date: date | None
    actual_end_date: date | None
    observations: int
    start_value: float | None
    end_value: float | None


@dataclass(frozen=True)
class IndexPeriodAnalysis:
    """Selected-window analytics and a causal, independently copied frame.

    ``causal_frame`` contains only rows inside the requested visible period. Its
    moving averages, drawdown, daily changes, and rolling volatility are never
    warmed up with observations before ``metadata.actual_start_date``.
    """

    metadata: IndexPeriodMetadata
    causal_frame: pd.DataFrame
    period_return_percent: float | None
    period_high: float | None
    period_low: float | None
    latest_value: float | None
    annualized_volatility_percent: float | None
    maximum_drawdown_percent: float | None
    trend_consistency_percent: float | None
    period_momentum_percent: float | None
    latest_vs_sma20_percent: float | None
    latest_vs_sma50_percent: float | None
    volume_participation_ratio: float | None


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_period(period: str) -> None:
    if period not in INDEX_PERIOD_OPTIONS:
        raise ValueError(
            f"Unsupported index period {period!r}; expected one of "
            f"{', '.join(INDEX_PERIOD_OPTIONS)}"
        )


def _normalize_index_code(index_code: str) -> str:
    normalized = str(index_code).strip().upper()
    if not normalized:
        raise ValueError("index_code cannot be empty")
    return normalized


def _empty_period_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.iloc[0:0].copy(deep=True)
    for column in ("index_code", "date", "value", "volume"):
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    return frame


def filter_index_period(
    data: pd.DataFrame,
    index_code: str,
    period: str,
) -> pd.DataFrame:
    """Return one index's own-latest, chronological period as a deep copy.

    The index is selected before its latest date is resolved. A lagging index is
    therefore never shortened merely because another index has newer data.
    """
    _validate_period(period)
    normalized_code = _normalize_index_code(index_code)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        return _empty_period_frame(data)

    required = {"index_code", "date", "value"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "index period data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = data.loc[
        data["index_code"].astype(str).str.strip().str.upper().eq(normalized_code)
    ].copy(deep=True)
    if frame.empty:
        return _empty_period_frame(data)

    frame["index_code"] = normalized_code
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    else:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values(
        "date", kind="stable"
    )
    if frame.empty:
        return _empty_period_frame(data)
    if frame["date"].duplicated().any():
        duplicate_dates = (
            frame.loc[frame["date"].duplicated(keep=False), "date"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"{normalized_code} contains duplicate period dates: "
            + ", ".join(duplicate_dates[:5])
        )

    if period != "Maximum":
        latest_date = frame["date"].iloc[-1]
        cutoff = latest_date - _INDEX_PERIOD_OFFSETS[period]
        frame = frame.loc[frame["date"].ge(cutoff)]
    return frame.reset_index(drop=True).copy(deep=True)


def combine_index_periods(
    data: pd.DataFrame,
    index_codes: Sequence[str],
    period: str,
) -> pd.DataFrame:
    """Combine independently anchored period frames for comparison charts."""
    _validate_period(period)
    frames = [filter_index_period(data, code, period) for code in index_codes]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_period_frame(data)
    return (
        pd.concat(non_empty, ignore_index=True)
        .sort_values(["date", "index_code"], kind="stable")
        .reset_index(drop=True)
    )


def _causal_frame(selected: pd.DataFrame) -> pd.DataFrame:
    frame = selected.copy(deep=True)
    if frame.empty:
        for column in (
            "daily_change",
            "daily_change_percent",
            "ma_20",
            "ma_50",
            "drawdown_percent",
            "rolling_volatility_20_percent",
        ):
            frame[column] = pd.Series(dtype="float64")
        return frame

    values = pd.to_numeric(frame["value"], errors="coerce")
    daily_returns = values.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    running_peak = values.cummax().replace(0, np.nan)
    frame["daily_change"] = values.diff()
    frame["daily_change_percent"] = daily_returns.mul(100)
    frame["ma_20"] = values.rolling(window=20, min_periods=20).mean()
    frame["ma_50"] = values.rolling(window=50, min_periods=50).mean()
    frame["drawdown_percent"] = (
        values.div(running_peak).sub(1).mul(100).replace([np.inf, -np.inf], np.nan)
    )
    frame["rolling_volatility_20_percent"] = (
        daily_returns.rolling(window=20, min_periods=20)
        .std(ddof=0)
        .mul(math.sqrt(252) * 100)
    )
    return frame.reset_index(drop=True)


def _period_momentum(daily_returns: pd.Series) -> float | None:
    """Return second-half minus first-half mean return, scaled to 21 sessions."""
    returns = daily_returns.dropna().astype(float)
    if len(returns) < 2:
        return None
    split = len(returns) // 2
    earlier = returns.iloc[:split]
    later = returns.iloc[split:]
    if earlier.empty or later.empty:
        return None
    return _finite((later.mean() - earlier.mean()) * 21 * 100)


def analyze_index_period(
    data: pd.DataFrame,
    index_code: str,
    period: str,
) -> IndexPeriodAnalysis:
    """Filter and calculate one reproducible selected-period analysis."""
    normalized_code = _normalize_index_code(index_code)
    selected = filter_index_period(data, normalized_code, period)
    causal = _causal_frame(selected)
    values = pd.to_numeric(causal["value"], errors="coerce").dropna()

    observations = int(len(values))
    start_value = _finite(values.iloc[0]) if observations else None
    end_value = _finite(values.iloc[-1]) if observations else None
    actual_start = (
        pd.Timestamp(causal["date"].iloc[0]).date() if observations else None
    )
    actual_end = (
        pd.Timestamp(causal["date"].iloc[-1]).date() if observations else None
    )
    metadata = IndexPeriodMetadata(
        contract_version=INDEX_PERIOD_VERSION,
        index_code=normalized_code,
        requested_period=period,
        actual_start_date=actual_start,
        actual_end_date=actual_end,
        observations=observations,
        start_value=start_value,
        end_value=end_value,
    )

    if observations:
        period_high = _finite(values.max())
        period_low = _finite(values.min())
        latest_value = end_value
    else:
        period_high = period_low = latest_value = None
    period_return = (
        _finite((end_value / start_value - 1) * 100)
        if observations >= 2 and start_value not in (None, 0) and end_value is not None
        else None
    )

    daily_returns = (
        pd.to_numeric(causal["daily_change_percent"], errors="coerce")
        .div(100)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    annualized_volatility = (
        _finite(daily_returns.std(ddof=0) * math.sqrt(252) * 100)
        if len(daily_returns) >= 2
        else None
    )
    drawdowns = pd.to_numeric(
        causal["drawdown_percent"], errors="coerce"
    ).dropna()
    maximum_drawdown = _finite(drawdowns.min()) if observations >= 2 else None
    trend_consistency = None
    if not daily_returns.empty:
        positive = int(daily_returns.gt(0).sum())
        flat = int(daily_returns.eq(0).sum())
        trend_consistency = _finite(
            (positive + 0.5 * flat) / len(daily_returns) * 100
        )
    momentum = _period_momentum(daily_returns)

    latest_vs_sma20 = None
    latest_vs_sma50 = None
    if end_value is not None and observations:
        latest_ma20 = _finite(causal["ma_20"].iloc[-1])
        latest_ma50 = _finite(causal["ma_50"].iloc[-1])
        if latest_ma20 not in (None, 0):
            latest_vs_sma20 = _finite((end_value / latest_ma20 - 1) * 100)
        if latest_ma50 not in (None, 0):
            latest_vs_sma50 = _finite((end_value / latest_ma50 - 1) * 100)

    volume_ratio = None
    volumes = pd.to_numeric(causal["volume"], errors="coerce").dropna()
    if len(volumes) >= 2:
        average_volume = _finite(volumes.mean())
        latest_volume = _finite(volumes.iloc[-1])
        if average_volume not in (None, 0) and latest_volume is not None:
            volume_ratio = _finite(latest_volume / average_volume)

    return IndexPeriodAnalysis(
        metadata=metadata,
        causal_frame=causal.copy(deep=True),
        period_return_percent=period_return,
        period_high=period_high,
        period_low=period_low,
        latest_value=latest_value,
        annualized_volatility_percent=annualized_volatility,
        maximum_drawdown_percent=maximum_drawdown,
        trend_consistency_percent=trend_consistency,
        period_momentum_percent=momentum,
        latest_vs_sma20_percent=latest_vs_sma20,
        latest_vs_sma50_percent=latest_vs_sma50,
        volume_participation_ratio=volume_ratio,
    )


def analyze_index_periods(
    data: pd.DataFrame,
    index_codes: Sequence[str],
    period: str,
) -> dict[str, IndexPeriodAnalysis]:
    """Calculate independent period analyses for an ordered index collection."""
    _validate_period(period)
    return {
        _normalize_index_code(code): analyze_index_period(data, code, period)
        for code in index_codes
    }
