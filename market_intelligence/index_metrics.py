"""Reusable trading-observation-based PSX index metrics."""

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd


@dataclass(frozen=True)
class IndexMetrics:
    index_code: str
    latest_value: float | None
    latest_date: date | None
    latest_daily_change: float | None
    latest_daily_change_percent: float | None
    previous_value: float | None
    one_week_return: float | None
    one_month_return: float | None
    three_month_return: float | None
    six_month_return: float | None
    year_to_date_return: float | None
    one_year_return: float | None
    maximum_return: float | None
    rolling_volatility_20: float | None
    latest_volume: int | None
    average_volume_20: float | None
    versus_ma_20_percent: float | None
    versus_ma_50_percent: float | None


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _observation_return(values: pd.Series, observations: int) -> float | None:
    if len(values) <= observations:
        return None
    prior = _finite(values.iloc[-observations - 1])
    latest = _finite(values.iloc[-1])
    if prior in (None, 0) or latest is None:
        return None
    return (latest / prior - 1) * 100


def calculate_index_metrics(data: pd.DataFrame, index_code: str) -> IndexMetrics:
    frame = data.loc[data["index_code"].astype(str) == index_code].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date", kind="stable")
    if frame.empty:
        return IndexMetrics(index_code, *([None] * 17))
    values = frame["value"]
    latest = _finite(values.iloc[-1])
    previous = _finite(values.iloc[-2]) if len(values) >= 2 else None
    change = latest - previous if latest is not None and previous is not None else None
    change_percent = change / previous * 100 if change is not None and previous else None
    ytd = frame.loc[frame["date"].dt.year == frame["date"].iloc[-1].year, "value"]
    ytd_return = None
    if len(ytd) >= 2 and ytd.iloc[0] != 0:
        ytd_return = (ytd.iloc[-1] / ytd.iloc[0] - 1) * 100
    returns = values.pct_change().dropna()
    volatility = returns.tail(20).std(ddof=0) * (252 ** 0.5) * 100 if len(returns) >= 20 else None
    def versus(window: int) -> float | None:
        if len(values) < window or latest is None:
            return None
        average = values.tail(window).mean()
        return (latest / average - 1) * 100 if average else None
    return IndexMetrics(
        index_code=index_code, latest_value=latest,
        latest_date=frame["date"].iloc[-1].date(), latest_daily_change=change,
        latest_daily_change_percent=change_percent, previous_value=previous,
        one_week_return=_observation_return(values, 5),
        one_month_return=_observation_return(values, 21),
        three_month_return=_observation_return(values, 63),
        six_month_return=_observation_return(values, 126),
        year_to_date_return=_finite(ytd_return), one_year_return=_observation_return(values, 252),
        maximum_return=_observation_return(values, len(values) - 1),
        rolling_volatility_20=_finite(volatility),
        latest_volume=int(frame["volume"].iloc[-1]) if pd.notna(frame["volume"].iloc[-1]) else None,
        average_volume_20=_finite(frame["volume"].tail(20).mean()) if len(frame) >= 20 else None,
        versus_ma_20_percent=versus(20), versus_ma_50_percent=versus(50),
    )
