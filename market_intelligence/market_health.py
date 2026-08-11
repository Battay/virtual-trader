"""Transparent rule-based descriptive market-condition score."""

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from .index_metrics import IndexMetrics
from .index_periods import IndexPeriodAnalysis
from .market_breadth import MarketBreadth

MARKET_HEALTH_WEIGHTS = {
    "KSE100 trend": 15, "KSE30 trend": 10, "KMI30 trend": 10,
    "All Share trend": 15, "Advance/decline ratio": 15,
    "Advancing securities": 10, "Volume participation": 10,
    "Moving-average position": 10, "Volatility penalty": 5,
}
assert sum(MARKET_HEALTH_WEIGHTS.values()) == 100

# Each index uses this same transparent 100-point model. Missing components are
# excluded and the available weights are rescaled to 100.
INDEX_HEALTH_WEIGHTS = {
    "1-day direction": 10,
    "5-trading-day return": 15,
    "20-trading-day return": 20,
    "Value relative to SMA-20": 15,
    "Value relative to SMA-50": 15,
    "Short-term momentum": 10,
    "Rolling volatility penalty": 10,
    "Volume participation": 5,
}
assert sum(INDEX_HEALTH_WEIGHTS.values()) == 100

TRAILING_INDEX_HEALTH_VERSION = "index_health_v1"
PERIOD_INDEX_HEALTH_VERSION = "index_period_health_v1"

# This is a separate horizon-sensitive methodology. It does not alter the
# established overall-market score or the legacy latest/trailing index score.
PERIOD_INDEX_HEALTH_WEIGHTS = {
    "Selected-period return": 20,
    "Trend consistency": 15,
    "Period momentum": 10,
    "Selected-period volatility resilience": 15,
    "Maximum-drawdown resilience": 15,
    "Value relative to SMA-20": 10,
    "Value relative to SMA-50": 10,
    "Volume participation": 5,
}
assert sum(PERIOD_INDEX_HEALTH_WEIGHTS.values()) == 100


@dataclass(frozen=True)
class MarketHealth:
    score: float | None
    label: str
    component_scores: Mapping[str, float | None]
    explanations: tuple[str, ...]
    reference_date: date | None


@dataclass(frozen=True)
class IndexHealth:
    """Independent rule-based health result for one index series."""

    index_code: str
    score: float | None
    label: str
    component_scores: Mapping[str, float | None]
    explanations: tuple[str, ...]
    reference_date: date | None
    observation_count: int | None
    available_weight: int
    unavailable_components: tuple[str, ...]
    coverage_percentage: float
    methodology_version: str = TRAILING_INDEX_HEALTH_VERSION
    requested_period: str | None = None
    component_inputs: Mapping[str, float | None] = field(
        default_factory=dict
    )
    component_factors: Mapping[str, float | None] = field(
        default_factory=dict
    )
    normalized_contributions: Mapping[str, float | None] = field(
        default_factory=dict
    )


def market_health_label(score: float) -> str:
    if score < 30: return "Strongly Bearish"
    if score < 45: return "Bearish"
    if score <= 55: return "Neutral"
    if score <= 70: return "Bullish"
    return "Strongly Bullish"


def _trend(metric: IndexMetrics) -> float | None:
    if metric.one_month_return is None: return None
    return max(0.0, min(1.0, 0.5 + metric.one_month_return / 20))


def _clamp_factor(value: float) -> float:
    return max(0.0, min(1.0, value))


def _centred_factor(value: float | None, full_scale: float) -> float | None:
    if value is None:
        return None
    return _clamp_factor(0.5 + value / (2 * full_scale))


def calculate_index_health(
    metric: IndexMetrics,
    *,
    observation_count: int | None = None,
) -> IndexHealth:
    """Calculate health solely from one index's trailing observations.

    Return and moving-average factors are centred at 50%. Positive readings add
    points and negative readings subtract points up to documented full-scale
    thresholds. Volatility loses all points at 50% annualized volatility.
    Volume receives full participation points at 1.5 times its 20-day average.
    """
    daily_direction = None
    if metric.latest_daily_change is not None:
        daily_direction = (
            1.0 if metric.latest_daily_change > 0
            else 0.0 if metric.latest_daily_change < 0
            else 0.5
        )
    momentum = None
    if metric.one_week_return is not None and metric.one_month_return is not None:
        weekly_acceleration = metric.one_week_return - metric.one_month_return / 4
        momentum = _centred_factor(weekly_acceleration, 5.0)
    volume_participation = None
    if (
        metric.latest_volume is not None
        and metric.average_volume_20 is not None
        and metric.average_volume_20 > 0
    ):
        volume_participation = _clamp_factor(
            metric.latest_volume / metric.average_volume_20 / 1.5
        )
    factors: dict[str, float | None] = {
        "1-day direction": daily_direction,
        "5-trading-day return": _centred_factor(metric.one_week_return, 10.0),
        "20-trading-day return": _centred_factor(metric.one_month_return, 20.0),
        "Value relative to SMA-20": _centred_factor(
            metric.versus_ma_20_percent, 10.0
        ),
        "Value relative to SMA-50": _centred_factor(
            metric.versus_ma_50_percent, 20.0
        ),
        "Short-term momentum": momentum,
        "Rolling volatility penalty": (
            None
            if metric.rolling_volatility_20 is None
            else _clamp_factor(1 - metric.rolling_volatility_20 / 50)
        ),
        "Volume participation": volume_participation,
    }
    weighted = {
        name: None if factor is None else factor * INDEX_HEALTH_WEIGHTS[name]
        for name, factor in factors.items()
    }
    used_weight = sum(
        INDEX_HEALTH_WEIGHTS[name]
        for name, factor in factors.items()
        if factor is not None
    )
    unavailable = tuple(name for name, factor in factors.items() if factor is None)
    coverage = round(used_weight / sum(INDEX_HEALTH_WEIGHTS.values()) * 100, 1)
    if not used_weight:
        return IndexHealth(
            metric.index_code,
            None,
            "Unavailable",
            weighted,
            ("Insufficient history is available to calculate index health.",),
            metric.latest_date,
            observation_count,
            0,
            unavailable,
            0.0,
        )
    raw_points = sum(value for value in weighted.values() if value is not None)
    score = round(_clamp_factor(raw_points / used_weight) * 100, 1)
    explanations = tuple(
        (
            f"{name}: unavailable."
            if weighted[name] is None
            else f"{name}: {weighted[name]:.1f} of "
            f"{INDEX_HEALTH_WEIGHTS[name]} points."
        )
        for name in INDEX_HEALTH_WEIGHTS
    )
    return IndexHealth(
        metric.index_code,
        score,
        market_health_label(score),
        weighted,
        explanations,
        metric.latest_date,
        observation_count,
        used_weight,
        unavailable,
        coverage,
    )


def calculate_index_health_scores(
    metrics: Mapping[str, IndexMetrics],
    *,
    observation_counts: Mapping[str, int] | None = None,
) -> dict[str, IndexHealth]:
    """Calculate one isolated health result for every supplied index metric."""
    counts = observation_counts or {}
    return {
        code: calculate_index_health(metric, observation_count=counts.get(code))
        for code, metric in metrics.items()
    }


def calculate_period_index_health(analysis: IndexPeriodAnalysis) -> IndexHealth:
    """Calculate a versioned health score from one exact selected-period window.

    Component scales are deliberately explicit and bounded:

    - selected-period return reaches its extrema at +/-20%;
    - trend consistency is the percentage of positive sessions (flat sessions
      receive half credit);
    - period momentum is the 21-session-equivalent difference between mean
      second-half and first-half daily returns and reaches its extrema at +/-10%;
    - volatility loses all points at 50% annualized volatility;
    - drawdown resilience loses all points at a 40% selected-window drawdown;
    - SMA-20 and SMA-50 position reach their extrema at +/-10% and +/-20%;
    - volume receives full points at 1.5 times selected-period average volume.

    Missing components are excluded, and the available raw points are rescaled
    to 100. ``normalized_contributions`` therefore reproduce ``score`` (before
    the final one-decimal rounding) without treating unavailable inputs as zero.
    """
    inputs: dict[str, float | None] = {
        "Selected-period return": analysis.period_return_percent,
        "Trend consistency": analysis.trend_consistency_percent,
        "Period momentum": analysis.period_momentum_percent,
        "Selected-period volatility resilience": (
            analysis.annualized_volatility_percent
        ),
        "Maximum-drawdown resilience": analysis.maximum_drawdown_percent,
        "Value relative to SMA-20": analysis.latest_vs_sma20_percent,
        "Value relative to SMA-50": analysis.latest_vs_sma50_percent,
        "Volume participation": analysis.volume_participation_ratio,
    }
    factors: dict[str, float | None] = {
        "Selected-period return": _centred_factor(
            inputs["Selected-period return"], 20.0
        ),
        "Trend consistency": (
            None
            if inputs["Trend consistency"] is None
            else _clamp_factor(inputs["Trend consistency"] / 100)
        ),
        "Period momentum": _centred_factor(inputs["Period momentum"], 10.0),
        "Selected-period volatility resilience": (
            None
            if inputs["Selected-period volatility resilience"] is None
            else _clamp_factor(
                1 - inputs["Selected-period volatility resilience"] / 50
            )
        ),
        "Maximum-drawdown resilience": (
            None
            if inputs["Maximum-drawdown resilience"] is None
            else _clamp_factor(
                1 - abs(inputs["Maximum-drawdown resilience"]) / 40
            )
        ),
        "Value relative to SMA-20": _centred_factor(
            inputs["Value relative to SMA-20"], 10.0
        ),
        "Value relative to SMA-50": _centred_factor(
            inputs["Value relative to SMA-50"], 20.0
        ),
        "Volume participation": (
            None
            if inputs["Volume participation"] is None
            else _clamp_factor(inputs["Volume participation"] / 1.5)
        ),
    }
    raw_points = {
        name: None
        if factor is None
        else factor * PERIOD_INDEX_HEALTH_WEIGHTS[name]
        for name, factor in factors.items()
    }
    available_weight = sum(
        PERIOD_INDEX_HEALTH_WEIGHTS[name]
        for name, factor in factors.items()
        if factor is not None
    )
    unavailable = tuple(name for name, factor in factors.items() if factor is None)
    coverage = round(
        available_weight / sum(PERIOD_INDEX_HEALTH_WEIGHTS.values()) * 100,
        1,
    )
    normalized = {
        name: None
        if points is None or not available_weight
        else points / available_weight * 100
        for name, points in raw_points.items()
    }
    metadata = analysis.metadata

    if not available_weight:
        return IndexHealth(
            index_code=metadata.index_code,
            score=None,
            label="Unavailable",
            component_scores=dict(raw_points),
            explanations=(
                "Insufficient selected-period history is available to calculate "
                "index health.",
            ),
            reference_date=metadata.actual_end_date,
            observation_count=metadata.observations,
            available_weight=0,
            unavailable_components=unavailable,
            coverage_percentage=0.0,
            methodology_version=PERIOD_INDEX_HEALTH_VERSION,
            requested_period=metadata.requested_period,
            component_inputs=dict(inputs),
            component_factors=dict(factors),
            normalized_contributions=dict(normalized),
        )

    total = sum(points for points in raw_points.values() if points is not None)
    score = round(_clamp_factor(total / available_weight) * 100, 1)
    explanations = tuple(
        (
            f"{name}: unavailable."
            if raw_points[name] is None
            else f"{name}: input {inputs[name]:.2f}; "
            f"{normalized[name]:.1f} normalized points "
            f"({raw_points[name]:.1f} raw of "
            f"{PERIOD_INDEX_HEALTH_WEIGHTS[name]})."
        )
        for name in PERIOD_INDEX_HEALTH_WEIGHTS
    )
    return IndexHealth(
        index_code=metadata.index_code,
        score=score,
        label=market_health_label(score),
        component_scores=dict(raw_points),
        explanations=explanations,
        reference_date=metadata.actual_end_date,
        observation_count=metadata.observations,
        available_weight=available_weight,
        unavailable_components=unavailable,
        coverage_percentage=coverage,
        methodology_version=PERIOD_INDEX_HEALTH_VERSION,
        requested_period=metadata.requested_period,
        component_inputs=dict(inputs),
        component_factors=dict(factors),
        normalized_contributions=dict(normalized),
    )


def calculate_period_index_health_scores(
    analyses: Mapping[str, IndexPeriodAnalysis],
) -> dict[str, IndexHealth]:
    """Calculate independently versioned period health for each supplied index."""
    return {
        code: calculate_period_index_health(analysis)
        for code, analysis in analyses.items()
    }


def calculate_market_health(
    metrics: Mapping[str, IndexMetrics], breadth: MarketBreadth
) -> MarketHealth:
    factors: dict[str, float | None] = {
        "KSE100 trend": _trend(metrics["KSE100"]) if "KSE100" in metrics else None,
        "KSE30 trend": _trend(metrics["KSE30"]) if "KSE30" in metrics else None,
        "KMI30 trend": _trend(metrics["KMI30"]) if "KMI30" in metrics else None,
        "All Share trend": _trend(metrics["ALLSHR"]) if "ALLSHR" in metrics else None,
        "Advance/decline ratio": None if breadth.advance_decline_ratio is None else min(1.0, breadth.advance_decline_ratio / 2),
        "Advancing securities": None if breadth.advancing_percent is None else min(1.0, breadth.advancing_percent / 70),
        "Volume participation": None,
        "Moving-average position": None,
        "Volatility penalty": None,
    }
    available_volumes = [m for m in metrics.values() if m.latest_volume is not None and m.average_volume_20]
    if available_volumes:
        factors["Volume participation"] = min(1.0, sum(m.latest_volume / m.average_volume_20 for m in available_volumes) / len(available_volumes))
    ma = [m.versus_ma_20_percent for m in metrics.values() if m.versus_ma_20_percent is not None]
    if ma: factors["Moving-average position"] = sum(1.0 if value > 0 else 0.0 for value in ma) / len(ma)
    volatility = [m.rolling_volatility_20 for m in metrics.values() if m.rolling_volatility_20 is not None]
    if volatility: factors["Volatility penalty"] = max(0.0, 1 - sum(volatility) / len(volatility) / 50)
    weighted = {name: (None if factor is None else factor * MARKET_HEALTH_WEIGHTS[name]) for name, factor in factors.items()}
    used_weight = sum(MARKET_HEALTH_WEIGHTS[name] for name, value in factors.items() if value is not None)
    if not used_weight:
        return MarketHealth(None, "Unavailable", weighted, ("No usable index or breadth inputs are available.",), breadth.reference_date)
    raw = sum(value for value in weighted.values() if value is not None)
    score = round(max(0.0, min(100.0, raw / used_weight * 100)), 1)
    explanations = tuple(
        f"{name}: {weighted[name]:.1f} of {MARKET_HEALTH_WEIGHTS[name]} points."
        for name in MARKET_HEALTH_WEIGHTS if weighted[name] is not None
    )
    return MarketHealth(score, market_health_label(score), weighted, explanations, breadth.reference_date)
