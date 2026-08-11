"""Versioned reward semantics and lightweight transition diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import fmean, median, pstdev
from typing import Final, Iterable, Mapping

from .action_validity import ActionOutcome, BUY_ACTION, HOLD_ACTION, SELL_ACTION


SECTOR_REWARD_VERSION: Final = "sector_reward_v1"
SECTOR_REWARD_EQUATION: Final = (
    "portfolio_growth_weight * log(current_portfolio_value / "
    "previous_portfolio_value) - transaction_cost_weight * "
    "transaction_cost / previous_portfolio_value - "
    "drawdown_increment_weight * max(0, current_drawdown - "
    "previous_drawdown) - invalid_action_penalty when action is invalid"
)


@dataclass(frozen=True)
class SectorRewardConfig:
    """Complete semantic and coefficient contract for sector reward v1."""

    reward_version: str = SECTOR_REWARD_VERSION
    portfolio_growth_weight: float = 1.0
    transaction_cost_weight: float = 0.0
    drawdown_increment_weight: float = 0.1
    invalid_action_penalty: float = 0.0001

    def __post_init__(self) -> None:
        if self.reward_version != SECTOR_REWARD_VERSION:
            raise ValueError(f"reward_version must be {SECTOR_REWARD_VERSION!r}")
        for field_name in (
            "portfolio_growth_weight",
            "transaction_cost_weight",
            "drawdown_increment_weight",
            "invalid_action_penalty",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")

    def to_metadata(self) -> dict[str, object]:
        """Return semantics as well as coefficients for reproducible identity."""
        return {
            "reward_version": self.reward_version,
            "reward_equation": SECTOR_REWARD_EQUATION,
            "portfolio_growth_definition": (
                "natural log of current portfolio value divided by previous "
                "portfolio value, after next-open execution and next-close marking"
            ),
            "transaction_cost_treatment": (
                "commission and slippage already affect portfolio accounting; "
                "the optional additional penalty is transaction_cost_weight times "
                "transaction cost divided by previous portfolio value"
            ),
            "drawdown_increment_definition": (
                "positive increment only: max(0, current drawdown minus previous "
                "drawdown)"
            ),
            "invalid_action_treatment": (
                "invalid selections are no-ops in penalty mode and subtract the "
                "configured fixed invalid-action penalty"
            ),
            "portfolio_growth_weight": self.portfolio_growth_weight,
            "transaction_cost_weight": self.transaction_cost_weight,
            "drawdown_increment_weight": self.drawdown_increment_weight,
            "invalid_action_penalty": self.invalid_action_penalty,
        }


@dataclass(frozen=True)
class RewardComponents:
    """Signed reward contributions whose exact sum is ``total_reward``."""

    portfolio_growth_reward: float
    transaction_cost_penalty: float
    drawdown_penalty: float
    invalid_action_penalty: float
    total_reward: float
    reward_version: str = SECTOR_REWARD_VERSION

    def __post_init__(self) -> None:
        values = (
            self.portfolio_growth_reward,
            self.transaction_cost_penalty,
            self.drawdown_penalty,
            self.invalid_action_penalty,
            self.total_reward,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("reward components must all be finite")
        component_sum = math.fsum(float(value) for value in values[:-1])
        if not math.isclose(
            component_sum,
            float(self.total_reward),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("reward components do not sum to total_reward")

    @classmethod
    def zero(cls) -> "RewardComponents":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "reward_version": self.reward_version,
            "portfolio_growth_reward": self.portfolio_growth_reward,
            "transaction_cost_penalty": self.transaction_cost_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "total_reward": self.total_reward,
        }

    def to_legacy_dict(self) -> dict[str, float]:
        """Preserve the single_symbol_env_v1 public dictionary contract."""
        return {
            "portfolio_growth": self.portfolio_growth_reward,
            "transaction_cost_penalty": self.transaction_cost_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "RewardComponents":
        """Reconstruct the canonical value object from ``reward_breakdown``."""
        return cls(
            portfolio_growth_reward=float(values["portfolio_growth_reward"]),
            transaction_cost_penalty=float(values["transaction_cost_penalty"]),
            drawdown_penalty=float(values["drawdown_penalty"]),
            invalid_action_penalty=float(values["invalid_action_penalty"]),
            total_reward=float(values["total_reward"]),
            reward_version=str(values["reward_version"]),
        )


def calculate_reward_components(
    *,
    config: SectorRewardConfig,
    previous_portfolio_value: float,
    current_portfolio_value: float,
    transaction_cost: float,
    previous_drawdown: float,
    current_drawdown: float,
    action_invalid: bool,
) -> RewardComponents:
    """Calculate sector_reward_v1 without changing portfolio accounting."""
    values = {
        "previous_portfolio_value": previous_portfolio_value,
        "current_portfolio_value": current_portfolio_value,
        "transaction_cost": transaction_cost,
        "previous_drawdown": previous_drawdown,
        "current_drawdown": current_drawdown,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("reward inputs must all be finite")
    if previous_portfolio_value <= 0 or current_portfolio_value <= 0:
        raise ValueError("portfolio values must be positive")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")
    if previous_drawdown < 0 or current_drawdown < 0:
        raise ValueError("drawdown values cannot be negative")

    growth = config.portfolio_growth_weight * math.log(
        current_portfolio_value / previous_portfolio_value
    )
    cost = -(
        config.transaction_cost_weight
        * transaction_cost
        / previous_portfolio_value
    )
    drawdown = -(
        config.drawdown_increment_weight
        * max(0.0, current_drawdown - previous_drawdown)
    )
    invalid = -config.invalid_action_penalty if action_invalid else 0.0
    total = math.fsum((growth, cost, drawdown, invalid))
    return RewardComponents(
        portfolio_growth_reward=float(growth),
        transaction_cost_penalty=float(cost),
        drawdown_penalty=float(drawdown),
        invalid_action_penalty=float(invalid),
        total_reward=float(total),
        reward_version=config.reward_version,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    count = len(values)
    return {
        "count": count,
        "mean": fmean(values),
        "median": median(values),
        "std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "p05": _percentile(values, 0.05),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "positive_fraction": sum(value > 0 for value in values) / count,
        "negative_fraction": sum(value < 0 for value in values) / count,
        "zero_fraction": sum(value == 0 for value in values) / count,
    }


class RewardDiagnosticsAccumulator:
    """Collect lightweight aggregate and per-symbol reward/action diagnostics."""

    def __init__(self) -> None:
        self._records: dict[str, list[tuple[RewardComponents, ActionOutcome]]] = (
            defaultdict(list)
        )

    def update(
        self,
        *,
        symbol: str,
        reward: RewardComponents,
        action: ActionOutcome,
    ) -> None:
        canonical_symbol = str(symbol).strip()
        if not canonical_symbol:
            raise ValueError("symbol cannot be empty")
        self._records[canonical_symbol].append((reward, action))

    def update_from_info(self, *, symbol: str, info: Mapping[str, object]) -> None:
        """Consume the public transition-info mappings emitted by the env."""
        breakdown = info.get("reward_breakdown")
        if not isinstance(breakdown, Mapping):
            raise ValueError("transition info is missing reward_breakdown")
        self.update(
            symbol=symbol,
            reward=RewardComponents.from_mapping(breakdown),
            action=ActionOutcome.from_mapping(info),
        )

    @staticmethod
    def _summary(
        records: Iterable[tuple[RewardComponents, ActionOutcome]],
    ) -> dict[str, object]:
        selected = list(records)
        rewards = [float(reward.total_reward) for reward, _ in selected]
        components = {
            name: math.fsum(float(getattr(reward, name)) for reward, _ in selected)
            for name in (
                "portfolio_growth_reward",
                "transaction_cost_penalty",
                "drawdown_penalty",
                "invalid_action_penalty",
            )
        }
        components["total_reward"] = math.fsum(rewards)
        counts: Counter[str] = Counter()
        for _, outcome in selected:
            validity = "valid" if outcome.action_valid else "invalid"
            counts[f"{validity}_{outcome.selected_action_name.lower()}"] += 1
        invalid_count = sum(
            count for name, count in counts.items() if name.startswith("invalid_")
        )
        execution_failures = sum(
            outcome.execution_failure_reason is not None for _, outcome in selected
        )
        return {
            "reward_distribution": _distribution(rewards),
            "cumulative_components": components,
            "action_validity_counts": {
                "valid_hold": counts[f"valid_{ACTION_NAMES_BY_ID[HOLD_ACTION]}"],
                "valid_buy": counts[f"valid_{ACTION_NAMES_BY_ID[BUY_ACTION]}"],
                "valid_sell": counts[f"valid_{ACTION_NAMES_BY_ID[SELL_ACTION]}"],
                "invalid_buy": counts[f"invalid_{ACTION_NAMES_BY_ID[BUY_ACTION]}"],
                "invalid_sell": counts[f"invalid_{ACTION_NAMES_BY_ID[SELL_ACTION]}"],
                "invalid_action_rate": (
                    invalid_count / len(selected) if selected else 0.0
                ),
                "execution_failure_count": execution_failures,
            },
        }

    def summary(self) -> dict[str, object]:
        all_records = [record for records in self._records.values() for record in records]
        aggregate = self._summary(all_records)
        aggregate["per_symbol"] = {
            symbol: self._summary(records)
            for symbol, records in sorted(self._records.items())
        }
        return aggregate


ACTION_NAMES_BY_ID: Final = {
    HOLD_ACTION: "hold",
    BUY_ACTION: "buy",
    SELL_ACTION: "sell",
}


__all__ = (
    "RewardComponents",
    "RewardDiagnosticsAccumulator",
    "SECTOR_REWARD_EQUATION",
    "SECTOR_REWARD_VERSION",
    "SectorRewardConfig",
    "calculate_reward_components",
)
