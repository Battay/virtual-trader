"""Read-only symbol-history policy for future recurrent transfer research.

This policy is deliberately separate from the current ``rl_partition_v1``
MLP PPO readiness gate.  It classifies real, usable observations only and does
not change dataset generation, split artifacts, or trainer eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real


MATURE_MINIMUM_USABLE_OBSERVATIONS = 126
COLD_START_MINIMUM_USABLE_OBSERVATIONS = 100


class HistoryClass(str, Enum):
    """Approved future recurrent/transfer history categories."""

    MATURE = "MATURE"
    COLD_START = "COLD_START"
    INSUFFICIENT = "INSUFFICIENT"

    @property
    def label(self) -> str:
        return {
            HistoryClass.MATURE: "Mature",
            HistoryClass.COLD_START: "Cold Start",
            HistoryClass.INSUFFICIENT: "Insufficient",
        }[self]


FUTURE_TRAINING_ROUTES = {
    HistoryClass.MATURE: (
        "Eligible for recurrent symbol fine-tuning; sector pretraining planned."
    ),
    HistoryClass.COLD_START: (
        "Sector-pretrained transfer route planned; use only real company history."
    ),
    HistoryClass.INSUFFICIENT: (
        "Not enough own history for safe symbol-specific fine-tuning."
    ),
}


@dataclass(frozen=True)
class HistoryClassification:
    """One symbol's future history class and intended research route."""

    history_class: HistoryClass
    usable_observations: int
    future_training_route: str

    @property
    def label(self) -> str:
        return self.history_class.label


def _usable_observation_count(value: object) -> int:
    """Return a validated whole usable-observation count without guessing."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("usable observations must be a non-negative whole number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("usable observations must be a non-negative whole number")
    return int(numeric)


def classify_usable_history(usable_observations: object) -> HistoryClassification:
    """Classify usable post-cleaning/post-warm-up observations.

    Calendar dates and raw row counts are intentionally not accepted.  The
    caller must supply the canonical usable feature-observation count.
    """
    count = _usable_observation_count(usable_observations)
    if count >= MATURE_MINIMUM_USABLE_OBSERVATIONS:
        history_class = HistoryClass.MATURE
    elif count >= COLD_START_MINIMUM_USABLE_OBSERVATIONS:
        history_class = HistoryClass.COLD_START
    else:
        history_class = HistoryClass.INSUFFICIENT
    return HistoryClassification(
        history_class=history_class,
        usable_observations=count,
        future_training_route=FUTURE_TRAINING_ROUTES[history_class],
    )
