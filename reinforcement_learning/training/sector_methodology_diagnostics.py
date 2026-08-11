"""Research diagnostics for detecting degenerate sector-policy behaviour.

These thresholds are warnings, not model-selection criteria.  They are fixed
before the future three-seed experiment and must not be interpreted as an
automatic validation pass/fail rule.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Mapping, Sequence


ACTION_DOMINANCE_WARNING_THRESHOLD = 0.80
INVALID_ACTION_WARNING_THRESHOLD = 0.80
CASH_POLICY_EXPOSURE_WARNING_THRESHOLD_PERCENT = 5.0


def action_pattern_digest(actions: Sequence[int]) -> str:
    """Return a stable digest for one symbol's ordered selected actions."""

    normalized = [int(action) for action in actions]
    if any(action not in {0, 1, 2} for action in normalized):
        raise ValueError("action patterns may contain only Hold/Buy/Sell")
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_percentage(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite percentage")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise ValueError(f"{label} must be between 0 and 100")
    return numeric


def detect_action_collapse(
    *,
    selected_action_counts: Mapping[str, int],
    invalid_action_rate: float,
    per_symbol_exposure_percentages: Mapping[str, float],
    per_symbol_trade_counts: Mapping[str, int],
    per_symbol_action_digests: Mapping[str, str],
) -> dict[str, object]:
    """Summarize predeclared warning signals without rejecting a model.

    ``selected_action_counts`` uses readable lower-case keys (``hold``,
    ``buy``, ``sell``).  Exposure is an observed transition percentage, not a
    synthetic sector portfolio.  Identical behaviour means exact equality of
    complete per-symbol action sequences and is therefore deliberately strict.
    """

    required = {"hold", "buy", "sell"}
    if set(selected_action_counts) != required:
        raise ValueError("selected action counts must contain hold, buy, and sell")
    counts: dict[str, int] = {}
    for name in sorted(required):
        value = selected_action_counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("selected action counts must be non-negative integers")
        counts[name] = value
    if isinstance(invalid_action_rate, bool):
        raise ValueError("invalid_action_rate must be a fraction")
    invalid_rate = float(invalid_action_rate)
    if not math.isfinite(invalid_rate) or not 0.0 <= invalid_rate <= 1.0:
        raise ValueError("invalid_action_rate must be between 0 and 1")
    exposures = {
        str(symbol): _finite_percentage(value, label="exposure")
        for symbol, value in per_symbol_exposure_percentages.items()
    }
    trades: dict[str, int] = {}
    for symbol, value in per_symbol_trade_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("trade counts must be non-negative integers")
        trades[str(symbol)] = value
    if set(exposures) != set(trades) or set(exposures) != set(
        per_symbol_action_digests
    ):
        raise ValueError("per-symbol diagnostic populations must match")

    total = sum(counts.values())
    dominant_action = max(counts, key=counts.get) if total else None
    dominant_fraction = counts[dominant_action] / total if dominant_action else 0.0
    sorted_exposure = sorted(exposures.values())
    if not sorted_exposure:
        median_exposure = None
    elif len(sorted_exposure) % 2:
        median_exposure = sorted_exposure[len(sorted_exposure) // 2]
    else:
        middle = len(sorted_exposure) // 2
        median_exposure = (
            sorted_exposure[middle - 1] + sorted_exposure[middle]
        ) / 2.0
    zero_trade_symbols = tuple(
        symbol for symbol, count in sorted(trades.items()) if count == 0
    )
    digest_groups: dict[str, list[str]] = {}
    for symbol, digest in sorted(per_symbol_action_digests.items()):
        digest_groups.setdefault(str(digest), []).append(str(symbol))
    identical_groups = tuple(
        tuple(symbols)
        for symbols in digest_groups.values()
        if len(symbols) > 1
    )

    warnings: list[str] = []
    if dominant_fraction > ACTION_DOMINANCE_WARNING_THRESHOLD:
        warnings.append("possible_action_collapse")
    if invalid_rate > INVALID_ACTION_WARNING_THRESHOLD:
        warnings.append("possible_invalid_action_attractor")
    if (
        median_exposure is not None
        and median_exposure < CASH_POLICY_EXPOSURE_WARNING_THRESHOLD_PERCENT
    ):
        warnings.append("possible_cash_policy_collapse")
    if zero_trade_symbols:
        warnings.append("zero_trade_symbols_observed")
    if identical_groups:
        warnings.append("identical_policy_behavior_across_symbols")

    return {
        "thresholds_are_warnings_not_selection_rules": True,
        "action_dominance_threshold": ACTION_DOMINANCE_WARNING_THRESHOLD,
        "invalid_action_threshold": INVALID_ACTION_WARNING_THRESHOLD,
        "cash_policy_median_exposure_threshold_percent": (
            CASH_POLICY_EXPOSURE_WARNING_THRESHOLD_PERCENT
        ),
        "selected_action_counts": counts,
        "dominant_action": dominant_action,
        "dominant_action_fraction": dominant_fraction,
        "invalid_action_rate": invalid_rate,
        "median_exposure_percentage": median_exposure,
        "zero_trade_symbols": list(zero_trade_symbols),
        "identical_action_pattern_symbol_groups": [
            list(group) for group in identical_groups
        ],
        "warnings": warnings,
    }


def combine_action_counts(
    per_symbol_actions: Mapping[str, Sequence[int]],
) -> dict[str, int]:
    """Combine ordered action sequences into the canonical readable counts."""

    counter: Counter[int] = Counter()
    for actions in per_symbol_actions.values():
        for action in actions:
            selected = int(action)
            if selected not in {0, 1, 2}:
                raise ValueError("action sequences may contain only 0, 1, or 2")
            counter[selected] += 1
    return {"hold": counter[0], "buy": counter[1], "sell": counter[2]}


__all__ = (
    "ACTION_DOMINANCE_WARNING_THRESHOLD",
    "CASH_POLICY_EXPOSURE_WARNING_THRESHOLD_PERCENT",
    "INVALID_ACTION_WARNING_THRESHOLD",
    "action_pattern_digest",
    "combine_action_counts",
    "detect_action_collapse",
)
