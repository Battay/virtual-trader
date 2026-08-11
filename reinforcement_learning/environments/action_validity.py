"""Canonical long-only action-validity semantics for RL environments.

The production action space remains ``0=Hold, 1=Buy, 2=Sell``.  This module
separates portfolio-state validity from execution feasibility: a Buy selected
while flat is state-valid, but may still fail because there is not enough cash
to buy one whole share.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Final, Mapping


SECTOR_ACTION_VALIDITY_VERSION: Final = "sector_action_validity_v1"
PENALTY_ACTION_VALIDITY_VERSION: Final = "sector_action_validity_penalty_v1"
MASK_ACTION_VALIDITY_VERSION: Final = "sector_action_validity_mask_v1"

HOLD_ACTION: Final = 0
BUY_ACTION: Final = 1
SELL_ACTION: Final = 2
ACTION_NAMES: Final = {HOLD_ACTION: "Hold", BUY_ACTION: "Buy", SELL_ACTION: "Sell"}

PENALTY_MODE: Final = "penalty"
MASK_MODE: Final = "mask"
INVALID_ACTION_MODES: Final = (PENALTY_MODE, MASK_MODE)
MASKING_STATUS: Final = "unsupported_or_deferred"
MASKING_LIMITATION: Final = (
    "sb3-contrib RecurrentPPO/MlpLstmPolicy does not provide a verified native "
    "action-mask training interface in the installed architecture; mask mode "
    "must not be emulated by rewriting a selected action"
)


class PortfolioState(str, Enum):
    """Long-only portfolio states relevant to action validity."""

    FLAT = "flat"
    LONG = "long"


def action_name(action: int) -> str:
    """Return the readable name for a supported discrete action."""
    try:
        return ACTION_NAMES[int(action)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported action {action!r}; expected 0, 1, or 2") from exc


def portfolio_state_from_shares(shares_held: int) -> PortfolioState:
    """Resolve a valid long-only state from the current whole-share holding."""
    if isinstance(shares_held, bool) or not isinstance(shares_held, Integral):
        raise TypeError("shares_held must be an integer")
    if shares_held < 0:
        raise ValueError("shares_held cannot be negative in a long-only portfolio")
    return PortfolioState.LONG if shares_held > 0 else PortfolioState.FLAT


def _portfolio_state(value: PortfolioState | str) -> PortfolioState:
    if isinstance(value, PortfolioState):
        return value
    try:
        return PortfolioState(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported portfolio state {value!r}") from exc


def valid_actions(portfolio_state: PortfolioState | str) -> tuple[int, ...]:
    """Return canonical valid actions for a flat or long-only portfolio."""
    state = _portfolio_state(portfolio_state)
    if state is PortfolioState.FLAT:
        return (HOLD_ACTION, BUY_ACTION)
    return (HOLD_ACTION, SELL_ACTION)


def action_mask(portfolio_state: PortfolioState | str) -> tuple[bool, bool, bool]:
    """Return a diagnostic/future-integration mask without altering an action."""
    allowed = set(valid_actions(portfolio_state))
    return (
        HOLD_ACTION in allowed,
        BUY_ACTION in allowed,
        SELL_ACTION in allowed,
    )


def is_action_valid(action: int, portfolio_state: PortfolioState | str) -> bool:
    """Return whether an action is semantically valid for the portfolio state."""
    action_name(action)
    return int(action) in valid_actions(portfolio_state)


@dataclass(frozen=True)
class ActionValidityDecision:
    """Pre-execution validity decision based only on portfolio state."""

    selected_action: int
    selected_action_name: str
    portfolio_state: str
    state_valid: bool
    invalid_reason: str | None
    valid_action_ids: tuple[int, ...]
    valid_action_mask: tuple[bool, bool, bool]
    action_validity_version: str = SECTOR_ACTION_VALIDITY_VERSION


@dataclass(frozen=True)
class ActionOutcome:
    """Final action metadata after state validity and execution feasibility."""

    selected_action: int
    selected_action_name: str
    portfolio_state: str
    state_valid: bool
    action_valid: bool
    action_executed: bool
    trade_executed: bool
    invalid_reason: str | None
    execution_failure_reason: str | None
    valid_action_ids: tuple[int, ...]
    valid_action_mask: tuple[bool, bool, bool]
    action_validity_version: str = SECTOR_ACTION_VALIDITY_VERSION

    def __post_init__(self) -> None:
        state = _portfolio_state(self.portfolio_state)
        expected_actions = valid_actions(state)
        if self.selected_action_name != action_name(self.selected_action):
            raise ValueError("selected action id/name are inconsistent")
        if self.valid_action_ids != expected_actions:
            raise ValueError("valid_action_ids are inconsistent with portfolio state")
        if self.valid_action_mask != action_mask(state):
            raise ValueError("valid_action_mask is inconsistent with portfolio state")
        if self.state_valid != is_action_valid(self.selected_action, state):
            raise ValueError("state_valid is inconsistent with canonical semantics")
        if self.action_valid != self.state_valid:
            raise ValueError("action_valid must represent portfolio-state validity")
        if self.action_executed and not self.action_valid:
            raise ValueError("an invalid action cannot be executed")
        if self.trade_executed and not self.action_executed:
            raise ValueError("a trade cannot execute when its action did not execute")
        if self.execution_failure_reason and self.action_executed:
            raise ValueError("an action with an execution failure cannot be executed")
        if self.action_validity_version != SECTOR_ACTION_VALIDITY_VERSION:
            raise ValueError(
                "action_validity_version must be "
                f"{SECTOR_ACTION_VALIDITY_VERSION!r}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_action": self.selected_action,
            "selected_action_name": self.selected_action_name,
            "portfolio_state_before_action": self.portfolio_state,
            "action_state_valid": self.state_valid,
            "action_valid": self.action_valid,
            "action_executed": self.action_executed,
            "trade_executed": self.trade_executed,
            # Keep the historical key populated for execution failures while
            # exposing the two causes separately in canonical metadata.
            "invalid_action_reason": (
                self.invalid_reason or self.execution_failure_reason
            ),
            "semantic_invalid_action_reason": self.invalid_reason,
            "execution_failure_reason": self.execution_failure_reason,
            "valid_action_ids": self.valid_action_ids,
            "valid_action_mask": self.valid_action_mask,
            "action_validity_version": self.action_validity_version,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ActionOutcome":
        """Reconstruct diagnostics from a transition info/history mapping."""
        selected = int(values["selected_action"])
        action_ids = tuple(int(value) for value in values["valid_action_ids"])
        mask = tuple(bool(value) for value in values["valid_action_mask"])
        if len(action_ids) != 2 or len(mask) != 3:
            raise ValueError("transition action-validity metadata has invalid shape")
        return cls(
            selected_action=selected,
            selected_action_name=str(
                values.get("selected_action_name") or action_name(selected)
            ),
            portfolio_state=str(values["portfolio_state_before_action"]),
            state_valid=bool(values["action_state_valid"]),
            action_valid=bool(values["action_valid"]),
            action_executed=bool(values["action_executed"]),
            trade_executed=bool(values["trade_executed"]),
            invalid_reason=(
                str(values["semantic_invalid_action_reason"])
                if values.get("semantic_invalid_action_reason") is not None
                else None
            ),
            execution_failure_reason=(
                str(values["execution_failure_reason"])
                if values.get("execution_failure_reason") is not None
                else None
            ),
            valid_action_ids=action_ids,
            valid_action_mask=(mask[0], mask[1], mask[2]),
            action_validity_version=str(values["action_validity_version"]),
        )


def evaluate_action_validity(
    action: int,
    portfolio_state: PortfolioState | str,
) -> ActionValidityDecision:
    """Evaluate the fixed long-only action table before execution."""
    selected = int(action)
    name = action_name(selected)
    state = _portfolio_state(portfolio_state)
    allowed = valid_actions(state)
    valid = selected in allowed
    reason: str | None = None
    if not valid:
        if selected == SELL_ACTION and state is PortfolioState.FLAT:
            reason = "No shares are held to sell"
        elif selected == BUY_ACTION and state is PortfolioState.LONG:
            reason = "A long position is already held; Buy is redundant"
        else:  # pragma: no cover - exhaustive guard for future action additions
            reason = f"{name} is invalid while portfolio is {state.value}"
    return ActionValidityDecision(
        selected_action=selected,
        selected_action_name=name,
        portfolio_state=state.value,
        state_valid=valid,
        invalid_reason=reason,
        valid_action_ids=allowed,
        valid_action_mask=action_mask(state),
    )


def finalize_action_outcome(
    decision: ActionValidityDecision,
    *,
    execution_failure_reason: str | None = None,
    trade_executed: bool = False,
) -> ActionOutcome:
    """Resolve final validity/execution metadata without rewriting the action.

    A state-valid action can still be infeasible at execution time, for example
    when a flat portfolio cannot afford one whole share.  Hold is considered an
    executed action even though it does not execute a trade.
    """
    action_valid = decision.state_valid
    action_executed = action_valid and execution_failure_reason is None
    return ActionOutcome(
        selected_action=decision.selected_action,
        selected_action_name=decision.selected_action_name,
        portfolio_state=decision.portfolio_state,
        state_valid=decision.state_valid,
        action_valid=action_valid,
        action_executed=action_executed,
        trade_executed=bool(trade_executed and action_executed),
        invalid_reason=decision.invalid_reason,
        execution_failure_reason=execution_failure_reason,
        valid_action_ids=decision.valid_action_ids,
        valid_action_mask=decision.valid_action_mask,
    )


def normalize_invalid_action_mode(value: str) -> str:
    """Validate a configured invalid-action mode."""
    mode = str(value).strip().lower()
    if mode not in INVALID_ACTION_MODES:
        raise ValueError(
            f"invalid_action_mode must be one of {INVALID_ACTION_MODES}, got {value!r}"
        )
    return mode


def require_supported_action_mode(value: str) -> str:
    """Fail closed instead of pretending to mask unsupported recurrent actions."""
    mode = normalize_invalid_action_mode(value)
    if mode == MASK_MODE:
        raise NotImplementedError(f"masking_status={MASKING_STATUS}: {MASKING_LIMITATION}")
    return mode


def action_validity_metadata(mode: str = PENALTY_MODE) -> dict[str, object]:
    """Return the semantic action table and masking status for fingerprints."""
    selected = normalize_invalid_action_mode(mode)
    methodology_version = (
        PENALTY_ACTION_VALIDITY_VERSION
        if selected == PENALTY_MODE
        else MASK_ACTION_VALIDITY_VERSION
    )
    return {
        "action_validity_version": SECTOR_ACTION_VALIDITY_VERSION,
        "methodology_version": methodology_version,
        "invalid_action_mode": selected,
        "action_mapping": dict(ACTION_NAMES),
        "flat_valid_actions": valid_actions(PortfolioState.FLAT),
        "long_valid_actions": valid_actions(PortfolioState.LONG),
        "hold_always_valid": True,
        "state_invalid_action_treatment": (
            "no-op plus configured fixed penalty"
            if selected == PENALTY_MODE
            else "must be unavailable to policy before selection"
        ),
        "execution_failure_treatment": (
            "state-valid action remains valid, action_executed is false, and no "
            "state-invalid-action penalty is charged"
        ),
        "masking_status": (
            "not_applicable" if selected == PENALTY_MODE else MASKING_STATUS
        ),
        "masking_limitation": (
            None if selected == PENALTY_MODE else MASKING_LIMITATION
        ),
    }


__all__ = (
    "ACTION_NAMES",
    "ActionOutcome",
    "ActionValidityDecision",
    "BUY_ACTION",
    "HOLD_ACTION",
    "INVALID_ACTION_MODES",
    "MASK_ACTION_VALIDITY_VERSION",
    "MASKING_LIMITATION",
    "MASKING_STATUS",
    "MASK_MODE",
    "PENALTY_ACTION_VALIDITY_VERSION",
    "PENALTY_MODE",
    "PortfolioState",
    "SECTOR_ACTION_VALIDITY_VERSION",
    "SELL_ACTION",
    "action_mask",
    "action_name",
    "action_validity_metadata",
    "evaluate_action_validity",
    "finalize_action_outcome",
    "is_action_valid",
    "normalize_invalid_action_mode",
    "portfolio_state_from_shares",
    "require_supported_action_mode",
    "valid_actions",
)
