"""Pure state helpers for the Streamlit historical-backfill preview."""

from collections.abc import Callable, Mapping, MutableMapping
from datetime import date
from typing import Any

from data_pipeline.src.backfill import BackfillPlan, BackfillState


PREVIEW_PLAN_KEY = "backfill_preview_plan"
PREVIEW_INPUTS_KEY = "backfill_preview_inputs"
PREVIEW_ERROR_KEY = "backfill_preview_error"


def build_preview_inputs(
    *,
    start_date: date,
    end_date: date,
    delay_seconds: float,
    max_dates: int,
    retry_failed: bool,
) -> dict[str, object]:
    """Return the normalized input snapshot associated with a preview plan."""
    return {
        "start_date": start_date,
        "end_date": end_date,
        "delay_seconds": float(delay_seconds),
        "max_dates": int(max_dates),
        "retry_failed": bool(retry_failed),
    }


def store_backfill_preview(
    state: MutableMapping[str, Any],
    plan: BackfillPlan,
    inputs: Mapping[str, object],
) -> None:
    """Persist a successful plan and its inputs in session-like state."""
    state[PREVIEW_PLAN_KEY] = plan
    state[PREVIEW_INPUTS_KEY] = dict(inputs)
    state.pop(PREVIEW_ERROR_KEY, None)


def clear_backfill_preview(state: MutableMapping[str, Any]) -> None:
    """Remove a previous plan when it must no longer authorize a live run."""
    state.pop(PREVIEW_PLAN_KEY, None)
    state.pop(PREVIEW_INPUTS_KEY, None)


def preview_is_stale(
    stored_inputs: Mapping[str, object] | None,
    current_inputs: Mapping[str, object],
) -> bool:
    """Return whether current controls differ from the previewed input snapshot."""
    return stored_inputs is None or dict(stored_inputs) != dict(current_inputs)


def state_for_preview_range(
    state: BackfillState | None,
    *,
    start_date: date,
    end_date: date,
) -> BackfillState | None:
    """Use saved classifications only when they belong to the previewed range."""
    if state is None:
        return None
    if (
        state.requested_start_date != start_date
        or state.requested_end_date != end_date
    ):
        return None
    return state


def resume_is_eligible(
    plan: BackfillPlan | None,
    *,
    stale: bool,
) -> bool:
    """Allow resume only for a current preview that contains live requests."""
    return plan is not None and not stale and plan.estimated_request_count > 0


def preview_status_message(plan: BackfillPlan) -> str:
    """Return explicit feedback for zero- and non-zero-request previews."""
    if plan.estimated_request_count == 0:
        return (
            "No requests are required for this range. Existing files and "
            "non-trading dates already cover it."
        )
    return (
        f"{plan.estimated_request_count:,} request date(s) are ready for a "
        "bounded backfill run."
    )


def create_preview_safely(
    planner: Callable[..., BackfillPlan],
    *args: object,
    **kwargs: object,
) -> tuple[BackfillPlan | None, str | None]:
    """Run only the injected planner and convert its errors to safe UI text."""
    try:
        return planner(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - UI boundary must remain usable.
        detail = str(exc).strip() or type(exc).__name__
        return None, f"Could not preview the backfill plan: {detail}"
