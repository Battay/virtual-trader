"""Pure state helpers for the Streamlit historical-backfill preview."""

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from data_pipeline.src.backfill import BackfillPlan, BackfillRunResult, BackfillState
from data_pipeline.src.automation import karachi_today
from data_pipeline.src.config import PSX_HISTORICAL_MIN_DATE


PREVIEW_PLAN_KEY = "backfill_preview_plan"
PREVIEW_INPUTS_KEY = "backfill_preview_inputs"
PREVIEW_ERROR_KEY = "backfill_preview_error"


@dataclass(frozen=True)
class BackfillBatchSummary:
    """Mutually exclusive counts produced by the latest operation."""

    requests_attempted: int
    downloads_successful: int
    existing_csv_reconciled: int
    non_trading_resolved: int
    temporarily_unavailable: int
    failed: int

    @property
    def total_dates_resolved(self) -> int:
        """Return the sum of the documented outcome categories."""
        return (
            self.downloads_successful
            + self.existing_csv_reconciled
            + self.non_trading_resolved
            + self.temporarily_unavailable
            + self.failed
        )


def summarize_backfill_batch(result: BackfillRunResult) -> BackfillBatchSummary:
    """Separate live downloads from reconciliation and other outcomes."""
    attempted = set(result.attempted_dates)
    reconciled = {
        outcome.trading_date
        for outcome in result.outcomes
        if outcome.status == "successful" and outcome.reconciled
    }
    downloads_successful = {
        outcome.trading_date
        for outcome in result.outcomes
        if outcome.status == "successful"
        and not outcome.reconciled
        and outcome.trading_date in attempted
    } - reconciled
    non_trading = {
        outcome.trading_date
        for outcome in result.outcomes
        if outcome.status == "non_trading"
    } - reconciled - downloads_successful
    temporary = {
        outcome.trading_date
        for outcome in result.outcomes
        if outcome.status == "temporary_unavailable"
        and outcome.trading_date in attempted
    } - reconciled - downloads_successful - non_trading
    failed = {
        outcome.trading_date
        for outcome in result.outcomes
        if outcome.status == "failed" and outcome.trading_date in attempted
    } - reconciled - downloads_successful - non_trading - temporary

    return BackfillBatchSummary(
        requests_attempted=len(result.attempted_dates),
        downloads_successful=len(downloads_successful),
        existing_csv_reconciled=len(reconciled),
        non_trading_resolved=len(non_trading),
        temporarily_unavailable=len(temporary),
        failed=len(failed),
    )


def backfill_date_bounds(now: datetime | None = None) -> tuple[date, date]:
    """Return the configured historical floor and current Karachi date."""
    return PSX_HISTORICAL_MIN_DATE, karachi_today(now)


def clamp_backfill_end_date(
    start_date: date,
    end_date: date,
    *,
    latest_allowed_date: date,
) -> date:
    """Preserve a valid end date and clamp only to the actual date limits."""
    if start_date > latest_allowed_date:
        raise ValueError("start date cannot be later than the latest allowed date")
    return min(max(end_date, start_date), latest_allowed_date)


def initial_backfill_dates(
    *,
    saved_state: BackfillState | None,
    default_start: date,
    default_end: date,
) -> tuple[date, date]:
    """Restore an exact saved range, otherwise return the supplied defaults."""
    if saved_state is not None:
        return (
            saved_state.requested_start_date,
            saved_state.requested_end_date,
        )
    return default_start, default_end


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
