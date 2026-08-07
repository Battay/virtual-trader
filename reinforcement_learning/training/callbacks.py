"""Coarse-grained Stable-Baselines3 progress callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from stable_baselines3.common.callbacks import BaseCallback


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingProgress:
    """One compact progress event for a single-symbol training run."""

    symbol: str
    phase: str
    current_timesteps: int
    requested_timesteps: int
    progress_percent: float
    timestamp: str


ProgressHandler = Callable[[TrainingProgress], bool | None]


class PPOProgressCallback(BaseCallback):
    """Emit bounded progress events and support cooperative cancellation."""

    def __init__(
        self,
        *,
        symbol: str,
        requested_timesteps: int,
        interval_steps: int,
        handler: ProgressHandler | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.symbol = symbol
        self.requested_timesteps = requested_timesteps
        self.interval_steps = max(1, interval_steps)
        self.handler = handler
        self.cancel_requested = False
        self._next_event = self.interval_steps

    def _event(self, phase: str) -> TrainingProgress:
        current = int(self.num_timesteps)
        percent = min(100.0, 100.0 * current / self.requested_timesteps)
        return TrainingProgress(
            symbol=self.symbol,
            phase=phase,
            current_timesteps=current,
            requested_timesteps=self.requested_timesteps,
            progress_percent=percent,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _emit(self, phase: str) -> None:
        event = self._event(phase)
        LOGGER.info(
            "ppo_progress symbol=%s phase=%s timesteps=%s requested=%s percent=%.1f",
            event.symbol,
            event.phase,
            event.current_timesteps,
            event.requested_timesteps,
            event.progress_percent,
        )
        if self.handler is not None and self.handler(event) is False:
            self.cancel_requested = True

    def _on_training_start(self) -> None:
        self._emit("started")

    def _on_step(self) -> bool:
        if self.cancel_requested:
            return False
        if self.num_timesteps >= self._next_event:
            self._emit("progress")
            while self._next_event <= self.num_timesteps:
                self._next_event += self.interval_steps
        return not self.cancel_requested

    def _on_training_end(self) -> None:
        self._emit("interrupted" if self.cancel_requested else "completed")
