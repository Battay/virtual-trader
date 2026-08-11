"""Progress, diagnostics, and rollout-continuity instrumentation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .callbacks import PPOProgressCallback


def _state_equal(left: object, right: object) -> bool:
    try:
        pairs = (
            (left.pi[0], right.pi[0]),
            (left.pi[1], right.pi[1]),
            (left.vf[0], right.vf[0]),
            (left.vf[1], right.vf[1]),
        )
    except AttributeError:
        return False
    return all(torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in pairs)


class RecurrentProgressCallback(PPOProgressCallback):
    """Prove rollout updates preserve state until a real episode boundary."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.first_episode_start: bool | None = None
        self.rollout_boundaries_observed = 0
        self.rollout_continuity_checks = 0
        self.rollout_continuity_failures = 0
        self.environment_episode_resets = 0
        self.rollout_start_episode_flags: list[bool] = []
        self._previous_rollout_end_state: object | None = None

    def _on_rollout_start(self) -> None:
        episode_starts = np.asarray(self.model._last_episode_starts, dtype=bool)
        self.rollout_start_episode_flags.append(bool(episode_starts.all()))
        if self.first_episode_start is None:
            self.first_episode_start = bool(episode_starts.all())
        elif self._previous_rollout_end_state is not None:
            self.rollout_continuity_checks += 1
            if not _state_equal(
                self._previous_rollout_end_state,
                self.model._last_lstm_states,
            ):
                self.rollout_continuity_failures += 1

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.environment_episode_resets += int(
                np.asarray(dones, dtype=bool).sum()
            )
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        self.rollout_boundaries_observed += 1
        self._previous_rollout_end_state = deepcopy(self.model._last_lstm_states)

    @property
    def rollout_continuity_verified(self) -> bool:
        return (
            self.rollout_continuity_checks > 0
            and self.rollout_continuity_failures == 0
        )
