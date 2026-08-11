"""Bounded diagnostics for balanced-window sector RecurrentPPO runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import numpy as np

from reinforcement_learning.environments.reward import RewardDiagnosticsAccumulator

from .recurrent_callbacks import RecurrentProgressCallback
from .results import PPOTrainingDiagnostics
from .sector_methodology_diagnostics import (
    action_pattern_digest,
    combine_action_counts,
    detect_action_collapse,
)


class BalancedSectorProgressCallback(RecurrentProgressCallback):
    """Retain periodic PPO summaries without logging every gradient update."""

    def __init__(
        self,
        *,
        diagnostic_interval_rollouts: int = 10,
        **kwargs: object,
    ) -> None:
        if (
            isinstance(diagnostic_interval_rollouts, bool)
            or not isinstance(diagnostic_interval_rollouts, int)
            or diagnostic_interval_rollouts < 1
        ):
            raise ValueError("diagnostic_interval_rollouts must be positive")
        super().__init__(**kwargs)
        self.diagnostic_interval_rollouts = diagnostic_interval_rollouts
        self.periodic_diagnostics: list[PPOTrainingDiagnostics] = []
        self._reward_diagnostics = RewardDiagnosticsAccumulator()
        self._actions_by_symbol: dict[str, list[int]] = defaultdict(list)
        self._invested_steps_by_symbol: Counter[str] = Counter()
        self._observed_steps_by_symbol: Counter[str] = Counter()
        self._trade_counts_by_symbol: Counter[str] = Counter()
        self.policy_episode_start_flags: list[bool] = []

    def _capture_logger_diagnostics(self) -> None:
        logger = getattr(self.model, "logger", None)
        values = getattr(logger, "name_to_value", None)
        diagnostic = PPOTrainingDiagnostics.from_sb3_logger_values(
            values,
            timesteps=int(self.num_timesteps),
        )
        has_update_values = any(
            value is not None
            for name, value in diagnostic.to_dict().items()
            if name != "timesteps"
        )
        if not has_update_values:
            return
        if (
            self.periodic_diagnostics
            and self.periodic_diagnostics[-1].timesteps == diagnostic.timesteps
        ):
            return
        self.periodic_diagnostics.append(diagnostic)

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()
        if (
            self.rollout_boundaries_observed > 0
            and self.rollout_boundaries_observed
            % self.diagnostic_interval_rollouts
            == 0
        ):
            self._capture_logger_diagnostics()

    def _on_training_end(self) -> None:
        super()._on_training_end()
        if not self.cancel_requested:
            self._capture_logger_diagnostics()

    def _on_step(self) -> bool:
        # RecurrentPPO later mutates its rollout reward at TimeLimit truncations
        # to add gamma * V(terminal_observation).  Copy the raw, auditable
        # environment reward components from info immediately instead of
        # retaining a reference to ``locals()['rewards']``.
        episode_starts = self.locals.get("episode_starts")
        if episode_starts is not None:
            if hasattr(episode_starts, "detach"):
                starts = episode_starts.detach().cpu().numpy()
            else:
                starts = np.asarray(episode_starts)
            flattened = np.asarray(starts, dtype=bool).reshape(-1)
            if len(flattened) != 1:
                raise RuntimeError(
                    "balanced methodology supports exactly one vector environment"
                )
            self.policy_episode_start_flags.append(bool(flattened[0]))
        infos = self.locals.get("infos", ())
        for raw_info in infos:
            if not isinstance(raw_info, Mapping):
                continue
            symbol = str(raw_info.get("sector_symbol", "")).strip()
            breakdown = raw_info.get("reward_breakdown")
            selected_action = raw_info.get("selected_action")
            if not symbol or not isinstance(breakdown, Mapping) or selected_action is None:
                continue
            action = int(selected_action)
            self._reward_diagnostics.update_from_info(
                symbol=symbol,
                info=dict(raw_info),
            )
            self._actions_by_symbol[symbol].append(action)
            self._observed_steps_by_symbol[symbol] += 1
            if int(raw_info.get("shares_held", 0)) > 0:
                self._invested_steps_by_symbol[symbol] += 1
            if bool(raw_info.get("trade_executed", False)):
                self._trade_counts_by_symbol[symbol] += 1
        return super()._on_step()

    @property
    def methodology_diagnostics(self) -> dict[str, object]:
        """Return raw reward/action aggregates and warning-only collapse flags."""

        reward_summary = self._reward_diagnostics.summary()
        symbols = tuple(sorted(self._actions_by_symbol))
        action_counts = combine_action_counts(self._actions_by_symbol)
        exposures = {
            symbol: (
                100.0
                * self._invested_steps_by_symbol[symbol]
                / self._observed_steps_by_symbol[symbol]
                if self._observed_steps_by_symbol[symbol]
                else 0.0
            )
            for symbol in symbols
        }
        trades = {
            symbol: int(self._trade_counts_by_symbol[symbol]) for symbol in symbols
        }
        pattern_digests = {
            symbol: action_pattern_digest(self._actions_by_symbol[symbol])
            for symbol in symbols
        }
        invalid_rate = float(
            reward_summary["action_validity_counts"]["invalid_action_rate"]
        )
        collapse = detect_action_collapse(
            selected_action_counts=action_counts,
            invalid_action_rate=invalid_rate,
            per_symbol_exposure_percentages=exposures,
            per_symbol_trade_counts=trades,
            per_symbol_action_digests=pattern_digests,
        )
        return {
            "raw_environment_reward_semantics": (
                "excludes RecurrentPPO timeout value-bootstrap adjustment"
            ),
            "reward_and_action": reward_summary,
            "selected_action_counts": action_counts,
            "per_symbol_exposure_percentages": exposures,
            "per_symbol_trade_counts": trades,
            "per_symbol_action_pattern_digests": pattern_digests,
            "collapse_diagnostics": collapse,
        }


__all__ = ("BalancedSectorProgressCallback",)
