"""Deterministic non-AI baseline policies and episode runner."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from reinforcement_learning.environments.single_symbol_env import (
    SingleSymbolTradingEnv,
)

from .metrics import calculate_episode_metrics


class BaselinePolicy(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def action(self, observation: np.ndarray, info: dict[str, object]) -> int: ...


class AlwaysHoldPolicy:
    """Hold on every transition."""

    def reset(self, seed: int | None = None) -> None:
        del seed

    def action(self, observation: np.ndarray, info: dict[str, object]) -> int:
        del observation, info
        return 0


class BuyAndHoldPolicy:
    """Buy on the first transition and hold afterward."""

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._first = True

    def action(self, observation: np.ndarray, info: dict[str, object]) -> int:
        del observation, info
        if self._first:
            self._first = False
            return 1
        return 0


class RandomPolicy:
    """Uniform random discrete actions from an isolated fixed-seed generator."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def action(self, observation: np.ndarray, info: dict[str, object]) -> int:
        del observation, info
        return int(self._rng.integers(0, 3))


@dataclass(frozen=True)
class BaselineResult:
    history: pd.DataFrame
    metrics: dict[str, object]


def run_baseline(
    env: SingleSymbolTradingEnv,
    policy: BaselinePolicy,
    *,
    seed: int = 0,
) -> BaselineResult:
    """Execute one complete deterministic baseline episode."""
    observation, info = env.reset(seed=seed)
    policy.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.action(observation, info)
        observation, _, terminated, truncated, info = env.step(action)
    history = env.get_history()
    return BaselineResult(history, calculate_episode_metrics(history))
