"""Isolated multi-symbol environment for sector RecurrentPPO training.

Every constituent remains a complete single-symbol episode.  Gymnasium emits a
real termination at the end of a symbol, allowing SB3-Contrib to set
``episode_start=True`` and clear recurrent state before the next symbol.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .config import SingleSymbolEnvConfig
from .single_symbol_env import SingleSymbolTradingEnv


EQUAL_SYMBOL_EPISODE_SAMPLING = "equal_symbol_episode_sampling_v1"


class SectorEnvironmentError(RuntimeError):
    """Raised when symbol isolation or accounting becomes unsafe."""


class EqualSymbolEpisodeSampler:
    """Deterministic shuffled cycles with exactly one selection per symbol."""

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        seed: int,
        universe_hash: str,
    ) -> None:
        canonical = tuple(str(symbol).strip() for symbol in symbols)
        if not canonical or any(not symbol for symbol in canonical):
            raise ValueError("sampler requires non-empty symbols")
        if len(set(canonical)) != len(canonical):
            raise ValueError("sampler symbols must be unique")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler seed must be a non-negative integer")
        if not isinstance(universe_hash, str) or len(universe_hash) != 64:
            raise ValueError("sampler requires a SHA-256 universe hash")
        try:
            int(universe_hash, 16)
        except ValueError as exc:
            raise ValueError("universe hash must be hexadecimal") from exc
        material = f"{universe_hash}:{seed}:{EQUAL_SYMBOL_EPISODE_SAMPLING}".encode()
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        self.symbols = canonical
        self.seed = seed
        self.universe_hash = universe_hash
        self._rng = np.random.default_rng(derived_seed)
        self._cycle: tuple[str, ...] = ()
        self._index = 0
        self._cycle_number = 0
        self._sequence: list[str] = []

    def next_symbol(self) -> str:
        if self._index >= len(self._cycle):
            permutation = self._rng.permutation(len(self.symbols))
            self._cycle = tuple(self.symbols[int(index)] for index in permutation)
            self._index = 0
            self._cycle_number += 1
        symbol = self._cycle[self._index]
        self._index += 1
        self._sequence.append(symbol)
        return symbol

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(self._sequence)

    @property
    def sequence_digest(self) -> str:
        payload = json.dumps(self._sequence, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def completed_cycles(self) -> int:
        return max(0, self._cycle_number - int(self._index < len(self._cycle)))


class SectorTrainingEnv(gym.Env[np.ndarray, int]):
    """Gymnasium controller that selects one isolated symbol episode per reset."""

    metadata = SingleSymbolTradingEnv.metadata

    def __init__(
        self,
        symbol_train_data: Mapping[str, pd.DataFrame],
        *,
        universe_hash: str,
        seed: int,
        config: SingleSymbolEnvConfig | None = None,
    ) -> None:
        super().__init__()
        if not symbol_train_data:
            raise ValueError("sector environment requires constituent TRAIN data")
        self.config = config or SingleSymbolEnvConfig()
        self._frames = {
            str(symbol).strip(): frame.copy(deep=True)
            for symbol, frame in symbol_train_data.items()
        }
        if any(not symbol for symbol in self._frames):
            raise ValueError("sector symbols cannot be empty")
        self.symbols = tuple(self._frames)
        self.sampler = EqualSymbolEpisodeSampler(
            self.symbols,
            seed=seed,
            universe_hash=universe_hash,
        )
        probe = SingleSymbolTradingEnv(self._frames[self.symbols[0]], self.config)
        self.action_space: spaces.Discrete = probe.action_space
        self.observation_space: spaces.Box = probe.observation_space
        self.observation_feature_names = probe.observation_feature_names
        probe.close()
        self._environment: SingleSymbolTradingEnv | None = None
        self.current_symbol: str | None = None
        self.episode_counts_started: Counter[str] = Counter()
        self.episode_counts_completed: Counter[str] = Counter()
        self.timesteps_by_symbol: Counter[str] = Counter()
        self.termination_reasons: Counter[str] = Counter()
        self.symbol_transition_count = 0
        self.reset_snapshots: list[dict[str, object]] = []
        self._last_symbol: str | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        if self._environment is not None:
            self._environment.close()
        symbol = self.sampler.next_symbol()
        if self._last_symbol is not None and symbol != self._last_symbol:
            self.symbol_transition_count += 1
        self.current_symbol = symbol
        self._last_symbol = symbol
        self._environment = SingleSymbolTradingEnv(self._frames[symbol], self.config)
        observation, info = self._environment.reset(seed=seed)
        self.episode_counts_started[symbol] += 1
        self._validate_runtime(observation, 0.0, info)
        snapshot = {
            "symbol": symbol,
            "episode_start": True,
            "cash": float(self._environment.cash),
            "shares_held": int(self._environment.shares_held),
            "realized_profit_loss": float(self._environment.realized_profit_loss),
            "drawdown": float(self._environment.current_drawdown),
            "portfolio_value": float(self._environment.total_portfolio_value),
        }
        if (
            snapshot["cash"] != self.config.initial_cash
            or snapshot["shares_held"] != 0
            or snapshot["realized_profit_loss"] != 0.0
            or snapshot["drawdown"] != 0.0
            or snapshot["portfolio_value"] != self.config.initial_cash
        ):
            raise SectorEnvironmentError("portfolio state did not reset at symbol boundary")
        self.reset_snapshots.append(snapshot)
        return observation, {**info, "sector_symbol": symbol, "episode_start": True}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._environment is None or self.current_symbol is None:
            raise SectorEnvironmentError("reset must be called before step")
        observation, reward, terminated, truncated, info = self._environment.step(action)
        self.timesteps_by_symbol[self.current_symbol] += 1
        self._validate_runtime(observation, reward, info)
        reason = None
        if terminated:
            reason = "natural_train_partition_end"
        elif truncated:
            reason = "explicit_truncation"
        if reason is not None:
            self.episode_counts_completed[self.current_symbol] += 1
            self.termination_reasons[reason] += 1
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                **info,
                "sector_symbol": self.current_symbol,
                "sector_episode_end_reason": reason,
            },
        )

    def _validate_runtime(
        self,
        observation: np.ndarray,
        reward: float,
        info: Mapping[str, object],
    ) -> None:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != self.observation_space.shape or not np.isfinite(values).all():
            raise SectorEnvironmentError("sector observation is invalid or non-finite")
        if not math.isfinite(float(reward)):
            raise SectorEnvironmentError("sector reward is non-finite")
        for name in ("cash", "portfolio_value"):
            value = float(info[name])
            if not math.isfinite(value) or value < 0:
                raise SectorEnvironmentError(f"sector {name} is invalid")
        shares = int(info["shares_held"])
        if shares < 0:
            raise SectorEnvironmentError("sector shares held cannot be negative")
        execution_price = info.get("execution_price")
        if execution_price is not None and (
            not math.isfinite(float(execution_price)) or float(execution_price) <= 0
        ):
            raise SectorEnvironmentError("sector execution price must be positive")

    @property
    def sampling_sequence(self) -> tuple[str, ...]:
        return self.sampler.sequence

    @property
    def sampling_sequence_digest(self) -> str:
        return self.sampler.sequence_digest

    def get_history(self) -> pd.DataFrame:
        if self._environment is None:
            return pd.DataFrame()
        return self._environment.get_history()

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None


__all__ = (
    "EQUAL_SYMBOL_EPISODE_SAMPLING",
    "EqualSymbolEpisodeSampler",
    "SectorEnvironmentError",
    "SectorTrainingEnv",
)
