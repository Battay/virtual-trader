"""Deterministic TRAIN-only windows for fair sector recurrent experiments.

The schedule is data infrastructure, not a training entry point.  In
particular, importing this module cannot start the predeclared three-seed
experiment.  A window of 512 *transitions* contains 513 source rows because
the environment observes row ``t`` and executes at row ``t + 1``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd

from reinforcement_learning.environments.config import SingleSymbolEnvConfig
from reinforcement_learning.environments.single_symbol_env import (
    SingleSymbolTradingEnv,
)


SECTOR_BALANCED_WINDOW_SAMPLING_VERSION = "sector_sampling_balanced_windows_v1"
FAIR_SECTOR_EXPERIMENT_VERSION = "sector_recurrent_fair_multiseed_experiment_v1"
DEFAULT_REWARD_VERSION = "sector_reward_v1"
DEFAULT_ACTION_VALIDITY_VERSION = "sector_action_validity_v1"
DEFAULT_WINDOW_TRANSITIONS = 512
DEFAULT_WINDOW_SOURCE_ROWS = DEFAULT_WINDOW_TRANSITIONS + 1
DEFAULT_BALANCED_ROUNDS = 20
DEFAULT_DATA_SCHEDULE_SEED = 42
PREDECLARED_MODEL_SEEDS = (42, 43, 44)
COMMERCIAL_BANKS_CONSTITUENT_COUNT = 19
PREDECLARED_TRANSITIONS_PER_SYMBOL = (
    DEFAULT_WINDOW_TRANSITIONS * DEFAULT_BALANCED_ROUNDS
)
PREDECLARED_TOTAL_SECTOR_TRANSITIONS = (
    COMMERCIAL_BANKS_CONSTITUENT_COUNT * PREDECLARED_TRANSITIONS_PER_SYMBOL
)


class BalancedWindowScheduleError(ValueError):
    """Raised when a window schedule would violate the research contract."""


class BalancedWindowScheduleExhausted(RuntimeError):
    """Raised when a bounded schedule has no next independent episode."""


def canonical_payload_hash(payload: Mapping[str, object]) -> str:
    """Return a portable SHA-256 over canonical JSON content."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BalancedWindowScheduleError(
            "deterministic identity must contain finite JSON values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BalancedWindowScheduleError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BalancedWindowScheduleError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BalancedWindowScheduleError(
            f"{label} must be a non-negative integer"
        )
    return value


def _iso_date(value: object) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise BalancedWindowScheduleError("window dates cannot be missing")
    return parsed.date().isoformat()


def _derived_seed(*parts: object) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class BalancedWindowRecord:
    """One independently reset chronological symbol window.

    Row bounds are zero-based and inclusive.  ``final_observation_date`` is
    the observation used to choose the final action; that action executes on
    ``final_execution_date``.  The resulting terminal observation is also the
    final execution row.
    """

    schedule_index: int
    round_number: int
    position_in_round: int
    symbol: str
    window_number_for_symbol: int
    source_start_row: int
    source_end_row: int
    source_row_count: int
    expected_transition_count: int
    actual_transition_count: int
    start_date: str
    final_observation_date: str
    final_execution_date: str
    terminal_observation_date: str
    start_chronology_quartile: str
    overlaps_previous_windows: bool
    overlapping_transition_count: int
    new_unique_transition_count: int
    source_window_reused: bool
    cumulative_unique_coverage_percentage: float
    boundary_kind: str
    expected_terminated: bool
    expected_truncated: bool
    partition: str = "train"
    episode_start: bool = True
    recurrent_state_reset: bool = True
    portfolio_state_reset: bool = True

    def __post_init__(self) -> None:
        if self.partition != "train":
            raise BalancedWindowScheduleError("balanced windows must be TRAIN-only")
        if self.source_start_row < 0 or self.source_end_row < self.source_start_row:
            raise BalancedWindowScheduleError("window source row bounds are invalid")
        if self.source_row_count != self.source_end_row - self.source_start_row + 1:
            raise BalancedWindowScheduleError("window source row count is inconsistent")
        if self.actual_transition_count != self.source_row_count - 1:
            raise BalancedWindowScheduleError(
                "window transition count must equal source row count minus one"
            )
        if self.actual_transition_count != self.expected_transition_count:
            raise BalancedWindowScheduleError(
                "constructed window cannot supply its expected transitions"
            )
        if self.start_chronology_quartile not in {"Q1", "Q2", "Q3", "Q4"}:
            raise BalancedWindowScheduleError("window chronology quartile is invalid")
        if (
            self.overlapping_transition_count < 0
            or self.new_unique_transition_count < 0
            or self.overlapping_transition_count + self.new_unique_transition_count
            != self.actual_transition_count
        ):
            raise BalancedWindowScheduleError(
                "window overlap and unique-transition counts are inconsistent"
            )
        if self.overlaps_previous_windows != (self.overlapping_transition_count > 0):
            raise BalancedWindowScheduleError("window overlap flag is inconsistent")
        if not 0.0 <= self.cumulative_unique_coverage_percentage <= 100.0:
            raise BalancedWindowScheduleError(
                "window cumulative historical coverage must be within 0-100"
            )
        if not (
            self.episode_start
            and self.recurrent_state_reset
            and self.portfolio_state_reset
        ):
            raise BalancedWindowScheduleError(
                "every balanced window must reset recurrent and portfolio state"
            )
        if self.boundary_kind not in {
            "artificial_window_truncation",
            "natural_train_partition_end",
        }:
            raise BalancedWindowScheduleError("window boundary kind is unsupported")
        if self.boundary_kind == "artificial_window_truncation":
            if self.expected_terminated or not self.expected_truncated:
                raise BalancedWindowScheduleError(
                    "artificial window ends must use truncation semantics"
                )
        elif not self.expected_terminated or self.expected_truncated:
            raise BalancedWindowScheduleError(
                "natural TRAIN ends must use termination semantics"
            )

    @property
    def transition_source_rows(self) -> range:
        """Source row indices at which the scheduled decisions are observed."""

        return range(self.source_start_row, self.source_end_row)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChronologyQuartileCoverage:
    quartile: str
    available_unique_transitions: int
    unique_transitions_used: int
    scheduled_transition_occurrences: int
    window_start_count: int
    coverage_percentage: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolWindowCoverage:
    symbol: str
    source_row_count: int
    available_unique_train_transitions: int
    scheduled_transitions: int
    unique_transitions_used: int
    unused_unique_train_transitions: int
    repeated_transition_occurrences: int
    overlap_percentage: float
    minimum_usage_count: int
    maximum_usage_count: int
    coverage_percentage: float
    window_count: int
    unique_window_count: int
    reused_window_count: int
    overlapping_window_count: int
    natural_boundary_count: int
    artificial_boundary_count: int
    chronological_quartiles: tuple[ChronologyQuartileCoverage, ...]
    all_chronological_quartiles_represented: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["chronological_quartiles"] = [
            item.to_dict() for item in self.chronological_quartiles
        ]
        return value


def _schedule_identity(
    *,
    sampling_version: str,
    universe_hash: str,
    partition: str,
    window_transition_count: int,
    source_rows_per_window: int,
    rounds: int,
    data_schedule_seed: int,
    symbols: Sequence[str],
    target_symbol: str | None,
    target_excluded_from_pretraining: bool,
    normalization_contributors: Sequence[str],
    records: Sequence[BalancedWindowRecord],
    symbol_statistics: Sequence[SymbolWindowCoverage],
) -> dict[str, object]:
    statistics = {item.symbol: item for item in symbol_statistics}
    expected_transitions = rounds * window_transition_count
    equal_exposure = {
        symbol: {
            "requested_transitions": expected_transitions,
            "actual_constructed_transitions": sum(
                record.actual_transition_count
                for record in records
                if record.symbol == symbol
            ),
            "contribution_percentage": 100.0 / len(symbols),
            "window_count": rounds,
            "unique_transitions_used": statistics[symbol].unique_transitions_used,
            "repeated_transition_occurrences": statistics[
                symbol
            ].repeated_transition_occurrences,
        }
        for symbol in symbols
    }
    return {
        "sampling_version": sampling_version,
        "universe_hash": universe_hash,
        "partition": partition,
        "window_transition_count": window_transition_count,
        "source_rows_per_window": source_rows_per_window,
        "rounds": rounds,
        "data_schedule_seed": data_schedule_seed,
        "symbols": list(symbols),
        "target_symbol": target_symbol,
        "target_excluded_from_pretraining": target_excluded_from_pretraining,
        "normalization_contributors": list(normalization_contributors),
        "records": [record.to_dict() for record in records],
        "symbol_statistics": [item.to_dict() for item in symbol_statistics],
        "equal_exposure": equal_exposure,
    }


@dataclass(frozen=True)
class BalancedWindowSchedule:
    sampling_version: str
    universe_hash: str
    partition: str
    window_transition_count: int
    source_rows_per_window: int
    rounds: int
    data_schedule_seed: int
    symbols: tuple[str, ...]
    target_symbol: str | None
    target_excluded_from_pretraining: bool
    normalization_contributors: tuple[str, ...]
    records: tuple[BalancedWindowRecord, ...]
    symbol_statistics: tuple[SymbolWindowCoverage, ...]
    schedule_digest: str

    def __post_init__(self) -> None:
        if self.sampling_version != SECTOR_BALANCED_WINDOW_SAMPLING_VERSION:
            raise BalancedWindowScheduleError("sampling version is incompatible")
        _valid_sha256(self.universe_hash, label="universe_hash")
        if self.partition != "train":
            raise BalancedWindowScheduleError("schedule must be TRAIN-only")
        if self.source_rows_per_window != self.window_transition_count + 1:
            raise BalancedWindowScheduleError(
                "source rows per window must equal transitions plus one"
            )
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise BalancedWindowScheduleError("schedule symbols must be non-empty and unique")
        if self.normalization_contributors != self.symbols:
            raise BalancedWindowScheduleError(
                "only scheduled symbols may contribute normalization metadata"
            )
        if self.target_symbol is not None:
            if not self.target_excluded_from_pretraining:
                raise BalancedWindowScheduleError("declared target must be excluded")
            if self.target_symbol in self.symbols:
                raise BalancedWindowScheduleError("target appears in its own schedule")
        elif self.target_excluded_from_pretraining:
            raise BalancedWindowScheduleError("target exclusion requires a target")
        if len(self.records) != self.rounds * len(self.symbols):
            raise BalancedWindowScheduleError("schedule record count is inconsistent")
        if tuple(item.symbol for item in self.symbol_statistics) != self.symbols:
            raise BalancedWindowScheduleError(
                "coverage statistics must correspond to every scheduled symbol"
            )
        if tuple(record.schedule_index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise BalancedWindowScheduleError("schedule indices must be contiguous")
        expected = self.rounds * self.window_transition_count
        per_symbol = Counter(record.symbol for record in self.records)
        transitions = Counter()
        for record in self.records:
            if record.symbol not in self.symbols:
                raise BalancedWindowScheduleError("record contains an unknown symbol")
            if (
                record.expected_transition_count != self.window_transition_count
                or record.actual_transition_count != self.window_transition_count
                or record.source_row_count != self.source_rows_per_window
            ):
                raise BalancedWindowScheduleError(
                    "record does not match the schedule window dimensions"
                )
            transitions[record.symbol] += record.actual_transition_count
        if any(per_symbol[symbol] != self.rounds for symbol in self.symbols):
            raise BalancedWindowScheduleError("every round must schedule every symbol once")
        if any(transitions[symbol] != expected for symbol in self.symbols):
            raise BalancedWindowScheduleError("scheduled symbol exposure is unequal")
        for round_number in range(1, self.rounds + 1):
            round_symbols = [
                record.symbol
                for record in self.records
                if record.round_number == round_number
            ]
            if len(round_symbols) != len(self.symbols) or set(round_symbols) != set(
                self.symbols
            ):
                raise BalancedWindowScheduleError(
                    "each balanced round must contain every symbol exactly once"
                )
            round_records = [
                record
                for record in self.records
                if record.round_number == round_number
            ]
            if tuple(item.position_in_round for item in round_records) != tuple(
                range(1, len(self.symbols) + 1)
            ) or any(
                item.window_number_for_symbol != round_number
                for item in round_records
            ):
                raise BalancedWindowScheduleError(
                    "round positions or per-symbol window numbers are inconsistent"
                )
        statistics = self.statistics_by_symbol
        if any(
            statistics[symbol].window_count != self.rounds
            or statistics[symbol].scheduled_transitions != expected
            for symbol in self.symbols
        ):
            raise BalancedWindowScheduleError(
                "coverage statistics disagree with scheduled exposure"
            )
        if canonical_payload_hash(self.deterministic_identity()) != self.schedule_digest:
            raise BalancedWindowScheduleError("schedule digest is stale")

    @property
    def expected_transitions_per_symbol(self) -> int:
        return self.rounds * self.window_transition_count

    @property
    def expected_total_scheduled_transitions(self) -> int:
        return len(self.symbols) * self.expected_transitions_per_symbol

    @property
    def full_commercial_banks_research_contract(self) -> bool:
        return (
            self.target_symbol is None
            and len(self.symbols) == COMMERCIAL_BANKS_CONSTITUENT_COUNT
            and self.window_transition_count == DEFAULT_WINDOW_TRANSITIONS
            and self.rounds == DEFAULT_BALANCED_ROUNDS
            and self.expected_transitions_per_symbol
            == PREDECLARED_TRANSITIONS_PER_SYMBOL
            and self.expected_total_scheduled_transitions
            == PREDECLARED_TOTAL_SECTOR_TRANSITIONS
        )

    @property
    def statistics_by_symbol(self) -> dict[str, SymbolWindowCoverage]:
        return {item.symbol: item for item in self.symbol_statistics}

    @property
    def exposure_by_symbol(self) -> dict[str, dict[str, int | float]]:
        """Exact equal optimization exposure, separate from unique coverage."""

        contribution = 100.0 / len(self.symbols)
        return {
            symbol: {
                "requested_transitions": self.expected_transitions_per_symbol,
                "actual_constructed_transitions": sum(
                    record.actual_transition_count
                    for record in self.records
                    if record.symbol == symbol
                ),
                "contribution_percentage": contribution,
                "window_count": self.rounds,
                "unique_transitions_used": self.statistics_by_symbol[
                    symbol
                ].unique_transitions_used,
                "repeated_transition_occurrences": self.statistics_by_symbol[
                    symbol
                ].repeated_transition_occurrences,
            }
            for symbol in self.symbols
        }

    def deterministic_identity(self) -> dict[str, object]:
        """Canonical schedule identity; contains no time, host, or path fields."""

        return _schedule_identity(
            sampling_version=self.sampling_version,
            universe_hash=self.universe_hash,
            partition=self.partition,
            window_transition_count=self.window_transition_count,
            source_rows_per_window=self.source_rows_per_window,
            rounds=self.rounds,
            data_schedule_seed=self.data_schedule_seed,
            symbols=self.symbols,
            target_symbol=self.target_symbol,
            target_excluded_from_pretraining=self.target_excluded_from_pretraining,
            normalization_contributors=self.normalization_contributors,
            records=self.records,
            symbol_statistics=self.symbol_statistics,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.deterministic_identity(), "schedule_digest": self.schedule_digest}

    def assert_full_research_contract(self) -> None:
        if not self.full_commercial_banks_research_contract:
            raise BalancedWindowScheduleError(
                "schedule is not the frozen 19 x 20 x 512 Commercial Banks design"
            )


def _validate_train_frame(
    symbol: str,
    frame: pd.DataFrame,
    *,
    source_rows_per_window: int,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise BalancedWindowScheduleError(f"{symbol}: TRAIN source is not a dataframe")
    required = {"symbol", "date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BalancedWindowScheduleError(
            f"{symbol}: TRAIN source lacks {', '.join(missing)}"
        )
    if len(frame) < source_rows_per_window:
        raise BalancedWindowScheduleError(
            f"{symbol}: needs at least {source_rows_per_window} source rows for "
            f"{source_rows_per_window - 1} transitions"
        )
    symbol_values = frame["symbol"].astype("string").str.strip()
    if symbol_values.isna().any() or not symbol_values.eq(symbol).all():
        raise BalancedWindowScheduleError(f"{symbol}: TRAIN source symbol differs")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise BalancedWindowScheduleError(
            f"{symbol}: TRAIN dates must be unique and chronological"
        )
    return frame.copy(deep=True).reset_index(drop=True)


def _stratified_window_start(
    *,
    valid_start_count: int,
    round_index: int,
    rounds: int,
    symbol: str,
    universe_hash: str,
    data_schedule_seed: int,
) -> int:
    """Choose one seeded location from each chronological start-position stratum."""

    if valid_start_count >= rounds:
        lower = math.floor(round_index * valid_start_count / rounds)
        upper = math.floor((round_index + 1) * valid_start_count / rounds) - 1
    else:
        # Extremely short eligible histories necessarily reuse start locations.
        lower = upper = round(
            round_index * (valid_start_count - 1) / max(1, rounds - 1)
        )
    width = upper - lower + 1
    offset = _derived_seed(
        universe_hash,
        data_schedule_seed,
        SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
        "window_location",
        symbol,
        round_index + 1,
    ) % width
    return lower + offset


def _round_symbol_order(
    symbols: Sequence[str],
    *,
    round_number: int,
    universe_hash: str,
    data_schedule_seed: int,
) -> tuple[str, ...]:
    order = list(symbols)
    random.Random(
        _derived_seed(
            universe_hash,
            data_schedule_seed,
            SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
            "round_order",
            round_number,
        )
    ).shuffle(order)
    return tuple(order)


def _coverage_for_symbol(
    symbol: str,
    *,
    source_row_count: int,
    records: Sequence[BalancedWindowRecord],
) -> SymbolWindowCoverage:
    available = source_row_count - 1
    usage: Counter[int] = Counter()
    window_starts: Counter[int] = Counter()
    overlapping_windows = 0
    previously_used: set[int] = set()
    for record in sorted(records, key=lambda item: item.window_number_for_symbol):
        transitions = set(record.transition_source_rows)
        if transitions.intersection(previously_used):
            overlapping_windows += 1
        previously_used.update(transitions)
        usage.update(record.transition_source_rows)
        window_starts[record.source_start_row] += 1
    scheduled = sum(usage.values())
    unique_used = len(usage)
    repeated = scheduled - unique_used
    quartiles: list[ChronologyQuartileCoverage] = []
    for quartile_index in range(4):
        available_indices = [
            index
            for index in range(available)
            if min(3, (4 * index) // available) == quartile_index
        ]
        used = sum(index in usage for index in available_indices)
        occurrences = sum(usage[index] for index in available_indices)
        starts = sum(
            count
            for index, count in window_starts.items()
            if min(3, (4 * index) // available) == quartile_index
        )
        quartiles.append(
            ChronologyQuartileCoverage(
                quartile=f"Q{quartile_index + 1}",
                available_unique_transitions=len(available_indices),
                unique_transitions_used=used,
                scheduled_transition_occurrences=occurrences,
                window_start_count=starts,
                coverage_percentage=(
                    100.0 * used / len(available_indices)
                    if available_indices
                    else 0.0
                ),
            )
        )
    starts = [record.source_start_row for record in records]
    return SymbolWindowCoverage(
        symbol=symbol,
        source_row_count=source_row_count,
        available_unique_train_transitions=available,
        scheduled_transitions=scheduled,
        unique_transitions_used=unique_used,
        unused_unique_train_transitions=available - unique_used,
        repeated_transition_occurrences=repeated,
        overlap_percentage=(100.0 * repeated / scheduled if scheduled else 0.0),
        minimum_usage_count=min(usage.values()),
        maximum_usage_count=max(usage.values()),
        coverage_percentage=100.0 * unique_used / available,
        window_count=len(records),
        unique_window_count=len(set(starts)),
        reused_window_count=len(starts) - len(set(starts)),
        overlapping_window_count=overlapping_windows,
        natural_boundary_count=sum(
            record.boundary_kind == "natural_train_partition_end"
            for record in records
        ),
        artificial_boundary_count=sum(
            record.boundary_kind == "artificial_window_truncation"
            for record in records
        ),
        chronological_quartiles=tuple(quartiles),
        all_chronological_quartiles_represented=all(
            item.unique_transitions_used > 0 for item in quartiles
        ),
    )


def build_balanced_window_schedule(
    train_data: Mapping[str, pd.DataFrame],
    *,
    universe_hash: str,
    rounds: int = DEFAULT_BALANCED_ROUNDS,
    window_transition_count: int = DEFAULT_WINDOW_TRANSITIONS,
    data_schedule_seed: int = DEFAULT_DATA_SCHEDULE_SEED,
    partition: str = "train",
    target_symbol: str | None = None,
) -> BalancedWindowSchedule:
    """Build an immutable, equal-exposure, chronology-spread TRAIN schedule.

    Model seed is deliberately not an argument: model seeds 42/43/44 consume
    identical window locations and round order.  ``data_schedule_seed`` may be
    changed only by declaring a different data schedule.
    """

    _valid_sha256(universe_hash, label="universe_hash")
    rounds = _positive_integer(rounds, label="rounds")
    window_transition_count = _positive_integer(
        window_transition_count, label="window_transition_count"
    )
    data_schedule_seed = _non_negative_integer(
        data_schedule_seed, label="data_schedule_seed"
    )
    if partition != "train":
        raise BalancedWindowScheduleError(
            "balanced scheduler may not load VALIDATION or TEST"
        )
    if not train_data:
        raise BalancedWindowScheduleError("balanced schedule requires TRAIN data")
    canonical_input: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in train_data.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in canonical_input:
            raise BalancedWindowScheduleError("TRAIN symbols must be non-empty and unique")
        canonical_input[symbol] = frame
    normalized_target = str(target_symbol).strip().upper() if target_symbol else None
    if normalized_target is not None:
        if normalized_target in canonical_input:
            raise BalancedWindowScheduleError(
                "leave-one-out target must be excluded before TRAIN data is supplied"
            )
        if not canonical_input:
            raise BalancedWindowScheduleError(
                "leave-one-out schedule requires at least one peer symbol"
            )
    symbols = tuple(sorted(canonical_input))
    source_rows_per_window = window_transition_count + 1
    frames = {
        symbol: _validate_train_frame(
            symbol,
            canonical_input[symbol],
            source_rows_per_window=source_rows_per_window,
        )
        for symbol in symbols
    }
    records: list[BalancedWindowRecord] = []
    schedule_index = 0
    prior_transitions: dict[str, set[int]] = {symbol: set() for symbol in symbols}
    prior_window_starts: dict[str, set[int]] = {symbol: set() for symbol in symbols}
    for round_index in range(rounds):
        starts = {
            symbol: _stratified_window_start(
                valid_start_count=len(frames[symbol]) - window_transition_count,
                round_index=round_index,
                rounds=rounds,
                symbol=symbol,
                universe_hash=universe_hash,
                data_schedule_seed=data_schedule_seed,
            )
            for symbol in symbols
        }
        order = _round_symbol_order(
            symbols,
            round_number=round_index + 1,
            universe_hash=universe_hash,
            data_schedule_seed=data_schedule_seed,
        )
        for position, symbol in enumerate(order, start=1):
            frame = frames[symbol]
            start = starts[symbol]
            end = start + window_transition_count
            natural = end == len(frame) - 1
            transition_rows = set(range(start, end))
            overlap_count = len(transition_rows.intersection(prior_transitions[symbol]))
            new_unique_count = len(transition_rows) - overlap_count
            start_quartile_index = min(
                3, (4 * start) // (len(frame) - 1)
            )
            record = BalancedWindowRecord(
                schedule_index=schedule_index,
                round_number=round_index + 1,
                position_in_round=position,
                symbol=symbol,
                window_number_for_symbol=round_index + 1,
                source_start_row=start,
                source_end_row=end,
                source_row_count=end - start + 1,
                expected_transition_count=window_transition_count,
                actual_transition_count=end - start,
                start_date=_iso_date(frame["date"].iloc[start]),
                final_observation_date=_iso_date(frame["date"].iloc[end - 1]),
                final_execution_date=_iso_date(frame["date"].iloc[end]),
                terminal_observation_date=_iso_date(frame["date"].iloc[end]),
                start_chronology_quartile=f"Q{start_quartile_index + 1}",
                overlaps_previous_windows=overlap_count > 0,
                overlapping_transition_count=overlap_count,
                new_unique_transition_count=new_unique_count,
                source_window_reused=start in prior_window_starts[symbol],
                cumulative_unique_coverage_percentage=(
                    100.0
                    * len(prior_transitions[symbol].union(transition_rows))
                    / (len(frame) - 1)
                ),
                boundary_kind=(
                    "natural_train_partition_end"
                    if natural
                    else "artificial_window_truncation"
                ),
                expected_terminated=natural,
                expected_truncated=not natural,
            )
            records.append(record)
            prior_transitions[symbol].update(transition_rows)
            prior_window_starts[symbol].add(start)
            schedule_index += 1
    statistics = tuple(
        _coverage_for_symbol(
            symbol,
            source_row_count=len(frames[symbol]),
            records=[record for record in records if record.symbol == symbol],
        )
        for symbol in symbols
    )
    fields = {
        "sampling_version": SECTOR_BALANCED_WINDOW_SAMPLING_VERSION,
        "universe_hash": universe_hash,
        "partition": "train",
        "window_transition_count": window_transition_count,
        "source_rows_per_window": source_rows_per_window,
        "rounds": rounds,
        "data_schedule_seed": data_schedule_seed,
        "symbols": symbols,
        "target_symbol": normalized_target,
        "target_excluded_from_pretraining": normalized_target is not None,
        "normalization_contributors": symbols,
        "records": tuple(records),
        "symbol_statistics": statistics,
    }
    identity = _schedule_identity(**fields)
    return BalancedWindowSchedule(
        **fields,
        schedule_digest=canonical_payload_hash(identity),
    )


def materialize_scheduled_window(
    record: BalancedWindowRecord,
    train_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return an independent 513-row copy for one 512-transition episode."""

    if record.partition != "train":
        raise BalancedWindowScheduleError("only TRAIN windows may be materialized")
    if record.symbol not in train_data:
        raise BalancedWindowScheduleError(
            f"scheduled TRAIN symbol is unavailable: {record.symbol}"
        )
    source = train_data[record.symbol]
    if record.source_end_row >= len(source):
        raise BalancedWindowScheduleError("scheduled window exceeds TRAIN source")
    window = source.iloc[
        record.source_start_row : record.source_end_row + 1
    ].copy(deep=True).reset_index(drop=True)
    if len(window) != record.source_row_count:
        raise BalancedWindowScheduleError("materialized source row count differs")
    if len(window) - 1 != record.expected_transition_count:
        raise BalancedWindowScheduleError("materialized transition count differs")
    if (
        _iso_date(window["date"].iloc[0]) != record.start_date
        or _iso_date(window["date"].iloc[-2]) != record.final_observation_date
        or _iso_date(window["date"].iloc[-1]) != record.final_execution_date
    ):
        raise BalancedWindowScheduleError("materialized TRAIN dates differ")
    return window


class BalancedWindowTrainingEnv(gym.Env[np.ndarray, int]):
    """Execute prescribed windows as isolated Gymnasium episodes.

    The underlying 513-row environment naturally exhausts its slice after 512
    steps.  This controller converts that end to ``truncated=True`` when the
    slice ended before the original TRAIN partition.  Gymnasium/SB3 can then
    bootstrap the terminal value; the returned info explicitly carries
    ``TimeLimit.truncated`` and ``terminal_observation`` as an additional audit
    trail.  A genuine source-partition end remains ``terminated=True``.
    """

    metadata = SingleSymbolTradingEnv.metadata

    def __init__(
        self,
        schedule: BalancedWindowSchedule,
        train_data: Mapping[str, pd.DataFrame],
        *,
        config: SingleSymbolEnvConfig | None = None,
        cycle_schedule: bool = False,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.config = config or SingleSymbolEnvConfig()
        if self.config.max_episode_steps is not None:
            raise BalancedWindowScheduleError(
                "balanced windows own truncation; max_episode_steps must be None"
            )
        self._frames = {
            symbol: frame.copy(deep=True).reset_index(drop=True)
            for symbol, frame in train_data.items()
            if symbol in schedule.symbols
        }
        if set(self._frames) != set(schedule.symbols):
            raise BalancedWindowScheduleError("window controller lacks TRAIN symbols")
        probe_record = schedule.records[0]
        probe = SingleSymbolTradingEnv(
            materialize_scheduled_window(probe_record, self._frames), self.config
        )
        self.action_space = probe.action_space
        self.observation_space = probe.observation_space
        self.observation_feature_names = probe.observation_feature_names
        probe.close()
        self.cycle_schedule = bool(cycle_schedule)
        self._next_record_index = 0
        self._environment: SingleSymbolTradingEnv | None = None
        self.current_record: BalancedWindowRecord | None = None
        self._current_transition_count = 0
        self.actual_transitions_by_symbol: Counter[str] = Counter()
        self.completed_window_counts: Counter[str] = Counter()
        self.reset_snapshots: list[dict[str, object]] = []
        self.completed_records: list[int] = []
        self.passive_post_schedule_reset_count = 0
        self._passive_schedule_complete = False

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
        if self._next_record_index >= len(self.schedule.records):
            if self.cycle_schedule:
                self._next_record_index = 0
            else:
                # DummyVecEnv immediately resets a completed environment before
                # returning the terminal step.  Supply one valid, non-counted
                # observation so this mandatory reset cannot invent another
                # scheduled episode or exposure.  Stepping it is prohibited.
                probe_record = self.schedule.records[-1]
                probe = SingleSymbolTradingEnv(
                    materialize_scheduled_window(probe_record, self._frames),
                    self.config,
                )
                observation, info = probe.reset(seed=seed)
                probe.close()
                self._environment = None
                self.current_record = None
                self._current_transition_count = 0
                self._passive_schedule_complete = True
                self.passive_post_schedule_reset_count += 1
                return observation, {
                    **info,
                    "schedule_complete": True,
                    "passive_post_schedule_reset": True,
                    "episode_start": False,
                    "recurrent_state_reset_required": True,
                }
        record = self.schedule.records[self._next_record_index]
        self._next_record_index += 1
        window = materialize_scheduled_window(record, self._frames)
        self._environment = SingleSymbolTradingEnv(window, self.config)
        observation, info = self._environment.reset(seed=seed)
        self.current_record = record
        self._current_transition_count = 0
        self._passive_schedule_complete = False
        snapshot = {
            "schedule_index": record.schedule_index,
            "symbol": record.symbol,
            "episode_start": True,
            "recurrent_state_reset_required": True,
            "cash": float(self._environment.cash),
            "shares_held": int(self._environment.shares_held),
            "average_entry_price": float(
                self._environment.average_entry_price
            ),
            "current_position_value": float(
                self._environment.current_position_value
            ),
            "realized_profit_loss": float(self._environment.realized_profit_loss),
            "unrealized_profit_loss": float(
                self._environment.unrealized_profit_loss
            ),
            "total_transaction_costs": float(
                self._environment.total_transaction_costs
            ),
            "number_of_trades": int(self._environment.number_of_trades),
            "drawdown": float(self._environment.current_drawdown),
            "portfolio_value": float(self._environment.total_portfolio_value),
            "peak_portfolio_value": float(
                self._environment.peak_portfolio_value
            ),
        }
        if (
            snapshot["cash"] != self.config.initial_cash
            or snapshot["shares_held"] != 0
            or snapshot["average_entry_price"] != 0.0
            or snapshot["current_position_value"] != 0.0
            or snapshot["realized_profit_loss"] != 0.0
            or snapshot["unrealized_profit_loss"] != 0.0
            or snapshot["total_transaction_costs"] != 0.0
            or snapshot["number_of_trades"] != 0
            or snapshot["drawdown"] != 0.0
            or snapshot["portfolio_value"] != self.config.initial_cash
            or snapshot["peak_portfolio_value"] != self.config.initial_cash
        ):
            raise BalancedWindowScheduleError(
                "portfolio state did not reset at window boundary"
            )
        self.reset_snapshots.append(snapshot)
        return observation, {
            **info,
            "sector_symbol": record.symbol,
            "balanced_window_schedule_index": record.schedule_index,
            "episode_start": True,
            "recurrent_state_reset_required": True,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._passive_schedule_complete:
            raise BalancedWindowScheduleExhausted(
                "balanced schedule is complete; passive reset cannot be stepped"
            )
        if self._environment is None or self.current_record is None:
            raise BalancedWindowScheduleError("reset must be called before step")
        observation, reward, terminated, truncated, info = self._environment.step(action)
        self._current_transition_count += 1
        self.actual_transitions_by_symbol[self.current_record.symbol] += 1
        if not np.isfinite(observation).all() or not math.isfinite(float(reward)):
            raise BalancedWindowScheduleError("window produced non-finite output")
        if terminated or truncated:
            if self._current_transition_count != self.current_record.expected_transition_count:
                raise BalancedWindowScheduleError(
                    "window ended with an off-by-one transition count"
                )
            if self.current_record.boundary_kind == "artificial_window_truncation":
                terminated, truncated = False, True
            else:
                terminated, truncated = True, False
            self.completed_window_counts[self.current_record.symbol] += 1
            self.completed_records.append(self.current_record.schedule_index)
            info = {
                **info,
                "TimeLimit.truncated": bool(truncated and not terminated),
                "terminal_observation": np.asarray(observation, dtype=np.float32).copy(),
            }
        return observation, reward, terminated, truncated, {
            **info,
            "sector_symbol": self.current_record.symbol,
            "balanced_window_schedule_index": self.current_record.schedule_index,
            "balanced_window_boundary_kind": self.current_record.boundary_kind,
            "window_transition_number": self._current_transition_count,
        }

    def get_history(self) -> pd.DataFrame:
        if self._environment is None:
            return pd.DataFrame()
        return self._environment.get_history()

    def assert_exact_schedule_completion(self) -> None:
        """Fail if runtime exposure differs from the frozen schedule."""

        expected_records = [record.schedule_index for record in self.schedule.records]
        if self.completed_records != expected_records:
            raise BalancedWindowScheduleError(
                "runtime did not complete every scheduled window exactly once"
            )
        expected_transitions = {
            symbol: self.schedule.expected_transitions_per_symbol
            for symbol in self.schedule.symbols
        }
        if dict(self.actual_transitions_by_symbol) != expected_transitions:
            raise BalancedWindowScheduleError(
                "runtime transition exposure differs from the balanced schedule"
            )
        expected_windows = {
            symbol: self.schedule.rounds for symbol in self.schedule.symbols
        }
        if dict(self.completed_window_counts) != expected_windows:
            raise BalancedWindowScheduleError(
                "runtime window counts differ from the balanced schedule"
            )

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None


def build_predeclared_fair_experiment_spec(
    schedule: BalancedWindowSchedule,
    *,
    taxonomy_version: str,
    trainer_version: str,
    recurrent_contract_version: str,
    feature_version: str,
    environment_version: str,
    ppo_configuration: Mapping[str, object],
    recurrent_architecture: Mapping[str, object],
    reward_configuration: Mapping[str, object],
    action_validity_configuration: Mapping[str, object],
    git_commit: str | None,
    reward_version: str = DEFAULT_REWARD_VERSION,
    action_validity_version: str = DEFAULT_ACTION_VALIDITY_VERSION,
    normalization_scope: str = "symbol",
) -> dict[str, object]:
    """Freeze, but never execute, the future fair three-model-seed protocol."""

    if schedule.window_transition_count != DEFAULT_WINDOW_TRANSITIONS:
        raise BalancedWindowScheduleError("frozen experiment requires 512 transitions/window")
    if schedule.rounds != DEFAULT_BALANCED_ROUNDS:
        raise BalancedWindowScheduleError("frozen experiment requires 20 rounds")
    for label, value in (
        ("taxonomy_version", taxonomy_version),
        ("trainer_version", trainer_version),
        ("recurrent_contract_version", recurrent_contract_version),
        ("feature_version", feature_version),
        ("environment_version", environment_version),
        ("reward_version", reward_version),
        ("action_validity_version", action_validity_version),
    ):
        if not str(value).strip():
            raise BalancedWindowScheduleError(f"{label} cannot be empty")
    if normalization_scope != "symbol":
        raise BalancedWindowScheduleError("6E.2 retains per-symbol normalization")
    reward = dict(reward_configuration)
    action_validity = dict(action_validity_configuration)
    if reward.get("reward_version") != reward_version:
        raise BalancedWindowScheduleError(
            "reward configuration/version identity is inconsistent"
        )
    required_reward_semantics = {
        "reward_equation",
        "portfolio_growth_definition",
        "transaction_cost_treatment",
        "drawdown_increment_definition",
        "invalid_action_treatment",
    }
    if not required_reward_semantics.issubset(reward):
        raise BalancedWindowScheduleError(
            "frozen reward configuration must include mathematical semantics"
        )
    if action_validity.get("action_validity_version") != action_validity_version:
        raise BalancedWindowScheduleError(
            "action-validity configuration/version identity is inconsistent"
        )
    if action_validity.get("invalid_action_mode") != "penalty":
        raise BalancedWindowScheduleError(
            "the frozen 6E.2 protocol retains transparent penalty mode"
        )
    ppo = dict(ppo_configuration)
    template_model_seed = ppo.pop("seed", PREDECLARED_MODEL_SEEDS[0])
    if (
        isinstance(template_model_seed, bool)
        or not isinstance(template_model_seed, int)
        or template_model_seed not in PREDECLARED_MODEL_SEEDS
    ):
        raise BalancedWindowScheduleError(
            "PPO template seed must belong to the frozen model-seed set"
        )
    ppo["seed_source"] = "seed_policy.experiment_model_seeds"
    n_steps = ppo.get("n_steps")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise BalancedWindowScheduleError(
            "frozen PPO configuration requires a positive integer n_steps"
        )
    if schedule.expected_total_scheduled_transitions % n_steps:
        raise BalancedWindowScheduleError(
            "scheduled transitions must divide evenly into PPO rollouts"
        )
    declared_timesteps = ppo.get("total_timesteps")
    if declared_timesteps is not None and (
        isinstance(declared_timesteps, bool)
        or not isinstance(declared_timesteps, int)
        or declared_timesteps != schedule.expected_total_scheduled_transitions
    ):
        raise BalancedWindowScheduleError(
            "PPO total_timesteps must equal the exact scheduled transitions"
        )
    ppo["total_timesteps"] = schedule.expected_total_scheduled_transitions
    identity: dict[str, object] = {
        "experiment_version": FAIR_SECTOR_EXPERIMENT_VERSION,
        "sector_universe_hash": schedule.universe_hash,
        "taxonomy_version": taxonomy_version,
        "trainer_version": trainer_version,
        "recurrent_contract_version": recurrent_contract_version,
        "feature_version": feature_version,
        "environment_version": environment_version,
        "sampling_version": schedule.sampling_version,
        "reward_version": reward_version,
        "action_validity_version": action_validity_version,
        "reward_configuration": reward,
        "action_validity_configuration": action_validity,
        "window_transition_count": schedule.window_transition_count,
        "source_rows_per_window": schedule.source_rows_per_window,
        "rounds": schedule.rounds,
        "constituent_count": len(schedule.symbols),
        "constituent_symbols": list(schedule.symbols),
        "target_symbol": schedule.target_symbol,
        "target_excluded_from_pretraining": schedule.target_excluded_from_pretraining,
        "normalization_contributors": list(schedule.normalization_contributors),
        "expected_transitions_per_symbol": schedule.expected_transitions_per_symbol,
        "expected_total_scheduled_transitions": schedule.expected_total_scheduled_transitions,
        "normalization_scope": normalization_scope,
        "symbol_identity_observation": "excluded_observation_shape_remains_17",
        "schedule_digest": schedule.schedule_digest,
        "unique_history_coverage": [
            item.to_dict() for item in schedule.symbol_statistics
        ],
        "recurrent_architecture": dict(recurrent_architecture),
        "ppo_configuration": ppo,
        "seed_policy": {
            "data_schedule_seed": schedule.data_schedule_seed,
            "experiment_model_seeds": list(PREDECLARED_MODEL_SEEDS),
            "ppo_seed_source": "experiment_model_seeds",
            "ppo_template_seed_is_not_part_of_spec_identity": True,
            "window_locations_constant_across_model_seeds": True,
            "round_order_constant_across_model_seeds": True,
            "model_seed_controls": [
                "python_rng",
                "numpy_rng",
                "pytorch_rng",
                "ppo_initialization",
                "environment_stochasticity",
            ],
            "data_schedule_seed_controls": [
                "chronology_stratified_window_locations",
                "per_round_symbol_order",
            ],
        },
        "rollout_alignment": {
            "scheduled_environment_transitions_are_authoritative": True,
            "n_steps_must_divide_total_scheduled_transitions": True,
            "requested_total_must_not_be_relabelled_after_sb3_padding": True,
            "artificial_window_end": "truncated_with_terminal_observation_for_bootstrap",
            "natural_train_end": "terminated_without_truncation",
            "post_schedule_vecenv_reset": (
                "passive_valid_observation_with_zero_additional_exposure; "
                "subsequent_step_prohibited"
            ),
        },
        "state_isolation": {
            "episode_start_at_every_window": True,
            "recurrent_state_reset_at_every_window": True,
            "portfolio_state_reset_at_every_window": True,
            "rollout_boundary_alone_does_not_reset": True,
        },
        "controls": {
            "sector_balanced": "10240_transitions_per_constituent",
            "independent_target": "10240_target_transitions",
            "total_compute_matched_independent": "separately_predeclared_not_executed",
        },
        "evaluation_methodology": "complete_per_symbol_validation_episode_with_fresh_state",
        "test_sealing_rule": "TEST_metadata_only_never_loaded_or_evaluated",
        "execution_status": "specification_only_not_executed",
        "full_three_seed_run_requires_future_explicit_authorization": True,
    }
    spec_hash = canonical_payload_hash(identity)
    fingerprint = {
        "git_commit": git_commit,
        "experiment_spec_hash": spec_hash,
        "schedule_digest": schedule.schedule_digest,
        "sector_universe_hash": schedule.universe_hash,
        "taxonomy_version": taxonomy_version,
        "trainer_version": trainer_version,
        "recurrent_contract_version": recurrent_contract_version,
        "sampling_version": schedule.sampling_version,
        "reward_version": reward_version,
        "action_validity_version": action_validity_version,
        "data_schedule_seed": schedule.data_schedule_seed,
        "model_seed_set": list(PREDECLARED_MODEL_SEEDS),
    }
    return {
        **identity,
        "experiment_spec_hash": spec_hash,
        "reproducibility_fingerprint": fingerprint,
    }


__all__ = (
    "BalancedWindowRecord",
    "BalancedWindowSchedule",
    "BalancedWindowScheduleError",
    "BalancedWindowScheduleExhausted",
    "BalancedWindowTrainingEnv",
    "ChronologyQuartileCoverage",
    "COMMERCIAL_BANKS_CONSTITUENT_COUNT",
    "DEFAULT_ACTION_VALIDITY_VERSION",
    "DEFAULT_BALANCED_ROUNDS",
    "DEFAULT_DATA_SCHEDULE_SEED",
    "DEFAULT_REWARD_VERSION",
    "DEFAULT_WINDOW_SOURCE_ROWS",
    "DEFAULT_WINDOW_TRANSITIONS",
    "FAIR_SECTOR_EXPERIMENT_VERSION",
    "PREDECLARED_MODEL_SEEDS",
    "PREDECLARED_TOTAL_SECTOR_TRANSITIONS",
    "PREDECLARED_TRANSITIONS_PER_SYMBOL",
    "SECTOR_BALANCED_WINDOW_SAMPLING_VERSION",
    "SymbolWindowCoverage",
    "build_balanced_window_schedule",
    "build_predeclared_fair_experiment_spec",
    "canonical_payload_hash",
    "materialize_scheduled_window",
)
