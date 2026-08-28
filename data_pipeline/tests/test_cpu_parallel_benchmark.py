"""Contract tests for the bounded recurrent CPU concurrency benchmark."""

from __future__ import annotations

import pandas as pd
import pytest

from reinforcement_learning.training.cpu_parallel_benchmark import (
    CPU_PARALLEL_BENCHMARK_VERSION,
    CPU_PARALLEL_SELECTION_POLICY,
    CPUParallelBenchmarkContract,
    recommend_worker_count,
    select_representative_symbols,
)
from reinforcement_learning.training.recurrent_orchestrator import (
    ELIGIBLE_TRAINABLE,
    RecurrentUniverseDiscovery,
)


def _discovery() -> RecurrentUniverseDiscovery:
    return RecurrentUniverseDiscovery(
        records=pd.DataFrame(
            {
                "symbol": ["E", "A", "D", "B", "C", "NO_VALIDATION"],
                "category": [ELIGIBLE_TRAINABLE] * 6,
                "train_rows": [500, 100, 400, 200, 300, 10_000],
                "validation_available": [True, True, True, True, True, False],
            }
        ),
        universe_version="current_common_equity_universe_v1",
        universe_hash="a" * 64,
        identity_count=6,
        category_counts={ELIGIBLE_TRAINABLE: 6},
        source_inventory_hash="b" * 64,
    )


def _contract() -> CPUParallelBenchmarkContract:
    return CPUParallelBenchmarkContract(
        benchmark_version=CPU_PARALLEL_BENCHMARK_VERSION,
        symbols=("A", "C", "E"),
        symbol_selection_policy=CPU_PARALLEL_SELECTION_POLICY,
        worker_candidates=(1, 2, 4),
        thread_policy=((1, 8), (2, 4), (4, 2)),
        requested_timesteps_per_symbol=100_000,
        seed=42,
    )


def test_representative_symbol_selection_is_deterministic_and_validation_capable() -> None:
    first = select_representative_symbols(_discovery(), count=3)
    second = select_representative_symbols(_discovery(), count=3)

    assert first == second == tuple(sorted(first))
    assert len(first) == len(set(first)) == 3
    assert "NO_VALIDATION" not in first


def test_benchmark_contract_fingerprint_is_deterministic_and_test_sealed() -> None:
    first = _contract()
    second = _contract()

    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["test_partition_loaded"] is False
    assert first.to_dict()["requested_device"] == "cpu"
    with pytest.raises(ValueError, match="TEST"):
        CPUParallelBenchmarkContract(
            **{**first.__dict__, "test_partition_loaded": True}
        )


def test_worker_recommendation_prefers_safe_throughput_with_headroom() -> None:
    candidates = [
        {"workers": 1, "safe": True, "aggregate_steps_per_second": 100.0, "agents_per_hour": 10.0},
        {"workers": 2, "safe": True, "aggregate_steps_per_second": 170.0, "agents_per_hour": 17.0},
        {"workers": 4, "safe": True, "aggregate_steps_per_second": 175.0, "agents_per_hour": 17.5},
    ]

    assert recommend_worker_count(candidates, full_budget=True)["worker_count"] == 2
    smoke = recommend_worker_count(candidates, full_budget=False)
    assert smoke["worker_count"] is None
    assert smoke["status"] == "qualification_only_not_permanent_selection"


def test_worker_recommendation_keeps_one_without_material_parallel_gain() -> None:
    candidates = [
        {"workers": 1, "safe": True, "aggregate_steps_per_second": 100.0, "agents_per_hour": 10.0},
        {"workers": 2, "safe": True, "aggregate_steps_per_second": 108.0, "agents_per_hour": 10.8},
        {"workers": 4, "safe": False, "aggregate_steps_per_second": 150.0, "agents_per_hour": 15.0},
    ]

    result = recommend_worker_count(candidates, full_budget=True)
    assert result["worker_count"] == 1


def test_thread_policy_must_cover_each_candidate() -> None:
    with pytest.raises(ValueError, match="thread policy"):
        CPUParallelBenchmarkContract(
            benchmark_version=CPU_PARALLEL_BENCHMARK_VERSION,
            symbols=("A",),
            symbol_selection_policy=CPU_PARALLEL_SELECTION_POLICY,
            worker_candidates=(1, 2),
            thread_policy=((1, 1),),
            requested_timesteps_per_symbol=512,
            seed=42,
        )
