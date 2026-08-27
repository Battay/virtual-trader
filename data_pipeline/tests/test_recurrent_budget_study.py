"""Offline protocol tests for the bounded recurrent budget study."""

from __future__ import annotations

import pandas as pd
import pytest

from reinforcement_learning.training.recurrent_budget_study import (
    BUDGET_STUDY_VERSION,
    BudgetStudyManifest,
    STUDY_BUDGETS,
    STUDY_RUN_COUNT,
    STUDY_SEEDS,
    select_budget,
    select_representative_symbols,
    study_schedule,
    main,
)


def _descriptors() -> pd.DataFrame:
    rows = []
    for index in range(30):
        rows.append(
            {
                "symbol": f"S{index:02d}",
                "train_rows": 130 + index * 50,
                "train_start": "2017-01-01",
                "train_end": "2023-01-01",
                "active_span_coverage": 0.55 + index / 100,
                "median_volume": 1_000 + 10_000 * index,
                "average_volume": 2_000 + 12_000 * index,
                "zero_volume_ratio": (index % 3) / 100,
                "zero_ohl_ratio": (index % 4) / 100,
                "non_positive_close_rows": 0,
                "validation_available": True,
                "recurrent_contract_version": "rl_recurrent_partition_v1",
                "feature_version": "feature_v1",
                "environment_version": "single_symbol_env_v1",
            }
        )
    return pd.DataFrame(rows)


def _manifest(symbols: tuple[str, ...]) -> BudgetStudyManifest:
    return BudgetStudyManifest(
        study_version=BUDGET_STUDY_VERSION,
        selection_version="train_regime_archetypes_v1",
        decision_version="smallest_technically_mature_v1",
        universe_version="current_common_equity_universe_v1",
        universe_hash="a" * 64,
        source_inventory_hash="b" * 64,
        symbols=symbols,
        symbol_descriptors=tuple({"symbol": symbol} for symbol in symbols),
        seeds=STUDY_SEEDS,
        budgets=STUDY_BUDGETS,
        requested_device="cpu",
        config_except_seed_budget={"algorithm": "RecurrentPPO"},
        run_count=STUDY_RUN_COUNT,
    )


def test_representative_selection_is_deterministic_train_only_and_six_regimes() -> None:
    source = _descriptors()
    before = source.copy(deep=True)

    first = select_representative_symbols(source)
    second = select_representative_symbols(source.sample(frac=1, random_state=9))

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(source, before)
    assert len(first) == 6
    assert first["symbol"].is_unique
    assert first["selection_regime"].nunique() == 6
    assert not any("validation" in column.lower() and column != "validation_available" for column in first.columns)


def test_schedule_has_exact_fixed_54_runs_and_never_mentions_test() -> None:
    manifest = _manifest(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
    first = study_schedule(manifest)
    second = study_schedule(manifest)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 54
    assert first["run_id"].is_unique
    assert set(first["seed"]) == {42, 123, 2026}
    assert set(first["requested_timesteps"]) == {50_000, 100_000, 250_000}
    assert set(first["requested_device"]) == {"cpu"}
    assert not first["test_partition_loaded"].any()


def _results(*, ev_50: float, ev_100: float, ev_250: float) -> pd.DataFrame:
    rows = []
    values = {50_000: ev_50, 100_000: ev_100, 250_000: ev_250}
    for symbol in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"):
        for seed in STUDY_SEEDS:
            for budget in STUDY_BUDGETS:
                rows.append(
                    {
                        "symbol": symbol,
                        "seed": seed,
                        "requested_timesteps": budget,
                        "status": "completed",
                        "validation_status": "completed",
                        "approximate_kl": 0.01,
                        "clip_fraction": 0.10,
                        "explained_variance": values[budget],
                        "value_loss": 10.0,
                        "recurrent_continuity_verified": True,
                        # Deliberately enormous profit differences: decision must
                        # remain governed by technical diagnostics only.
                        "validation_return": budget * (seed + 1),
                    }
                )
    return pd.DataFrame(rows)


def test_smallest_mature_budget_rule_ignores_validation_profit() -> None:
    decision = select_budget(_results(ev_50=0.20, ev_100=0.21, ev_250=0.22))

    assert decision["decision"] == "FREEZE_BUDGET_50000"
    assert decision["validation_profit_used"] is False


def test_rule_selects_100k_only_when_50k_not_mature_and_250k_not_material() -> None:
    decision = select_budget(_results(ev_50=-0.10, ev_100=0.20, ev_250=0.21))

    assert decision["decision"] == "FREEZE_BUDGET_100000"


def test_too_many_failures_block_budget_selection() -> None:
    results = _results(ev_50=0.2, ev_100=0.2, ev_250=0.2)
    mask = results["seed"].isin((42, 123))
    results.loc[mask, "status"] = "failed"
    results.loc[mask, "validation_status"] = "not_run"

    assert select_budget(results)["decision"] == "BLOCKED_BUDGET_SELECTION"


def test_manifest_rejects_non_cpu_or_test_access() -> None:
    base = dict(
        study_version=BUDGET_STUDY_VERSION,
        selection_version="train_regime_archetypes_v1",
        decision_version="smallest_technically_mature_v1",
        universe_version="v1",
        universe_hash="a" * 64,
        source_inventory_hash="b" * 64,
        symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
        symbol_descriptors=tuple(),
        seeds=STUDY_SEEDS,
        budgets=STUDY_BUDGETS,
        config_except_seed_budget={},
        run_count=54,
    )
    with pytest.raises(ValueError, match="CPU-only"):
        BudgetStudyManifest(**base, requested_device="mps")
    with pytest.raises(ValueError, match="TEST"):
        BudgetStudyManifest(**base, requested_device="cpu", test_partition_loaded=True)


def test_worker_cli_does_not_require_parent_output_directory(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "recurrent_budget_study",
            "--worker",
            "--symbol",
            "AAA",
            "--seed",
            "1",
            "--budget",
            "50000",
        ],
    )

    # Invalid seed is rejected by the worker protocol, proving parsing reached
    # worker validation without a parent-only --output-directory argument.
    assert main() == 2
