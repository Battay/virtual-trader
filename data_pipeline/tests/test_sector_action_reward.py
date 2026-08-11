"""Deterministic offline tests for the 6E.2 action/reward contract."""

import math
import inspect

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from reinforcement_learning.environments import (
    MASKING_STATUS,
    SECTOR_ACTION_VALIDITY_VERSION,
    SECTOR_REWARD_VERSION,
    PortfolioState,
    RewardDiagnosticsAccumulator,
    SectorRewardConfig,
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
    action_mask,
    action_validity_metadata,
    calculate_reward_components,
    evaluate_action_validity,
    is_action_valid,
    valid_actions,
)
from reinforcement_learning.environments.action_validity import (
    finalize_action_outcome,
)


FEATURES = ("simple_return",)


def _prices() -> pd.DataFrame:
    opens = np.asarray([10.0, 10.0, 11.0, 12.0, 11.0, 11.5])
    closes = np.asarray([10.0, 10.5, 11.5, 11.0, 11.25, 12.0])
    return pd.DataFrame(
        {
            "symbol": "BANK",
            "date": pd.date_range("2024-01-01", periods=len(opens), freq="D"),
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": np.arange(1_000, 1_000 + len(opens)),
            "simple_return": np.linspace(-0.1, 0.1, len(opens)),
        }
    )


def _config(**values: object) -> SingleSymbolEnvConfig:
    return SingleSymbolEnvConfig(feature_columns=FEATURES, **values)


def _env(**values: object) -> SingleSymbolTradingEnv:
    return SingleSymbolTradingEnv(_prices(), _config(**values))


def test_canonical_action_validity_table_and_masks() -> None:
    assert SECTOR_ACTION_VALIDITY_VERSION == "sector_action_validity_v1"
    assert valid_actions(PortfolioState.FLAT) == (0, 1)
    assert valid_actions(PortfolioState.LONG) == (0, 2)
    assert action_mask("flat") == (True, True, False)
    assert action_mask("long") == (True, False, True)

    assert is_action_valid(0, "flat")
    assert is_action_valid(0, "long")
    assert is_action_valid(1, "flat")
    assert not is_action_valid(1, "long")
    assert not is_action_valid(2, "flat")
    assert is_action_valid(2, "long")

    flat_sell = evaluate_action_validity(2, "flat")
    long_buy = evaluate_action_validity(1, "long")
    assert not flat_sell.state_valid
    assert flat_sell.invalid_reason == "No shares are held to sell"
    assert not long_buy.state_valid
    assert "already held" in str(long_buy.invalid_reason)

    metadata = action_validity_metadata("penalty")
    assert metadata["flat_valid_actions"] == (0, 1)
    assert metadata["long_valid_actions"] == (0, 2)
    assert metadata["methodology_version"] == "sector_action_validity_penalty_v1"
    assert metadata["masking_status"] == "not_applicable"


def test_penalty_mode_invalid_actions_are_noops_with_explicit_metadata() -> None:
    env = _env(
        initial_cash=100.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        drawdown_penalty_weight=0.0,
    )
    env.reset()
    before = (env.cash, env.shares_held, env.number_of_trades)
    _, reward, _, _, info = env.step(2)

    assert (env.cash, env.shares_held, env.number_of_trades) == before
    assert reward == pytest.approx(-0.0001)
    assert info["selected_action"] == 2
    assert info["action_state_valid"] is False
    assert info["action_valid"] is False
    assert info["action_executed"] is False
    assert info["trade_executed"] is False
    assert info["shares_traded"] == 0
    assert info["invalid_action_reason"] == "No shares are held to sell"
    assert info["reward_breakdown"]["invalid_action_penalty"] == -0.0001
    assert info["reward_breakdown"]["total_reward"] == pytest.approx(reward)

    history = env.get_history().iloc[0]
    assert not history["action_valid"]
    assert not history["action_executed"]
    assert history["invalid_action_reason"] == "No shares are held to sell"
    assert history["reward_breakdown"]["portfolio_growth_reward"] == 0.0
    assert history["reward_breakdown"]["total_reward"] == pytest.approx(reward)


def test_long_buy_is_invalid_noop_but_hold_is_always_valid() -> None:
    env = _env(
        initial_cash=100.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        drawdown_penalty_weight=0.0,
    )
    env.reset()
    _, _, _, _, buy = env.step(1)
    shares_after_buy = env.shares_held
    trades_after_buy = env.number_of_trades
    assert buy["action_valid"] is True
    assert buy["trade_executed"] is True

    _, _, _, _, redundant = env.step(1)
    assert env.shares_held == shares_after_buy
    assert env.number_of_trades == trades_after_buy
    assert redundant["portfolio_state_before_action"] == "long"
    assert redundant["action_valid"] is False
    assert redundant["action_executed"] is False
    assert redundant["shares_traded"] == 0
    assert "redundant" in redundant["invalid_action_reason"]
    assert redundant["reward_breakdown"]["invalid_action_penalty"] == -0.0001

    _, _, _, _, hold = env.step(0)
    assert hold["portfolio_state_before_action"] == "long"
    assert hold["action_state_valid"] is True
    assert hold["action_valid"] is True
    assert hold["action_executed"] is True
    assert hold["trade_executed"] is False
    assert "invalid_action_reason" not in hold or hold["invalid_action_reason"] is None
    assert hold["reward_breakdown"]["invalid_action_penalty"] == 0.0


def test_flat_buy_affordability_is_execution_failure_not_invalid_action() -> None:
    env = _env(
        initial_cash=1.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        drawdown_penalty_weight=0.0,
    )
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert info["portfolio_state_before_action"] == "flat"
    assert info["action_state_valid"] is True
    assert info["action_valid"] is True
    assert info["action_executed"] is False
    assert info["trade_executed"] is False
    assert info["semantic_invalid_action_reason"] is None
    assert "Insufficient cash" in info["execution_failure_reason"]
    assert info["reward_breakdown"]["invalid_action_penalty"] == 0.0
    assert reward == 0.0


def test_mask_mode_fails_closed_instead_of_rewriting_recurrent_actions() -> None:
    assert MASKING_STATUS == "unsupported_or_deferred"
    config = _config(invalid_action_mode="mask")
    with pytest.raises(NotImplementedError, match="unsupported_or_deferred"):
        SingleSymbolTradingEnv(_prices(), config)
    with pytest.raises(ValueError, match="invalid_action_mode"):
        _config(invalid_action_mode="rewrite")


def test_installed_recurrent_ppo_has_no_verified_native_mask_interface() -> None:
    """Prevent a future no-op action_masks hook from being called 'masking'."""

    assert not issubclass(RecurrentActorCriticPolicy, MaskableActorCriticPolicy)
    assert RecurrentPPO is not MaskablePPO
    assert "action_masks" not in inspect.signature(RecurrentPPO.predict).parameters
    assert (
        "action_masks"
        not in inspect.signature(RecurrentActorCriticPolicy.forward).parameters
    )


def test_sector_reward_v1_metadata_defines_equation_and_semantics() -> None:
    config = SectorRewardConfig()
    metadata = config.to_metadata()
    assert SECTOR_REWARD_VERSION == "sector_reward_v1"
    assert metadata["reward_version"] == SECTOR_REWARD_VERSION
    assert "log(current_portfolio_value" in metadata["reward_equation"]
    assert "next-open execution" in metadata["portfolio_growth_definition"]
    assert "commission and slippage" in metadata["transaction_cost_treatment"]
    assert "positive increment only" in metadata["drawdown_increment_definition"]
    assert "no-ops in penalty mode" in metadata["invalid_action_treatment"]
    assert metadata["portfolio_growth_weight"] == 1.0
    assert metadata["transaction_cost_weight"] == 0.0
    assert metadata["drawdown_increment_weight"] == 0.1
    assert metadata["invalid_action_penalty"] == 0.0001


def test_reward_components_sum_exactly_and_invalid_penalty_is_isolated() -> None:
    config = SectorRewardConfig(
        transaction_cost_weight=0.25,
        drawdown_increment_weight=0.1,
        invalid_action_penalty=0.0001,
    )
    valid = calculate_reward_components(
        config=config,
        previous_portfolio_value=100.0,
        current_portfolio_value=105.0,
        transaction_cost=2.0,
        previous_drawdown=0.01,
        current_drawdown=0.04,
        action_invalid=False,
    )
    invalid = calculate_reward_components(
        config=config,
        previous_portfolio_value=100.0,
        current_portfolio_value=105.0,
        transaction_cost=2.0,
        previous_drawdown=0.01,
        current_drawdown=0.04,
        action_invalid=True,
    )

    for result in (valid, invalid):
        assert result.total_reward == math.fsum(
            (
                result.portfolio_growth_reward,
                result.transaction_cost_penalty,
                result.drawdown_penalty,
                result.invalid_action_penalty,
            )
        )
    assert valid.invalid_action_penalty == 0.0
    assert invalid.invalid_action_penalty == -0.0001
    assert invalid.total_reward == pytest.approx(valid.total_reward - 0.0001)
    assert invalid.portfolio_growth_reward == valid.portfolio_growth_reward
    assert invalid.transaction_cost_penalty == valid.transaction_cost_penalty
    assert invalid.drawdown_penalty == valid.drawdown_penalty


def test_deterministic_reward_counterfactual_paths_are_finite_and_additive() -> None:
    paths = {
        "stay_cash": [0, 0, 0, 0, 0],
        "flat_sell": [2, 2, 2, 2, 2],
        "buy_then_hold": [1, 0, 0, 0, 0],
        "buy_then_sell": [1, 2, 0, 0, 0],
    }
    totals: dict[str, float] = {}
    for name, actions in paths.items():
        env = _env(initial_cash=100.0, commission_rate=0.0, slippage_rate=0.0)
        env.reset(seed=42)
        transition_rewards = []
        for action in actions:
            _, reward, _, _, info = env.step(action)
            breakdown = info["reward_breakdown"]
            assert reward == math.fsum(
                (
                    breakdown["portfolio_growth_reward"],
                    breakdown["transaction_cost_penalty"],
                    breakdown["drawdown_penalty"],
                    breakdown["invalid_action_penalty"],
                )
            )
            assert reward == breakdown["total_reward"]
            assert math.isfinite(reward)
            transition_rewards.append(reward)
        totals[name] = math.fsum(transition_rewards)

    assert totals["stay_cash"] == 0.0
    assert totals["flat_sell"] == pytest.approx(-0.0005)
    assert totals["buy_then_hold"] != totals["stay_cash"]
    assert totals["buy_then_sell"] != totals["flat_sell"]


def test_reward_inputs_and_coefficients_reject_nan_or_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        SectorRewardConfig(drawdown_increment_weight=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        SectorRewardConfig(transaction_cost_weight=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        calculate_reward_components(
            config=SectorRewardConfig(),
            previous_portfolio_value=100.0,
            current_portfolio_value=float("nan"),
            transaction_cost=0.0,
            previous_drawdown=0.0,
            current_drawdown=0.0,
            action_invalid=False,
        )


def test_reward_diagnostics_aggregate_validity_and_components_by_symbol() -> None:
    accumulator = RewardDiagnosticsAccumulator()
    config = SectorRewardConfig()
    cases = (("AAA", 0, "flat"), ("AAA", 2, "flat"), ("BBB", 1, "flat"))
    for symbol, action, state in cases:
        decision = evaluate_action_validity(action, state)
        outcome = finalize_action_outcome(decision, trade_executed=action == 1)
        reward = calculate_reward_components(
            config=config,
            previous_portfolio_value=100.0,
            current_portfolio_value=100.0,
            transaction_cost=0.0,
            previous_drawdown=0.0,
            current_drawdown=0.0,
            action_invalid=not outcome.action_valid,
        )
        accumulator.update(symbol=symbol, reward=reward, action=outcome)

    summary = accumulator.summary()
    assert summary["reward_distribution"]["count"] == 3
    assert summary["action_validity_counts"] == {
        "valid_hold": 1,
        "valid_buy": 1,
        "valid_sell": 0,
        "invalid_buy": 0,
        "invalid_sell": 1,
        "invalid_action_rate": pytest.approx(1 / 3),
        "execution_failure_count": 0,
    }
    assert summary["cumulative_components"]["invalid_action_penalty"] == -0.0001
    assert summary["per_symbol"]["AAA"]["reward_distribution"]["count"] == 2
    assert summary["per_symbol"]["BBB"]["reward_distribution"]["count"] == 1

    emitted = RewardDiagnosticsAccumulator()
    environment = _env()
    environment.reset()
    _, _, _, _, info = environment.step(2)
    emitted.update_from_info(symbol="BANK", info=info)
    emitted_summary = emitted.summary()
    assert emitted_summary["reward_distribution"]["count"] == 1
    assert emitted_summary["action_validity_counts"]["invalid_sell"] == 1


def test_legacy_reward_component_contract_remains_unchanged() -> None:
    env = _env()
    _, reset_info = env.reset()
    _, _, _, _, step_info = env.step(0)
    expected = {
        "portfolio_growth",
        "transaction_cost_penalty",
        "drawdown_penalty",
        "invalid_action_penalty",
    }
    assert set(reset_info["reward_components"]) == expected
    assert set(step_info["reward_components"]) == expected
    assert set(step_info["reward_breakdown"]) == {
        "reward_version",
        "portfolio_growth_reward",
        "transaction_cost_penalty",
        "drawdown_penalty",
        "invalid_action_penalty",
        "total_reward",
    }
