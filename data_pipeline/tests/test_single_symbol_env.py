"""Deterministic offline coverage for Gymnasium environment v1."""

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from reinforcement_learning.environments import (
    ENVIRONMENT_VERSION,
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
    action_name,
)
from reinforcement_learning.environments.validation import (
    EnvironmentDataError,
    environment_readiness_for_path,
    prepare_single_symbol_data,
    validate_environment,
)
from reinforcement_learning.evaluation import (
    AlwaysHoldPolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    calculate_episode_metrics,
    run_baseline,
)


FEATURES = ("simple_return", "rsi_14")


def _data(symbol: str = "786", rows: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    opens = np.arange(10.0, 10.0 + rows)
    closes = opens + 1.0
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": opens,
            "high": closes + 1,
            "low": opens - 1,
            "close": closes,
            "volume": np.arange(1_000, 1_000 + rows),
            "simple_return": np.linspace(-0.2, 0.2, rows),
            "rsi_14": np.linspace(-1.0, 1.0, rows),
        }
    )


def _config(**values) -> SingleSymbolEnvConfig:
    return SingleSymbolEnvConfig(feature_columns=FEATURES, **values)


def _env(**config_values) -> SingleSymbolTradingEnv:
    return SingleSymbolTradingEnv(_data(), _config(**config_values))


def test_one_symbol_validation_preserves_numeric_symbol_and_sorts() -> None:
    source = _data().sort_values("date", ascending=False).reset_index(drop=True)
    prepared = prepare_single_symbol_data(source, feature_columns=FEATURES)

    assert prepared["date"].is_monotonic_increasing
    assert str(prepared["symbol"].iloc[0]) == "786"
    with pytest.raises(EnvironmentDataError, match="exactly one symbol"):
        prepare_single_symbol_data(
            pd.concat([source, _data("ABC")], ignore_index=True),
            feature_columns=FEATURES,
        )


def test_duplicate_dates_and_missing_or_invalid_values_are_rejected() -> None:
    duplicate = _data()
    duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
    with pytest.raises(EnvironmentDataError, match="duplicate"):
        prepare_single_symbol_data(duplicate, feature_columns=FEATURES)
    invalid = _data()
    invalid.loc[0, "rsi_14"] = np.nan
    with pytest.raises(EnvironmentDataError, match="finite"):
        prepare_single_symbol_data(invalid, feature_columns=FEATURES)


def test_source_dataframe_is_not_mutated() -> None:
    source = _data().sort_values("date", ascending=False).reset_index(drop=True)
    original = source.copy(deep=True)
    SingleSymbolTradingEnv(source, _config())
    pdt.assert_frame_equal(source, original)


def test_action_mapping_and_validation() -> None:
    assert [action_name(value) for value in range(3)] == ["Hold", "Buy", "Sell"]
    with pytest.raises(ValueError):
        action_name(3)
    with pytest.raises(ValueError):
        _env().step(3)


def test_next_open_execution_is_look_ahead_safe() -> None:
    env = _env(initial_cash=100, commission_rate=0, slippage_rate=0)
    observation, _ = env.reset()
    assert observation[0] == pytest.approx(_data().iloc[0]["simple_return"])

    _, _, _, _, info = env.step(1)

    assert info["date"] == _data().iloc[0]["date"]
    assert info["next_date"] == _data().iloc[1]["date"]
    assert info["execution_price"] == 11.0
    assert info["shares_traded"] == 9


def test_buy_maximum_whole_share_accounting_commission_and_slippage() -> None:
    env = _env(initial_cash=100, commission_rate=0.1, slippage_rate=0.01)
    env.reset()
    _, reward, _, _, info = env.step(1)

    assert info["execution_price"] == pytest.approx(11.11)
    assert info["shares_traded"] == 8
    assert env.cash == pytest.approx(2.232)
    assert env.average_entry_price == pytest.approx(12.221)
    assert env.current_position_value == pytest.approx(96.0)
    assert env.total_portfolio_value == pytest.approx(env.cash + 96.0)
    assert env.unrealized_profit_loss == pytest.approx(-1.768)
    assert info["transaction_cost"] == pytest.approx(9.768)
    assert env.total_transaction_costs == pytest.approx(9.768)
    assert math.isfinite(reward)


def test_sell_accounting_realized_pnl_and_complete_liquidation() -> None:
    env = _env(initial_cash=100, commission_rate=0.1, slippage_rate=0.01)
    env.reset()
    env.step(1)
    _, _, _, _, info = env.step(2)

    assert info["execution_price"] == pytest.approx(11.88)
    assert info["shares_traded"] == -8
    assert env.shares_held == 0
    assert env.average_entry_price == 0
    assert env.current_position_value == 0
    assert env.unrealized_profit_loss == 0
    expected_net = 8 * 11.88 * 0.9
    assert env.realized_profit_loss == pytest.approx(expected_net - 97.768)
    assert env.cash >= 0


def test_hold_accounting_makes_no_transaction() -> None:
    env = _env(initial_cash=100)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert env.cash == 100
    assert env.shares_held == 0
    assert info["transaction_cost"] == 0
    assert env.number_of_trades == 0


def test_invalid_buy_and_sell_are_reported_without_negative_balances() -> None:
    buy_env = _env(initial_cash=1)
    buy_env.reset()
    _, _, _, _, buy_info = buy_env.step(1)
    assert "Insufficient cash" in buy_info["invalid_action_reason"]
    assert buy_env.cash == 1 and buy_env.shares_held == 0

    sell_env = _env()
    sell_env.reset()
    _, _, _, _, sell_info = sell_env.step(2)
    assert "No shares" in sell_info["invalid_action_reason"]
    assert sell_env.cash >= 0 and sell_env.shares_held == 0


def test_drawdown_peak_and_portfolio_identity() -> None:
    falling = _data()
    falling["close"] = [11, 12, 8, 7, 6, 5]
    falling["low"] = 4
    env = SingleSymbolTradingEnv(
        falling,
        _config(initial_cash=100, commission_rate=0, slippage_rate=0),
    )
    env.reset()
    env.step(1)
    env.step(0)

    assert env.current_drawdown > 0
    assert env.total_portfolio_value == pytest.approx(
        env.cash + env.current_position_value
    )
    assert env.peak_portfolio_value >= env.total_portfolio_value


def test_observation_contract_names_shape_dtype_and_finiteness() -> None:
    env = _env()
    observation, _ = env.reset()
    assert env.observation_feature_names == (
        *FEATURES,
        "portfolio_cash_ratio",
        "portfolio_position_value_ratio",
        "portfolio_position_indicator",
        "portfolio_unrealized_return_ratio",
        "portfolio_current_drawdown",
    )
    assert observation.shape == (len(FEATURES) + 5,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert env.observation_space.contains(observation)


def test_reset_restores_initial_state_and_script_is_deterministic() -> None:
    env = _env(initial_cash=100, commission_rate=0.01, slippage_rate=0.01)
    actions = [1, 0, 2, 0, 0]
    histories = []
    for _ in range(2):
        observation, _ = env.reset(seed=42)
        assert env.cash == 100 and env.shares_held == 0 and env.current_step == 0
        assert observation[-5] == 1.0
        for action in actions:
            env.step(action)
        histories.append(env.get_history())
    pdt.assert_frame_equal(histories[0], histories[1])


def test_history_copy_is_isolated() -> None:
    env = _env()
    env.reset()
    env.step(0)
    history = env.get_history()
    history.loc[0, "cash"] = -1
    assert env.get_history().loc[0, "cash"] >= 0


def test_reward_components_and_no_future_reward_lookahead() -> None:
    first = _data()
    second = _data()
    second.loc[2:, ["open", "high", "low", "close"]] *= 100
    env_a = SingleSymbolTradingEnv(first, _config(initial_cash=100))
    env_b = SingleSymbolTradingEnv(second, _config(initial_cash=100))
    env_a.reset()
    env_b.reset()
    _, reward_a, _, _, info_a = env_a.step(1)
    _, reward_b, _, _, info_b = env_b.step(1)

    assert reward_a == pytest.approx(reward_b)
    assert info_a["reward_components"] == info_b["reward_components"]
    assert set(info_a["reward_components"]) == {
        "portfolio_growth",
        "transaction_cost_penalty",
        "drawdown_penalty",
        "invalid_action_penalty",
    }


def test_natural_termination_and_optional_truncation() -> None:
    env = _env()
    env.reset()
    for step in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is (step == 4)
        assert truncated is False
    with pytest.raises(RuntimeError):
        env.step(0)

    short = _env(max_episode_steps=2)
    short.reset()
    short.step(0)
    _, _, terminated, truncated, _ = short.step(0)
    assert terminated is False and truncated is True


def test_baseline_policies_and_fixed_seed_random() -> None:
    hold = run_baseline(_env(), AlwaysHoldPolicy(), seed=7)
    buy_hold = run_baseline(_env(), BuyAndHoldPolicy(), seed=7)
    random_a = run_baseline(_env(), RandomPolicy(seed=99), seed=99)
    random_b = run_baseline(_env(), RandomPolicy(seed=99), seed=99)

    assert hold.history["action"].eq(0).all()
    assert hold.metrics["number_of_trades"] == 0
    assert buy_hold.history["action"].iloc[0] == 1
    assert buy_hold.history["action"].iloc[1:].eq(0).all()
    pdt.assert_frame_equal(random_a.history, random_b.history)


def test_episode_evaluation_metrics_and_short_sample_safety() -> None:
    result = run_baseline(_env(initial_cash=100), BuyAndHoldPolicy())
    metrics = result.metrics
    assert metrics["initial_portfolio_value"] == 100
    assert metrics["final_portfolio_value"] == pytest.approx(
        result.history["portfolio_value"].iloc[-1]
    )
    assert metrics["total_return"] == pytest.approx(
        metrics["final_portfolio_value"] / 100 - 1
    )
    assert metrics["maximum_drawdown"] >= 0
    assert len(metrics["daily_returns"]) == len(result.history)

    one = calculate_episode_metrics(result.history.head(1))
    assert one["sharpe_ratio"] is None
    assert one["annualized_volatility"] is None


def test_gymnasium_environment_checker() -> None:
    result = validate_environment(_env())
    assert result.environment_version == ENVIRONMENT_VERSION
    assert result.valid, result.errors
    assert result.observation_shape == (len(FEATURES) + 5,)


def test_readiness_helper_handles_missing_and_insufficient_files(tmp_path) -> None:
    missing = environment_readiness_for_path(tmp_path / "missing.csv")
    assert missing.status == "Not Implemented"
    path = tmp_path / "tiny.csv"
    _data(rows=1).to_csv(path, index=False)
    tiny = environment_readiness_for_path(path, minimum_rows=2)
    assert tiny.status == "Validation Failed"


def test_configuration_version_and_feature_safety() -> None:
    assert _config().environment_version == ENVIRONMENT_VERSION
    with pytest.raises(ValueError, match="symbol and date"):
        SingleSymbolEnvConfig(feature_columns=("symbol",))
