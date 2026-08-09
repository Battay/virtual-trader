"""Offline validation-only PPO evaluation and baseline comparison tests."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from data_pipeline.src.config import (
    MODEL_REGISTRY_PATH,
    MODELS_DATA_DIR,
    SAVED_MODELS_DIR,
)
from feature_engineering.schemas import FEATURE_COLUMNS
from feature_engineering.splitting import chronological_split, persist_split_artifacts
from reinforcement_learning.data_contract import (
    DEFAULT_OBSERVATION_FEATURES,
    EXECUTION_ACCOUNTING_COLUMNS,
    RL_CONTRACT_FILENAME,
    RLDataContractError,
    RL_OBSERVATION_SCALER_FILENAME,
    load_rl_partition,
    scaled_observation_column,
)
from reinforcement_learning.environments import (
    SingleSymbolEnvConfig,
    SingleSymbolTradingEnv,
)
from reinforcement_learning.evaluation import comparison as comparison_module
from reinforcement_learning.evaluation import ppo_evaluator as evaluator_module
from reinforcement_learning.evaluation.comparison import (
    CandidateValidationCriteria,
    compare_candidate_on_validation,
    decide_candidate_validation,
)
from reinforcement_learning.evaluation.metrics import (
    TRADING_DAYS_PER_YEAR,
    calculate_episode_metrics,
)
from reinforcement_learning.evaluation.ppo_evaluator import (
    EXPECTED_HISTORY_COLUMNS,
    evaluate_ppo_validation,
    policy_parameter_hash,
)
from reinforcement_learning.integrity import sha256_file
from reinforcement_learning.training.config import PPOConfig
from reinforcement_learning.training.ppo_trainer import train_single_symbol
from reinforcement_learning.training import ppo_trainer as trainer_module


def _processed(rows: int = 40, symbol: str = "MCB") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    wave = 2.5 * np.sin(index / 2.5)
    opens = 100.0 + 0.2 * index + wave
    data: dict[str, object] = {
        "symbol": pd.Series([symbol] * rows, dtype="string"),
        "date": pd.date_range("2024-01-01", periods=rows),
        "open": opens,
        "high": opens + 2.0,
        "low": opens - 2.0,
        "close": opens + np.cos(index / 2.0),
        "volume": 10_000.0 + 100 * index,
    }
    for position, feature in enumerate(FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = np.sin(index / (position + 2)) + position * 0.01
    return pd.DataFrame(data)


def _persist_splits(root: Path, *, rows: int = 40):
    source = _processed(rows=rows)
    split = chronological_split(source, scope="symbol")
    artifacts = persist_split_artifacts(split, root / "symbols" / "MCB")
    return source, split, artifacts


def _tiny_config(seed: int = 42) -> PPOConfig:
    return replace(
        PPOConfig(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        total_timesteps=16,
        seed=seed,
    )


def _hash_files(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def trained_candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("ppo-validation")
    _, split, _ = _persist_splits(root)
    training = train_single_symbol(
        "MCB",
        config=_tiny_config(seed=17),
        splits_dir=root,
        smoke_test=True,
    )
    assert training.succeeded and training.model is not None
    return root, split, training


def _metric_history(returns: list[float]) -> pd.DataFrame:
    initial = 100.0
    values = initial * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    count = len(returns)
    shares_traded = np.zeros(count)
    shares_held = np.zeros(count)
    realized = np.zeros(count)
    if count >= 1:
        shares_traded[0] = 10
        shares_held[: min(2, count)] = 10
    if count >= 3:
        shares_traded[2] = -10
        realized[2:] = 2.0
    return pd.DataFrame(
        {
            "initial_portfolio_value": initial,
            "portfolio_value": values,
            "shares_traded": shares_traded,
            "transaction_cost": np.full(count, 0.5),
            "realized_profit_loss": realized,
            "unrealized_profit_loss": np.where(shares_held > 0, 1.0, 0.0),
            "shares_held": shares_held,
        }
    )


def test_evaluator_loads_validation_only_predicts_deterministically_and_is_immutable(
    trained_candidate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, split, training = trained_candidate
    model = training.model
    loader_calls: list[tuple[str, str, Path]] = []
    original_loader = load_rl_partition

    def guarded_loader(symbol: str, partition: str, *, splits_dir: Path):
        loader_calls.append((symbol, partition, Path(splits_dir)))
        if partition != "validation":
            raise AssertionError("train/test partitions are sealed during evaluation")
        return original_loader(symbol, partition, splits_dir=splits_dir)

    prediction_flags: list[bool] = []
    original_predict = model.predict

    def guarded_predict(observation, *args, deterministic=False, **kwargs):
        prediction_flags.append(deterministic)
        return original_predict(
            observation, *args, deterministic=deterministic, **kwargs
        )

    def forbidden_learning(*args, **kwargs):
        raise AssertionError("validation evaluation must never call learn")

    monkeypatch.setattr(evaluator_module, "load_rl_partition", guarded_loader)
    monkeypatch.setattr(model, "predict", guarded_predict)
    monkeypatch.setattr(model, "learn", forbidden_learning)
    artifact_root = root / "symbols" / "MCB"
    artifacts_before = _hash_files(artifact_root)
    parameter_before = policy_parameter_hash(model)
    timesteps_before = model.num_timesteps

    result = evaluate_ppo_validation(model, "MCB", seed=23, splits_dir=root)

    assert loader_calls == [("MCB", "validation", root)]
    assert prediction_flags == [True] * (len(split.validation) - 1)
    assert result.evaluation_partition == "validation"
    assert result.validation_rows == len(split.validation)
    assert result.model_parameters_unchanged
    assert result.parameter_hash_before == parameter_before
    assert policy_parameter_hash(model) == parameter_before
    assert result.model_timesteps_before == timesteps_before
    assert model.num_timesteps == timesteps_before
    contract_path = (artifact_root / RL_CONTRACT_FILENAME).resolve()
    scaler_path = (artifact_root / RL_OBSERVATION_SCALER_FILENAME).resolve()
    scaler_metadata_path = scaler_path.with_suffix(".json")
    assert result.source_rl_contract_path == str(contract_path)
    assert result.source_rl_contract_sha256 == sha256_file(contract_path)
    assert result.source_observation_scaler_path == str(scaler_path)
    assert result.source_observation_scaler_sha256 == sha256_file(scaler_path)
    assert result.source_observation_scaler_metadata_path == str(
        scaler_metadata_path
    )
    assert result.source_observation_scaler_metadata_sha256 == sha256_file(
        scaler_metadata_path
    )
    assert result.observation_features == DEFAULT_OBSERVATION_FEATURES
    history = result.strategy_result.history
    assert tuple(history.columns) == EXPECTED_HISTORY_COLUMNS
    assert len(history) == len(split.validation) - 1
    pdt.assert_series_equal(
        pd.to_datetime(history["observation_date"]).reset_index(drop=True),
        split.validation["date"].iloc[:-1].reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        pd.to_datetime(history["execution_date"]).reset_index(drop=True),
        split.validation["date"].iloc[1:].reset_index(drop=True),
        check_names=False,
    )
    assert _hash_files(artifact_root) == artifacts_before


def test_validation_preserves_real_execution_and_train_scaled_observations(
    trained_candidate,
) -> None:
    root, split, _ = trained_candidate
    loaded = load_rl_partition("MCB", "validation", splits_dir=root)
    raw = pd.read_csv(
        root / "symbols/MCB/validation.csv",
        dtype={"symbol": "string"},
        parse_dates=["date"],
    )
    artifact = pd.read_csv(
        root / "symbols/MCB/validation_rl.csv",
        dtype={"symbol": "string"},
        parse_dates=["date"],
    )
    for column in EXECUTION_ACCOUNTING_COLUMNS:
        np.testing.assert_array_equal(loaded.data[column], raw[column])
        np.testing.assert_allclose(raw[column], split.validation[column])
    for feature in DEFAULT_OBSERVATION_FEATURES:
        np.testing.assert_allclose(
            loaded.data[feature], artifact[scaled_observation_column(feature)]
        )
    assert loaded.contract["scaler_fit_partition"] == "train"
    environment = SingleSymbolTradingEnv(loaded.data)
    try:
        observation, _ = environment.reset(seed=42)
        expected = artifact.iloc[0][
            [scaled_observation_column(name) for name in DEFAULT_OBSERVATION_FEATURES]
        ].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(observation[: len(expected)], expected)
    finally:
        environment.close()


def test_loader_rejects_non_train_scaler_provenance(tmp_path: Path) -> None:
    _, _, artifacts = _persist_splits(tmp_path)
    contract_path = artifacts.rl_artifacts.contract_path
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["scaler_fit_partition"] = "validation"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RLDataContractError, match="fitted on the train"):
        load_rl_partition("MCB", "validation", splits_dir=tmp_path)


def test_comparison_is_apples_to_apples_and_reproducible(
    trained_candidate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, split, training = trained_candidate
    config = SingleSymbolEnvConfig(
        initial_cash=250_000.0,
        commission_rate=0.002,
        slippage_rate=0.001,
    )
    seen_configs: list[dict[str, object]] = []
    real_environment = SingleSymbolTradingEnv

    def recording_environment(data, environment_config):
        seen_configs.append(asdict(environment_config))
        return real_environment(data, environment_config)

    monkeypatch.setattr(
        evaluator_module, "SingleSymbolTradingEnv", recording_environment
    )
    monkeypatch.setattr(
        comparison_module, "SingleSymbolTradingEnv", recording_environment
    )
    first = compare_candidate_on_validation(
        training.model,
        "MCB",
        trainer_result=training,
        environment_config=config,
        deterministic_seed=31,
        random_seed=99,
        splits_dir=root,
    )
    second = compare_candidate_on_validation(
        training.model,
        "MCB",
        trainer_result=training,
        environment_config=config,
        deterministic_seed=31,
        random_seed=99,
        splits_dir=root,
    )

    assert len(seen_configs) == 8
    assert all(item == asdict(config) for item in seen_configs)
    assert first.validation_rows == len(split.validation)
    assert first.validation_start == split.validation["date"].min().date().isoformat()
    assert first.validation_end == split.validation["date"].max().date().isoformat()
    assert first.initial_cash == 250_000.0
    assert first.commission_rate == 0.002
    assert first.slippage_rate == 0.001
    assert first.ppo_model_unchanged
    assert first.ppo_training_metadata["ppo_config_version"] == training.ppo_config_version
    assert first.source_rl_contract_sha256 == training.source_rl_contract_sha256
    assert (
        first.source_observation_scaler_sha256
        == training.source_observation_scaler_sha256
    )
    assert (
        first.source_observation_scaler_metadata_sha256
        == training.source_observation_scaler_metadata_sha256
    )
    assert first.observation_features == training.observation_features
    strategies_first = (
        first.ppo,
        first.buy_and_hold,
        first.always_hold,
        first.random,
    )
    strategies_second = (
        second.ppo,
        second.buy_and_hold,
        second.always_hold,
        second.random,
    )
    reference_observation_dates = strategies_first[0].history["observation_date"]
    reference_execution_dates = strategies_first[0].history["execution_date"]
    for left, right in zip(strategies_first, strategies_second, strict=True):
        pdt.assert_frame_equal(left.history, right.history)
        assert left.metrics.keys() == right.metrics.keys()
        pdt.assert_series_equal(
            left.metrics["daily_returns"], right.metrics["daily_returns"]
        )
        assert left.history["observation_date"].equals(reference_observation_dates)
        assert left.history["execution_date"].equals(reference_execution_dates)
        assert left.metrics["initial_portfolio_value"] == 250_000.0
    assert first.buy_and_hold.history["action"].iloc[0] == 1
    assert first.buy_and_hold.history["action"].iloc[1:].eq(0).all()
    assert first.always_hold.history["action"].eq(0).all()
    assert first.always_hold.metrics["final_portfolio_value"] == 250_000.0
    pdt.assert_frame_equal(first.random.history, second.random.history)
    json.dumps(first.to_dict(include_history=True))


@pytest.mark.parametrize(
    ("field_name", "stale_value"),
    (
        ("environment_version", "stale_environment"),
        ("rl_contract_version", "stale_contract"),
        ("feature_version", "stale_features"),
    ),
)
def test_comparison_rejects_stale_training_provenance(
    trained_candidate,
    field_name: str,
    stale_value: str,
) -> None:
    root, _, training = trained_candidate
    stale_result = replace(training, **{field_name: stale_value})
    with pytest.raises(
        evaluator_module.ValidationEvaluationError,
        match=field_name,
    ):
        compare_candidate_on_validation(
            training.model,
            "MCB",
            trainer_result=stale_result,
            splits_dir=root,
        )


def test_comparison_rejects_same_version_artifact_drift(tmp_path: Path) -> None:
    _persist_splits(tmp_path)
    training = train_single_symbol(
        "MCB",
        config=_tiny_config(seed=37),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert training.succeeded and training.model is not None
    contract_path = (
        tmp_path / "symbols" / "MCB" / RL_CONTRACT_FILENAME
    )
    original = contract_path.read_text(encoding="utf-8")
    contract_path.write_text(original + "\n", encoding="utf-8")

    with pytest.raises(
        evaluator_module.ValidationEvaluationError,
        match="source_rl_contract_sha256",
    ):
        compare_candidate_on_validation(
            training.model,
            "MCB",
            trainer_result=training,
            splits_dir=tmp_path,
        )


def test_comparison_rejects_training_observation_order_drift(
    trained_candidate,
) -> None:
    root, _, training = trained_candidate
    stale_result = replace(
        training,
        observation_features=tuple(reversed(training.observation_features)),
    )

    with pytest.raises(
        evaluator_module.ValidationEvaluationError,
        match="observation_features",
    ):
        compare_candidate_on_validation(
            training.model,
            "MCB",
            trainer_result=stale_result,
            splits_dir=root,
        )


def test_evaluator_rejects_same_shape_wrong_feature_semantics(
    trained_candidate,
) -> None:
    root, _, training = trained_candidate
    wrong_order = replace(
        SingleSymbolEnvConfig(),
        feature_columns=tuple(reversed(DEFAULT_OBSERVATION_FEATURES)),
    )
    assert len(wrong_order.feature_columns) == len(DEFAULT_OBSERVATION_FEATURES)
    with pytest.raises(
        evaluator_module.ValidationEvaluationError,
        match="canonical RL contract order",
    ):
        evaluate_ppo_validation(
            training.model,
            "MCB",
            environment_config=wrong_order,
            splits_dir=root,
        )


def test_future_validation_feature_does_not_change_earlier_policy_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _processed(rows=60)
    split_a = chronological_split(source, scope="symbol")
    future_position = 3
    future_date = split_a.validation["date"].iloc[future_position]
    source_b = source.copy(deep=True)
    source_b.loc[
        source_b["date"].eq(future_date),
        DEFAULT_OBSERVATION_FEATURES[0],
    ] += 1_000.0
    split_b = chronological_split(source_b, scope="symbol")
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    persist_split_artifacts(split_a, root_a / "symbols" / "MCB")
    persist_split_artifacts(split_b, root_b / "symbols" / "MCB")
    training = train_single_symbol(
        "MCB",
        config=_tiny_config(seed=29),
        splits_dir=root_a,
        smoke_test=True,
    )
    assert training.succeeded and training.model is not None

    observed_a: list[np.ndarray] = []
    observed_b: list[np.ndarray] = []

    def record_hold(target: list[np.ndarray]):
        def predict(observation, *args, deterministic=False, **kwargs):
            del args, kwargs
            assert deterministic
            target.append(np.asarray(observation).copy())
            return np.asarray(0), None

        return predict

    monkeypatch.setattr(training.model, "predict", record_hold(observed_a))
    evaluate_ppo_validation(training.model, "MCB", seed=29, splits_dir=root_a)
    monkeypatch.setattr(training.model, "predict", record_hold(observed_b))
    evaluate_ppo_validation(training.model, "MCB", seed=29, splits_dir=root_b)

    np.testing.assert_array_equal(
        np.stack(observed_a[:future_position]),
        np.stack(observed_b[:future_position]),
    )
    assert not np.array_equal(
        observed_a[future_position],
        observed_b[future_position],
    )


def test_extended_metrics_match_known_return_path() -> None:
    returns = np.asarray([0.01, -0.02, 0.03, -0.01])
    history = _metric_history(returns.tolist())
    metrics = calculate_episode_metrics(
        history,
        minimum_annualization_observations=2,
    )
    expected_final = 100.0 * float(np.prod(1.0 + returns))
    daily_std = float(returns.std(ddof=0))
    downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(returns, 0)))))
    assert metrics["final_portfolio_value"] == pytest.approx(expected_final)
    assert metrics["total_return"] == pytest.approx(expected_final / 100.0 - 1)
    assert metrics["annualized_return"] == pytest.approx(
        (expected_final / 100.0) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
    )
    assert metrics["annualized_volatility"] == pytest.approx(
        daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert metrics["sharpe_ratio"] == pytest.approx(
        returns.mean() / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert metrics["sortino_ratio"] == pytest.approx(
        returns.mean() / downside_deviation * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert metrics["maximum_drawdown"] == pytest.approx(0.02)
    assert metrics["number_of_trades"] == 2
    assert metrics["total_transaction_costs"] == 2.0
    assert metrics["realized_profit_loss"] == 2.0
    assert metrics["final_unrealized_profit_loss"] == 0.0
    assert metrics["exposure_percentage"] == 50.0
    assert metrics["completed_trades"] == 1
    assert metrics["profitable_completed_trades"] == 1
    assert metrics["completed_trade_win_rate"] == 1.0


def test_metrics_handle_short_zero_downside_and_nonfinite_data() -> None:
    short = calculate_episode_metrics(_metric_history([0.01]))
    assert short["annualized_return"] is None
    assert short["annualized_volatility"] is None
    assert short["sharpe_ratio"] is None
    assert short["sortino_ratio"] is None
    positive = calculate_episode_metrics(_metric_history([0.01] * 20))
    assert positive["annualized_return"] is not None
    assert positive["sortino_ratio"] is None
    assert "zero_downside_deviation" in positive["metric_warnings"]
    invalid = _metric_history([0.01, -0.01])
    invalid.loc[1, "portfolio_value"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        calculate_episode_metrics(invalid)
    extreme = calculate_episode_metrics(_metric_history([1_000_000.0] * 20))
    assert extreme["annualized_return"] is None
    assert "non_finite_annualized_return" in extreme["metric_warnings"]
    with pytest.raises(ValueError, match="trading_days_per_year"):
        calculate_episode_metrics(pd.DataFrame(), trading_days_per_year=0)


def _candidate_metrics(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    drawdown: float,
) -> dict[str, float]:
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": drawdown,
    }


def test_candidate_criteria_pass_fail_insufficient_and_nonfinite() -> None:
    benchmark = _candidate_metrics(
        total_return=0.10, sharpe=0.8, sortino=1.0, drawdown=0.15
    )
    passing = _candidate_metrics(
        total_return=0.14, sharpe=1.0, sortino=1.2, drawdown=0.14
    )
    decision = decide_candidate_validation(
        passing, benchmark, validation_observations=200
    )
    assert decision.status == "validation_pass" and decision.passed

    positive_but_worse = _candidate_metrics(
        total_return=0.05, sharpe=0.9, sortino=1.1, drawdown=0.14
    )
    failed = decide_candidate_validation(
        positive_but_worse, benchmark, validation_observations=200
    )
    assert failed.status == "validation_fail" and not failed.passed
    assert any("return" in reason for reason in failed.reasons)

    drawdown = _candidate_metrics(
        total_return=0.14, sharpe=1.0, sortino=1.2, drawdown=0.50
    )
    assert decide_candidate_validation(
        drawdown, benchmark, validation_observations=200
    ).status == "validation_fail"
    assert decide_candidate_validation(
        passing, benchmark, validation_observations=20
    ).status == "insufficient_validation_data"
    nonfinite = dict(passing, sharpe_ratio=np.inf)
    assert decide_candidate_validation(
        nonfinite, benchmark, validation_observations=200
    ).status == "evaluation_error"
    assert decide_candidate_validation(
        passing,
        benchmark,
        validation_observations=200,
        evaluation_error="predict failed",
    ).status == "evaluation_error"
    with pytest.raises(ValueError, match="at least 2"):
        CandidateValidationCriteria(minimum_validation_observations=1)


def test_tiny_train_to_validation_workflow_is_sealed_and_write_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, split, _ = _persist_splits(tmp_path)
    production_before = {
        "registry": Path(MODEL_REGISTRY_PATH).read_bytes(),
        "saved_models": _hash_files(Path(SAVED_MODELS_DIR)),
        "model_data": _hash_files(Path(MODELS_DATA_DIR)),
    }
    fixture_before = _hash_files(tmp_path / "symbols" / "MCB")
    calls: list[str] = []
    original_loader = load_rl_partition

    def traced_loader(symbol: str, partition: str, *, splits_dir: Path):
        calls.append(partition)
        if partition == "test":
            raise AssertionError("test partition must remain sealed")
        return original_loader(symbol, partition, splits_dir=splits_dir)

    monkeypatch.setattr(trainer_module, "load_rl_partition", traced_loader)
    monkeypatch.setattr(evaluator_module, "load_rl_partition", traced_loader)
    training = train_single_symbol(
        "MCB",
        config=_tiny_config(seed=5),
        splits_dir=tmp_path,
        smoke_test=True,
    )
    assert training.succeeded and training.model is not None
    result = compare_candidate_on_validation(
        training.model,
        "MCB",
        trainer_result=training,
        deterministic_seed=5,
        random_seed=7,
        splits_dir=tmp_path,
    )
    assert calls == ["train", "validation"]
    assert result.validation_rows == len(split.validation)
    assert result.ppo_model_unchanged
    assert all(
        strategy.metrics["final_portfolio_value"] is not None
        for strategy in (
            result.ppo,
            result.buy_and_hold,
            result.always_hold,
            result.random,
        )
    )
    assert _hash_files(tmp_path / "symbols" / "MCB") == fixture_before
    assert Path(MODEL_REGISTRY_PATH).read_bytes() == production_before["registry"]
    assert _hash_files(Path(SAVED_MODELS_DIR)) == production_before["saved_models"]
    assert _hash_files(Path(MODELS_DATA_DIR)) == production_before["model_data"]
