# Milestone 5B-2: PPO Validation Evaluator and Baseline Comparison

## Objective

Milestone 5B-2 adds deterministic, validation-only evaluation for one already
trained in-memory PPO candidate. It compares that candidate with Buy and Hold,
Always Hold, and a fixed-seed Random policy under identical simulation inputs.
It does not evaluate the test partition, tune hyperparameters, persist a model,
write the model registry, or promote a candidate.

The implementation uses Stable-Baselines3 2.9.0, PyTorch 2.13.0, Gymnasium
1.3.0, `rl_partition_v1`, `single_symbol_env_v1`, and the existing
`ppo_single_symbol_v1` trainer.

## Training, validation, and sealed test separation

The lifecycle is deliberately split:

1. PPO gradient updates use only `train_rl.csv`, loaded by the canonical RL
   partition loader.
2. Candidate scoring uses only `validation_rl.csv`, also loaded by the canonical
   loader. Validation is never passed to `learn()`.
3. `test_rl.csv` remains sealed for a later final evaluation milestone. Neither
   the evaluator nor the comparison orchestration contains a test-partition
   load path.

Validation informs a conservative candidate decision; it is not a final
out-of-sample performance claim. Keeping test sealed prevents iterative model
or threshold choices from leaking information from the final holdout.

## Architecture

- `reinforcement_learning/evaluation/ppo_evaluator.py` loads the canonical
  validation context, runs deterministic PPO inference through the complete
  episode, validates the history boundaries, and verifies model integrity.
- `reinforcement_learning/evaluation/comparison.py` runs the PPO candidate and
  all three baselines on the same loaded validation frame and environment
  configuration. It also contains the explicit candidate criteria and the safe
  single-symbol developer CLI.
- `reinforcement_learning/evaluation/results.py` defines immutable structured
  strategy, PPO, comparison, and candidate-decision results.
- `reinforcement_learning/evaluation/metrics.py` derives the common performance
  and accounting metrics from environment history.
- `reinforcement_learning/evaluation/baselines.py` remains the shared baseline
  policy and episode runner.

The evaluator consumes `env.get_history()` and requires exactly one transition
for each adjacent pair of validation rows. It records observation date,
execution date, action, execution price, shares traded, transaction cost, cash,
shares held, portfolio value, realized and unrealized P&L, drawdown, and reward.
There is no synthetic-data fallback.

## Deterministic inference and integrity checks

PPO evaluation calls `model.predict(observation, deterministic=True)` inside a
no-gradient context. It never calls `learn()`. For a fixed model, validation
artifact, environment configuration, and seed, the resulting history and
metrics are reproducible.

Before and after evaluation, the evaluator hashes the complete policy state
dictionary and records `model.num_timesteps`. A successful result requires both
to remain unchanged. The policy's original training/evaluation mode is restored
after the episode.

When trainer metadata is supplied, its symbol, in-memory model identity,
environment version, RL contract version, feature version, and observation
shape must match the validation model and artifacts. A same-shape but reordered
observation feature list is rejected because shape alone cannot establish
semantic compatibility.

## Baseline methodology

All four strategies receive the exact same:

- validation DataFrame and date boundaries;
- initial cash;
- commission and slippage assumptions;
- reward and environment configuration; and
- complete episode length.

Buy and Hold buys on the first available environment transition and then holds.
Always Hold remains in cash for the full episode. Random selects uniformly from
Hold, Buy, and Sell with an isolated, explicit fixed seed. Each strategy uses a
fresh environment instance, so portfolio state cannot leak between runs.

## Metric definitions

Daily returns are transition returns from the path consisting of initial
portfolio value followed by every post-step portfolio value.

- Total return: `final / initial - 1`.
- Annualized return: geometric annualization with 252 trading days. It is only
  reported for at least 20 transitions and is returned as unavailable, with a
  warning, if it would be non-finite.
- Annualized volatility: population standard deviation of daily returns times
  the square root of 252, requiring at least two transitions.
- Sharpe ratio: mean daily excess return divided by its population standard
  deviation, annualized by the square root of 252. The annual risk-free rate is
  explicitly 0% for this milestone.
- Sortino ratio: mean daily excess return divided by the root mean square of
  negative excess returns, annualized by the square root of 252.
- Maximum drawdown: recomputed from the initial-plus-post-step portfolio value
  path and its running peak.
- Number of trades: transitions with a non-zero executed share quantity.
- Transaction costs: sum of the environment's commission and slippage costs.
- Realized P&L: final cumulative realized P&L from environment accounting.
- Final unrealized P&L: the final open-position unrealized amount.
- Exposure: percentage of post-transition observations with shares held.
- Completed and profitable completed trades: executed sells, with profitability
  derived from a positive change in cumulative realized P&L. This is robust for
  the environment's all-in/all-out position semantics.

Undefined ratios are represented by `None`, never infinity or a fabricated
number, and are accompanied by metric warnings. Examples include zero return
volatility for Sharpe and zero downside deviation for Sortino.

## Candidate-validation foundation

Criteria version `ppo_validation_criteria_v1` is explicit and configurable.
The conservative defaults are:

| Criterion | Default |
|---|---:|
| Minimum validation observations | 126 |
| Minimum PPO return advantage over Buy and Hold | 0.00 |
| Minimum absolute PPO Sharpe | 0.00 |
| Minimum PPO Sharpe advantage over Buy and Hold | 0.00 |
| Minimum absolute PPO Sortino | 0.00 |
| Maximum PPO drawdown | 30% |
| Maximum drawdown disadvantage versus Buy and Hold | 2 percentage points |

Possible decisions are `validation_pass`, `validation_fail`,
`insufficient_validation_data`, and `evaluation_error`. Positive PPO return by
itself cannot pass. Missing or non-finite required PPO/Buy-and-Hold metrics are
handled conservatively as `evaluation_error`; they are not substituted with
zero. Evaluation exceptions remain explicit exceptions at the orchestration
boundary and can be mapped to the helper's `evaluation_error` input by a future
caller. These thresholds were not optimized against the sealed test set, and a
validation pass would still not promote a model.

## Leakage safeguards

- Validation is hard-coded at the evaluator loader boundary; no partition
  argument can redirect scoring to train, test, or complete history.
- The RL contract must state that its observation scaler was fitted on train.
  Scaler training-row metadata must match the contract's train row count.
- Validation observations come from the precomputed training-scaled feature
  columns, while identity, OHLCV execution, and accounting fields must exactly
  match the unscaled validation partition.
- The canonical observation feature names and order are enforced.
- Environment v1 observes row `t`, executes at row `t+1` open, and never includes
  the next open in the row-`t` observation. Regression tests also perturb a
  later validation feature and prove earlier policy inputs are unchanged.
- Complete observation and execution date sequences are checked against the
  validation frame.
- Policy hashes and training timesteps must be unchanged by inference.

## Production-data smoke methodology and diagnostics

The safe developer command is:

```bash
.venv/bin/python -m reinforcement_learning.evaluation.comparison \
  --symbol MCB --timesteps 512 --seed 42 --random-seed 42 --smoke-test
```

`--smoke-test` is mandatory and capped at 1,024 timesteps. The run remains in
memory and has no save or registry path.

On 8 August 2026 (Asia/Karachi), one MCB integration smoke run used:

- training: 1,704 rows, 6 October 2016 through 23 August 2023;
- requested/actual PPO timesteps: 512 / 512, CPU;
- validation: 365 rows and 364 transitions, 24 August 2023 through
  12 February 2025;
- initial cash: PKR 1,000,000;
- commission: 0.10%; slippage: 0.05%;
- deterministic PPO seed: 42; Random seed: 42; and
- observation shape: `(17,)`.

Actual validation diagnostics were:

| Metric | PPO | Buy and Hold | Always Hold | Random (seed 42) |
|---|---:|---:|---:|---:|
| Initial value | 1,000,000.00 | 1,000,000.00 | 1,000,000.00 | 1,000,000.00 |
| Final value | 1,141,956.07 | 1,975,057.51 | 1,000,000.00 | 1,706,240.77 |
| Total return | 14.1956% | 97.5058% | 0.0000% | 70.6241% |
| Annualized return | 9.6254% | 60.1888% | 0.0000% | 44.7582% |
| Annualized volatility | 13.9857% | 30.5134% | 0.0000% | 24.2442% |
| Sharpe | 0.725923 | 1.697411 | Not available | 1.646593 |
| Sortino | 1.255720 | 2.703857 | Not available | 2.823766 |
| Maximum drawdown | 11.2335% | 13.9824% | 0.0000% | 12.5249% |
| Executed trades | 11 | 1 | 0 | 124 |
| Transaction costs | 18,821.38 | 1,498.20 | 0.00 | 245,590.54 |
| Realized P&L | 173,710.36 | 0.00 | 0.00 | 706,240.77 |
| Final unrealized P&L | -31,754.28 | 975,057.51 | 0.00 | 0.00 |
| Exposure | 12.6374% | 100.0000% | 0.0000% | 48.3516% |
| Completed sells | 5 | 0 | 0 | 62 |
| Profitable completed sells | 4 | 0 | 0 | 37 |

Relative to Buy and Hold, PPO total return was lower by 83.3101 percentage
points, Sharpe was lower by 0.971488, and maximum drawdown was lower by 2.7489
percentage points. The candidate decision was `validation_fail` because it did
not meet the configured return or Sharpe advantage thresholds.

This 512-step smoke result proves integration only. It is not evidence of model
quality, profitability, or production readiness. Test data was not evaluated.
The PPO policy hash and timestep count were unchanged during validation.

## Verification and artifact safety

The complete offline suite passed: 291 tests. Pre/post manifests matched for the
MCB split/scaler/contract artifacts, raw-data file metadata, `backfill_state.json`,
`model_registry.csv`, `data/models/`, and
`reinforcement_learning/saved_models/`. No model was saved, registered, or
promoted, and no live HTTP request was made.

## Limitations and next milestone

This milestone evaluates one candidate on one fixed validation partition. It
does not perform multiple-seed robustness analysis, walk-forward evaluation,
hyperparameter selection, final test evaluation, production persistence, or
registry promotion. Commission and slippage are explicit development
assumptions rather than a complete broker fee schedule.

Milestone 5B-3 should add atomic model persistence and registry integration with
clear promotion controls. Final test evaluation must remain separately gated
and should occur only after candidate selection is frozen.
