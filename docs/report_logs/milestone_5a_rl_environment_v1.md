# Milestone 5A: Gymnasium single-symbol environment v1

## Objective

Implement and validate a reusable Gymnasium trading simulator for one PSX
symbol without training PPO. The explicit version is
`single_symbol_env_v1`.

## Environment as a trading simulator

The environment is a deterministic accounting simulator, not a prediction
model. It consumes one processed symbol dataset, checks identity and OHLCV
structure, preserves numeric-looking symbols as strings, rejects duplicate
dates and invalid values, sorts a deep copy chronologically, and never mutates
the caller's DataFrame.

## Observation design

The flat `float32` observation uses an explicitly configured tuple of
externally preprocessed market and technical features. The environment never
fits a scaler. It appends cash ratio, position-value ratio, position indicator,
unrealized-return ratio, and drawdown. Symbol and raw date are excluded.
Non-finite inputs and observations are errors rather than silently becoming
zero. The only explicit zero convention is portfolio unrealized return while
no position exists.

## Action design

`Discrete(3)` maps `0` to Hold, `1` to Buy, and `2` to Sell. Buy uses all
available cash for the maximum whole number of shares. Sell closes all shares.
Invalid buys and sells are non-transactions reported in `info`. V1 has no
shorting, leverage, margin, fractional shares, or continuous actions.

## Next-open execution and look-ahead avoidance

At step `t`, the policy observes features through date `t`. Its action executes
at the next available date's open, with slippage, and the resulting portfolio
is valued at that next date's close. Reward is calculated only after this
transition. The next open is not included in the decision observation. The
transition into the final row terminates cleanly.

## Portfolio accounting

Cash, whole shares, commission-inclusive average entry price, position value,
total value, realized and unrealized P&L, cumulative costs, peak value,
drawdown, and trade count are tracked. The accounting identity is:

```text
portfolio value = cash + shares held × current close
```

Buy cash outflow is slipped execution value plus commission. Sell cash inflow
is slipped execution value minus commission. Realized P&L subtracts stored cost
basis from net proceeds. Reported transaction costs contain commission and the
slippage impact relative to the unadjusted open.

## Costs and slippage

Development defaults are PKR 1,000,000 cash, 0.10% commission, and 0.05%
slippage. These are configurable simulation parameters and are not claimed to
match an exact PSX broker fee schedule.

## Reward v1

The base reward is log portfolio growth. Configurable deductions are normalized
transaction cost, positive drawdown increase, and an invalid-action penalty.
Default transaction-cost weight is zero to avoid double-counting costs already
embedded in portfolio value; drawdown weight is `0.1`; invalid action is
`0.0001`. Components are returned separately in `info`, remain finite, and use
no arbitrary profit bonus or future data.

## Baselines and evaluation

Always Hold, Buy and Hold, and fixed-seed Random policies provide deterministic
non-AI reference episodes. Metrics include initial/final value, total return,
maximum drawdown, trades, total costs, transition return series, volatility,
and Sharpe with a zero risk-free rate and 252-day annualization. Annualized
metrics are unavailable for fewer than two transition returns and are not
presented as meaningful for short samples.

## Validation methodology and testing

Reusable validation runs Gymnasium's environment checker, verifies observation
shape/dtype/finiteness, action behavior, deterministic resets and episodes,
non-negative balances, accounting identity, and termination. Offline tests
cover next-open execution, costs, slippage, maximum whole-share sizing, P&L,
drawdown, history isolation, reward components, truncation, baselines,
evaluation metrics, source immutability, and insufficient-history readiness.
No live requests or PPO training are used.

## Local-data limitation

At implementation time, raw local history tops out at 143 rows and processed
history at 94 usable rows per symbol. No processed per-symbol files or symbol
scalers exist, so no symbol meets the configured 252-row gate. Notebook 04
therefore reports non-readiness and offers a clearly labelled deterministic
fixture demonstration that cannot support research conclusions.

## Next milestone

Milestone 5B may add a controlled PPO pilot, training configuration,
walk-forward evaluation, artifact persistence, and registry metadata. It must
not weaken the timing, accounting, validation, or preprocessing contracts
established here.
