# Milestone 6E.2 — sector recurrent methodology repair

Audit/implementation date: 2026-08-12 PKT  
Sector: Commercial Banks  
Universe hash: `589485c8adfe6170a6c2391687202ac3a287de9eb30737a2cb1f57a34f111e5b`  
Trainer: `recurrent_ppo_sector_balanced_v1`  
Status: methodology implementation and tiny engine smoke only

## Objective and research separation

This milestone repairs the training methodology exposed by 6E.1. It does not
replace the blocked 6E model with a validation-selected alternative. The work
has two deliberately separate layers:

1. **Engine/methodology repair:** explicit action validity, complete reward
   attribution, exact balanced TRAIN windows, recurrent/portfolio isolation,
   exact SB3 timestep accounting, warning-only collapse diagnostics, and a
   frozen reproducibility contract.
2. **Future research experiment:** three separately initialized models with
   seeds 42, 43, and 44, each consuming the same predeclared 19-bank TRAIN
   schedule. That experiment was specified but was **not executed** here.

No reward coefficient, action mode, normalizer, observation, architecture, or
window location was selected by comparing replacement candidates on the
already-observed validation period. Validation and TEST were not evaluated in
6E.2.

## Why the 6E foundation model remains blocked

The read-only 6E.1 audit reconstructed 6,014 validation Sell selections, of
which 5,917 were selected while already flat. That was 98.39% of all Sell
selections and 88.95% of all actions. Median exposure was 4.12%, and the PPO
return beat Buy & Hold for only 1 of 19 banks. It also found unequal
symbol-specific exposure: MCB received 1,703 sector timesteps, compared with
10,240 in the independent recurrent experiment. The retained PPO diagnostics
were finite and did not support a numerical-instability explanation.

The 6E notebook also requested 50,000 rather than the specified 25,000 steps;
SB3 completed 50,176 after rollout alignment. This protocol deviation is
documented rather than normalized away. Those validation results motivated a
predeclared diagnostic and fairness repair, not reward tuning.

## Canonical action-validity contract

Semantic version: `sector_action_validity_v1`  
Selected mode version: `sector_action_validity_penalty_v1`

| Portfolio state | Hold | Buy | Sell |
|---|---|---|---|
| Flat | Valid | Valid | Invalid: no shares to sell |
| Long | Valid | Invalid/redundant | Valid |

The shared helper records the selected action, state validity, whether the
action executed, whether a trade executed, semantic invalid reason, and any
separate execution failure. A flat Buy that cannot afford one whole share is
still semantically valid; it is unexecuted with an execution-failure reason
and does not receive the state-invalid penalty.

Penalty mode preserves the three-action space. State-invalid selections are
no-ops and receive the configured fixed penalty. No action is rewritten after
selection.

### Masking decision

True recurrent masking is **not supported safely** by the installed
sb3-contrib 2.9.0 stack. `RecurrentPPO` uses a recurrent actor-critic policy and
recurrent rollout buffer with LSTM state and episode-start inputs, but exposes
no action-mask path. `MaskablePPO` is a separate stateless algorithm/policy/
buffer stack. Supplying an `action_masks()` method would have no effect, and
rewriting a sampled action would invalidate PPO log-probability semantics.

Therefore `invalid_action_mode="mask"` is represented for future research but
fails closed as `unsupported_or_deferred`. No masking dependency was added.

## Reward governance and instrumentation

Reward version: `sector_reward_v1`

```text
total_reward =
    1.0 * log(current_portfolio_value / previous_portfolio_value)
  - 0.0 * transaction_cost / previous_portfolio_value
  - 0.1 * max(0, current_drawdown - previous_drawdown)
  - 0.0001 when the selected action is state-invalid
```

Commission and slippage continue to affect portfolio accounting even though
the additional transaction-cost reward weight remains zero. The 0.1 drawdown
coefficient and 0.0001 invalid-action penalty were **not tuned** from 6E
validation results. Any coefficient change requires a separately predeclared
experiment.

Every transition now exposes signed, exactly additive components:

- portfolio growth reward;
- transaction-cost penalty;
- positive-increment drawdown penalty;
- invalid-action penalty;
- total raw environment reward.

The trainer copies these values immediately from `info`. This matters because
RecurrentPPO legitimately adds `gamma * V(terminal_observation)` to its rollout
reward for an artificial timeout after the callback has seen the raw reward.
That algorithmic bootstrap is not misreported as an environment reward
component.

Normal training retains aggregate/per-symbol reward distributions; cumulative
components; valid Hold/Buy/Sell and invalid Buy/Sell counts; invalid rate;
execution failures; action frequencies; exposure; trades; action-pattern
digests; final PPO diagnostics; and bounded periodic PPO summaries. It does
not retain every gradient update or every raw row.

## Balanced-window methodology

Sampling version: `sector_sampling_balanced_windows_v1`  
Data-schedule seed: 42  
Future model seeds: 42, 43, 44  
Window size: 512 transitions / 513 chronological TRAIN observations  
Rounds: 20  
Constituents per round: 19  
Expected exposure: 10,240 transitions per symbol  
Expected total per seed: 194,560 transitions

Each symbol's valid window-start range is divided into 20 chronological
strata. A deterministic choice is made within every stratum using the sector
universe hash, data-schedule seed, symbol, round, and sampling version. Symbol
order is deterministically shuffled per round from the same data identity.
The model seed is not an input to either operation, so seeds 42/43/44 see the
same locations and ordering while independently controlling Python, NumPy,
PyTorch, PPO initialization, and environment stochasticity.

The scheduler never concatenates symbols. Every record includes source row
bounds, source/transition counts, start date, final observation date, final
execution/terminal-observation date, chronology quartile, overlap/reuse
metadata, reset declarations, and expected termination/truncation flags.
Windows are deep copies; repeated exposure cannot mutate the canonical TRAIN
frame or change historical prices/observations.

### Termination and SB3 semantics

- A 512-transition window requires rows `t` through `t+512`—513 observations.
- A cutoff before the true TRAIN end returns
  `terminated=False, truncated=True`.
- A genuine TRAIN-partition end returns
  `terminated=True, truncated=False`.
- `DummyVecEnv` retains the final state as `terminal_observation`, returns the
  next reset observation, and marks only artificial ends as
  `TimeLimit.truncated`.
- RecurrentPPO bootstraps an artificial cutoff with the continuing critic
  state and `gamma * V(terminal_observation)`, then blocks GAE propagation
  across the done boundary.
- A PPO rollout boundary inside a live window does not reset the environment
  or LSTM.
- After the final scheduled transition, one passive vector-environment reset
  supplies a valid observation but starts no window and consumes no exposure;
  a subsequent step fails.

The future full total is exactly `380 * 512`, so one environment with
`n_steps=512` has no rollout padding. Trainer completion requires optimizer
timesteps, scheduled environment transitions, per-symbol transitions, and
per-symbol window counts all to match exactly. A padded or unequal run fails
instead of being relabelled compliant.

## Exposure and chronology audit

Full-schedule digest: `09b16449fa796a8cf4039fd0eec7fb192d32419c20237fa240ea927e7c66231f`  
Windows: 380  
Artificial truncations: 380  
Natural TRAIN ends: 0  
Exact duplicate window starts in current data: 0

All four TRAIN chronology quartiles are represented for every bank. Because
10,240 optimization transitions exceed each bank's 1,216–1,703 available
unique transitions, overlap is explicit and substantial; this is repeated
optimization exposure, not new history.

| Symbol | Available TRAIN transitions | Scheduled | Unique used | Unused | Repeated occurrences | Overlap | Unique coverage | Min positive use | Max use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABL | 1,659 | 10,240 | 1,629 | 30 | 8,611 | 84.09% | 98.19% | 1 | 10 |
| AKBL | 1,702 | 10,240 | 1,629 | 73 | 8,611 | 84.09% | 95.71% | 1 | 10 |
| BAFL | 1,702 | 10,240 | 1,619 | 83 | 8,621 | 84.19% | 95.12% | 1 | 10 |
| BAHL | 1,703 | 10,240 | 1,626 | 77 | 8,614 | 84.12% | 95.48% | 1 | 10 |
| BIPL | 1,648 | 10,240 | 1,603 | 45 | 8,637 | 84.35% | 97.27% | 1 | 10 |
| BML | 1,702 | 10,240 | 1,697 | 5 | 8,543 | 83.43% | 99.71% | 1 | 10 |
| BOK | 1,483 | 10,240 | 1,480 | 3 | 8,760 | 85.55% | 99.80% | 1 | 12 |
| BOP | 1,703 | 10,240 | 1,629 | 74 | 8,611 | 84.09% | 95.65% | 1 | 10 |
| FABL | 1,700 | 10,240 | 1,612 | 88 | 8,628 | 84.26% | 94.82% | 1 | 10 |
| HBL | 1,703 | 10,240 | 1,674 | 29 | 8,566 | 83.65% | 98.30% | 1 | 9 |
| HMB | 1,674 | 10,240 | 1,605 | 69 | 8,635 | 84.33% | 95.88% | 1 | 10 |
| JSBL | 1,645 | 10,240 | 1,606 | 39 | 8,634 | 84.32% | 97.63% | 1 | 10 |
| MCB | 1,703 | 10,240 | 1,669 | 34 | 8,571 | 83.70% | 98.00% | 1 | 10 |
| MEBL | 1,690 | 10,240 | 1,617 | 73 | 8,623 | 84.21% | 95.68% | 1 | 10 |
| NBP | 1,703 | 10,240 | 1,687 | 16 | 8,553 | 83.53% | 99.06% | 1 | 10 |
| SBL | 1,216 | 10,240 | 1,177 | 39 | 9,063 | 88.51% | 96.79% | 1 | 15 |
| SCBPL | 1,494 | 10,240 | 1,452 | 42 | 8,788 | 85.82% | 97.19% | 1 | 12 |
| SNBL | 1,605 | 10,240 | 1,532 | 73 | 8,708 | 85.04% | 95.45% | 1 | 10 |
| UBL | 1,703 | 10,240 | 1,687 | 16 | 8,553 | 83.53% | 99.06% | 1 | 10 |

The repeated-exposure range is 83.43%–88.51%, and individual available
transitions can be revisited up to 9–15 times depending on history depth. This
is a meaningful overfitting risk to examine across the predeclared model seeds;
it is not hidden by calling the run 194,560 unique observations.

## State and observation isolation

Every window constructs a fresh `SingleSymbolTradingEnv` and resets cash,
holdings, average entry price, position value, realized/unrealized P&L,
transaction costs, trade count, peak value, drawdown, and current portfolio
value. Each window's first policy transition is directly observed with
`episode_start=True`; every other transition in the fixed 512-step window is
observed with `episode_start=False`. This is checked against the complete
runtime sequence, not inferred only from schedule metadata.

Observation shape remains exactly `(17,)`: 12 canonical market features and
five dynamic portfolio features. No symbol ID, sector ID, or embedding was
added. Normalization remains the existing per-symbol TRAIN-fitted scaler, and
real OHLCV execution fields remain unscaled. This isolates the fairness/action
repair from identity and normalization ablations.

## Frozen future experiment and controls

The immutable specification is stored at
`docs/config/sector_recurrent_fair_experiment_v1.json`.

- Experiment-spec hash:
  `0c91e883f2e49d11d33c8e9891df8c89f0453865fb1f9f1c63309f342fa1345f`
- Model seeds: `[42, 43, 44]`
- Data-schedule seed: `42`
- Per seed: 19 symbols × 20 windows × 512 transitions = 194,560
- Across the future three-model-seed experiment: 583,680 optimizer/environment
  transitions, while each seed uses the same historical schedule.

The production training function refuses the full schedule unless a future
caller supplies all of: the predeclared research purpose, an explicit
authorization flag, the exact specification path, and the exact specification
hash. There is no CLI, notebook, dashboard, or import-time route that starts
the full run.

Predeclared controls are represented without execution:

- **Control A:** balanced sector model, 10,240 transitions per constituent;
- **Control B:** independent recurrent target model, 10,240 target transitions;
- **Control C:** separately predeclared total-compute-matched independent
  control.

The scheduler also accepts a leave-one-symbol-out peer universe only when the
target is already absent from supplied TRAIN frames. The target receives zero
windows and is absent from normalization contributors. This validates the
scheduler foundation but does not authorize transfer training.

## Tiny real-data methodology smoke

The final smoke used ABL, AKBL, and BAFL; seed 42; CPU; two windows per symbol;
512 transitions per window; and `n_steps=128` so three nonterminal rollout
boundaries occur inside every window. Total scheduled and optimizer exposure
was 3,072 transitions, exactly 1,024 per bank, with zero padding.

Final measured result:

| Field | Result |
|---|---:|
| Status | Completed |
| Duration | 7.165 seconds |
| Schedule digest | `a6e9acdf0695c30aebf978d7157b7fa4d8d983d536c43c88b2f588195951b7c2` |
| Scheduled / optimizer transitions | 3,072 / 3,072 |
| Rollout padding | 0 |
| Per-symbol transitions | ABL 1,024; AKBL 1,024; BAFL 1,024 |
| Per-symbol windows | 2 each |
| Artificial truncations | 6 |
| Non-counted passive final resets | 1 |
| Rollout boundaries / continuity checks | 24 / 23 |
| Episode-start sequence verified | Yes |
| Recurrent reset verified | Yes |
| Portfolio reset verified | Yes |
| Rollout continuity verified | Yes |
| Actual device | CPU |

The raw environment reward distribution over 3,072 transitions had mean
`-0.000750`, median `-0.000100`, standard deviation `0.009546`, minimum
`-0.142698`, and maximum `0.055817`. Fractions were 15.20% positive, 64.88%
negative, and 19.92% zero. Cumulative attribution was portfolio growth
`-1.586564`, drawdown `-0.561097`, invalid-action penalty `-0.155700`,
additional transaction-cost penalty `0`, and total `-2.303361`.

Selected actions were Hold 832, Buy 526, and Sell 1,714. Canonical validity
counts were valid Hold 832, valid Buy 343, valid Sell 340, invalid Buy 183,
and invalid Sell 1,374, for a 50.68% invalid rate. Median per-symbol exposure
was 24.80%; all three symbols traded; no action-dominance, invalid-attractor,
cash-collapse, zero-trade, or identical-pattern warning threshold fired. This
does not show that a longer model is healthy; it only proves the diagnostics
work and report an honest tiny-run outcome.

Final PPO diagnostics were: approximate KL `0.026967`, clip fraction
`0.167969`, entropy loss `-0.575791`, explained variance `-0.055376`, policy
gradient loss `-0.014327`, value loss `0.000176`, learning rate `0.0003`, and
240 updates. Twelve bounded periodic diagnostic summaries were retained.

This run is an engine verification, not an economic evaluation. It performs no
validation or TEST inference, persists no model, and cannot support a claim
about profitability, sector knowledge, or transfer benefit.

## Limitations

- Penalty mode makes invalidity explicit but does not mathematically prevent a
  policy from selecting invalid actions; native recurrent masking remains
  unavailable.
- The reward coefficients are preserved, so the 6E.1 drawdown/investment
  incentive remains a research question rather than a tuned-away result.
- Heavy overlap is unavoidable under 20 × 512 exposure with the available
  history and may overfit individual regimes.
- Per-symbol normalization remains economically heterogeneous under one shared
  identity-free policy.
- One tiny seed-42 smoke establishes mechanics only. The full multi-seed result
  does not exist.
- Leave-one-out transfer additionally needs a dedicated target-excluded data
  loader/manifest and approved fine-tuning/control protocol before any claim
  about knowledge transfer.

## Final decisions

### BALANCED SECTOR TRAINING METHODOLOGY: **GO**

The implementation now enforces equal scheduled exposure, TRAIN-only windows,
correct Gym/SB3 timeout semantics, direct episode-start observation, portfolio
and LSTM isolation, exact timestep reconciliation, immutable inputs, and
diagnostic retention.

### INVALID-ACTION ATTRACTOR: **PARTIALLY_RESOLVED**

Invalid selections can now be identified precisely and audited per symbol;
state validity no longer depends on duplicated evaluator logic, and fake
masking is prohibited. Penalty mode still permits invalid choices, so only the
future predeclared experiment can show whether the attractor remains.

### FULL 3-SEED COMMERCIAL BANKS EXPERIMENT: **CONDITIONAL**

The executable methodology and immutable specification are ready, but the run
remains gated for review. Before authorization, review the overlap/reuse table,
reward/action smoke diagnostics, exact specification hash, runtime estimate,
and criteria for interpreting—not tuning from—the three fixed-seed outcomes.

### LEAVE-ONE-OUT TRANSFER: **BLOCKED**

Do not fine-tune a target until a non-collapsed, predeclared multi-seed sector
foundation has been evaluated; then use a target-excluded manifest/data loader,
target-excluded normalization contributors, fixed target history, and the
predeclared scratch/exposure controls.

## Integrity and verification

- Full pytest result: **565 passed, 2 skipped in 23.26 seconds**
- `git diff --check`: **passed**
- `.venv/bin/python -m pip check`: **passed — no broken requirements found**
- Production registry SHA-256: **`e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a` (unchanged)**
- Production saved-model directories: **unchanged; only tracked `.gitkeep` placeholders are present**
- Full `194,560 × 3` experiment: **not run**
- Validation-driven coefficient/action selection: **not performed**
- TEST: **not loaded or evaluated**
- Live HTTP: **not used**
- Commit: **not created**
