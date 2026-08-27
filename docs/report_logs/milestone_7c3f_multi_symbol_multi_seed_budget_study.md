# Milestone 7C.3f — Multi-Symbol Multi-Seed Recurrent Budget Study

Study date: 2026-08-27  
Study version: `recurrent_budget_study_v1`  
Study fingerprint: `1f2f95d2b816e0887fc1acee788a002bd4d2689dfdf7e5b273d0a01ea13ad53f`

## Outcome

All **54/54** predeclared scientific runs completed training and VALIDATION
evaluation successfully. The frozen decision rule returns:

**BLOCKED_BUDGET_SELECTION**

This is not a runtime or training failure. The evidence does not satisfy one
of the three predeclared freeze branches: 50k and 100k are technically mature,
but their next larger budget shows a widespread material training-diagnostic
improvement; 250k shows the strongest explained variance but narrowly misses
the frozen diagnostic maturity fraction because 2/18 final KL values exceed
0.10. The rule is not changed after seeing this result.

No TEST values were loaded, no model was persisted, and no registry entry or
full-universe authorization was created.

## Representative panel

Selection used only canonical TRAIN partitions from eligible identities with
VALIDATION available. Each symbol received percentile ranks for TRAIN rows,
active-span coverage over actual observed TRAIN market dates, and log median
volume. Six fixed archetype targets were matched by squared rank distance,
with small zero-volume/zero-OHL quality terms and symbol as the deterministic
tie-breaker. Reward, validation return, Sharpe, runtime, and previous PPO
results were not inputs.

| Regime | Symbol | TRAIN rows | TRAIN dates | Active-span coverage | Median volume | Zero volume/OHL |
|---|---|---:|---|---:|---:|---:|
| Long history / high coverage | MARI | 1,704 | 2016-10-06–2023-08-23 | 100.00% | 21,222 | 0% / 0% |
| Medium history | AGP | 1,421 | 2018-05-15–2024-02-15 | 99.79% | 86,106 | 0% / 0% |
| Shorter valid history | LSEFSL | 341 | 2024-08-16–2025-12-29 | 99.13% | 8,361 | 0% / 0% |
| High liquidity | WAVES | 1,628 | 2017-03-10–2023-10-09 | 100.00% | 489,000 | 0% / 0% |
| Medium liquidity | GLAXO | 1,703 | 2016-10-06–2023-08-22 | 100.00% | 44,500 | 0% / 0% |
| Lower-liquidity / sparser valid | AKDHL | 709 | 2021-10-07–2025-04-21 | 81.03% | 800 | 0% / 0% |

The complete 432-symbol TRAIN descriptor population and selected rows are in
`train_descriptors.csv` and `selected_symbols.csv`.

## Frozen experiment and model configuration

- Symbols: exactly the six above.
- Seeds: exactly `42`, `123`, `2026`.
- Requested budgets: exactly `50,000`, `100,000`, `250,000`.
- Total: 6 × 3 × 3 = **54 runs**, 7,200,000 requested and 7,216,128
  rollout-rounded actual timesteps.
- Device: explicit CPU for every run; no MPS or CUDA attempt.
- Algorithm/policy: RecurrentPPO / MlpLstmPolicy.
- LSTM: one 64-unit layer; `shared_lstm=False`, critic LSTM enabled.
- Actor/critic network: `[64]`, Tanh, shared feature extractor, orthogonal init.
- Learning rate: `3e-4`.
- Rollout / batch / epochs: 512 / 64 / 10.
- Gamma / GAE lambda: 0.99 / 0.95.
- Clip range: 0.20.
- Entropy / value coefficients: 0.01 / 0.50.
- Maximum gradient norm: 0.50.
- Environment: `single_symbol_env_v1`, observation `(17,)`.
- Trainer: `recurrent_ppo_single_symbol_v1`.
- Parameter count: 51,076 in all 54 runs.

TRAIN alone was visible to optimization. Each completed in-memory model was
evaluated deterministically on its matching complete VALIDATION episode with
a fresh hidden state. Parameters and timestep counters were unchanged in all
54 evaluations. Models were then discarded.

## Complete results and generated tables

The canonical per-run table contains all 54 symbol × seed × budget records and
43 fields, including every requested training/validation diagnostic:

- `per_run_results.csv`
- `per_symbol_aggregates.csv`
- `global_aggregates.csv`
- `runtime_scaling.csv`
- `budget_decision.json`

All are under `docs/report_logs/milestone_7c3f_budget_study/`. Per-run actual
timesteps were 50,176, 100,352, and 250,368 respectively. Every status is
`completed`, every effective device is `cpu`, and every recurrent continuity
flag is true. Continuity checks were 97, 195, and 488 per run by budget.

## Global training stability

| Budget | Success | Mean KL | Mean clip fraction | Mean entropy loss | EV mean ± std | EV median | Mean value loss | Continuity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50k | 18/18 | 0.0304 | 0.1349 | -0.4592 | 0.7450 ± 0.1960 | 0.7461 | 0.001510 | 18/18 |
| 100k | 18/18 | 0.0385 | 0.1420 | -0.3727 | 0.6814 ± 0.2831 | 0.7480 | 0.000855 | 18/18 |
| 250k | 18/18 | 0.0670 | 0.1699 | -0.3273 | 0.9583 ± 0.0320 | 0.9652 | 0.000781 | 18/18 |

Final diagnostic ranges were finite throughout. The 250k KL exceptions were:

- WAVES, seed 123: KL 0.150537, clip fraction 0.205078, EV 0.923948.
- AKDHL, seed 123: KL 0.130680, clip fraction 0.159570, EV 0.992019.

Clip fraction stayed below the frozen 0.30 bound in every run. These two KL
exceptions make the 250k maturity fraction 16/18 = 88.89%, just below the
predeclared 90% requirement. This is not described as numerical collapse;
it is a failed frozen selection gate.

Paired budget evidence:

| Comparison | Median EV change | EV improved pairs | Median relative value-loss reduction | Value-loss improved pairs | Frozen widespread result |
|---|---:|---:|---:|---:|---|
| 50k → 100k | -0.0282 | 8/18 | 26.64% | 13/18 | Yes, through value-loss criterion |
| 100k → 250k | +0.2152 | 16/18 | 24.77% | 13/18 | Yes |

## Seed sensitivity

| Budget | Seed | Mean EV across symbols | EV std | Mean validation return | Return std | Mean validation Sharpe |
|---:|---:|---:|---:|---:|---:|---:|
| 50k | 42 | 0.8181 | 0.1187 | 0.0977 | 0.2998 | 0.1466 |
| 50k | 123 | 0.6739 | 0.2620 | 0.1830 | 0.5182 | 0.1619 |
| 50k | 2026 | 0.7431 | 0.2075 | 0.0772 | 0.3167 | 0.2408 |
| 100k | 42 | 0.6564 | 0.3183 | 0.2733 | 0.4601 | 0.6286 |
| 100k | 123 | 0.6913 | 0.3103 | 0.1579 | 0.3539 | 0.3216 |
| 100k | 2026 | 0.6964 | 0.2996 | -0.0106 | 0.2233 | -0.0706 |
| 250k | 42 | 0.9462 | 0.0448 | 0.1858 | 0.1735 | 0.9616 |
| 250k | 123 | 0.9664 | 0.0258 | 0.0363 | 0.2461 | 0.1806 |
| 250k | 2026 | 0.9621 | 0.0271 | 0.1951 | 0.4337 | 0.5999 |

250k produced materially tighter explained variance across seeds/symbols, but
validation returns and Sharpe remain visibly seed- and symbol-sensitive. No
budget is selected by the return or Sharpe columns.

## Symbol sensitivity and validation consistency

Mean explained variance across seeds:

| Symbol | 50k | 100k | 250k | 50k→100k | 100k→250k |
|---|---:|---:|---:|---:|---:|
| AGP | 0.6687 | 0.8206 | 0.9584 | +0.1519 | +0.1378 |
| AKDHL | 0.9482 | 0.8946 | 0.9829 | -0.0535 | +0.0883 |
| GLAXO | 0.6378 | 0.6323 | 0.9409 | -0.0055 | +0.3085 |
| LSEFSL | 0.9829 | 0.9942 | 0.9899 | +0.0113 | -0.0043 |
| MARI | 0.6267 | 0.1664 | 0.9631 | -0.4603 | +0.7966 |
| WAVES | 0.6059 | 0.5800 | 0.9144 | -0.0259 | +0.3344 |

Mean VALIDATION return across seeds:

| Symbol | 50k | 100k | 250k |
|---|---:|---:|---:|
| AGP | 21.55% | 43.64% | 44.41% |
| AKDHL | 24.02% | 15.20% | 20.57% |
| GLAXO | 27.43% | 36.79% | 18.59% |
| LSEFSL | -43.97% | -40.46% | -23.41% |
| MARI | 39.40% | 26.31% | 23.72% |
| WAVES | 3.14% | 2.64% | -0.44% |

The economic results are not monotonic and illustrate why selecting the
largest mean return would be unsound. The training-stability improvement from
100k to 250k is broad (five of six symbol means improve), but 50k to 100k is
mixed and qualifies only through the predeclared value-loss branch.

## Runtime and full-universe CPU cost

| Budget | Mean runtime | Std | Mean steps/s | Sequential 435-agent estimate |
|---:|---:|---:|---:|---:|
| 50k | 109.40 s | 4.54 s | 459.40 | 13.22 h |
| 100k | 218.17 s | 6.70 s | 460.40 | 26.36 h |
| 250k | 542.63 s | 17.63 s | 461.87 | 65.57 h |

The scientific runs consumed 15,663.54 training seconds (4.351 hours). These
are sequential CPU estimates. No parallel, MPS, or CUDA speedup is assumed.

## Failure and interruption audit

Before scientific training began, nine MARI workers exited at zero timesteps
because the worker CLI mistakenly inherited a parent-only output-directory
argument. The parent was stopped immediately, the parser was corrected and
tested, and all nine exact identities were explicitly retried. Their original
JSON records remain under each run's `attempts/infrastructure_attempt_000.json`.
No symbol, seed, or budget was substituted. The final scientific table contains
54 successful runs and zero training/validation failures.

## Frozen decision application

Predeclared maturity required at least 17/18 successful training and validation
runs, at least two seeds for every symbol, complete recurrent continuity, at
least 90% of runs within KL ≤ 0.10 and clip fraction ≤ 0.30, and median EV ≥ 0.
Material next-budget improvement required median EV gain ≥ 0.05 with at least
12 positive pairs, or median absolute value-loss reduction ≥ 20% with at least
12 improved pairs.

- 50k: mature, but 100k has widespread material value-loss improvement.
- 100k: mature, but 250k has widespread material EV and value-loss improvement.
- 250k: not mature under the frozen 90% diagnostic-fraction gate (88.89%).

Changing the KL bound or the 90% fraction now would be post-result threshold
tuning. Freezing 100k would ignore a broad predeclared 250k improvement;
freezing 250k would waive its failed gate. Therefore the only defensible result
under this study is a blocker. A follow-up methodology must be declared before
additional data are inspected; this milestone does not improvise one.

## Safety

- Optimization loader: TRAIN only.
- Evaluation loader: VALIDATION only.
- TEST flags: false in the manifest and all 54 records.
- Validation parameter/timestep immutability: 54/54.
- MPS/CUDA attempts: none.
- Persisted production models: none.
- Model registry writes: none.
- Full 435-agent training: not started.
- Source Parquet: unchanged.
- Commit: none.

## Verification and files

- Complete pytest suite: **696 passed, 2 skipped in 27.50 seconds**.
- `pip check`: **No broken requirements found**. The pip cache ownership
  warning was environmental and did not affect dependency checking.
- `git diff --check`: passed.
- Final result-table accounting: 55 CSV lines including header; 54 unique runs.
- Per-symbol aggregate accounting: 18 symbol-budget rows.
- Global aggregate accounting: three budget rows.

Files added:

- `reinforcement_learning/training/recurrent_budget_study.py`
- `data_pipeline/tests/test_recurrent_budget_study.py`
- `docs/report_logs/milestone_7c3f_multi_symbol_multi_seed_budget_study.md`
- `docs/report_logs/milestone_7c3f_budget_study/` containing the immutable
  manifest/schedule, descriptors, 54 final run records, nine archived
  zero-timestep infrastructure attempts, and all requested aggregate tables

## Decision

**BLOCKED_BUDGET_SELECTION**
