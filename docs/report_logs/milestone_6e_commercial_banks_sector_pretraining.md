# Milestone 6E — Commercial Banks Sector RecurrentPPO Pretraining

Run date: 2026-08-12 (Asia/Karachi)

Git commit used by the research run:
`5790e0697d6ae281e9b5d24a71df4f290b9b7107`

## Objective and research boundary

Milestone 6E proves that one RecurrentPPO policy can train across isolated,
full-partition Commercial Banks episodes and can be evaluated independently on
each constituent's VALIDATION episode. The result is a **Commercial Banks
sector foundation RecurrentPPO**. It is not a transfer-learning result, a Cold
Start result, a symbol-specific model, or a production trading model.

No individual fine-tuning, target-excluded model training, TEST evaluation,
hyperparameter search, production persistence, registry entry, or promotion
occurred.

## Versions and reproducibility identity

- sector trainer: `recurrent_ppo_sector_v1`;
- algorithm/policy: RecurrentPPO / MlpLstmPolicy;
- sector taxonomy: `psx_sector_taxonomy_v1`;
- sector manifest: `sector_universe_v1`;
- recurrent contract: `rl_recurrent_partition_v1`;
- environment: `single_symbol_env_v1`;
- universe hash:
  `589485c8adfe6170a6c2391687202ac3a287de9eb30737a2cb1f57a34f111e5b`;
- sampling: `equal_symbol_episode_sampling_v1`;
- normalization: existing per-symbol TRAIN-fitted scalers;
- Python 3.11.9, sb3-contrib 2.9.0, Stable-Baselines3 2.9.0,
  PyTorch 2.13.0, Gymnasium 1.3.0;
- seed 42; requested device CPU; actual device CPU.

The run result records the complete PPO/LSTM configuration, dependencies,
source commit, universe, sampling schedule/digest, and per-symbol episode and
timestep contributions.

## Fail-closed universe audit

Exact constituents, in canonical manifest order:

`ABL, AKBL, BAFL, BAHL, BIPL, BML, BOK, BOP, FABL, HBL, HMB, JSBL, MCB,
MEBL, NBP, SBL, SCBPL, SNBL, UBL`

All 19 passed these checks before model creation:

- unique symbol and Mature history class;
- currently verified Commercial Banks sector evidence;
- current recurrent, feature, and environment versions;
- identical ordered 12 market features plus five dynamic portfolio features;
- observation shape exactly `(17,)`, with no symbol/filename/metadata feature;
- canonical TRAIN artifact, recurrent contract, scaler, and scaler metadata
  present with matching SHA-256;
- per-symbol TRAIN-fitted normalization;
- positive real execution OHLC and non-negative real volume;
- one full TRAIN episode boundary and no validation/TEST frame load.

The 19 artifacts contain 31,157 referenced TRAIN rows. No incompatible symbol
is silently dropped; any mismatch aborts the entire universe load.

## Recurrent architecture and optimizer

The baseline deliberately reuses the proven 6C architecture:

- `lstm_hidden_size=64`;
- `n_lstm_layers=1`;
- `shared_lstm=false`;
- `enable_critic_lstm=true`;
- policy/value `net_arch=[64]`;
- Tanh activation;
- orthogonal initialization enabled;
- 51,076 policy parameters.

PPO settings remained unchanged: learning rate 3e-4, n_steps 512, batch size
64, 10 epochs, gamma 0.99, GAE lambda 0.95, clip 0.20, entropy coefficient
0.01, value coefficient 0.50, and max gradient norm 0.50. No result-driven
configuration adjustment was made.

## Multi-symbol environment and reset semantics

Price frames are never concatenated. A deterministic shuffled-cycle scheduler
selects a symbol, then a fresh `SingleSymbolTradingEnv` runs the complete TRAIN
partition. A natural Gymnasium termination occurs before another symbol is
selected. At every transition the controller verifies and resets:

- `episode_start=True` for the new episode;
- RecurrentPPO hidden state through the termination/done mask;
- cash to the configured initial amount;
- holdings to zero;
- realized/unrealized P&L;
- transaction costs;
- portfolio peak and drawdown;
- environment history and step state.

Rollout boundaries do not reset the environment or LSTM. Instrumentation
verified recurrent continuity across rollout updates and counted real symbol
terminations separately. Runtime assertions reject non-finite observations,
rewards, model parameters/losses, invalid accounting state, or non-positive
execution prices.

## Sampling fairness

The scheduler emits one random permutation of all 19 symbols per cycle. The
permutation is deterministic for universe hash, seed, and sampling strategy;
different seeds can produce different schedules. This prevents indefinite
symbol omission, but equal episode selection is not equal timestep exposure
when full episodes have different lengths.

The research schedule digest was
`609eecfc4ba44451e986541ed3f1cb3a311e9dcf09f71c9beebeace0b1d6914e`.

| Symbol | Episodes started | Completed | Timesteps | Contribution |
|---|---:|---:|---:|---:|
| ABL | 2 | 2 | 3,318 | 6.61% |
| AKBL | 2 | 2 | 3,404 | 6.78% |
| BAFL | 2 | 2 | 3,404 | 6.78% |
| BAHL | 1 | 1 | 1,703 | 3.39% |
| BIPL | 1 | 1 | 1,648 | 3.28% |
| BML | 1 | 1 | 1,702 | 3.39% |
| BOK | 2 | 2 | 2,966 | 5.91% |
| BOP | 2 | 2 | 3,406 | 6.79% |
| FABL | 2 | 2 | 3,400 | 6.78% |
| HBL | 2 | 1 | 2,557 | 5.10% |
| HMB | 1 | 1 | 1,674 | 3.34% |
| JSBL | 2 | 2 | 3,290 | 6.56% |
| MCB | 1 | 1 | 1,703 | 3.39% |
| MEBL | 2 | 2 | 3,380 | 6.74% |
| NBP | 2 | 2 | 3,406 | 6.79% |
| SBL | 1 | 1 | 1,216 | 2.42% |
| SCBPL | 2 | 2 | 2,988 | 5.96% |
| SNBL | 1 | 1 | 1,605 | 3.20% |
| UBL | 2 | 2 | 3,406 | 6.79% |

All symbols received exposure. Thirty episodes ended naturally, none was
truncated or failed, and HBL's second episode was still active when the fixed
budget ended. The largest contribution was 2.80 times the smallest; therefore
the run is episode-selection-balanced, not timestep-balanced. A future
predeclared capped/window design should be evaluated before compute-matched
transfer claims.

## Integration smoke

The real 2,048-step CPU smoke completed successfully:

- actual steps: 2,048;
- duration: 5.37 seconds;
- throughput: 381.07 steps/second;
- BAFL completed one natural 1,702-step episode; FABL contributed 346 steps;
- 17 banks received no exposure at this small budget and were explicitly
  reported as unexposed;
- rollout continuity verified; one true environment reset;
- all captured diagnostics were finite.

This smoke demonstrated integration only and was not evaluated economically.

## Predefined research run and compute

The research budget was fixed at **50,000 total sector timesteps** before
VALIDATION because one complete full-partition cycle requires approximately
31,138 environment transitions. It is not 50,000 steps per symbol.

- requested steps: 50,000;
- rollout-aligned actual steps: 50,176;
- duration: 111.81 seconds;
- throughput: 448.76 steps/second;
- device: CPU;
- constituents encountered: 19/19;
- natural completed episodes: 30;
- truncations/failures: 0.

Final finite diagnostics:

| Diagnostic | Value |
|---|---:|
| Updates | 980 |
| Approximate KL | 0.008941 |
| Clip fraction | 0.097656 |
| Entropy loss | -0.502825 |
| Explained variance | 0.630388 |
| Policy-gradient loss | -0.019343 |
| Value loss | 0.000581 |
| Learning rate | 0.000300 |

No NaN, infinity, invalid observation/reward, or non-finite policy parameter was
detected. A first notebook execution completed training and validation but
stopped in a post-validation display-only DataFrame expression. Because no
artifact was written, the notebook was rerun with the identical predeclared
configuration after fixing only that presentation expression. No change was
made in response to validation performance.

## Independent per-symbol VALIDATION

Each bank used its own complete VALIDATION episode, fresh LSTM state, fresh
portfolio, and the same PKR 1,000,000 initial capital. Baselines used the exact
same dates, capital, commission, slippage, and execution semantics. These are
statistics over 19 independent episodes—not a compounded sector portfolio.

| Symbol | PPO return | Buy & Hold | PPO Sharpe | Sortino | Volatility | Max DD | Trades | Costs | Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABL | -7.36% | 99.68% | -0.47 | -0.63 | 10.35% | 13.04% | 16 | 24,414 | 8.19% |
| AKBL | 9.97% | 160.87% | 0.60 | 1.15 | 12.21% | 11.70% | 8 | 12,460 | 12.36% |
| BAFL | 4.35% | 83.63% | 0.82 | 1.71 | 3.70% | 2.69% | 8 | 12,488 | 1.65% |
| BAHL | 15.59% | 154.69% | 1.58 | 3.02 | 6.46% | 4.00% | 6 | 10,086 | 5.22% |
| BIPL | -6.06% | 40.62% | -0.99 | -1.07 | 4.44% | 6.50% | 8 | 11,853 | 1.99% |
| BML | 2.33% | 119.67% | 0.26 | 0.49 | 7.07% | 5.23% | 4 | 5,946 | 1.93% |
| BOK | 23.52% | 26.89% | 0.63 | 0.93 | 38.97% | 26.60% | 12 | 22,674 | 32.18% |
| BOP | 48.06% | 207.29% | 1.35 | 2.04 | 22.02% | 18.09% | 28 | 57,465 | 25.55% |
| FABL | -5.37% | 109.18% | -0.50 | -0.57 | 7.11% | 9.34% | 10 | 14,874 | 2.48% |
| HBL | 1.28% | 63.99% | 0.14 | 0.22 | 9.21% | 6.14% | 14 | 20,660 | 7.14% |
| HMB | 7.92% | 163.15% | 0.79 | 1.18 | 7.12% | 6.22% | 16 | 26,584 | 9.22% |
| JSBL | -1.84% | 62.43% | -0.20 | -0.29 | 5.77% | 3.68% | 4 | 5,997 | 1.42% |
| MCB | 15.74% | 97.51% | 1.72 | 4.25 | 5.98% | 5.68% | 20 | 33,255 | 8.52% |
| MEBL | 0.00% | 101.17% | Not Available | Not Available | 0.00% | 0.00% | 0 | 0 | 0.00% |
| NBP | 6.31% | 212.20% | 0.54 | 0.84 | 8.55% | 7.28% | 12 | 18,476 | 4.12% |
| SBL | -6.70% | -24.21% | -0.76 | -0.78 | 8.34% | 8.35% | 4 | 5,661 | 1.54% |
| SCBPL | -17.05% | 81.34% | -1.52 | -1.57 | 9.39% | 18.92% | 8 | 10,702 | 3.45% |
| SNBL | -5.59% | 72.70% | -0.47 | -0.76 | 8.35% | 12.07% | 12 | 16,768 | 4.66% |
| UBL | 1.68% | 180.19% | 1.36 | Not Available | 0.85% | 0.00% | 4 | 6,048 | 0.55% |

The executed notebook contains the complete 76-row RecurrentPPO, Buy & Hold,
Always Hold, and fixed-seed Random metric table, including final value,
annualized return, volatility, Sharpe, Sortino, drawdown, trades, costs, and
exposure.

## Aggregate validation summary

| Metric | Mean | Median | Minimum | Q25 | Q75 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| PPO total return | 4.57% | 1.68% | -17.05% | -5.48% | 8.95% | 48.06% |
| Buy & Hold total return | 105.95% | 99.68% | -24.21% | 68.35% | 157.78% | 212.20% |
| PPO Sharpe (defined) | 0.27 | 0.40 | -1.52 | -0.47 | 0.81 | 1.72 |
| PPO maximum drawdown | 8.71% | 6.50% | 0.00% | 4.62% | 11.88% | 26.60% |
| PPO transaction costs | 16,653 | 12,488 | 0 | 8,067 | 21,667 | 57,465 |
| PPO exposure | 6.96% | 4.12% | 0.00% | 1.79% | 8.35% | 32.18% |

- PPO beat Buy & Hold on return for 1/19 symbols (5.26%) and lost for 18/19;
- PPO beat Buy & Hold Sharpe for 2/18 comparable symbols (11.11%);
- 11 positive PPO returns, seven negative, one exactly zero;
- one undefined PPO Sharpe (MEBL);
- zero evaluation failures.

Positive average return is not classified as success. The validation regime was
strongly favorable to Buy & Hold, and the shared policy generally held too
little exposure to participate.

## Policy-collapse diagnosis

Across all independent validations:

- Hold: 372 actions (5.59%);
- Buy: 266 actions (4.00%);
- Sell: 6,014 actions (90.41%);
- median exposure: 4.12%;
- one zero-trade symbol: MEBL;
- no identical full action-sequence group.

The evaluator flags this as a **possible sell-dominant/mostly-cash policy
collapse**. Although the pipeline is technically healthy, this policy behavior
prevents an unconditional transfer-parent recommendation.

## Baseline/comparison interpretation

Buy & Hold, Always Hold, and fixed-seed Random were evaluated on all 19 symbols.
The optional independent RecurrentPPO/MLP retraining subset was not run: matching
total budget would still misrepresent target-specific exposure, while true
compute/exposure matching needs a predeclared separate experiment. No winner is
claimed against an old model trained under a different budget.

## Leave-one-out foundation

The helper derived an MCB-excluded Commercial Banks universe with hash
`0e90c86759b3ac5e4fa0d3cf8c4edac802ffb158d2720311b525e5b92f8e60c3`.
MCB is absent from both pretraining constituents and normalization contributors,
and target exclusion is explicit. No leave-one-out model was trained.

## Temporary persistence and registry v3 design

A temporary directory outside the project proved RecurrentPPO save, reload,
deterministic action/state equality, metadata integrity, and cleanup. The
temporary model hash was
`c76e7a7406799beaab19c9867bbf653632e506497a0adabdd255f7339f4a946a`.
No production registry or saved-model directory changed.

Future `model_registry_v3` needs sector scope/name, taxonomy/universe identity,
constituent count and manifest path, sampling and normalization policy,
recurrent contract/trainer versions, parent model ID, target/exclusion fields,
and transfer/fine-tune lineage. No migration or row was created in 6E.

Safe interruption checkpoints are deferred. A correct resume must restore the
model, optimizer, rollout state, shuffled-cycle RNG/cursor, active symbol and
environment offset, and recurrent state. A partial `model.save()` alone could
be mistaken for a final candidate, so 6E fails/intercepts without retaining a
model instead of implementing an unsafe checkpoint.

## Survivorship and other limitations

- Sector grouping uses current verified sector evidence, not proof of
  historical sector membership throughout TRAIN.
- One sector, one seed, one predefined configuration, and one validation
  regime were tested.
- Equal episode sampling still yielded a 2.8x timestep-exposure range.
- The model was not trained under target exclusion and cannot support a
  transfer-benefit claim.
- Validation influenced no retraining, configuration, universe, or budget.
- TEST remained completely sealed.

## Next-milestone readiness

1. Multi-symbol recurrent training functioned correctly: **yes**.
2. All 19 symbols received training exposure: **yes**.
3. Recurrent and portfolio state isolation was verified: **yes**.
4. Training was numerically stable: **yes**.
5. Validation was broad: all 19 completed, but performance was weak versus Buy
   & Hold and heterogeneous across symbols.
6. Policy collapse was observed: **possible sell-dominant/mostly-cash collapse**.
7. Technically suitable as a transfer parent: **conditional**, not yet a
   research-quality parent.
8. Before leave-one-out transfer: predeclare collapse acceptance criteria,
   timestep/exposure matching, multi-seed targets, target-excluded universes and
   scalers, and a normalization/sampling refinement experiment independent of
   these validation results.

**SECTOR RECURRENT PRETRAINING ENGINE: GO**

The TRAIN-only loader, deterministic scheduler, episode/accounting/LSTM reset,
diagnostics, per-symbol validation, aggregation, leave-one-out metadata, and
temporary persistence are technically sound.

**COMMERCIAL BANKS FOUNDATION MODEL FOR TRANSFER EXPERIMENTS: CONDITIONAL**

The artifact is technically loadable and reproducible, but its sell-dominant,
low-exposure validation behavior, one-seed evidence, current-sector limitation,
and unequal effective exposure must be addressed by a separately specified
experiment before it is used to claim transfer benefit.

## Final verification

- Complete test suite: **523 passed, 2 skipped** in 23.01 seconds. The skips
  are hardware-gated tests; there were no failures.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: no broken requirements found.
- Executed Notebook 07: 11/11 code cells executed in order, with zero error
  outputs.
- Installed compatibility set: Stable-Baselines3 2.9.0, sb3-contrib 2.9.0,
  PyTorch 2.13.0, and Gymnasium 1.3.0 under Python 3.11.9.
- Production registry SHA-256 remained
  `e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a`.
- Production model roots still contain only their existing `.gitkeep` files.
- No HTTP request, TEST partition load/evaluation, production persistence,
  registry migration, promotion, raw/backfill mutation, or commit occurred.
