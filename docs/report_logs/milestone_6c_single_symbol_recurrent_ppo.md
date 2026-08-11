# Milestone 6C — Single-Symbol RecurrentPPO Baseline

Execution timestamp: 2026-08-12 00:35:42 PKT (Asia/Karachi)

Code-base commit: `9779484c272e3df3795db9c53779a70beed81cb6`

Research symbol/seed: MCB / 42

## Objective and safety boundary

Milestone 6C proves a production-safe, single-symbol recurrent architecture
using `sb3-contrib` RecurrentPPO and `MlpLstmPolicy`. It adds an independent
trainer/evaluator path over `rl_recurrent_partition_v1` and compares a staged
MCB run with an equal-timestep `ppo_single_symbol_v1` MLP baseline.

This is architecture validation, not final model selection. It performs no
sector pretraining, cross-symbol transfer, bulk training, TEST evaluation,
hyperparameter tuning, registry write, production persistence, or promotion.
The existing MLP trainer/evaluator/persistence and `rl_partition_v1` were not
refactored or behaviorally changed.

## Dependency integration

The exact dependency `sb3-contrib==2.9.0` was declared and installed only in:

```text
/Users/m.abdulbasit/Downloads/virtual-trader/.venv/bin/python
```

The resolution dry run showed that all existing versions already satisfied the
new wheel; no installed dependency was upgraded or downgraded.

| Package/runtime | Version |
|---|---|
| Python | 3.11.9 |
| sb3-contrib | 2.9.0 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.13.0 |
| Gymnasium | 1.3.0 |

`pip check` reports no broken requirements.

## Recurrent architecture

Trainer/config version: `recurrent_ppo_single_symbol_v1`

| Setting | Value |
|---|---:|
| Algorithm | RecurrentPPO |
| Policy | MlpLstmPolicy |
| Learning rate | 0.0003 |
| Rollout steps | 512 |
| Batch size | 64 |
| Epochs/update | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.20 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.50 |
| Maximum gradient norm | 0.50 |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Shared actor/critic LSTM | No |
| Critic LSTM | Yes, separate |
| Shared feature extractor | Yes |
| Post-LSTM policy head | `[64]` |
| Post-LSTM value head | `[64]` |
| Activation | Tanh |
| Orthogonal initialization | Yes |
| Default device | CPU |

The installed sb3-contrib policy source explicitly forbids combining
`shared_lstm=True` with `enable_critic_lstm=True`. The baseline therefore uses
separate actor and critic LSTMs (`shared_lstm=False`,
`enable_critic_lstm=True`) and validates invalid combinations before model
construction. No sb3-contrib architecture default is relied upon silently.

## Data and training contract

The recurrent trainer calls only the canonical
`load_recurrent_partition(symbol, "train")` API. It rejects missing/stale
recurrent contracts, incompatible environment/feature/reset semantics, and
anything other than a Mature, independent-training-ready symbol.

MCB source:

| Partition | Rows | Start | End | Use |
|---|---:|---|---|---|
| TRAIN | 1,704 | 2016-10-06 | 2023-08-23 | Gradient updates only |
| VALIDATION | 365 | 2023-08-24 | 2025-02-12 | Research evaluation only |
| TEST | 366 metadata rows | 2025-02-13 | 2026-08-05 | Sealed, not loaded |

The environment remains `single_symbol_env_v1`: `(17,)` float32 observations,
real unscaled execution OHLCV, scaled ordered market features, dynamic
portfolio features, deterministic seeding, and unchanged next-open execution
semantics.

## Hidden-state and episode-start handling

### Training

RecurrentPPO receives the existing full-partition Gymnasium environment through
one `DummyVecEnv`. At the first observation, SB3 has
`_last_episode_starts=True`. During `collect_rollouts`, it copies the previous
LSTM state and carries `_last_lstm_states` into the next rollout call. A normal
512-step buffer/update boundary therefore does not reset state.

Instrumentation on the 10,240-step run observed:

- 20 rollout buffers;
- 19 state-continuity checks across rollout/update boundaries;
- 19/19 continuity checks passed;
- six actual full-TRAIN environment completions/resets;
- first episode-start flag `True`;
- later rollout starts included `False`, proving rollout boundary != episode
  boundary.

### VecEnv completion

At full-episode termination, `DummyVecEnv` places the true final observation in
`info["terminal_observation"]`, resets the environment, and returns the new
episode's first observation with `done=True`. RecurrentPPO assigns that `done`
array to `_last_episode_starts`, so the next policy call zeros the recurrent
state. The terminal observation is retained for terminal/truncation accounting
and is not mistaken for the next episode's first observation.

### Validation

The recurrent evaluator does not use generic `evaluate_policy`. It explicitly:

1. resets the validation environment;
2. initializes `lstm_state=None`;
3. calls the first prediction with `episode_start=[True]`;
4. passes the returned state into every later prediction with
   `episode_start=[False]`;
5. stops immediately on validation termination/truncation instead of accepting
   an auto-reset second cycle.

The executed MCB validation propagated recurrent state for 363 subsequent
steps, recorded exactly one episode reset, started with `True`, supplied no
TRAIN state, and preserved the policy hash and model timestep counter.

Every evaluator invocation initializes state independently, and a symbol
mismatch is rejected, preventing hidden-state carry between TRAIN/VALIDATION or
between symbols.

## 512-step production-data smoke

The executed Notebook 06 smoke used CPU, MCB, seed 42, and 512 timesteps:

| Model | Actual timesteps | Parameters | Training duration |
|---|---:|---:|---:|
| RecurrentPPO | 512 | 51,076 | 1.856 s |
| MLP PPO | 512 | 10,884 | 0.524 s |

Recurrent training and the complete 365-row VALIDATION episode completed.
State propagation/reset checks passed. This smoke establishes integration only;
its portfolio result is not evidence of model quality.

After the complete test suite, a separate final 512-step RecurrentPPO smoke
completed in 1.982 seconds using all 1,704 MCB TRAIN rows. Its validation run
used all 365 VALIDATION rows, initialized a new hidden state, propagated that
state for 363 subsequent prediction steps, recorded exactly one validation
reset, and left policy parameters and the model timestep counter unchanged.
The single-rollout 512-step smoke has no cross-rollout boundary to inspect;
cross-rollout continuity is instead established by the 19/19 checks in the
10,240-step run and by deterministic offline tests.

## Isolated CPU versus MPS recurrent benchmark

Both device arms ran in separate child processes with identical MCB TRAIN data,
seed, policy/configuration, a 512-step same-device warm-up, synchronized timing,
and 5,120 timed environment steps. Silent MPS-to-CPU fallback was disabled.

| Device | Actual device | Status | Duration | Throughput | Warning |
|---|---|---|---:|---:|---|
| CPU | `cpu` | Completed | 11.080 s | 462.080 steps/s | None |
| MPS | `mps:0` | Completed | 38.207 s | 134.007 steps/s | None |

The recorded CPU-duration/MPS-duration ratio was 0.290. Expressed directly,
CPU was approximately **3.45× faster** than MPS for this RecurrentPPO workload.
Both arms had 51,076 parameters and matching contract/config/data hashes. CPU is
therefore the recommended recurrent device. This conclusion is specific to the
measured MCB/architecture/timestep workload, not a universal LSTM benchmark.

## 10,240-step MCB comparison

Both learning algorithms used CPU, MCB, seed 42, identical TRAIN/VALIDATION
partitions, and exactly 10,240 environment timesteps. No parameters were tuned
after observing validation.

| Model | Parameters | Duration | Timesteps | Final update KL | Updates |
|---|---:|---:|---:|---:|---:|
| RecurrentPPO | 51,076 | 21.809 s | 10,240 | 0.014221 | 200 |
| MLP PPO | 10,884 | 5.024 s | 10,240 | 0.007383 | 200 |

Equal environment timesteps do not mean equal parameter count, wall-clock
compute, network memory, optimization difficulty, or effective recurrent
sequence computation. RecurrentPPO used about 4.69× as many parameters and
about 4.34× the observed training time.

### Validation metrics

| Strategy | Final value | Total return | Annualized return | Volatility | Sharpe | Sortino | Max drawdown | Trades | Costs | Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RecurrentPPO | 2,457,645.67 | 145.76% | 86.36% | 25.55% | 2.565 | 4.823 | 11.50% | 23 | 66,112.98 | 80.49% |
| MLP PPO | 1,298,314.44 | 29.83% | 19.81% | 21.52% | 0.947 | 1.531 | 10.61% | 37 | 62,929.71 | 65.93% |
| Buy and Hold | 1,975,057.51 | 97.51% | 60.19% | 30.51% | 1.697 | 2.704 | 13.98% | 1 | 1,498.20 | 100.00% |
| Always Hold | 1,000,000.00 | 0.00% | 0.00% | 0.00% | Not available | Not available | 0.00% | 0 | 0.00 | 0.00% |
| Random (seed 42) | 1,706,240.77 | 70.62% | 44.76% | 24.24% | 1.647 | 2.824 | 12.52% | 124 | 245,590.54 | 48.35% |

Additional P&L:

| Strategy | Realized P&L | Final unrealized P&L |
|---|---:|---:|
| RecurrentPPO | 1,526,884.54 | -69,238.87 |
| MLP PPO | 325,020.97 | -26,706.52 |
| Buy and Hold | 0.00 | 975,057.51 |
| Always Hold | 0.00 | 0.00 |
| Random | 706,240.77 | 0.00 |

### Interpretation

For this one MCB/seed-42 validation run, RecurrentPPO produced higher return and
risk-adjusted ratios than the equal-timestep MLP candidate and Buy and Hold,
while MLP PPO had slightly lower maximum drawdown. This is an observed
validation result only. It does **not** establish that recurrent PPO is better:
a defensible FYP comparison requires predeclared multi-seed, multi-symbol
experiments and uncertainty reporting. Hyperparameters were not changed after
viewing this outcome.

## Training diagnostics

The callback captures only finite metrics actually exposed by the final
sb3-contrib/SB3 logger update. The recurrent 10,240-step run reported:

- approximate KL: 0.0142208;
- clip fraction: 0.116016;
- entropy loss: -0.749043;
- explained variance: 0.178934;
- policy-gradient loss: -0.0188678;
- value loss: 0.0006860;
- learning rate: 0.0003;
- updates: 200.

Unavailable values are preserved as unavailable; none are fabricated.

## Temporary persistence audit

Production PPO persistence remains standard-PPO-specific and was not made to
silently load recurrent archives. A separate temporary-only compatibility
helper:

- refuses paths under the project/production model roots;
- writes `recurrent_model.zip` through `RecurrentPPO.save()`;
- reloads only through `RecurrentPPO.load()`;
- verifies the exact recurrent policy class;
- verifies hidden size, layer count, shared/critic layout and parameter count;
- checks deterministic action and recurrent-state equality from a fresh
  `episode_start=True` prediction;
- confirms the registry bytes are unchanged;
- relies on `TemporaryDirectory` cleanup.

Notebook 06 verified 51,076 saved/reloaded parameters, matching architecture,
action, and recurrent state; the temporary directory was deleted.

## Registry compatibility roadmap

`model_registry_v2` remains unchanged and contains no recurrent row. A future
v3 migration should add explicit fields for:

- recurrent algorithm and policy;
- recurrent contract/trainer versions;
- LSTM hidden size and layer count;
- actor/critic sharing configuration;
- parent/pretrained model ID;
- transfer lineage;
- sector/universe scope and hash;
- normalization scope;
- recurrent bundle schema/loader identity.

Promotion semantics must not be introduced merely by adding these columns.

## Notebook 06

`notebooks/06_recurrent_ppo_baseline.ipynb` was executed end to end with the
project kernel: 10/10 code cells executed sequentially and produced zero error
outputs. It uses production contract/trainer/evaluator/comparison/persistence
APIs, executes a short smoke comparison, displays the measured isolated device
and 10,240-step evidence, demonstrates TEST guards, and performs a fully cleaned
temporary round trip.

## TEST seal and integrity

Partition tracing for the 10,240-step research run was exactly:

```text
recurrent trainer -> train
MLP trainer       -> train
recurrent evaluator -> validation
MLP evaluator       -> validation
```

The pandas file guard recorded zero `test.csv` or `test_rl.csv` reads. TEST
contributed only its recurrent-contract row/date metadata. The benchmark never
invoked an evaluator. Temporary persistence loaded VALIDATION only for one
deterministic prediction.

No live HTTP request occurred. No raw/backfill/index/dashboard logic changed.
No production model was saved, registered, or promoted. No model was retained
after the in-memory research processes exited.

## Final verification

The final verification was run with the project virtual environment:

- `.venv/bin/python -m pytest -v`: **494 passed, 2 skipped** in 13.57 seconds;
- the skips are the pre-existing MLP and new recurrent real-MPS tests when MPS
  is unavailable to the sandboxed pytest process;
- focused recurrent suite: **16 passed, 1 skipped**;
- `.venv/bin/python -m pip check`: **No broken requirements found**;
- `git diff --check`: passed;
- executed Notebook 06: 10/10 code cells, zero error outputs;
- final smoke partition trace: TRAIN, VALIDATION, VALIDATION for the temporary
  reload proof; no TEST partition;
- final temporary bundle: architecture, deterministic action, and recurrent
  state all matched after reload, then the directory was removed;
- production registry and saved-model tree hashes matched before and after the
  final smoke.

The worktree remains uncommitted by design.

## Limitations and next milestone

- One symbol and one seed cannot establish architectural superiority.
- 10,240 timesteps is a moderate integration experiment, not converged model
  selection.
- Validation has now been observed and must not be repeatedly tuned against in
  the same experiment definition.
- TEST remains sealed for a later predeclared final evaluation.
- Sector cohort construction, pooled TRAIN-only normalization, sector recurrent
  pretraining, Cold Start transfer, and lineage-aware registry/persistence are
  explicitly deferred.

The next milestone is sector-universe construction and recurrent pretraining
design, followed by multi-seed/multi-symbol experiments before any final TEST
evaluation.
