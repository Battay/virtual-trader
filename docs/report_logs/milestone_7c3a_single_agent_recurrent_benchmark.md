# Milestone 7C.3a — Single-Agent RecurrentPPO + LSTM Benchmark

Benchmark date: **2026-08-26 (Asia/Karachi)**

Status: **COMPLETED ON CPU; MPS REJECTED AS UNRELIABLE FOR THIS RUN**

## Scope and conclusion

This is a compute benchmark of one existing single-symbol recurrent execution
agent. It does not redesign soft clustering, train a soft-group agent, compare
model quality, select hyperparameters, persist a model, or use TEST.

On this MacBook Air M2, the validated CPU path took approximately:

- **3.5 minutes for 100,000 requested timesteps** (100,352 actual);
- **8.7 minutes for 250,000 requested timesteps** (250,368 actual).

For conservative sequential planning, use **about 3.6 minutes per 100k agent**
or **8.8 minutes per 250k agent** under the measured single-symbol architecture.
These figures do not prove that a future multi-stock/group agent will have the
same per-step cost.

## Existing recurrent architecture inspected

The benchmark reused the production path without trainer changes:

- trainer: `recurrent_ppo_single_symbol_v1`;
- algorithm: `sb3_contrib.RecurrentPPO`;
- policy: `MlpLstmPolicy`;
- environment: `single_symbol_env_v1`, `Discrete(3)`, observation `(17,)`;
- contract: `rl_recurrent_partition_v1` over canonical `rl_partition_v1`;
- training loader: canonical recurrent `train` partition only;
- evaluator: deterministic complete `validation` episode with fresh recurrent
  state, parameter/timestep immutability checks, and no TEST loader;
- device handling: explicit CPU/MPS resolution, model and tensor-device
  verification, no silent explicit-MPS fallback, and synchronized MPS timing;
- persistence: none. Every model remained in memory and was discarded when its
  isolated process exited.

Runtime dependencies were Python 3.11.9, PyTorch 2.13.0,
Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, and Gymnasium 1.3.0.

## Representative-symbol rule

Selection used no runtime, reward, validation metric, or economic-return
information.

1. Intersect the 508 official-listing-backed current common equities with valid
   Mature recurrent contracts.
2. Load only canonical TRAIN partitions.
3. Require TRAIN rows and active-span coverage at or above their respective
   75th percentiles.
4. Require zero-OHL and zero-volume ratios at or below their medians, with zero
   non-positive closes and zero negative-volume rows.
5. Among the qualified cohort, select the deterministic medoid of TRAIN rows,
   active-span coverage, log median volume, zero-volume ratio, and zero-OHL
   ratio after IQR scaling; break exact ties by symbol.

There were 432 authoritative common-equity/recurrent-contract candidates and
83 qualified high-history/high-coverage candidates. The deterministic result
was **PIAHCLA — PIA Holding Company Limited**.

Selected TRAIN evidence:

| Field | Value |
|---|---:|
| TRAIN rows | 1,704 |
| TRAIN dates | 2016-10-06 through 2023-08-23 |
| Active-span coverage | 100.00% |
| Median TRAIN volume | 680,000 |
| Zero-volume ratio | 0.00% |
| Zero-OHL ratio | 0.00% |
| Non-positive close rows | 0 |
| Negative-volume rows | 0 |

The registry and current listing snapshot classify the symbol as a currently
listed ordinary equity. No local registry alias/corporate-action flag is set.

## Fixed configuration

All valid runs used the same symbol, seed, contract, environment, scaler,
policy architecture, and optimizer configuration.

| Setting | Value |
|---|---:|
| Configuration version | `recurrent_ppo_single_symbol_v1` |
| Seed | 42 |
| Learning rate | `3e-4` |
| Rollout length (`n_steps`) | 512 |
| Batch size | 64 |
| Epochs per update | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.20 |
| Entropy coefficient | 0.01 |
| Value-function coefficient | 0.50 |
| Maximum gradient norm | 0.50 |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Shared LSTM | False |
| Critic LSTM | Enabled |
| Shared feature extractor | True |
| Actor/critic `net_arch` | `[64]` / `[64]` |
| Activation | Tanh |
| Orthogonal initialization | Enabled |
| Trainable/non-trainable parameters | 51,076 |

Each isolated process first ran an untimed, same-device 512-step warm-up. The
reported wall-clock time covers the complete subsequent trainer call: contract
and TRAIN loading, environment validation, model construction, and learning.
The warm-up and optional validation evaluation are excluded. Peak RSS is the
isolated Python process high-water mark and is not a pure model-memory figure.

Because PPO collects complete 512-step rollouts, SB3 rounded the requested
budgets upward. Requested and actual values are both reported.

## CPU benchmark results

| Run | Requested | Actual | Wall clock | Throughput | Peak RSS | Effective device |
|---|---:|---:|---:|---:|---:|---|
| 50k | 50,000 | 50,176 | 110.116 s (1m 50.1s) | 455.66 steps/s | 466.3 MiB | `cpu` |
| 100k repeat 1 | 100,000 | 100,352 | 213.610 s (3m 33.6s) | 469.79 steps/s | 463.5 MiB | `cpu` |
| 100k repeat 2 | 100,000 | 100,352 | 208.127 s (3m 28.1s) | 482.17 steps/s | 457.7 MiB | `cpu` |
| 250k | 250,000 | 250,368 | 524.036 s (8m 44.0s) | 477.77 steps/s | 466.2 MiB | `cpu` |

The two valid same-seed 100k repeats had:

- mean duration: **210.868 seconds**;
- population standard deviation: **2.742 seconds**;
- coefficient of variation: **1.30%**;
- range: **5.483 seconds**, or **2.60%** of the mean;
- mean throughput: approximately **475.98 steps/s**.

The deterministic CPU repeats produced identical final diagnostics and
validation results. Runtime scaled approximately linearly: the 250k runtime
normalized to 100k was 209.6 seconds, close to the measured 210.9-second 100k
mean.

One earlier 100k training process completed learning but its ad-hoc reporting
wrapper raised an `AttributeError` while serializing post-training validation
metrics. It wrote no artifact and was excluded because it did not emit a valid
timing record. The wrapper was corrected; the trainer/configuration were not
changed.

## Final SB3 training diagnostics

These are the final logger snapshot after the final PPO update, not averages
over the whole run and not convergence claims.

| Metric | 50k | 100k (both repeats) | 250k |
|---|---:|---:|---:|
| Updates | 980 | 1,960 | 4,890 |
| Approximate KL | 0.015184 | 0.027360 | 0.059002 |
| Clip fraction | 0.116992 | 0.145898 | 0.171289 |
| Entropy loss | -0.449712 | -0.428891 | -0.381813 |
| Explained variance | 0.343615 | 0.830819 | 0.865004 |
| Policy-gradient loss | -0.014104 | -0.017383 | -0.020425 |
| Value loss | 0.001620 | 0.001980 | 0.002603 |
| Learning rate | 0.0003 | 0.0003 | 0.0003 |

Recurrent-state instrumentation passed in every valid run:

- first episode start was `True`;
- 50k: 97/97 rollout-continuity checks passed;
- each 100k repeat: 195/195 checks passed;
- 250k: 488/488 checks passed;
- rollout boundaries did not reset hidden state unless the environment episode
  actually ended.

## Validation-only observations

The existing recurrent evaluator was used after each valid training run. It
loaded only the complete 365-row VALIDATION episode from 2023-08-24 through
2025-02-12, initialized a fresh recurrent state, and verified that model
parameters and timestep state were unchanged. TEST was not loaded.

| Requested training budget | Total return | Sharpe | Sortino | Max drawdown | Trades | Exposure |
|---|---:|---:|---:|---:|---:|---:|
| 50k | 51.06% | 0.887 | 1.396 | 45.65% | 5 | 28.85% |
| 100k | 7.92% | 0.359 | 0.545 | 46.50% | 10 | 40.66% |
| 250k | 87.88% | 1.135 | 1.765 | 38.82% | 40 | 30.49% |

These single-symbol, single-seed validation values are included only because
the existing safe evaluator was available. Their non-monotonic behavior is a
specific warning that more timesteps do not by themselves prove convergence or
model quality. They were not used to alter configuration or choose a budget.

## Apple MPS result

PyTorch reports MPS built and available on the actual Apple M2 when run outside
the restricted sandbox. The explicit MPS path completed the 512-step warm-up,
passed the trainer's no-silent-fallback device checks, and began the synchronized
50k timed run. At approximately 5,000 steps, the process aborted in
MetalPerformanceShaders with an `MPSNDArrayDescriptor` slicing assertion.

Therefore:

- there is **no valid MPS timing** for this milestone;
- the failed arm is not reported as a CPU fallback or partial success;
- 100k and 250k MPS runs were not attempted because the current long-run MPS
  path was demonstrably unreliable;
- CPU remains the only supported recommendation for this experiment.

This is consistent with, but stronger than, the earlier 5,120-step MCB device
benchmark: that shorter run completed on MPS but found CPU approximately 3.45x
faster. The present result shows that current long-run MPS reliability is also
insufficient.

## Conservative sequential scaling estimates

The 100k column uses the slower valid repeat, 213.610 seconds per agent. The
250k column uses the measured 524.036 seconds per agent. These are sequential
arithmetic estimates, not promises.

| Agents | 100k per agent | 250k per agent |
|---:|---:|---:|
| 10 | 35m 36s | 1h 27m 20s |
| 15 | 53m 24s | 2h 11m 01s |
| 20 | 1h 11m 12s | 2h 54m 41s |
| 25 | 1h 29m 00s | 3h 38m 21s |
| 30 | 1h 46m 48s | 4h 22m 01s |
| 50 | 2h 58m 01s | 7h 16m 42s |
| 500 | 29h 40m 05s | 72h 46m 58s |

The exact answer to give the supervisor is:

> On the measured MacBook Air M2 CPU, one existing 51,076-parameter
> RecurrentPPO agent with a one-layer, 64-unit LSTM takes about 3.5 minutes for
> 100k requested timesteps and about 8.7 minutes for 250k. For conservative
> sequential planning, budget roughly 3.6 and 8.8 minutes per agent,
> respectively. Fifteen agents are about 53 minutes at 100k each or 2 hours 11
> minutes at 250k each; 30 agents are about 1 hour 47 minutes or 4 hours 22
> minutes. These figures assume the future agent retains the same per-step
> single-symbol environment cost.

## Parallelization caveats

Dividing the sequential totals by two or four gives only an unattainable-or-
best-case arithmetic lower bound unless scaling is measured. Actual concurrent
training will contend for performance cores, memory bandwidth, Python/Gymnasium
environment work, and the fanless M2 thermal envelope.

Measured peak process RSS was about 458–466 MiB, so two or four isolated workers
also multiply process/model/data overhead before accounting for shared library
pages and OS pressure. MPS is not a viable parallel accelerator from this run.
A dedicated concurrency benchmark is required before claiming 2x or 4x speedup.

## Safety and integrity

- Learning loaded only `train` through the canonical recurrent loader.
- Evaluation loaded only `validation` and supplied no TRAIN hidden state.
- TEST observations/returns were never loaded or evaluated.
- No model was saved, registered, promoted, or written to a benchmark path.
- Production model directories and `model_registry.csv` remained unchanged.
- Source Parquet and processed RL artifacts were read-only.
- No hyperparameter changed between budgets or after viewing validation.
- No live HTTP request was made.

## Verification

Final verification commands and results are recorded after the benchmark:

- `.venv/bin/python -m pytest -q`: **645 passed, 2 skipped in 25.78s**;
- `.venv/bin/python -m pip check`: **No broken requirements found** (pip
  disabled its unwritable user-cache directory; this did not affect checking);
- `git diff --check`: **passed**;
- `git status --short`: only this untracked benchmark report.

No commit was created.
