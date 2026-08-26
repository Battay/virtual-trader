# Milestone 7C.3e — Full-Universe Recurrent Run Preparation

Audit date: 2026-08-27 (Asia/Karachi)

## Outcome

The software and deterministic run contract are **ready for an explicit CUDA
benchmark**. Full-universe training is not authorized. The two deliberate open
gates are (1) a successful benchmark on real NVIDIA CUDA hardware and (2) a
predeclared final training-budget decision.

No agent was trained, no validation or TEST market values were loaded, and no
production model or registry entry was created.

## Frozen candidate training configuration

Specification version: `recurrent_full_universe_run_spec_v1`

| Field | Candidate value |
|---|---|
| Trainer/config | `recurrent_ppo_single_symbol_v1` |
| Algorithm / implementation | `sb3_contrib.RecurrentPPO` 2.9.0 |
| Policy | `MlpLstmPolicy` |
| Environment | `single_symbol_env_v1` |
| Artifact contracts | Compatible symbol-scoped `rl_recurrent_partition_v1` or `rl_recurrent_train_partition_v2`, recorded per job |
| Features | 12 ordered, scaled market features plus 5 dynamic portfolio features |
| Observation shape | `(17,)` |
| Learning rate | `3e-4` |
| Rollout length | 512 |
| Batch size | 64 |
| Epochs per update | 10 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip range | 0.20 |
| Entropy / value coefficients | 0.01 / 0.50 |
| Maximum gradient norm | 0.50 |
| LSTM | 64 units, one layer, separate actor path, critic LSTM enabled |
| Network / activation | `[64]`, Tanh, orthogonal initialization |
| Seed policy | Fixed seed 42 for each independent symbol job |
| Candidate budget | 100,000 requested timesteps; cost-planning candidate, not research-frozen |
| Candidate device / workers | Explicit CUDA / one worker |
| Validation | Complete VALIDATION-only evaluation after training when available; TEST sealed |
| Models | Isolated per symbol and attempt, atomic persistence, no silent overwrite |
| Checkpoints/resume | No true PPO-state checkpoint; explicit restart from timestep zero |

The dependency fingerprint records Python 3.11.9, PyTorch 2.13.0,
Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, and Gymnasium 1.3.0.

## Timestep-budget decision support

The prior PIAHCLA CPU benchmark measured the unchanged architecture as follows.
The budgets are cost points, not evidence that one budget is statistically or
economically optimal.

| Requested budget | Actual PPO timesteps | One-agent CPU time | Throughput | Sequential 435-agent estimate |
|---:|---:|---:|---:|---:|
| 50,000 | 50,176 | 110.116 s | 455.66 steps/s | 13.31 h |
| 100,000 | 100,352 | 210.868 s mean over two repeats | 475.98 steps/s mean | 25.48 h |
| 250,000 | 250,368 | 524.036 s | 477.77 steps/s | 63.32 h |

The 100k repeat coefficient of variation was 1.30%. One symbol and one seed
cannot establish a universal convergence budget. In particular, observed
validation-return differences would confound symbol, seed, and training
variance and must not be used to choose the full-run budget after the fact.

Recommendation: retain **100k as the provisional capacity-planning budget**,
then predeclare a small, stratified multi-symbol/multi-seed convergence-budget
study over 50k/100k/250k. Freeze its diagnostics and decision rule before
examining results. Full-universe execution remains blocked until that decision
is recorded. TEST is not part of budget selection.

## Materialized 508-identity dry-run plan

The dry-run plan uses the frozen identity and current artifact inventory.

| State | Count |
|---|---:|
| `QUEUED` / trainable | 435 |
| `INELIGIBLE` / no recent trading activity | 53 |
| `INELIGIBLE` / legacy insufficient usable history | 13 |
| `INELIGIBLE` / Cold Start, not independent training | 2 |
| `INELIGIBLE` / insufficient under canonical TRAIN-only v2 policy | 5 |
| **Total** | **508** |

Among eligible jobs, 432 use `rl_recurrent_partition_v1` and three use
`rl_recurrent_train_partition_v2`. Every row records symbol, eligibility and
reason, contract/feature/environment versions, source contract hash, candidate
timesteps, seed, requested device, initial status, and unique model,
checkpoint, and validation paths.

- Universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`
- Source inventory hash: `4e59e97bf00cb5d39b5b98deea96657cf724de14e549ce68fd708e63356ab44d`
- Plan content hash: `a5972d9ebb0f834f96ca54c1921fe7ccc385b9067959fbbb17e7a6e3ee0c6318`
- Candidate run ID: `rppo-symbols-5756b2b952aa357c4ae3`
- Specification fingerprint: `c1c3ca51a5806750f6bb262392c9e6c2f32238034ddae76f8675ed9500f0fba6`
- Execution authorized in plan: `false`

## CUDA benchmark contract

Representative symbol: **PIAHCLA**, retained from 7C.3a. It was selected by a
deterministic TRAIN-only high-history/high-coverage data-quality medoid rule;
runtime, validation returns, and model performance were not selection inputs.

On this Mac, PyTorch reports `torch.cuda.is_available() == False` and
`torch.version.cuda == None`. The preflight correctly terminated with:

> CUDA was explicitly requested but is unavailable; CPU fallback is disabled
> for explicit CUDA requests.

No CUDA timing is reported or inferred. On an NVIDIA host, use:

```bash
.venv/bin/python -m reinforcement_learning.training.cuda_benchmark \
  --preflight

.venv/bin/python -m reinforcement_learning.training.cuda_benchmark \
  --run --device cuda --symbol PIAHCLA --seed 42 \
  --warmup-timesteps 512 --budgets 50000 100000 250000 \
  --workers 1 --output-json cuda_benchmark_workers_1.json
```

The isolated worker verifies the effective model/tensor device is CUDA and
records GPU name/count/memory, compute capability, CUDA and PyTorch versions,
requested and actual timesteps, wall time, steps/second, peak process RSS,
peak allocated CUDA memory, parameter count, TRAIN provenance, and contract
version. A worker that exits, times out, reports CPU, or cannot produce valid
telemetry fails closed.

## Bounded CUDA concurrency contract

After the one-worker benchmark succeeds, repeat the same fixed 50k/100k
workload with `--workers 2`, then `--workers 4`. The implementation starts
isolated same-budget workers concurrently and records aggregate throughput,
per-worker timings, memory, and any failure/OOM. CUDA utilization and thermal
state are explicitly reported as unavailable unless obtained safely; no
optional monitoring dependency was added.

Do not proceed to the next worker count after instability or unsafe memory
pressure. Worker count is frozen only after evidence; `2 != 2x` and `4 != 4x`
are explicit assumptions. One worker remains the candidate default.

## Storage estimate

There are currently no saved recurrent `.zip`, `.pt`, or `.pth` model or
checkpoint files to measure in the repository. The estimate therefore uses a
conservative 1 MiB final archive allowance for the existing 51,076-parameter
architecture, plus 256 KiB validation output, 64 KiB job state, and 5 MiB logs
per applicable job. Checkpoint copies are zero because true continuation is
not implemented.

| Component | Conservative amount |
|---|---:|
| 435 final models | 435.00 MiB |
| Validation results | 108.75 MiB |
| 508 job-state records | 31.75 MiB |
| Logs | 2,175.00 MiB |
| Optional checkpoints | 0 MiB |
| **Expected total** | **2,750.50 MiB (2.686 GiB)** |
| Required safety capacity (2x, minimum 5 GiB) | 5.372 GiB |
| Current free disk | 29.922 GiB |
| Margin above requirement | 24.550 GiB |

The current storage gate passes. It must be recomputed on the actual training
host immediately before creating a run. If durable checkpoints are later
implemented, their measured size and retention count must be added first.

## Interruption and recovery

- Completed compatible job: verify contract/model hashes and skip.
- Failed job: retain error; retry only by explicit requeue.
- Interrupted job: mark `INTERRUPTED`; restart from timestep zero only.
- Stale job: mark explicitly and require recovery/requeue under the new contract.

The system does not persist optimizer, RNG, rollout buffer, environment, and
recurrent hidden state together, so it does not claim true resume. At the
candidate one-worker budget, an interruption can waste at most the unfinished
portion of one 100k job (about 3.5 CPU minutes from current evidence; CUDA cost
is unknown). With 2/4 workers, up to 2/4 in-flight jobs can be lost. This is
acceptable for the current short independent-job design, conditional on the
actual CUDA benchmark. It must be revisited if per-job duration becomes large.

## Progress demonstration

Initial dry-run state:

| Metric | Value |
|---|---:|
| Total identities | 508 |
| Trainable / queued | 435 / 435 |
| Ineligible | 73 |
| Training / validating / completed | 0 / 0 / 0 |
| Failed / interrupted | 0 / 0 |
| Overall progress | 0.00% |
| Effective device | Not assigned |
| Elapsed | 0 s |
| ETA | Not available |

The existing job status table supplies per-symbol requested/completed
timesteps, percentage, device, start time, elapsed time, validation status,
model availability, error, and an ETA only after observed nonzero progress.

## Full-run safety gates

| Gate | Current status |
|---|---|
| 508/508 identities accounted | PASS |
| 435 trainable jobs reproducibly materialized | PASS |
| 73 explicit exclusions | PASS |
| Source/universe hashes present and frozen | PASS |
| TEST sealed | PASS |
| Storage safe | PASS |
| Explicit device policy | PASS |
| CUDA benchmark for a CUDA run | **OPEN** |
| Worker count benchmarked if greater than one | PASS for candidate one worker; future 2/4 OPEN |
| Isolated unique model paths | PASS |
| No incompatible stale run | PASS; dry-run created no persisted run store |
| Training budget frozen | **OPEN** |
| Validation policy frozen | PASS |

Because the two open gates are intentionally future decisions, the dry plan's
`execution_authorized` value is false. This milestone does not authorize a
call to the orchestrator's execution path.

## Artifacts

- `docs/report_logs/milestone_7c3e_full_run_plan/full_universe_run_plan.csv`
- `docs/report_logs/milestone_7c3e_full_run_plan/full_universe_run_spec.json`
- `docs/report_logs/milestone_7c3e_full_run_plan/full_universe_storage_estimate.json`

## Verification

- Complete test suite: **689 passed, 2 skipped in 26.87 seconds**.
- Dependency check: **No broken requirements found**. The pip cache ownership
  warning is environmental and did not affect dependency validation.
- `git diff --check`: passed with no whitespace errors.
- CUDA preflight: failed closed as designed; no training subprocess started.
- Production model directories: no recurrent model archive was created.
- Model registry and source Parquet: unchanged.
- Commit: none.

Files added by 7C.3e:

- `reinforcement_learning/training/full_universe_run.py`
- `reinforcement_learning/training/cuda_benchmark.py`
- `data_pipeline/tests/test_full_universe_run.py`
- `data_pipeline/tests/test_cuda_benchmark_contract.py`
- this report and the three planning artifacts listed above

## Decision

**READY_TO_BENCHMARK_CUDA**

This means only that the software/run contract is ready for measurement on
real NVIDIA hardware. Full-universe training remains unauthorized until CUDA
and any requested concurrency are benchmarked and the training budget is
frozen under a predeclared methodology.
