# Milestone 7C.3g — Safe Parallel Recurrent Training Orchestrator

Date: 2026-08-28  
Branch: `feat/rl-environment-v1`  
Decision: **READY_CPU_PARALLEL_BENCHMARK**

## Outcome

The existing persistent per-symbol recurrent orchestrator now supports bounded
CPU process concurrency at 1, 2, or 4 workers. The default remains the original
one-worker sequential implementation. `--max-jobs` still limits how many queued
jobs an invocation may process; the new `--workers` option controls concurrent
processes and does not change that meaning.

This milestone ran only a 512-timestep qualification. It did not start the
435-symbol training run, did not load TEST, did not change PPO/LSTM
hyperparameters, and did not publish a registry or production model artifact.

## Existing contract preserved

The following versions and semantics remain unchanged:

- job schema `recurrent_training_job_v1`;
- run schema `recurrent_training_run_v1`;
- orchestrator/model identity `recurrent_multi_symbol_orchestrator_v1`;
- `QUEUED → TRAINING → VALIDATING → COMPLETED`;
- explicit `FAILED`, `INTERRUPTED`, `STALE`, and `INELIGIBLE` states;
- restart from zero after interruption—never false optimizer-checkpoint resume;
- attempt-specific model, validation, checkpoint, and log paths;
- TRAIN-only optimization and VALIDATION only after successful TRAIN;
- TEST sealed.

The v1 run fingerprint retains its historical `worker_limit=1` field so old run
identities remain reproducible. Runtime process concurrency is deliberately
invocation metadata rather than model identity. Each process invocation has a
parent-written `logs/invocations/*.json` record containing worker count, thread
policy, selected symbols, terminal job statuses, registry-integrity result, and
`test_partition_loaded=false`.

## Process and state ownership

The implementation uses Python's `spawn` process context. Each worker receives
only one symbol, the frozen recurrent configuration, the canonical split root,
an isolated temporary workspace, and a validation flag. It never receives the
run manifest, job JSON path, registry path, production model path, or TEST
partition/path.

Workers may write only:

```text
<run>/workspaces/<symbol>/attempt_NNN/model.zip
<run>/workspaces/<symbol>/attempt_NNN/validation.json
```

The parent is the sole writer of:

- job state/progress JSON;
- active-worker state;
- invocation logs;
- final per-symbol logs;
- model and validation promotion into isolated run paths.

Before promotion, the parent validates the worker protocol, symbol, PID,
terminal status, requested/actual timesteps, explicit CPU device, configured
thread policy, model filename/hash, validation reference, and sealed-TEST flag.
Artifact promotion uses `os.replace`, refuses existing destinations, and rolls
back promoted files if finalization fails.

A nonblocking run-level execution lock prevents two orchestrator invocations
from operating on the same run simultaneously. Existing per-job atomic writes
and the state lock remain in force.

## CLI

Example bounded run:

```bash
.venv/bin/python -m reinforcement_learning.training.recurrent_orchestrator \
  --run \
  --run-directory <run-directory> \
  --timesteps 100000 \
  --device cpu \
  --max-jobs 20 \
  --workers 2 \
  --cpu-threads-per-worker 4
```

Here, 20 is the job limit and 2 is the concurrency limit. Values supported for
`--workers` are 1, 2, and 4. Omitting `--workers` keeps the legacy sequential
path. More than one worker requires an explicit thread limit; the orchestrator
does not guess a permanent policy.

`--status` now prints a global count summary followed by the existing
per-symbol table. Active rows include the worker PID when available. Global
counts expose total, eligible, queued, active, completed, failed, and
interrupted jobs.

## CPU thread policy

For spawned CPU workers the parent sets these variables in the child process's
inherited environment and immediately restores the parent environment:

- `OMP_NUM_THREADS`;
- `MKL_NUM_THREADS`;
- `OPENBLAS_NUM_THREADS`;
- `VECLIB_MAXIMUM_THREADS`;
- `NUMEXPR_NUM_THREADS`.

The child then verifies and records `torch.set_num_threads()` and the Torch
interop setting. Worker telemetry must match the requested intra-op thread
limit or the parent fails the job closed. This makes oversubscription
benchmarkable without choosing a permanent worker/thread count prematurely.

## Failure and interruption behavior

- One worker exception becomes that symbol's `FAILED` result and does not stop
  peers unless fail-fast was explicit.
- A malformed result, wrong PID/device, incomplete budget, missing hash, wrong
  validation reference, or thread-policy mismatch fails only that job and does
  not promote its temporary artifacts.
- A stale temporary workspace fails closed rather than overwriting evidence.
- Ctrl-C stops new launches, terminates and joins every active child, removes
  temporary workspaces, and marks launched unfinished jobs `INTERRUPTED`.
- An unexpected parent exception terminates children and uses the existing
  stale in-flight recovery to prevent abandoned `TRAINING`/`VALIDATING` jobs.
- Completed peers remain completed; unlaunched jobs remain queued.
- The model registry is byte-compared before/after every invocation.

## Benchmark path

The benchmark module uses the production discovery, run store, process worker,
trainer, and validation evaluator. It selects four trainable symbols at
deterministic TRAIN-row-depth quantiles rather than cherry-picking speed or
performance. Candidate runs use separate temporary roots and are removed after
telemetry is captured.

The full Apple M2 command is:

```bash
.venv/bin/python -m reinforcement_learning.training.cpu_parallel_benchmark \
  --run \
  --timesteps 100000 \
  --workers 1 2 4 \
  --thread-policy 1:8 2:4 4:2 \
  --symbol-count 4 \
  --output-json docs/report_logs/cpu_parallel_100k_benchmark.json
```

The `1:8`, `2:4`, and `4:2` values are benchmark candidates for this 8-core
Mac, not permanent defaults. A 100k run is required before selecting the
production Mac worker count. The selection rule requires safe completion,
explicit CPU, no TEST access, completed validation, protected-artifact
integrity, and at least 10% aggregate throughput improvement over one worker.
When a lower worker count is within 5% of the highest agents/hour, the lower
count is preferred for resource headroom. Validation returns never influence
the selection.

## Bounded 512-timestep qualification

Current discovery reported 510 identities and 435 trainable contracts. The
trainable count matches the milestone baseline; the identity count is two above
the older 508-identity full-run assumption and must be reconciled separately
before authorizing a full-universe run.

Selection policy: `train_row_depth_quantiles_v1`  
Symbols: `786`, `CCM`, `DCL`, `ECOP`  
Seed: 42  
Validation: enabled only after TRAIN  
TEST: not loaded

| Workers | Threads/worker | Jobs | Wall seconds | Aggregate steps/s | Agents/hour | Max worker RSS | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 4/4 | 18.513 | 110.627 | 777.8 | 493.8 MB | safe |
| 2 | 4 | 4/4 | 10.267 | 199.469 | 1,402.5 | 498.7 MB | safe |
| 4 | 2 | 4/4 | 6.488 | 315.675 | 2,219.6 | 501.4 MB | safe |

All 12 tiny agent runs completed, reported CPU, transitioned through validation,
and produced deterministic isolated model paths. Registry, production saved
models, `data/models`, and source recurrent artifacts were unchanged. Temporary
run directories were cleaned. These figures are process/mechanics qualification
only: spawn overhead is large relative to a single 512-step rollout, so they do
not select 4 workers for 100k training.

## Progress and telemetry

Per-symbol records retain completed/requested timesteps, capped progress,
effective device, timestamps, elapsed time, validation state, model
availability, and error. Active-worker state adds PID. Worker logs retain:

- actual/requested timesteps and duration;
- training versus validation duration;
- TRAIN rows/date range;
- final PPO diagnostics;
- actual device;
- worker and parent peak RSS;
- CPU thread policy;
- explicit `test_partition_loaded=false`.

The benchmark aggregates wall time, completed/failed/interrupted jobs, actual
timesteps, steps/second, agents/hour, per-agent telemetry, system memory when
portable, validation behavior, and protected-artifact checks.

## Verification

- bounded production smoke: 12/12 completed, zero failures/interruption;
- final strict-protocol production check: JSML completed 512/512 on CPU with
  validation in 5.405 seconds wall time;
- TEST loaded: false;
- production registry/models changed: false;
- source recurrent artifacts changed: false;
- full 435-symbol run: not started;
- `.venv/bin/python -m pytest -q`: 759 passed, 2 skipped in 55.88s;
- `.venv/bin/python -m pip check`: no broken requirements;
- `git diff --check`: passed.

## Decision

**READY_CPU_PARALLEL_BENCHMARK**

The process/state engine and benchmark path are ready for the explicit 100k
Apple M2 comparison. This decision does not select a worker count and does not
authorize full-universe training.
