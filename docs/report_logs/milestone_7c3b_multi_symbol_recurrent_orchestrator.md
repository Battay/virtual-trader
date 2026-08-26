# Milestone 7C.3b — Multi-Symbol RecurrentPPO Orchestration

Date: 2026-08-26  
Branch: `feat/rl-environment-v1`  
Orchestrator version: `recurrent_multi_symbol_orchestrator_v1`

## Outcome

The repository now has a persistent, deterministic, single-worker orchestration
layer for independent single-symbol RecurrentPPO agents. It accounts for every
member of the frozen 508-symbol current-common-equity identity universe, queues
only compatible Mature recurrent contracts, records every stage atomically,
isolates each attempt's artifacts, continues after per-symbol failure, and can
recover abandoned in-flight state honestly.

This milestone did not start full-universe training. It did not implement soft
clustering or a lead allocator, and it did not load TEST.

## Existing infrastructure reused

- `train_recurrent_single_symbol` and
  `recurrent_ppo_single_symbol_v1` for TRAIN-only optimization;
- `RecurrentPPOConfig` for the validated MlpLstmPolicy architecture and PPO
  hyperparameters;
- `rl_recurrent_partition_v1` and its metadata-only contract loader for
  discovery;
- `SingleSymbolTradingEnv` and `single_symbol_env_v1`;
- `RecurrentProgressCallback` for bounded timestep progress events;
- `evaluate_recurrent_on_validation` for isolated validation-only evaluation;
- `sha256_file`, `atomic_write_json`, safe path components, and advisory file
  locking;
- the frozen authoritative current-common-equity identity loader and exact
  universe hash;
- existing model-registry and production-model paths only as protected
  boundaries. The orchestrator does not publish to either.

The existing temporary recurrent persistence proof was not reused directly
because it intentionally requires an external temporary directory and loads
validation as part of a round trip. The new run-local persistence follows the
same non-production principle but provides atomic, per-attempt model archives.

## Universe discovery

Real local discovery produced:

| Category | Count | Meaning |
|---|---:|---|
| Eligible/trainable | 432 | Mature, compatible canonical recurrent contract |
| Insufficient data | 15 | 13 Insufficient plus 2 Cold Start identities |
| Missing required artifacts | 4 | Mature candidates missing/incompatible recurrent artifacts |
| Unsupported | 57 | Not active/recently traded or unsupported by the current independent pipeline |
| Total identity universe | 508 | Every frozen identity has a job record |

The 432 count is lower than the earlier 454 recurrent-compatible count because
454 was measured over a broader 563-security current metadata inventory. This
milestone intersects recurrent compatibility with the subsequently frozen 508
current-common-equity identity universe.

Universe version: `current_common_equity_universe_v1`  
Universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`

Discovery loads and hashes recurrent contract metadata. It does not load TRAIN,
VALIDATION, or TEST frames. Each source-data hash is the SHA-256 of the symbol's
validated recurrent contract, whose own provenance contains the canonical
partition/scaler hashes.

## Persistent job schema

Schema version: `recurrent_training_job_v1`

Each job records:

- deterministic job ID and run ID;
- symbol and explicit trainability category/reason;
- agent, environment, recurrent-data, feature, and universe versions;
- universe hash and per-symbol source-contract hash;
- requested and completed timesteps;
- seed and complete hyperparameter hash;
- requested device, effective device, and CUDA device name when applicable;
- current status and complete state-transition history;
- created, started, updated, and completed timestamps;
- wall-clock duration;
- attempt-specific checkpoint, model, and validation paths;
- model hash and model availability;
- validation status/reference;
- retained error message and explicit retry count.

The run manifest has a deterministic fingerprint over the universe, source
inventory, configuration, seed, timestep budget, requested device, validation
policy, agent version, and worker limit. Volatile creation time is excluded
from IDs.

## State machine

Primary path:

`QUEUED → TRAINING → VALIDATING → COMPLETED`

Other explicit paths:

- `TRAINING/VALIDATING → FAILED` on controlled exception;
- `TRAINING/VALIDATING → INTERRUPTED` on cancellation or stale-process
  recovery;
- `COMPLETED/QUEUED/PENDING/FAILED/INTERRUPTED → STALE` when current contract
  or retained model no longer matches;
- `FAILED/INTERRUPTED/STALE → QUEUED` only through an explicit restart;
- `INELIGIBLE` is terminal and cannot be queued accidentally.

Illegal transitions raise `TrainingJobStateError`. A job is marked TRAINING
before device resolution or trainer execution, so device errors cannot leave a
queued job with no recorded failure.

## Orchestration flow

1. discover and classify every frozen identity;
2. create a deterministic run manifest and 508 atomic job JSON files;
3. preflight completed and pending job contracts for staleness;
4. select queued jobs in deterministic symbol order;
5. run one symbol at a time in v1;
6. persist bounded progress events;
7. atomically save the successful in-memory model under an attempt-specific
   path without overwriting any prior attempt;
8. optionally transition to validation and evaluate the complete VALIDATION
   episode only;
9. persist validation metrics without episode history;
10. mark completed or retain a precise per-symbol failure and continue unless
    fail-fast was explicitly selected.

`max_jobs` bounds each invocation. Parallel workers are deliberately deferred;
the v1 manifest enforces `worker_limit = 1` and makes no parallel-speedup claim.

## Progress interface

The CLI provides discovery, run creation, bounded execution, and a
download-state-style status table. Status rows contain:

- symbol and status;
- completed/requested timesteps and capped progress percentage;
- effective device;
- start time and observed elapsed time;
- ETA only when the job is actively training with nonzero observed progress;
- validation status;
- model availability;
- error summary.

Example entry point:

```text
python -m reinforcement_learning.training.recurrent_orchestrator --discover
```

Run execution defaults to one job and requires an explicit run directory. There
is no accidental all-symbol command.

## Resume and checkpoint policy

Current capability:
`restart_from_zero_only_no_optimizer_checkpoint_v1`.

The validated recurrent trainer does not currently save a complete SB3
optimizer, rollout buffer, random state, environment state, and recurrent state
checkpoint during learning. Therefore:

- an interrupted run retains its actual observed timestep count and error;
- stale TRAINING or VALIDATING markers recover to INTERRUPTED;
- a restart must be explicit;
- restart resets progress to zero and increments the retry count;
- restart uses a new `attempt_NNN` directory;
- this is never described as checkpoint resume.

Checkpoint directories are reserved in the artifact contract but remain empty
until a scientifically valid full-state checkpoint implementation exists.

## Device policy

- CPU: validated local production path and current default.
- CUDA: accepted when `torch.cuda.is_available()` and at least one CUDA device
  are reported. `auto` prefers CUDA and records device count/name.
- MPS: explicit opt-in only. It is never selected by `auto` because sustained
  recurrent LSTM training crashed in the measured Apple-M2 environment.
- `auto`: CUDA when available, otherwise CPU.
- Explicit CUDA or MPS requests fail when unavailable. Neither may silently
  fall back to CPU.
- The orchestrator independently revalidates the resolver result, so an
  injected or future resolver cannot silently substitute devices.

CUDA support is implemented but was not hardware-validated on this MacBook Air
M2. CPU remains the only locally validated full training device.

## Artifact isolation

```text
data/training_runs/<run_id>/
  run_manifest.json
  jobs/<symbol>.json
  models/<symbol>/attempt_000/model.zip
  checkpoints/<symbol>/attempt_000/
  validation/<symbol>/attempt_000.json
  logs/
```

Retries use `attempt_001`, `attempt_002`, and so forth. Paths are run-relative,
portable, unique by symbol/attempt, and checked against traversal. Existing
model paths are never overwritten. The orchestration root is rejected if it is
inside the production saved-model directory.

## Real acceptance run

Selection rule: first three trainable symbols in deterministic frozen-universe
order. Configuration: 512 timesteps, seed 42, explicit CPU, validation enabled.

| Symbol | TRAIN rows | State path | Timesteps | Effective device | Duration | Validation | Model |
|---|---:|---|---:|---|---:|---|---|
| 786 | 1,152 | QUEUED→TRAINING→VALIDATING→COMPLETED | 512/512 | CPU | 2.219 s | completed | isolated |
| AABS | 1,317 | QUEUED→TRAINING→VALIDATING→COMPLETED | 512/512 | CPU | 1.637 s | completed | isolated |
| AATM | 343 | QUEUED→TRAINING→VALIDATING→COMPLETED | 512/512 | CPU | 1.381 s | completed | isolated |

Total bounded acceptance execution: 5.297 seconds. This was an integration
check, not a performance or model-quality result.

- registry unchanged: yes;
- production saved-model directories unchanged: yes;
- TEST loaded/evaluated: no;
- temporary run directory cleaned: yes.

Offline acceptance tests additionally simulated an isolated first-symbol
failure, proved the second symbol still completed, proved a compatible completed
job was skipped, recovered abandoned TRAINING to INTERRUPTED, and required an
explicit zero-start retry with a new attempt path.

## Before full-universe training

The orchestration engine is ready for a separately authorized bounded run, but
full-universe execution should not start until:

1. the supervisor chooses 100k versus 250k or another fixed budget;
2. the 15 insufficient and 57 unsupported identities remain explicitly out of
   independent training, and the four missing/incompatible artifacts are
   reviewed without fabrication;
3. available disk capacity is checked for hundreds of isolated model and
   validation bundles;
4. the approximately 25.9-hour sequential CPU estimate for 432 × 100k, or
   63.4-hour estimate for 432 × 250k, is accepted;
5. any CUDA deployment is benchmarked on the actual target GPU;
6. true checkpoint continuation is implemented if restarting a long interrupted
   job from zero is operationally unacceptable.

No parallel speedup should be assumed. Bounded parallel workers remain a future
extension after resource-contention measurements.

## Verification

- `.venv/bin/python -m pytest -q`: **665 passed, 2 skipped in 24.81s**;
- `.venv/bin/python -m pip check`: no broken requirements;
- `git diff --check`: passed;
- no live HTTP requests;
- no commit created.

## Safety statement

No full-universe training ran. No source Parquet, recurrent source artifacts,
TEST data, model registry, production model, soft-clustering code, or lead
allocator was modified. No commit was created.
