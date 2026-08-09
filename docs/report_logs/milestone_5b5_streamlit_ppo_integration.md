# Milestone 5B-5: Safe Streamlit PPO Integration

Implementation date: **2026-08-10 (Asia/Karachi)**

## Objective

Milestone 5B-5 adds the presentation and control layer for the existing
single-symbol PPO infrastructure on the **Training & Model Management** page.
It turns readiness inspection, one-symbol training, validation comparison,
candidate persistence, and registry history into an explicit research
workflow without weakening the boundaries established in Milestones 5B-0
through 5B-4.

The page is a controller, not a second training implementation. It delegates
to the canonical production APIs:

- `train_single_symbol` for TRAIN-only PPO learning;
- `compare_candidate_on_validation` for VALIDATION-only evaluation and
  baseline comparison;
- `persist_ppo_candidate` for guarded, atomic candidate persistence; and
- `load_model_registry` for model history.

Non-visual state and eligibility rules live in `dashboard/ppo_workflow.py` so
they can be tested without starting Streamlit. The page remains responsible
for rendering, session state, progress, and explicit user actions. It does not
train a universe, tune hyperparameters, evaluate TEST, or promote a model to
production.

## Safety invariants

The UI preserves these non-negotiable data boundaries:

| Partition | Permitted use in 5B-5 | Prohibited use |
|---|---|---|
| TRAIN | PPO gradient updates through the canonical trainer | Candidate scoring or final performance claims |
| VALIDATION | Deterministic candidate evaluation and baseline comparison | Gradient updates or final TEST claims |
| TEST | Row count and date boundaries from contract metadata only | Loading `test_rl.csv`, inference, charts, metrics, selection, or tuning |

The metadata-only boundary uses
`reinforcement_learning.data_contract.load_rl_contract_metadata`. Reading the
TEST section of `rl_contract.json` does not read the TEST frame. The page
labels this boundary as **FINAL TEST SET: SEALED** and exposes no TEST
evaluation control or evaluator path.

Training, validation, and persistence are all button-gated. Page load and
ordinary Streamlit reruns perform readiness and metadata inspection only. No
synthetic or demo fallback is permitted when a requested symbol, contract,
scaler, device, or artifact is unavailable.

## Page workflow

The PPO workflow is organized into seven progressive sections. Later actions
remain unavailable until their prerequisites are satisfied.

| Section | Purpose | Write or compute boundary |
|---|---|---|
| **A. RL Readiness** | Summarize versions, eligible symbols, insufficient-history symbols, device guidance, and registry count. | Read-only inspection |
| **B. Training Configuration** | Select one ready symbol, timestep preset, deterministic seed, and requested device; show fixed PPO defaults. | Session configuration only |
| **C. Selected Symbol Data Summary** | Show TRAIN, VALIDATION, and metadata-only TEST ranges and row counts plus contract compatibility. | No frame is loaded for TEST |
| **D. Train PPO Candidate** | Confirm the exact configuration and deliberately launch one TRAIN-only job. | In-memory candidate only |
| **E. Validation Results** | Deliberately evaluate the in-memory policy on VALIDATION and compare four strategies. | No learning and no TEST access |
| **F. Candidate Persistence** | Offer an explicit save only for a compatible `validation_pass`. | Atomic candidate bundle and registry append; no promotion |
| **G. Model Registry / History** | Show useful lifecycle and provenance fields for persisted records. | Read-only registry display |

This ordering makes the lifecycle visible:

```text
ready data -> configured run -> trained in-memory candidate
           -> validation decision -> optional persisted candidate
           -> future, separately authorized promotion
```

## A. RL readiness

The readiness summary displays:

- environment version `single_symbol_env_v1`;
- RL contract version `rl_partition_v1`;
- trainer/config version `ppo_single_symbol_v1`;
- current training-ready symbol count;
- insufficient-history count when the readiness artifact provides it;
- the current device recommendation; and
- the number of registry records.

Counts are derived from current local readiness/contract artifacts and the
current registry, rather than from a hard-coded list. The confirmed starting
state for this milestone is 454 ready active ordinary equities and a
header-only registry. The wording distinguishes **environment and data ready**
from **model exists** so a zero-model registry is represented honestly.

The symbol selector accepts one ready symbol at a time. The proposed pilot
universe—OGDC, UBL, FFC, PPL, MEBL, LUCK, HUBC, PSO, MLCF, and TRG—is displayed
separately as a read-only readiness table. Each proposed symbol is checked
against current readiness dynamically. There is no Train All action.

## B. Training configuration

The editable controls are deliberately narrow:

- timesteps, with safe presets of 10,000, 25,000, 50,000, and 100,000;
- deterministic seed, defaulting to `42`; and
- requested device: `CPU`, `MPS`, or `AUTO`, defaulting to `CPU`.

Selecting 100,000 timesteps does not launch a run. The user must still press
**Train PPO Candidate**.

The remaining versioned `ppo_single_symbol_v1` defaults are read-only:

| Parameter | Value |
|---|---:|
| Policy | `MlpPolicy` |
| Learning rate | `3e-4` |
| `n_steps` | `512` |
| Batch size | `64` |
| Epochs | `10` |
| Gamma | `0.99` |
| GAE lambda | `0.95` |
| Clip range | `0.20` |
| Entropy coefficient | `0.01` |
| Value-function coefficient | `0.50` |
| Maximum gradient norm | `0.50` |

No Optuna, search space, or tuning control is exposed.

### Device guidance

CPU is labelled **Recommended for current PPO workload**. MPS remains an
explicit supported choice when PyTorch reports it available, and AUTO remains
an explicit request rather than a silent device change.

The recommendation is grounded in the controlled MCB 5,120-step benchmark:
CPU delivered approximately **6.81 times** the measured throughput of MPS for
the current single-environment MLP PPO workload on the MacBook Air M2. This is
a workload-specific result, not a claim that CPU is always faster. An explicit
unavailable MPS request fails clearly rather than silently falling back.

## C. Selected-symbol data summary

The selected-symbol card reports metadata for:

- TRAIN: row count, first date, and last date;
- VALIDATION: row count, first date, and last date; and
- TEST: row count and date boundaries from `rl_contract.json` metadata only.

It also reports feature version, observation shape `(17,)`, RL contract
version, and environment version. Missing, stale, or incompatible contract and
scaler metadata blocks training with a concrete error; the page does not infer
readiness from filenames alone.

## D. Explicit training and session safety

Immediately above **Train PPO Candidate**, the page repeats the selected
symbol, timesteps, seed, requested device, and TRAIN date range. Training begins
only on that button event.

The workflow assigns each configuration a stable identity derived from the
symbol, timesteps, seed, requested device, contract/scaler hashes, and current
TRAIN/VALIDATION artifact fingerprint. Session state associates the in-memory
training result and later validation result with that exact identity.
Changing any identity field invalidates incompatible prior results so, for
example, an MCB validation result cannot appear after selecting OGDC.

Long-running job safeguards include:

- an in-progress guard that rejects duplicate or overlapping starts;
- disabling action paths while the corresponding job is running;
- progress updates through the existing trainer callback rather than per-step
  log flooding;
- retaining a completed result in the current Streamlit session;
- clean display of training failure, interruption, or unavailable-device
  errors; and
- no registry or saved-model write during training itself.

Streamlit runs work in the page process. Cooperative cancellation is only
honest when the trainer callback can observe the request; closing a tab or
interrupting the server cannot be presented as guaranteed transactional job
cancellation. Milestone 5B-5 does not introduce a worker queue or background
job service. Completed in-memory candidates are session-scoped and disappear
when that session ends unless the user later persists an eligible candidate.

The training result is labelled **candidate**, never profitable or production,
and displays symbol, status, actual/requested timesteps, duration, TRAIN rows
and dates, requested/resolved device, seed, environment version, feature
version, and RL contract version.

## E. Validation workflow

**Evaluate on Validation** appears only after a compatible completed training
result exists. Validation is a separate deliberate action. It calls the
canonical comparison workflow and evaluates only `validation_rl.csv`; it does
not call `learn()` and never loads TEST.

The comparison contains:

- deterministic PPO;
- Buy and Hold;
- Always Hold; and
- Random with fixed seed `42`.

All four receive the same validation frame, environment configuration, initial
capital, fees, and episode boundaries. The results table displays Final
Portfolio Value, Total Return, Annualized Return, Annualized Volatility,
Sharpe, Sortino, Maximum Drawdown, Trades, Transaction Costs, Exposure,
Realized P&L, and Unrealized P&L. Undefined metrics are shown as unavailable,
not coerced to zero or infinity.

Two focused charts use the evaluator histories: validation portfolio value and
validation drawdown. They are validation diagnostics, not TEST results or
profitability claims.

The `ppo_validation_criteria_v1` decision is displayed with all reasons. The
supported outcomes are:

- `validation_pass`;
- `validation_fail`;
- `insufficient_validation_data`; and
- `evaluation_error`.

The page explains that a validation pass does not equal production promotion
and that validation failure does not mean the pipeline is broken.

## F. Candidate persistence

Persistence controls appear only after completed validation for the same
configuration and in-memory policy. **Save Candidate** is offered only for
`validation_pass` and must be pressed explicitly. A validation failure,
insufficient-data result, or evaluation error cannot invoke production
candidate persistence.

The page delegates to `persist_ppo_candidate`; it does not reproduce version
allocation, compatibility validation, atomic publication, or registry logic.
Before saving, it can display the next model identity as a preview. The
persistence transaction remains authoritative because concurrent state may
change between preview and save. After success, the page shows the actual model
ID/version returned by persistence and refreshes registry history.

Duplicate-rerun protection records the persisted identity against the current
candidate within the Streamlit session. A separate read-only registry/filesystem
audit supplies the provisional next version, while the persistence transaction
performs the authoritative collision-safe allocation. Existing paths are never
silently overwritten, and a version collision or partial registry commit is
reported as an error for deliberate recovery.

Saved artifacts remain lifecycle status **candidate**. There is no active
Promote to Production button in this milestone. Promotion eligibility may be
shown read-only, but promotion remains a later, separately authorized action.

## G. Registry and model history

Registry history is loaded through `load_model_registry` and handles a
schema-correct, header-only registry without implying that a model exists. The
display prioritizes lifecycle and reproducibility fields:

- model ID, symbol, version, and algorithm;
- validation status, promotion status, and model status;
- environment and feature versions;
- created/trained date;
- training and validation date ranges; and
- seed.

Internal bundle paths and filesystem bookkeeping remain available to the
model-management layer but are not the focus of the page.

## Error handling

The workflow surfaces, without synthetic fallback:

- a selected symbol that is no longer ready;
- missing, stale, or incompatible RL contract metadata;
- incompatible scaler metadata;
- training failure or interruption;
- validation failure at the evaluation boundary;
- persistence failure or version collision; and
- an explicitly requested device that is unavailable.

Errors preserve the last known-safe persistent state. Failed or interrupted
runs do not create registry records, and validation errors cannot unlock
persistence.

## Limitations

Milestone 5B-5 remains intentionally narrow:

- one symbol and one in-process PPO job at a time;
- overlap prevention is session-local; separate browser sessions are not a
  substitute for a process-wide worker queue;
- no durable background job or guaranteed cancellation after browser/server
  loss;
- no multi-seed robustness study;
- no walk-forward evaluation;
- no hyperparameter search;
- no bulk training;
- no TEST evaluation;
- no automatic candidate save; and
- no production promotion.

The local CPU/MPS result applies to the measured single-environment workload
and should be re-benchmarked if model architecture, vectorization, batch size,
hardware, or PyTorch/Stable-Baselines3 versions change materially.

## Preparation for Milestone 5B-6

The read-only pilot table prepares a small, explicit universe without starting
work. Milestone 5B-6 can use the verified readiness state of OGDC, UBL, FFC,
PPL, MEBL, LUCK, HUBC, PSO, MLCF, and TRG to define a deliberate pilot protocol
covering run ordering, reproducible seeds, resource limits, failure isolation,
and review gates.

Before any pilot is launched, 5B-6 must retain the same TRAIN-only learning,
VALIDATION-only selection, and sealed TEST boundaries. It must not reinterpret
the 5B-5 page as authorization for all-symbol training or automatic promotion.
