# Milestone 7C.3i — Production Training & Models Control Center

## Decision

`READY_PRODUCTION_TRAINING_UI`

The Training & Models page is now a read/command surface for the persistent
single-symbol recurrent orchestrator. Opening or refreshing the page does not
train a model. No production run or worker was launched during this milestone.

## Legacy-page audit

The previous page was a synchronous, browser-owned MLP PPO candidate workflow.
It exposed editable symbol, timestep, seed, and device controls; invoked the
legacy `dashboard.ppo_workflow` trainer directly during a Streamlit request;
kept the candidate model in session memory; and offered separate validate/save
actions. It described recurrent PPO as future work and had no persistent
multi-symbol run state.

Those controls were removed from the registered page. Dataset construction and
split preparation remain owned by their existing pipeline surfaces; their
backend APIs were not removed. Historical MLP workflow modules and registry
records were not deleted, but the production page no longer represents them as
the current execution path.

## Frozen production plan

The page renders immutable backend-owned values and exposes no configuration
editors:

| Field | Frozen value |
|---|---|
| Execution policy | `TRAINABLE_MEMBERS_OF_FROZEN_RESEARCH_UNIVERSE_V1` |
| Frozen identities | 508 |
| Eligible recurrent symbols | 435 |
| Explicit exclusions | 73 |
| Universe hash | `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5` |
| Trainable-symbol hash | `44efa67c6c1aa5ac27d559f85835493206617a63fa24c25648e2da0d9f38a4a2` |
| Algorithm / policy | RecurrentPPO / MlpLstmPolicy |
| Budget | 100,000 timesteps per symbol |
| Seed | 42 |
| Device | CPU |
| Concurrency | 4 workers, 2 CPU threads per worker |
| Validation | after TRAIN |
| TEST | sealed |

A real metadata-only preparation acceptance run in a temporary directory
produced exactly 508 persistent records: 435 `QUEUED` and 73 `INELIGIBLE`.
It produced no controller state and no model artifact.

## Page architecture

The page is organized as:

1. system readiness and immutable plan;
2. production/benchmark/smoke/legacy run selection;
3. explicit prepare/start/stop/interrupt controls;
4. download-style aggregate progress and observed-only ETA;
5. active worker table and searchable all-job table;
6. failed/interrupted restart-from-zero console;
7. read-only model registry view;
8. per-symbol TRAIN-contract, diagnostics, VALIDATION, and artifact detail;
9. bounded controller log and persisted orchestration events.

The pre-run view does not display a misleading zero-percent running bar. A
completed view states: “TRAIN and VALIDATION complete. TEST remains sealed.”

### Final pre-production polish

A native `st.space("medium")` now precedes the page title so Streamlit's top
navigation does not clip the heading or introductory caption on ordinary
desktop Safari/Chrome layouts. No CSS or browser-specific override was added.

The production-plan table now separates frozen experiment identity from its
underlying authoritative contract. It explicitly renders the frozen research
universe role, snapshot date `2026-08-02`, 508 research identities, frozen
manifest version, frozen universe hash, execution-training policy, 435
trainable agents, and trainable-symbol hash. The retained
`current_common_equity_universe_v1` provenance is labelled **Underlying identity
contract**, not frozen universe version. A nearby statement makes clear that
the current operational identity universe is outside this training run.

The readiness cards and immutable PPO/LSTM production configuration remain
unchanged. The pre-run state continues to contain no progress bars.

## Detached process and recovery design

Streamlit launches a detached Python controller with a new process session,
closed stdin, bounded stdout/stderr log, and an atomic PID/state record. A
per-run advisory launch lock prevents duplicate launchers. A PID hand-off
barrier prevents a fast child from having its `RUNNING` record overwritten by
the parent process's `STARTING` record.

Only controller states that are actively running are eligible for PID liveness
checks. This prevents an old final-state PID from later blocking a continuation
if that operating-system PID is reused. A dead active controller causes its
recorded workers to be terminated and in-flight jobs to become honestly
`INTERRUPTED`; the user must explicitly restart them from timestep zero.

The browser is not the process owner. Closing a tab, refreshing the browser, or
restarting Streamlit does not signal the detached controller.

## Stop, interrupt, and retry semantics

- **Stop after current jobs** atomically records a stop request. Active workers
  finish; no additional queued workers launch.
- **Interrupt active run** sends `SIGINT` to the controller. Its process workers
  are joined/terminated and active jobs become `INTERRUPTED`.
- **Retry failed** and **requeue interrupted** are disabled while a controller is
  live and require explicit confirmation.
- Retry is explicitly a restart from timestep zero. It is not described as
  optimizer/checkpoint resume.
- Benchmark, smoke, and legacy runs are viewable but cannot be started as the
  production run from this page.

## Progress and diagnostics

The persistent job records remain the source of truth. The UI shows completed,
active, queued, validating, failed, interrupted, stale, and ineligible counts.
Per active job it shows symbol metadata, actual/requested timesteps, progress,
runtime, worker PID/display slot, effective device, CPU threads, and timestamps.

An aggregate rate/ETA is withheld until at least two agents have completed over
at least one observed minute. The qualified 46.81 agents/hour benchmark remains
labelled as prior benchmark evidence, not live ETA. Training diagnostics are
read from bounded per-attempt logs and never fabricated.

### Per-symbol progress-bar addendum

Progress now has one backend-owned contract. For every eligible job, persisted
`completed_timesteps` is divided by positive `requested_timesteps`; graphical
rendering is clamped to 0–100%, while the unmodified actual timestep value stays
visible. Missing, non-integral, negative, or zero-budget fields fail closed.

Each active TRAINING worker has a compact native Streamlit card with its symbol,
company, sector, state, graphical progress bar, actual/requested timesteps,
percentage, elapsed time, worker slot/PID, device, and CPU-thread policy.
VALIDATING retains a 100% training bar and is labelled “Training complete —
validating.” Advanced PPO diagnostics remain collapsed in a per-symbol
expander.

FAILED and INTERRUPTED jobs retain their last persisted percentage in a native
progress-style dataframe column. COMPLETED jobs render 100% and do not remain
in the active section. INELIGIBLE identities have a null progress value and no
training bar.

Overall run progress is now:

`sum(min(actual_timesteps, requested_timesteps) for eligible jobs)`

divided by:

`sum(requested_timesteps for eligible jobs)`

The 73 ineligible identities never enter this denominator. Completed-agent
count remains a separate `completed / 435` metric. Every browser refresh reloads
the atomic job records; no progress depends on Streamlit session state.

## Registry and artifact behavior

The existing v2 model registry is rendered read-only and split into recurrent
and legacy families. Artifact presence and registry-manifest hash integrity are
reported. A malformed registry is surfaced as an integrity error rather than
silently displayed as empty. Run-isolated recurrent artifacts remain visible in
job/detail state but are not automatically registered or promoted.

## TEST isolation

The page only reads job state, recurrent TRAIN contract metadata, training
diagnostics, and VALIDATION artifacts. A validation artifact claiming TEST
access is rejected. No TEST partition path or loader is exposed to the control
module. The run manifest continues to require `test_partition_loaded=false`.

## Acceptance evidence

| State | Result | Evidence |
|---|---|---|
| No production run | PASS | AppTest renders readiness and Prepare without writes |
| Prepared/not started | PASS | mocked page state plus real 508-record temp preparation |
| Active/running | PASS | mocked ranged-progress state renders worker and stop controls |
| Partial failure | PASS | mocked failure state renders error and explicit retry controls |
| Completed | PASS | mocked completed state renders final TRAIN/VALIDATION message |
| Main application registration | PASS | `app.py` AppTest loaded without exception |
| Local Streamlit process | PASS | server bound on port 8513 and health returned `ok` |
| Browser refresh safety | PASS | AppTest reruns remained read-only; process is detached by contract |

The temporary Streamlit server was stopped after the health check. No screenshot
was required or produced; acceptance notes and deterministic AppTest coverage
are retained instead.

## Verification

- Focused control-center suite: 30 passed.
- Streamlit page-state acceptance: pre-run, prepared, running, partial failure,
  and completed all passed without an exception.
- Full repository pytest: **794 passed, 2 skipped in 60.02 seconds**.
- `pip check`: **No broken requirements found**.
- `git diff --check`: **passed**.
- No recurrent training occurred.
- No production run was created under the repository training-run root.
- No production model or registry row was created.
- TEST remained sealed.
