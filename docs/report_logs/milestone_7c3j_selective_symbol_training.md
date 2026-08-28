# Milestone 7C.3j — Selective Symbol Training & Model Coverage

## Objective

Add first-class selective recurrent training without changing or replacing the
immutable 435-symbol full-production plan. No training was started by this
implementation or its tests, and TEST remains sealed.

## Coverage contract

Coverage is reconstructed from the frozen 435 eligible identities and every
valid persisted recurrent run inventory. A symbol is `TRAINED` only when all of
the following are true:

- its recurrent job reached `COMPLETED` at the qualified 100,000-timestep budget;
- the isolated model exists and matches the job's SHA-256;
- VALIDATION is marked complete and its artifact exists;
- the validation artifact identifies the same symbol, the validation partition,
  compatible environment/feature/recurrent contract versions, and unchanged
  model parameters;
- neither run nor validation evidence reports TEST access.

Registry presence is not used as proof. Active and terminal coverage states are
`UNTRAINED`, `QUEUED`, `TRAINING`, `VALIDATING`, `TRAINED`, `FAILED`, and
`INTERRUPTED`. `Untrained` in summary/filtering means any eligible symbol without
a verified completed model, including queued or failed symbols.

At implementation time, the preserved full run contains four verified trained
symbols (`786`, `AABS`, `AATM`, `ABL`) and 431 queued/untrained symbols.

## Selected-run contract

Selected runs use `recurrent_selected_run_v1` metadata and are classified as
`SELECTED`, never `FULL_PRODUCTION`. The immutable sidecar records:

- requested, skipped, and effective selected membership;
- deterministic requested and effective membership hashes;
- frozen research universe version/hash and frozen trainable hash;
- selected attempt version and execution policy;
- RecurrentPPO budget/seed and CPU worker policy;
- validation enablement and explicit TEST sealing;
- a future-facing device-policy version while CUDA execution remains
  unauthorized in this milestone.

Already verified symbols are skipped by default. Explicit retraining operates
only on selected `TRAINED` symbols and creates a new run identity/attempt, so an
existing model is never overwritten. Repeated preparation of the same normal
selection reuses its compatible queue instead of creating a duplicate.

## Training & Models interface

The page now provides:

- persistent coverage metrics for eligible, trained, untrained, training,
  validating, failed, and interrupted symbols;
- status, sector, and symbol/company filters;
- an explicit multiselect with no default selection, plus **Select visible** and
  **Clear selection**;
- a coverage table showing selection, latest persisted progress, model and
  validation status, and latest run/attempt;
- exact selected membership/hash and qualified 100k CPU confirmation before
  launch;
- a separately confirmed retraining action;
- selected-run progress through the existing persisted job UI;
- run-history classification for `FULL_PRODUCTION`, `SELECTED`, `BENCHMARK`,
  `SMOKE`, and `LEGACY`, including selected count/hash.

The stopped full-production run and all its queued/completed records remain
unchanged. Its existing stop-after-current and explicit restart controls remain
available.

## Safety boundaries

- No full or selected training was launched during implementation.
- No model was overwritten, promoted, or registered.
- No TRAIN, VALIDATION, or TEST market dataframe is loaded for coverage.
- No TEST route was added.
- The full frozen 508/435 identity contract and both frozen hashes are unchanged.
- Browser/session state is used only for transient selection; coverage and
  progress always come from persisted run state and artifacts.
