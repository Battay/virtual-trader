# Milestone 5B-3: Atomic PPO Model Persistence and Registry Integration

Implementation date: **2026-08-09 (Asia/Karachi)**

## Objective

Milestone 5B-3 adds an explicit persistence boundary between an in-memory PPO
candidate and the project's production model inventory. A model can be
persisted only after compatible training and validation results have been
checked together. Persistence creates an immutable, versioned artifact bundle,
verifies that bundle, and then appends one matching registry row.

This milestone does **not** perform production promotion. A validation-passing
model is saved with candidate status only. A separate helper can report whether
that candidate would be eligible for a future explicit promotion action, but it
does not change any file or lifecycle status.

The implementation remains single-symbol PPO only. It does not train the full
universe, evaluate TEST, tune hyperparameters, enable Streamlit training, or
implement master-model persistence.

## Pre-change audit

The pre-existing model-management foundation established useful naming and
registry concepts, but it was not yet safe for production artifact writes.

| Area | Pre-change inconsistency or risk | 5B-3 treatment |
|---|---|---|
| Bundle definition | `ModelArtifactPaths` reserved only a model, generic scaler, and generic metrics file. | Adds a complete `ppo_artifact_bundle_v1` layout while retaining the older helpers for compatibility. |
| Scaler convention | The reserved name was `standard_scaler.joblib`, while the canonical RL contract uses a train-fitted `rl_observation_scaler.joblib` plus JSON metadata. | Copies both canonical RL scaler artifacts into every immutable bundle and records their source hashes. |
| Registry schema | The registry did not encode RL contract, PPO config, validation, manifest, observation, or source-code provenance. | Extends the registry to version `model_registry_v2` with explicit compatibility and artifact columns. |
| Lifecycle | `trained` did not distinguish a validation candidate from a production model. | Adds candidate/experiment/production/superseded lifecycle vocabulary plus separate validation and promotion statuses. |
| Record creation | Arbitrary known-column overrides could replace identity-critical values. | Identity-critical fields are protected and every final record is validated as a complete row. |
| Version validation | Numeric-but-fractional versions and duplicate scope/symbol/version identities were not rejected. | Requires positive integral versions and uniqueness of both model ID and `(scope, symbol, version)`. |
| Version allocation | The next version was derived only from the registry and did not account for filesystem-only bundles. | Audits registry and filesystem under a lock; disagreement or malformed version directories block allocation. |
| Concurrent registry writes | A load/append/replace sequence could lose another concurrent append. | Serializes allocation, publication, and append with an advisory registry lock. |
| Durability | Registry replacement was atomic but did not fsync the completed temporary file and containing directory. | Flushes and fsyncs the temporary CSV, validates it, atomically replaces the destination, and fsyncs the parent directory. |
| Loading | No exact-version PPO bundle loader or compatibility verification existed. | Adds explicit registry resolution, manifest verification, compatibility checks, and CPU PPO reload. |
| Freshness | Held-out validation and TEST dates could be counted as new data because staleness used `training_data_end`. | Uses `dataset_latest_date` or complete-history end as the freshness cutoff, with a legacy fallback. |

At implementation time, `data/models/model_registry.csv` contained only its
header and the production saved-model directories contained only `.gitkeep`
files. Therefore the schema could be upgraded without migrating a previously
promoted model.

## Versioned identity and bundle layout

The established model identity convention is preserved:

```text
ppo-symbol-<SYMBOL>-v<four-digit-version>
```

For example, MCB's first persisted version is:

```text
ppo-symbol-MCB-v0001
reinforcement_learning/saved_models/symbol_models/MCB/v0001/
```

Versions are immutable positive integers. The allocator examines both validated
registry rows and canonical filesystem directories. Allocation is refused if
the two sides disagree, if a version directory is malformed, if the target
already exists, or if a registry identity is duplicated. Symbols must already
be collision-safe path components; persistence does not silently normalize two
different symbols to the same directory.

Each `ppo_artifact_bundle_v1` directory has exactly these files:

```text
v0001/
├── ppo_model.zip
├── model_metadata.json
├── ppo_config.json
├── validation_metrics.json
├── baseline_comparison_metrics.json
├── rl_contract.json
├── rl_observation_scaler.joblib
├── rl_observation_scaler.json
├── registry_record.json
└── artifact_manifest.json
```

The files serve distinct purposes:

- `ppo_model.zip` is the Stable-Baselines3 PPO serialization.
- `model_metadata.json` contains identity, lifecycle, dependency versions,
  source-code provenance, partition ranges, observation/action contracts,
  training provenance, validation decision, and the explicit statement that
  TEST was not evaluated.
- `ppo_config.json` stores the complete versioned PPO configuration.
- `validation_metrics.json` stores PPO validation diagnostics and the
  candidate decision without episode history.
- `baseline_comparison_metrics.json` stores the apples-to-apples validation
  comparison with Buy and Hold, Always Hold, and fixed-seed Random.
- `rl_contract.json` is the exact canonical contract used by the candidate.
- `rl_observation_scaler.joblib` and `rl_observation_scaler.json` are immutable
  copies of the exact train-fitted scaler and its metadata.
- `registry_record.json` stores the planned registry row used for recovery if
  publication succeeds but the registry append does not.
- `artifact_manifest.json` records SHA-256 and byte size for every other bundle
  file. The manifest deliberately does not hash itself.

Copying the small RL contract and scaler artifacts makes the bundle
self-contained if processed split artifacts are rebuilt later. Their original
paths and hashes are still retained as provenance.

## Persistence eligibility and compatibility boundary

The production API accepts a `PPOTrainingResult`, a
`ValidationComparisonResult`, the exact symbol, and optional notes. Before any
production path is created, it requires all of the following:

- the trainer result is `completed`, exposes its in-memory PPO model, and has
  consistent actual timestep metadata;
- training and validation refer to the same symbol and the same in-memory
  policy state;
- validation is completed on the `validation` partition;
- the policy hash and PPO timestep counter did not change during validation;
- the validation decision is `validation_pass` for a production candidate;
- algorithm, PPO config version, feature version, environment version, and RL
  contract version agree;
- observation feature names and order match the canonical 12-feature contract;
- observation shape is `(17,)`, the action space is `Discrete(3)`, and action
  semantics remain Hold/Buy/Sell;
- training and validation source-contract/scaler paths and SHA-256 values agree
  with the artifacts still present under the requested split root;
- the RL contract declares a train-fitted scaler and has valid train,
  validation, and TEST metadata; and
- train and validation row/date metadata agree with the trainer and evaluator
  results.

Failed or interrupted training cannot be persisted. An insufficient or errored
validation result cannot be persisted. A normal production-candidate call also
rejects `validation_fail`.

The separate developer-smoke API accepts `validation_pass` or
`validation_fail`, but only when both registry and saved-model roots are
explicit non-production locations. Such a bundle is labelled `experiment` and
`not_eligible`; it cannot be confused with a production candidate. Validation
thresholds are not relaxed merely to make a smoke run pass.

## Atomic filesystem publication

Persistence follows this sequence:

1. Validate training, validation, and source-artifact provenance before taking
   a version.
2. Acquire the registry transaction lock.
3. Validate the registry and audit its versions against the filesystem.
4. Allocate the next version and create a hidden staging directory beside the
   final version directory. Staging on the same filesystem allows an atomic
   directory rename.
5. Save the PPO zip, copy the exact RL contract/scaler artifacts, write the JSON
   files, flush and fsync file contents, and fsync the staging directory where
   supported.
6. Generate SHA-256/byte-size manifest entries and fully verify the staged
   bundle, including an actual PPO reload.
7. Recheck that the final directory does not exist, atomically rename the
   complete staging directory to the final version path, and fsync its parent
   directory where supported.
8. Validate and append the matching registry row through a separate fsynced
   temporary CSV and atomic replacement.

If any write or validation fails before publication, the hidden staging
directory is removed and no final version or success row exists. Earlier model
versions are never touched. Existing final version paths are never silently
overwritten.

The advisory lock covers allocation, bundle publication, and registry append,
so cooperating project processes cannot allocate the same version or lose an
append. The lock also prevents another supported writer from observing a
half-completed transaction.

## Registry atomicity and orphan recovery

Filesystem directory publication and CSV replacement are two separate atomic
operations; no portable filesystem primitive can commit both as one unit. The
implementation therefore commits the complete, verified bundle first and the
registry row second. This ordering prevents a registry success row from
pointing to a partial bundle.

If the registry update fails after directory publication, persistence raises
`RegistryCommitPendingError` with the exact model ID, bundle path, and registry
path. The completed bundle is intentionally retained as a recoverable
filesystem-only artifact. It cannot be loaded through the registry and the
registry/filesystem consistency audit blocks allocation of another version
until the discrepancy is resolved.

`reconcile_persisted_bundle()` is explicit recovery. It re-verifies the entire
bundle, reconstructs the final row from `registry_record.json` plus the actual
manifest hash, checks for conflicting model ID or version identities, and then
performs the locked atomic append. It is idempotent only when the existing row
is exactly equal; conflicting rows fail closed.

The registry writer validates the complete schema before and after temporary
serialization. It flushes and fsyncs the temporary file, re-loads that file,
uses `os.replace()` for atomic publication, and fsyncs the registry directory.
Existing rows remain immutable and duplicate model IDs, duplicate model paths,
or duplicate scope/symbol/version identities are rejected.

## Registry schema and lifecycle

Registry schema version: **`model_registry_v2`**

The original date, row-count, freshness, path, duration, seed, and notes fields
remain. The schema adds:

- `registry_schema_version` and `artifact_schema_version`;
- `rl_contract_version` and `ppo_config_version`;
- `validation_status` and `promotion_status`;
- RL contract, scaler-metadata, model-metadata, configuration, validation,
  baseline, recovery-record, and manifest paths;
- the manifest SHA-256;
- serialized observation shape and ordered observation features; and
- source Git commit and worktree-dirty provenance when Git metadata is
  available.

The corrected date contract remains explicit:

- complete availability is stored separately as
  `complete_available_history_start` and
  `complete_available_history_end`;
- actual train, validation, and TEST boundaries and row counts occupy their
  respective fields; and
- `dataset_latest_date` is the complete-data cutoff used for freshness rather
  than the earlier training-partition end.

Lifecycle dimensions are separate:

| Artifact purpose | `model_status` | `training_status` | `validation_status` | `promotion_status` |
|---|---|---|---|---|
| Validation-passing production candidate | `candidate` | `completed` | `validation_pass` | `candidate` |
| Non-production developer smoke/experiment | `experiment` | `completed` | `validation_pass` or `validation_fail` | `not_eligible` |
| Future promoted model | `production` | `completed` | `validation_pass` | `production` |

The final row above describes a future lifecycle state only. Milestone 5B-3
does not implement or invoke the mutation that would create it. Legacy status
values remain accepted for backward compatibility, but model loading still
requires an exact registry identity and a valid bundle.

## Exact loading and integrity validation

`load_persisted_ppo()` deliberately has no implicit latest behavior. A caller
must specify exactly one of:

- an exact `model_id`; or
- an exact symbol and positive integer `model_version`.

Missing, ambiguous, or multiply resolved selections fail. The loader derives
the canonical expected path under the caller-supplied saved-model root and
rejects a registry model path that escapes or disagrees with it.

Before Stable-Baselines3 loading, bundle verification requires:

- a real directory, no symlink entries, and exactly the expected file set;
- a compatible `ppo_artifact_bundle_v1` manifest using SHA-256;
- matching hash and byte size for every payload file;
- consistent model identity across manifest, metadata, PPO config, and planned
  registry record;
- PPO algorithm and single-symbol scope;
- compatible `rl_partition_v1`, `single_symbol_env_v1`, and
  `ppo_single_symbol_v1` versions;
- exact ordered observation features, `(17,)` shape, and three-action space;
- a contract and scaler both declaring the canonical feature order and a
  train-fitted scaler;
- a loadable scaler whose fitted feature width is 12;
- validation-only metric artifacts and an explicit false TEST-evaluation flag;
- a structurally valid PPO zip; and
- exact equality between the registry row and the bundle recovery record after
  adding the actual manifest hash.

The PPO model is loaded on CPU. Its saved timestep count, policy-parameter
SHA-256, observation space, and action space must match metadata. A missing,
extra, corrupt, stale, reordered, or incompatible artifact causes a clear
failure; the loader never falls back to another version.

`check_promotion_eligibility()` performs the same exact load and integrity
checks, then confirms candidate, validation-pass, and candidate-promotion
statuses plus the persisted decision. It returns a read-only eligibility
result. It does not promote, supersede, rename, or edit anything.

## Sealed TEST guarantee

Milestone 5B-3 does not evaluate TEST and does not call the TEST evaluator. It
uses TEST metadata from the already-written RL contract only to preserve known
date boundaries, row counts, complete data availability, and freshness.
`test_rl.csv` is not loaded for scoring.

Persisted metadata marks the TEST partition as `sealed_not_evaluated` and sets
`test_evaluation_performed` to false. Validation and baseline metric artifacts
identify their partition as `validation`. No TEST return, Sharpe, drawdown, or
other performance metric is invented or stored.

## Controlled smoke methodology

The final integration verification is designed to use one production-ready
symbol, MCB, without creating a production artifact:

1. Hash the production registry, `data/models`, production saved-model roots,
   and relevant source RL artifacts.
2. Train one 512-timestep in-memory PPO smoke candidate on TRAIN only.
3. Evaluate that candidate on VALIDATION only and run the three baselines.
4. Create a temporary directory containing a temporary registry and temporary
   saved-model root.
5. Persist through `persist_developer_smoke_bundle()`. A validation failure is
   retained only as a non-promotable experiment; it is not relabelled as a
   validation pass.
6. Verify every manifest entry, reload the exact model from the temporary
   registry, and confirm deterministic prediction works for a canonical
   validation observation.
7. Remove the temporary directory and compare all protected production hashes.

This smoke proves serialization, atomic publication, registry integration,
integrity verification, and reload mechanics. It is not a profitability,
research-quality, or final out-of-sample result.

### Verification results

| Check | Result |
|---|---|
| Complete `.venv/bin/python -m pytest -v` | **317 passed in 9.45 seconds** |
| `git diff --check` | **Passed; no whitespace errors** |
| `git status` review | **Completed; no commit created** |
| `.venv/bin/python -m pip check` | **No broken requirements found** |
| MCB temporary persistence/reload | **Passed** |
| Temporary smoke directory cleanup | **Confirmed** |
| Production registry unchanged | **Confirmed byte-for-byte during smoke** |
| Production saved-model roots unchanged | **Confirmed by file-size/SHA-256 snapshot** |
| TEST evaluator calls | **0** |
| Production promotions | **0** |
| Live HTTP requests | **0** |

The controlled MCB run used 1,704 TRAIN rows from 2016-10-06 through
2023-08-23 and requested/completed exactly 512 PPO timesteps on CPU in 1.48
seconds. VALIDATION used 365 rows from 2023-08-24 through 2025-02-12. The
partition trace was exactly `trainer:train` followed by
`evaluator:validation`; no TEST partition was loaded.

The smoke model had observation shape `(17,)`, contract version
`rl_partition_v1`, environment version `single_symbol_env_v1`, and feature
version `psx-4a-126450ec6355`. The complete ten-file bundle passed manifest and
semantic verification, reloaded by exact model ID, and produced a repeatable
deterministic action. Its honest validation decision was `validation_fail`, so
the temporary registry labelled it `experiment` / `not_eligible`. The result
was not relabelled, promoted, or interpreted as evidence of model quality.

Installed integration versions were Stable-Baselines3 2.9.0, PyTorch 2.13.0,
and Gymnasium 1.3.0. The temporary directory was removed automatically, and
the production registry, production saved-model tree, and source RL contract
and scaler hashes matched their pre-smoke snapshots.

## Limitations and next milestone

- Persistence and registry replacement are recoverably coordinated, not one
  indivisible operating-system transaction. A crash between them can leave a
  verified filesystem-only bundle requiring explicit reconciliation.
- The advisory lock coordinates project writers that use the same lock; it
  cannot prevent an unrelated external process from modifying files directly.
- `fcntl` locking is POSIX-specific.
- SHA-256 detects accidental or unauthorized modification after persistence,
  but the manifest is not a cryptographic signature and provides no external
  identity attestation.
- Registry paths are resolved local paths. Self-contained contract/scaler
  copies protect runtime compatibility, but moving a bundle requires an
  explicit registry migration rather than silent path rewriting.
- Only single-symbol PPO bundles are persisted. Master/universal models, bulk
  orchestration, automatic retraining, and UI training remain out of scope.
- Candidate eligibility is read-only. There is no production promotion API,
  no automatic promotion, and no supersession mutation in this milestone.
- TEST remains sealed. Final TEST evaluation must occur only after candidate
  selection and promotion policy are frozen in a later milestone.

No production model is promoted by Milestone 5B-3.
