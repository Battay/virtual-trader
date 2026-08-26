# Milestone 7C.3d — Canonical TRAIN-Only Recurrent Artifact Recovery

Audit/build timestamp: 2026-08-26T21:12:26Z  
Identity universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`  
Source Parquet SHA-256: `45e3d396c3472fa53b20016c153fdd529308600274ecf77d0ef942417793d7d5`  
Decision: **READY_FOR_FULL_UNIVERSE_TRAINING_PREP**

## Outcome

Three of the eight pipeline-limited identities have been recovered through an
isolated successor contract:

- MUGHALC — 143 final usable TRAIN features
- GEMMEL — 126 final usable TRAIN features
- GEMPAPL — 240 final usable TRAIN features

ASIC, EWIC, ZUMA, GEMNETS, and GEMPACRA remain excluded because their
TRAIN-only prefixes produce only 80–97 usable feature observations. They are no
longer ambiguous pipeline exclusions: the new path handles their identity and
security type, but the data available to TRAIN does not meet the versioned
Mature minimum of 126.

No v1 artifact was changed. No VALIDATION or TEST frame was loaded or created.

## Why the legacy 252-row gate existed

`AI_MINIMUM_USABLE_ROWS = 252` belongs to the legacy processed-symbol dataset
policy. The old builder also requires `security_type == ordinary_equity` and
current activity before producing the base processed CSV. The chronological
split, RL scaler/partitions, and recurrent contract are downstream of that
builder, so a symbol with 126–251 usable observations—or a GEM common equity—can
never reach recurrent artifact generation even though RecurrentPPO itself does
not require 252 rows.

The audit separates the requirements as follows:

| Requirement | Actual constraint | Interpretation |
|---|---:|---|
| Backward-looking feature warm-up | 49 retained rows | Mathematical indicator warm-up; SMA-50 is the longest configured lookback |
| Environment episode | 2 usable observations | One observation plus a next-row execution/mark-to-market step |
| StandardScaler | Mechanically fit-capable with at least one row | No 252-row dependency; two rows are required here because the environment is stricter |
| Recurrent full-partition episode | 2 usable observations | No fixed 512-row episode requirement; PPO rollouts may cross normal episode resets |
| Approved independent-training policy | 126 final usable TRAIN rows | Scientific Mature threshold, now applied without consulting later-partition values |
| Minimum quality-retained source rows for Mature TRAIN | 175 | 49 warm-up + 126 final usable rows, before any additional quality losses |
| Legacy 252 threshold | 252 usable rows | Older one-trading-year symbol-model eligibility policy, not a feature/LSTM mathematical minimum |

The v2 path does not lower the approved 126-observation Mature standard to
recover a named symbol. It strengthens leakage control by requiring all 126
qualifying observations to exist inside TRAIN.

## Eight-symbol evidence

| Symbol | Type | Raw TRAIN dates | Quality-retained OHLCV | Quality removed | Warm-up loss | Final TRAIN features | Feature span | Episode/scaler feasible | Recovery |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| ASIC | ordinary | 293 | 146 | 147 | 49 | 97 | 2025-01-14 to 2025-12-24 | Yes | No — below 126 |
| EWIC | ordinary | 324 | 144 | 180 | 49 | 95 | 2021-12-13 to 2025-09-15 | Yes | No — below 126 |
| MUGHALC | ordinary | 192 | 192 | 0 | 49 | 143 | 2025-08-26 to 2026-04-02 | Yes | **Recovered** |
| ZUMA | ordinary | 137 | 137 | 0 | 49 | 88 | 2026-01-02 to 2026-05-11 | Yes | No — below 126 |
| GEMMEL | GEM | 196 | 175 | 21 | 49 | 126 | 2024-10-17 to 2025-12-31 | Yes | **Recovered** |
| GEMNETS | GEM | 140 | 139 | 1 | 49 | 90 | 2025-08-07 to 2026-02-13 | Yes | No — below 126 |
| GEMPACRA | GEM | 132 | 129 | 3 | 49 | 80 | 2025-09-24 to 2026-03-18 | Yes | No — below 126 |
| GEMPAPL | GEM | 349 | 289 | 60 | 49 | 240 | 2022-03-25 to 2025-06-18 | Yes | **Recovered** |

All eight have zero additional missing-feature rows after warm-up. ASIC and
EWIC fail because the existing OHLC-quality policy removes 147 and 180 TRAIN
rows respectively, not because their raw date counts are small. Every symbol is
mechanically capable of a scaler and two-row episode; the five exclusions are
the scientific 126-row decision.

The machine-readable evidence is in
`docs/report_logs/milestone_7c3d_pipeline_limited_equity_evidence.csv`.

## GEM compatibility

The local authoritative metadata classifies GEMMEL, GEMNETS, GEMPACRA, and
GEMPAPL as current `COMMON_EQUITY` with `security_type=gem_equity`. Their
canonical data uses the same daily OHLCV schema and the environment uses the
same unscaled real execution fields, long-only whole-share accounting,
commission/slippage configuration, observation ordering, and reset semantics.
No local environment field or feature calculation depends on main-board versus
GEM board identity.

Therefore GEM is technically compatible with the existing feature/environment
semantics for this research path. The v2 allowlist is deliberately limited to
authoritative `ordinary_equity` and `gem_equity`; ETFs, funds, rights,
preference shares, debt, government securities, and unknown instruments remain
fail-closed. This is technical compatibility, not a claim that GEM liquidity or
microstructure is economically identical to main-board equities.

## New versioned TRAIN-only contract

The following versions are separate from all historical v1 artifacts:

- Feature contract: `canonical_symbol_features_v2`
- RL TRAIN partition: `rl_train_partition_v2`
- Recurrent TRAIN contract: `rl_recurrent_train_partition_v2`
- History policy: `canonical_recurrent_train_history_v2`
- Episode boundaries: `rl_recurrent_train_episode_boundaries_v2`
- Environment remains: `single_symbol_env_v1`

Each recovered symbol has an isolated directory under
`data/processed/canonical_recurrent_train_v2/symbols/<SYMBOL>/` containing:

- `train_features.csv`
- `train_rl.csv`
- `observation_scaler.joblib`
- `observation_scaler.json`
- `episode_boundaries.json`
- `recurrent_train_contract.json`

The root `recovery_audit.json` records all eight outcomes and lets discovery
retain precise reasons for non-recovered symbols.

Contract provenance includes the source Parquet hash, frozen universe hash,
portable source reference, TRAIN cutoff, raw/quality/warm-up/final row counts,
feature implementation version, observation order and shape, unscaled execution
fields, scaler and partition hashes, one full-TRAIN episode boundary, and a
canonical deterministic contract identity. Paths stored inside the contract
are relative and portable.

The scaler is fitted only on the final TRAIN feature frame. Scaled observation
columns are stored separately; open/high/low/close/volume remain unscaled and
byte-for-byte numerically aligned between feature and RL frames. No bars are
synthesized and no returns are forward-filled.

## Artifact identities

| Symbol | Deterministic contract identity | TRAIN RL SHA-256 | Scaler SHA-256 |
|---|---|---|---|
| GEMMEL | `f53aa1e270076e2ca9aa5a2ea3c9c979c8e20eeb06451058bbac673f2c09af2e` | `383938f3ba31564ced8d7af176f08c2c514223b87a8a66b69149b21541c2060c` | `6fcc27928d94ce51916abd3b84703b46e13cfe2547d865b17d75ea982af5ccf8` |
| GEMPAPL | `013197ad72d881aaec3c261a048b754a4b1449a28291076ea169252271fd765b` | `e1103f58f49fa8a3463f60cdee52af7981ca87c28125dcce4bc6990dec2944ea` | `abf479776001df40a302d919c311886321f95f43b3a069ada04a1c52bdb2a7a9` |
| MUGHALC | `c68cac430ccfc9be3ff718d5fb43f4e908d423c9bfdfd6f3c78138fcef9a0990` | `337fc2e848d33262bbcd0c208bb3fd1a9660fd8367fee03a3940bece25ebcb5a` | `af01f83ff6b8afc15051da2375d802c8b212083aa85f8031d18ab106f78ef0e5` |

## Recurrent and orchestrator compatibility

The unified training loader prefers a present v2 contract and otherwise uses
the unchanged v1 loader. An invalid v2 contract fails closed; it never silently
falls back to v1. Recovered frames validate with observation shape `(17,)`, one
episode start, chronological ordering, finite scaled observations, and unscaled
execution prices.

Discovery after generation:

| State | Before | After |
|---|---:|---:|
| Trainable | 432 | 435 |
| No recent trading activity | 53 | 53 |
| Original insufficient | 13 | 13 |
| Original Cold Start | 2 | 2 |
| New insufficient TRAIN history | 0 | 5 |
| Pipeline-limited | 8 | 0 |
| Total | 508 | 508 |

The existing orchestrator creates 508 deterministic jobs, with 435 queued and
73 explicit ineligible records. Recovered v2 jobs record
`validation_status=not_available_train_only_contract`: this milestone proves
TRAIN compatibility and does not pretend that a VALIDATION artifact exists.
The training executor skips validation for that explicit status instead of
loading a v1 or later-partition substitute.

## Mechanics-only smoke

Three in-memory CPU runs used the same RecurrentPPO/MlpLstmPolicy path with
`n_steps=8`, `batch_size=4`, `n_epochs=1`, seed 42, and eight requested
timesteps. No model was saved.

| Symbol | TRAIN rows | TRAIN dates | Timesteps | Duration | Shape | Result |
|---|---:|---|---:|---:|---|---|
| MUGHALC | 143 | 2025-08-26 to 2026-04-02 | 8 | 0.5610 s | `(17,)` | Completed |
| GEMMEL | 126 | 2024-10-17 to 2025-12-31 | 8 | 0.0477 s | `(17,)` | Completed |
| GEMPAPL | 240 | 2022-03-25 to 2025-06-18 | 8 | 0.0637 s | `(17,)` | Completed |

These runs establish mechanics only. They provide no performance, convergence,
validation, or profitability evidence.

## Data-integrity and scope statement

- Existing v1 feature, split, scaler, RL, recurrent, and saved-model artifacts
  were not modified.
- Source Parquet was read-only and not rewritten.
- VALIDATION and TEST values were not loaded.
- No production model, checkpoint, registry row, clustering artifact, allocator,
  or full-size PPO agent was created.
- No live HTTP request was made.

## Final decision

**READY_FOR_FULL_UNIVERSE_TRAINING_PREP**

All 508 frozen identities are now either trainable through an existing v1 or
validated v2 recurrent TRAIN contract (435), or explicitly excluded for a
defensible data/current-activity reason (73). This decision does not authorize
full-universe training. A later milestone must decide how validation artifacts
for v2-recovered symbols are created without weakening TEST sealing.
