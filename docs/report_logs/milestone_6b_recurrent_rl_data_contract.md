# Milestone 6B — Recurrent RL Data Contract

Audit/build timestamp: 2026-08-11 23:08:16 PKT (Asia/Karachi)

## Objective and boundary

Milestone 6B adds the versioned `rl_recurrent_partition_v1` metadata and
loading layer required by a future single-symbol RecurrentPPO / LSTM baseline.
It does **not** install `sb3-contrib`, implement a recurrent policy, train a
model, evaluate TEST, or change the existing MLP PPO path.

The recurrent contract is deliberately separate from `rl_partition_v1`.
`rl_partition_v1` remains the canonical real-price/scaled-observation source
for the current `ppo_single_symbol_v1` + `MlpPolicy` baseline. Recurrent-only
episode, reset, cohort, and sequence semantics therefore cannot silently alter
the established MLP contract.

## Contract and artifact layout

For each compatible Mature symbol, the builder creates two small JSON files
under the existing symbol split directory:

```text
data/processed/splits/symbols/<SYMBOL>/
├── rl_contract.json                         # existing rl_partition_v1
├── train.csv / train_rl.csv                 # existing canonical data
├── validation.csv / validation_rl.csv       # existing canonical data
├── test.csv / test_rl.csv                   # existing sealed data
├── rl_observation_scaler.joblib/.json       # existing TRAIN-fitted scaler
└── recurrent/
    ├── recurrent_contract.json              # rl_recurrent_partition_v1
    └── episode_boundaries.json              # recurrent_episode_boundaries_v1
```

No recurrent CSV is copied. The contract references and hashes existing TRAIN
and VALIDATION raw/RL artifacts and the observation scaler. TEST contributes
only its row count and date boundaries; the recurrent contract deliberately
contains no TEST frame path.

The contract records:

- symbol, company, verified sector when available;
- recurrent, source RL, feature, and environment versions;
- ordered 12 market features, 5 dynamic portfolio features, and shape `(17,)`;
- unscaled execution/accounting columns `open, high, low, close, volume`;
- TRAIN-fitted scaler paths and SHA-256 identities;
- TRAIN, VALIDATION, and sealed TEST row/date boundaries;
- sequence and reset semantics;
- history class and recurrent eligibility;
- single-symbol universe ID/hash, constituent list, cohort cutoff, and source
  registry snapshot provenance;
- explicit TEST-sealed/evaluation-not-performed declarations.

## Initial episode and sequence semantics

The safer initial baseline is option **A: one full partition episode**.

- TRAIN is one complete recurrent learning episode.
- VALIDATION is one separate complete recurrent evaluation episode.
- An episode never crosses a partition or symbol boundary.
- The first observation of every episode has `episode_start=True`.
- The environment and recurrent hidden state must reset together at every
  episode start.
- Any future explicit window start, symbol transition, or partition transition
  also forces a reset.
- No reset is inserted inside a continuous episode unless an explicit future
  window boundary is supplied.

Fixed-length rolling/sampled windows are not enabled in v1. Metadata therefore
records:

| Field | `rl_recurrent_partition_v1` value |
|---|---:|
| `episode_strategy` | `full_partition` |
| `sequence_length` | Not configured (`null`) |
| `burn_in_length` | Not used (`null`) |
| `episode_length` | Complete partition (`null`) |
| `minimum_sequence_rows` | 2 |
| `fixed_windows_enabled` | `false` |
| `reset_on_symbol_change` | `true` |
| `reset_on_partition_change` | `true` |

Burn-in is not invented before a future recurrent algorithm and rollout design
justify it.

## Deterministic reset representation

`recurrent_episode_start_mask()` produces the future algorithm-facing Boolean
reset representation. It accepts only row identities (symbol, partition, and
optional explicit-window flags), never feature values. This guarantees:

- row zero is always a reset;
- a symbol change cannot inherit hidden state;
- a partition change cannot inherit hidden state;
- an explicit window boundary cannot inherit hidden state;
- feature values from present or future rows cannot affect reset placement.

The canonical loader cross-checks the generated mask against the separately
hashed episode-boundary artifact.

## Observation, execution, and look-ahead integrity

The recurrent loader delegates frame loading to the established
`load_rl_partition()` path. It therefore retains:

- real, unscaled OHLCV for execution and accounting;
- TRAIN-fitted scaled market observations;
- deterministic market-feature order followed by dynamic portfolio state;
- chronological, unique `(symbol, date)` rows;
- the existing environment timing: observe row *t*, execute at row *t+1* open,
  and mark at row *t+1* close.

The recurrent contract records `lookahead_observations=0`. It does not build
shifted labels, forward-looking windows, or cross-boundary sequences.

## Canonical loader and TEST sealing

`load_recurrent_partition(symbol, partition)` accepts only literal `train` or
`validation`. Passing `test` fails before the canonical frame loader is called.
It returns:

- an isolated environment-ready DataFrame;
- partition rows and date bounds;
- ordered observation and execution fields;
- a Boolean `episode_start` / reset mask;
- structured episode boundaries;
- recurrent/source/feature/environment versions;
- scaler identity and recurrent universe/history metadata.

`load_recurrent_contract_metadata()` may expose TEST row count and start/end
dates, but it does not parse any partition CSV. The build path loads canonical
TRAIN and VALIDATION only. Offline guards explicitly fail if `test.csv` or
`test_rl.csv` is opened during recurrent build/readiness/loading.

## Normalization policy

Per-symbol TRAIN-fitted scaling remains valid for the initial single-symbol
recurrent baseline and preserves the current leakage-safe behavior. The v1
contract records `normalization_scope=symbol` and exact scaler hashes.

Future pooled sector/generalized training will require a new, explicit scaler
fit over the pooled **TRAIN-only** cohort. The contract already reserves
`normalization_scope` values `symbol`, `sector`, and `global`, but sector/global
fitting is intentionally not implemented in 6B. Per-symbol scaling must not be
silently reused as if it were a pooled normalization policy.

## Future sector/generalized compatibility

The single-symbol artifact uses the same metadata concepts required later by a
pooled builder:

- `training_scope`: `symbol` now; `sector` and `generalized` reserved;
- `universe_id` and deterministic `universe_hash`;
- exact `constituent_symbols`;
- verified sector when the local company registry supplies it;
- TRAIN-end `cohort_cutoff`;
- company-registry source path/hash;
- `historical_membership_fabricated=false`.

This does not claim or fabricate historical sector membership. A future pooled
milestone must define time-correct cohort membership and pooled TRAIN-only
normalization explicitly.

## History-policy integration

The recurrent policy remains separate from current MLP readiness:

| History class | Usable observations | Generate independent recurrent artifact | Independent fitting | Transfer/fine-tune route |
|---|---:|---:|---:|---:|
| Mature | `>=126` | Yes, if all source checks pass | Eligible | Eligible |
| Cold Start | `100–125` | No | Not eligible | Future transfer/fine-tune only |
| Insufficient | `<100` | No | Not eligible | Not eligible |

No Cold Start or Insufficient symbol is promoted into current MLP PPO readiness,
and no sector transfer is performed.

## Production migration results

The local active ordinary-equity universe reconciled to the approved counts:

| Result | Count |
|---|---:|
| Mature symbols inspected | 458 |
| Recurrent-compatible symbols generated | 454 |
| Cold Start symbols | 2 |
| Insufficient symbols | 13 |
| Mature source-contract failures | 4 |
| Recurrent JSON artifacts generated | 908 |

The four failures are honest source-readiness failures, not forced conversions:

| Symbol | Usable observations | Reason |
|---|---:|---|
| ASIC | 221 | Canonical `rl_contract.json` is absent |
| EWIC | 231 | Canonical `rl_contract.json` is absent |
| MUGHALC | 225 | Canonical `rl_contract.json` is absent |
| ZUMA | 147 | Canonical `rl_contract.json` is absent |

## Representative local verification

| Symbol | Usable rows | Class | Artifact eligibility/result | TRAIN | VALIDATION | sealed TEST metadata | Episodes / first reset | Contract |
|---|---:|---|---|---|---|---|---|---|
| MCB | 2,437 | Mature | Generated | 1,704; 2016-10-06 to 2023-08-23 | 365; 2023-08-24 to 2025-02-12 | 366; 2025-02-13 to 2026-08-05 | 1 TRAIN + 1 VALIDATION; both `True` | `rl_recurrent_partition_v1` |
| OGDC | 2,437 | Mature | Generated | 1,704; 2016-10-06 to 2023-08-23 | 365; 2023-08-24 to 2025-02-12 | 366; 2025-02-13 to 2026-08-05 | 1 TRAIN + 1 VALIDATION; both `True` | `rl_recurrent_partition_v1` |
| BLUEX | 102 | Cold Start | Not generated; transfer-only | Not available | Not available | Not available | Not applicable | Not generated |
| ANLNV | 16 | Insufficient | Rejected | Not available | Not available | Not available | Not applicable | Not generated |

The usable-row history classification reflects the latest local source view;
the existing immutable split boundaries still end on 2026-08-05 and are the
only rows referenced by the recurrent artifacts.

## Environment compatibility audit

No environment change was required. `single_symbol_env_v1` already provides:

- Gymnasium `reset(seed=...) -> (observation, info)`;
- `step(...) -> (observation, reward, terminated, truncated, info)`;
- deterministic seeding for the current deterministic environment;
- float32 observations with shape `(17,)`;
- isolated history copies;
- explicit portfolio cash, holdings, realized/unrealized P&L, costs, drawdown,
  and action information;
- termination/truncation and a no-resource `close()` lifecycle.

The future recurrent runner must pass the contract mask to RecurrentPPO and
call environment reset at the same episode boundaries. That algorithm adapter
is intentionally deferred.

## Limitations and next milestone

- No `sb3-contrib`, RecurrentPPO, or `MlpLstmPolicy` dependency is installed.
- No recurrent network configuration, LSTM hidden size, rollout length,
  sequence minibatching, or burn-in policy is selected.
- Only single-symbol artifacts exist; pooled sector membership/scaling remains
  design metadata.
- JSON files reference current local source hashes; rebuilding canonical split
  or scaler artifacts intentionally makes the recurrent contract stale until
  regenerated.
- TEST remains sealed and has not been evaluated.

The next milestone is a single-symbol RecurrentPPO baseline that consumes this
contract, explicitly propagates `episode_start`, learns on TRAIN only, evaluates
on VALIDATION only, and continues to leave TEST sealed.

## Data-integrity statement

This milestone made no live HTTP request, changed no raw/backfill data, installed
no dependency, trained no model, wrote no model/registry record, and performed
no TEST evaluation. Production generation added only the ignored recurrent JSON
metadata beneath existing processed symbol split directories. Existing source
CSVs, scalers, `rl_contract.json` files, model registry, and model directories
were not overwritten.

## Verification

- Complete test suite: **478 passed, 1 skipped** in 12.54 seconds. The skip is
  the pre-existing hardware-gated MPS smoke test.
- Focused recurrent contract suite: **23 passed**.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: `No broken requirements found.`
- `sb3-contrib`: not installed.
- Final local artifact inventory: 454 recurrent contracts plus 454 boundary
  files.
- MCB and OGDC final reload: schema valid, one TRAIN and one VALIDATION episode,
  first reset `True`, TEST bounds metadata-only.
