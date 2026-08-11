# Recurrent PPO and Sector-Transfer Architecture Audit

Audit date: **2026-08-11 (Asia/Karachi)**

Audit mode: **read-only architecture revision**

## Executive decision

The first recurrent research milestone should implement a **sector-pretrained
recurrent PPO parent followed by optional symbol fine-tuning**, while retaining
the current single-symbol MLP PPO as the baseline. It should not begin with one
universal model across all 454 symbols, and it should not train 454 independent
LSTMs as the primary design.

The safest sequence is:

1. preserve `single_symbol_env_v1` semantics and prove a single-symbol
   recurrent baseline;
2. introduce a new, immutable `rl_recurrent_partition_v1` contract;
3. build a point-in-time sector universe and pooled sector TRAIN-only scaler;
4. pretrain one sector parent on separate, boundary-reset symbol episodes;
5. compare zero-shot sector transfer with fine-tuning on a new symbol's real
   observations; and
6. add generalized and parallel training only after correctness and leakage
   tests pass.

No OHLC row may be created before a symbol's first real trading observation.
Knowledge transfer means **weight initialization/pretraining**, not synthetic
price history and not “agent-to-agent learning.”

## Scope and evidence

The audit inspected `reinforcement_learning/`, `feature_engineering/`,
`data_pipeline/`, `requirements.txt`, the local company registry, current
readiness outputs, split contracts, and model-registry implementation. It made
no production-code, dependency, dataset, split, registry, or model changes.

The repository baseline is:

| Concern | Current implementation | Consequence for recurrent/shared training |
|---|---|---|
| Environment | `single_symbol_env_v1` in `reinforcement_learning/environments/` | Core trading semantics are reusable. Episode orchestration must expand. |
| Observation | 12 scaled market features plus 5 dynamic portfolio features, shape `(17,)` | Compatible with an LSTM; ordering and scaling provenance must remain exact. |
| Action | `Discrete(3)`: Hold, Buy, Sell | Compatible with recurrent PPO without redesign. |
| Accounting | Long-only, all-in/all-out; next-open execution and next-close marking | Must reset at every symbol/window boundary. |
| Reward | Log portfolio growth with configured cost, drawdown, and invalid-action penalties | Reusable; experiments must keep it identical across architectures. |
| Trainer | SB3 `PPO` with `MlpPolicy`, one `DummyVecEnv`, one symbol, TRAIN only | Hard-coded model and policy types must be abstracted. |
| Evaluation | Deterministic PPO prediction on VALIDATION; Buy & Hold, Always Hold, and fixed-seed Random | Recurrent inference must carry hidden state and episode-start masks. |
| Split contract | `rl_partition_v1`, per-symbol 70/15/15 chronological split, per-symbol TRAIN scaler | Not sufficient for a shared/sector model. |
| Registry | `model_registry_v2`, scopes `symbol`/`master`, algorithm forced to `PPO` | Missing sector, recurrent, transfer-parent, and training-universe identity. |
| Readiness | 252 usable rows required today | Current readiness is stricter than the proposed six-month age policy. |

Relevant local implementation points include:

- environment version/features: `reinforcement_learning/environments/config.py`;
- reset and terminal behavior: `reinforcement_learning/environments/single_symbol_env.py`;
- fixed MLP configuration: `reinforcement_learning/training/config.py`;
- one-symbol `DummyVecEnv` trainer: `reinforcement_learning/training/ppo_trainer.py`;
- validation-only evaluator: `reinforcement_learning/evaluation/ppo_evaluator.py`;
- current contract/scaler validation: `reinforcement_learning/data_contract.py`;
- chronological split and TRAIN-fitted scaling: `feature_engineering/splitting.py`
  and `feature_engineering/preprocessing.py`; and
- registry constraints: `reinforcement_learning/model_management/registry.py`
  and `persistence.py`.

## A. Recurrent PPO compatibility

### Dependency implication

Stable-Baselines3 2.9.0 does not itself provide recurrent PPO. The compatible
ecosystem package is **`sb3-contrib==2.9.0`**, whose `RecurrentPPO` supports
`MlpLstmPolicy`, Box observations, Discrete actions, and vectorized
environments. Its 2.9.0 package metadata requires
`stable_baselines3>=2.9.0,<3.0` and Python 3.10 or newer, aligning with this
project's SB3 2.9.0 and Python 3.11 environment.

Official sources:

- [SB3-Contrib RecurrentPPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- [SB3-Contrib 2.9.0 release](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/releases/tag/v2.9.0)
- [SB3-Contrib 2.9.0 package requirements](https://raw.githubusercontent.com/Stable-Baselines-Team/stable-baselines3-contrib/v2.9.0/setup.py)

This is a recommendation only. No package was installed. SB3-Contrib describes
its algorithms as contributed/experimental, so the exact version must be
pinned, tested, and recorded in every artifact.

### Environment and episode audit

The current environment is recurrent-compatible because observations are
finite fixed-width vectors, the action is discrete, and `reset()` clears cash,
positions, cost basis, P&L, counters, and terminal flags. A natural end returns
`terminated=True`; an optional step limit returns `truncated=True`; stepping
after either state is rejected.

Required recurrent additions are outside the trading mechanics:

- reset LSTM state at every environment reset, symbol change, window change,
  and partition boundary;
- pass `episode_start` masks during prediction; a recurrent evaluator cannot
  keep calling `model.predict(observation)` without the returned state;
- never carry hidden state from TRAIN into VALIDATION, from one symbol to
  another, or across unrelated windows;
- treat either `terminated` or `truncated` as an episode boundary; and
- keep portfolio/accounting reset synchronized with the recurrent-state reset.

SB3-Contrib explicitly requires `lstm_states` and `episode_start` during
prediction. Training through a `VecEnv` handles done/reset masks, but the
project's evaluator and compatibility tests must verify them explicitly.

### Sequence behavior

`RecurrentPPO` does not expose a simple independent `sequence_length`
hyperparameter. Its rollout buffer derives recurrent sequences from rollout
and environment episode boundaries. Therefore the project must distinguish:

- `n_steps`: rollout steps **per environment**;
- `n_envs`: parallel symbol/window environments;
- total rollout buffer: `n_steps * n_envs`;
- explicit window length, if the project introduces fixed chronological
  training windows; and
- an analytical sequence/unroll target used to set minimum window size.

No burn-in should be claimed for the first version. An explicit burn-in that
excludes early recurrent outputs from the loss would require custom buffer or
training logic beyond stock SB3-Contrib and should be a later experiment.

## B. Cross-symbol architecture comparison

| Architecture | Advantages | Disadvantages / leakage risks | Compute and data | Registry impact | FYP suitability |
|---|---|---|---|---|---|
| **A. Independent LSTM-PPO per symbol** | Clean isolation; simplest extension of the existing trainer; interpretable per-symbol results | No knowledge transfer; sparse symbols remain weak; 454 full training runs; inconsistent scalers/hyperparameters can impair comparisons | Highest repeated compute; each symbol needs enough TRAIN/VALIDATION data | Existing symbol scope can extend, but needs recurrent fields | Required baseline, not the recommended main architecture |
| **B. One generalized recurrent PPO** | Maximum data reuse; one parent serves unknown symbols; efficient amortized training | Strong regime/sector heterogeneity; long histories can dominate; symbol identity may cause memorization; global temporal leakage is easy | Large balanced universe, global cutoffs, multi-env sampling | Generalized scope, universe manifest/hash, global scaler | Valuable later, too broad for the first controlled milestone |
| **C. One recurrent PPO per sector** | Shares economically related patterns; supports zero-shot new listings; smaller heterogeneity | Sparse sectors; sector labels are current rather than historical; survivorship/sector-change bias | One training job per sector; pooled sector TRAIN data | Sector scope and sector snapshot required | Strong intermediate architecture |
| **D. Sector parent -> symbol fine-tune** | Directly satisfies cold-start requirement; reuses parent checkpoints; lets real target history specialize the policy | Fine-tuning may overfit short histories or forget parent knowledge; cutoff and scaler must remain fixed | Sector pretraining once, then short target-specific runs | Parent ID, transfer stage, fine-tune range, freeze policy | **Recommended first research architecture** |
| **E. General -> sector -> symbol hierarchy** | Broad fallback for sparse/unknown sectors; richest transfer story | Most complex provenance and evaluation; multiple leakage surfaces; attribution is difficult | Highest engineering complexity despite checkpoint reuse | General, sector, symbol lineage graph | Future extension after D is validated |

Recommendation: implement D for one evidence-rich sector, with A as the
recurrent baseline and the existing MLP as the primary baseline. Insurance is
a strong first cold-start case because it has 26 mature active ordinary-equity
donors and the two short-history symbols PAKQATAR (100 usable observations) and
PQGTL (72). PAKQATAR can support the narrowly defined fine-tuning pilot; PQGTL
should initially be sector-parent zero-shot only. BLUEX has 102 observations
but Transport has only five mature active donors, making it a useful later
generalized-fallback case rather than the first sector experiment.

## C. Symbol-age and operational eligibility policy

Calendar age is technically indefensible: holidays, suspensions, illiquidity,
missing rows, invalid OHLC, and feature warm-up can make years of calendar age
contain very few usable observations. The threshold must be based on **real PSX
trading observations remaining after OHLC quality filtering and feature
warm-up**.

Recommended versioned policy:

| Category | Definition | Permitted behavior |
|---|---:|---|
| **MATURE** | `usable_post_cleaning_observations >= 126` | Eligible for recurrent symbol fine-tuning. Independent recurrent training still needs an evaluation-sufficiency gate. |
| **NEW / COLD START** | `100 <= usable observations < 126` | Sector-parent inference plus a tightly controlled fine-tuning pilot using only real rows. Report limited evidence. |
| **INSUFFICIENT FOR SYMBOL FITTING** | `< 100 usable observations` | No symbol-specific gradient update or claim. A compatible sector/general parent may operate zero-shot from the first real feature-ready observation. |

The 126 threshold approximates six PSX trading months (`21 * 6`) and is an age
gate, not a guarantee of statistical power. The 100-observation lower floor is
for the first cold-start experiment: it permits at least 60 TRAIN, 20
VALIDATION, and 20 sealed TEST observations under a proposed short-history
60/20/20 contract, with a 32-observation recurrent design horizon. This is
still limited evidence and must never be presented as equivalent to a mature
experiment.

The existing `minimum_validation_observations=126` criterion is incompatible
with a total six-month history under the current 70/15/15 split. It must not be
silently weakened. Recurrent research needs separately versioned mature and
cold-start validation criteria, and symbols without enough validation data
must receive an honest insufficient-evidence result.

### Current local universe under the proposed policy

The active recently traded ordinary-equity universe contains **473** symbols:

- **458 MATURE** (`>=126` usable observations);
- **2 NEW / COLD START fine-tuning candidates** (`100–125`): PAKQATAR (100)
  and BLUEX (102); and
- **13 INSUFFICIENT for symbol-specific fitting** (`<100`).

The age-only six-month split is 458 mature and 15 younger symbols. The current
production readiness gate remains 252 observations, producing 454 Ready and
19 Insufficient History; therefore “age mature” and “current RL-contract ready”
are intentionally different concepts.

The 13 below the cold-start fine-tuning floor are:

| Symbol | Sector | Usable observations |
|---|---|---:|
| NATM | Textile Spinning | 0 |
| SELECT | Technology & Communication | 0 |
| SLM | Automobile Parts & Accessories | 0 |
| SPAC2 | Investment Banks / Investment Companies / Securities Companies | 0 |
| SPSL | Oil & Gas Marketing Companies | 3 |
| SPAC1 | Investment Banks / Investment Companies / Securities Companies | 11 |
| WAHDAT | Food & Personal Care Products | 12 |
| ANLNV | Textile Composite | 16 |
| ARMG | Investment Banks / Investment Companies / Securities Companies | 35 |
| AWTX | Textile Spinning | 63 |
| GDL | Food & Personal Care Products | 65 |
| ITANZ | Technology & Communication | 68 |
| PQGTL | Insurance | 72 |

AWTX spans multiple calendar years but has only 63 usable observations, which
directly demonstrates why calendar days must not define maturity.

## D. Sector metadata and cold-start safety

The current company registry contains 4,741 symbol records:

- 563 are in the current official listing snapshot and all 563 have a sector;
- 4,178 are historical/not currently listed and **none has a populated local
  sector**;
- current listed types include 530 ordinary equities, 10 preference shares, 9
  ETFs, 5 GEM equities, 3 rights, and 6 other instruments; and
- historical records are largely `other` (2,473) or `unknown` (1,705).

The registry has `previous_symbol`, `successor_symbol`, and
`corporate_action_type` columns, but they currently contain no reconstructed
links. Sector membership has no effective-from/effective-to history.

Consequences:

1. Current active symbols can be mapped to a current sector reliably.
2. Historical/delisted same-sector constituents cannot be reconstructed safely
   from present local evidence.
3. Names, ticker similarity, and price behavior must not be used to invent
   historical sector labels.
4. Initial sector parents must use an explicitly versioned snapshot of symbols
   whose sector is known locally, and the report must disclose point-in-time
   and survivorship bias.
5. A later historical-sector dataset requires official, dated evidence before
   delisted constituents can enter a sector universe.

Across the active ordinary universe, 18 sectors have at least 10 mature donors
and 18 have fewer than 10. Sparse-sector symbols therefore require a
generalized fallback or an “insufficient sector evidence” status rather than
automatic pooling with an unrelated sector.

Cold-start lifecycle:

```text
versioned known-sector donor universe (TRAIN dates only)
    -> pooled sector recurrent pretraining
    -> immutable parent checkpoint
    -> new symbol's first real feature-ready row onward
    -> zero-shot inference, then optional real-row-only fine-tuning
    -> VALIDATION decision
    -> sealed TEST only for the final selected experiment
```

No step inserts pre-listing rows, interpolates missing company prices, or
copies a donor's prices into the target.

## E. Multi-symbol training-data design

Each vectorized environment should own one symbol/window episode. A sampler
selects a donor symbol and a contiguous TRAIN-only window, creates a fresh
environment, and resets both accounting and recurrent state. Symbols must not
be concatenated into one apparently continuous time series.

Recommended first design:

- symbol-balanced sampling so a long history does not dominate the gradient;
- shuffled episode order across environments, while each episode remains
  strictly chronological;
- fixed or bucketed window lengths with enough steps for recurrent context;
- explicit symbol and date ranges in the sampled-universe manifest;
- no cross-partition windows; and
- a global research cutoff or rolling-origin cohort cutoff so one donor's
  future cannot leak into another target's earlier validation period.

Normalization should be **pooled sector TRAIN-only scaling**, fitted with
symbol-balanced contributions. Real execution OHLCV remains unscaled. The
parent scaler is frozen for zero-shot and fine-tuned descendants. A future
generalized model should use a balanced global TRAIN-only scaler. The current
per-symbol scaler architecture cannot support sector/generalized transfer
safely because different symbols map the same observation to different scales
and a new symbol has no mature scaler history.

No symbol identity feature is needed for the first sector model. A generalized
model may later compare no identity, a sector category, and a learned
symbol/sector embedding through a custom feature extractor. Adding a 454-way
symbol identity immediately risks memorization and has no natural value for an
unseen ticker.

## F. Proposed recurrent contract

Create **`rl_recurrent_partition_v1`** rather than silently extending
`rl_partition_v1`. The existing contract describes one symbol and one
per-symbol scaler; changing those meanings in place would make old artifacts
ambiguous.

Required fields:

- schema, feature, environment, PPO/recurrent-policy, and library versions;
- unchanged real execution columns: date, symbol, open, high, low, close,
  volume and required accounting inputs;
- exact ordered 12 market observation features and 5 portfolio features;
- recurrent policy class, hidden size, layer count, actor/critic/shared-LSTM
  settings;
- `n_steps`, `n_envs`, total rollout size, sampler/window length, minimum
  episode length, and batch divisibility;
- an explicit `burn_in=none` statement for v1;
- symbol/window boundary rules and recurrent `episode_start` mask rules;
- accounting and LSTM reset invariants;
- sector code/name, metadata snapshot timestamp/source/hash, and unknown-sector
  policy;
- training-universe manifest, ordered symbols, per-symbol real first/last
  dates, row counts, global/cohort cutoffs, and universe hash;
- TRAIN/VALIDATION/TEST boundaries per symbol, with TEST sealed;
- normalization scope (`sector` or `global`), fit partition (`train`), donor
  symbols, symbol-balancing method, fitted rows, artifact paths, and hashes;
- parent/pretrained model ID, fine-tuned-from ID, transfer stage, frozen layers,
  and target fine-tune range where applicable; and
- source raw/processed/contract hashes sufficient to detect rebuild drift.

The current observation/action/reward semantics can remain versioned as
`single_symbol_env_v1` for single-symbol episodes. A separate multi-symbol
sampler/orchestrator version should describe how those episodes are selected;
the environment itself should not infer sector membership.

## G. Efficient training architecture

1. Begin with 4–8 deterministic `DummyVecEnv` sector environments to validate
   resets, sampling, and buffer mathematics.
2. Benchmark 8–16 environments and, on CPU, compare `SubprocVecEnv` only after
   deterministic correctness is established.
3. Select `n_steps` and `n_envs` together; ensure `batch_size` divides or
   sensibly partitions `n_steps * n_envs`.
4. Reuse immutable sector checkpoints. Fine-tuning should update the target
   from a selected parent instead of retraining the sector universe.
5. Save explicit parent/child artifacts and never overwrite a checkpoint.
6. Use VALIDATION-only checkpoint selection or early stopping. TEST cannot
   drive stopping, architecture, hyperparameters, or parent selection.
7. Balance episode selection and report per-symbol contribution counts.

The prior MCB result that CPU was approximately 6.81x faster than MPS applies
only to the current one-environment MLP workload. A recurrent benchmark must
use identical sector universe, windows, `n_envs`, seed, scaler, `n_steps`, and
total transitions on both devices; include a warm-up; synchronize MPS before
and after timing; reject `PYTORCH_ENABLE_MPS_FALLBACK`; record throughput,
duration, memory, and actual policy parameter device; and repeat enough seeds
or runs to show variability.

## H. Fair research evaluation

Compare, without changing environment economics:

1. current MLP PPO;
2. independent recurrent PPO;
3. sector-pretrained recurrent PPO used zero-shot; and
4. the same sector parent fine-tuned on the target symbol.

For every target, use the same real rows, chronological/cohort cutoffs,
transaction costs, slippage, initial cash, reward, seeds, and evaluation
baselines. Donor pretraining consumes donor TRAIN only. Fine-tuning consumes
target TRAIN only. Architecture and checkpoint decisions consume VALIDATION
only. Buy & Hold, Always Hold, and fixed-seed Random run on the identical
target evaluation frame. The final TEST is opened once for the selected,
pre-registered experiment and is never returned to tuning.

Use multiple seeds and report distributions/confidence intervals, not one best
seed. Compare transfer benefit against both MLP and independent recurrent
baselines. A positive return or a validation pass is not evidence that sector
transfer caused the improvement.

## I. Registry and artifact impact

`model_registry_v2` cannot fully express the proposed lineage. It permits only
`symbol` and `master` scopes, validates the algorithm as exactly `PPO`, and the
production persistence path is fixed to a symbol candidate and SB3 `PPO`.

Use an explicit `model_registry_v3` migration while preserving v2 records.
Add:

- scopes: `symbol`, `sector`, `generalized` (retain legacy `master`);
- `algorithm_family=PPO`, `algorithm_variant=mlp|recurrent`, and policy class;
- recurrent architecture/config version, hidden size, layers, shared/critic
  LSTM flags, rollout/window parameters, and `n_envs`;
- sector identifier/name and sector-metadata snapshot source/hash;
- `parent_model_id`, `pretrained_model_id`, `fine_tuned_from_model_id`, transfer
  stage, frozen-layer policy, and target fine-tune range;
- universe ID, ordered-symbol manifest path/hash, symbol count, date/cohort
  cutoffs, episode-sampling policy, and per-symbol contribution summary;
- normalization scope, scaler-contract version, scaler donor universe, and
  scaler hashes;
- recurrent contract path/hash and source-data hashes; and
- dependency versions, device, seed, and full validation-criteria version.

Artifact paths need `sector_models/<sector>/vNNNN` and
`generalized_models/vNNNN` roots in addition to symbol bundles. Persistence,
manifest validation, exact loading, compatibility checks, and prediction types
must accept `RecurrentPPO` without weakening the current atomic publication,
hash, candidate-only, or sealed-TEST guarantees.

## Risks and go/no-go gates

| Risk | Required gate |
|---|---|
| Historical sector survivorship or sector-change leakage | Do not include a historical constituent without dated official membership evidence; version the snapshot. |
| Cross-symbol temporal leakage | Global/cohort cutoff tests; each sampled row date must be permitted for that target evaluation cutoff. |
| Hidden-state contamination | Boundary tests proving state and accounting reset on every symbol/window/partition transition. |
| Long-history dominance | Symbol-balanced sampling and contribution audit. |
| Normalization leakage | Scaler fitted only on permitted donor TRAIN rows; immutable provenance hashes. |
| Short-history overclaim | Separate age, fine-tuning, and evaluation sufficiency statuses. |
| TEST reuse | Metadata-only seal until the single final experiment. |
| Registry ambiguity | Explicit v3 migration and lineage-compatible immutable bundle. |

## Proposed implementation milestones

1. **6A — architecture and contract hardening:** pin the proposed dependency;
   implement recurrent interfaces, recurrent result types, state-reset tests,
   global/cohort split design, and `rl_recurrent_partition_v1` builder without
   training a universe.
2. **6B — single-symbol recurrent baseline:** train/evaluate one recurrent
   symbol with stateful VALIDATION prediction and compare it with the MLP
   baseline.
3. **6C — sector universe builder:** create versioned point-in-time donor
   manifests, sector sample-size gates, pooled TRAIN-only scalers, and leakage
   diagnostics.
4. **6D — sector recurrent pretraining:** train one evidence-rich sector parent
   using balanced boundary-reset episodes and multiple seeds.
5. **6E — cold-start transfer:** zero-shot and fine-tune PAKQATAR from the
   Insurance parent; keep PQGTL as an honest insufficient/zero-shot case.
6. **6F — generalized and vectorized efficiency:** add generalized fallback,
   multi-env benchmarking, and recurrent CPU/MPS evidence.
7. **6G — comparative evaluation:** execute the pre-defined MLP, independent
   recurrent, sector zero-shot, and sector fine-tuned VALIDATION study; open
   TEST only for the selected final experiment.
8. **6H — registry/persistence and UI consolidation:** migrate registry/bundles
   and expose only safe recurrent/transfer metadata and explicit actions.
9. **6I — controlled pilot:** run a small sector/symbol cohort with documented
   go/no-go criteria before any broad universe training.

## Data-integrity statement

This audit did not install `sb3-contrib`, train or evaluate a model, load TEST
for inference, create a model artifact, modify the model registry, alter raw or
processed data, or change production code. All counts are observational and
must be recomputed by the future versioned universe builder at implementation
time.

## Verification

- Complete test suite: **436 passed, 1 skipped** from 437 collected tests.
  The skip is the existing hardware-gated MPS smoke test.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: **No broken requirements found**. The pip
  cache emitted a non-fatal permissions warning and was disabled.
- A path/size/mtime manifest of every file under `data/` was identical before
  and after verification.
- No commit was created.
