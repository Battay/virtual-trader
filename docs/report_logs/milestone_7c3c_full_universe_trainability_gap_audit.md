# Milestone 7C.3c — Full-Universe Recurrent Trainability Gap Audit

Audit timestamp: 2026-08-26T20:47:23Z  
Audit version: `recurrent_trainability_gap_audit_v1`  
Frozen identity universe: `current_common_equity_universe_v1`  
Universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`  
Decision: **BLOCKED_TRAINABILITY_GAPS**

## Outcome

All 508 frozen identities reconcile exactly: 432 have compatible Mature
single-symbol recurrent contracts and 76 are explicitly audited below. No
identity was silently discarded.

The audit found 68 data/policy-limited identities and eight pipeline-limited
identities. The eight pipeline gaps are not scientifically defensible permanent
exclusions, so full-universe training preparation remains blocked pending a
separately reviewed canonical TRAIN-only artifact path. No model was trained and
no recurrent, feature, split, registry, source Parquet, VALIDATION-value, or
TEST-value artifact was created or changed.

## Methodology and sealed-data boundary

- Membership and pre-audit categories came from the frozen authoritative
  508-stock identity universe and the existing recurrent orchestrator discovery.
- Only `market_date` and `symbol` identity metadata were read across the full
  consolidated Parquet history. No price, return, or volume value after the
  TRAIN cutoff entered memory.
- Each non-trainable symbol's candidate TRAIN boundary is the first 70% of its
  distinct chronological raw market dates, matching the current symbol split
  proportion. Market values were then predicate-loaded only through that
  inclusive cutoff.
- `train_row_count` is the count remaining inside that TRAIN prefix after the
  existing OHLC quality rule, causal feature calculation, 49-row warm-up, and
  required-feature completeness checks.
- Active-span coverage uses actual global PSX market dates between the symbol's
  first observation and candidate TRAIN end, not calendar days.
- Existing `usable_observations` metadata is retained where the already-built
  readiness audit evaluated complete usable history. This is provenance, not a
  new load of later market values.
- The audit is descriptive. It does not lower the Mature threshold of 126,
  reclassify Cold Start, infer aliases, or fabricate an artifact.

The exact deterministic 76-row table is in
`docs/report_logs/milestone_7c3c_recurrent_trainability_gap_audit.csv`.

## Before and precise root-cause breakdown

| View | Category | Count |
|---|---|---:|
| Before | Eligible/trainable | 432 |
| Before | Insufficient / Cold Start | 15 |
| Before | Missing or incompatible recurrent artifacts | 4 |
| Before | Unsupported | 57 |
| After audit | `NO_RECENT_TRADING_ACTIVITY` | 53 |
| After audit | `INSUFFICIENT_HISTORY` | 13 |
| After audit | `COLD_START` | 2 |
| After audit | `FEATURE_BUILD_GAP` | 4 |
| After audit | `LEGACY_PIPELINE_ONLY` | 4 |

Limitation accounting:

| Limitation | Count | Interpretation |
|---|---:|---|
| `DATA_LIMITED` | 68 | 13 insufficient, two Cold Start, and 53 without recent trading evidence |
| `PIPELINE_LIMITED` | 8 | Four Mature ordinary equities blocked by the legacy 252-row build gate and four authoritative GEM equities blocked by the ordinary-only gate |
| `CONTRACT_LIMITED` | 0 | No stale/incompatible contract file was found; the four broad cases are absent contracts caused upstream |
| `IDENTITY_LIMITED` | 0 | Every audited identity has an exact canonical Parquet symbol match |

## Four missing/incompatible cases

All four recurrent contracts and processed feature artifacts are absent. None
contains an incompatible existing contract. Existing evidence classifies all
four as Mature, but the old base feature builder requires 252 usable rows before
it creates the split/RL artifacts on which recurrent generation depends.

| Symbol | Existing usable | Candidate TRAIN usable | Candidate TRAIN span | Canonical data sufficient under current Mature policy | Root cause |
|---|---:|---:|---|---|---|
| ASIC | 221 | 97 | 2025-01-14 to 2025-12-24 | Yes | Legacy 252-row base-artifact gate |
| EWIC | 231 | 95 | 2021-12-13 to 2025-09-15 | Yes | Legacy 252-row base-artifact gate after substantial OHLC-quality removals |
| MUGHALC | 225 | 143 | 2025-08-26 to 2026-04-02 | Yes | Legacy 252-row base-artifact gate |
| ZUMA | 147 | 88 | 2026-01-02 to 2026-05-11 | Yes | Legacy 252-row base-artifact gate |

These are `FEATURE_BUILD_GAP` / `PIPELINE_LIMITED`, not insufficient history.
The safe future correction is a versioned canonical Parquet-to-TRAIN feature,
scaler, base RL, and recurrent artifact path that never needs to expose sealed
later values during training preparation. Implementing that boundary is a
substantial pipeline change and was intentionally not improvised in this audit.

## The 57 previously unsupported identities

### 53 without recent trading evidence

The legacy `UNSUPPORTED` label hid a precise current-activity rule. These are
authoritative current identity members, but the registry does not classify them
as `recently_traded`; the canonical Parquet confirms their last observations
predate the current dataset tail. They remain explicitly tracked but are not
approved for current independent recurrent agents.

AAL, AASM, ABSON, AMSL, ANNT, APOT, AZMT, BIIC, DBSL, DCTL, DKTM, DMTM, DSFL,
ENGL, FAEL, FTHM, GIL, GLOT, GUTM, HADC, HAJT, HATM, HKKT, HMIM, HSPI, KAKL,
MFTM, MOHE, MUBT, NAFL, NINA, NMFL, PHDL, PICL, PRIB, PRIC, REGAL, SANE, SCHT,
SDOT, SFAT, SGABL, SHCI, SLCL, SLL, SPCL, SSIC, SURAJ, SWL, TAJT, TRIBL, USMT,
ZELP.

Fifteen have at least 126 audited candidate TRAIN rows—DKTM, DMTM, DSFL, HADC,
HMIM, HSPI, MFTM, MUBT, PHDL, SANE, SFAT, SLCL, SLL, SPCL, and TRIBL—but the
blocking fact is stale current activity, not historical row quantity. The other
38 also fall below 126 candidate TRAIN rows. The rule should be revisited only
if authoritative trading resumes; old history alone does not establish a
currently executable symbol agent.

### Four GEM common equities

| Symbol | Existing usable | Candidate TRAIN usable | Candidate TRAIN span | Root cause |
|---|---:|---:|---|---|
| GEMMEL | 208 | 126 | 2024-10-17 to 2025-12-31 | Legacy ordinary-equity-only builder |
| GEMNETS | 149 | 90 | 2025-08-07 to 2026-02-13 | Legacy ordinary-equity-only builder |
| GEMPACRA | 135 | 80 | 2025-09-24 to 2026-03-18 | Legacy ordinary-equity-only builder |
| GEMPAPL | 381 | 240 | 2022-03-25 to 2025-06-18 | Legacy ordinary-equity-only builder |

All four are authoritatively inside the frozen common-equity identity universe
and have at least 126 existing usable observations. Their exclusion is
`LEGACY_PIPELINE_ONLY` / `PIPELINE_LIMITED`, not an instrument-identity failure.
They require an explicit research decision and a contract-compatible TRAIN-only
GEM artifact path; the audit does not silently treat GEM execution semantics as
ordinary-board semantics.

## The 15 insufficient / Cold Start identities

| Symbol | Class | Existing usable | Candidate TRAIN usable | Required | Deficit to Mature |
|---|---|---:|---:|---:|---:|
| BLUEX | Cold Start | 102 | 57 | 126 | 24 |
| PAKQATAR | Cold Start | 100 | 56 | 126 | 26 |
| ANLNV | Insufficient | 16 | 0 | 126 | 110 |
| ARMG | Insufficient | 35 | 10 | 126 | 91 |
| AWTX | Insufficient | 63 | 0 | 126 | 63 |
| GDL | Insufficient | 65 | 31 | 126 | 61 |
| ITANZ | Insufficient | 68 | 0 | 126 | 58 |
| NATM | Insufficient | 0 | 0 | 126 | 126 |
| PQGTL | Insufficient | 72 | 36 | 126 | 54 |
| SELECT | Insufficient | 0 | 0 | 126 | 126 |
| SLM | Insufficient | 0 | 0 | 126 | 126 |
| SPAC1 | Insufficient | 11 | 0 | 126 | 115 |
| SPAC2 | Insufficient | 0 | 0 | 126 | 126 |
| SPSL | Insufficient | 3 | 0 | 126 | 123 |
| WAHDAT | Insufficient | 12 | 0 | 126 | 114 |

Zero candidate TRAIN rows means the available prefix does not survive the
existing quality/49-row warm-up/feature-completeness contract; it does not mean
the source contains no identifier rows. Relaxing the threshold is not justified.
BLUEX and PAKQATAR remain Cold Start and are not eligible for independent
pretraining.

## Narrow corrective implementation and orchestrator impact

The implementation adds a read-only symbol/date inventory method and a
versioned gap-audit module. It replaces the diagnostic catch-all with evidenced
categories, verifies complete accounting, distinguishes limitation types, and
asserts that every market-value load ends at its declared TRAIN cutoff.

No feature or recurrent artifact was regenerated. Therefore no symbol changed
to trainable status in this milestone and discovery remains 432 trainable plus
76 explicit ineligible records. The existing orchestrator already creates a
persistent job record for all 508 identities, so status tracking and isolated
paths are unchanged. The eight future-recoverable pipeline cases will
automatically require normal compatible job discovery after their artifacts are
built; no special job-state transition is needed.

## Final accounting and decision

| Final state | Count |
|---|---:|
| Compatible trainable | 432 |
| Data-limited / current-activity-limited | 68 |
| Pipeline-limited | 8 |
| Contract-limited | 0 |
| Identity-limited | 0 |
| Total | 508 |

**BLOCKED_TRAINABILITY_GAPS**

The 68 data/policy exclusions are explicit and defensible under current rules.
Preparation is nevertheless blocked by the eight pipeline-limited identities.
Before full-universe training preparation can be declared ready, the project
must either implement and validate the canonical TRAIN-only artifact route for
the four Mature ordinary equities and make an explicit GEM compatibility
decision, or formally version and justify their exclusion. This decision does
not authorize full-universe training.
