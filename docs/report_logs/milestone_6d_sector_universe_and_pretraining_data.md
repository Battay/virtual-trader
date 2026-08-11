# Milestone 6D — Sector Universe and Pretraining Data Foundation

Audit/build date: 2026-08-12 (Asia/Karachi)

Code-base commit at generation: `3b7a41b9271af7f310b66c79e748e1f9ee7c82f3`

## Objective and boundary

Milestone 6D creates a research-safe, versioned sector taxonomy, current-sector
evidence table, standard sector-universe manifests, and TRAIN-only episode
indexes for a future sector RecurrentPPO. It does not train, fine-tune,
evaluate, persist, register, or promote a model. It does not alter the
single-symbol MLP/RecurrentPPO results or their contracts.

The generated universes are explicitly **current-sector grouped research
cohorts**. They are not presented as proof that a company remained in the same
sector throughout its 2016–2026 observations.

Versions:

- taxonomy: `psx_sector_taxonomy_v1`;
- manifest: `sector_universe_v1`;
- episode index: `sector_train_episode_index_v1`;
- normalization metadata: `per_symbol_train_scaler_v1`;
- sampling metadata: `equal_symbol_episode_sampling_v1`;
- source recurrent contract: `rl_recurrent_partition_v1`.

## Local sector and instrument evidence audit

### Sources inspected

| Source | Evidence available | Safe interpretation |
|---|---|---|
| `data/metadata/listings/current_listings.csv` | 563 symbols; company, raw PSX sector, board/segment, listing status, snapshot date 2026-08-02, official source URL | Locally authoritative current listing/sector snapshot |
| `listings_2026-07-30.csv` and `listings_2026-08-02.csv` | 563 symbols in each; company/security/sector/status content is identical | Two close current snapshots, not a historical taxonomy archive |
| `data/master/company_registry.csv` | 4,741 unique symbols; current listing metadata merged with observed market dates | Current sector evidence for 563; no sector for 4,178 historical-only records |
| `data/master/psx_master.csv` | 1,528,985 market rows and 4,741 symbols; OHLCV/date only | Price-history evidence, not instrument or sector evidence |
| `data/processed/master/psx_ai_master.csv` | Feature/security metadata derived from the current registry | Cannot independently verify historical sector membership |
| 454 recurrent contracts | Current registry sector copied with source hash; TRAIN/VALIDATION metadata | Valid contract compatibility and current-sector provenance; not historical membership proof |
| Company override/alias evidence | No local override file; historical registry alias fields are empty | No safe historical alias/sector reconstruction |
| Git-tracked data history | Registry/listing data are ignored; no older committed sector snapshots | Cannot recover historical sectors from repository history |

### Verified, inferred, missing

- **Locally verified:** raw current sector labels and current listing status from
  the cached official PSX listing tables.
- **Code-inferred:** `security_type` is conservatively inferred by the listing
  parser from board, company name, sector, and symbols. It is not treated as an
  explicit historical PSX instrument record.
- **Normalized:** 38 official raw sector labels map through explicit aliases;
  unknown labels fail to `unknown` and are never slug-guessed into a sector.
- **Missing:** legal listing dates, delisting dates for historical-only rows,
  historical sector-effective intervals, historical company names/types, and
  historical ticker aliases.
- **Unverifiable locally:** sector membership for all 4,178 historical-only
  symbols and sector membership of current companies at old TRAIN dates.

## Historical-only instrument classification

The 4,178 historical-only records are not 4,178 historical companies.

| Classification | Count | Equity-sector treatment |
|---|---:|---|
| Month-coded / contract symbols | 3,263 | Excluded |
| Odd-lot segment securities (`-ODL`) | 487 | Excluded |
| Rights / security-entitlement patterns | 136 | Excluded |
| Debt/preference/other instrument markers | 15 | Excluded |
| Bare symbols with no local company/type evidence | 277 | Unknown; authoritative investigation required |
| Locally verified historical ordinary-equity candidates | **0** | None can be sector-mapped safely yet |

Thus 3,901 are pattern-supported non-equity/contract-like exclusions and 277
remain unknown. The 277 are not promoted to ordinary equities merely because
their ticker looks simple. Historical GEM, preference, ETF, REIT, and ordinary
equity status cannot be established from current local files.

For comparison, the 563 current official records are retained distinctly as:

- 530 ordinary equities;
- 5 GEM equities;
- 10 preference shares;
- 9 ETFs;
- 6 REITs;
- 3 rights/security entitlements.

Only active ordinary equities can enter independent sector pretraining.

## Taxonomy design

`psx_sector_taxonomy_v1` contains 38 explicit canonical IDs and display names.
Normalization removes harmless whitespace/case/punctuation differences and
accepts only declared aliases. It does not merge genuinely distinct sectors.
For example, Oil & Gas Exploration and Oil & Gas Marketing remain separate;
ETFs and REITs remain instrument sectors and do not enter the ordinary-equity
research cohort.

The taxonomy hash is derived from sorted canonical content. `generated_at`,
machine paths, and host information do not affect it.

## Cohort and survivorship controls

The generated research snapshot records:

- observed-data cohort cutoff: **2026-08-07**;
- official sector snapshot: **2026-08-02**;
- first observed date;
- first usable date;
- TRAIN start/end;
- TRAIN rows available at the cutoff;
- `eligible_at_cutoff`;
- current/cutoff/historical verification flags.

A symbol is excluded if it first appears after the cutoff or if its complete
canonical TRAIN partition extends past the cutoff. The generator does not
silently trim a source while retaining a scaler fitted on later data.

The listing snapshot is five days earlier than the observed-data cutoff.
Therefore `sector_verified_current=true`, but
`sector_verified_at_cutoff=false`. Every constituent also records:

- `historical_sector_membership_verified=false`;
- `historical_sector_membership_unknown=true`;
- `sector_changed_over_time=unknown_no_local_evidence`.

Current membership is not back-projected to 2016. Universe selection uses
only current official metadata, TRAIN availability/quality, and contract
compatibility. VALIDATION performance and TEST are irrelevant to membership.

## Eligibility and diversity policy

A pretraining constituent must be:

1. a current, recently traded ordinary equity;
2. in a recognized official current sector;
3. Mature under the approved history policy;
4. backed by a valid current recurrent contract;
5. fully TRAIN-available on or before the cohort cutoff;
6. homogeneous in feature, recurrent contract, environment, observation
   ordering/shape, execution semantics, and normalization provenance.

A sector is `READY_FOR_SECTOR_RESEARCH` only when it has at least five
independent compatible Mature symbols, at least 5,120 TRAIN rows, median TRAIN
depth of at least 252, no more than 40% proportional concentration in one
symbol, verified current sector evidence, and homogeneous contracts. Five
symbols preserve four peers after a future target exclusion; 5,120 rows align
with the measured recurrent benchmark scale; one-year median depth avoids a
row-rich but shallow cohort.

`LIMITED` requires at least three compatible symbols and 1,000 TRAIN rows.
Smaller or instrument-only sectors are `INSUFFICIENT`. Historical claims remain
`LIMITED_CURRENT_SECTOR_ONLY` regardless of row count.

## Full sector readiness summary

Overall:

- 38 canonical sectors;
- 454 approved Mature recurrent constituents;
- 556,586 referenced TRAIN rows;
- 26 READY, 5 LIMITED, 7 INSUFFICIENT;
- all 454 approved contracts are homogeneous under the current versions.

| Sector | Verified | Mature | Cold | Insuff. | Compatible | TRAIN rows | Median | Max share | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Textile Spinning | 56 | 42 | 0 | 2 | 42 | 38,249 | 838.5 | 4.4% | READY_FOR_SECTOR_RESEARCH |
| Textile Composite | 49 | 36 | 0 | 1 | 36 | 36,580 | 1,039.0 | 4.7% | READY_FOR_SECTOR_RESEARCH |
| Investment Banks / Investment Companies / Securities Companies | 38 | 31 | 0 | 3 | 31 | 36,445 | 1,152.0 | 4.7% | READY_FOR_SECTOR_RESEARCH |
| Sugar & Allied Industries | 29 | 28 | 0 | 0 | 28 | 29,761 | 1,133.5 | 5.6% | READY_FOR_SECTOR_RESEARCH |
| Chemical | 26 | 24 | 0 | 0 | 24 | 32,975 | 1,491.5 | 5.2% | READY_FOR_SECTOR_RESEARCH |
| Insurance | 33 | 26 | 1 | 1 | 24 | 30,002 | 1,288.5 | 5.7% | READY_FOR_SECTOR_RESEARCH |
| Food & Personal Care Products | 27 | 23 | 0 | 2 | 23 | 29,252 | 1,351.0 | 5.8% | READY_FOR_SECTOR_RESEARCH |
| Commercial Banks | 19 | 19 | 0 | 0 | 19 | 31,157 | 1,701.0 | 5.5% | READY_FOR_SECTOR_RESEARCH |
| Modarabas | 23 | 19 | 0 | 0 | 19 | 18,739 | 859.0 | 8.6% | READY_FOR_SECTOR_RESEARCH |
| Cement | 20 | 18 | 0 | 0 | 18 | 28,328 | 1,697.0 | 6.0% | READY_FOR_SECTOR_RESEARCH |
| Technology & Communication | 22 | 19 | 0 | 2 | 18 | 23,686 | 1,698.0 | 7.2% | READY_FOR_SECTOR_RESEARCH |
| Miscellaneous | 21 | 17 | 0 | 0 | 17 | 15,530 | 787.0 | 11.0% | READY_FOR_SECTOR_RESEARCH |
| Power Generation & Distribution | 16 | 15 | 0 | 0 | 15 | 22,645 | 1,680.0 | 7.5% | READY_FOR_SECTOR_RESEARCH |
| Engineering | 19 | 16 | 0 | 0 | 15 | 20,774 | 1,607.0 | 8.2% | READY_FOR_SECTOR_RESEARCH |
| Pharmaceuticals | 14 | 14 | 0 | 0 | 14 | 17,529 | 1,387.5 | 9.7% | READY_FOR_SECTOR_RESEARCH |
| Automobile Parts & Accessories | 12 | 11 | 0 | 1 | 11 | 13,042 | 1,472.0 | 13.1% | READY_FOR_SECTOR_RESEARCH |
| Automobile Assembler | 10 | 10 | 0 | 0 | 10 | 16,545 | 1,703.0 | 10.3% | READY_FOR_SECTOR_RESEARCH |
| Paper, Board & Packaging | 13 | 10 | 0 | 0 | 10 | 14,454 | 1,624.5 | 11.7% | READY_FOR_SECTOR_RESEARCH |
| Oil & Gas Marketing Companies | 10 | 9 | 0 | 1 | 9 | 14,687 | 1,704.0 | 11.6% | READY_FOR_SECTOR_RESEARCH |
| Glass & Ceramics | 9 | 8 | 0 | 0 | 8 | 11,451 | 1,699.0 | 14.9% | READY_FOR_SECTOR_RESEARCH |
| Cable & Electrical Goods | 7 | 7 | 0 | 0 | 7 | 9,362 | 1,471.0 | 18.2% | READY_FOR_SECTOR_RESEARCH |
| Leather & Tanneries | 6 | 6 | 0 | 0 | 6 | 5,797 | 1,097.0 | 28.5% | READY_FOR_SECTOR_RESEARCH |
| Textile Weaving | 9 | 6 | 0 | 0 | 6 | 5,450 | 830.5 | 31.2% | READY_FOR_SECTOR_RESEARCH |
| Fertilizer | 6 | 5 | 0 | 0 | 5 | 8,399 | 1,703.0 | 20.3% | READY_FOR_SECTOR_RESEARCH |
| Transport | 6 | 5 | 1 | 0 | 5 | 6,101 | 1,563.0 | 27.9% | READY_FOR_SECTOR_RESEARCH |
| Property | 6 | 5 | 0 | 0 | 5 | 5,400 | 1,359.0 | 31.5% | READY_FOR_SECTOR_RESEARCH |
| Synthetic & Rayon | 9 | 5 | 0 | 0 | 5 | 4,928 | 1,042.0 | 24.7% | LIMITED |
| Oil & Gas Exploration Companies | 4 | 4 | 0 | 0 | 4 | 6,816 | 1,704.0 | 25.0% | LIMITED |
| Refinery | 4 | 4 | 0 | 0 | 4 | 6,816 | 1,704.0 | 25.0% | LIMITED |
| Apparel | 5 | 4 | 0 | 0 | 4 | 2,994 | 774.0 | 35.6% | LIMITED |
| Closed-End Mutual Funds | 4 | 3 | 0 | 0 | 3 | 3,601 | 1,165.0 | 38.6% | LIMITED |
| Tobacco | 2 | 2 | 0 | 0 | 2 | 2,784 | 1,392.0 | 55.6% | INSUFFICIENT |
| Vanaspati & Allied Industries | 3 | 2 | 0 | 0 | 2 | 2,316 | 1,158.0 | 51.6% | INSUFFICIENT |
| Leasing Companies | 8 | 2 | 0 | 0 | 2 | 1,801 | 900.5 | 56.9% | INSUFFICIENT |
| Jute | 2 | 2 | 0 | 0 | 2 | 800 | 400.0 | 67.0% | INSUFFICIENT |
| Woollen | 1 | 1 | 0 | 0 | 1 | 1,390 | 1,390.0 | 100.0% | INSUFFICIENT |
| Exchange Traded Funds | 9 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | INSUFFICIENT |
| Real Estate Investment Trust | 6 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | INSUFFICIENT |

### Representative verification

| Sector | Verified | Mature | Cold | Compatible | TRAIN rows | Historical evidence | Status |
|---|---:|---:|---:|---:|---:|---|---|
| Commercial Banks | 19 | 19 | 0 | 19 | 31,157 | Current sector only | READY |
| Oil & Gas Exploration | 4 | 4 | 0 | 4 | 6,816 | Current sector only | LIMITED: only four independent peers |
| Textile Composite | 49 | 36 | 0 | 36 | 36,580 | Current sector only | READY |
| Insurance | 33 | 26 | 1 | 24 | 30,002 | Current sector only | READY; PAKQATAR Cold Start |
| Transport | 6 | 5 | 1 | 5 | 6,101 | Current sector only | READY; BLUEX Cold Start |

## Manifest and episode-index design

Each sector directory contains:

```text
universe_manifest.json
train_episode_index.json
```

The manifest records taxonomy/cohort/version identity, exact approved symbols,
exclusions and reasons, security/history classes, current/historical
verification flags, TRAIN bounds/rows, compatibility metadata, scaler/source
hashes, normalization contributors, sampling policy, target exclusion fields,
source registry/listing hashes, Git commit, and deterministic universe hash.
It contains no validation/test values or performance metrics.

All artifact paths are project-relative. The universe hash excludes
`generated_at`, absolute paths, machine identity, and Git working-directory
details. It changes if constituents, exclusions, cutoff, taxonomy, contract,
feature, environment, normalization, sampling, target, or source registry hash
changes. Regeneration of the same canonical identity returns the same hash.

### Standard versus transfer mode

- `standard_sector_pretraining`: all approved constituents; reusable sector
  foundation model.
- `leave_one_symbol_out_transfer`: explicit target; target is absent from both
  `pretraining_constituent_symbols` and normalization contributors.

Any future statement that other companies helped target X must use the second
mode. The manifest rejects a target appearing in its own pretraining or scaler
universe.

### Symbol and portfolio isolation

The episode index never concatenates price frames. Each entry references one
canonical `train_rl.csv` and begins with:

- `episode_start=true`;
- environment reset;
- cash/holdings/realized-P&L reset;
- portfolio peak/drawdown reset;
- recurrent hidden-state reset.

No validation or TEST path is included. Symbol A can never transition directly
into symbol B with portfolio or LSTM state intact.

## Sampling fairness

Every episode records two contributions:

- proportional row share: `symbol TRAIN rows / sector TRAIN rows`;
- equal-symbol selection share: `1 / constituent count`.

The first 6E baseline should select symbols uniformly and enforce equal planned
symbol exposure. Full-partition episode lengths still differ, so 6E must report
both episode-selection frequency and actual environment-step contribution. If
long histories remain dominant in step count, use a predeclared equal per-symbol
step cap/window schedule while preserving resets; do not introduce opaque
prioritized sampling yet.

## Normalization decision

The initial foundation references the existing per-symbol, TRAIN-fitted
observation scalers. This is the safest 6D/first-6E baseline because it:

- preserves existing real-price execution fields;
- avoids future rows;
- avoids cross-company price-scale domination;
- requires no replacement of validated symbol contracts;
- lets leave-one-out pretraining exclude the target completely.

A later sector-wide scaler may improve a common representation, but it must be
fit only on approved cohort TRAIN rows. A future listing or excluded target may
not influence its statistics. Global normalization has the largest leakage and
heterogeneity surface and is not recommended first.

No sector identity or high-cardinality symbol embedding is recommended for the
first sector model. Sector identity is constant within one model; symbol
identity could encourage memorization. Revisit learned identity only after an
identity-free baseline.

## Cold Start and pseudo-Cold-Start feasibility

Current Cold Start symbols:

| Symbol | Sector | Usable rows | Usable dates | Current standalone design |
|---|---|---:|---|---|
| BLUEX | Transport | 102 | 2026-03-09 to 2026-08-07 | No canonical split; not enough independent validation evidence |
| PAKQATAR | Insurance | 100 | 2026-03-11 to 2026-08-07 | No canonical split; not enough independent validation evidence |

Under the current 70/15/15 methodology, 100 rows yield 70 TRAIN, 15 VALIDATION,
15 sealed TEST rows; 102 yield 71/15/16. A 15-row validation episode is too
short for a statistically meaningful standalone trading claim, and neither
symbol has a canonical base RL contract because the production gate remains
252 rows.

Safe alternatives are limited TRAIN-only fine-tuning followed by delayed
evaluation after more real post-listing observations, or a predeclared
rolling/expanding validation design once minimum history is reached. TEST must
not be opened to compensate.

Future pseudo-Cold-Start experiments should use multiple Mature targets:

1. exclude the target from sector pretraining and sector-scaler fitting;
2. pretrain only on verified peer TRAIN episodes;
3. truncate the target to its first predeclared 100–125 real usable rows;
4. fine-tune on only those rows;
5. evaluate on chronologically later target VALIDATION observations;
6. compare scratch-on-truncated-history, pretrained-plus-fine-tuned, Buy &
   Hold, and fixed baselines across multiple seeds/targets.

No pre-listing rows may be synthesized. These experiments are not run in 6D.

## Four Mature source-contract exclusions

No symbol was repaired merely to reach 458:

| Symbol | Sector | Usable | Usable dates | Evidence and decision |
|---|---|---:|---|---|
| ASIC | Insurance | 221 | 2025-01-14 to 2026-08-07 | Base processed/split/`rl_contract.json` absent; below canonical 252-row gate after OHLC quality filtering; remain excluded |
| EWIC | Insurance | 231 | 2021-12-13 to 2026-08-07 | Base contract absent; substantial non-positive-OHLC filtering and below 252; remain excluded |
| MUGHALC | Engineering | 225 | 2025-08-26 to 2026-08-07 | Recent usable history below 252; base contract absent; remain excluded |
| ZUMA | Technology & Communication | 147 | 2026-01-02 to 2026-08-07 | Recent usable history below 252; base contract absent; remain excluded |

Rebuilding them would require weakening or special-casing the canonical base RL
pipeline. That is not deterministic parity with the other symbols and was not
done.

## Sector recommendations

### Best sectors for general sector-pretraining research

1. **Commercial Banks** — 19/19 current ordinary equities are Mature and
   compatible, median 1,701 TRAIN rows, 31,157 total, 5.5% maximum
   concentration, and manageable compute.
2. **Cement** — 18 compatible, median 1,697, 28,328 total.
3. **Power Generation & Distribution** — 15 compatible, median 1,680, 22,645
   total.
4. **Automobile Assembler** — 10/10 compatible, median 1,703, 16,545 total.
5. **Chemical** — 24 compatible and 32,975 total, but larger compute and a
   wider depth distribution.

**Milestone 6E initial standard sector recommendation: Commercial Banks.** It
offers complete compatible coverage, deep histories, low concentration, and a
smaller, cleaner universe than the largest textile sectors.

### Best sectors for Cold Start transfer research

1. **Insurance** — PAKQATAR is Cold Start; 24 compatible Mature peers and
   30,002 TRAIN rows remain after honest ASIC/EWIC exclusion.
2. **Transport** — BLUEX is Cold Start; five compatible Mature peers and 6,101
   TRAIN rows, but greater concentration and less diversity.

Insurance is the stronger future real-Cold-Start sector, but current PAKQATAR
validation depth is not yet sufficient for a robust result.

## Compute estimate for Commercial Banks

- constituents: 19;
- complete referenced episode rows: 31,157;
- one conceptual pass over every complete episode: about 31,157 environment
  timesteps, 18.28 times MCB's 1,704-row TRAIN episode;
- linear estimate from measured 6C CPU timings: roughly 66–67 seconds for one
  full data pass, with a practical planning range of 1–2 minutes;
- 100,000 environment steps: about 3.5–5 minutes if current throughput holds;
- referenced TRAIN CSV size: 21.3 MiB; pandas/Torch working memory will be
  higher but remains modest relative to a 16-GB MacBook Air M2;
- PPO rollout memory remains bounded mainly by `n_steps=512`, observation shape
  17, and the 64-unit recurrent state, not all historical rows simultaneously.

This is an engineering estimate, not a runtime promise. No sector training was
performed to obtain it.

## Authoritative historical reconstruction plan

A future controlled process should use only authoritative evidence:

1. acquire dated official PSX listing/security-master or company-profile
   records where PSX makes them available;
2. capture symbol, legal company name, instrument type, sector identifier,
   listing/delisting/effective dates, previous/successor ticker, snapshot date,
   source URL/reference, retrieval date, and raw response hash;
3. preserve sector-effective intervals rather than one timeless value;
4. cross-check official corporate-action notices for aliases/mergers;
5. assign confidence `verified_at_cutoff`, `verified_current_only`, or
   `manual_review_required`;
6. route conflicts and unavailable archives to manual review without guessing.

No live scraping or external metadata update occurred in 6D.

## Generated artifact layout

```text
data/processed/sector_universes/
    sector_taxonomy.json
    current_verified_symbols.csv
    historical_instrument_audit.csv
    sector_universe_summary.csv
    <38 canonical sector ids>/
        universe_manifest.json
        train_episode_index.json
```

The directory contains 80 files and about 2.9 MiB. It duplicates no symbol
TRAIN data; it references canonical artifacts by portable path and SHA-256.
Generated files remain ignored consistently with existing processed-data
policy.

## Final go/no-go decisions

**SECTOR PRETRAINING DATA FOUNDATION: GO**

The taxonomy, current evidence, compatibility gate, portable deterministic
manifests, target-exclusion mechanism, and reset-safe TRAIN episode indexes are
ready.

**HISTORICAL SECTOR CLAIMS: LIMITED**

Current-sector grouping is documented honestly. Any claim of sector membership
at historical date T is blocked until dated authoritative evidence is stored.

**GENERAL SECTOR PRETRAINING: CONDITIONAL**

6E may implement a Commercial Banks TRAIN-only baseline after it implements
the uniform symbol sampler, enforces portfolio/LSTM resets, records actual step
contributions, and predeclares validation/multi-seed methodology. It must not
claim historical taxonomy correctness.

**COLD-START TRANSFER EXPERIMENT: CONDITIONAL**

Pseudo-Cold-Start research can proceed later with target-excluded Mature
symbols and multiple seeds. Real BLUEX/PAKQATAR claims require sufficient later
validation history or a predeclared rolling/expanding design; TEST remains
sealed.

## Verification

- Complete test suite: **509 passed, 2 skipped in 14.22 seconds**. The skips are
  existing hardware-gated MPS tests.
- Offline sector-universe regressions: **15 passed**.
- Generated-artifact integrity: 38 manifests and 38 episode indexes validated;
  all deterministic universe hashes reproduced; 454 unique constituent symbols
  and 556,586 TRAIN rows reconciled to the summary.
- Portability/partition scan: no `/Users/...` paths and no `validation_rl.csv`
  or `test_rl.csv` references in generated artifacts.
- Production model registry SHA-256 remained
  `e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a`.
- Company registry SHA-256 remained
  `4e1dd1206831d43722283d4aa157218a2050c8bd809348ecf673e4599082fa02`.
- Both production saved-model roots still contain only their tracked
  `.gitkeep` files; no model bundle was created.
- No live HTTP request, sector training, model persistence, registry mutation,
  validation evaluation, or TEST-frame load occurred.
