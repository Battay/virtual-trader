# Milestone 7C.3h — Frozen vs Current Identity Reconciliation

Audit date: 2026-08-28 (Asia/Karachi)  
Repository baseline: `b60c5e0`  
Scope: identity, TRAIN-contract, and downstream provenance audit only. No model
training, dataset regeneration, registry write, or TEST-partition load occurred.

## Exact identity reconciliation

| Identity role | Snapshot | Identities | Universe hash |
|---|---:|---:|---|
| `FROZEN_RESEARCH_UNIVERSE` | 2026-08-02 | 508 | `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5` |
| `CURRENT_OPERATIONAL_IDENTITY` | 2026-08-28 | 510 | `2f767ae6d718806bb3ec06e9080a5c8bba0b9f10ac28cd1e526cee0681621b20` |

- Common symbols: **508**
- Current-only symbols: **2** — `GCWLPRS`, `TISL`
- Frozen-only symbols: **0**

The tracked frozen manifest is
`docs/config/frozen_research_common_equity_identity_v1.json`. Its 508 sorted
members reproduce the original hash. It does not derive membership from the
mutable `current_listings.csv` file.

### Current-only records

| Symbol | Company | Sector | Current inferred security type | Source | Snapshot | Listing status | Recurrent category | Reason |
|---|---|---|---|---|---|---|---|---|
| GCWLPRS | Ghani Chemworld Limited(PRS) | CHEMICAL | ordinary_equity | PSX main normal-counter listing table | 2026-08-28 | listed / officially listed | missing_required_artifacts | recurrent_contract_missing |
| TISL | Tasdeeq Information Services Limited | MISCELLANEOUS | ordinary_equity | PSX main normal-counter listing table | 2026-08-28 | listed / officially listed | missing_required_artifacts | recurrent_contract_missing |

Both records have source URL
`https://dps.psx.com.pk/listings-table/main/nc` and listing refresh timestamp
`2026-08-28T02:28:13+05:00`.

`ordinary_equity` is the existing listing classifier's result, not an explicit
security-type field supplied by the listing table. `GCWLPRS` contains `(PRS)`
in its company name, while the current classifier recognizes `PREF` and
`PREFERENCE`, not `PRS`. It therefore merits a future current-identity manual
classification review. This does not affect the frozen training set because it
has no pre-cutoff TRAIN data or recurrent contract.

## Why the symbols appeared

The retained 2026-07-30 and 2026-08-02 listing snapshots each contain 563
records and contain neither symbol. The retained 2026-08-28 snapshot contains
568 records and contains both. The other three new listing rows are rights
(`FPRMR2`, `SGPLR`, `STLR`), so only `GCWLPRS` and `TISL` enter the current
common-equity identity under the existing classifier.

The registry records both as newly listed/recently traded, with first observed
market date 2026-08-24, last observed date 2026-08-27, and three trading dates.
This is consistent with new authoritative listing/data evidence, not a change
to the classification code and not an alias mismatch. The local snapshots do
not provide an exchange effective-listing date, so the strongest local claim is
that both entered the retained listing evidence after 2026-08-02 and by
2026-08-28. They are not merely stale-registry recoveries: exact-symbol market
observations and official listing rows are both present.

## TRAIN-only contract evidence

The frozen TRAIN boundary ends on **2023-08-03**. A predicate-pushed market
read for only `GCWLPRS` and `TISL` through that date returned zero rows.

| Symbol | TRAIN rows through cutoff | Recurrent contract | Feature/TRAIN RL artifact | Validation availability in recurrent discovery | Outcome |
|---|---:|---|---|---|---|
| GCWLPRS | 0 | absent | absent | false | not trainable |
| TISL | 0 | absent | absent | false | not trainable |

No artifact was regenerated. No validation values were loaded. TEST metadata
and values were not read by the reconciliation/discovery path.

## Why both policies still yield 435 trainable symbols

| Policy | Trainable | Insufficient | Unsupported/inactive | Missing artifacts |
|---|---:|---:|---:|---:|
| Frozen 508 | 435 | 20 | 53 | 0 |
| Current 510 | 435 | 20 | 53 | 2 |

The two current-only identities account exactly for the two missing-artifact
records. Across the 508 common members, no trainability category, reason,
contract version, environment version, feature version, source-contract hash,
TRAIN row/date evidence, or validation-availability flag changed.

- Frozen trainable-set hash: `44efa67c6c1aa5ac27d559f85835493206617a63fa24c25648e2da0d9f38a4a2`
- Current trainable-set hash: `44efa67c6c1aa5ac27d559f85835493206617a63fa24c25648e2da0d9f38a4a2`
- Current-only trainable symbols: none
- Frozen-only trainable symbols: none

## Training-universe decision

The full recurrent research run will use:

`TRAINABLE_MEMBERS_OF_FROZEN_RESEARCH_UNIVERSE_V1`

That is the 435 compatible symbols within the immutable 508-member 2026-08-02
research snapshot. This preserves comparability with frozen clustering,
relationship, trainability, and budget-study evidence. It also omits no
currently trainable symbol because the current and frozen trainable sets are
identical.

The 510-member identity remains valid for operational listing/registry views.
It must not silently replace research membership. A future operational model
extension may evaluate newly listed identities after they accumulate a valid
chronological TRAIN contract; it would require a separately versioned universe
and cannot be represented as a resume of this run.

## Downstream impact audit

| Consumer | Required identity behavior | Finding/action |
|---|---|---|
| `recurrent_orchestrator.py` | Supports operational discovery and explicit frozen input | Default remains current operational; discovery now records identity role, snapshot, execution policy, and trainable-set hash. |
| `full_universe_run.py` | Frozen research identity only | Now loads the tracked 508 manifest and fails closed if passed current operational identity. |
| `sector_universe.py` | Versioned sector-prototype manifests | Existing manifests retain their own universe hashes and constituent lists; frozen prototype work is unchanged. Current listing defaults apply only to regeneration, which did not occur. |
| Clustering/relationship artifacts | Frozen methodology | Existing reports reference frozen hash `571f...ef5`; no artifact was rewritten. |
| Lead allocator | Future architecture | No executable lead allocator exists; representation documentation explicitly does not allocate capital. It must consume an explicit model/run universe later. |
| Model registry | Model-level immutable records | Registry contains only its header and received no row. Future models must retain run-manifest provenance. |
| Training run manifests | Immutable resume identity | New manifests record identity policy/snapshot, universe hash, trainable count/hash, and execution policy. Persisted job inventory is rejected if universe version/hash or trainable-symbol count/hash differs. |

## Reproducibility and safety controls

- Frozen membership, snapshot, selection methodology, and universe hash are
  stored in a tracked manifest and verified on load.
- Full-run planning rejects `CURRENT_OPERATIONAL_IDENTITY` even if counts happen
  to match.
- The deterministic run fingerprint includes identity role, snapshot,
  execution-training policy, trainable count, and trainable-set hash.
- A resumed job store fails closed when its jobs do not reconcile with the
  manifest universe or trainable set.
- Full-run dry planning resolves to 508 identities, 435 queued trainable jobs,
  73 explicit exclusions, and the frozen trainable hash above.
- No training was started and `execution_authorized` remains false.

## Decision

The discrepancy is fully reconciled and the training membership ambiguity is
removed. Subject to the separately frozen runtime/budget gates, the identity
contract is **READY_FULL_RECURRENT_TRAINING**.
