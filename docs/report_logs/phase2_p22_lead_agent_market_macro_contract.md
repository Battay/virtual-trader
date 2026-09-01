# Phase 2 P2.2 — Leakage-Safe Market + Macro Data Contract

Decision: **BLOCKED_PHASE2_DATA_CONTRACT**

Contract artifact: `docs/config/phase2_lead_agent_data_contract_v1.json`

Current contract evidence hash after P2.2a: `d06c228a80918969a5d2786feaea6878897301becce930828c3f98f19687d7a5`

> P2.2a update: PBS evidence is now preserved locally, while both required SBP
> histories remain blocked behind official-server HTTP 403 responses. See
> `docs/report_logs/phase2_p22a_authoritative_macro_evidence_acquisition.md` for
> the current evidence inventory and manual-ingest procedure. The text below
> preserves the original P2.2 contract findings.

## Objective and outcome

P2.2 established the executable, offline-tested boundary for causal official-index and macro observations. It did not create a canonical macro Parquet, freeze a common split, or fit a scaler because a complete provenance-preserved first-party macro evidence set is not locally available. Reporting exact macro coverage or split dates in that state would be fabricated evidence.

This is a data-readiness block, not an architecture block. Phase 1 remains closed as `REJECTED_CLUSTER_STRUCTURE`, and the Phase-2 Lead Agent remains independent of hard clusters, soft prototypes, sector allocations, and Phase-3 recurrent agents.

## Authoritative source inventory

| Series | First-party source | Definition and units | Frequency | Release / effective semantics | Machine-readable and vintage finding | P2.2 status |
|---|---|---|---|---|---|---|
| SBP policy rate | [SBP Structure of Interest Rates](https://www.sbp.org.pk/ecodata/sir.pdf), with [SBP EasyData](https://easydata.sbp.org.pk/) series `TS_GP_IR_SIRPR_AH.SBPOL0030` | State Bank of Pakistan Policy (Target) Rate; percent per annum | Event-driven | Use no earlier than both the official announcement and the documented `w.e.f.` effective time. A date-only `w.e.f.` entry is conservatively usable from the start of that effective date. | The historical table is a PDF; announcements are dated, but there is no locally preserved complete timestamped event file. Corrections must remain separate vintages. | Blocked pending preserved raw evidence. |
| PBS CPI | [PBS Price Statistics](https://www.pbs.gov.pk/price-statistics/), [PBS historical CPI table](https://www.pbs.gov.pk/wp-content/uploads/2020/07/indices_and_growth_rates_historical-1.pdf), and [PBS release-schedule FAQ](https://www.pbs.gov.pk/faqs/) | National CPI year-over-year inflation, base 2015-16=100; percent YoY | Monthly | Month M is unavailable until its actual publication. Reference month alone never grants availability. If only a publication date is authoritative, use the start of the following local day. | Official history is primarily PDF. Recent publication posts expose timestamps, but a complete first-release calendar is not preserved locally. Later revisions must not be projected backward. | Blocked pending complete first-release evidence. |
| SBP USD/PKR | [SBP historical conversion rates](https://archive.sbp.org.pk/ecodata/CRates/index.asp) | Intended initial field: daily USD M2M revaluation/conversion rate in PKR per USD; exact workbook field remains subject to schema verification | Business day | Reference date d cannot enter decision d without authoritative publication-time evidence. The conservative rule is the first decision cutoff after publication. | First-party annual XLSX histories are linked, but the files could not be preserved and their exact field schema was not verified in this run. Historical replacements require new source vintages. | Blocked pending raw workbooks and exact field verification. |

These are public first-party statistical publications. No explicit reusable-data license was identified during the bounded source review, so the contract requires source attribution, checksums, retrieval timestamps, and preservation of the exact response.

## Raw evidence store

The repository now defines:

```text
data/raw/macro/
  sbp_policy_rate/
  pbs_cpi/
  sbp_usdpkr/
```

Every evidence file requires a sibling `macro_raw_evidence_manifest_v1` record containing source identity/URL, retrieval timestamp, byte size, SHA-256, media type, parse status, and source version. Evidence filenames must be retrieval-unique; silent overwrite is prohibited.

No raw evidence file was persisted. Bounded PBS research responses reached temporary storage, but not a complete reproducible first-release set. The SBP file requests were refused with HTTP 403 in the available download path. Temporary or partial evidence was not promoted into the repository or represented as canonical.

## Canonical macro schema

Version: `lead_agent_macro_observation_v1`

Intended path: `data/processed/macro/lead_agent_macro.parquet`

Required fields are:

```text
schema_version
series
reference_date
value
unit
native_frequency
release_date
release_timestamp
effective_available_timestamp
availability_method
vintage_id
source
source_version
retrieved_at
provenance_hash
point_in_time_safe
```

The validator fails closed on absent required series, malformed timestamps, missing or non-finite values, unsafe vintages, duplicate vintage keys, invalid hashes, non-positive USD/PKR, negative policy rates, CPI releases not later than their reference month, and availability that precedes release/effective semantics.

No canonical Parquet was created. Point-in-time-safe and unsafe rows cannot be mixed into training data.

## Point-in-time methodology

The conservative Phase-2 decision cutoff is the start of decision-session calendar day t in `Asia/Karachi`. This intentionally delays a same-day publication when its intraday release time cannot be proved.

At decision session t:

- market features contain official index information only through trading session t-1;
- macro features use the latest vintage satisfying `effective_available_timestamp <= decision_cutoff(t)`;
- the realized target is KSE100 close(t) to close(t+1), matching P2.1;
- no same-session close or volume enters the observation;
- no forward interpolation is permitted;
- a later revision becomes visible only at its own effective availability time and never rewrites earlier decisions;
- dates before all required macro series have a safely released observation are excluded.

The implementation reconstructs a release-vintage state timeline. A late revision to an older CPI reference month cannot displace a newer reference month as the latest-known CPI state.

## Market feature schema

Version: `lead_agent_market_features_v1`

Returns are recomputed from official index levels, not stored `daily_change` fields. This removes the known refresh-boundary null dependency.

The nine causal market features are:

1. KSE100 1-session return, lag 1
2. KSE100 5-session return, lag 1
3. KSE100 21-session return, lag 1
4. KSE100 20-session volatility, lag 1
5. KSE100 expanding drawdown, lag 1
6. KSE100 distance from 20-session mean, lag 1
7. KSE100 distance from 50-session mean, lag 1
8. KSE100 volume / 20-session mean volume, lag 1
9. cross-index 1-session return dispersion across KSE100, KSE30, KMI30, and ALLSHR, lag 1

The local official index source contains 5,012 rows, four indices, and 1,253 common dates from 2021-08-06 through 2026-08-27, with zero duplicate `(index_code, date)` keys and zero required nulls. Its SHA-256 is `09244e0d3484276843b00159f76e69b2e782ce3f1256ae5b2b21512c1b51cb6d`.

After the 50-session market warm-up and requiring a complete close(t)→close(t+1) target, the market-only candidate has 1,202 decision rows from 2021-10-20 through 2026-08-25; its final target ends 2026-08-27. The deterministic candidate hash is `473412db280716ad3391714e7375c4582874f88201964dfca81cac89b8b87ba9`.

## Macro feature schema

Version: `lead_agent_macro_features_v1`

The ten initial, interpretable macro features are:

- latest known SBP policy rate, last known setting change, and days since availability;
- latest known PBS CPI YoY, release-to-release change, and days since availability;
- latest known SBP USD/PKR M2M level, lagged observation return, 20-observation volatility, and days since availability.

The combined market+macro feature shape is 19. Symbol identity, sector identity, current-only breadth, Phase-1 relationship features, and Phase-3 agent outputs are excluded.

## Common coverage and split

Exact common market+macro coverage is **not available**. The authoritative market-only upper bound is documented above, but the lower/upper usable dates and row count depend on safely released CPI, policy-rate, and FX vintages.

Consequently, `lead_agent_common_calendar_split_v1` remains unfrozen:

| Partition | Rule | Exact dates | Rows | Status |
|---|---|---:|---:|---|
| TRAIN | first `floor(70% of N)` common eligible decision dates | not derivable | not derivable | not created |
| VALIDATION | next `floor(15% of N)` | not derivable | not derivable | not created |
| TEST | remainder | not opened | metadata unavailable | **SEALED** |

The offline split implementation is deterministic, stores date hashes and non-overlapping boundaries, and refuses any TEST-dated row supplied to the TRAIN/VALIDATION selector. It will be run only after authoritative common coverage exists.

## Normalization

Version: `lead_agent_train_standard_scaler_v1`

- fit: TRAIN observations only;
- validation: transform only;
- TEST: no load and no transform during P2.2;
- identity: ordered feature list, TRAIN date hash/row count, fitted mean/variance/scale, scikit-learn version, and deterministic scaler hash.

No scaler was fitted because there is no defensible common TRAIN partition.

## Leakage and quality audit

Offline regression coverage proves:

- CPI joins on release availability, not reference month;
- policy settings are invisible before their effective timestamp;
- FX reference values remain unavailable until the conservative availability boundary;
- future macro releases are not interpolated backward;
- later revisions to older periods do not replace the latest reference period;
- same-session index close changes do not affect session-t observations;
- index returns survive stored-change refresh gaps because they come from levels;
- split boundaries are exact, chronological, deterministic, and non-overlapping;
- the TRAIN/VALIDATION selector rejects TEST rows;
- StandardScaler statistics are fitted on TRAIN only;
- malformed, missing, or unsafe macro evidence fails closed;
- logical frame and decision-artifact hashes are reproducible;
- raw evidence manifest inspection does not modify its source file.

TEST observations were not loaded. No Lead Agent environment, PPO/SAC training, model artifact mutation, or Phase-3 recurrent-agent modification occurred.

## Data-management ownership

P2.2 adds only path/configuration ownership and contract/provenance validation. It does not add a dashboard page or modify Fetch Data / Automation. A later bounded Phase-2 data-management follow-up may expose macro freshness only after authoritative acquisition and parse workflows are reproducible.

## Required unblock work

1. Preserve the exact SBP policy-rate source and a complete announcement/effective-date mapping.
2. Preserve PBS CPI source files plus a complete actual or explicitly conservative first-release calendar; never infer availability from the represented month.
3. Preserve the SBP annual USD/PKR workbooks, verify the exact USD M2M/conversion field and units, and freeze the date-only publication lag.
4. Parse into `lead_agent_macro_observation_v1`, retain vintages, and pass the real coverage/quality audit.
5. Derive exact common eligible decision dates, freeze the 70/15/15 metadata, keep TEST sealed, and fit the scaler on TRAIN only.

P2.3 must not begin before those items complete.

## Verification

- Focused P2.2 tests: 19 passed.
- Complete repository suite: **907 passed, 2 skipped in 360.00 seconds**.
- Dependency check: **No broken requirements found** (the read-only pip cache warning is environmental).
- Whitespace check: **PASS** (`git diff --check` produced no output).

**BLOCKED_PHASE2_DATA_CONTRACT**
