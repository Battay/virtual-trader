# Phase 2 P2.2a — Authoritative Macro Evidence Acquisition

Decision: **BLOCKED_PHASE2_DATA_CONTRACT**

Audit timestamp: `2026-09-01T14:30:02Z`

Contract artifact: `docs/config/phase2_lead_agent_data_contract_v1.json`

Contract evidence hash: `d06c228a80918969a5d2786feaea6878897301becce930828c3f98f19687d7a5`

## Outcome

The official PBS CPI history, release-schedule statement, and recent publication
metadata were acquired and preserved with checksums. The two required SBP
histories could not be preserved because normal direct requests to the official
file hosts returned HTTP 403. No access control was bypassed.

P2.2 remains blocked. `lead_agent_macro.parquet`, the common 70/15/15 calendar,
and the TRAIN-fitted scaler were not created. Treating the current PBS history
as every historical first-release vintage, or substituting SBP conversion rates
for M2M without an identity proof, would violate the point-in-time contract.

## First-party source findings

| Series | Official evidence | Semantics frozen by P2.2a | Local result |
|---|---|---|---|
| SBP policy target rate | [Structure of Interest Rates](https://www.sbp.org.pk/ecodata/sir.pdf), [EasyData](https://easydata.sbp.org.pk/), and dated SBP monetary-policy decisions | Use the target rate no earlier than both its announcement and official `w.e.f.` effective boundary. An official 27-Apr-2026 decision states that its rate became effective 28-Apr-2026, proving announcement and effective dates are distinct concepts. | Official pages were inspectable, but the complete source file returned HTTP 403 to normal machine retrieval. Zero raw files preserved; manual ingest required. |
| PBS National CPI YoY | [Historical CPI table](https://www.pbs.gov.pk/wp-content/uploads/2020/07/indices_and_growth_rates_historical-1.pdf), [PBS FAQ](https://www.pbs.gov.pk/faqs/), and official PBS post metadata | Reference month never grants availability. Exact CMS publication timestamps are used where preserved. For legacy releases lacking timestamps, `first day of M+2` is a candidate conservative availability bound based on PBS's stated first/second-day publication schedule; it must be labelled as a bound, not an actual release date. | Three first-party evidence files preserved and verified. Historical first-release vintages remain incomplete. |
| SBP USD/PKR M2M | [M2M history](https://www.sbp.org.pk/ecodata/rates/m2m/M2M-History.asp), [SBP economic-data archive](https://archive.sbp.org.pk/ecodata/index2.asp), and SBP Statistical Bulletin table 6.9 | Use the daily USD M2M revaluation rate in PKR per USD. With no authoritative intraday release time, date `d` is first eligible at the next Phase-2 decision cutoff. | Official pages establish the series and daily frequency, but a complete daily source file returned HTTP 403 or was unavailable through the machine path. Zero raw files preserved; manual ingest and schema verification required. |

The annual SBP conversion-rate workbooks are not assumed to be the M2M series.
They may be used only if first-party documentation or field-level comparison
proves the identity and units.

## Preserved PBS evidence

All files are under `data/raw/macro/pbs_cpi/` and remain local raw evidence.
Each has a sibling `macro_raw_evidence_manifest_v1` record with series owner,
source URL/id, retrieval time, media type, parse status, byte size, and SHA-256.

| Evidence | Bytes | SHA-256 | Parsed finding |
|---|---:|---|---|
| `20260901T143002Z_indices_and_growth_rates_historical-1.pdf` | 203,422 | `0b6ba1e88084e00075065ad01b7bbeae3c464fffee6b474d056e56f989343070` | 108 unique National CPI YoY reference months, 2017-07 through 2026-06 |
| `20260901T143002Z_monthly_inflation_posts.json` | 138,905 | `e7577edec707b45e49e04aba87aec69d0b57eb2c127160d8396ebbb3493aaad7` | 11 exact official CMS timestamps, 2025-10 through 2026-08; nine overlap the history PDF |
| `20260901T143002Z_pbs_faq.html` | 243,678 | `f38c3320e740e3a37ee979a7fe714ec72027c5b34fd3071d6af463b37846c4a0` | PBS states CPI is normally released on the first or second day of each month |

The PDF is a current historical table, not a first-release vintage archive.
Consequently its older values cannot be projected backward into historical
decision dates until original releases are preserved or PBS establishes that
the relevant values are unrevised.

## HTTP 403 and manual ingest

Normal HTTPS requests, including ordinary browser headers and official archive
alternates, were attempted for the SBP evidence. They returned HTTP 403. The
review stopped there; no authentication, anti-bot mechanism, or access control
was bypassed.

`macro_evidence_ingest_v1` provides the bounded fallback:

1. A user saves the public official SBP file through a normal browser or an
   official EasyData export.
2. The user computes and supplies the observed SHA-256.
3. `python -m data_pipeline.src.macro_evidence --ingest-file ...` verifies the
   checksum, source series, HTTPS provenance, and retrieval timestamp.
4. The utility writes a retrieval-stamped file and manifest atomically and
   refuses any existing destination.
5. `python -m data_pipeline.src.macro_evidence --audit` re-hashes every file and
   fails closed on missing, changed, or malformed evidence.

This path does not download, parse into canonical observations, overwrite raw
evidence, or fit any scaler.

## Remaining unblock evidence

### SBP policy rate

Preserve the complete official `sir.pdf` (or an official EasyData export) and
the dated announcement/effective evidence needed for every event in the
Phase-2 market period. Parsing must retain both announcement and `w.e.f.` dates.

### PBS CPI

Preserve original monthly releases for the Phase-2 period, or authoritative PBS
evidence demonstrating that the historical YoY values equal their first
releases. Exact CMS timestamps can cover the recent months. An explicitly
labelled conservative M+2 availability bound may replace a missing timestamp,
but it cannot replace missing first-release values.

### SBP USD/PKR

Preserve a complete official daily M2M history covering the Phase-2 period.
Verify the USD field, PKR-per-USD units, reference-date convention, duplicate
policy, and source publication semantics before canonicalization.

## Safety and reproducibility

- First-party evidence only; no aggregator values entered the project.
- No release date was fabricated.
- No historical value was represented as an original vintage without proof.
- No canonical macro Parquet was created.
- No common split was frozen and no scaler was fitted.
- TEST observations remained sealed and were not loaded.
- No environment, PPO, SAC, or Phase-3 work occurred.
- No model or training-run state was modified.
- No commit was made.

**BLOCKED_PHASE2_DATA_CONTRACT**
