# Phase 2 P2.2c — SBP Policy-Rate Evidence Conflict Diagnosis

Audit timestamp: `2026-09-01T19:39:04Z`  
Decision: **BLOCKED_POLICY_RATE_EVIDENCE**

The preserved official PDF is unchanged and checksum-valid:

- source: `https://www.sbp.org.pk/ecodata/sir.pdf`
- local evidence id: `sbp_structure_of_interest_rates_sir`
- SHA-256: `6b4a03a2c1d25529575e33eed561e1948fecf2a9b841003fd9087dd447f3065e`
- PDF title metadata: `SIR_31-Aug-26.xlsx`
- pages: 6

No replacement evidence was fetched. No canonical macro data, split, scaler,
environment, or model was created.

## Root cause

The old parser flattened all six pages and treated any date followed by three
numbers as:

`w.e.f. / Reverse Repo / Repo / Policy (Target) Rate`

That shape is not unique to the policy table. Pages II through VI contain
Treasury-bill rates, KIBOR, PIB yields, Sukuk rental rates, lending/deposit
rates, and National Savings rates with the same textual shape. The generic
error said “one effective date”, but the pre-fix extraction actually produced
six same-date conflicts. None came from the SBP Policy (Target) Rate table.

This is deterministic parser column/table conflation (causes C and D), not an
official revision or duplicated policy-history ambiguity.

## Pre-fix false conflict inventory

The values below are retained exactly as the old regex matched them. Its
labels “reverse repo”, “repo”, and “target” were false interpretations for all
of these non-policy rows. The PDF supplies no announcement date in these
tables.

| Date | Page / actual section | Raw extracted match | Old reverse-repo interpretation | Old repo interpretation | Old target interpretation |
|---|---|---|---:|---:|---:|
| 2025-09-17 | 2 / SIR-II Treasury Bills Auctions | `17-Sep-25 10.74 10.85 10.84` | 10.74 | 10.85 | 10.84 |
| 2025-09-17 | 6 / SIR-VI National Savings rates | `17-Sep-25 10.80 11.42 10.40` | 10.80 | 11.42 | 10.40 |
| 2025-10-15 | 2 / SIR-II Treasury Bills Auctions | `15-Oct-25 11.11 11.05 11.05` | 11.11 | 11.05 | 11.05 |
| 2025-10-15 | 3 / SIR-III PIB auction yields | `15-Oct-25 11.33 11.35 11.50` | 11.33 | 11.35 | 11.50 |
| 2026-06-10 | 2 / SIR-II Treasury Bills Auctions | `10-Jun-26 12.19 12.50 12.49` | 12.19 | 12.50 | 12.49 |
| 2026-06-10 | 6 / SIR-VI National Savings rates | `10-Jun-26 12.24 12.19 12.40` | 12.24 | 12.19 | 12.40 |
| 2026-07-22 | 2 / SIR-II daily rates | `22-Jul-26 11.89 11.67 11.67` | 11.89 | 11.67 | 11.67 |
| 2026-07-22 | 2 / SIR-II Treasury Bills Auctions | `22-Jul-26 11.35 11.52 11.80` | 11.35 | 11.52 | 11.80 |
| 2026-08-05 | 2 / SIR-II daily rates | `5-Aug-26 11.90 11.70 11.75` | 11.90 | 11.70 | 11.75 |
| 2026-08-05 | 2 / SIR-II Treasury Bills Auctions | `5-Aug-26 11.35 11.51 11.70` | 11.35 | 11.51 | 11.70 |
| 2026-08-19 | 2 / SIR-II daily rates | `19-Aug-26 11.90 11.70 11.79` | 11.90 | 11.70 | 11.79 |
| 2026-08-19 | 2 / SIR-II Treasury Bills Auctions | `19-Aug-26 11.47 11.65 11.80` | 11.47 | 11.65 | 11.80 |

Every row has the same evidence hash stated above.

## Source semantics and versioned resolution

Parser version `sbp_policy_target_rate_pdf_v2` now requires both:

1. the exact `Structure of Interest Rates -I` page/table label; and
2. the explicit `SBP Policy (Target) Rate` header.

Within that identified table, the first three numeric columns after `w.e.f.`
are retained separately as `SBP Reverse Repo Rate`, `SBP Repo Rate`, and `SBP
Policy (Target) Rate`. Only the third is the required Phase-2 value. Page-I
footnotes state that Reverse Repo was the policy rate only through 24-May-2015
and that the explicit Policy (Target) Rate was introduced from 25-May-2015.
The parser therefore never substitutes Reverse Repo, Repo, floor, ceiling, or
another interest-rate series for the target column.

Identical same-date target rows from overlapping copies are retained in the
raw audit and deterministically deduplicated for the event series. Different
target values for one effective date still raise with the exact date(s).
Announcement dates remain null because this PDF does not publish them in the
table; availability remains conservatively delayed to the next Lead-Agent
decision boundary rather than inventing an announcement date.

## Correct target-rate event inventory

All 13 rows are on page 1, `Structure of Interest Rates -I`, and have the same
source hash. No deduplication was needed.

| w.e.f. | Reverse Repo | Repo | Policy (Target) Rate | Announcement date |
|---|---:|---:|---:|---|
| 2023-01-24 | 18.00 | 16.00 | 17.00 | not present |
| 2023-03-03 | 21.00 | 19.00 | 20.00 | not present |
| 2023-04-05 | 22.00 | 20.00 | 21.00 | not present |
| 2023-06-27 | 23.00 | 21.00 | 22.00 | not present |
| 2024-06-11 | 21.50 | 19.50 | 20.50 | not present |
| 2024-07-30 | 20.50 | 18.50 | 19.50 | not present |
| 2024-09-13 | 18.50 | 16.50 | 17.50 | not present |
| 2024-11-05 | 16.00 | 14.00 | 15.00 | not present |
| 2024-12-17 | 14.00 | 12.00 | 13.00 | not present |
| 2025-01-28 | 13.00 | 11.00 | 12.00 | not present |
| 2025-05-06 | 12.00 | 10.00 | 11.00 | not present |
| 2025-12-16 | 11.50 | 9.50 | 10.50 | not present |
| 2026-04-28 | 12.50 | 10.50 | 11.50 | not present |

Audit summary:

- earliest effective date: `2023-01-24`
- latest effective date: `2026-04-28`
- raw rows: 13
- unique policy events: 13
- duplicate source rows: 0
- conflicting source rows: 0
- missing parsed target values: 0
- effective dates monotonic: yes

## Readiness

The cross-table conflict is fixed, but `POLICY_RATE` remains `INVALID`. The
Phase-2 market interval begins on `2021-08-06`; the preserved target-rate table
begins on `2023-01-24` and therefore does not establish the setting in force at
the Phase-2 start. Treating unrelated 2020–2022 rates elsewhere in the PDF as
policy evidence would be precisely the conflation this correction prevents.

Required remaining policy evidence is an official SBP Policy (Target) Rate
history with an effective setting on or before `2021-08-06`. CPI and USD/PKR
remain independently blocked. The canonical macro dataset was not created and
TEST remained sealed.

**BLOCKED_POLICY_RATE_EVIDENCE**
