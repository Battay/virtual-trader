# Phase 2 P2.2f — SBP Policy Circular PDF Support

Decision: **READY_POLICY_RATE_EVIDENCE**

P2.2 remains blocked by CPI release evidence and daily SBP USD/PKR M2M
evidence. P2.3 did not start, TEST remained sealed, and no model was trained.

## Root cause and parser separation

The preserved C06/2022 and C09/2022 files are official one-page, image-only
PDFs. They contain one JPEG page image and no extractable PDF text layer. The
readiness loop previously routed every `.pdf` to
`sbp_policy_target_rate_pdf_v2`, which correctly requires the explicitly
labelled `Structure of Interest Rates -I` table and therefore rejected these
circulars.

Routing is now determined by the manifest source identifier, media type,
official URL, and parser contract:

| Evidence | Parser |
|---|---|
| `sbp_structure_of_interest_rates_sir`, PDF, official `/ecodata/sir.pdf` | `sbp_policy_target_rate_pdf_v2` |
| canonical DMMD circular identifier, HTML, matching official archive URL | `sbp_policy_circular_html_v1` |
| C06/C09 canonical DMMD circular identifier, PDF, matching official circular URL | `sbp_policy_circular_pdf_v1` |
| canonical EasyData target-rate export identifier and delimited media | existing explicit export parser |

Any unsupported identifier, source-path mismatch, media mismatch, unsupported
circular PDF, missing OCR capability, or ambiguous page layout fails closed.
Circular PDFs cannot enter the SIR parser. The actual `Policy (Target) Rate`
sentence is required; Repo, Reverse Repo, floor, ceiling, KIBOR, T-bill, PIB,
and other rates cannot substitute for it.

For image-only circulars on macOS, the parser extracts the single preserved
page image into a temporary directory and uses native Vision OCR. Temporary
images and module cache are removed after parsing. No evidence file is edited,
converted, overwritten, or re-saved. The circular number and announcement
date/year must be present in visible OCR text; values are not derived from a
hard-coded event table.

## Parsed circular PDF records

### C06/2022

- Circular: 06 of 2022
- Announcement: 2022-04-07
- Previous target rate: 9.75%
- New target rate: 12.25%
- Effective: 2022-04-08
- Referenced circular: C23 dated 2021-12-14
- Source identifier: `sbp_dmmd_circular_06_2022`
- SHA-256: `232ff6017f61a9636f76c87336e680930a28e81b870b68a0cd92c7177a4d3eb8`

### C09/2022

- Circular: 09 of 2022
- Announcement: 2022-05-23
- Previous target rate: 12.25%
- New target rate: 13.75%
- Effective: 2022-05-24
- Referenced circular: C06 dated 2022-04-07
- Source identifier: `sbp_dmmd_circular_09_2022`
- SHA-256: `6f3c6385e55f39634f7683c63548c37366b06fa12552855a0418cbf922865691`

Both hashes matched before and after real parsing.

## Combined policy inventory

The circular chain establishes a 7.00% policy target rate at the Phase-2 start
date of 2021-08-06 without creating a synthetic start event.

| Effective date | Target rate | Announcement | Evidence |
|---|---:|---|---|
| 2021-09-21 | 7.25 | 2021-09-20 | C15/2021 HTML |
| 2021-11-22 | 8.75 | 2021-11-19 | C21/2021 HTML |
| 2021-12-15 | 9.75 | 2021-12-14 | C23/2021 HTML |
| 2022-04-08 | 12.25 | 2022-04-07 | C06/2022 PDF |
| 2022-05-24 | 13.75 | 2022-05-23 | C09/2022 PDF |
| 2022-07-13 | 15.00 | 2022-07-07 | C13/2022 HTML |
| 2022-11-28 | 16.00 | 2022-11-25 | C20/2022 HTML |
| 2023-01-24 | 17.00 | not present in SIR table | SIR PDF |
| 2023-03-03 | 20.00 | not present in SIR table | SIR PDF |
| 2023-04-05 | 21.00 | not present in SIR table | SIR PDF |
| 2023-06-27 | 22.00 | not present in SIR table | SIR PDF |
| 2024-06-11 | 20.50 | not present in SIR table | SIR PDF |
| 2024-07-30 | 19.50 | not present in SIR table | SIR PDF |
| 2024-09-13 | 17.50 | not present in SIR table | SIR PDF |
| 2024-11-05 | 15.00 | not present in SIR table | SIR PDF |
| 2024-12-17 | 13.00 | not present in SIR table | SIR PDF |
| 2025-01-28 | 12.00 | not present in SIR table | SIR PDF |
| 2025-05-06 | 11.00 | not present in SIR table | SIR PDF |
| 2025-12-16 | 10.50 | not present in SIR table | SIR PDF |
| 2026-04-28 | 11.50 | not present in SIR table | SIR PDF |

The seven circulars pass previous-rate/new-rate continuity. The combined set
contains 20 unique effective-date events from 2021-09-21 through 2026-04-28.
The SIR rows deliberately retain null announcement dates rather than inventing
them.

## Readiness and remaining P2.2 blockers

The real offline readiness command reports:

- `POLICY_RATE`: `READY`
- policy events: 20
- circular chain events: 7
- Phase-2 start rate: 7.00%
- `CPI`: `MISSING_RELEASE_EVIDENCE`
- `USD_PKR`: `MISSING`
- `CANONICAL_MACRO`: `BLOCKED`
- TEST observations loaded: false

The CPI blocker is one missing required value month (2026-07) and exact
first-release evidence for 51 required months. The FX blocker is the absent
official daily SBP USD/PKR M2M history covering the Phase-2 range. No canonical
macro Parquet, common split, or scaler was created.

## Safety

- First-party SBP domains remain mandatory.
- Preserved evidence and checksums were unchanged.
- No replacement evidence was fetched.
- No TEST observations were accessed.
- No training, validation, Phase-3 work, or model mutation occurred.
- No commit was created.
