# Phase 2 P2.2d — Official SBP Circular-Chain Policy Evidence

Audit timestamp: `2026-09-01T21:53:20Z`  
Decision: **BLOCKED_POLICY_RATE_EVIDENCE**

The official SBP circular chain was identified and its deterministic ingestion,
parsing, and validation path is implemented. No circular HTML was stored,
however: bounded direct requests to both the archive and current SBP pages
returned HTTP 403. Search-engine-rendered text was used only to audit which
first-party pages are required. It was not reconstructed into fake HTML or
accepted as checksum-preserved raw evidence.

The existing SIR PDF remains unchanged at
`data/raw/macro/sbp_policy_rate/20260901T192400Z_sir.pdf`, with SHA-256
`6b4a03a2c1d25529575e33eed561e1948fecf2a9b841003fd9087dd447f3065e`.
No canonical macro dataset, split, scaler, environment, or model was created.

## Required official circular pages

| Circular | Official archive URL | Announcement | Explicit target-rate change | Effective |
|---|---|---|---:|---|
| DMMD C15/2021 | `https://archive.sbp.org.pk/dmmd/2021/C15.htm` | 2021-09-20 | 7.00% → 7.25% | 2021-09-21 |
| DMMD C21/2021 | `https://archive.sbp.org.pk/dmmd/2021/C21.htm` | 2021-11-19 | 7.25% → 8.75% | 2021-11-22 |
| DMMD C23/2021 | `https://archive.sbp.org.pk/dmmd/2021/C23.htm` | 2021-12-14 | 8.75% → 9.75% | 2021-12-15 |
| DMMD C06/2022 | `https://archive.sbp.org.pk/dmmd/2022/C6.htm` | 2022-04-07 | 9.75% → 12.25% | 2022-04-08 |
| DMMD C09/2022 | `https://archive.sbp.org.pk/dmmd/2022/C9.htm` | 2022-05-23 | 12.25% → 13.75% | 2022-05-24 |
| DMMD C13/2022 | `https://archive.sbp.org.pk/dmmd/2022/C13.htm` | 2022-07-07 | 13.75% → 15.00% | 2022-07-13 |
| DMMD C20/2022 | `https://archive.sbp.org.pk/dmmd/2022/C20.htm` | 2022-11-25 | 15.00% → 16.00% | 2022-11-28 |

The current SBP site exposes corresponding pages under
`https://www.sbp.org.pk/circulars/dmmd-circular-no-NN-of-YYYY`. Both hostnames
are first-party and accepted by the ingestion boundary. The archive URLs above
are the canonical chain identifiers because they preserve the original DMMD
page structure.

## Chain and Phase-2 start semantics

The rate sequence is internally continuous:

`7.00 → 7.25 → 8.75 → 9.75 → 12.25 → 13.75 → 15.00 → 16.00`

C15 explicitly refers to the prior DMMD C12 dated 2020-06-25 and states that
the prior target rate was 7.00%. Its new 7.25% setting does not take effect
until 2021-09-21. Therefore the source-driven policy rate effective on the
Phase-2 start date, 2021-08-06, is **7.00%**. The validator records that as
start-state evidence; it does not fabricate a synthetic 2021-08-06 event.

C13/2022 contains a first-party reference anomaly: its page says “Circular No.
07 dated May 23, 2022”, while the official 2022 index identifies the May 23
policy circular as C09. The economically relevant explicit chain remains
consistent because C13 states 13.75% as the previous target rate and C09 states
13.75% as its new rate. The parser preserves the circular's literal reference;
the chain gate validates the previous/new target-rate values and does not
silently rewrite official text.

## Versioned parser and fail-closed chain gate

- HTML parser: `sbp_policy_circular_html_v1`
- chain validator: `sbp_policy_circular_chain_v1`
- modern PDF parser: `sbp_policy_target_rate_pdf_v2`

An ingested circular must be write-once, checksum-valid HTML attributed to
`sbp.org.pk` or `archive.sbp.org.pk`. The parser requires the explicit SBP
Policy Rate (Target Rate) wording and extracts the circular number,
announcement date, previous rate, new rate, effective date, source URL,
evidence identifier, and SHA-256. It does not treat Repo, Reverse Repo,
floor/ceiling, or other rates as the target rate.

The chain gate requires the seven circular identities above, rejects duplicate
effective dates, gaps, and previous/new-rate contradictions, proves the
Phase-2 start state from C15's explicit predecessor, and requires a preserved
modern SIR PDF bridge. Search snippets or a partly preserved chain cannot make
the series READY.

## Combined policy inventory

The complete source-audited chain plus the already preserved SIR history would
contain 20 unique effective-date events:

| Effective date | Policy target rate | Evidence state |
|---|---:|---|
| 2021-09-21 | 7.25% | circular identified; raw HTML not preserved |
| 2021-11-22 | 8.75% | circular identified; raw HTML not preserved |
| 2021-12-15 | 9.75% | circular identified; raw HTML not preserved |
| 2022-04-08 | 12.25% | circular identified; raw HTML not preserved |
| 2022-05-24 | 13.75% | circular identified; raw HTML not preserved |
| 2022-07-13 | 15.00% | circular identified; raw HTML not preserved |
| 2022-11-28 | 16.00% | circular identified; raw HTML not preserved |
| 2023-01-24 | 17.00% | checksum-preserved SIR PDF |
| 2023-03-03 | 20.00% | checksum-preserved SIR PDF |
| 2023-04-05 | 21.00% | checksum-preserved SIR PDF |
| 2023-06-27 | 22.00% | checksum-preserved SIR PDF |
| 2024-06-11 | 20.50% | checksum-preserved SIR PDF |
| 2024-07-30 | 19.50% | checksum-preserved SIR PDF |
| 2024-09-13 | 17.50% | checksum-preserved SIR PDF |
| 2024-11-05 | 15.00% | checksum-preserved SIR PDF |
| 2024-12-17 | 13.00% | checksum-preserved SIR PDF |
| 2025-01-28 | 12.00% | checksum-preserved SIR PDF |
| 2025-05-06 | 11.00% | checksum-preserved SIR PDF |
| 2025-12-16 | 10.50% | checksum-preserved SIR PDF |
| 2026-04-28 | 11.50% | checksum-preserved SIR PDF |

The 7.00% start state is not listed as an invented event. It is metadata proven
by C15's explicit previous-rate statement and predecessor reference.

## Current readiness

The real local readiness command still reports:

- `POLICY_RATE: INVALID`
- first locally parsed event: `2023-01-24`
- last locally parsed event: `2026-04-28`
- reason: no checksum-preserved effective setting at the 2021-08-06 Phase-2
  start
- `CPI: MISSING_RELEASE_EVIDENCE`
- `USD_PKR: MISSING`
- `CANONICAL_MACRO: BLOCKED`
- `test_observations_loaded: false`

To resolve the policy blocker, the seven original first-party pages must be
saved through a normal browser and ingested with their original bytes, exact
URLs, retrieval timestamps, and SHA-256 checksums. The commands are documented
in `data/raw/macro/README.md`. Until that happens, the correct decision remains
**BLOCKED_POLICY_RATE_EVIDENCE**.

## Safety statement

No access control was bypassed. No third-party rate data was used. No raw
evidence was fabricated or overwritten. No TRAIN, VALIDATION, or TEST dataset
was built or loaded. No training ran, Phase 3 was untouched, and no commit was
made.
