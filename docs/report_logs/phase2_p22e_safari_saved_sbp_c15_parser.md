# Phase 2 P2.2e — Safari-Saved SBP C15 Parser Repair

Audit timestamp: `2026-09-01T22:20:48Z`  
C15 evidence decision: **READY_C15_POLICY_EVIDENCE**  
Overall policy chain: **INCOMPLETE**

## Preserved evidence

- source URL: `https://archive.sbp.org.pk/dmmd/2021/C15.htm`
- source identifier: `sbp_dmmd_circular_15_2021`
- ingested file: `20260901T221400Z_sbp_dmmd_C15_2021.html`
- byte size: `26,355`
- SHA-256: `f7770636af25b9be99b16e8e1a8ffc16448bc05a3acc7f1981d474928990563f`
- declared encoding: `iso-8859-1`
- browser-compatible byte encoding: `Windows-1252`

The source and manifest were read only. Their bytes, timestamps, and checksums
were not changed.

## DOM and visible-text audit

The legacy document contains two head titles:

1. `State Bank of Pakistan`
2. stale template title `DMMD Circular No. 01 of 2018`

The authoritative circular content is in the body’s nested table:

| Field | DOM structure | Normalized visible text |
|---|---|---|
| identity | `table > tr > td > strong`, split by a line break | `DMMD Circular No. 15 of 2021` |
| announcement | adjacent `td > b` | `September 20, 2021` |
| target-rate change | `blockquote > ol > li`, with nested span/bold elements | `SBP has decided to increase its ‘Policy Rate’ (Target Rate) from 7.00% to 7.25%.` |
| effective date | `blockquote > p` | `Above changes are effective from September 21, 2021.` |

The page is ordinary legacy HTML rather than a Safari webarchive wrapper.
Nested tags, table layout, non-breaking spaces, and line breaks are present but
become unambiguous after body-only visible-text normalization.

## Exact failure

BeautifulSoup initially honored the page’s literal ISO-8859-1 declaration.
The source nevertheless uses Windows-1252 bytes `0x91` and `0x92` around
`Policy Rate`. They emerged as C1 control characters `\x91` and `\x92`, which
the strict smart-quote regex did not accept. Consequently the previous/new
target-rate match was missing; the circular number, announcement date, and
effective date already matched.

A second latent error would then have selected the stale head title’s C01/2018
identity because identity and announcement were searched independently.

## Repair

Parser version remains `sbp_policy_circular_html_v1`; its evidence semantics
did not change. It now:

1. reparses a detected legacy ISO-8859-1 page using the browser-standard
   Windows-1252 mapping;
2. extracts visible body text only and excludes scripts, styles, templates,
   and non-visible head titles;
3. normalizes whitespace and non-breaking spaces;
4. derives circular number, year, and announcement date from the same visible
   heading match;
5. continues to require explicit previous rate, new target rate, and effective
   date;
6. continues to enforce first-party source domains and checksum-valid
   manifests.

No missing field is inferred. Repo, Reverse Repo, floor, ceiling, and other
rate concepts remain unacceptable substitutes.

## Corrected real C15 extraction

| Field | Parsed value |
|---|---|
| circular number | `15` |
| circular year | `2021` |
| announcement date | `2021-09-20` |
| previous target rate | `7.00` |
| new target rate | `7.25` |
| effective date | `2021-09-21` |
| predecessor | `DMMD Circular No. 12`, dated `2020-06-25` |
| source checksum | `f7770636af25b9be99b16e8e1a8ffc16448bc05a3acc7f1981d474928990563f` |

## Readiness after repair

`POLICY_RATE` is `INVALID` only because the circular chain is incomplete. C15
is listed as preserved and parseable. Missing checksum-preserved pages are:

- DMMD Circular No. 21 of 2021
- DMMD Circular No. 23 of 2021
- DMMD Circular No. 06 of 2022
- DMMD Circular No. 09 of 2022
- DMMD Circular No. 13 of 2022
- DMMD Circular No. 20 of 2022

The readiness output no longer describes C15 as malformed. `CANONICAL_MACRO`
remains blocked because this policy chain, CPI release evidence, and USD/PKR
M2M evidence are incomplete.

## Safety

No replacement source was fetched. The preserved C15 HTML and SIR PDF were not
altered or overwritten. No canonical macro artifact or split was created. TEST
remained sealed, no training ran, Phase 3 was untouched, and no commit was
made.
