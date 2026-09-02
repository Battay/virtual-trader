# Phase 2 P2.2b — Manual Authoritative Macro Evidence Completion

Decision: **READY_FOR_MANUAL_MACRO_INGEST**

Audit timestamp: `2026-09-01T15:11:27Z`

Contract evidence hash: `2313ca5d0d863c87c5dda53f7b2c6663f3b4e1b51f07143065511a728cf457d9`

The manual ingestion and parser infrastructure is ready. The macro dataset is
not ready: no SBP policy/M2M files have been ingested, and PBS CPI first-release
evidence remains incomplete. No canonical Parquet, split, or scaler was made.

## 1. SBP policy target rate

### Browser acquisition

1. Open the official SBP [Structure of Interest Rates PDF](https://www.sbp.org.pk/ecodata/sir.pdf) in a normal browser.
2. Save the original PDF, without printing or converting it, as
   `sbp_structure_of_interest_rates_2026-09-01.pdf`.
3. Confirm that the document includes the `w.e.f.` column and the separate
   `SBP Reverse Repo Rate`, `SBP Repo Rate`, and `SBP Policy (Target) Rate`
   columns. The required event history must include a setting effective on or
   before `2021-08-06` and remain current through the acquisition date.
4. If SBP EasyData is used instead, export the official series
   `TS_GP_IR_SIRPR_AH.SBPOL0030` directly. Preserve the untouched export. It
   must include an effective date and target-rate value; an announcement date
   is optional but retained when the official export supplies it.

### Checksum and ingest

```bash
shasum -a 256 /path/to/sbp_structure_of_interest_rates_2026-09-01.pdf

.venv/bin/python -m data_pipeline.src.macro_evidence \
  --ingest-file /path/to/sbp_structure_of_interest_rates_2026-09-01.pdf \
  --series sbp_policy_target_rate \
  --source-url https://www.sbp.org.pk/ecodata/sir.pdf \
  --source-id sbp_structure_of_interest_rates_sir \
  --retrieved-at 2026-09-01T15:11:27Z \
  --media-type application/pdf \
  --sha256 PASTE_THE_64_CHARACTER_SHA256 \
  --source-version retrieved_2026-09-01 \
  --provenance-notes "Saved directly from the official SBP PDF in a browser"
```

The parser outputs `policy_rate`, nullable `announcement_date`,
`effective_date`, `effective_available_timestamp`, evidence id, and evidence
hash. It never infers an announcement date. If announcement predates the
effective date, availability begins at the effective-date boundary. If the
announcement is same-day or unavailable, availability is delayed to the next
Lead-Agent decision boundary.

Parser status: **READY_FOR_MANUAL_EVIDENCE**.

## 2. SBP USD/PKR Mark-to-Market rate

### Browser acquisition

1. Open the official SBP [M2M history](https://www.sbp.org.pk/ecodata/rates/m2m/M2M-History.asp).
2. Preserve official daily M2M PDFs covering `2021-08-05` through
   `2026-08-26`, or use a single official M2M export if the SBP interface offers
   one. Do not use the `CRates` conversion-rate workbooks.
3. For a daily PDF, retain the original SBP filename such as `26-Aug-26.pdf`.
   It must say `Exchange Rates for Mark to Market Revaluation`, show its
   observation date, and include the `USD` row. The parser selects the first
   or `Ready` USD value, which SBP identifies as the daily USD M2M rate.
4. An official delimited export must contain a date column and an explicit USD
   M2M/Ready value column. It must be attributable to the official M2M URL.

### Checksum and ingest template for each official daily PDF

```bash
shasum -a 256 /path/to/26-Aug-26.pdf

.venv/bin/python -m data_pipeline.src.macro_evidence \
  --ingest-file /path/to/26-Aug-26.pdf \
  --series sbp_usd_pkr_m2m \
  --source-url https://www.sbp.org.pk/ecodata/rates/m2m/2026/Aug/26-Aug-26.pdf \
  --source-id sbp_usd_pkr_m2m_daily_2026_08_26 \
  --retrieved-at 2026-09-01T15:11:27Z \
  --media-type application/pdf \
  --sha256 PASTE_THE_64_CHARACTER_SHA256 \
  --source-version retrieved_2026-09-01 \
  --provenance-notes "Saved directly from the official SBP M2M archive"
```

The command is intentionally one-file-at-a-time so every official response has
its own URL and checksum. A genuine official bulk M2M export can instead be
ingested once with its exact official URL and media type.

The parser outputs `observation_date`, `usd_pkr_m2m`, the next-decision
`effective_available_timestamp`, evidence id, and evidence hash. It rejects
conversion-rate sources even if hosted by SBP.

Parser status: **READY_FOR_MANUAL_EVIDENCE**.

## 3. PBS CPI release timing

The preserved historical table supplies 108 National CPI YoY values from
2017-07 through 2026-06. Official CMS metadata supplies 11 exact recent
release timestamps. For the Phase-2 interval, one value month (`2026-07`) and
exact first-release evidence for 51 months (`2021-07` through `2025-09`) remain
missing.

The PBS FAQ says CPI is normally released on the first or second day of each
month. That statement is not a historical guarantee: a preserved July-2026
release was published on August 3. It also does not prove that values in the
current history equal every original release. Therefore no synthetic
first/second-day or M+2 release date was assigned.

To unblock CPI, preserve the original first-party monthly review/press-release
PDF and its official PBS publication page or CMS timestamp for each missing
month. An official archive-page publication timestamp may be used as a later,
conservative availability bound for documents linked by that page, but must be
labelled as such; it is not the original release timestamp.

CPI decision: **MISSING_RELEASE_EVIDENCE**.

## 4. Readiness command

```bash
.venv/bin/python -m data_pipeline.src.macro_evidence --readiness
```

Current result:

| Component | Status |
|---|---|
| POLICY_RATE | MISSING |
| CPI | MISSING_RELEASE_EVIDENCE |
| USD_PKR | MISSING |
| CANONICAL_MACRO | BLOCKED |

The command is read-only and lists every missing evidence class/month. The
canonical build guard raises before any output write unless all three series
are `READY`.

## 5. Scientific and safety outcome

- First-party SBP/PBS hosts are enforced by series ownership.
- Ingest is checksum-required, atomic, write-once, and overwrite-refusing.
- Policy announcement/effective dates remain distinct.
- USD/PKR is strictly the SBP M2M Ready/spot value, not conversion-rate data.
- No historical CPI release date or first-release value was fabricated.
- TEST remained sealed; no TEST observations were loaded.
- No canonical macro dataset, calendar split, or scaler was created.
- No environment, PPO/SAC, Phase-3, model, or training-state work occurred.
- No commit was made.

**READY_FOR_MANUAL_MACRO_INGEST**
