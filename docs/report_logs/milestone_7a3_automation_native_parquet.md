# Milestone 7A.3 — Automation/native Parquet integration

Audit date: 2026-08-28 (Asia/Karachi)

## Outcome

The Streamlit manual action, launchd/scheduled action, and CLI now delegate to
one orchestration function. A successful source acquisition always enters the
native incremental pipeline; rebuilding legacy feature/AI products remains an
optional later stage. No RL training or TEST-partition access is part of this
flow.

Decision: **READY_AUTOMATION_NATIVE_PARQUET**

## Previous flow and observed failure

The old flow was:

`Streamlit button -> run_manual_update -> run_incremental_update -> PSX request
loop -> index refresh -> legacy master CSV -> listing/registry refresh ->
optional feature builds`

It never invoked `native_market_pipeline.incremental_update`. Missing-date
discovery enumerated every calendar day from the earliest local CSV, so
weekends, holidays, and past empty dates were repeatedly placed ahead of recent
dates. The observed run therefore began processing 1,178 old gaps rather than
the few recent gaps. It also ran in a process that had loaded the old code, so
`automation.json` remained `running` until that entire loop returned. The
retained lock recorded a PID, but shell-level liveness could not be established
reliably in the restricted audit environment. The legacy process subsequently
completed and wrote an old-format `success` record (20 successful and 1,178
skipped), confirming that the observed delay was the oversized legacy queue.
Production state was not manually rewritten during this milestone.

The replacement lock checks both age and owner-PID liveness. On the next page
load or run, an abandoned `running` record is finalized as `failed`, the stale
lock is removed, and the recovery is recorded in the durable run audit.

## Canonical orchestration

The shared workflow is:

1. discover valid local daily CSVs and the trusted native source manifest;
2. exclude weekends and recorded non-request dispositions;
3. request only genuinely unresolved weekdays;
4. validate downloaded daily CSVs;
5. refresh authoritative listings when native inputs exist;
6. call the native incremental pipeline;
7. update the sector-enriched master CSV, affected symbol CSVs, affected daily
   Parquets, consolidated Parquet, and native state atomically;
8. refresh the company registry;
9. optionally rebuild legacy index/feature/AI products, only after native
   success;
10. finalize automation status and audit details.

New latest dates and recovered old dates use the same incremental path. The
native merge is keyed and deterministically ordered by market date and symbol,
so an old date is inserted without requiring it to be later than the existing
maximum. Conflicting replacements fail closed.

## Source-evidence policy and real-state audit

Daily source CSV provenance—not an obsolete legacy master maximum—is the
freshness source of truth. A trusted native manifest entry also counts because
it retains the originating source CSV name and hash. A date found only in
analytical Parquet does not count as source evidence.

At audit time:

- valid local CSV dates: 2,508, spanning 2016-07-26 through 2026-08-27;
- trusted native manifest dates: 2,489, spanning 2016-07-26 through 2026-08-20;
- consolidated Parquet dates: 2,489 over the same range;
- Parquet-only dates without CSV-manifest/local evidence: 0;
- native dates retaining external/bootstrap provenance: 2,489;
- local CSV dates not yet represented in native Parquet: 19;
- of those 19, eight are Sunday files created by the legacy run and are not
  eligible for native ingestion; one additional weekend file (2018-12-16) is
  already present in the bootstrapped native source set and is reported rather
  than changed;
- eligible pending local native inputs: 11 dates / 6,915 rows:
  2026-08-10, 2026-08-11, 2026-08-12, 2026-08-13, 2026-08-17,
  2026-08-18, 2026-08-19, 2026-08-21, 2026-08-24, 2026-08-25,
  and 2026-08-27;
- unresolved weekday requests through 2026-08-28: 2026-08-14,
  2026-08-26, and 2026-08-28. Each currently has a retained 903-byte empty
  HTML response but no valid source CSV. This is reported as unresolved source
  evidence, not silently treated as a holiday.

The 2026-08-20 native manifest entry is determinably bootstrap/external: its
32,318-byte file hash exactly matches
`../psx-data-sync/data/raw/market_2026-08-20.csv`. The subsequently created
local counterpart is 57,433 bytes and has a different hash. The native
manifest preserves the external source hash and provenance; no source was
rewritten.

## Incremental write behavior

Normal updates no longer rewrite every symbol CSV or every daily Parquet. The
pipeline stages existing immutable files into its transaction, rewrites only
affected symbol/date artifacts, and still regenerates the consolidated master
CSV and consolidated Parquet required for a deterministic complete dataset.
Promotion remains atomic and rollback-safe. Result metadata reports rows added
or replaced and exact per-symbol/daily artifact write counts.

## Status and UI behavior

The durable latest-run record includes start/finish times, target date,
discovered/attempted/downloaded/skipped/failed dates, native result, row and
daily-file counts, latest consolidated date, AI stage, source-evidence warnings,
stale recovery, and failure reason. Returned runs end in `success`,
`partial_success`, `failed`, or `no_update_needed`.

The Streamlit page uses a native status container fed by backend stage events:
source check, fetching, listing refresh, native CSVs, affected symbol CSVs,
daily Parquets, consolidated Parquet, optional AI rebuild, and completion. It
does not fabricate percentages.

## Safety and acceptance

All acceptance tests are offline and use temporary artifacts. The real audit
was read-only; no live HTTP request was made and no production native update
was launched. The three unresolved dates therefore remain for a later bounded
manual run, and the 11 eligible local inputs remain ready for the canonical
incremental path. Raw CSV/HTML files, TEST data, RL code, and `psx-data-sync`
were not modified.

Final verification:

- `.venv/bin/python -m pytest -q`: 723 passed, 2 skipped;
- `.venv/bin/python -m pip check`: no broken requirements;
- `git diff --check`: passed;
- no Streamlit process was listening on port 8501, so no app process required
  stopping or restarting.
