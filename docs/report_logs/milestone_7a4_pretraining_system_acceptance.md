# Milestone 7A.4 — Pre-Training System Hardening and Application Acceptance

Audit timestamp: 2026-08-28T14:39:11+05:00  
Branch/starting commit: `feat/rl-environment-v1` / `d99feb6`  
Decision: **PRETRAINING_SYSTEM_READY**

## Scope and safety

This acceptance did not train an RL model, load TEST observations, modify the
`psx-data-sync` repository, rebuild AI/feature products, install a scheduler,
or create a commit. Source CSV and retained raw-HTML evidence were not changed.
The only production-data writes were the authorized native ingestion of 11
already-valid local CSV dates, current listing/registry refresh, and atomic
regeneration of Virtual Trader's derived native artifacts.

## 7A.3 orchestration acceptance

Manual Streamlit, scheduled launchd, and CLI entry points delegate to the same
`run_update_orchestration` function. The UI calls `run_manual_update`, the CLI
calls `run_manual_update` or `run_scheduled_update`, and the scheduler invokes
the scheduled CLI. There is no second update implementation.

An offline CLI acceptance through 2026-08-27 returned
`no_update_needed`, processed zero dates, performed no HTTP call, and finalized
the persisted automation state. The state distinguishes current/not-required
stages from incomplete stages.

## Source-date reconciliation

The following 11 validated local CSV dates were ingested without HTTP:

`2026-08-10`, `2026-08-11`, `2026-08-12`, `2026-08-13`,
`2026-08-17`, `2026-08-18`, `2026-08-19`, `2026-08-21`,
`2026-08-24`, `2026-08-25`, and `2026-08-27`.

The incremental operation added 6,915 rows, wrote exactly 11 daily Parquets,
updated 713 affected per-symbol CSVs, and atomically updated the canonical
master and consolidated Parquet. It covered both recovered old-date insertion
and newest-date extension without regenerating all existing daily files.

Final source evidence:

- Local valid CSV dates: 2,508, from 2016-07-26 through 2026-08-27.
- Native manifest dates: 2,500, from 2016-07-26 through 2026-08-27.
- Eligible local dates pending native ingestion: 0.
- Local-not-native dates: 8; every one is a Sunday artifact from the historical
  legacy workflow and is intentionally ineligible.
- Native provenance: 2,489 `external_validated_csv` dates and 11
  `virtual_trader_raw_csv` dates.

### Unresolved source dates

| Date | Classification | Evidence and retry policy |
|---|---|---|
| 2026-08-14 | `confirmed_non_trading` | Official PSX Independence Day closure plus retained 903-byte empty response; automatic retry disabled. |
| 2026-08-26 | `source_anomaly` | Retained 903-byte empty response while adjacent 2026-08-25 and 2026-08-27 contain valid data. It is explicitly **not** inferred to be a holiday; automatic retry is disabled pending manual source review. |
| 2026-08-28 | `not_final` | The bounded acceptance check occurred at 03:20 PKT before the current date was final; retained response was empty and automatic retry remains enabled. The latest available accepted PSX date is 2026-08-27. |

The bounded direct recheck attempted only these three dates. Local DNS
resolution failed before a PSX response was received, so it made no data
change and did not loop. The dispositions above are persisted and auditable.

## Canonical market master

There is one human-facing market master:

`data/master/psx_master.csv`

Its schema is:

`market_date, symbol, ldcp, open, high, low, close, change, change_percent,
volume, sector_current, sector_source, sector_snapshot_date`.

Legacy consumers receive an in-memory `date` alias; the canonical file is not
downgraded. An explicitly named legacy compatibility path exists only for an
opt-in legacy build and was not generated. The transitional
`data/processed/master/psx_market_master.csv` was retired after its 1,527,792
core rows were proved to be an exact subset of the canonical master: zero
missing keys and zero changed core values. The extra 6,915 rows are the newly
ingested dates. `data/processed/master/psx_ai_master.csv` remains a distinct,
untouched AI/feature artifact.

## Sector coverage

- Rows with current-sector provenance: 828,384.
- Rows without a current sector: 706,323.
- Symbols with current-sector provenance: 510.
- Symbols without current-sector provenance: 4,171.
- Sector snapshot date: 2026-08-28.
- Current authoritative common equities covered: 510 of 510.
- The previously frozen 508-symbol research snapshot remains conceptually
  distinct; the refreshed current listing snapshot added two current
  identifiers and was not used to train or silently revise a model universe.

Missing sectors are expected for historical-only, inactive, entitlement,
contract-like, debt/fund, and otherwise non-current identifiers. Current sector
metadata is deliberately not presented as effective-dated historical
membership.

## Historical Backfill acceptance

- Date bounds use actual PSX support beginning 2016-07-26.
- AppTest load and control rendering passed.
- Controlled Preview Backfill Plan passed with no exception and reported that
  normal requests are not required for the current stored range.
- Resume, Retry Failed Dates, Retry Temporary Dates, weekend classification,
  bounded retries, persisted attempt evidence, and circuit-breaker behavior
  remain covered by the complete offline suite.
- A successful or safely reconciled backfill CSV is now passed into the shared
  lock-protected native reconciliation flow, updating daily/consolidated/master
  artifacts and registry metadata.
- No multi-year live backfill was run.

## Automation and rebuild acceptance

- Settings save passed and persisted the existing disabled/AI-rebuild-off
  configuration.
- Fetch, canonical rebuild, and macOS scheduler controls rendered with truthful
  enabled states; install was available, while trigger/uninstall were disabled
  because no launch agent is installed.
- Stale PID-lock recovery was exercised after an interrupted acceptance worker;
  the abandoned lock was recovered and the rerun completed atomically.
- The explicit canonical rebuild preserved all 2,500 source-manifest entries,
  including the external provenance of 2026-08-20, and regenerated derived
  artifacts in about 39 seconds.
- The rebuild button is labelled **Rebuild canonical market artifacts**. It
  rebuilds market master, per-symbol CSVs, daily Parquets, consolidated Parquet,
  sector enrichment, and validation. It does not rebuild RL or AI products.
- No persistent scheduler was installed.

## Page-by-page Streamlit acceptance

The live server health endpoint returned `ok`, and the server was stopped after
acceptance. AppTest results:

| Page | Status | Runtime (s) |
|---|---:|---:|
| Application navigation / default page | PASS | 6.596 |
| Market Overview | PASS | 0.598 |
| Fetch Data | PASS | 0.107 |
| Market Indices | PASS | 0.142 |
| Dataset Explorer | PASS | 2.920 |
| Stock Explorer | PASS | 4.080 |
| Automation | PASS | 8.285 |
| Company Registry | PASS | 0.670 |
| Training & Models | PASS | 53.812 |
| Historical Backfill | PASS | 7.413 |

Every registered page exists, compiles, resolves project-local paths, renders
controls/empty states without an uncaught exception, and contains no runtime
reference to `psx-data-sync` or a user-specific absolute path.

## Final data-integrity snapshot

| Check | Result |
|---|---:|
| Canonical rows | 1,534,707 |
| Unique market dates | 2,500 |
| Unique symbols | 4,681 |
| Date range | 2016-07-26 to 2026-08-27 |
| Duplicate `(market_date, symbol)` rows | 0 |
| Required-field nulls | 0 |
| Negative-volume rows | 0 |
| Non-positive-close rows | 0 |
| Deterministic ordering | PASS |
| Daily Parquet files | 2,500 |
| Daily Parquet rows | 1,534,707 |
| Daily vs consolidated logical hash | PASS |
| Canonical master vs Parquet core/full hash | PASS |

Source-set hash:
`75ea704968ed5535dde78d2c335bd2590446855f5817464fd0fd8b670066a245`

Canonical logical content hash:
`645a35f9c141a94c1e72ed134401440b9ad5b27689d65a430b205761ece3d357`

Consolidated physical SHA-256:
`0177ad84c8588be3e5b96068f2da0b7b9b3cbb118078904f2ed27bb3326f9be6`

Zero O/H/L values remain source-policy-allowed availability indicators and were
not changed. The read-only audit reports 318,297 zero-open rows, 247,501
zero-high rows, 306,392 zero-low rows, 318,335 rows with any zero O/H/L, and
21,940 positive-comparable-value OHLC inconsistency rows for later policy
review.

## State and performance findings

- Automation state: `no_update_needed`; no stale running marker.
- Native state: `completed`, latest date 2026-08-27.
- Backfill state: intentionally `paused` from its historical workflow, not
  `running`; no lock is held.
- Scheduler: not installed or loaded.
- Automation lock: absent.
- Native staging directories: absent.
- Streamlit server after acceptance: stopped.
- Ordinary incremental ingestion touched 11 affected daily files and 713
  symbols, not all 2,500 days.
- Full daily regeneration occurs only under the explicit repair/rebuild action.
- Normal page loads do not invoke a dataset rebuild; the slowest AppTest page
  was Training & Models at 53.812 seconds because it audits many local training
  artifacts. This is a performance follow-up, not a correctness blocker.

## Verification

- Complete pytest: **733 passed, 2 skipped**.
- `pip check`: **No broken requirements found**.
- `git diff --check`: **passed**.
- No commit was created.
