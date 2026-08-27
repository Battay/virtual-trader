# Milestone 7A.2 — Native CSV to Sector-Enriched Parquet Pipeline

Audit/build date: 2026-08-27 (Asia/Karachi)

## Decision

`READY_TO_MIGRATE_LOCAL_PARQUET`

The native full build, isolated incremental acceptance tests, exact bootstrap
comparison, atomic failure tests, and local reader verification passed. No RL
training, TEST access, download, or write to `psx-data-sync` occurred.

## Previous data flow

1. `virtual-trader/data/raw/csv/market_YYYY-MM-DD.csv` fed
   `data/master/psx_master.csv` through `csv_store.py`.
2. The feature builder read that master plus the company registry and wrote
   feature-engineered AI products to
   `data/processed/master/psx_ai_master.csv` and
   `data/processed/symbols/<SYMBOL>.csv`.
3. The read-only analytical boundary defaulted to the sibling-project file
   `../psx-data-sync/data/parquet/market.parquet`.
4. No native sector-enriched market CSV, daily Parquet partition set,
   consolidated Parquet writer, or unified incremental transaction existed.

The pre-existing `psx_ai_master.csv` and `data/processed/symbols/` files are AI
feature products, not normalized market exports. They were preserved. The new
market exports use non-colliding paths.

## New source-of-truth flow

```text
validated immutable daily CSV evidence
        -> normalize and validate once
        -> attach current authoritative sector context once
        -> data/processed/master/psx_market_master.csv
        -> data/processed/market_symbols/<SYMBOL>.csv
        -> data/parquet/daily/market_YYYY-MM-DD.parquet
        -> data/parquet/market.parquet
```

The normalized in-memory records are the logical source for every generated
format. The master CSV is a human-readable/research export, symbol CSVs are
convenience artifacts, daily Parquets are inspectable date partitions, and the
consolidated Parquet is the canonical analytical/RL read artifact. Existing RL
code and artifacts were not changed.

## Canonical record contract

Schema version: `native_market_record_v1`

| Field | Physical Parquet type | Rule |
|---|---|---|
| `market_date` | `date32` | non-null; source column or validated filename date |
| `symbol` | `string` | non-null, stripped, uppercase |
| `ldcp`, `open`, `high`, `low`, `close`, `change`, `change_percent` | `double` | finite; zero O/H/L remains allowed by source policy; close must be positive |
| `volume` | `int64` | finite, integral, non-negative |
| `sector_current` | nullable `string` | current authoritative common-equity sector |
| `sector_source` | nullable `string` | official listing source URL |
| `sector_snapshot_date` | nullable `string` | current listing snapshot date |

Rows are ordered by `(market_date, symbol)`. Duplicate business keys are
rejected. No imputation, cleaning, inferred sector, missing bar, or fabricated
history is introduced.

## Sector enrichment contract

The input is the cached authoritative PSX listing snapshot dated 2026-08-02.
Sector provenance is attached only to the frozen common-equity policy:
authoritative `ordinary_equity` or `gem_equity`, excluding fund/REIT sectors
and Modarabas that the listing table can label with an equity-like type. This
produced exactly 508 matched symbols and 823,439 enriched historical rows.

The metadata label is
`current_listing_context_not_historical_membership`. It provides present-day
identity/context only and must not be interpreted as historical effective-dated
sector membership. Unmatched and non-common instruments retain null sector
fields; ticker names are never used to infer sectors.

## Output and update design

- Master CSV: 1,527,792 rows, 13 columns, deterministic order.
- Per-symbol CSVs: 4,668 isolated files, chronological and duplicate-free.
- Daily Parquet: 2,489 ZSTD-compressed files with schema, source-set, sector
  context, and content-hash metadata.
- Consolidated Parquet: 1,527,792 rows, 35,829,331 bytes, ZSTD-compressed,
  atomically promoted after complete validation.
- Pipeline state: `data/metadata/native_market_pipeline_state.json`, containing
  source hashes, source dates, row counts, sector snapshot, date range, symbol
  count, schema/pipeline versions, output hashes, status, and timestamp.
  Failures are recorded separately so the last valid state is not replaced.

Incremental updates normalize the new daily evidence, reject conflicting
existing keys, treat byte/value-identical reprocessing as an idempotent no-op,
refresh current sector annotations consistently, stage a complete bundle, and
atomically rewrite/promote it. Rewriting the approximately 34 MiB consolidated
file was chosen over pretending that a single Parquet file supports safe row
append. Transaction rollback restores every prior output if any promotion step
fails. A later full rebuild also fails closed if its source set would remove or
silently alter an already accepted canonical record; no implicit deletion or
replacement policy exists.

## Rebuild/incremental equivalence acceptance

The deterministic offline fixture contained 4 rows, 3 symbols, and 2 market
dates. Building both dates together and building one date followed by one
incremental update produced identical:

- schema and values;
- row ordering and business keys;
- symbol/date sets and sector fields;
- source-set hash and canonical content hash.

Reprocessing the same source did not rewrite the master or consolidated file.
A conflicting same-key row was rejected. A simulated mid-promotion exception
restored master, symbol, Parquet, and state hashes exactly.

## Real migration comparison

The acceptance build used the 2,489 validated daily CSV files in the sibling
project strictly as read-only ingestion evidence. `virtual-trader`'s local CSV
set currently has 2,488 dates and lacks 2026-08-20, so it was not sufficient for
an exact bootstrap comparison by itself.

| Check | Native | Bootstrap | Result |
|---|---:|---:|---|
| Rows | 1,527,792 | 1,527,792 | equal |
| Market dates | 2,489 | 2,489 | equal |
| Symbols | 4,668 | 4,668 | equal |
| Duplicate keys | 0 | 0 | equal |
| Date range | 2016-07-26–2026-08-20 | 2016-07-26–2026-08-20 | equal |
| Canonical overlapping-core hash | `8c6d1f0670475789c75c786f557570c573c7358501ddb11efdbf925f28bf5d7f` | same | equal |

Native enriched content hash:
`09f54611ace9cdeeff537cb07e6acb4fa21b8178bd4e2530bb1613f570b3ac25`.

Source-set hash:
`2d5904cc769f6e39b57b9c21fcf59271dd1f212a4352c2440e6465d04c6c731a`.

The bootstrap Parquet was not modified or deleted.

## Reader migration

Resolution is now:

1. explicit function argument;
2. `PSX_MARKET_PARQUET_PATH` environment override;
3. `virtual-trader/data/parquet/market.parquet`.

The sibling Parquet remains an explicit migration comparison constant only; it
is no longer an implicit runtime fallback.

## Quality audit

The local reader reported a valid required schema, no required-column nulls,
no negative volume, no non-positive close, and no duplicate keys. Source-policy
zero O/H/L availability remains unmodified: 318,282 zero-open rows, 247,486
zero-high rows, 306,377 zero-low rows, and 318,320 rows with any zero O/H/L.
Comparable positive-value OHLC inconsistencies are reported, not cleaned:
680 high-below-open, 10,007 high-below-close, 435 low-above-open, 10,723
low-above-close, 0 high-below-low, covering 21,840 distinct rows.

This pipeline is a preservation and provenance boundary, not a source-data
repair step.
