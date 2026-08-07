# OHLC row-level quality policy implementation

Implementation date: **2026-08-08 (Asia/Karachi)**  
Scope: AI feature generation, processed datasets, readiness, and model-management presentation.  
Minimum usable-history requirement: **252 rows** (unchanged).

## Outcome

The AI pipeline now removes invalid OHLC observations before causal feature generation instead of rejecting an otherwise valid symbol because one row contains a zero or otherwise invalid required price. Raw PSX history remains authoritative and unchanged. No price is imputed, substituted, forward-filled, backward-filled, or interpolated.

| Active ordinary-equity measure | Before | After |
|---|---:|---:|
| Active ordinary equities | 473 | 473 |
| Whole-symbol OHLC quality exclusions | 342 | 0 |
| Symbols with one or more OHLC rows removed | Not applicable | 342 |
| Invalid OHLC rows removed | Not applicable | 23,073 |
| Ready for training | 119 | 454 |
| Insufficient after exact usable-row evaluation | 12 | 19 |

The after-count was produced by rebuilding local data; it was not forced to equal the earlier simulation.

## Audit evidence and rationale

The preceding read-only audit found 23,073 affected rows across 342 of 473 active ordinary equities. No affected active symbol contained a negative price, only five contained a non-positive close, and direct checks showed representative zeros already existed in preserved official PSX HTML and matched parsed CSV/master values. The old rule rejected the complete history when any OHLC value was non-positive, including a single isolated zero-open record.

That whole-symbol response was disproportionate: it discarded hundreds or thousands of valid observations because of one source observation. Row filtering was chosen because it preserves valid official observations while preventing invalid values from contaminating returns and rolling indicators. Price imputation was rejected because LDCP substitution, filling, or interpolation would fabricate a market path and could alter returns, volatility, indicator windows, and downstream rewards.

## Implemented policy

A copied AI working frame classifies a row as invalid when any required OHLC value is missing, non-numeric, or non-positive; high is below low; volume is missing/non-numeric or negative; or another retained row-level integrity condition fails. Invalid rows are excluded before feature calculation.

Negative prices remain a symbol-level fatal condition. Invalid dates, duplicate symbol/date keys, missing/invalid volume, and negative volume also retain structural fatal handling. Duplicate output `(symbol, date)` keys are explicitly rejected. Unsupported security-type rules are unchanged.

Close `<= 0` receives no substitute. Positive-volume partial-zero rows are also removed; volume does not justify inventing an open, high, low, or close. The source master and Dataset Explorer continue to expose official values.

## Processing order and date gaps

The enforced order is:

1. Copy source market data.
2. Identify structural symbol-level failures.
3. Remove invalid OHLC rows from the AI-only working copy.
4. Sort remaining observations by symbol and date.
5. Calculate causal technical indicators.
6. Remove the 49-observation feature warm-up and rows missing required features.
7. Apply the unchanged 252-usable-row readiness requirement.
8. Create chronological train/validation/test splits.
9. Fit scalers on training partitions only and transform later partitions.

Removing a source observation creates a natural date gap. No synthetic date or replacement row is inserted. Rolling indicators operate on the ordered sequence of remaining valid observations, and adding future data cannot alter previously calculated feature rows.

## Quality metadata

Readiness and model-management now expose, per active symbol:

- raw rows;
- invalid OHLC rows removed;
- valid rows before features;
- quality-retention percentage;
- explicit `invalid_ohlc_row` removal reason when applicable;
- warm-up rows removed;
- rows after warm-up;
- exact usable feature rows;
- first and last usable dates;
- train, validation, and test rows;
- readiness status.

The Training and Models page presents these fields with human-readable labels and full dates/counts.

## Rebuilt universe funnel

### Full master AI universe

| Stage | Symbols | Excluded at stage | Rule |
|---|---:|---:|---|
| Raw/master history | 4,741 | — | Unique local master symbols |
| Supported master security types | 3,033 | 1,708 | Existing security-type policy, unchanged |
| Structural quality gate | 3,033 | 0 | Fatal date/key/negative-price/volume integrity |
| At least one valid pre-feature row | 2,776 | 257 | Row-level OHLC filtering |
| At least one usable post-warm-up row | 1,899 | 877 | Existing indicators and 49-row warm-up |
| Processed master output | 1,899 | 0 | 940,565 rows; no duplicate keys |

The full-master filter removed **320,139** invalid rows across all supported and unsupported histories examined by the shared preparation path. This broader number is not the same scope as the 23,073 removed from active ordinary equities.

### Active ordinary-equity readiness

| Stage | Symbols | Excluded at stage |
|---|---:|---:|
| Active ordinary equities | 473 | — |
| Pass structural fatal checks | 473 | 0 |
| Retain at least one valid OHLC row | 473 | 0 |
| Meet 252 exact usable rows after warm-up | 454 | 19 |
| Processed per-symbol datasets | 454 | 0 |
| Chronological symbol splits/scalers | 454 | 0 |
| Final ready for training | 454 | 0 |

Retention distribution across all 473 active ordinary equities:

- 342 have at least one row removed;
- 246 retain at least 99% of raw rows (115 of these are affected symbols);
- 351 retain at least 95% of raw rows (220 of these are affected symbols);
- 122 retain less than 95%;
- no active ordinary symbol is excluded solely by the new quality gate;
- 19 remain excluded for insufficient exact usable history.

## Remaining active ordinary exclusions

All remaining active ordinary exclusions have readiness status **Insufficient History**. Four have zero post-warm-up rows; none fails for a missing processed file, split/scaler error, duplicate output key, or structural quality error.

| Symbol | Raw rows | Invalid rows removed | Valid pre-feature rows | Usable rows | Reason |
|---|---:|---:|---:|---:|---|
| ANLNV | 87 | 24 | 63 | 14 | Below 252 usable rows |
| ARMG | 84 | 2 | 82 | 33 | Below 252 usable rows |
| ASIC | 416 | 148 | 268 | 219 | Below 252 usable rows |
| AWTX | 298 | 186 | 112 | 63 | Below 252 usable rows |
| BLUEX | 149 | 0 | 149 | 100 | Below 252 usable rows |
| EWIC | 460 | 182 | 278 | 229 | Below 252 usable rows |
| GDL | 112 | 0 | 112 | 63 | Below 252 usable rows |
| ITANZ | 265 | 150 | 115 | 66 | Below 252 usable rows |
| MUGHALC | 272 | 0 | 272 | 223 | Below 252 usable rows |
| NATM | 183 | 170 | 13 | 0 | No post-warm-up usable rows |
| PAKQATAR | 147 | 0 | 147 | 98 | Below 252 usable rows |
| PQGTL | 119 | 0 | 119 | 70 | Below 252 usable rows |
| SELECT | 18 | 0 | 18 | 0 | No post-warm-up usable rows |
| SLM | 36 | 0 | 36 | 0 | No post-warm-up usable rows |
| SPAC1 | 58 | 0 | 58 | 9 | Below 252 usable rows |
| SPAC2 | 18 | 0 | 18 | 0 | No post-warm-up usable rows |
| SPSL | 50 | 0 | 50 | 1 | Below 252 usable rows |
| WAHDAT | 59 | 0 | 59 | 10 | Below 252 usable rows |
| ZUMA | 194 | 0 | 194 | 145 | Below 252 usable rows |

## Rebuild results

| Measure | Result |
|---|---:|
| Raw master rows | 1,527,808 |
| Raw master symbols | 4,741 |
| Processed master rows | 940,565 |
| Processed master symbols | 1,899 |
| Per-symbol processed datasets | 454 |
| Ready for training | 454 |
| Insufficient history | 19 |
| Split/scaler sets | 455 (454 symbol + 1 master) |
| Processed date range | 2016-10-06 through 2026-08-05 |
| Duplicate processed `(symbol, date)` keys | 0 |
| Non-positive OHLC rows in processed master | 0 |
| Rebuild errors | 0 |

## Reproducibility and limitations

- The quality rule is part of `FEATURE_VERSION`, so derived artifacts identify the changed preprocessing contract.
- Raw files and master source rows are not rewritten by the filter. The ordered rebuild reproduced the master CSV byte-for-byte; registry refresh metadata changed as part of the standard rebuild.
- Filtering changes observation adjacency. Indicators measure valid-observation sequences, not a synthetic calendar-complete series.
- This policy does not identify corporate actions, exchange suspensions, or every semantic cause of a zero field. It conservatively excludes the row.
- A high row-removal percentage is visible in metadata but does not independently lower the fixed 252-row requirement or fabricate observations.
- No model was trained, no live HTTP request was made, and no backfill/raw source file was changed.
