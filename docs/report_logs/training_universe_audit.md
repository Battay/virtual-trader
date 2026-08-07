# PSX AI Training Universe Audit

- Audit timestamp (Asia/Karachi): `2026-08-08T02:52:44+05:00`
- Raw/master symbols: **4,741**
- Processed-master symbols: **213**
- Per-symbol ready datasets: **119**
- Configured feature warm-up: **49 rows**
- Configured minimum usable history: **252 rows**

## Executive conclusion

The reported **119 is internally correct under the current code and artifacts**, but it should not be interpreted as the true number of trainable PSX ordinary equities. The dominant reduction is the fatal data-quality rule: one zero/non-positive open, high, low, or close anywhere in a symbol's complete history excludes the whole symbol. This removes **3,969/4,741 symbols**, including **342/473 active ordinary equities**. No missing split, scaler, duplicate-date, or processed-file failure explains the current 119.

The first PPO experiment should use a **curated liquid pilot of 20 symbols from the current 119**, ranked by median `close × volume` over each symbol's latest 60 observations. This limits compute and execution-quality risk while the zero-price policy is reviewed. It is not evidence that the other 99 ready symbols are invalid.

## Funnel summary

### Master AI funnel

| Stage | Input | Excluded | Output | Rule and owner | Assessment |
|---|---:|---:|---:|---|---|
| Raw/master historical universe | 4,741 | 0 | 4,741 | `data_pipeline/src/csv_store.py::build_master_dataset` preserves unique symbol/date history | Expected as a broad instrument universe |
| Fatal raw quality screen | 4,741 | 3,969 | 772 | `feature_engineering/preprocessing.py::fatal_quality_errors_by_symbol`; any invalid date/OHLCV, duplicate date, inverted range, negative volume, or non-positive OHLC is fatal | **Suspiciously strict in this data: all 3,969 fail for non-positive price** |
| Master-supported security types | 772 | 239 | 533 | `feature_engineering/schemas.py::DEFAULT_MASTER_SECURITY_TYPES`; excludes `unknown` and `right` | Expected by current master schema, but historical `unknown` is coarse |
| Feature warm-up / usable output | 533 | 320 | 213 | `feature_engineering/schemas.py::FEATURE_WARMUP_ROWS` and `feature_engineering/indicators.py::calculate_features`; 49 warm-up rows, output requires at least one usable row | Expected; all 320 exclusions have ≤49 raw rows |
| Processed master | 213 | 0 | 213 | `build_master_ai_dataset`; no 252-row minimum at master scope | Expected |

### Active per-symbol readiness funnel

| Stage | Input | Excluded | Output | Rule and owner | Assessment |
|---|---:|---:|---:|---|---|
| Registry symbols | 4,741 | 4,240 | 501 | Officially listed **and** `activity_status == recently_traded`; `data_pipeline/src/company_registry.py::build_company_registry` and `data_pipeline/src/official_listings.py` | Expected scope reduction, though “recent” is a 30-calendar-day rule |
| Fatal quality among active | 501 | 364 | 137 | `feature_engineering/preprocessing.py::fatal_quality_errors_by_symbol` | **Suspicious: excludes 342 active ordinary equities** |
| Ordinary-equity-only training | 137 | 6 | 131 | `feature_engineering/dataset_builder.py::build_symbol_datasets` | Expected current policy |
| Minimum usable rows | 131 | 12 | 119 | Minimum 252; `feature_engineering/dataset_builder.py::build_symbol_datasets` | Expected threshold behavior |
| Dataset/split/scaler validation | 119 | 0 | 119 | `feature_engineering/splitting.py::chronological_split` and `persist_split_artifacts` and `feature_engineering/readiness.py::build_training_readiness_report` | Healthy; all artifacts present |

## Exact explanation of 213 → 119

| Exclusive reason within processed-master symbols | Count |
|---|---:|
| Historical/not active under current registry scope | 79 |
| Active but unsupported for per-symbol training (2 ETFs, 4 REIT/other) | 6 |
| Active ordinary equity below 252 usable rows | 9 |
| Missing processed dataset | 0 |
| Split generation failure or missing split | 0 |
| Missing scaler | 0 |
| Processed data-quality/duplicate-date failure | 0 |
| Ready for training | 119 |
| **Total processed-master symbols** | **213** |

Three additional active ordinary equities are counted as insufficient in the full readiness population but never reach the processed master because they have no post-warm-up row. Therefore the full insufficient-history count is 12, while the 213→119 bridge contains 9.

## What the 4,741 symbols represent

| Category | Count |
|---|---:|
| other | 2,479 |
| unknown | 1,705 |
| ordinary_equity | 530 |
| preference_share | 10 |
| etf | 9 |
| gem_equity | 5 |
| right | 3 |

| Category | Symbols |
|---|---:|
| historical_only | 4,178 |
| listed_recently_traded | 462 |
| non_compliant | 92 |
| listed_not_recently_traded | 7 |
| newly_listed | 2 |

- **563** are currently listed; **4,178** are historical-only.
- The 530 ordinary equities, 10 preference shares, 9 ETFs, 5 GEM equities, 3 rights, and 6 listed REIT/other instruments are identified from the official listing snapshot.
- The **2,473 historical `other` symbols** are contract-like symbols ending in month codes under the registry's futures-contract heuristic. Across the complete master, **2,473** symbols match that suffix pattern.
- The **1,705 historical `unknown` symbols** do not have a current official-listing match and do not match the limited futures suffix heuristic. They may contain delisted cash securities, old ticker identities, or other instruments, but the current registry evidence cannot safely subdivide them.
- The registry contains **0 populated previous/successor-symbol links**, so ticker/name changes are not presently quantified. It would be unsafe to count historical aliases as distinct companies without additional evidence.
- REITs are represented as `other`; rights are explicitly `right`. The master count is therefore an instrument/symbol-history universe, not a count of trainable companies.

## Current active readiness statuses

| Category | Symbols |
|---|---:|
| Data Quality Issue | 364 |
| Ready | 119 |
| Insufficient History | 12 |
| Unsupported Security Type | 6 |

## Suspicious exclusions

1. **Whole-symbol rejection for any zero price:** all 3,969 fatal symbols fail `price is not positive`. PSX zero OHLC fields can occur on no-trade rows. Rejecting an otherwise long, valid history because of one such row is likely over-broad and removes 342 currently active ordinary equities. This requires a separate policy investigation before changing or imputing anything.
2. **Historical classification is coarse:** 1,705 symbols are `unknown`, and previous/successor ticker links are empty. This is expected from current evidence but prevents reliable company-level interpretation of the 4,741 figure.
3. **No downstream artifact failures:** among symbols that pass current eligibility, there are no missing datasets, split failures, missing scalers, or duplicate-date failures. The 119 is not caused by the recently fixed merge/scaler bugs.

## Recommended first PPO universe

Recommendation: **option (b), a curated liquid subset of 20 current-ready ordinary equities**. Rank by recent median traded value, retain the current leakage-safe chronological splits, and use this as a pipeline/experiment validation cohort before scaling to all 119. Do not admit the 342 zero-price-excluded active ordinary equities until their row semantics and treatment are formally resolved.

| symbol | company_name | sector | median_recent_traded_value | recent_observations |
|---|---|---|---:|---:|
| MLCF | Maple Leaf Cement Factory Limited | CEMENT | 1,303,293,751 | 60 |
| OGDC | Oil & Gas Development Company Limited | OIL & GAS EXPLORATION COMPANIES | 1,135,508,078 | 60 |
| DGKC | D.G. Khan Cement Company Limited | CEMENT | 964,802,520 | 60 |
| UBL | United Bank Limited | COMMERCIAL BANKS | 879,460,311 | 60 |
| PPL | Pakistan Petroleum Limited | OIL & GAS EXPLORATION COMPANIES | 843,587,154 | 60 |
| NBP | National Bank of Pakistan | COMMERCIAL BANKS | 779,360,603 | 60 |
| FFC | Fauji Fertilizer Company Limited | FERTILIZER | 711,998,340 | 60 |
| PSO | Pakistan State Oil Company Limited | OIL & GAS MARKETING COMPANIES | 704,665,765 | 60 |
| LUCK | Lucky Cement Limited | CEMENT | 702,666,974 | 60 |
| BOP | The Bank of Punjab | COMMERCIAL BANKS | 695,797,618 | 60 |
| HUBC | The Hub Power Company Limited | POWER GENERATION & DISTRIBUTION | 571,205,907 | 60 |
| MEBL | Meezan Bank Limited | COMMERCIAL BANKS | 540,079,598 | 60 |
| ENGROH | Engro Holdings Limited | INV. BANKS / INV. COS. / SECURITIES COS. | 485,959,649 | 60 |
| HBL | Habib Bank Limited | COMMERCIAL BANKS | 429,374,460 | 60 |
| MARI | Mari Energies Limited | OIL & GAS EXPLORATION COMPANIES | 383,149,583 | 60 |
| ATRL | Attock Refinery Limited | REFINERY | 371,684,255 | 60 |
| PTC | Pakistan Telecommunication Company Ltd | TECHNOLOGY & COMMUNICATION | 360,834,128 | 60 |
| NRL | National Refinery Limited | REFINERY | 334,191,597 | 60 |
| FCCL | Fauji Cement Company Limited | CEMENT | 322,788,627 | 60 |
| TRG | TRG Pakistan Limited | TECHNOLOGY & COMMUNICATION | 313,211,927 | 60 |


## Per-symbol artifact

The companion CSV contains one row for every one of the 4,741 symbols, preserves symbols as strings, and includes raw/processed/split counts, registry classification, exact current exclusion reason, and final readiness.

## Audit-only assurance

This audit did not change thresholds, eligibility logic, raw/backfill data, generated datasets, splits, scalers, or model artifacts. No models were trained and no commit was created. Only this report and `training_universe_audit.csv` were generated.
