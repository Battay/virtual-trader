# Non-positive OHLC quality audit

Audit timestamp: **2026-08-08T03:03:27+05:00**  
Scope: local master dataset and registry only; active, officially listed, recently traded ordinary equities.  
Configured minimum usable history: **252 rows**.

## Executive conclusion

The current whole-symbol fatal exclusion is **too strict for this local PSX history**. It excludes an entire equity for one non-positive OHLC field on one date. Of **473** active ordinary equities, **342** are affected, covering **23,073** rows. **9** symbols have exclusively evidence-compatible zero-volume/no-trade-style invalid rows, while **0** symbols contain a genuinely negative price and **5** contain a non-positive close. The audit supports row-level, explicitly classified handling with strict rejection of negative prices, zero closes, and unexplained positive-volume partial zeros; it does not support price fabrication or blanket acceptance.

## Scope and methodology

- Active ordinary means `officially_listed == true`, `activity_status == recently_traded`, and `security_type == ordinary_equity` from `company_registry.csv`.
- An affected row has at least one of open/high/low/close `<= 0`, exactly matching the non-positive portion of `fatal_quality_errors_by_symbol`.
- All simulations use the existing causal `calculate_features`, its warm-up/missing-feature logic, and the configured 252-row minimum. No future observations are used.
- Policy row counts below mean rows remaining in quality-retained symbols after that policy's row removal. Rows in rejected symbols therefore count as removed from that candidate universe.
- No-trade-style means: close > 0; at least one of open/high/low <= 0; volume == 0; and no price is negative. This is an evidence-based pattern label, not proof of exchange status.
- “Surrounding normal” means the immediately preceding or following symbol observation has all OHLC > 0.

## Headline counts

| Measure | Count |
|---|---:|
| Active ordinary equities | 473 |
| Affected active ordinary equities | 342 |
| Unaffected active ordinary equities | 131 |
| Active-ordinary raw rows | 842,901 |
| Non-positive-OHLC rows | 23,073 |
| Symbols with ≤4 affected rows (isolated definition) | 52 |
| Symbols with a non-positive close | 5 |
| Symbols with a negative price | 0 |
| Symbols with at least one likely no-trade-style row | 283 |
| Symbols whose every affected row is likely no-trade-style | 9 |

## Pattern classification

Primary classes are mutually exclusive, applied in this order: negative price; zero close; affected share ≥5%; ≤4 isolated rows; exclusively no-trade-style rows; remaining partial-zero patterns.

| Primary pattern | Symbols |
|---|---|
| HIGH_FREQUENCY_QUALITY_PROBLEM | 119 |
| ISOLATED_BAD_ROWS | 52 |
| NO_TRADE_STYLE_ROW | 2 |
| PARTIAL_ZERO_OHLC | 164 |
| ZERO_CLOSE | 5 |

Row-level patterns overlap only where explicitly meaningful (for example, a partial O/H/L-zero row can also satisfy the no-trade evidence rule). Full per-symbol counts are in the CSV artifact.

## Policy simulations

| policy | quality_retained | quality_excluded | rows_retained | rows_removed | training_eligible | newly_eligible_vs_current | still_insufficient |
|---|---:|---:|---:|---:|---:|---:|---:|
| Policy 0 — current whole-symbol rejection | 131 | 342 | 259,357 | 583,544 | 119 | 0 | 12 |
| Policy 1 — filter non-positive rows | 473 | 0 | 819,828 | 23,073 | 454 | 335 | 19 |
| Policy 2 — remove clear zero-volume/no-trade rows; reject other invalid patterns | 140 | 333 | 270,940 | 571,961 | 128 | 9 | 12 |
| Policy 3 — invalid-row tolerance ≤0.1% | 162 | 311 | 334,700 | 508,201 | 150 | 31 | 12 |
| Policy 3 — invalid-row tolerance ≤0.5% | 209 | 264 | 434,276 | 408,625 | 197 | 78 | 12 |
| Policy 3 — invalid-row tolerance ≤1% | 246 | 227 | 512,107 | 330,794 | 234 | 115 | 12 |
| Policy 3 — invalid-row tolerance ≤2% | 277 | 196 | 571,009 | 271,892 | 265 | 146 | 12 |
| Policy 3 — invalid-row tolerance ≤5% | 351 | 122 | 696,583 | 146,318 | 338 | 219 | 13 |

“Training eligible” is the number with at least 252 exact usable feature rows after the stated handling. “Still insufficient” is within quality-retained symbols; it does not include quality-excluded symbols. Policy 1 is a diagnostic upper bound: it filters every non-positive row without claiming all such rows are benign. Policy 2 is deliberately conservative and performs no imputation. Threshold policies filter affected rows only for symbols at or below the stated tolerance and reject the remainder.

## Concrete examples

| sample | symbol | date | ldcp | open | high | low | close | volume |
|---|---|---|---|---|---|---|---|---|
| exactly one bad row | GGL | 2019-07-25 | 7.3 | 0.0 | 7.35 | 7.35 | 7.3 | 2000 |
| fewer than five bad rows | GGGL | 2018-12-19 | 9.1 | 0.0 | 9.1 | 9.1 | 9.1 | 3000 |
| fewer than five bad rows | GGGL | 2019-09-12 | 4.75 | 0.0 | 5.01 | 5.01 | 4.75 | 1000 |
| 1–2% affected | RICL | 2017-06-19 | 9.28 | 0.0 | 9.28 | 9.28 | 9.28 | 187000 |
| 1–2% affected | RICL | 2018-03-08 | 8.0 | 0.0 | 8.0 | 8.0 | 8.0 | 500 |
| many bad rows | ORM | 2018-01-02 | 12.05 | 0.0 | 12.05 | 12.05 | 12.05 | 500 |
| many bad rows | ORM | 2018-01-30 | 12.75 | 0.0 | 12.75 | 12.75 | 12.75 | 2000 |
| zero close | PCAL | 2016-11-11 | 279.86 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| zero close | PCAL | 2017-06-02 | 349.68 | 0.0 | 349.68 | 349.68 | 349.68 | 1500 |
| only O/H/L zero with valid close | GGL | 2019-07-25 | 7.3 | 0.0 | 7.35 | 7.35 | 7.3 | 2000 |
| liquid/major by median turnover | UNITY | 2017-10-11 | 13.99 | 0.0 | 14.7666 | 13.99 | 14.7666 | 0 |
| liquid/major by median turnover | UNITY | 2017-10-12 | 14.7666 | 0.0 | 15.7459 | 0.0 | 15.7459 | 0 |
| liquid/major by median turnover | TPLP | 2018-01-16 | 10.0 | 0.0 | 10.0 | 10.0 | 10.0 | 800000 |
| liquid/major by median turnover | TPLP | 2018-01-17 | 10.0 | 0.0 | 10.0 | 10.0 | 10.0 | 800000 |
| liquid/major by median turnover | SAZEW | 2019-05-20 | 226.69 | 0.0 | 226.2158 | 226.2158 | 226.2158 | 0 |
| liquid/major by median turnover | SAZEW | 2019-06-10 | 237.66 | 0.0 | 235.7 | 235.7 | 235.7 | 0 |
| liquid/major by median turnover | GGL | 2019-07-25 | 7.3 | 0.0 | 7.35 | 7.35 | 7.3 | 2000 |
| liquid/major by median turnover | HALEON | 2023-08-21 | 160.21 | 0.0 | 0.0 | 0.0 | 160.21 | 500 |

The “liquid/major” examples are selected reproducibly by median local `close × volume`, not by a subjective company list.

### Surrounding-observation examples

| Symbol | Previous observation | Affected observation | Next observation | Interpretation |
|---|---|---|---|---|
| GGL | 2019-07-24: O/H/L/C 7.25/7.30/7.25/7.30, volume 12,000 | 2019-07-25: 0/7.35/7.35/7.30, volume 2,000 | 2019-07-26: 7.10/7.20/7.10/7.10, volume 15,000 | Isolated zero-open row between normal observations; positive volume means it does **not** meet the narrow no-trade-style rule. |
| PCAL | 2016-11-10: 279.90/282/277/279.86, volume 14,900 | 2016-11-11: 0/0/0/0, volume 0 | 2016-11-15: 281/281/275/275, volume 1,000 | All-zero OHLC and volume between normal observations; close is invalid, so conservative handling must not treat it as a usable price. |
| UNITY | 2017-10-10: 195.20/195.24/176.66/176.70, volume 162,000 | 2017-10-11: 0/14.7666/13.99/14.7666, volume 0 | 2017-10-12: 0/15.7459/0/15.7459, volume 0 | Adjacent discontinuity and repeated zeros suggest a more complex event (potential adjustment/corporate-action pattern), not evidence for blind imputation. |
| ORM | 2017-12-28: 12/12.05/12/12.05, volume 2,500 | 2018-01-02: 0/12.05/12.05/12.05, volume 500 | 2018-01-03: 12.25/12.50/12.25/12.50, volume 65,000 | Zero-open row between normal observations; positive volume excludes it from the narrow no-trade-style rule. |

## Source-versus-pipeline checks

| Symbol | Date | Raw daily CSV | Raw equals master OHLCV | Raw values | Processed representation |
|---|---|---|---|---|---|
| GGL | 2019-07-25 | `market_2019-07-25.csv` | Yes | ldcp=7.3, open=0.0, high=7.35, low=7.35, close=7.3, volume=2000 | Absent: whole symbol is excluded before feature generation |
| GGGL | 2018-12-19 | `market_2018-12-19.csv` | Yes | ldcp=9.1, open=0.0, high=9.1, low=9.1, close=9.1, volume=3000 | Absent: whole symbol is excluded before feature generation |
| GGGL | 2019-09-12 | `market_2019-09-12.csv` | Yes | ldcp=4.75, open=0.0, high=5.01, low=5.01, close=4.75, volume=1000 | Absent: whole symbol is excluded before feature generation |
| RICL | 2017-06-19 | `market_2017-06-19.csv` | Yes | ldcp=9.28, open=0.0, high=9.28, low=9.28, close=9.28, volume=187000 | Absent: whole symbol is excluded before feature generation |
| RICL | 2018-03-08 | `market_2018-03-08.csv` | Yes | ldcp=8.0, open=0.0, high=8.0, low=8.0, close=8.0, volume=500 | Absent: whole symbol is excluded before feature generation |
| ORM | 2018-01-02 | `market_2018-01-02.csv` | Yes | ldcp=12.05, open=0.0, high=12.05, low=12.05, close=12.05, volume=500 | Absent: whole symbol is excluded before feature generation |
| ORM | 2018-01-30 | `market_2018-01-30.csv` | Yes | ldcp=12.75, open=0.0, high=12.75, low=12.75, close=12.75, volume=2000 | Absent: whole symbol is excluded before feature generation |
| PCAL | 2016-11-11 | `market_2016-11-11.csv` | Yes | ldcp=279.86, open=0.0, high=0.0, low=0.0, close=0.0, volume=0 | Absent: whole symbol is excluded before feature generation |
| PCAL | 2017-06-02 | `market_2017-06-02.csv` | Yes | ldcp=349.68, open=0.0, high=349.68, low=349.68, close=349.68, volume=1500 | Absent: whole symbol is excluded before feature generation |
| GGL | 2019-07-25 | `market_2019-07-25.csv` | Yes | ldcp=7.3, open=0.0, high=7.35, low=7.35, close=7.3, volume=2000 | Absent: whole symbol is excluded before feature generation |


Preserved official-response HTML was also checked directly for three distinct patterns:

| Symbol | Date | Preserved HTML bytes | HTML `data-value` sequence (symbol, LDCP, O, H, L, C, change, %, volume) | HTML = CSV = master |
|---|---|---:|---|---|
| GGL | 2019-07-25 | 398,359 | `GGL, 7.3, 0, 7.35, 7.35, 7.3, 0, 0, 2000` | Yes |
| PCAL | 2016-11-11 | 359,009 | `PCAL, 279.86, 0, 0, 0, 0, -279.86, -100, 0` | Yes |
| UNITY | 2017-10-11 | 280,321 | `UNITY, 13.99, 0, 14.7666, 13.99, 14.7666, 0.7766…, 5.5511…, 0` | Yes |


Every sampled value was already present in the parsed daily CSV and matched the master OHLCV values exactly. The three direct HTML checks also match exactly, including an isolated zero open, an all-zero OHLC row, and a zero-volume partial-zero row. The affected symbols are absent from the processed representation because the fatal screen runs before feature calculation; there is no processed transformation that could have introduced these zeros. The audited zeros therefore originate in preserved official PSX responses, not master consolidation or feature preprocessing. No network request was made.

## Recommended future policy (not implemented)

1. Preserve raw data unchanged and attach explicit row-quality flags.
2. Reject negative OHLC, non-positive close, inverted ranges, negative volume, and unexplained positive-volume partial-zero records as genuinely invalid pending review.
3. For the narrow zero-volume/no-trade-style pattern, exclude the row from return/indicator generation rather than inventing OHLC values. Do not forward-fill/back-fill prices across dates.
4. Evaluate eligibility after row-level filtering, retaining symbols with sufficient exact usable history. Use a documented tolerance only as a secondary symbol-level safeguard, with per-symbol exclusion reasons.
5. Version the policy, record removed dates and reasons, fit scalers on training partitions only, and retain cardinality/date-order checks for reproducibility and leakage prevention.
6. Extend the preserved-HTML sample before implementation if a policy depends on finer subtypes. This audit's direct HTML checks already distinguish official-response values from parser/master/preprocessing corruption for representative patterns.

The current one-row-fails-the-whole-symbol policy is not proportionate to the observed evidence. A future implementation should remain conservative: row filtering is preferable to price imputation, while ambiguous or genuinely invalid patterns remain excluded and visible for audit.

## Artifact notes

`zero_ohlc_quality_audit.csv` contains one row per affected active ordinary symbol, quoted symbol identifiers, exact field counts, affected volume summary, adjacency evidence, and primary pattern. This audit created reports only. It did not modify raw HTML/CSV, rejected data, master/registry, processed datasets, backfill state, splits/scalers, or model files; it made no live HTTP requests and trained no model.
