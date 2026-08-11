# Dashboard correctness repair before Milestone 5B-6

Audit date: 2026-08-10 (Asia/Karachi)

## Objective and safety boundary

This repair was limited to Market Overview index-period analytics, Market Indices
period analytics and charts, and Training & Models PPO-readiness reconciliation.
It did not train a PPO model, evaluate the sealed TEST partition, change trading
environment semantics, make a live HTTP request, modify raw/backfill data, write a
model-registry record, or promote a model.

## Root-cause audit

### Market Overview

The selected-period frame was already passed into the lower per-index health
panel, so this was not a stale Streamlit object or cache-key defect. The legacy
health method only consumed the latest 1, 5, 20, and 50 observations, plus a
trailing 20-observation volatility and volume average. Once a selected window
contained at least 50 observations, adding earlier observations could not change
any component. Selected-period return, full-window volatility, and drawdown were
not score inputs. As a result, the real KSE-30 score was 62.4 for 3M, 6M, 1Y,
and Maximum even though those windows had materially different returns,
volatility, and drawdowns.

The separate upper “Index Health Overview” was also calculated from full history
before the period selector. That visually reinforced the impression that the
selector was disconnected. It has been removed from the period-dependent area;
overall market health remains a separate, intentionally non-period-sensitive
indicator.

The old combined filter also anchored every index to the newest date found in the
combined dataset. A lagging index could therefore receive a shortened window.
Canonical filtering now selects an index first and anchors that window to that
index's own latest observation.

### Market Indices

This page had a direct full-history-input bug. It created a period-filtered
`shown` frame for its chart, table, and download, but passed the full combined
index dataset to `calculate_index_metrics`. Its metric cards and health score
therefore remained full-history values when the period changed. The page also
maintained its own date-offset mapping, included a noncanonical YTD option, and
reported the full stored range beside a filtered chart.

The page now creates one canonical `IndexPeriodAnalysis` for the selected
`(index, period)` and uses that same object for its contract, health score,
breakdown, summary, charts, data table, and download.

### Training & Models PPO readiness

The zero count was not caused by missing files, an RL schema mismatch, a feature
version mismatch, symbol normalization, or PPO training state. The readiness row
combined two different snapshots:

- `first_usable_date` and `last_usable_date` described current live/master data,
  which had advanced as far as 2026-08-07.
- train/validation/test row counts described the persisted processed dataset and
  RL contracts, which ended on 2026-08-05.

`build_ready_symbol_catalog` compared the current-source dates with the persisted
RL contract bounds, so all 454 otherwise-valid symbols were rejected. In the real
data, 449 symbols differed by two calendar days, four by one day, and one by three
days. Every processed dataset's bounds matched its RL contract.

Readiness now exposes distinct `processed_first_date` and
`processed_last_date` fields and compares RL contracts to those processed bounds.
The current-source dates remain available as freshness information and are not
discarded. Compatibility failures are classified as missing, stale, incompatible
feature version, incompatible contract version, incompatible environment version,
or other failure.

## One canonical period contract

Both index pages now use `index_period_v1` with exactly these options:

- 1M: selected index's latest date minus one calendar month, inclusive
- 3M: selected index's latest date minus three calendar months, inclusive
- 6M: selected index's latest date minus six calendar months, inclusive
- 1Y: selected index's latest date minus one calendar year, inclusive
- Maximum: every locally available observation for that index

Every analysis exposes the requested period, actual first and last dates,
observation count, start value, and end value. Period calculations operate on a
deep copy and do not mutate source data. Each index is filtered independently.

Daily return, daily change, MA20, MA50, drawdown, and rolling volatility are
calculated after filtering. No observations before the visible period are used as
warm-up data. MA20/MA50 require 20/50 visible observations; 20-observation rolling
volatility uses `ddof=0` and annualizes by `sqrt(252)`. Drawdown is
`value / cumulative_max(value) - 1`.

## Index health methodology

The period-sensitive score is versioned as `index_period_health_v1` and is
separate from the established overall-market-health formula. This is a justified
replacement for the selected-index panel because the legacy trailing-only score
could not represent the selected horizon.

| Component | Weight | Selected-period input and bounded factor |
|---|---:|---|
| Selected-period return | 20 | Centred at 50%; reaches 0/100% factor at -20%/+20% return |
| Trend consistency | 15 | Percentage of positive visible sessions; flat sessions receive half credit |
| Period momentum | 10 | 21-session-equivalent difference between mean second-half and first-half returns, centred at 50% and bounded at +/-10% |
| Selected-period volatility resilience | 15 | `clamp(1 - annualized_volatility / 50)` |
| Maximum-drawdown resilience | 15 | `clamp(1 - abs(max_drawdown) / 40)` |
| Value relative to SMA20 | 10 | Latest value versus causal visible-period SMA20, centred and bounded at +/-10% |
| Value relative to SMA50 | 10 | Latest value versus causal visible-period SMA50, centred and bounded at +/-20% |
| Volume participation | 5 | Latest volume divided by visible-period average; full factor at 1.5x |

Each available factor is clamped to `[0, 1]`. Raw points equal
`factor * configured weight`. Unavailable components are not assigned zero;
available raw points are normalized with:

`health = clamp(sum(raw points) / available configured weight, 0, 1) * 100`

The result is therefore bounded from 0 to 100. The UI shows each input, unit,
configured weight, factor, raw points, normalized contribution, unavailable
components, and coverage. The displayed normalized contributions reproduce the
score before final one-decimal rounding. A typical 1M window lacks SMA50 and has
90% configured coverage; longer current windows have 100% coverage.

The pre-existing overall market health remains unchanged and independent of the
index-period selector.

## Chart and presentation corrections

For a selected index, both pages now show:

- index level, causal MA20, and causal MA50 on one level chart;
- hover values for date, level, daily change, daily-change percentage, MA20,
  and MA50;
- selected-period return, high, low, latest level, full-window annualized
  volatility, maximum drawdown, health, dates, observations, and start/end values;
- a separate drawdown chart; and
- a separate causal rolling-20-observation volatility chart.

Fundamentally different scales are not combined. Source CSV caches include path,
modification time, and byte size. PPO readiness caches include processed/readiness
artifact identity plus the feature, RL-contract, and environment versions.

## Real local index verification

All values below were computed from the same local official PSX index master used
by the pages. Returns, volatility, and drawdown are percentages.

| Index | Period | Start | End | Rows | Return | Volatility | Max Drawdown | Health |
|---|---|---|---|---:|---:|---:|---:|---:|
| KSE100 | 1M | 2026-07-06 | 2026-08-05 | 23 | -3.97% | 24.64% | -8.77% | 58.2 |
| KSE100 | 3M | 2026-05-05 | 2026-08-05 | 62 | +9.27% | 22.61% | -8.77% | 59.0 |
| KSE100 | 6M | 2026-02-06 | 2026-08-05 | 121 | -2.23% | 35.78% | -20.45% | 48.5 |
| KSE100 | 1Y | 2025-08-05 | 2026-08-05 | 251 | +25.85% | 27.41% | -22.57% | 58.0 |
| KSE100 | Maximum | 2021-08-06 | 2026-08-05 | 1,239 | +279.06% | 20.25% | -22.57% | 66.0 |
| KSE30 | 1M | 2026-07-06 | 2026-08-05 | 23 | -4.11% | 26.52% | -9.23% | 58.7 |
| KSE30 | 3M | 2026-05-05 | 2026-08-05 | 62 | +8.75% | 24.38% | -9.23% | 59.0 |
| KSE30 | 6M | 2026-02-06 | 2026-08-05 | 121 | -4.37% | 37.61% | -21.05% | 47.3 |
| KSE30 | 1Y | 2025-08-05 | 2026-08-05 | 251 | +22.33% | 29.02% | -23.67% | 57.4 |
| KSE30 | Maximum | 2021-08-06 | 2026-08-05 | 1,239 | +183.79% | 21.82% | -27.19% | 64.4 |
| KMI30 | 1M | 2026-07-06 | 2026-08-05 | 23 | -4.57% | 25.99% | -9.43% | 57.5 |
| KMI30 | 3M | 2026-05-05 | 2026-08-05 | 62 | +7.19% | 23.49% | -9.43% | 57.1 |
| KMI30 | 6M | 2026-02-06 | 2026-08-05 | 121 | -2.45% | 37.28% | -19.19% | 47.7 |
| KMI30 | 1Y | 2025-08-05 | 2026-08-05 | 251 | +24.51% | 29.20% | -22.06% | 57.8 |
| KMI30 | Maximum | 2021-08-06 | 2026-08-05 | 1,239 | +229.97% | 23.18% | -22.06% | 64.2 |
| ALLSHR | 1M | 2026-07-06 | 2026-08-05 | 23 | -4.26% | 22.59% | -8.27% | 57.9 |
| ALLSHR | 3M | 2026-05-05 | 2026-08-05 | 62 | +9.92% | 20.48% | -8.27% | 59.2 |
| ALLSHR | 6M | 2026-02-06 | 2026-08-05 | 121 | -2.16% | 32.99% | -20.58% | 49.5 |
| ALLSHR | 1Y | 2025-08-05 | 2026-08-05 | 251 | +22.56% | 25.07% | -22.34% | 58.9 |
| ALLSHR | Maximum | 2021-08-06 | 2026-08-05 | 1,239 | +233.00% | 18.43% | -22.34% | 66.6 |

Similar scores can still occur when their independently calculated weighted
factors offset one another. They are no longer identical because a full-history
or trailing-only object was reused.

## PPO readiness reconciliation

| Measure | Actual count |
|---|---:|
| Current eligible/Ready symbols | 454 |
| Compatible `rl_partition_v1` symbols among the eligible set | 454 |
| Eligibility/compatibility intersection | 454 |
| Missing contracts/artifacts | 0 |
| Stale contracts | 0 |
| Incompatible feature versions | 0 |
| Incompatible contract versions | 0 |
| Incompatible environment versions | 0 |
| Other compatibility failures | 0 |
| Insufficient-history symbols | 19 |
| Registered models | 0 |

All 454 compatible symbols were checked through the metadata-only RL contract
loader for the contract, TRAIN/VALIDATION/TEST artifact existence, scaler and
scaler-metadata compatibility, versions, symbol, dates, counts, and chronology.
The TEST CSV was not loaded. All ten requested pilot symbols are Ready: OGDC,
UBL, FFC, PPL, MEBL, LUCK, HUBC, PSO, MLCF, and TRG.

## Regression and Streamlit smoke verification

The complete offline suite completed with **436 passed, 1 skipped**. The skip is
the existing hardware-gated MPS test. `git diff --check` passed and
`.venv/bin/python -m pip check` reported no broken requirements.

In-process Streamlit page smokes (not browser automation) produced zero
exceptions:

- Market Overview, KSE-30: 6M = 47.3 health / 121 observations / -4.37%;
  3M = 59.0 / 62 / +8.75%; 1Y = 57.4 / 251 / +22.33%.
- Market Indices: KSE-100 6M = 48.5 / 121 / -2.23%; KSE-100 1M = 58.2 /
  23 / -3.97%; KSE-30 1M = 58.7 / 23 / -4.11%.
- Training & Models reported `single_symbol_env_v1`, `rl_partition_v1`,
  `ppo_single_symbol_v1`, 454 training-ready, 19 insufficient-history, and zero
  registered models. The selector contained 454 symbols (default OGDC); Train
  was visible and unclicked; validation remained disabled without a candidate.

Manual visual verification can be repeated by opening each index page, selecting
one index, and changing 1M through Maximum while comparing the displayed contract,
summary, health breakdown, and chart date extent. On Training & Models, confirm
the top count is 454, inspect the pilot table, and leave Train unclicked.

## Integrity and remaining limitations

The production registry SHA-256 remained
`e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a`.
The two production saved-model roots still contain only their original empty
`.gitkeep` files. No production model or registry record was created.

The score is a transparent descriptive heuristic, not a fitted statistical model
or investment recommendation. Calendar-month windows have different trading-row
counts by design. Short periods honestly omit long-window components. The two
pages retain five-minute data caches, but their keys now include local source or
artifact identity so ordinary file updates invalidate them. No HTTP refresh
button was clicked during verification.

