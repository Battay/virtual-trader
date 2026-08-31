# Milestone 7C.4 — Global Recurrent VALIDATION Comparison

## Scope and outcome

This comparison is read-only and uses the persisted VALIDATION artifacts for
all 16 globally verified recurrent models. No model was loaded, retrained,
reevaluated, promoted, or modified. TEST remains sealed and no TEST observation,
return, or dataframe was opened.

All 16 persisted validation artifacts passed provenance checks. Fourteen models
have positive validation return. The median validation return is 20.42%, median
Sharpe is 0.975, and median maximum drawdown is 12.02%.

The artifacts do not contain Buy-and-Hold metrics, turnover, average reward, or
candidate acceptance decisions. Those fields remain explicitly unavailable;
they were not reconstructed or fabricated. Consequently, excess-return ranks
and positive-excess-return counts are unavailable.

## Methodology

- Source: persisted run-isolated validation JSON and training-diagnostic JSON.
- Model population: latest verified model per trained symbol across all valid
  full-production and selected-run history.
- Required provenance: symbol, validation partition, model parameters unchanged,
  unchanged model timesteps, compatible feature/environment/recurrent contract,
  and matching persisted model-observed VALIDATION dates and rows.
- Default display order: descending validation Sharpe, then symbol.
- Transparent ranks: return and Sharpe/Sortino descending; maximum drawdown
  ascending; excess return descending when a persisted benchmark exists.
- Aggregates include only valid persisted validation artifacts.
- Each model retains its own `rl_partition_v1` model-observed validation window;
  comparisons are descriptive and are not a claim of statistical significance.

## Validation inventory

| Symbol | Run type | Rows | Model-observed VALIDATION range | Return | Sharpe | Sortino | Max drawdown | Trades | Return rank | Sharpe rank |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| UBL | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 132.98% | 2.796 | 5.453 | 11.00% | 10 | 1 | 1 |
| UNITY | SELECTED | 319 | 2024-01-03 to 2025-04-22 | 63.39% | 2.210 | 6.327 | 7.25% | 10 | 3 | 2 |
| MCB | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 103.98% | 1.876 | 3.018 | 12.46% | 9 | 2 | 3 |
| LUCK | SELECTED | 364 | 2023-08-21 to 2025-02-06 | 18.12% | 1.480 | 3.344 | 7.93% | 4 | 10 | 4 |
| OGDC | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 44.83% | 1.436 | 2.983 | 11.13% | 14 | 5 | 5 |
| SYS | SELECTED | 363 | 2023-08-28 to 2025-02-12 | 55.41% | 1.268 | 2.279 | 21.47% | 25 | 4 | 6 |
| FFC | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 20.53% | 1.120 | 1.992 | 5.86% | 16 | 8 | 7 |
| PSO | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 16.27% | 0.976 | 1.598 | 9.94% | 12 | 11 | 8 |
| 786 | FULL_PRODUCTION | 246 | 2024-08-08 to 2025-08-07 | 37.66% | 0.974 | 1.747 | 16.43% | 34 | 6 | 9 |
| ENGROH | SELECTED | 362 | 2023-09-01 to 2025-02-17 | 13.65% | 0.787 | 1.441 | 11.58% | 8 | 12 | 10 |
| MARI | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 31.41% | 0.723 | 1.121 | 42.14% | 5 | 7 | 11 |
| AATM | FULL_PRODUCTION | 73 | 2025-12-30 to 2026-04-14 | 6.31% | 0.654 | 1.054 | 26.10% | 15 | 13 | 12 |
| TRG | SELECTED | 365 | 2023-08-24 to 2025-02-12 | 20.32% | 0.588 | 0.931 | 19.91% | 48 | 9 | 13 |
| AABS | FULL_PRODUCTION | 282 | 2024-04-23 to 2025-06-17 | 2.24% | 0.209 | 0.289 | 8.58% | 8 | 14 | 14 |
| HUBC | SELECTED | 365 | 2023-08-24 to 2025-02-12 | -4.99% | -0.047 | -0.062 | 33.63% | 17 | 15 | 15 |
| ABL | FULL_PRODUCTION | 355 | 2023-09-06 to 2025-02-25 | -5.94% | -0.194 | -0.280 | 19.78% | 14 | 16 | 16 |

## Sector summary

| Sector | Models | Median return | Median Sharpe | Median max drawdown | Positive returns |
|---|---:|---:|---:|---:|---:|
| Commercial Banks | 3 | 103.98% | 1.876 | 12.46% | 2 |
| Investment Banks / Companies / Securities Companies | 2 | 25.65% | 0.880 | 14.00% | 2 |
| Oil & Gas Exploration Companies | 2 | 38.12% | 1.080 | 26.64% | 2 |
| Technology & Communication | 2 | 37.86% | 0.928 | 20.69% | 2 |
| Cement | 1 | 18.12% | 1.480 | 7.93% | 1 |
| Fertilizer | 1 | 20.53% | 1.120 | 5.86% | 1 |
| Food & Personal Care Products | 1 | 63.39% | 2.210 | 7.25% | 1 |
| Oil & Gas Marketing Companies | 1 | 16.27% | 0.976 | 9.94% | 1 |
| Power Generation & Distribution | 1 | -4.99% | -0.047 | 33.63% | 0 |
| Sugar & Allied Industries | 1 | 2.24% | 0.209 | 8.58% | 1 |
| Textile Spinning | 1 | 6.31% | 0.654 | 26.10% | 1 |

## Comparability warnings

- AATM has 73 validation rows, below the existing 126-observation validation
  sufficiency criterion, and is explicitly marked as short-history.
- AATM and 786 have validation row counts materially below the 365-row median.
- All models lack persisted benchmark return, turnover, average reward, and
  acceptance-decision fields. The comparison does not infer them.
- Model-specific validation dates differ because `rl_partition_v1` is applied
  independently to each symbol. Rankings are descriptive validation summaries,
  not evidence that windows are identical or that one model is universally best.

## Export and leakage guard

The Training & Models page provides a deterministic CSV export containing
model/run provenance, partition metadata, persisted metrics, transparent ranks,
artifact hashes, warnings, and `TEST_status=SEALED`. It excludes persisted daily
return arrays and contains no TEST values. Optional detail charts derive only
from the selected artifact's already persisted VALIDATION daily-return list.
