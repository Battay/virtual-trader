# Milestone 7C.5 — VALIDATION benchmark and acceptance-policy freeze

Frozen at: **2026-08-31T16:24:43Z**

Status: **VALIDATION ONLY · TEST SEALED**

## Outcome

All 16 verified recurrent models were compared with deterministic Buy-and-Hold
on each model's exact canonical `rl_partition_v1` VALIDATION membership. No
model was loaded, retrained, promoted, or modified. No TEST dataframe, TEST
return, or TEST metric was opened or accepted.

The frozen artifacts are:

- benchmark contract: `validation_buy_and_hold_v1`;
- acceptance policy: `recurrent_validation_acceptance_v1`;
- source criteria: `ppo_validation_criteria_v1`;
- model inventory count: 16;
- model inventory hash:
  `9b35bd969491e6688509efc78affc1d8a89e8e30f4885c99f84e60fd5ad34241`;
- benchmark-result hash:
  `d53379cba1a585912cf0d1dff94de19b110772939fa2f9f9a617725901105013`;
- classification-result hash:
  `bcc4ee0d9fa24c79bd0ac37451a6d6811a2a18700f539ba65b4b8f1ea9f88fb9`.

The machine-readable freeze is
`docs/config/recurrent_validation_acceptance_v1.json`.

## Benchmark methodology

For each symbol, the canonical recurrent loader is called only with
`partition="validation"`. The benchmark verifies the symbol, row count, first
and last dates, recurrent/feature/environment contracts, exact observation and
execution dates, and sealed TEST metadata before accepting a result.

The agent observes the first VALIDATION row, buys the maximum whole-share
position at the next stored VALIDATION row's open, and then holds. The open
position is marked at the last VALIDATION close; it is not forcibly sold.
Adjacent stored usable observations are used as-is—no missing bar is fabricated,
forward-filled, or interpolated. Annualization uses the existing 252-trading-day
metric implementation. Seed 42 is fixed.

The existing `single_symbol_env_v1` benchmark method explicitly includes its
configured 0.10% commission and 0.05% slippage. This milestone did not add,
remove, tune, or estimate transaction costs.

## Frozen acceptance policy

The pre-existing criteria values are reused rather than tuned from these model
results:

- minimum VALIDATION observations: 126;
- positive RL return, Sharpe, and Sortino for standalone acceptance;
- maximum RL drawdown: 30%;
- minimum return advantage over Buy-and-Hold: 0;
- minimum Sharpe advantage over Buy-and-Hold: 0;
- maximum drawdown disadvantage versus Buy-and-Hold: 2 percentage points.

The categories are explicit:

- `INVALID_VALIDATION`: invalid/missing/incompatible validation or benchmark
  evidence, TEST contamination, or an unavailable required metric.
- `INSUFFICIENT_VALIDATION_HISTORY`: otherwise valid evidence with fewer than
  126 VALIDATION observations.
- `STRONG_VALIDATION`: sufficient history and all absolute plus relative gates
  pass.
- `ACCEPTABLE_VALIDATION`: sufficient history and all standalone absolute gates
  pass, but one or more conservative relative gates do not; every miss remains
  visible in the classification reason.
- `WEAK_VALIDATION`: valid and sufficient evidence that does not meet the
  standalone acceptable rule.

This is a transparent classification, not a composite score, model promotion,
or profitability claim.

## Per-model comparison

`DD improvement = Buy-and-Hold DD - RL DD`; positive is favorable to RL.

| Symbol | Rows | RL return | B&H return | Excess | RL Sharpe | B&H Sharpe | Sharpe Δ | RL Sortino | B&H Sortino | Sortino Δ | RL DD | B&H DD | DD improvement | RL vol | B&H vol | Sufficiency | Frozen classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 786 | 246 | 37.6574% | 76.7544% | -39.0970% | 0.9739 | 1.1941 | -0.2202 | 1.7468 | 1.9171 | -0.1703 | 16.4255% | 35.3293% | 18.9038% | 42.9947% | 68.4881% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| AABS | 282 | 2.2446% | 45.9088% | -43.6642% | 0.2090 | 1.1806 | -0.9716 | 0.2892 | 1.8885 | -1.5992 | 8.5817% | 21.2134% | 12.6317% | 14.8017% | 33.3867% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| AATM | 73 | 6.3066% | 5.4780% | 0.8286% | 0.6542 | 0.6073 | 0.0469 | 1.0538 | 0.9716 | 0.0822 | 26.1018% | 28.5743% | 2.4726% | 57.7561% | 63.2594% | INSUFFICIENT | INSUFFICIENT_VALIDATION_HISTORY |
| ABL | 355 | -5.9410% | 99.6839% | -105.6249% | -0.1936 | 1.8208 | -2.0144 | -0.2795 | 3.0186 | -3.2981 | 19.7754% | 15.7369% | -4.0385% | 15.9752% | 29.4144% | SUFFICIENT | WEAK_VALIDATION |
| ENGROH | 362 | 13.6462% | 102.4827% | -88.8365% | 0.7865 | 1.4060 | -0.6195 | 1.4410 | 2.2353 | -0.7943 | 11.5839% | 27.9859% | 16.4020% | 12.3024% | 41.0301% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| FFC | 365 | 20.5259% | 308.2101% | -287.6842% | 1.1202 | 3.0364 | -1.9162 | 1.9915 | 5.7439 | -3.7524 | 5.8574% | 12.8037% | 6.9463% | 12.1978% | 34.0102% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| HUBC | 365 | -4.9940% | 52.6114% | -57.6053% | -0.0472 | 1.0061 | -1.0533 | -0.0617 | 1.4971 | -1.5587 | 33.6300% | 41.5915% | 7.9614% | 22.2265% | 35.2982% | SUFFICIENT | WEAK_VALIDATION |
| LUCK | 364 | 18.1231% | 96.1765% | -78.0534% | 1.4795 | 1.7735 | -0.2940 | 3.3438 | 3.0054 | 0.3385 | 7.9317% | 15.3792% | 7.4475% | 8.0318% | 28.7050% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| MARI | 365 | 31.4081% | -64.6640% | 96.0720% | 0.7233 | 0.2880 | 0.4353 | 1.1210 | 0.3168 | 0.8041 | 42.1434% | 88.9808% | 46.8374% | 34.2216% | 85.6787% | SUFFICIENT | WEAK_VALIDATION |
| MCB | 365 | 103.9815% | 97.5058% | 6.4757% | 1.8762 | 1.6974 | 0.1788 | 3.0176 | 2.7039 | 0.3138 | 12.4605% | 13.9824% | 1.5219% | 28.4819% | 30.5134% | SUFFICIENT | STRONG_VALIDATION |
| OGDC | 365 | 44.8296% | 107.7651% | -62.9355% | 1.4363 | 1.4662 | -0.0300 | 2.9835 | 2.4219 | 0.5616 | 11.1289% | 30.6222% | 19.4933% | 19.1036% | 39.9610% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| PSO | 365 | 16.2717% | 193.2790% | -177.0073% | 0.9764 | 2.0219 | -1.0455 | 1.5984 | 3.4707 | -1.8723 | 9.9436% | 35.0056% | 25.0620% | 11.3490% | 41.0026% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| SYS | 363 | 55.4070% | 26.6834% | 28.7236% | 1.2683 | 0.6781 | 0.5901 | 2.2789 | 1.1612 | 1.1177 | 21.4721% | 21.4700% | -0.0021% | 27.0533% | 31.4741% | SUFFICIENT | STRONG_VALIDATION |
| TRG | 365 | 20.3183% | -28.5856% | 48.9040% | 0.5878 | -0.2477 | 0.8355 | 0.9309 | -0.3734 | 1.3043 | 19.9145% | 54.5506% | 34.6361% | 28.7399% | 47.9887% | SUFFICIENT | STRONG_VALIDATION |
| UBL | 365 | 132.9787% | 180.1914% | -47.2127% | 2.7958 | 2.8100 | -0.0142 | 5.4526 | 4.9744 | 0.4781 | 11.0014% | 11.0113% | 0.0099% | 21.8070% | 26.6754% | SUFFICIENT | ACCEPTABLE_VALIDATION |
| UNITY | 319 | 63.3887% | 8.3354% | 55.0534% | 2.2095 | 0.3539 | 1.8556 | 6.3270 | 0.5886 | 5.7384 | 7.2539% | 35.0843% | 27.8304% | 18.3576% | 36.4975% | SUFFICIENT | STRONG_VALIDATION |

## Aggregate findings

- `STRONG_VALIDATION`: 4
- `ACCEPTABLE_VALIDATION`: 8
- `WEAK_VALIDATION`: 3
- `INSUFFICIENT_VALIDATION_HISTORY`: 1
- `INVALID_VALIDATION`: 0
- models outperforming Buy-and-Hold on total return: 6/16
- models improving on Buy-and-Hold Sharpe: 6/16
- median excess return: -45.4384%
- median Sharpe delta: -0.125075

AATM remains visible with 73 observations and is explicitly insufficient. It
is not excluded or relabelled based on its otherwise favorable point metrics.

## Leakage and integrity proof

- The benchmark module has no training, `.learn()`, promotion, or registry path.
- Its canonical loader call is fixed to `validation`; a non-VALIDATION return
  fails closed.
- Exact VALIDATION dates, rows, contracts, execution chronology, and source hash
  are verified per model.
- Inventory columns or flags reporting TEST evidence are rejected.
- The policy artifact records `test_status_at_freeze=SEALED` and
  `test_observations_accessed=false`.
- Source recurrent artifacts were read only; no model or partition artifact was
  changed.

## Interpretation

The models collectively show positive standalone validation returns in 14 of
16 cases, but relative evidence is mixed: only 6 beat Buy-and-Hold on return and
6 improve Sharpe. This policy freeze prevents later TEST results from changing
the predeclared validation classification rules. It does not authorize TEST
evaluation, retraining, or model promotion.
