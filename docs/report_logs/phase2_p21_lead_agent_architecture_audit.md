# Phase 2 P2.1 — Lead Agent Data, Objective & Architecture Audit

Audit date: 2026-09-01

Decision artifact: `docs/config/phase2_lead_agent_architecture_audit_v1.json`

Artifact version: `phase2_lead_agent_architecture_audit_v1`
Architecture evidence hash: `93cbbcab32be4f48a83833f9343cefdb1d06ef6352c964b97ee0a94430f725e5`

## Executive decision

`READY_PHASE2_ARCHITECTURE`

This means the research question, inherited constraint, observation timing, scalar action, auditable reward, temporal protocol, environment proposal, algorithm comparison, and Phase-2 work sequence are now explicit and internally coherent.

It does **not** authorize training. Current training readiness is:

`BLOCKED_PENDING_P2_2_DATA_CONTRACT`

The repository has usable local PSX index history, but it has no authoritative point-in-time Pakistan macro dataset or release calendar. P2.2 must establish those sources, causal alignment, a common date set, and a TRAIN-only scaler before any Lead-Agent optimizer may run.

No Lead Agent was trained, no TEST observations were opened, no Phase-3 agents or model artifacts were changed, and no network request was made during this audit.

## 1. Existing Lead-Agent inventory

Repository-wide search found no executable Lead Agent, macro allocator, top-level capital allocator, market-regime policy, Lead-specific environment, or Lead-specific trainer/evaluator. Several similarly named foundations exist, but none is the Phase-2 policy.

| Capability | Classification | Exact evidence | Finding |
|---|---|---|---|
| Executable Lead Agent / allocator | MISSING | None | No Lead observation, action, reward, environment, trainer, evaluator, config, artifact, or test existed before P2.1. |
| Legacy master AI data/split | PARTIAL; obsolete as a Lead contract | `feature_engineering/dataset_builder.py`; `feature_engineering/splitting.py`; `data/processed/master/psx_ai_master.csv`; `data/processed/splits/master/metadata.json` | A global all-symbol feature table and TRAIN-fitted scaler exist. It is an equity-row legacy artifact, has no macro release timing or market-allocation objective, and deliberately has no Lead RL contract. TEST files exist but were not opened. |
| Legacy `master` model namespace | PARTIAL | `reinforcement_learning/model_management/paths.py`; `registry.py`; `persistence.py`; `reinforcement_learning/saved_models/master_models/.gitkeep` | The generic namespace is only scaffolding. Registry v2 accepts `symbol/master`, forces algorithm `PPO`, has zero data rows, and has no master/Lead trainer. “Master” must not be mistaken for “Lead Agent.” |
| Single-symbol PPO/RecurrentPPO | COMPLETE for its original scope; not Lead | `reinforcement_learning/environments/single_symbol_env.py`; `training/ppo_trainer.py`; `training/recurrent_trainer.py`; `evaluation/ppo_evaluator.py`; `evaluation/recurrent_evaluator.py` | Generic device, seeding, callback, metrics, and atomic-write patterns are reusable later. Its Discrete Hold/Buy/Sell action, per-symbol data contract, and execution semantics are Phase-3 concerns and remain untouched. |
| Multi-symbol recurrent orchestrator | COMPLETE for independent-symbol jobs; not Lead | `reinforcement_learning/training/recurrent_orchestrator.py`; `job_state.py`; `global_validation.py`; `model_details.py` | “Global” means inventory/comparison of independent symbol models, not one global allocation policy. |
| Sector recurrent prototype | EXPERIMENTAL / OBSOLETE for Phase 2 | `reinforcement_learning/environments/sector_training_env.py`; `training/sector_balanced_trainer.py`; Milestone 6E reports | Same-sector recurrent pretraining is retained as prototype evidence. It neither allocates capital nor provides an accepted cluster basis. |
| Soft NMF decoder | EXPERIMENTAL / BLOCKED | `data_pipeline/src/soft_relationship_representation.py`; `docs/report_logs/milestone_7d_soft_relationship_representation_audit.md` | The decoder is mathematically capital-conserving, but no NMF representation passed Phase-1 acceptance. It is not an approved action basis. |
| Market-intelligence layer | PARTIAL | `market_intelligence/index_parser.py`; `index_store.py`; `refresh_indices.py`; `feature_joiner.py`; `index_metrics.py`; `market_health.py` | Official index storage and causal calculation helpers exist, but there is no versioned point-in-time Lead dataset. |
| SAC path | MISSING | No project SAC trainer/config/persistence/evaluator | SB3 includes SAC, but repository integration and replay-buffer safety do not exist. |

Reusable foundations are limited to generic patterns: atomic storage, SHA-256 integrity, fail-closed device resolution, deterministic seeds/callbacks, job lifecycle concepts, portfolio metric math, and official index parsing. Existing per-symbol and sector contracts must not be edited or treated as the Lead architecture.

## 2. Market-level data inventory

### Canonical consolidated market data

| Field | Audit result |
|---|---:|
| Path | `data/parquet/market.parquet` |
| File SHA-256 | `8843f4573baf07a0c1e266efcb9ba37f96a839a42b41f75528d4f1c67b78fb34` |
| Rows | 1,535,391 |
| Historical identifiers | 4,681 |
| Market dates | 2,501 |
| Range | 2016-07-26 through 2026-08-28 |
| Duplicate `(market_date, symbol)` keys | 0 |
| Required nulls | 0 |
| Negative volume | 0 |
| Non-positive close | 0 |

This is canonical local market evidence, but the full 4,681-identifier history is not an all-equity point-in-time universe. It includes variants and other instruments. Consequently, historical breadth or aggregate equity volume cannot be built by blindly summing every symbol. Current-listing membership would create survivorship bias; a historical effective-date policy would be needed.

### Official PSX index master

Source: official PSX timeseries endpoint, stored locally at `data/indices/master/psx_indices_master.csv`.

| Field | Audit result |
|---|---:|
| File SHA-256 | `09244e0d3484276843b00159f76e69b2e782ce3f1256ae5b2b21512c1b51cb6d` |
| Total rows | 5,012 |
| Codes | KSE100, KSE30, KMI30, ALLSHR |
| Rows/dates per code | 1,253 |
| Range | 2021-08-06 through 2026-08-27 |
| Duplicate `(index_code, date)` keys | 0 |
| Missing/non-positive levels | 0 |
| Missing/negative volume | 0 |
| Missing `open` | 0 |
| Canonical market sessions missing from all indices within the span | 1: 2021-10-19 |

All four index calendars are identical. The combined context has 1,253 dates. When recomputed from levels, each 1-day return has one warm-up null, KSE100 5-day return has five, and KSE100 20-session volatility has 20. Expanding drawdown has no nulls.

Two stored `daily_change` and `daily_change_percent` values per index are null: 2021-08-06 and 2021-08-30. The second is not a missing level; it is a refresh-boundary artifact. The parser calculated changes within each fetched payload before the store merged 14 older cached dates with the later full payload. Therefore Lead features must recompute returns from sorted index levels and must not trust stored change columns.

The raw current snapshot holds 1,240 source records per index with a duplicate 2024-01-22 record; parsing correctly collapses it to 1,239 and storage adds the 14 earlier non-overlapping dates. Raw snapshots are overwritten and all retained `fetched_at` values are from 2026, so the repository has no historical vintage/revision archive. Index levels are usable under a documented conservative lag, but vintage limitations must remain visible.

The endpoint's fourth positional field is stored as `open`, yet local code does not establish a versioned execution-price meaning. P2.1 excludes it from execution semantics.

### Existing and proposed market features

| Feature family | Local status | Point-in-time finding |
|---|---|---|
| Index returns, momentum, volatility, drawdown, moving-average distance | Derivable | Safe only when recomputed from levels, backward-looking, and shifted so session `t` market outcomes are never in the decision for the session-`t` rebalance. |
| Official index volume | Available | Prefer over an all-identifier market aggregate; lag one completed session. |
| Historical breadth | Missing canonical history | Current-registry breadth leaks survivor membership; all-security breadth is instrument-contaminated. Excluded from v1. |
| Aggregate equity volume | Unsafe without a point-in-time instrument universe | Not an initial feature. |
| Market Health score | Latest descriptive calculation only | Not a canonical historical point-in-time series. Excluded from v1. |
| Legacy AI-master index columns | Partial and stale | They duplicate market context across symbol rows, end on 2026-08-05, and have no Lead contract. They are not the source of truth. |

## 3. Macro-data inventory

An exhaustive local search found no authoritative Pakistan macro dataset, loader, schema, release calendar, update state, provider dependency, or tests. Government-security and TFC-like market tickers are traded instrument observations, not a macro/yield time series and not a substitute.

| Candidate | Status | Release/revision risk | P2.1 disposition |
|---|---|---|---|
| SBP policy rate | MISSING | Announcement and effective timestamps differ | Minimal v1 candidate; P2.2 blocker |
| Pakistan CPI/inflation | MISSING | Reference month is not release date; revisions possible | Minimal v1 candidate; first-release history required |
| PKR/USD | MISSING | Rate type, timezone, and publication cutoff matter | Minimal v1 candidate; authoritative timestamped series required |
| Yields/KIBOR/money-market | MISSING | Instrument and observation-time semantics vary | Deferred, not needed for minimal v1 |
| FX reserves/money supply | MISSING | Weekly/monthly lags and revisions | Deferred |
| Production/trade/current account | MISSING | Long release lags and revisions | Deferred |
| Commodities | MISSING | Vendor/timezone/close alignment unresolved | Not needed for v1 |

No macro candidate is marked `AVAILABLE`. The proposed observation dimension therefore remains deliberately unfrozen.

## 4. Point-in-time and leakage findings

The Lead decision contract is:

1. A decision is formed before an abstract benchmark rebalance at session `t` close.
2. Market observations may extend only through completed session `t-1`; the close or volume of `t` is never visible.
3. The chosen exposure applies to the benchmark return from close `t` to close `t+1`, after one explicit turnover charge.
4. A macro observation enters only when `release_timestamp <= decision_cutoff`. `reference_period` and `effective_date` alone never prove availability.
5. Low-frequency values may be carried forward only after release, by backward as-of join, with release age and a predeclared staleness rule. No backward fill, future interpolation, or pre-release period-date fill is allowed.
6. Revised series require first-release vintages. If historical vintages cannot be established, exclude the series rather than backfilling future-known revisions.
7. Every rolling feature is backward-looking and shifted before joining to its decision date.
8. Scalers/imputation statistics fit on TRAIN only. VALIDATION applies the frozen transformer; TEST remains sealed.

This close-to-next-close proxy is intentionally conservative because the local endpoint does not prove that its positional `open` field is an executable index price. P2.3 must make the market-proxy limitation explicit; no claim of direct index execution is allowed.

## 5. Phase-2 research question

> On one common PSX market calendar, can a leakage-safe long-only market-level RL policy use only prior-session PSX index dynamics and point-in-time Pakistan macro releases to choose scalar equity-market exposure that improves validation-period net risk-adjusted performance and drawdown relative to predeclared cash, fixed-exposure, full-market, and rule-based baselines?

A required market-only versus market-plus-macro ablation distinguishes whether macro inputs add evidence. This is a market-risk question, not a Phase-3 stock-selection or execution question.

## 6. Action-space decision

The recommended contract is `lead_agent_scalar_market_exposure_v1`:

- Gymnasium `Box(low=0, high=1, shape=(1,), dtype=float32)`.
- Action means target long-only PSX benchmark exposure; the remainder is cash.
- No shorting, leverage, per-symbol weights, sector buckets, clusters, or soft prototypes.

Alternatives were rejected as the primary action:

- Defensive/neutral/aggressive fixed levels are useful deterministic baselines but unnecessarily quantize exposure.
- Binary cash/market is transparent but too restrictive and may induce switching.
- Per-symbol allocation is Phase 3, while cluster/sector/prototype vectors violate the Phase-1 constraint.

## 7. Reward decision

The recommended v1 reward is `lead_agent_net_growth_reward_v1`:

`reward_t = log(net_portfolio_value_t+1 / net_portfolio_value_t)`

The net value reflects `cost_rate × absolute change in target exposure`, deducted exactly once. Cash return is nominally zero until a safe cash-yield series exists.

Drawdown, volatility, Sharpe, Sortino, turnover, and downside risk are evaluation metrics. A separately predeclared drawdown-increment reward is permitted only as a later sensitivity experiment; its coefficient must not be selected after looking at VALIDATION. Episode Sharpe/Sortino is not used as a sparse step reward.

## 8. PPO versus SAC recommendation

Decision: `BOUNDED_PPO_VS_SAC_COMPARISON`.

| Dimension | PPO | SAC |
|---|---|---|
| Scalar continuous action | Supported | Native strength |
| Existing project reuse | Strong generic SB3/device/callback patterns | None beyond the installed SB3 capability |
| Sampling | On-policy, simpler audit | Off-policy, potentially more sample-efficient |
| New safety work | New Lead trainer/data loader | Trainer, replay-buffer leakage/resume policy, persistence, diagnostics |
| Reproducibility | Familiar local path | More state to freeze and verify |

P2.4 should use the same common TRAIN environment, environment transitions, costs, feature contract, and seeds `[42, 43, 44]`. Defaults/bounded candidates must be predeclared before VALIDATION. The initial policy is nonrecurrent MLP: lagged and rolling inputs already encode history, and an LSTM would add an unproven variable. CPU is the reference path; CUDA must be explicitly qualified; MPS is not auto-selected.

## 9. Temporal policy

Proposed version: `lead_agent_common_calendar_split_v1`.

After P2.2 performs release-aware alignment and backward-looking warm-up, it freezes the ordered common decision-date list:

- TRAIN: first `floor(70% × N)` common dates.
- VALIDATION: next `floor(15% × N)` common dates.
- TEST: remaining dates, sealed.

Exact dates are intentionally not invented in P2.1 because macro coverage is unknown. The Phase-1 fixed relationship window is not reused: it answers a different question and starts years before local official index history. `rl_partition_v1` also does not apply; that contract splits each symbol's own usable observations independently rather than one shared market/macro timeline.

## 10. Proposed Phase-2 data contract

Contract proposal: `lead_agent_market_macro_contract_v1`

Current status: `SCHEMA_PROPOSAL_BLOCKED_ON_MACRO_DATA`

Market candidates are:

- KSE100 1-, 5-, and 21-session returns.
- KSE100 20-session volatility and expanding drawdown.
- KSE100 20-/50-session moving-average distance.
- KSE100 volume-to-20-session-average ratio.
- One-day return dispersion across KSE100, KSE30, KMI30, and ALLSHR.

Minimal macro candidates are:

- SBP policy-rate level and last announced change.
- PBS CPI YoY and release-to-release change.
- Authoritative SBP USD/PKR backward-looking changes/volatility.

Dynamic causal state is prior target exposure and current portfolio drawdown. Every observation is one deterministic `float32` row per decision date. The final order/shape, source and release-calendar hashes, staleness policy, transformer type, and exact split dates belong to P2.2.

Explicitly excluded inputs are stored index changes, ambiguous index `open`, current-only breadth, all-identifier volume, latest-only health score, revised macro without vintages, Phase-1 relationship/sector features, and Phase-3 symbol-agent outputs.

## 11. Proposed environment

Version: `lead_agent_market_risk_env_v1` (proposal only; not implemented).

- Observation: frozen market/macro vector plus causal portfolio state.
- Action: target market exposure `[0,1]`.
- Return proxy: close `t` to close `t+1`, chosen without observing session `t` market outcomes.
- Reward: net log growth after one turnover charge.
- Episode: common chronological TRAIN episode or predeclared chronological windows; never shuffled dates.
- State: exposure, net value, running peak, and drawdown carry within an episode and reset at episode boundaries.
- Policy: nonrecurrent initial baseline.
- TEST: environment loader must reject it.

The index is a research proxy rather than a directly tradable instrument. P2.3 must test action timing, finite observations/rewards, no same-session information, cost accounting, bounds, reset state, and deterministic chronology.

## 12. Binding Phase-1 constraint

Phase-1 result: `REJECTED_CLUSTER_STRUCTURE`.

Binding consequence: the Phase-2 Lead Agent cannot require hard clusters, accepted soft NMF memberships, or sectors relabelled as clusters. The chosen scalar market-risk architecture satisfies this constraint without a workaround.

## 13. Gap matrix

| Capability | Existing status | Evidence | Missing work | Owner |
|---|---|---|---|---|
| Index dataset | PARTIAL | Four official series; 2021-08-06..2026-08-27 | Snapshot/version policy; level-derived returns; timing/revision caveats | P2.2 |
| Macro dataset | MISSING | No authoritative local series/releases | Minimal authoritative series, releases/vintages, hashes | P2.2 |
| Point-in-time alignment | MISSING | Generic backward joins only | Release-aware as-of alignment and leakage tests | P2.2 |
| Feature engineering | PARTIAL | Index helpers exist | Versioned common Lead table | P2.2 |
| Normalization | PARTIAL | TRAIN-only symbol patterns exist | Common Lead TRAIN transformer | P2.2 |
| Environment | MISSING | Single-symbol/sector envs are incompatible | Implement `lead_agent_market_risk_env_v1` | P2.3 |
| Reward | PROPOSED | Net-growth contract frozen here | Component accounting/counterfactual tests | P2.3 |
| Action | PROPOSED | Scalar exposure frozen here | Box/turnover/baseline implementation | P2.3 |
| PPO | PARTIAL | Generic SB3 patterns | Lead-only loader/config/trainer | P2.4 |
| SAC | MISSING | No project integration | Config/trainer/replay safety/diagnostics | P2.4 |
| Persistence | PARTIAL | Atomic PPO bundles; registry v2 | Lead scope and algorithm-aware provenance | P2.5 |
| Validation | PARTIAL | Generic metrics/baselines | Common-calendar baselines/ablation/multi-seed comparison | P2.4/P2.6 |
| Dashboard | MISSING | No Lead controls | Add only after accepted contracts; keep TEST inaccessible | P2.6 |

## 14. Exact Phase-2 milestone sequence

1. **P2.1 — Architecture/data audit:** this decision; no training.
2. **P2.2 — Leakage-safe market+macro contract:** authoritative releases/vintages, causal features, common split, TRAIN scaler, source hashes, sealed TEST metadata.
3. **P2.3 — Lead-Agent environment:** scalar exposure, costs/reward, timing invariants, and deterministic baselines.
4. **P2.4 — Bounded PPO versus SAC benchmark:** same-budget, multi-seed TRAIN runs and VALIDATION-only comparison.
5. **P2.5 — Candidate training and persistence:** selected predeclared candidate, TRAIN only, atomic algorithm-aware bundle.
6. **P2.6 — VALIDATION acceptance and Phase-2 closure:** frozen gates, baselines/ablation, closure artifact, and limited controls. TEST remains sealed unless separately authorized.

No Phase-3 implementation is part of this plan.

## 15. Safety and limitations

- No Lead Agent training, optimizer call, validation run, or model write occurred.
- TEST observation files/dataframes were not opened. Existing TEST path/row metadata was treated only as inventory metadata.
- No Phase-3 recurrent agent, environment, data contract, run state, or model artifact was modified.
- No joint fine-tuning or Phase-4 integration started.
- No network fetch occurred.
- The official index history begins in 2021 and has no historical vintage archive.
- No local macro data currently satisfies the roadmap requirement.
- The proposed benchmark exposure is an abstract research proxy and needs an explicit friction sensitivity.
- The common-calendar exact boundaries and final observation dimension are intentionally deferred to P2.2.

## Final status

**Architecture:** `READY_PHASE2_ARCHITECTURE`

**Training:** `BLOCKED_PENDING_P2_2_DATA_CONTRACT`
