# Streamlit Feature Ownership and Duplication Audit

Audit date: **2026-08-11 (Asia/Karachi)**

Audit mode: **read-only architecture revision**

## Executive finding

The application has several strong shared backends, but page ownership is not
yet clean. The largest unnecessary overlaps are:

- Dataset Explorer and Stock Explorer both provide company/security selection,
  profile information, price-period filtering, price charts, paginated history,
  and downloads;
- Historical Backfill, Automation, and Training & Models all expose rebuild or
  dataset-preparation controls at different levels;
- Market Overview and Market Indices both render detailed index-period health;
  and
- Training & Models combines the safe one-symbol PPO lifecycle with dataset
  building, split creation, bulk-symbol preparation, and master-dataset tools.

The desired rule is: **one owner for each user intent, one canonical backend
for each calculation, and page scripts limited to presentation, explicit
actions, and session state**. Reuse of `dashboard` helpers is good duplication;
copying period maps, readiness definitions, rebuild order, or model lifecycle
rules into page files is not.

## Milestone 6A implementation update

Milestone 6A implemented the Training & Models portion of this ownership plan:

- removed the legacy user-editable minimum-history widget;
- fixed current MLP readiness to the canonical configured 252-row gate;
- removed AI dataset build/validation, split creation, bulk dataset
  preparation, and master-dataset/model controls from Training & Models;
- retained the underlying backend capabilities pending an approved Data
  Operations owner;
- retained only model lifecycle and read-only training metadata; and
- added a separate read-only 126/100-observation future history-class policy
  that does not change `rl_partition_v1` readiness.

Training & Models now presents sections A–H for readiness, model architecture,
configuration, selected-symbol metadata, training, VALIDATION, candidate
persistence, and registry history. The current local reconciliation is 454
processed-ready, 454 compatible-contract, and 454 selectable symbols.

## Audit method

Every page registered by `app.py` and each module under `dashboard/` was read.
The review followed the repository's Streamlit architecture conventions:
stable widget keys, explicit write buttons, correctly keyed caching, page
scripts as UI controllers, and testable non-visual logic in shared modules.

## Page and control inventory

| Page | Visible feature/control | Backend/helper | Overlap elsewhere | Assessment | Recommended owner |
|---|---|---|---|---|---|
| **Market Overview** | Overall market health and analytical disclaimer | `calculate_index_metrics`, `calculate_market_breadth`, `calculate_market_health_score`; presentation helpers | Index health detail on Market Indices | Broad health is legitimate here; detailed selected-index decomposition is duplicated intent | Market Overview owns aggregate snapshot only |
| Market Overview | All/single index period selector, period cards, health components, MA/level/drawdown/volatility charts | canonical `market_intelligence.index_periods`, `market_health`, `dashboard.index_period_presentation` in the current tree | Market Indices provides the same deep analysis | Shared backend is **good**; duplicate deep UI is unnecessary | Keep a compact index comparison/link here; Market Indices owns detail |
| Market Overview | Breadth metrics/tables | `calculate_market_breadth`, `dashboard.market_overview` | No equivalent analytical page | Legitimate | Market Overview |
| Market Overview | Pipeline, automation, index-data system status | config paths and `load_automation_config` | Automation owns scheduling/status; other pages show data freshness | Summary is legitimate, controls/detail are not | Overview shows compact status; Automation owns schedule detail |
| **Market Indices** | Index selection and 1M/3M/6M/1Y/Maximum period selection | `analyze_index_period`, presentation helpers | Overview index panel | Deep index work is legitimate owner | Market Indices |
| Market Indices | Refresh selected/all official indices | `refresh_indices` | Overview source caching/status | Legitimate explicit data action, but Overview must not duplicate it | Market Indices |
| Market Indices | Period contract, health score/components, period summary, MA, drawdown, rolling volatility, table/download | shared index-period core/presentation | Overview currently renders much of this | Good backend reuse; consolidate the visible detail here | Market Indices |
| **Fetch Data** | Single date or date-range form and explicit Fetch button | `collect_single_date`, `collect_date_range`, `load_csv_preview` | Automation “Fetch missing data now”; Backfill historical range | Different user intent is legitimate if labels/bounds remain clear | Fetch Data owns user-selected one-off collection |
| Fetch Data | Collection summary and CSV preview | structured collection result/presentation helpers | Backfill result summaries | Legitimate result-specific presentation | Fetch Data |
| **Dataset Explorer** | Source/dataset summary | `load_dashboard_dataset`, summary/filter helpers | Training readiness and Backfill rebuild outputs also show counts | Read-only dataset inspection is legitimate | Dataset Explorer |
| Dataset Explorer | Company/security filters and company table | `filter_company_registry`, registry display helpers | Stock Explorer and Company Registry | Unnecessary domain overlap | Move company metadata to Company Registry; symbol analysis to Stock Explorer |
| Dataset Explorer | Selected security profile/status | `filter_market_data`, registry row formatting | Stock Explorer duplicates it | Unnecessary | Stock Explorer |
| Dataset Explorer | Period-filtered price chart, paginated day data, CSV download | `filter_stock_history_period`, chart/pagination helpers | Stock Explorer duplicates almost exactly | Unnecessary visible duplication despite shared helper reuse | Stock Explorer |
| **Stock Explorer** | Listed/ordinary/lifecycle/type filters and symbol selector | `dashboard.registry_loader` filters | Dataset Explorer and Company Registry | Necessary selection for a stock-analysis page; use the shared registry backend | Stock Explorer |
| Stock Explorer | Profile, freshness/status, period chart, security information, paginated prices, download | `dashboard.data_loader` and presentation helpers | Dataset Explorer | Correct owner | Stock Explorer |
| **Automation** | Automation enabled/start-date/rebuild settings | `load/save_automation_config` | Overview system status; manual rebuild controls elsewhere | Legitimate configuration | Automation |
| Automation | “Fetch missing data now” | `run_manual_update` | Fetch Data one-off collection and Backfill | Manual incremental maintenance overlaps data operations, not scheduling | Move to a single Data Operations owner |
| Automation | “Rebuild master dataset” | `build_master_dataset` | Backfill full rebuild; Training dataset builds | Unnecessary duplicate mutation control | Data Operations owner only |
| Automation | Install/trigger/uninstall LaunchAgent and schedule status | `data_pipeline.src.launchd` | Overview status only | Legitimate | Automation |
| **Company Registry** | Official/current registry metrics, filters, security table, details, download | registry loader/filter/presentation helpers | Dataset/Stock company metadata | Correct owner for registry-wide metadata | Company Registry |
| Company Registry | Refresh and rebuild official registry | `refresh_and_build_registry` | Full rebuild orchestration refreshes registry internally | Legitimate scoped action if it delegates to one canonical backend | Company Registry |
| **Training & Models** | RL readiness, future history class, architecture status, selected contract summary | `build_model_readiness_table`, `build_ready_symbol_catalog`, `classify_usable_history`, metadata-only loader | No lower duplicate remains after Milestone 6A | Canonical and explicitly separated | Training & Models |
| Training & Models | One-symbol config, explicit train, VALIDATION comparison, candidate save, model history | `dashboard.ppo_workflow` -> trainer/evaluator/persistence/registry | No other model-training page | Correct owner | Training & Models |
| Training & Models | Former AI dataset build/validation and split controls | Backend functions remain, UI removed in Milestone 6A | Backfill and Automation still expose broader data operations | Training duplication resolved | Future Data Operations owner |
| Training & Models | Former bulk symbol dataset preparation | Selection/build helpers remain, UI removed in Milestone 6A | Future sector/generalized universe selection is not yet implemented | Legacy duplication resolved | Future training-universe workflow after contract approval |
| Training & Models | Former master dataset/model block | Backend remains, UI removed in Milestone 6A | Proposed sector/generalized scopes supersede ambiguous master wording | Legacy duplication resolved | Future versioned sector/generalized model workflow |
| **Historical Backfill** | Range, planner, Resume, Retry Failed, Retry Temporary, state/results | `dashboard.backfill_preview`, `run_backfill`, state helpers | Fetch Data range collection | Legitimate specialized recovery workflow | Historical Backfill |
| Historical Backfill | Rebuild Data Products | `rebuild_data_products` | Automation and Training controls | Unnecessary cross-domain control | Remove/move to single Data Operations owner; show a link/status instead |

## Dashboard-helper inventory

| Helper | Responsibility | Assessment | Recommendation |
|---|---|---|---|
| `dashboard.data_loader` | Canonical local market loading, filtering, stock periods, summaries, pagination, previews | **Good shared logic.** It prevents Dataset/Stock from maintaining separate filters. | Retain; let Dataset Explorer and Stock Explorer use narrower subsets of its API. |
| `dashboard.registry_loader` | Registry loading, lifecycle/security filters, display metrics | **Good shared logic.** | Retain as the only page-neutral registry query layer. |
| `dashboard.presentation` | Null-safe labels, date/number/status formatting | **Good shared presentation.** | Retain; pages should not introduce raw `NaN` formatting. |
| `dashboard.market_overview` | Snapshot and breadth/presentation helpers | Mostly good, page-domain specific | Keep aggregate-market helpers; do not reimplement index-period formulas here. |
| `dashboard.index_period_presentation` | Contract cards, health breakdown, level/MA, drawdown, and volatility chart preparation | **Good current consolidation.** Both index pages reuse it. | Make Market Indices the detailed renderer; Overview uses only compact summaries. |
| `market_intelligence.index_periods` | Canonical per-index period filtering and causal analytics | **Good business-logic ownership.** | Preserve as the sole period-analysis contract; do not restore page-local mappings. |
| `dashboard.backfill_preview` | Pure preview/state reconciliation presentation for Backfill | Legitimate page-specific helper | Retain; processing remains in `data_pipeline.src.backfill`. |
| `dashboard.ppo_workflow` | Ready catalog, metadata, identity/state gates, action wrappers, metrics/charts, registry history | Correct safety boundary but broad (catalog + controller + presentation) | Split later into `ppo_readiness`, `ppo_session_actions`, and `ppo_presentation` without duplicating rules. |

## Duplication by requested concern

| Concern | Current locations | Finding | Canonical owner |
|---|---|---|---|
| Dataset build/rebuild | Automation, Training & Models, Historical Backfill | Three manual mutation surfaces with overlapping dependency chains | New Data Operations page/service |
| Split creation | Training page loop and full rebuild pipeline | Page-local orchestration can drift from pipeline ordering | `rebuild_data_products`/central data-product service |
| Readiness calculations | Feature readiness, model status, PPO contract catalog, multiple Training panels | Different layers are valid, but labels/counts previously mixed snapshots | `feature_engineering.readiness` -> `model_management.status` -> `dashboard.ppo_workflow` |
| Model status/history | Training page and registry/status helpers | One visible owner; helper separation is good | Training & Models |
| PPO readiness | Top PPO catalog and lower general readiness table | Legitimate contract-compatible subset, but must display a mathematical reconciliation | Training & Models using canonical processed-bound provenance |
| Historical fetching | Fetch Data, Backfill, Automation incremental update | Three distinct workflows, but manual incremental action needs clearer ownership | Fetch one-off; Backfill recovery; Automation schedule only; Data Operations incremental/rebuild |
| Backfill controls | Historical Backfill only | No problematic duplication | Historical Backfill |
| Company/symbol selection | Dataset, Stock, Registry, Training | Some selection is contextual; Dataset's stock-analysis copy is unnecessary | Registry metadata; Stock analysis; Training universe; Dataset no company analysis |
| Index health | Overview and Indices | Calculation is now shared, but detailed rendering is duplicated | Overview compact; Indices deep |
| Market health | Overview only | Correct | Market Overview |
| Date/period filtering | Shared stock helper; shared index-period helper; page-specific controls | Good if helpers stay canonical; prior page-local index mappings caused drift | Domain core helpers |
| Chart preparation | Dataset and Stock duplicate stock presentation; index chart helper is shared | Stock chart UI duplicate is unnecessary; index chart reuse is good | Stock Explorer; index presentation helper |
| Dataset metadata | Dataset, Training, Backfill result | Contextual summaries are legitimate; schema/partition inspection belongs Dataset Explorer | Dataset Explorer for detail; compact status elsewhere |
| Automation status | Overview and Automation | Compact summary plus detailed owner is legitimate | Automation owns detail |

## Relationship to the three known dashboard defects

The known defects are architecture warnings, not isolated cosmetic bugs.

### 1. Market Overview period-health mismatch

The page filtered the chart by period, but the old score used only trailing
1/5/20/50-observation components. Once a period had at least 50 rows, 3M, 6M,
1Y, and Maximum could be identical despite different period returns,
volatility, and drawdown. A separate full-history “Index Health Overview” made
the static appearance more confusing.

The current working tree uses the canonical `index_periods` analysis and a
period-specific health method. Preserve that consolidation and reduce Overview
to a compact period comparison; detailed components belong on Market Indices.

### 2. Market Indices static health bars

The old page filtered `shown` for the chart/table but passed the unfiltered
`data` frame to `calculate_index_metrics`. It also maintained its own period
mapping. The duplicated data paths allowed chart and health to disagree.

The current tree routes chart, summary, and health through the same period
analysis. This shared contract is the correct permanent ownership model.

### 3. PPO-ready count 0 versus lower readiness 454

The general readiness table combined current raw history (through a newer
date) with processed-dataset row counts, while the PPO catalog compared those
raw first/last dates against the older but internally consistent RL contract.
All 454 valid contracts were rejected by a cross-snapshot date mismatch.

The correct reconciliation is explicit:

```text
active supported ordinary symbols
  -> processed-dataset Ready
  -> processed bounds match RL contract bounds
  -> feature/environment/contract/scaler compatible
  = PPO-ready
```

Current raw first/last dates remain freshness signals, not contract-boundary
identity. The current tree propagates processed first/last dates and reports
454 eligible, 454 compatible, and 454 in the intersection. Preserve this
layering and display rejection buckets whenever counts diverge.

## Recommended page ownership map

| Page | Sole responsibility | Controls to keep | Controls/details to move or remove |
|---|---|---|---|
| **Market Overview** | Broad market snapshot | Overall market health, compact index comparison, breadth, compact system freshness | Move detailed selected-index health/components/MA/drawdown/volatility to Market Indices |
| **Market Indices** | Deep official index analytics | Refresh, index/period selector, period contract, health breakdown, causal charts, table/download | No broad market health or automation controls |
| **Fetch Data** | Explicit one-off collection | Single date/range fetch and immediate result/preview | No backfill recovery or product rebuild |
| **Historical Backfill** | Historical acquisition, retry, reconciliation, audit state | Preview/Resume/Retry Failed/Retry Temporary and results | Remove Rebuild Data Products; show downstream-stale status/link |
| **Dataset Explorer** | Read-only dataset inspection | Source layers, schema, row/date/symbol counts, quality metadata, partition metadata | Remove company browser, selected-stock chart/history/download |
| **Stock Explorer** | One-security market analysis | Security selector/profile, period chart, history, security-level download | No registry-wide maintenance |
| **Company Registry** | Company, security, sector, listing/lifecycle metadata | Registry metrics/filters/detail/download and scoped official refresh | No price-history exploration |
| **Training & Models** | AI/RL readiness, training, validation, persistence, model lineage/history | Readiness reconciliation, one/sector/generalized research workflows, registry history | Move raw/AI dataset building and split orchestration out; replace ambiguous master controls |
| **Automation** | Scheduled workflows only | Configuration, scheduler status/install/trigger/uninstall, run logs | Move manual incremental fetch and rebuild buttons |
| **Data Operations** *(recommended new page)* | Single manual owner for derived-data operations | Incremental maintenance, master/registry/AI/split rebuild through one canonical orchestrator, validation report | No model training, backfill state machine, or browsing |

If adding a page is not acceptable, the same Data Operations section can live
under Fetch Data, but it must remain one UI owner and call the same central
orchestrator used by automation. It should not remain copied across Automation,
Backfill, and Training.

## Control migration plan

1. First extract a read-only operation-status/result contract around
   `rebuild_data_products`; do not move buttons before the backend dependency
   order is canonical and tested.
2. Add the single Data Operations owner, then replace old buttons with links or
   stale-status messages.
3. Remove stock browsing from Dataset Explorer only after Stock Explorer has
   equivalent filters/downloads and shared URLs/session state where needed.
4. Reduce Market Overview detail after the compact index comparison is tested;
   leave all deep analytics on Market Indices.
5. Split the large PPO workflow helper by responsibility without changing its
   safety gates.
6. Add page-level smoke tests proving page load never triggers fetch, rebuild,
   train, validate, persist, retry, or refresh actions.

## Streamlit-specific safety recommendations

- Keep write operations only in explicit button/form branches.
- Use stable, page-scoped widget keys; do not share mutable session values
  across unrelated page controls without a versioned identity.
- Cache only deterministic reads. Include file identity/version inputs in cache
  keys; never cache training, persistence, fetch, refresh, or backfill actions.
- Keep business calculations in core modules and chart/display preparation in
  dashboard helpers. Page scripts should compose and render results.
- Use schema-correct empty objects after read failures so a warning is not
  followed by a secondary page crash.
- Display a reason-count reconciliation whenever one readiness layer is a
  strict subset of another.
- Do not expose a generic partition selector capable of reaching sealed TEST.

## UI consolidation milestones

1. **Ownership tests:** codify one owner per mutation and zero side effects on
   page load.
2. **Data Operations consolidation:** centralize rebuild/split/manual update
   controls and remove duplicate buttons.
3. **Explorer separation:** Dataset Explorer becomes dataset-only; Stock and
   Registry retain their respective domain views.
4. **Index presentation consolidation:** Overview compact, Indices deep, one
   canonical period/health backend.
5. **Training page simplification:** retain readiness/training/evaluation/model
   lineage; move data-product orchestration.
6. **Recurrent UI extension:** only after recurrent contracts, registries, and
   evaluation are production-tested; expose explicit scope/sector/parent
   lineage and never add a Train All shortcut prematurely.

## Read-only statement

This audit did not move or remove a control, modify a Streamlit page/helper,
run a fetch/backfill/rebuild/training/evaluation action, change navigation,
write session state, alter data/model artifacts, or commit. It created this
documentation artifact only.
