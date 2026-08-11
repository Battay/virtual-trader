# Milestone 6A: Dashboard Ownership and Readiness

Implementation date: **2026-08-11 (Asia/Karachi)**

## Objective

Milestone 6A finishes the existing dashboard correctness work and limits
**Training & Model Management** to model-lifecycle responsibilities before any
recurrent PPO implementation begins. It also adds a read-only history policy
for the approved future sector-transfer research without changing current
`rl_partition_v1` eligibility.

No recurrent dependency was installed, no recurrent trainer was implemented,
and no model was trained or promoted.

## Manual issue and root cause

The removed lower page contained this widget:

```python
st.number_input(
    "Minimum usable rows for symbol eligibility",
    min_value=1,
    key="training_minimum_history",
)
```

The current production default is `AI_MINIMUM_USABLE_ROWS = 252`, and the page
used `setdefault` to seed the same session key before calculating readiness.
The widget had no explicit `value`, allowed values down to 1, was rendered only
after the readiness panels, and retained prior per-session widget state. This
created a second, user-editable readiness authority on the model-lifecycle
page.

A clean Streamlit test session displayed 252. Setting the legacy widget to 1
proved it did affect the next readiness calculation: the compatible PPO count
remained 454 while the Insufficient History count changed from 19 to 4. The PPO
count stayed 454 because the selectable universe is also capped by the 454
existing compatible `rl_partition_v1` artifact sets. Thus the control could
change one readiness layer while leaving the headline intersection unchanged,
which made it appear disconnected and could produce misleading/stale-looking
combinations.

The fix is not “change 1 to 126.” The widget and its session key were removed.
Current MLP readiness now always receives the canonical configured 252-row
gate on this page.

## Current versus future eligibility

The two concepts are now explicit and independent:

| Concept | Source | Current policy | Effect |
|---|---|---|---|
| **Current MLP PPO readiness** | `feature_engineering.readiness` -> `model_management.status` -> `dashboard.ppo_workflow` compatible-contract intersection | Fixed 252 usable observations plus valid processed dataset and compatible `rl_partition_v1` artifacts | Controls the current selectable PPO universe |
| **Future recurrent/transfer history class** | `reinforcement_learning.history_policy.classify_usable_history` applied to canonical `usable_rows` | Mature ≥126; Cold Start 100–125; Insufficient <100 | Read-only research metadata; changes no current contract, split, scaler, or eligibility |

The future helper accepts only a finite, non-negative whole usable-observation
count. It does not accept calendar dates and therefore cannot guess maturity
from listing age.

Current active ordinary-equity counts are:

- **Mature:** 458
- **Cold Start:** 2
- **Insufficient for symbol-specific fine-tuning:** 13
- **Current MLP PPO ready:** 454
- **Current production Insufficient History:** 19

The count difference is intentional. Four symbols have at least 126 usable
observations but do not meet the existing 252-row MLP production contract.

## Selected-symbol metadata

The selected-symbol panel now shows, without loading TEST:

- symbol;
- company name;
- sector;
- canonical usable-observation count;
- future history class;
- current MLP PPO readiness;
- future intended training route;
- TRAIN, VALIDATION, and metadata-only sealed TEST boundaries;
- observation shape;
- feature version;
- RL contract version; and
- environment version.

Routes are fixed presentation of the approved research policy:

| Class | Displayed route |
|---|---|
| Mature | Eligible for recurrent symbol fine-tuning; sector pretraining planned. |
| Cold Start | Sector-pretrained transfer route planned; use only real company history. |
| Insufficient | Not enough own history for safe symbol-specific fine-tuning. |

All ten current pilot symbols—OGDC, UBL, FFC, PPL, MEBL, LUCK, HUBC, PSO,
MLCF, and TRG—classify as **Mature**.

## Page ownership decisions

| Previous block | Decision | Result |
|---|---|---|
| RL/PPO readiness | **Keep** | Current contract-compatible count plus explicit future-class summary |
| Model architecture/status | **Keep/add read-only summary** | Current PPO/MlpPolicy/single-symbol lifecycle versus planned sector transfer |
| PPO training configuration | **Keep** | Explicit one-symbol timesteps, seed, and device controls |
| Selected-symbol data | **Keep/enhance read-only** | Company, sector, usable rows, history class, route, and partition metadata |
| In-memory training | **Keep** | Explicit TRAIN-only button |
| Validation comparison | **Keep** | Explicit VALIDATION-only action and baselines |
| Candidate persistence | **Keep** | Validation-pass candidate only; no promotion |
| Model registry/history | **Keep** | Full candidate/version history |
| Minimum usable rows widget | **Remove duplicate** | No user-editable second readiness authority |
| Build/Refresh AI Datasets | **Remove from page** | Backend preserved |
| Validate AI Datasets | **Remove from page** | Backend preserved |
| Create/Refresh Chronological Splits | **Remove from page** | Backend preserved |
| Generic bulk symbol selection/preparation | **Remove from page** | Selection/preparation helpers remain available to future universe tooling |
| Prepare/Train Master Model | **Remove legacy block** | Avoids conflict with future sector/generalized architecture |

No clean existing page owns the complete manual data-product lifecycle.
Historical Backfill is a recovery workflow, Automation should own schedules,
Dataset Explorer should remain read-only, and Fetch Data owns one-off
collection. Therefore Milestone 6A did not move these controls to an arbitrary
page. The canonical backend capabilities remain in:

- `data_pipeline.src.data_products.rebuild_data_products`;
- `feature_engineering.dataset_builder`;
- `feature_engineering.splitting`; and
- their CLI/test surfaces.

The recommended future owner remains a dedicated **Data Operations** page or a
single clearly scoped Data Operations section, implemented only after its
dependency order and side-effect contract are tested.

## Canonical readiness ownership

The page now follows one direction of data flow:

```text
AI_MINIMUM_USABLE_ROWS (252)
    -> build_model_readiness_table
    -> processed-dataset readiness and partition bounds
    -> build_ready_symbol_catalog
    -> current compatible rl_partition_v1 intersection
    -> Training-ready selector and count
```

Current raw-history end dates remain freshness indicators. Processed bounds
remain the comparison source for persisted RL contracts. Future history class
reads only the resulting usable-observation count and cannot mutate any layer.

## Streamlit smoke verification

A no-click `streamlit.testing.v1.AppTest` run against
`app_pages/6_Training_and_Models.py` reported:

- exceptions: 0;
- UI errors: 0;
- current MLP PPO ready: 454;
- current Insufficient History: 19;
- registered models: 0;
- future Mature / Cold Start / Insufficient: 458 / 2 / 13;
- selected symbol: OGDC;
- company: Oil & Gas Development Company Limited;
- sector: Oil & Gas Exploration Companies;
- usable observations: 2,437;
- history class: Mature;
- current MLP PPO ready: Yes;
- TEST seal displayed: Yes; and
- visible action buttons: Validate selected TRAIN environment, Train PPO
  candidate, and Evaluate on validation.

No action button was clicked. The legacy threshold and dataset-management
buttons were absent.

## Remaining limitations

- RecurrentPPO, sector universes, pooled scalers, transfer learning, and registry
  v3 are not implemented.
- The current MLP contract remains 252-row `rl_partition_v1` and should not be
  interpreted as the future six-month policy.
- Manual dataset-operation controls still require a separately approved owner;
  backend capability is preserved in the meantime.
- The readiness calculation is intentionally expensive and cached using source
  artifact fingerprints; future Data Operations must clear/invalidate those
  caches after writes.
- Streamlit training remains synchronous and overlap protection remains
  session-local.

## Safety and integrity

- Complete pytest suite: **455 passed, 1 skipped** from 456 collected tests in
  12.26 seconds. The skip is the existing hardware-gated MPS smoke test.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: no broken requirements. The local pip cache
  emitted a non-fatal permissions warning and was disabled.
- No Streamlit process was already listening on ports 850*; UI verification
  used the no-click Streamlit test harness.
- No live HTTP calls were made.
- No model training or validation action was started.
- TEST frames were not opened by the page smoke run.
- No dependency was installed.
- No model registry entry was created or modified.
- Registry SHA-256 remained
  `e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a`.
- No raw, processed, split, or model artifact was changed.
- No commit was created.
