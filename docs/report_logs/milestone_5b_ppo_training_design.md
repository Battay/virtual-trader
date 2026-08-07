# Milestone 5B PPO training architecture audit and design

Audit date: **2026-08-08 (Asia/Karachi)**  
Repository commit audited: **`4d708bd9c1d47cef541039d171dfd341e421bd57`**  
Scope: read-only architecture, local artifacts, installed dependencies, and future workflow design. No model was trained or registered.

## Executive assessment

Milestone 5A provides a well-tested, deterministic, Gymnasium-compatible single-symbol environment, causal execution timing, baseline policies, core episode metrics, chronological data artifacts, readiness selection, and most of an immutable model-registry schema. Milestone 5B does **not** yet have a trainer, PPO dependency, PyTorch runtime, PPO evaluator, candidate-selection protocol, atomic model-bundle persistence, interruption handling, or enabled UI workflow.

Two contract issues must be resolved before the first training run:

1. **Scaled execution-price conflict:** all 24 `FEATURE_COLUMNS`, including raw open/high/low/close/volume, are standardized in `*_scaled.csv`. Environment v1 uses `open` and `close` as currency prices and requires prices to be positive. MCB `train_scaled.csv`, for example, contains negative standardized opens/closes. It cannot safely be supplied directly to the environment.
2. **Registry range semantics:** `complete_history_training_metadata()` currently records the complete dataset's first/last dates as `training_data_start/end`, while row counts come from the training partition. This would misstate what PPO actually saw. `create_model_record()` also defaults `environment_version` to the stale value `pending_3b`, not `single_symbol_env_v1`.

Recommendation: begin with a **5B-0 contract-hardening step**, then implement a CPU-first, one-symbol-at-a-time PPO trainer. Do not expose training in Streamlit or use the test partition until these gates pass offline tests.

## 1. Existing architecture

### Implemented

| Area | Existing component | Assessment |
|---|---|---|
| Environment | `SingleSymbolTradingEnv` | Long-only, whole-share, all-in/all-out; action `Discrete(3)` = Hold/Buy/Sell; observation `(17,)`; deterministic accounting |
| Causality | Observation at *t*, execution at *t+1* open, valuation at *t+1* close | Explicit and extensively tested; next open is not observed |
| Costs/reward | Commission, slippage, drawdown and invalid-action penalties | Configured centrally; finite log-growth reward; exact PSX fee schedule remains a limitation |
| Validation | `prepare_single_symbol_data`, Gymnasium `check_env`, accounting and finite-value checks | Production MCB validation: `valid=True`, shape `(17,)`, no errors |
| Baselines | Always Hold, Buy & Hold, fixed-seed Random | Complete deterministic episode runner exists |
| Metrics | Final value, total return, maximum drawdown, trades, transaction costs, daily returns, Sharpe, annualized volatility | Solid initial evaluator, incomplete for 5B |
| Data | Per-symbol processed CSVs; chronological raw and scaled train/validation/test partitions; scaler + JSON metadata | 454 symbol split/scaler sets plus master set |
| Scaling | `StandardScaler` fit on training only, then applied to validation/test | Leakage-safe fitting is tested; environment integration is unresolved |
| Readiness | Exact processed/split readiness and symbol selectors | 454 active ordinary equities ready |
| Registry | Atomic CSV append, immutable version IDs, deterministic artifact paths, latest-version views | Useful foundation; no trainer lifecycle/payload implementation |
| UI | Multi-symbol selection, eligibility filters, dataset preparation, environment validation, registry history | Training buttons deliberately disabled; page load performs no training |
| Notebook | `04_rl_environment_design.ipynb` | Executed on 2,435 real MCB rows; environment and baselines validated |

### Missing

- Stable-Baselines3 and PyTorch dependencies/runtime.
- A hybrid execution/observation data adapter and strict split loader.
- PPO configuration schema and validation.
- Trainer core, callbacks, progress events, cancellation/checkpoint policy, and CLI/service boundary.
- PPO-policy episode evaluator and candidate-selection rules.
- Annualized return, Sortino, profitable closed-trade/win-loss statistics, benchmark deltas, and serializable metric schemas.
- Atomic staging/promotion of model, scaler reference/copy, metrics, configuration, and manifest.
- Correct training/validation/test registry ranges and `single_symbol_env_v1` registration.
- Structured PPO configuration, dependency versions, observation schema, artifact hashes, and source commit metadata.
- Notebook 05 and enabled, explicit Streamlit training flow.
- Pilot execution; no PPO model currently exists.

`reinforcement_learning/training/__init__.py` is only a placeholder and still refers to “Milestone 3B.”

## 2. Training data contract

### Artifact paths per symbol

For symbol `{SYMBOL}`:

| Purpose | Path |
|---|---|
| Complete processed history | `data/processed/symbols/{SYMBOL}.csv` |
| Raw-price training partition | `data/processed/splits/symbols/{SYMBOL}/train.csv` |
| Raw-price validation partition | `data/processed/splits/symbols/{SYMBOL}/validation.csv` |
| Raw-price test partition | `data/processed/splits/symbols/{SYMBOL}/test.csv` |
| Standardized training features | `data/processed/splits/symbols/{SYMBOL}/train_scaled.csv` |
| Standardized validation features | `data/processed/splits/symbols/{SYMBOL}/validation_scaled.csv` |
| Standardized test features | `data/processed/splits/symbols/{SYMBOL}/test_scaled.csv` |
| Training-fitted scaler | `data/processed/splits/symbols/{SYMBOL}/standard_scaler.joblib` |
| Transparent scaler metadata | `data/processed/splits/symbols/{SYMBOL}/standard_scaler.json` |
| Split contract | `data/processed/splits/symbols/{SYMBOL}/metadata.json` |

### Feature and observation schema

The scaler currently covers 24 feature columns:

`open, high, low, close, volume, simple_return, log_return, high_low_range, open_close_return, rolling_volatility_20, sma_20, sma_50, ema_20, ema_50, rsi_14, macd, macd_signal, macd_histogram, bollinger_middle, bollinger_upper, bollinger_lower, atr_14, obv, volume_ma_20`.

Environment v1 observes 12 market features:

`simple_return, log_return, high_low_range, open_close_return, rolling_volatility_20, rsi_14, macd, macd_signal, macd_histogram, atr_14, obv, volume_ma_20`.

It appends five live portfolio-state fields:

`portfolio_cash_ratio, portfolio_position_value_ratio, portfolio_position_indicator, portfolio_unrealized_return_ratio, portfolio_current_drawdown`.

Total observation shape: **17**.

Default environment configuration:

- environment version: `single_symbol_env_v1`;
- initial cash: PKR 1,000,000;
- commission: 0.10%;
- slippage: 0.05%;
- transaction-cost penalty weight: 0;
- drawdown penalty weight: 0.1;
- invalid-action penalty: 0.0001;
- maximum episode steps: full partition.

### Required 5B loader contract

The trainer should create a **hybrid frame** for each partition:

1. Load unscaled `train.csv`, `validation.csv`, or `test.csv` for identity, dates, executable OHLCV, accounting, and provenance.
2. Load the corresponding scaled partition.
3. Require identical row count and exact `(symbol, date)` identity/order between raw and scaled frames.
4. Copy only `SingleSymbolEnvConfig.feature_columns` from the scaled frame into the raw frame. Retain raw `open`, `high`, `low`, `close`, and `volume` for execution and validation.
5. Verify split metadata feature version, dataframe feature version, configured feature list, scaler feature order, and artifact hashes.
6. Reject missing, stale, cross-symbol, reordered, duplicated, non-finite, or mismatched artifacts. Never silently load complete history or synthetic data as fallback.

This uses the existing training-fitted scaler while keeping observations standardized and prices economically meaningful. A cleaner longer-term schema could store observation-only scaled columns separately, but 5B must not reinterpret standardized prices as money.

### Leakage guarantees

- Split dates are globally ordered and disjoint: training ends before validation begins; validation ends before testing begins.
- `StandardScaler.fit()` is called only on training rows. Validation and test use `transform()` from that fitted scaler; tests assert this behavior.
- Indicators were calculated causally before chronological splitting.
- PPO learns only from the training hybrid frame.
- Candidate configuration/seed selection and early stopping use validation only.
- Test is loaded/evaluated only after a candidate is frozen and selected. Test metrics must never feed configuration, seed, stopping, or model selection.
- Test access should be a separate explicit method/event and be recorded in the manifest.

## 3. MCB production contract example

Feature version for all current artifacts: **`psx-4a-126450ec6355`**.

| Partition | Rows | Dates | Use |
|---|---:|---|---|
| Train | 1,704 | 2016-10-06 through 2023-08-23 | PPO learning only |
| Validation | 365 | 2023-08-24 through 2025-02-12 | Candidate evaluation/selection |
| Test | 366 | 2025-02-13 through 2026-08-05 | One final locked-model evaluation |

The MCB scaler metadata records 1,704 fitting rows. Raw training opens range from 107.01 to 264.10; scaled opens range approximately −1.97 to 2.43, proving why scaled execution columns cannot be used directly.

## 4. Proposed Milestone 5B flow

1. Resolve one selected symbol from the readiness report; require `Ready` and ordinary-equity eligibility.
2. Load and validate split metadata, raw/scaled partition pairs, scaler metadata, and feature/environment versions.
3. Build the hybrid training frame and instantiate `SingleSymbolTradingEnv`.
4. Run project validation and SB3 `check_env`; assert observation `(17,)`, action `Discrete(3)`, finite rewards, and deterministic reset.
5. Apply deterministic seeds to Python, NumPy, Gymnasium/action space, PyTorch, and PPO. Record determinism settings and device.
6. Instantiate PPO with the frozen pilot configuration.
7. Train only on the training partition; emit structured progress and interruption-safe run logs.
8. Evaluate the candidate deterministically on validation; independently run Buy & Hold and the fixed-seed Random baseline on the same hybrid validation partition and environment configuration.
9. Select/reject the candidate using a predeclared validation rule. Do not inspect test metrics.
10. Stage the selected candidate bundle atomically without overwriting any prior version.
11. Freeze the candidate/configuration/seed, then perform exactly one final deterministic evaluation on test, with the same baselines.
12. Persist manifest, PPO configuration, validation/test metrics, environment/scaler references, hashes, logs, and model artifact.
13. Append the final immutable registry record only after the complete artifact bundle is durable. A per-symbol failure must not stop other requested symbols.

For the first pilot, use one frozen configuration and seed across all symbols. Treat results as feasibility evidence, not tuned production performance.

## 5. PPO dependencies

Local environment findings:

| Package | Current state |
|---|---|
| Stable-Baselines3 | Not installed; absent from `requirements.txt` |
| PyTorch | Not installed; absent from `requirements.txt` |
| Gymnasium | 1.3.0; requirement is `gymnasium>=1.0,<2.0` |
| Python | 3.11.9 |

The current official SB3 release information reports that **SB3 2.9.0** supports Gymnasium 1.3.0 and requires PyTorch 2.8 or newer. PPO supports discrete action spaces. Sources: [official SB3 releases](https://github.com/DLR-RM/stable-baselines3/releases), [official SB3 repository/install guidance](https://github.com/DLR-RM/stable-baselines3).

Proposed dependency change for implementation, after a dedicated install/compatibility PR:

```text
stable-baselines3==2.9.0
torch>=2.8,<3.0
```

Keep the existing Gymnasium range. Use base SB3 rather than `[extra]` initially; TensorBoard can be added explicitly if required, avoiding unrelated Atari/OpenCV dependencies. Generate and preserve a resolved lock/freeze for reproducibility. No package was installed during this audit.

## 6. Conservative pilot PPO configuration

| Parameter | Proposal | Basis |
|---|---:|---|
| Policy | `MlpPolicy` | Correct for flat 17-value `Box` observation |
| Learning rate | `3e-4` | SB3 PPO default; conservative first baseline |
| `n_steps` | 512 | Four-to-eight minibatches and several updates per ~1,700-row episode; lower memory than 2,048 |
| Batch size | 64 | Exactly divides 512; stable CPU minibatch size |
| `n_epochs` | 10 | SB3 default |
| Gamma | 0.99 | SB3 default; long-horizon daily reward |
| GAE lambda | 0.95 | SB3 default bias/variance balance |
| Clip range | 0.20 | SB3 default |
| Entropy coefficient | 0.01 | Small project-specific exploration pressure for three discrete actions |
| Value coefficient | 0.50 | SB3 default |
| Maximum gradient norm | 0.50 | SB3 default |
| Seed | 42 | Fixed reproducible pilot seed |
| Total timesteps | 100,000 per symbol | Feasibility-scale run; about 59 passes through a 1,704-row train partition |
| Device | `cpu` | Reproducible and universally available |

Also retain default policy network initially rather than adding architecture tuning. Log the effective SB3 configuration, not merely user overrides. This is a frozen pilot proposal, not a tuned optimum. Any future sweep must use validation only and record its search space before execution.

## 7. Pilot universe

All 20 previously audited liquid candidates are currently ready and have processed datasets, split metadata, and scaler artifacts:

`MLCF, OGDC, DGKC, UBL, PPL, NBP, FFC, PSO, LUCK, BOP, HUBC, MEBL, ENGROH, HBL, MARI, ATRL, PTC, NRL, FCCL, TRG`.

Recommended staged pilot:

- **Smoke/contract stage (3):** OGDC, UBL, FFC — distinct sectors and full 1,704-row training partitions.
- **First comparative pilot (10):** OGDC, UBL, FFC, PPL, MEBL, LUCK, HUBC, PSO, MLCF, TRG.
- **Expanded pilot (20):** all listed candidates only after the first ten complete without contract, persistence, or evaluation errors.

Nineteen candidates have roughly 1,690–1,704 train rows and 362–366 rows in each holdout. LUCK has 1,701/364/365; MEBL 1,691/362/364; ENGROH 1,690/362/363. Exact paths follow the template in section 2 and every artifact reports feature version `psx-4a-126450ec6355`.

## 8. Evaluation design

| Metric | Existing? | 5B definition/action |
|---|---|---|
| Initial/final portfolio value | Yes | Retain |
| Total return | Yes | Retain |
| Annualized return | No | Add only when period length supports it; annualize over observed trading transitions |
| Sharpe ratio | Yes | Retain zero risk-free convention; document 252-day annualization |
| Sortino ratio | No | Add downside-deviation implementation and zero-downside handling |
| Annualized volatility | Yes | Retain |
| Maximum drawdown | Yes | Retain |
| Number of trades | Yes | Retain executed buy/sell transitions |
| Transaction costs | Yes | Retain total and add as percentage of initial/final value |
| Profitable closed trades / win rate / average win-loss | No | Add by pairing completed buy/sell cycles; report open final positions separately |
| Invalid-action count | History has enough information only indirectly | Persist from episode `info`/history explicitly in 5B |
| Buy & Hold comparison | Baseline exists | Run same partition/config; add absolute and percentage-point return delta |
| Random comparison | Fixed-seed baseline exists | Use predeclared seed 42; optionally report a fixed multi-seed distribution without selecting seeds after results |

Create a unified evaluator that accepts any deterministic action provider, uses `model.predict(..., deterministic=True)` for PPO, and returns a JSON-safe result plus transition history. Validation and test output schemas must be identical and clearly labelled. Never rank models by test output.

## 9. Model registry and artifact design

### Existing conventions to preserve

- IDs such as `ppo-symbol-OGDC-v0001`.
- Per-symbol paths under `reinforcement_learning/saved_models/symbol_models/{SYMBOL}/vNNNN/`.
- Immutable version increments and atomic registry replacement.
- Existing algorithm, environment/feature versions, partition dates/rows, paths, duration, seed, status, and retraining fields.

### Required persisted model bundle

Each version directory should contain:

- `ppo_model.zip`;
- scaler copy or immutable verified reference;
- `metrics.json` with separately nested validation/test/baseline results;
- `ppo_config.json` with complete effective configuration;
- `environment_config.json`;
- `manifest.json` with symbol, versions, observation columns, split paths/ranges/rows, SHA-256 hashes, dependency versions, source commit, timestamps, device and determinism flags;
- optional structured training log/checkpoint files that cannot be mistaken for a selected final model.

The registry should add structured references/hashes for PPO config, environment config, manifest, validation metrics, test metrics, source commit, observation-schema hash, split/scaler hash, and dependency versions. Avoid packing machine-critical metadata into free-form `notes`.

Correct before use:

- set `environment_version=single_symbol_env_v1`, never `pending_3b`;
- derive `training_data_start/end` from `metadata.json["training"]`, not complete history;
- keep `dataset_latest_date` separately as the complete available cutoff;
- distinguish candidate/selected/tested/failed lifecycle without mutating or overwriting prior versions.

## 10. Training safeguards

- Test-access gate enforced in code, not only documentation.
- One immutable `TrainingRequest` containing symbol, config, seed, feature/environment versions and artifact hashes.
- Seed Python/NumPy/Gymnasium/PyTorch/SB3 and record deterministic-algorithm settings. Reproducibility should be described as best effort across platform/library versions.
- Refuse an existing version directory; stage into a same-filesystem temporary directory, fsync where practical, then atomically rename/promote.
- Append registry only after all selected artifacts validate; failed/interrupted attempts go to a separate atomic run log or an explicit failed version with no trained-model claim.
- Catch `KeyboardInterrupt`/termination at safe callback boundaries, close environments, preserve diagnostics, and never publish a partial candidate.
- Isolate every symbol in batch orchestration; aggregate successes/failures without stopping later symbols.
- CPU is the required baseline. Consider MPS later only behind an explicit device option after numerical/reproducibility tests; never silently change device.
- No complete-history, synthetic, random, or alternate-symbol fallback when an artifact is absent or invalid.
- Validate model reload and deterministic replay before registry publication.
- Use advisory/run locks so two UI/CLI processes cannot allocate the same next version.
- Log progress by symbol, timesteps, elapsed time, validation stage, artifact promotion and final registry status without logging per-step noise.

## 11. Future Streamlit workflow

Keep business logic out of the page. The page should call the same production service/CLI used by tests and Notebook 05.

1. Reuse the current readiness table and canonical multi-symbol selection; restrict training submission to `Ready` symbols.
2. Add explicit pilot presets (Smoke 3, Pilot 10, Expanded 20) that populate selection without erasing manually selected hidden rows.
3. Put PPO configuration inputs in a `st.form` so edits do not launch work or repeatedly recompute readiness. Show defaults, effective values, estimated timesteps and CPU warning.
4. Require an explicit **Train selected models** submit button plus a confirmation summary. Loading/rerunning the page must never train.
5. Submit a durable background job or external process; do not run a multi-symbol 100k-step loop inside the Streamlit rerun thread.
6. Show job-level and per-symbol states, progress, cancellation request, failures, and durable log paths. Poll status in an isolated fragment with bounded caching.
7. Display validation metrics and baseline deltas first. Expose final test evaluation only for a selected frozen candidate with an explicit confirmation.
8. Reuse native responsive containers/metrics and dataframes; no custom HTML is needed. Keep model history and latest status separate from active jobs.
9. Refresh registry/readiness caches only after durable state transitions.

The existing disabled training buttons are correct for the current repository and should remain disabled until the production backend and dependency gates exist.

## 12. Notebook 05 proposal

`notebooks/05_ppo_training_and_evaluation.ipynb` should validate production modules, not contain a second trainer implementation.

1. Scope, leakage rules, warnings, source commit and dependency versions.
2. Select one configurable ready symbol (default MCB or OGDC) and display split/scaler manifest.
3. Demonstrate the raw/scaled execution-price conflict and build the production hybrid frames.
4. Validate train and validation environments; display observation/action schemas.
5. Show frozen PPO configuration and deterministic seed/device setup.
6. Invoke the production trainer for a deliberately small smoke budget in a temporary notebook output directory, never the live registry by default.
7. Plot training diagnostics from structured logs.
8. Evaluate deterministic PPO, Buy & Hold and fixed-seed Random on validation.
9. Display total/annualized return, Sharpe, Sortino, drawdown, trades, costs and portfolio curves.
10. Demonstrate save/reload and deterministic evaluation equivalence.
11. Keep the test cell gated off by default; require an explicit `FINAL_TEST_EVALUATION = False` switch and explain that it is only for a frozen selected candidate.
12. Summarize limitations and production-readiness checks.

Expected outputs: validated artifact manifest, environment result, training summary, validation comparison table/plot, reload check, and no registry mutation unless explicitly routed through a later production workflow.

## 13. Recommended implementation sequence

1. **5B-0 — Data-contract hardening:** hybrid raw-price/scaled-observation loader; identity/hash/version checks; correct registry range/environment semantics; offline tests.
2. **5B-1 — PPO dependencies and trainer core:** pinned SB3/PyTorch, typed configuration/request/result, deterministic CPU trainer, callbacks and CLI; temporary outputs only.
3. **5B-2 — Evaluation and baselines:** unified PPO/baseline runner; missing risk/trade metrics; validation-only selection and explicit test gate.
4. **5B-3 — Atomic persistence and registry:** staged immutable bundle, reload validation, locks, run logs, failure isolation, expanded metadata and final registry append.
5. **5B-4 — Notebook 05 validation:** real production split smoke test without live registry mutation.
6. **5B-5 — Streamlit integration:** form-based submission to durable jobs, progress/cancel/status, validation comparison and model history; never train on page load.
7. **5B-6 — Pilot execution:** Smoke 3, then Pilot 10, then optional Expanded 20; one frozen configuration/seed; test only after validation selection.

## 14. Audit conclusion

The repository is ready to begin **implementation**, not training. Environment v1 and the leakage-safe split foundation are credible, and all 20 pilot candidates are ready. The scaled execution-price conflict, incomplete metrics/evaluator, absent SB3/PyTorch runtime, and incomplete persistence/registry semantics are hard blockers. Resolve them in 5B-0 through 5B-3, validate through Notebook 05, then enable UI orchestration and run the staged pilot.

This audit created only this report. It did not modify production code, dependencies, data artifacts, saved models, or `model_registry.csv`; it installed nothing, trained nothing, and made no commit.
