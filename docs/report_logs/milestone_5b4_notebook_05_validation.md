# Milestone 5B-4 — Notebook 05 PPO training and validation

Execution date: **2026-08-09 (Asia/Karachi)**  
Status: **Completed successfully**

## Objective

Notebook 05 demonstrates the complete production single-symbol PPO research
workflow without becoming a second training implementation:

1. inspect the canonical `rl_partition_v1` artifacts;
2. validate `single_symbol_env_v1`;
3. train one in-memory PPO model from TRAIN only;
4. compare the fixed model with three baselines on VALIDATION only;
5. apply `ppo_validation_criteria_v1` honestly;
6. prove the `model_registry_v2` bundle round-trip in a temporary directory;
7. prove that TEST and production model locations remain untouched.

The executed artifact is
[`notebooks/05_ppo_training_and_evaluation.ipynb`](../../notebooks/05_ppo_training_and_evaluation.ipynb).
All 18 code cells executed through the project `.venv` kernel, all three
Matplotlib figures are embedded, and the notebook contains no error output.

## Environment and experiment configuration

| Setting | Executed value |
|---|---:|
| Symbol | `MCB` |
| Seed | 42 |
| Requested / resolved / actual device | `cpu` / `cpu` / `cpu` |
| Notebook demonstration timesteps | 10,240 |
| Smoke setting shown, not executed | 512 |
| Production default shown, not executed | 100,000 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.13.0 |
| Gymnasium | 1.3.0 |
| Matplotlib | 3.11.1 |

The existing CPU default was retained. The prior controlled device benchmark
found CPU approximately 6.81 times faster than MPS for this single-environment
MLP PPO workload. The notebook verifies MPS availability but never switches to
it automatically.

The PPO research hyperparameters remain the unchanged
`ppo_single_symbol_v1` values: `MlpPolicy`, learning rate `3e-4`, rollout length
`512`, batch size `64`, 10 epochs, gamma `0.99`, GAE lambda `0.95`, clip range
`0.20`, entropy coefficient `0.01`, value coefficient `0.50`, and maximum
gradient norm `0.50`.

## Data contract and leakage safeguards

| Partition | Rows | Start | End | Frame loaded |
|---|---:|---|---|---|
| TRAIN | 1,704 | 2016-10-06 | 2023-08-23 | Yes |
| VALIDATION | 365 | 2023-08-24 | 2025-02-12 | Yes |
| TEST | 366 | 2025-02-13 | 2026-08-05 | **No — metadata only** |

The canonical access trace was exactly `train, validation, train, validation`:
the first pair supports notebook inspection, and the second pair is the
production trainer/evaluator workflow. A runtime read guard covered both
`test.csv` and `test_rl.csv`; it recorded zero TEST read attempts.

The notebook displays the ordered 12 market observation features, five dynamic
portfolio features, and resulting `(17,)` observation shape. The canonical
loader verified real, unscaled OHLCV execution fields against the raw split,
while observation features came from the scaler fitted on TRAIN. Scaler
provenance reported `fit_partition=train` and 1,704 fitting rows.

The production environment validation returned `valid=True`, `Discrete(3)`,
observation shape `(17,)`, and no errors.

## Training result and diagnostics

The production `train_single_symbol()` API completed 10,240 requested and
actual timesteps in **5.47109 seconds**. It returned an in-memory model only;
no output directory was supplied.

Stable-Baselines3 final completed-update diagnostics were captured through the
new structured trainer result rather than notebook-side SB3 introspection:

| Diagnostic | Value |
|---|---:|
| Timesteps | 10,240 |
| Updates | 200 |
| Approximate KL | 0.007383 |
| Clip fraction | 0.048828 |
| Entropy loss | -0.925302 |
| Explained variance | 0.280859 |
| Policy-gradient loss | -0.011729 |
| Value loss | 0.000668 |
| Learning rate | 0.000300 |

These are optimization diagnostics, not profitability measures. The reusable
instrumentation copies only a finite, explicit allowlist from the final SB3
logger update; unavailable values remain `None` rather than being fabricated.

## VALIDATION methodology and results

`compare_candidate_on_validation()` evaluated deterministic PPO, Buy & Hold,
Always Hold, and fixed-seed Random on fresh environments sharing the complete
365-row VALIDATION frame, PKR 1,000,000 initial capital, 0.10% commission,
0.05% slippage, and identical environment semantics. PPO parameters and
training timesteps were unchanged by evaluation.

| Metric | PPO | Buy & Hold | Always Hold | Random |
|---|---:|---:|---:|---:|
| Final portfolio value | PKR 1,298,314.44 | PKR 1,975,057.51 | PKR 1,000,000.00 | PKR 1,706,240.77 |
| Total return | 29.83% | 97.51% | 0.00% | 70.62% |
| Annualized return | 19.81% | 60.19% | 0.00% | 44.76% |
| Annualized volatility | 21.52% | 30.51% | 0.00% | 24.24% |
| Sharpe | 0.947 | 1.697 | Not Available | 1.647 |
| Sortino | 1.531 | 2.704 | Not Available | 2.824 |
| Maximum drawdown | 10.61% | 13.98% | 0.00% | 12.52% |
| Executed trades | 37 | 1 | 0 | 124 |
| Transaction costs | PKR 62,929.71 | PKR 1,498.20 | PKR 0.00 | PKR 245,590.54 |
| Exposure | 65.93% | 100.00% | 0.00% | 48.35% |
| Realized P&L | PKR 325,020.97 | PKR 0.00 | PKR 0.00 | PKR 706,240.77 |
| Unrealized P&L | PKR -26,706.52 | PKR 975,057.51 | PKR 0.00 | PKR 0.00 |

PPO action counts were 12 Hold, 235 Buy, and 117 Sell decisions. These produced
37 executed transactions, 18 completed sells, and 14 profitable completed
sells. Action frequency is presented descriptively and is not treated as a
performance claim.

The notebook contains one four-strategy portfolio-value chart, one separate
PPO-versus-Buy-and-Hold drawdown chart, and one compact action-frequency chart.

## Candidate decision

The honest result was **`validation_fail`** under
`ppo_validation_criteria_v1`. PPO trailed Buy & Hold by 67.67 percentage points
of total return and by 0.7505 Sharpe units. PPO's maximum drawdown was 3.37
percentage points lower, but the model failed the return-advantage and
Sharpe-advantage criteria.

This is not a pipeline failure. Training, deterministic evaluation, metrics,
and integrity checks all completed. It means this one model was economically
inferior to the critical passive benchmark under the predeclared criteria.
Likewise, a future `validation_pass` would identify only a candidate; it would
not constitute production promotion.

Always Hold correctly produced undefined Sharpe and Sortino values because its
return path had zero volatility/downside deviation. Those values are displayed
as Not Available.

## Temporary persistence demonstration

The notebook called `persist_developer_smoke_bundle()` with explicit registry
and saved-model roots inside an operating-system `TemporaryDirectory` outside
the project tree. The honest validation failure was stored as:

- model ID `ppo-symbol-MCB-v0001`;
- model status `experiment`;
- validation status `validation_fail`;
- promotion status `not_eligible`.

The complete immutable bundle was created, its manifest hashes and metadata
were verified, and `load_persisted_ppo()` reloaded the exact model ID. One
deterministic prediction succeeded with a valid action. The temporary directory
was then removed successfully. No temporary registry or model survived the
cell.

The verified bundle metadata recorded TEST as `sealed_not_evaluated`,
`test_evaluation_performed=false`, and validation-only metric payloads.

## TEST sealing and data integrity

The final notebook assertions reported:

- `FINAL TEST SET: UNTOUCHED`;
- zero TEST CSV read attempts;
- no TEST metrics or TEST evaluation result;
- production `model_registry.csv` unchanged byte-for-byte;
- production saved-model directory unchanged;
- production `data/models` tree unchanged;
- no production candidate created or promoted.

No live HTTP requests occurred. Raw/backfill data and market-intelligence logic
were outside this workflow.

## Verification

- Complete test suite: **362 passed, 1 hardware-gated MPS test skipped** in
  8.14 seconds. Notebook 05 itself used CPU and executed successfully.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: no broken requirements.
- Executed notebook: 31 cells, 18/18 code cells executed, zero error outputs,
  and three embedded Matplotlib figures.
- Production registry: header only; no model records added.
- Production saved-model directories: only their existing `.gitkeep` files.
- No commit was created automatically.

## Limitations and next milestone

This is one symbol, one seed, one fixed configuration, and one historical
market regime. It demonstrates methodology and integration, not general
profitability. Random's relatively strong result illustrates why a single
path can reward chance. Positive PPO return alone is insufficient when a
passive benchmark is materially stronger after risk and costs.

The next research milestone should define a controlled multi-symbol,
multi-seed PPO pilot with predeclared aggregation and failure reporting. It
should continue using TRAIN for learning and VALIDATION for selection while
keeping TEST sealed for one later, explicitly authorized final evaluation.
