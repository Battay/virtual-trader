# Milestone 6E.1 — Commercial Banks sector-policy collapse audit

Audit timestamp: 2026-08-12 02:40:55 PKT  
Mode: read-only diagnosis; no training, tuning, model loading, persistence, or
dataset mutation  
Sector: COMMERCIAL BANKS  
Universe hash:
`589485c8adfe6170a6c2391687202ac3a287de9eb30737a2cb1f57a34f111e5b`

## Executive finding

The multi-symbol environment, partition controls, recurrent resets, and
optimizer numerics are technically healthy. The retained 6E policy is not a
healthy transfer parent. Of 6,652 deterministic VALIDATION actions, 6,014 were
Sell. Exactly 97 Sell actions executed while long and **5,917 (98.39% of Sell
selections; 88.95% of all actions) occurred while already flat**. The policy
therefore selected an invalid, penalized action for most of validation.

The strongest supported explanation is a combination of:

1. a reward/regime incentive toward low exposure: cash earns zero, while a
   counterfactual fully invested policy over the permitted TRAIN histories had
   negative aggregate reward and substantial incremental-drawdown penalties;
2. a cheap invalid-action path: Sell while flat is a no-op with only a
   `0.0001` penalty and no transaction cost;
3. limited and unequal symbol-specific exposure under full-partition episodes;
4. one shared, identity-free policy receiving differently normalized economic
   regimes.

The evidence does **not** show exploding losses, non-finite values, excessive
clipping, state leakage, or TEST leakage.

## Protocol deviation

The specified research protocol required exactly **25,000 requested
timesteps**. Notebook 07 explicitly configured 50,000 and SB3 completed 50,176
after rollout alignment: 25,176 steps, or 100.704%, above the required budget.
This is a protocol deviation even though the 50,000 value was fixed before its
own validation run. The 50k validation outcome is used here only to diagnose
the already-observed policy; it is not used to select or tune a replacement
configuration.

## Evidence and limits

The executed notebook retained aggregate action counts, per-symbol strategy
metrics, training exposure, final PPO logger diagnostics, and hashes. It did
not retain the in-memory model, transition histories, per-symbol action counts,
training rewards, or update-by-update action distributions. The temporary
model bundle was correctly deleted.

Consequently:

- aggregate Sell-while-flat is exactly reconstructible from the action and
  transaction invariants;
- per-symbol executed Buy/Sell counts are reconstructible;
- per-symbol selected Buy/Hold/Sell and per-symbol Sell-while-flat counts are
  **not reconstructible** without replaying the deleted model;
- actual PPO TRAIN reward distributions and policy progression are unavailable;
- the reward-scale calculations below are explicitly labelled deterministic
  counterfactual simulations on canonical TRAIN data, not reconstructed PPO
  behavior.

No retraining was performed to fill these evidence gaps.

## 1. Action semantics

The environment is long-only and all-in/all-out:

| State and action | Environment effect | Cost | Reward consequence |
|---|---|---|---|
| Flat + Hold | No state change | None | Portfolio growth 0; total reward 0 |
| Flat + Sell | Invalid no-op; `No shares are held to sell` | No commission/slippage | Invalid penalty `-0.0001`; total reward `-0.0001` |
| Flat + Buy | Buys maximum affordable whole shares | Commission 0.1% and slippage 0.05% | Costs reduce portfolio growth; no separate cost penalty because its weight is 0 |
| Long + Hold | Keeps position and marks it at next close | None | Log portfolio growth less any increase in drawdown penalty |
| Long + Buy | Normally invalid because the first Buy is all-in and leaves insufficient cash | None if invalid | Market P&L still occurs; additionally `-0.0001` |
| Long + Sell | Liquidates all shares | Commission and slippage | Costs enter portfolio growth; any incremental drawdown is additionally penalized |

An invalid action does not trade, charge transaction costs, or otherwise
change holdings. It does change reward by `-0.0001`, so invalid actions are not
free. However, flat Sell has a deterministic, low-variance loss that is small
relative to daily market tails.

### Exact aggregate VALIDATION reconstruction

- Hold selections: 372
- Buy selections: 266
- Sell selections: 6,014
- Executed transactions: 194
- Executed Buys: 97
- Executed Sells / Sell while long: 97
- Invalid or redundant Buy selections: 169
- Sell while flat: **5,917**
- Sell-while-flat / all Sell: **98.39%**
- Sell-while-flat / all actions: **88.95%**

Every validation episode began flat. Valid transactions must alternate Buy and
Sell, all per-symbol trade counts are even, and every Sell while long fully
liquidates the position. Therefore half of the 194 executed trades are valid
Sells; all other Sell selections occurred while flat.

### Per-bank action evidence

`NR` means the selected-action sequence was not retained. It is not estimated.

| Symbol | Sell selected | Sell while flat | Sell while long | Buy selected | Hold selected | Exposure | Executed transactions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABL | NR | NR | 8 | NR | NR | 8.19% | 16 |
| AKBL | NR | NR | 4 | NR | NR | 12.36% | 8 |
| BAFL | NR | NR | 4 | NR | NR | 1.65% | 8 |
| BAHL | NR | NR | 3 | NR | NR | 5.22% | 6 |
| BIPL | NR | NR | 4 | NR | NR | 1.99% | 8 |
| BML | NR | NR | 2 | NR | NR | 1.93% | 4 |
| BOK | NR | NR | 6 | NR | NR | 32.18% | 12 |
| BOP | NR | NR | 14 | NR | NR | 25.55% | 28 |
| FABL | NR | NR | 5 | NR | NR | 2.48% | 10 |
| HBL | NR | NR | 7 | NR | NR | 7.14% | 14 |
| HMB | NR | NR | 8 | NR | NR | 9.22% | 16 |
| JSBL | NR | NR | 2 | NR | NR | 1.42% | 4 |
| MCB | NR | NR | 10 | NR | NR | 8.52% | 20 |
| MEBL | NR | NR | 0 | NR | NR | 0.00% | 0 |
| NBP | NR | NR | 6 | NR | NR | 4.12% | 12 |
| SBL | NR | NR | 2 | NR | NR | 1.54% | 4 |
| SCBPL | NR | NR | 4 | NR | NR | 3.45% | 8 |
| SNBL | NR | NR | 6 | NR | NR | 4.66% | 12 |
| UBL | NR | NR | 2 | NR | NR | 0.55% | 4 |

The next research run must retain per-symbol counts partitioned by position
state and validity; a digest alone cannot support this audit after the model is
deleted.

## 2. Exact reward equation

For transition `t`:

```text
growth_t = log(portfolio_value_t / portfolio_value_(t-1))

transaction_penalty_t =
    transaction_cost_penalty_weight
    * transaction_cost_t / portfolio_value_(t-1)

drawdown_penalty_t =
    drawdown_penalty_weight
    * max(0, drawdown_t - drawdown_(t-1))

invalid_penalty_t = invalid_action_penalty if action is invalid else 0

reward_t = growth_t
           - transaction_penalty_t
           - drawdown_penalty_t
           - invalid_penalty_t
```

Configured values are:

- commission: `0.001`;
- slippage: `0.0005`;
- transaction-cost penalty weight: `0.0`;
- drawdown penalty weight: `0.1`;
- invalid-action penalty: `0.0001`;
- no explicit holding reward, cash reward, volatility penalty, or exposure
  target.

Commission and slippage already reduce cash/portfolio value and therefore enter
`growth_t`; the explicit transaction penalty is disabled, so costs are not
double-penalized. Drawdown is asymmetrical: new drawdown is penalized, while a
later reduction does not reimburse previous penalties. Flat cash has exactly
zero reward. The design therefore prefers cash whenever expected risk-adjusted
growth is negative, but it does not prefer invalid Sell over flat Hold.

## 3. Representative reward decomposition

The following read-only counterfactual uses the first MCB TRAIN transitions.

| Scenario/step | Growth | Cost penalty | Drawdown penalty | Invalid penalty | Total reward |
|---|---:|---:|---:|---:|---:|
| Flat Hold | 0 | 0 | 0 | 0 | 0 |
| Flat Sell | 0 | 0 | 0 | -0.000100 | -0.000100 |
| Buy | -0.020761 | 0 | -0.002055 | 0 | -0.022815 |
| Buy then Hold: Hold step | +0.004594 | 0 | 0 | 0 | +0.004594 |
| Buy then Sell: Sell step | -0.004411 | 0 | -0.000431 | 0 | -0.004842 |
| Long redundant Buy: second step | +0.004594 | 0 | 0 | -0.000100 | +0.004494 |

The first Buy traded 4,379 shares and incurred Rs 1,498.12 of commission plus
slippage; these costs are already reflected in growth. The subsequent Sell
incurred Rs 1,464.29.

Over the complete MCB TRAIN trajectory:

- stay cash: cumulative reward `0`;
- repeated Sell while flat: `-0.170300`;
- Buy then Hold: growth `-0.529590`, drawdown penalty `-0.648999`, total
  `-1.178588`;
- Buy then Sell, then stay cash: growth `-0.025172`, drawdown penalty
  `-0.002486`, total `-0.027657`.

These are scenario diagnostics, not the deleted model's rewards.

## 4. Reward-scale audit

Actual PPO TRAIN transition rewards were not persisted, so their distribution
cannot be recovered read-only. The following distributions apply fixed actions
to all 31,138 canonical TRAIN transitions (31,157 rows minus one transition per
symbol):

| Counterfactual policy | Mean | Median | Std | Min | Max | Positive | Negative | Zero | Cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stay cash | 0 | 0 | 0 | 0 | 0 | 0% | 0% | 100% | 0 |
| Repeated flat Sell | -0.000100 | -0.000100 | 0 | -0.000100 | -0.000100 | 0% | 100% | 0% | -3.113800 |
| Buy then Hold | -0.000658 | -0.000134 | 0.027188 | -0.231099 | +0.487923 | 45.90% | 50.12% | 3.99% | -20.500753 |
| Buy then Sell, then cash | -0.00000465 | 0 | 0.000855 | -0.067957 | +0.097540 | 0.05% | 0.07% | 99.88% | -0.144640 |

For Buy then Hold, cumulative portfolio growth was `-3.594617` and cumulative
incremental-drawdown penalty was `-16.906135`. The drawdown term contributed
82.47% of the negative aggregate reward and was 4.70 times the absolute growth
term. Repeated invalid Sell was much worse than Hold but substantially less
negative and less variable than full exposure in these TRAIN histories.

The 1st/5th/25th/75th/95th/99th percentiles of the counterfactual invested
reward were `-0.081279`, `-0.040713`, `-0.011651`, `0.010139`, `0.040727`, and
`0.074452`. The invalid penalty is comparable to the median invested reward in
magnitude, but tiny relative to the invested standard deviation and tails.

### Per-symbol counterfactual invested reward scale

| Symbol | TRAIN rows | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| ABL | 1,660 | -0.000611 | 0.016594 | -0.082744 | 0.072307 |
| AKBL | 1,703 | -0.000617 | 0.019015 | -0.176620 | 0.072193 |
| BAFL | 1,703 | -0.000252 | 0.019522 | -0.143248 | 0.072312 |
| BAHL | 1,704 | -0.000336 | 0.017846 | -0.154958 | 0.072320 |
| BIPL | 1,649 | -0.000581 | 0.026863 | -0.132480 | 0.117922 |
| BML | 1,703 | -0.000846 | 0.045915 | -0.220616 | 0.487923 |
| BOK | 1,484 | -0.000805 | 0.028610 | -0.139597 | 0.115512 |
| BOP | 1,704 | -0.001257 | 0.024595 | -0.138259 | 0.165132 |
| FABL | 1,701 | -0.000343 | 0.022849 | -0.149089 | 0.073276 |
| HBL | 1,704 | -0.000801 | 0.019400 | -0.081269 | 0.072311 |
| HMB | 1,675 | -0.000436 | 0.019643 | -0.160203 | 0.072217 |
| JSBL | 1,646 | -0.000841 | 0.033693 | -0.146650 | 0.177358 |
| MCB | 1,704 | -0.000692 | 0.015938 | -0.083115 | 0.064189 |
| MEBL | 1,691 | -0.000089 | 0.020107 | -0.136206 | 0.072344 |
| NBP | 1,704 | -0.001042 | 0.019960 | -0.170920 | 0.072185 |
| SBL | 1,217 | -0.001345 | 0.065460 | -0.231099 | 0.230985 |
| SCBPL | 1,495 | -0.000434 | 0.027440 | -0.187916 | 0.072459 |
| SNBL | 1,606 | -0.000793 | 0.024070 | -0.144661 | 0.118059 |
| UBL | 1,704 | -0.000579 | 0.019208 | -0.098798 | 0.072348 |

Invested reward standard deviation differed by 4.11 times between MCB and SBL.
This heterogeneity can make pooled gradients disproportionately reflect
high-volatility episodes even when episode selection probabilities are equal.

## 5. Training action progression

The retained outputs contain only the final deterministic VALIDATION action
distribution. They contain no TRAIN action counts by rollout, policy entropy
history, reward-component moments, or update snapshots. It is therefore
unknown whether Sell dominance appeared immediately or emerged late.

A future research-only diagnostics stream should record, per completed rollout
and symbol: action counts split by flat/long and valid/invalid state, reward and
component moments, exposure, entropy, KL, clip fraction, value diagnostics,
episode identifier, and cumulative timesteps. It should be a non-model audit
artifact in a temporary/research directory and must not include VALIDATION or
TEST during learning.

## 6. PPO logger diagnostics

| Final completed update diagnostic | Value | Interpretation |
|---|---:|---|
| Approximate KL | 0.008941 | Modest; no evidence of an excessive update |
| Clip fraction | 0.097656 | Moderate, not excessive clipping |
| Entropy loss | -0.502825 | Implied entropy about 0.503 versus three-action maximum 1.099; concentrated but not numerically zero |
| Explained variance | 0.630388 | Positive value-function fit, not obvious value-learning failure |
| Policy-gradient loss | -0.019343 | Finite; magnitude alone does not identify collapse |
| Value loss | 0.000581 | Finite and small in the environment's reward scale |
| Learning rate | 0.000300 | Configured value |
| Updates | 980 | Final retained count |

All recorded quantities and parameters were finite. These terminal values do
not support optimizer instability, but one snapshot cannot determine when the
action distribution concentrated or whether entropy fell prematurely.

## 7. Multi-symbol sampling and history depth

| Symbol | Episodes started/completed | Training steps | Contribution | TRAIN rows |
|---|---:|---:|---:|---:|
| ABL | 2/2 | 3,318 | 6.61% | 1,660 |
| AKBL | 2/2 | 3,404 | 6.78% | 1,703 |
| BAFL | 2/2 | 3,404 | 6.78% | 1,703 |
| BAHL | 1/1 | 1,703 | 3.39% | 1,704 |
| BIPL | 1/1 | 1,648 | 3.28% | 1,649 |
| BML | 1/1 | 1,702 | 3.39% | 1,703 |
| BOK | 2/2 | 2,966 | 5.91% | 1,484 |
| BOP | 2/2 | 3,406 | 6.79% | 1,704 |
| FABL | 2/2 | 3,400 | 6.78% | 1,701 |
| HBL | 2/1 | 2,557 | 5.10% | 1,704 |
| HMB | 1/1 | 1,674 | 3.34% | 1,675 |
| JSBL | 2/2 | 3,290 | 6.56% | 1,646 |
| MCB | 1/1 | 1,703 | 3.39% | 1,704 |
| MEBL | 2/2 | 3,380 | 6.74% | 1,691 |
| NBP | 2/2 | 3,406 | 6.79% | 1,704 |
| SBL | 1/1 | 1,216 | 2.42% | 1,217 |
| SCBPL | 2/2 | 2,988 | 5.96% | 1,495 |
| SNBL | 1/1 | 1,605 | 3.20% | 1,606 |
| UBL | 2/2 | 3,406 | 6.79% | 1,704 |

The maximum/minimum contribution ratio was 2.80. MCB received 1,703 specific
steps, only 16.63% of the 10,240 steps used by the independent recurrent MCB
experiment.

Across 19 observations, training timesteps had weak Pearson correlation with
VALIDATION return (`0.126`), Sharpe (`0.076`), exposure (`0.160`), and trade
count (`0.027`). Spearman correlations were `0.278`, `0.314`, `-0.020`, and
`0.040`, respectively. Per-symbol selected-action distributions were not
retained, so contribution/action-frequency correlation is unavailable. These
small-sample associations do not establish causation.

## 8. Per-symbol normalization under one policy

All policy inputs have the same ordered 17-dimensional schema, and execution
OHLCV remains real and unscaled outside the observation. Twelve market features
are standardized by each symbol's own TRAIN mean and scale. Thus `z=1` always
means one symbol-specific standard deviation above that symbol's mean, but the
underlying economic quantity varies materially:

| Feature | Scale max/min | Raw value represented by z=1, min–max |
|---|---:|---:|
| Simple return | 4.15x | 0.0152 to 0.0661 |
| High-low range | 21.23x | 0.2883 to 6.9240 price units |
| Rolling volatility 20 | 5.01x | 0.0198 to 0.0866 |
| MACD | 36.48x | 0.1025 to 3.3910 price units |
| MACD signal | 36.73x | 0.0959 to 3.1532 price units |
| MACD histogram | 33.30x | 0.0340 to 1.1533 price units |
| ATR 14 | 25.21x | 0.2459 to 6.4033 price units |
| OBV | 933.45x | 1.33 million to 1.235 billion |
| Volume MA 20 | 782.95x | 22,289 to 16.479 million shares |

Per-symbol scaling is leakage-safe and reduces domination by nominal price or
volume. It also removes absolute liquidity and volatility scale that could be
useful to a shared policy. It does not make identical normalized observations
economically identical. This is a plausible shared-policy ambiguity, not proof
that sector-wide scaling is better. Sector-wide normalization must not be
introduced without a separately specified TRAIN-only, cohort-only ablation.

## 9. Identity and partial observability

The policy receives no symbol name, filename, price level, or embedding. It sees
12 standardized market features and five portfolio-state features. This is a
defensible first invariance baseline and avoids easy symbol memorization, but
after every reset the policy cannot directly know which bank's normalization,
liquidity, tail behavior, or historical regime applies. LSTM state can infer a
regime only after observing within-episode dynamics; it is intentionally reset
at symbol boundaries.

The next diagnostic should first test whether balanced windows and richer audit
instrumentation resolve collapse. Only then should a low-cardinality context
ablation be specified. If identity is tested later, it should be a declared
context feature with target-exclusion safeguards, not an accidental path or
absolute-price leak.

## 10. Full-partition episode design

- Current equal-episode sampling gives each symbol one selection per shuffled
  cycle, but longer episodes contribute more steps and gradients.
- Capped full episodes reduce extreme dominance but still vary in duration.
- Balanced timestep sampling can equalize contributions but can create opaque
  reset/window behavior.
- Equal-symbol fixed-window episodes make contribution and reset semantics
  explicit and are the most interpretable next baseline.

Full-partition episodes are valid mechanically, but one or two exposures per
symbol are too coarse for a compute-matched architecture comparison. A fixed
window must reset environment, cash, holdings, drawdown, P&L, and LSTM state;
rollout boundaries inside a window must not reset it.

## 11. TRAIN and VALIDATION regime evidence

Counterfactual Buy-and-Hold across each complete TRAIN episode produced mean
return `-4.02%`, median `-23.88%`, eight positive symbols, and eleven negative
symbols. The later VALIDATION periods were strongly bullish: Buy & Hold mean
return `105.95%`, median `99.68%`, 18 positive symbols, one negative symbol,
25th percentile `68.35%`, and 75th percentile `157.78%`.

The sector model's low exposure is defective in the narrower sense that it
repeatedly chose penalized Sell instead of the strictly better flat Hold.
However, avoiding exposure is partly consistent with its permitted TRAIN
reward landscape and cannot be judged solely against an unusually bullish
later regime. Lower drawdown does not rescue the model: the policy missed most
of the validation upside and chose invalid actions on most steps.

## 12. Independent MCB versus sector MCB

| Item | Independent recurrent MCB | Sector foundation on MCB |
|---|---:|---:|
| Total model training steps | 10,240 | 50,176 across 19 banks |
| MCB-attributable steps | 10,240 | 1,703 |
| Distinct training symbols | 1 | 19 |
| Validation return | 145.76% | 15.74% |
| Sharpe | 2.565 | 1.722 |
| Exposure | 80.49% | 8.52% |

The comparison is not compute- or exposure-matched. Plausible differences are
six times less MCB-specific experience, mixed gradients, full-partition
sampling, identity-free partial observability, symbol-specific normalization,
and a different total update/data composition. Action/reward semantics are the
same, so they cannot alone explain the cross-model difference. One symbol and
one seed do not establish that sector learning is intrinsically inferior.

## 13. Root-cause ranking

1. **TRAIN reward/regime preference for low exposure — strong evidence.** Flat
   Hold scores zero; the canonical invested counterfactual scored `-20.50`,
   dominated by `-16.91` of incremental-drawdown penalties. TRAIN was mixed to
   negative while VALIDATION was strongly bullish.
2. **Cheap invalid-action attractor — strong mechanism evidence.** Flat Sell is
   penalized and therefore not optimal, but costs only `0.0001`, changes no
   portfolio state, and generated 5,917 invalid validation actions. There is no
   action mask.
3. **Insufficient/unequal effective symbol exposure — moderate evidence.** Only
   one or two episodes per symbol were seen; contributions varied 2.80x; MCB
   received 1,703 rather than 10,240 target-specific steps. The 50k run also
   violated the specified 25k protocol.
4. **Identity-free pooled regimes plus per-symbol scaling — plausible.** The
   same normalized value maps to materially different absolute volatility,
   price-range, and liquidity regimes. No direct ablation evidence exists.
5. **Numerical PPO instability — unsupported by retained evidence.** KL,
   clipping, value fit, losses, observations, rewards, and parameters were
   finite and not obviously unstable. Progression data is missing, so early
   entropy/policy concentration remains unknown.

These are ranked hypotheses, not causal proof.

## 14. Predeclared compute/exposure-matched follow-up

Do not execute this design inside 6E.1.

### Shared-sector arm

- Universe: the same 19 symbols and immutable universe hash above.
- Seeds: exactly `42, 43, 44`.
- Recurrent/PPO config: unchanged `recurrent_ppo_sector_v1` architecture and
  optimizer settings.
- Device: CPU.
- Normalization: existing per-symbol TRAIN-fitted scalers; no scaler change in
  the sampling experiment.
- Episode form: 512-transition TRAIN-only windows.
- Schedule: 20 rounds; each round contains one deterministic shuffled window
  from every symbol.
- Exposure: exactly 10,240 transitions per symbol.
- Total sector steps: exactly `19 * 20 * 512 = 194,560` per seed, already
  divisible by PPO `n_steps=512`.
- Window starts: deterministic from universe hash, seed, round, and symbol;
  confined to canonical TRAIN. Each window resets environment/account/LSTM,
  while rollout updates within the window preserve state.

### Comparison controls

No single control can simultaneously equal both shared-model total compute and
target-specific exposure. Predeclare both axes:

1. exposure-matched independent RecurrentPPO: 10,240 target steps, one symbol,
   same seed/config;
2. total-compute control for the primary target: 194,560 target steps using the
   same fixed-window/reset semantics, reported separately and not presented as
   exposure matched.

For MCB, UBL, HBL, and MEBL, report total environment steps, target-attributable
steps, wall time, parameter count, and distinct symbols before any outcome
comparison.

### Evaluation and diagnostics

- VALIDATION only, once after the final predefined update for each seed.
- Fresh recurrent and portfolio state per symbol; same capital and costs.
- Report all per-symbol distributions and Buy & Hold/Always Hold/Random.
- Retain research-only per-rollout valid/invalid action counts, reward-component
  moments, entropy/KL/clipping/value diagnostics, and sampling exposure.
- TEST remains sealed.
- Do not change reward, action masking, normalization, architecture, or sampler
  after looking at these results. Each proposed ablation must be a separately
  specified experiment.

A later transfer claim additionally requires target-excluded pretraining and
normalization contribution, fixed fine-tuning history, and multiple seeds. The
current all-constituent foundation model cannot provide that evidence.

## 15. Decisions

### SECTOR TRAINING ENGINE: **GO**

The multi-symbol controller, TRAIN-only loading, state isolation, recurrent
resets, deterministic sampling, numerical safeguards, and per-symbol
VALIDATION evaluation worked as intended.

### CURRENT COMMERCIAL BANKS FOUNDATION MODEL: **BLOCKED**

The exact 6E model is not a healthy parent: 90.41% Sell, 98.39% of Sells while
flat, median exposure 4.12%, only 1/19 return wins versus Buy & Hold, a protocol
budget deviation, one seed, and insufficient retained diagnostic provenance.
It was temporary and no longer exists, so it cannot be responsibly reused.

### LEAVE-ONE-OUT TRANSFER TRAINING: **BLOCKED**

Do not begin transfer using this policy or these validation results. Before
transfer: execute a separately approved balanced-window, multi-seed foundation
experiment; retain action/reward progression; establish a non-collapsed parent
under predeclared criteria; then train a target-excluded universe/scaler and
compare fixed-history fine-tuning against exposure-matched scratch controls.

## Integrity statement

This audit read existing source, JSON metadata, TRAIN/VALIDATION CSV artifacts,
scaler metadata, and executed notebook outputs. Counterfactual environment
rollouts did not call a learning method or write an artifact. TEST data was not
loaded or evaluated. No production code, dataset, raw/backfill state, model,
registry, notebook, or prior report was modified. This Markdown report is the
only 6E.1 artifact created. No live HTTP request or commit occurred.

## Verification

- Complete test suite: **523 passed, 2 skipped** in 22.00 seconds.
- `git diff --check`: passed.
- `.venv/bin/python -m pip check`: no broken requirements found.
- Production registry SHA-256 remained
  `e99dadcbc00ad084a85763baf599601fb9172950977ed66b9ac407c86322e75a`.
- Production model roots still contain only their existing `.gitkeep` files.
- Final `git status --short` contains only this requested report.
