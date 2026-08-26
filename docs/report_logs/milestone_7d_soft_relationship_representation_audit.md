# Milestone 7D — TRAIN-Only Soft Relationship Representation Audit

Audit date: 2026-08-26  
Branch: `feat/rl-environment-v1`  
Decision: **BLOCKED_SOFT_REPRESENTATION**

## Outcome

The frozen TRAIN data does not support freezing the tested soft relationship
representation. None of the predeclared dimensions `K = {8, 10, 12, 15, 20,
25, 30}` passed every evidence gate. K=8 was closest, but its temporal subspace
stability was `0.798305` against the predeclared `0.80` floor, and every full
TRAIN NMF fit reached the fixed 2,000-iteration cap without convergence.
Stability and robustness generally weakened as K increased.

This result does not revive hard clusters and does not authorize another soft
method trial. A further attempt requires architecture review.

## Data boundary and provenance

- Identity universe: 508 authoritative current common equities.
- Universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`.
- TRAIN interval: 2016-07-26 through 2023-08-03.
- Source Parquet SHA-256: `45e3d396c3472fa53b20016c153fdd529308600274ecf77d0ef942417793d7d5`.
- TRAIN-return-capable identities: 448.
- Overlap-eligible identities at the frozen 120-observation floor: 336.
- Complete full-TRAIN fit core: 318.
- Common temporal comparison core: 301.
- VALIDATION values loaded: no.
- TEST values loaded: no.
- Sectors used during fitting: no.

Loading used the existing predicate-pushed TRAIN-only boundary. Return
construction preserved the existing no-forward-fill, no-zero-fill, and
no-gap-spanning rules. The audit did not modify the Parquet source.

## Method chosen

The predeclared method was deterministic nonnegative matrix factorization of a
complete long-only correlation-affinity matrix:

1. construct gap-safe stock returns on TRAIN only;
2. calculate Pearson correlation subject to the frozen pairwise-overlap floor;
3. choose the deterministic complete-pair fit core;
4. define finite negative correlation as zero *long-only affinity* while
   refusing missing correlations rather than filling them;
5. fit NMF with `nndsvda` initialization, coordinate-descent/Frobenius loss,
   seed 42, tolerance `1e-5`, and maximum 2,000 iterations;
6. normalize the nonnegative loading row for each stock into overlapping soft
   memberships;
7. project non-core identities by NNLS using only observed TRAIN core
   relationships, requiring at least 60 observed relationships and confidence
   at least 0.10;
8. mark any identity that cannot meet projection requirements unsupported.

This method was chosen because it produces a continuous representation and an
auditable nonnegative decoder. It does not claim that the factors are natural
hard PSX clusters.

## Mathematical contract

For supported stocks, `Z >= 0` is the continuous relationship matrix and
`M[i,k] = Z[i,k] / sum_k Z[i,k]` is the soft-membership matrix. Therefore each
membership row is nonnegative and sums to one.

The prototype-to-stock decoder `P` is confidence-weighted membership normalized
by prototype column:

- `P[i,k] >= 0`;
- `sum_i P[i,k] = 1` for every prototype;
- for nonnegative prototype capital `a`, stock budgets are `b = P @ a`;
- `sum_i b[i] = sum_k a[k]`.

Unit tests verify decoder nonnegativity, column normalization, and capital
conservation. This is a representation contract only; no executable allocator
was implemented.

## Identity handling

The direct-fit count is 318 for every K. Projection results vary slightly with
K because NNLS reconstruction confidence depends on the fitted basis.

| K | Fitted | Projected | Unsupported | Supported share |
|---:|---:|---:|---:|---:|
| 8 | 318 | 102 | 88 | 82.68% |
| 10 | 318 | 102 | 88 | 82.68% |
| 12 | 318 | 103 | 87 | 82.87% |
| 15 | 318 | 102 | 88 | 82.68% |
| 20 | 318 | 103 | 87 | 82.87% |
| 25 | 318 | 103 | 87 | 82.87% |
| 30 | 318 | 103 | 87 | 82.87% |

All 508 identity members receive one explicit state: fitted, projected, or
unsupported. Unsupported rows are never silently removed or supplied with
fabricated returns.

## Full K comparison

`Rob sub/mem/dec` are the worst results across simple-return/Pearson and
log-return/Spearman robustness fits. `Entropy` is mean normalized membership
entropy. `Eff stocks` is the median inverse-Herfindahl decoder support.

| K | Recon. error | Temporal sub/mem/dec | Rob sub/mem/dec | Entropy | Max mass | Eff stocks | Max stock weight | Condition | Converged | Plateau | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 8 | 0.252548 | 0.798305 / 0.859116 / 0.732036 | 0.907382 / 0.927019 / 0.790208 | 0.787236 | 0.248342 | 229.22 | 0.028260 | 9.85 | no | yes | no |
| 10 | 0.248053 | 0.800054 / 0.790994 / 0.661324 | 0.891768 / 0.869392 / 0.745456 | 0.791641 | 0.165966 | 211.84 | 0.041263 | 9.59 | no | yes | no |
| 12 | 0.244272 | 0.772700 / 0.779079 / 0.662310 | 0.863810 / 0.861032 / 0.725571 | 0.772647 | 0.193145 | 196.95 | 0.048731 | 11.25 | no | yes | no |
| 15 | 0.239040 | 0.731831 / 0.766558 / 0.635005 | 0.830208 / 0.798840 / 0.676871 | 0.781685 | 0.106764 | 174.25 | 0.051874 | 10.03 | no | no | no |
| 20 | 0.231108 | 0.702400 / 0.661880 / 0.573515 | 0.765270 / 0.764224 / 0.642210 | 0.776887 | 0.083126 | 158.65 | 0.051223 | 9.55 | no | no | no |
| 25 | 0.223861 | 0.694386 / 0.695291 / 0.553530 | 0.747946 / 0.683592 / 0.574565 | 0.759014 | 0.083296 | 138.58 | 0.084534 | 10.89 | no | no | no |
| 30 | 0.217048 | 0.685722 / 0.691821 / 0.547178 | 0.751907 / 0.691889 / 0.565541 | 0.750303 | 0.069109 | 115.13 | 0.076054 | 10.37 | no | n/a | no |

Reconstruction improved monotonically, but this was not sufficient to select a
larger K. The relative gain to the next candidate was 1.78%, 1.52%, 2.14%,
3.32%, 3.14%, and 3.04% for K 8 through 25. Under the predeclared 3% plateau
rule, only K 8, 10, and 12 reached a plateau; each still failed at least one
other gate.

## Temporal stability

The early and late representations were fitted independently, then prototypes
were aligned with the Hungarian assignment using decoder-column cosine. Both
windows contain 1,220 TRAIN market dates, overlap by 698 dates, and use a
301-symbol common complete core.

| K | Subspace | Mean membership cosine | P10 membership cosine | Mean alignment cosine | Mean decoder overlap |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.798305 | 0.859116 | 0.706444 | 0.804014 | 0.732036 |
| 10 | 0.800054 | 0.790994 | 0.543285 | 0.739053 | 0.661324 |
| 12 | 0.772700 | 0.779079 | 0.585366 | 0.741191 | 0.662310 |
| 15 | 0.731831 | 0.766558 | 0.520091 | 0.719436 | 0.635005 |
| 20 | 0.702400 | 0.661880 | 0.409605 | 0.624180 | 0.573515 |
| 25 | 0.694386 | 0.695291 | 0.439911 | 0.623443 | 0.553530 |
| 30 | 0.685722 | 0.691821 | 0.427981 | 0.631838 | 0.547178 |

K=8 missed the temporal subspace floor narrowly, while the higher-dimensional
representations showed substantively weaker temporal reproducibility.

## Return/correlation robustness

Each comparison was separately fitted on the same 318-symbol full-TRAIN core
and aligned to the primary log-return/Pearson fit.

| K | Variant | Subspace | Mean membership cosine | Mean alignment cosine | Mean decoder overlap |
|---:|---|---:|---:|---:|---:|
| 8 | simple/Pearson | 0.958464 | 0.967714 | 0.951168 | 0.868958 |
| 8 | log/Spearman | 0.907382 | 0.927019 | 0.882954 | 0.790208 |
| 10 | simple/Pearson | 0.976087 | 0.876177 | 0.899178 | 0.802231 |
| 10 | log/Spearman | 0.891768 | 0.869392 | 0.845781 | 0.745456 |
| 12 | simple/Pearson | 0.958424 | 0.888903 | 0.894585 | 0.818450 |
| 12 | log/Spearman | 0.863810 | 0.861032 | 0.821341 | 0.725571 |
| 15 | simple/Pearson | 0.901907 | 0.802849 | 0.814639 | 0.719486 |
| 15 | log/Spearman | 0.830208 | 0.798840 | 0.775662 | 0.676871 |
| 20 | simple/Pearson | 0.898970 | 0.808761 | 0.802153 | 0.696427 |
| 20 | log/Spearman | 0.765270 | 0.764224 | 0.729934 | 0.642210 |
| 25 | simple/Pearson | 0.855657 | 0.749986 | 0.716307 | 0.626180 |
| 25 | log/Spearman | 0.747946 | 0.683592 | 0.653833 | 0.574565 |
| 30 | simple/Pearson | 0.884092 | 0.753548 | 0.771155 | 0.660190 |
| 30 | log/Spearman | 0.751907 | 0.691889 | 0.641423 | 0.565541 |

The low-dimensional variants were reasonably robust, but the result degraded
materially above K=12, particularly under rank correlation.

## Decoder and sector diagnostics

Every candidate decoder passed the mathematical investability invariants:
nonnegative entries, unit column sums, and capital conservation. Decoder
concentration was not the blocking problem: median effective support ranged
from about 115 to 229 stocks and the largest single-stock prototype weight
ranged from 2.83% to 8.45%.

Current 2026 sectors were applied only post hoc:

| K | Sector NMI | Mean sector purity | Maximum sector purity | Sector dominated |
|---:|---:|---:|---:|:---:|
| 8 | 0.183040 | 0.115465 | 0.170048 | no |
| 10 | 0.231888 | 0.115988 | 0.165633 | no |
| 12 | 0.223511 | 0.124615 | 0.193094 | no |
| 15 | 0.269460 | 0.125982 | 0.230092 | no |
| 20 | 0.315995 | 0.131415 | 0.228297 | no |
| 25 | 0.321921 | 0.135971 | 0.226563 | no |
| 30 | 0.356283 | 0.134134 | 0.284091 | no |

The factors did not merely reproduce current-sector taxonomy. That observation
does not solve the temporal instability or convergence failures, and current
sector metadata is not historical point-in-time membership evidence.

## Why no K was selected

- K=8 had the strongest overall temporal/robustness profile and reached a
  reconstruction plateau, but missed the temporal subspace floor and failed
  the convergence requirement.
- K=10 cleared the temporal subspace floor by only `0.000054`, but also failed
  convergence.
- K=12 missed temporal subspace stability and failed convergence.
- K>=15 improved reconstruction only by adding dimensions that were less
  temporally stable and less robust; K>=15 also did not reach the 3% plateau.
- All K values hit the fixed iteration cap. Raising it after observing this
  result would alter the predeclared experiment and was not attempted.

Therefore no representation version, relationship matrix, membership matrix,
or decoder has been frozen. `soft_relationship_nmf_v1` remains an audit method,
not an approved downstream contract.

## Agent-count practicality

Using the validated conservative recurrent benchmark (3.6 minutes per 100k
agent or 8.8 minutes per 250k agent on CPU), illustrative sequential costs are:

| Agent/prototype count | 100k each | 250k each |
|---:|---:|---:|
| 8 | 28.8 min | 70.4 min |
| 10 | 36.0 min | 88.0 min |
| 12 | 43.2 min | 105.6 min |
| 15 | 54.0 min | 132.0 min |
| 20 | 72.0 min | 176.0 min |
| 25 | 90.0 min | 220.0 min |
| 30 | 108.0 min | 264.0 min |

These estimates describe compute practicality only. They did not select K and
do not justify training prototype agents for a blocked representation.

## Predeclared gates and limitations

The audit required at least 80% supported identity coverage; temporal subspace,
membership, and decoder stability of 0.80/0.75/0.55; worst robustness values of
0.85/0.80/0.60; non-dominant sector correspondence; converged NMF; acceptable
conditioning and decoder concentration; normalized membership entropy from
0.15 to 0.95; and a reconstruction plateau. Thresholds and the K grid were not
changed after viewing results.

The identity universe is current-common-equity and therefore retains the
previously documented survivorship limitation. Sparse stocks are projection
candidates only from their observed TRAIN relationships. VALIDATION, TEST, PPO,
and trading returns played no role in method or K selection.

## Final decision

**BLOCKED_SOFT_REPRESENTATION**

No PPO or LSTM was trained, no lead allocator or final RL architecture was
implemented, no VALIDATION or TEST values were loaded, no representation was
frozen, and no existing clustering evidence was overwritten.
