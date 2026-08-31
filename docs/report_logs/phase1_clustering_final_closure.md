# Phase 1 — Clustering Evidence Consolidation and Closure

Audit timestamp: 2026-08-31T21:20:46Z  
Closure version: `phase1_clustering_closure_v1`  
Final decision: **REJECTED_CLUSTER_STRUCTURE**

## Outcome

Phase 1 can close with a negative scientific result. The bounded hard-clustering
experiments did not produce groups that were jointly stable, cohesive,
balanced, and robust to reasonable return representations. The subsequent soft
NMF experiment also failed its predeclared temporal-stability and convergence
gates. No hard assignments, soft representation, or decoder is approved for
downstream use.

The two required method decisions are:

- **REJECT_HARD_CLUSTERING**
- **REJECT_SOFT_RELATIONSHIP**

Weak and unstable relationship structure is itself the Phase-1 finding. No
acceptance threshold was weakened and no new representation was tried during
this closure audit.

## Research question

Does TRAIN-only PSX return evidence support a reproducible hard cluster or soft
relationship structure for downstream hierarchical decision-making?

For the bounded methods tested, the answer is no.

## Experiment inventory

| Work | Version | Code and tests | Configuration / partition | Persisted report | Decision |
|---|---|---|---|---|---|
| Raw return-correlation diagnostics | `phase1_clustering_methodology_audit_v1` | `data_pipeline/src/clustering_methodology.py`; `test_clustering_methodology.py` | Log/Pearson primary, Spearman robustness, angular-chord distance, average linkage, overlap and K sweeps; TRAIN-only | No standalone report was committed | Non-final diagnostic |
| Protocol selection | `phase1_clustering_protocol_selection_v1` | `data_pipeline/src/clustering_protocol.py`; `test_clustering_protocol.py` | Average/complete/positive-correlation graph; overlap 120/252/504; K=8..20; TRAIN subwindow refits; Pareto evidence table | No standalone report was committed | `blocked_weak_cluster_structure` |
| Market-mode removal | `phase1_market_mode_noise_audit_v1` | `data_pipeline/src/clustering_market_mode.py`; `test_clustering_market_mode.py` | Raw and static equal-weight residual correlations; bounded shrinkage feasibility; frozen common TRAIN window | No standalone report was committed | `BLOCKED_WEAK_CLUSTER_STRUCTURE` |
| Sector-informed multiview | `phase1_sector_informed_multiview_audit_v1` | `data_pipeline/src/clustering_multiview.py`; `test_clustering_multiview.py` | Return distance plus categorical sector distance, lambda 0.0–0.3, K=8..20; frozen common TRAIN window | No standalone report was committed | `BLOCKED_WEAK_CLUSTER_STRUCTURE` |
| Soft NMF relationship representation | `soft_relationship_nmf_v1` / `soft_relationship_contract_v1` | `data_pipeline/src/soft_relationship_representation.py`; `test_soft_relationship_representation.py` | K=8/10/12/15/20/25/30; separately fitted temporal windows; two robustness representations; long-only decoder | `milestone_7d_soft_relationship_representation_audit.md` | `BLOCKED_SOFT_REPRESENTATION` |

The architecture review after the hard-clustering block was read-only and did
not create another fitted representation. The soft NMF audit was its one
bounded follow-up. Recurrent-training benchmarks and model work are not Phase-1
clustering experiments and are excluded from this inventory.

## Universe and temporal provenance

- Frozen identity: 508 authoritative current common equities.
- Identity snapshot: 2026-08-02.
- Universe hash: `571f32af6de4d864ded90bbc06e814cf309fdffe4f61151102895a93ec588ef5`.
- Common clustering TRAIN interval: 2016-07-26 through 2023-08-03.
- VALIDATION dates were retained only as boundary metadata; observations did
  not enter fitting.
- TEST date metadata remained sealed; TEST observations and returns were not
  loaded.
- The current-equity identity creates a documented survivorship limitation and
  is not a point-in-time historical PSX universe.

The clustering common-calendar protocol is not `rl_partition_v1`.
`rl_partition_v1` assigns the first 70%, next 15%, and final 15% of each
symbol's own usable chronological observations to TRAIN, VALIDATION, and sealed
TEST. Its symbol-specific dates do not define Phase-1's common research window.

The earliest methodology code derived its common 70/15/15 calendar boundaries
from the then-current dataset. That moving calculation is provenance-sensitive
after later data ingestion and has no standalone persisted report. The later
market-mode, multiview, and soft audits explicitly froze the common TRAIN end
at 2023-08-03; those fixed-window audits are the authoritative closure evidence.

The original soft audit recorded Parquet SHA-256
`45e3d396c3472fa53b20016c153fdd529308600274ecf77d0ef942417793d7d5`.
The current consolidated file hash is
`8843f4573baf07a0c1e266efcb9ba37f96a839a42b41f75528d4f1c67b78fb34`
because later observations were appended. Predicate-pushed fixed-TRAIN reruns
reproduced the later hard-audit counts and metrics, so the frozen TRAIN evidence
remains logically consistent despite whole-file hash evolution.

## Sector provenance and limitation

The 508-stock snapshot has 34 current sectors and no missing sector label. Its
source is the official PSX listing snapshot dated 2026-08-02. Those labels are
current company metadata, not historical effective-dated proof of sector
membership throughout 2016–2023.

- Raw, residual, and soft primary fits did not use sectors. Sectors were
  post-hoc interpretation only.
- The bounded multiview experiment deliberately used the fixed current-sector
  labels as a secondary categorical distance.
- That use introduces a temporal metadata/look-ahead limitation but no future
  return leakage. Its results cannot be called survivorship-free or historical
  point-in-time sector clustering.

## Hard-clustering evidence

The fixed-window rerun reconciled 508 identities, 448 with usable TRAIN returns,
336 passing the 120-overlap eligibility rule, 318 in the complete fit core, and
298 in the hard temporal comparison core.

### Raw versus market-residual reference

The complete-linkage K=15 reference illustrates the tradeoff:

| Representation | Silhouette | Return cohesion gap | Largest cluster | Size CV | Temporal ARI | Temporal NMI | Sector NMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw log/Pearson | -0.001650 | 0.102306 | 34.59% | 1.5071 | 0.491944 | 0.383490 | 0.289358 |
| Equal-weight market residual | 0.007311 | 0.045006 | 14.15% | 0.4439 | 0.152033 | 0.307921 | 0.374989 |

Residualization removed the average market mode and substantially improved
cluster-size balance. It also more than halved the return cohesion/separation
gap and materially worsened temporal stability. Raw-versus-residual assignment
ARI at K=15 was only 0.080206. This is transformation sensitivity, not a stable
latent grouping.

Strict Ledoit-Wolf shrinkage was not feasible without imputation: the 318-stock
core had zero dates on which every stock simultaneously had a valid return.
The audit correctly refused to fabricate a complete matrix.

### Sector-informed candidate

The strongest bounded hard candidate for diagnostic purposes was the raw-return
multiview result at lambda=0.10 and K=12:

- combined-distance silhouette: 0.068111;
- return cohesion/separation gap: 0.066167, down 0.022995 from lambda=0;
- largest cluster: 26.42%; size CV: 0.8869;
- temporal ARI/NMI: 0.557599 / 0.768607;
- sector NMI/purity: 0.787948 / 0.559748;
- simple-return robustness ARI: 0.486375;
- Spearman robustness ARI: 0.565217;
- raw-versus-residual ARI: 0.326461.

For the same K, raising lambda from 0 to 0.1 improved the combined-distance
silhouette from -0.000684 to 0.068111 and reduced the largest cluster from
34.59% to 26.42%. However, the return gap deteriorated from 0.089161 to
0.066167 while sector NMI jumped from 0.261019 to 0.787948. Lambda 0.2 and 0.3
raised sector NMI to 0.796977 without improving the return gap. The apparent
stability/balance improvement was therefore largely static-taxonomy alignment,
not stronger return structure.

### Hard acceptance rule and decision

The frozen hard protocol used a Pareto evidence table rather than inventing one
scalar score. It required a specific candidate to be materially defensible
across temporal stability, return cohesion/separation, balance, and
representation robustness without merely reproducing sectors. No numeric
silhouette or ARI pass floor was predeclared; this closure does not retrofit
one. Existing fixed controls include overlap 120, active coverage 0.50, at
least 20 overlapping peers, K=8..20, and bounded sector weights. The multiview
diagnostic's explicit domination floors were sector NMI/purity 0.80, and a
material return-gap improvement was at least 0.01 or 10%.

No candidate jointly met the scientific rule. Therefore:

**REJECT_HARD_CLUSTERING**

No K or hard assignment is frozen.

## Soft-relationship evidence

The persisted soft audit has SHA-256
`390bada329a8dfbb6b70a6900b299301a431660477f4a32695d2a439dc7c4237`.
It used 318 direct-fit identities, a 301-symbol common temporal core, and
explicit fitted/projected/unsupported states for all 508 identities.

| K | Reconstruction | Temporal subspace / membership / decoder | Worst robustness subspace / membership / decoder | Converged | Passed |
|---:|---:|---:|---:|:---:|:---:|
| 8 | 0.252548 | 0.798305 / 0.859116 / 0.732036 | 0.907382 / 0.927019 / 0.790208 | No | No |
| 10 | 0.248053 | 0.800054 / 0.790994 / 0.661324 | 0.891768 / 0.869392 / 0.745456 | No | No |
| 12 | 0.244272 | 0.772700 / 0.779079 / 0.662310 | 0.863810 / 0.861032 / 0.725571 | No | No |
| 15 | 0.239040 | 0.731831 / 0.766558 / 0.635005 | 0.830208 / 0.798840 / 0.676871 | No | No |
| 20 | 0.231108 | 0.702400 / 0.661880 / 0.573515 | 0.765270 / 0.764224 / 0.642210 | No | No |
| 25 | 0.223861 | 0.694386 / 0.695291 / 0.553530 | 0.747946 / 0.683592 / 0.574565 | No | No |
| 30 | 0.217048 | 0.685722 / 0.691821 / 0.547178 | 0.751907 / 0.691889 / 0.565541 | No | No |

The exact predeclared gates included: supported identity share at least 0.80;
temporal subspace/membership/decoder at least 0.80/0.75/0.55; worst robustness
at least 0.85/0.80/0.60; sector NMI and mean purity no more than 0.75; valid
decoder concentration, entropy, and conditioning; a reconstruction plateau;
and convergence.

K=8 was closest. It supported 82.68% of identities and its decoder was valid,
but temporal subspace stability was 0.798305 against the 0.80 floor. Every full
fit reached the fixed 2,000-iteration cap without convergence. Higher K reduced
reconstruction error but generally weakened stability. No K passed all gates.

Therefore:

**REJECT_SOFT_RELATIONSHIP**

No representation, membership matrix, or decoder is frozen.

## Final Phase-1 architecture decision

**REJECTED_CLUSTER_STRUCTURE**

Phase 1 is closed as a negative finding. The data did not support the bounded
hard cluster families, and the one approved soft follow-up did not meet its
predeclared contract. Repeatedly trying representations after these blocks
would be post-result method search, not closure.

## Constraint inherited by Phase 2

Phase 2 must not assume reliable discrete cluster membership or an accepted
soft prototype basis. Its architecture must remain valid without clusters. Any
later relationship feature would be exploratory and would require a separate,
predeclared justification; it is not an approved Phase-1 output.

This statement constrains Phase 2 but does not begin or design it.

## Safety and limitations

- No VALIDATION observations were used in fitting.
- TEST remained sealed; no TEST observations or returns were loaded.
- No PPO/LSTM or other model was trained.
- No Lead Agent work was performed.
- No Phase-3 recurrent agent code or artifact was modified.
- No hard assignments or soft artifacts were written.
- The current-equity universe and current-sector snapshot retain documented
  survivorship and temporal-metadata limitations.
- This closure consolidates existing methods; it does not claim that all
  possible relationship representations are impossible.

The machine-readable decision is
`docs/config/phase1_clustering_closure_v1.json`, with evidence identity
`fb4c2a404c1ca8acb3a022283e6ace45519347c38ee9518d7453a45a4488b16e`.
