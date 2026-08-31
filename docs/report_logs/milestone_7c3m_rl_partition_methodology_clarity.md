# Milestone 7C.3m — RL Partition Methodology Clarity

## Single-symbol RL partition protocol

The recurrent execution models use `rl_partition_v1`. Each symbol is split
independently in chronological order after causal feature generation and
warm-up removal:

- TRAIN: first `floor(70%)` of that symbol's usable observations.
- VALIDATION: next `floor(15%)`.
- TEST: remaining observations (approximately 15%), sealed.

Partition boundaries are aligned to distinct market dates so one market date
cannot cross partitions. The observation scaler is fitted on TRAIN only;
VALIDATION and TEST do not contribute to its fit. The Training & Models page
reads persisted boundary metadata and does not open partition dataframes.

Dates displayed for a verified model are therefore labelled **Model-observed
TRAIN range** and **Model-observed VALIDATION range**. They are properties of
that symbol's persisted model contract, not global research cutoffs. TEST is
shown as `SEALED`; only its already persisted row-count metadata is displayed.

## Distinct clustering/relationship protocol

Clustering and relationship studies use a separate common frozen temporal
protocol, including fixed research-window cutoffs such as 2023-08-03. Those
fixed common-calendar dates do not define `rl_partition_v1` and must not be
used to reinterpret or invalidate the existing recurrent models.

## Consistency audit

- `feature_engineering/splitting.py` performs the symbol-scoped chronological
  70/15/remaining split at distinct-date boundaries.
- `reinforcement_learning/data_contract.py` defines `rl_partition_v1` and
  requires TRAIN-only scaler fitting.
- `reinforcement_learning/recurrent_data_contract.py` layers
  `rl_recurrent_partition_v1` over that contract, permits recurrent loading of
  TRAIN or VALIDATION, and retains TEST as sealed metadata.
- `reinforcement_learning/training/recurrent_trainability_audit.py` uses a
  symbol-specific 70% raw-date prefix only for TRAIN-value-sealed eligibility
  diagnostics; it does not redefine persisted model partitions.
- `reinforcement_learning/training/model_details.py` validates the persisted
  per-symbol 70/15/remaining row counts, strict date ordering, TRAIN-fitted
  scaler metadata, and TEST sealing without opening partition frames.

No split algorithm, model artifact, run state, validation output, or TEST
observation was changed or accessed for this clarification.
