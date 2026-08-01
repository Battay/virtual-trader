# Milestone 3A — AI dataset and model management

## Objectives

This milestone establishes deterministic feature data, chronological research
splits, training-only scaling, model version metadata, retraining detection,
symbol selection, and a Streamlit readiness page. It intentionally stops before
the PPO environment and trainer, so it produces no model predictions or fake
trained artifacts.

## Symbol and master model requirements

The planned symbol family has one independent PPO model per user-selected,
eligible active symbol. A newly listed and recently traded ordinary equity is
discovered from the refreshed Company Registry and appears automatically in the
readiness table. Never-trained and outdated symbols can be selected singly or
in groups.

The master PPO model is a separate identity and version history. Its dataset
retains the symbol column and may include supported active, inactive, and
historical securities. It is not a fallback symbol model and its staleness is
calculated against the master processed dataset.

## Feature-engineering methodology

Raw OHLCV values are converted to numeric values and processed in stable
symbol/date order. Group boundaries are enforced before any lag, rolling, EWM,
ATR, or cumulative OBV calculation. Derived features comprise simple and log
returns, high-low range, open-close return, rolling volatility, SMA/EMA 20 and
50, RSI 14 with Wilder EWM smoothing, MACD 12/26/9, 20-period population-
standard-deviation Bollinger Bands, 14-period ATR, OBV, and volume MA 20.

The feature version is a deterministic hash of the documented calculation
parameters. The first 49 observations per symbol are marked as warm-up. Missing
values are not backfilled and are never silently changed to zero. Dataset build
metrics distinguish warm-up removal from later missing-feature removal.

## Time-series splitting and leakage prevention

The default 70%/15%/15% split is chronological. Symbol datasets partition their
own ordered dates. The master dataset partitions the global set of trading dates
and then assigns every symbol row by those boundaries. Consequently no date can
appear in more than one partition.

Features are computed before partitioning but use only current and earlier
observations. Scaling is performed after splitting: `StandardScaler.fit` sees
training rows only, and validation/test rows use `transform`. Categorical and
identity columns are excluded from scaling. Metadata records boundaries, row
counts, symbols, feature columns, and scaler parameters.

## Data quality and eligibility

Fatal symbol-level checks include invalid/duplicate dates, missing or invalid
OHLCV, non-positive prices, inverted high/low values, and negative volume.
Symbol models additionally require listed/recently-active status, ordinary-
equity classification, and the configurable minimum usable history. The
initial value of 252 post-warm-up rows approximates one trading year but is not
presented as a universal PPO sample-size rule.

The master universe admits supported security types across lifecycle states and
excludes unknown or configured unsupported types. Both dataset forms come from
the same feature implementation and are atomically persisted.

## Model lifecycle and retraining

The CSV model registry has deterministic IDs and monotonically increasing
versions per symbol or master identity. Appends preserve old versions. Malformed
registry files raise safely before writes. Status values remain machine-friendly
internally and are humanized only by dashboard presentation helpers.

Staleness compares distinct processed trading dates with the recorded complete-
history training cutoff. Calendar weekends and holidays do not inflate the new
data count. Retraining metadata deliberately starts at the earliest configured
history and ends at the latest available date, preventing an incremental-only
retrain from discarding earlier context.

## Newly listed symbol handling

Registry refreshes identify new listings independently from market activity.
An active new ordinary equity enters the symbol status table automatically. It
shows `Insufficient History` until it crosses the configured quality/history
gate, then `Never Trained` until a real model version is trained in a later
milestone.

## Jupyter research workflow

Three small notebooks inspect data quality, call production feature functions,
and review split/scaling boundaries. Outputs are cleared in source control.
JupyterLab and VS Code may both execute them; reusable calculations stay in
Python modules so notebook experiments cannot silently diverge from production.

## Testing

Offline tests cover cross-symbol isolation, causal prefix stability,
deterministic indicators, numeric symbols, symbol/master builders, active and
historical eligibility, atomic writes, chronological/global-date splitting,
date disjointness, training-only scaling, registry safety/versioning, new-date
counts, complete-history metadata, bulk selection, and readable UI labels. No
test calls PSX or performs PPO training.

## Current limitations

At inspection time, the local master had 20,845 rows across 818 symbols from
12 Jun through 30 Jul 2026. No symbol had more than 33 dates. This is shorter
than the 50-observation indicator warm-up and far below the 252 usable-row gate,
so every current symbol is officially insufficient for PPO preparation. The
official PSX listing source also cannot conclusively classify every historical
instrument; unsupported/unknown types remain excluded by default.

## Next milestone

Milestone 3B will design and test the trading environment, reward function,
portfolio state, transaction costs, PPO trainer, reproducible seeds, evaluation
metrics, and real artifact registration. Training buttons remain disabled until
that implementation exists.
