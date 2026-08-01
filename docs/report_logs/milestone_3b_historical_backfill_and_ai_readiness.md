# Milestone 3B — Historical backfill and AI dataset readiness

## Objective

This milestone adds a source-respectful, sequential, resumable historical PSX
backfill and connects newly collected daily data to the master, company
registry, feature datasets, chronological splits, scaling artifacts, and
training-readiness report. PPO training, signals, predictions, and model files
remain outside scope.

## Insufficient-history finding

Inspection found 20,845 master rows across 818 symbols and 33 valid locally
stored trading dates from 12 June through 30 July 2026. Per-symbol history
ranges from 1 to 33 rows, with a median and maximum of 33. No symbol has the 50
observations needed to finish the longest indicator warm-up, and none has the
252 post-warm-up usable rows required by the initial production gate.

A 17-day sample—and even the current 33-date sample—cannot support meaningful
PPO training. It cannot provide the required SMA/EMA-50 warm-up plus 252 usable
observations, robust chronological train/validation/test partitions, or enough
market regimes for credible evaluation. The 252-row value is an initial
engineering minimum, not a model-quality guarantee; more history is preferable.

## Backfill architecture

`data_pipeline.src.backfill` reuses `process_date()` for all network fetching,
HTML preservation, parsing, validation, rejection handling, and daily CSV
storage. It adds orchestration only:

- inclusive chronological planning;
- valid-daily-CSV discovery and exclusion;
- locally resolved weekends;
- historical-empty, recent-ambiguous, future, failure, success, and
  already-downloaded classifications;
- sequential requests with a configurable inter-request delay;
- bounded batches through `--max-dates`;
- dry-run planning;
- failure continuation and explicit failed-date retry; and
- interruption-safe, per-date persistence.

No parallel requests, browser automation, proxy rotation, cookie reuse, or
anti-bot bypass is used. A multi-year download is never launched automatically.

## Resumability and state safety

Progress is stored under `data/metadata/backfill_state.json`. It records the
requested range, last attempted/successful dates, successful and confirmed
non-trading dates, temporary skips, failures, already-downloaded dates,
timestamps, status, and last message. Writes use a temporary file followed by
atomic replacement. A malformed file is logged and ignored safely; valid daily
CSVs remain the source of truth.

Temporary skips remain eligible on resume. Failed dates are retried only when
explicitly requested. Ctrl+C records an interrupted state without discarding
earlier per-date outcomes.

## Date classification

- **Successful:** the one-date pipeline produced a valid, date-matching daily
  CSV.
- **Already downloaded:** an existing daily CSV passes the established
  structural/date validation.
- **Non-trading:** a weekend or an old historical response with no equity rows.
- **Temporarily unavailable:** today, a future weekday, or an empty response
  within the recent ambiguity window.
- **Failed:** network, invalid response, parsing, validation, or file failure.

This deliberately avoids treating recent ambiguity as a permanent holiday.

## Post-backfill rebuild

The explicit `data_pipeline.src.data_products rebuild` action runs:

1. deterministic master CSV rebuild;
2. company registry rebuild using the current official listing snapshot;
3. master AI dataset build;
4. eligible symbol AI dataset build;
5. chronological split and training-only scaler refresh; and
6. readiness-report recalculation.

The returned summary includes raw dates added, master rows/symbols, processed
rows/symbols, ready and insufficient symbols, split count, and errors. A stage
failure is reported while later diagnostic stages are still attempted.

## Readiness criteria

The shared readiness report calculates features using the production causal
feature implementation. For each active symbol it reports raw rows and dates,
warm-up rows, complete feature rows, configured minimum, additional usable rows
needed, split row counts, and one of:

- Ready
- Insufficient History
- Data Quality Issue
- Unsupported Security Type
- Missing Processed Dataset

The Training & Models page consumes this report. Notebook 3 uses the same
builder-metrics summary helper and cleanly skips splitting and scaling when a
symbol is not ready.

## Testing

Offline tests cover plan counts, existing-date exclusion, chronological order,
resumption, atomic writes, malformed-state recovery, interrupted persistence,
batch limits, dry runs, failed-date retry, temporary-skip retry, non-trading
classification, failure continuation, injected delay, rebuild order, exact
feature readiness, additional-row gaps, split counts, model-status integration,
and the Notebook 3 readiness summary. No test sleeps or calls PSX.

## Limitations

- Historical availability begins only where the operator explicitly chooses;
  the project does not assert that PSX supports a particular earliest date.
- An old weekday with a valid empty historical response is treated as
  non-trading. Operators can start a new range if source behavior later changes.
- Streamlit runs deliberately bounded foreground batches with progress output;
  the CLI is preferred for long unattended backfills.
- Meeting 252 usable rows does not guarantee PPO quality or sufficient market
  diversity.
- PPO environment design, training, evaluation, and predictions remain future
  work.
