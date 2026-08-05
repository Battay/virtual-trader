# PSX Virtual Trader

## Official market indices and market intelligence

Milestone 4A keeps indices separate from equity rows and uses only the official
PSX end-of-day route:

```text
GET https://dps.psx.com.pk/timeseries/eod/{index_code}
```

Supported codes are `KSE100`, `KSE30`, `KMI30`, and `ALLSHR`. PSX calls its
headline benchmark KSE-100; `PSX100` is not used internally. Each JSON
observation is `[unix_timestamp, index_value, volume, open_or_reference]`.
Timestamps are retained and converted to trading dates in `Asia/Karachi`; PSX
does not provide high/low fields and they are not fabricated.

Refresh all series or selected series with:

```bash
python -m market_intelligence.refresh_indices --all
python -m market_intelligence.refresh_indices --index KSE100 --index ALLSHR
```

Untouched responses are stored under `data/indices/raw/`, normalized per-index
CSVs under `data/indices/master/`, and the combined business-keyed dataset at
`data/indices/master/psx_indices_master.csv`. The source returns its complete
retained series without caller-defined date parameters; approximately five
years were observed during source investigation, but retention is not a
documented guarantee.

The Market Overview calculates breadth from valid securities on the latest
local equity date. It can also restrict breadth to currently listed ordinary
equities when a registry is supplied. The transparent Market Health Score uses
weighted index trends (50 points), advance/decline ratio (15), advancing share
(10), volume participation (10), moving-average position (10), and a volatility
penalty (5). Available components are normalized when inputs are missing. It is
a descriptive market-condition indicator, not investment advice.

AI datasets can attach same-date market context without changing the equity
master: index levels/returns, KSE-100 five-day return and volatility, breadth,
and market-health fields. Joins never use future observations; forward filling
defaults to zero days and is explicitly bounded when enabled. Dataset build
metadata records whether context was available. Models are not retrained
automatically.

PSX warns that dissemination and commercial use of market data, including
index levels, may require a licence. This project treats the source as
educational/research data; deployments should confirm rights with PSX Market
Data before redistributing it.

PSX Virtual Trader is a university FYP project for collecting and exploring
Pakistan Stock Exchange market data. The current system provides a
requests/BeautifulSoup data pipeline, persistent daily and master CSV data,
incremental updates, optional macOS automation, and a multipage Streamlit
dashboard.

## Setup and entry points

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Fetch one market date or an inclusive date range:

```bash
python -m data_pipeline.src.main --date 2026-07-27
python -m data_pipeline.src.main --start-date 2026-07-01 --end-date 2026-07-31
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run the offline test suite:

```bash
python -m pytest -v
```

## Historical backfill and AI readiness

The daily collector only creates rows for dates that have been requested. A
historical backfill is therefore required before rolling indicators,
chronological evaluation, or future PPO research can use a meaningful time
horizon. The earliest requested date is always an explicit CLI or dashboard
input; the project does not claim or hard-code an unsupported PSX history
boundary.

Preview a range without making requests or changing saved progress:

```bash
python -m data_pipeline.src.backfill \
  --start-date 2016-07-26 \
  --end-date 2026-07-31 \
  --delay-seconds 1 \
  --dry-run
```

Start with a safe 10-request batch:

```bash
python -m data_pipeline.src.backfill \
  --start-date 2016-07-26 \
  --end-date 2026-07-31 \
  --delay-seconds 1 \
  --max-dates 10
```

Resume the exact saved range, or explicitly retry dates that previously
failed:

```bash
python -m data_pipeline.src.backfill \
  --start-date 2016-07-26 \
  --end-date 2026-07-31 \
  --delay-seconds 1 \
  --max-dates 100 \
  --resume

python -m data_pipeline.src.backfill \
  --start-date 2016-07-26 \
  --end-date 2026-07-31 \
  --delay-seconds 1 \
  --max-dates 100 \
  --resume \
  --retry-failed
```

The backfill is sequential and uses the existing one-date pipeline. Valid
daily CSVs are never fetched again. Weekends are resolved locally; old empty
historical responses are recorded as non-trading dates; today, future dates,
recent ambiguous empty responses, and previously temporary skips remain
retryable. Network, parsing, validation, and file errors are recorded per date
without stopping later requests. The configured delay is applied between live
requests, with no parallel fetching, proxy rotation, cookie reuse, or anti-bot
circumvention.

Progress is atomically stored in `data/metadata/backfill_state.json` after each
attempt. Ctrl+C preserves completed work and marks the run interrupted. Use the
same start/end dates with `--resume`; a mismatched range is rejected clearly.
Generated state remains ignored by Git.

After a batch, explicitly rebuild all dependent data products in order:

```bash
python -m data_pipeline.src.data_products rebuild
```

This rebuilds the master CSV, rebuilds the company registry from its current
official snapshot, builds master and eligible symbol AI datasets, refreshes
chronological split/scaler metadata, and recalculates training readiness. It
never starts model training.

The reusable readiness report has one row per active symbol and distinguishes
**Ready**, **Insufficient History**, **Data Quality Issue**, **Unsupported
Security Type**, and **Missing Processed Dataset**. It reports raw and usable
rows, indicator warm-up removal, the remaining usable-row gap, date range, and
train/validation/test row counts. The Historical Backfill and Training & Models
dashboard pages expose these services without invoking CLI subprocesses.

## AI dataset and model-management foundation

Milestone 3A prepares deterministic datasets and model lifecycle metadata; it
does **not** implement or simulate PPO training. The architecture separates:

- one future PPO model per selected active, listed ordinary-equity symbol; and
- one independently versioned master PPO model whose supported universe can
  include active, inactive, and historical securities while retaining each
  symbol's identity.

New active symbols automatically enter the readiness table after the master
data and Company Registry are refreshed. A model becomes outdated when its
recorded complete-history cutoff predates distinct newly available trading
dates. Future retraining will rebuild features and train from the complete
configured history, not only the incremental rows.

### Feature datasets

`feature_engineering.indicators` calculates features separately per symbol in
symbol/date order. It provides raw OHLCV, simple/log returns, price ranges,
20-day volatility, SMA/EMA 20 and 50, RSI 14, MACD, Bollinger Bands, ATR 14,
OBV, and 20-day volume average. Every calculation is backward-looking. Missing
indicators remain missing; no future backfill or zero replacement occurs.

The first 49 observations per symbol are explicit warm-up rows because the
50-observation features are not yet available. Processed AI datasets remove
warm-up and other incomplete feature rows and record both counts in build
metrics. Symbol datasets and the master dataset use this same implementation.

Build selected symbol datasets:

```bash
python -m feature_engineering.dataset_builder symbols --symbols MCB OGDC
```

Build the combined master AI dataset:

```bash
python -m feature_engineering.dataset_builder master
```

Build both forms or validate generated files:

```bash
python -m feature_engineering.dataset_builder all
python -m feature_engineering.dataset_builder validate data/processed/master/psx_ai_master.csv
```

Generated processed datasets live under `data/processed/` and are atomically
replaced. They remain ignored by Git.

### Eligibility and current history limitation

The initial configurable symbol eligibility threshold is
`AI_MINIMUM_USABLE_ROWS = 252`. This approximates one trading year **after**
indicator warm-up and is an engineering readiness gate, not a claim that 252
rows are universally sufficient for PPO. More history is preferable, meeting
the minimum does not guarantee model quality, and the 49 indicator warm-up rows
are additional to the 252 usable rows. Research may justify increasing it.
Symbol eligibility additionally requires listed/recently-active status,
ordinary-equity classification, and no fatal data-quality errors.

The master-model universe does not require active status. Its configurable
default supports recognized ordinary equities, preference shares, GEM
equities, ETFs, and other classified instruments; `unknown` and transient
rights are excluded by default. Lifecycle identity remains in every row.

The current local master history covers only 12 Jun–30 Jul 2026, with at most
33 trading dates per symbol. Therefore no symbol currently passes SMA-50
warm-up or the 252-row gate, and the dashboard must report **Insufficient
History**. No PPO training should be attempted from this local sample.

### Chronological splitting and scaling

Only time-based 70%/15%/15% train, validation, and test splits are supported.
Symbol datasets split their own dates; the master dataset uses shared global
date boundaries so the same trading date cannot occur in multiple partitions.
No random shuffling is used.

Create symbol or master split artifacts:

```bash
python -m feature_engineering.splitting symbols --symbols MCB OGDC
python -m feature_engineering.splitting master
python -m feature_engineering.splitting all
```

`StandardScaler` is fitted only on the training partition. Validation and test
data use that fitted scaler; symbol, date, lifecycle, and security identity
columns are never scaled. Split metadata and scaler metadata are stored under
`data/processed/splits/`.

### Model registry and research workflow

Initialize the machine-friendly, atomic, version-preserving registry:

```bash
python -m reinforcement_learning.model_management.registry init
```

The registry at `data/models/model_registry.csv` tracks symbol and master model
versions independently. Previous versions are appended rather than silently
overwritten. It records data boundaries, row counts, feature/environment
versions, artifacts, failures, and distinct new-trading-day counts. No trained
record or model file is fabricated in Milestone 3A.

Launch the starter research notebooks:

```bash
jupyter lab notebooks
```

JupyterLab is useful for sequential exploration and charts. VS Code can open
the same `.ipynb` files with its Jupyter extension, while production changes
should stay in reusable Python modules and be exercised through pytest. The
notebooks deliberately import those modules and contain no copied indicator or
splitting implementation.

## Company and security registry

Milestone 2C adds `data/master/company_registry.csv`, a complete symbol-level
view formed by an outer merge of the current official PSX listings and the
local master market history. This preserves current listings with no local
price rows, as well as historical symbols that are absent from the current
official list.

Listing status and trading activity are deliberately separate:

- `official_status` comes from the official PSX listing tables or an
  evidence-based curated override. A missing recent trade never proves that a
  symbol is delisted.
- `activity_status` comes from the local master dataset and reports whether a
  symbol traded within the configurable 30-calendar-day window.
- `lifecycle_status` combines those concepts with explicit precedence for
  official delisting, suspension and non-compliance, followed by newly listed,
  listed/active, listed/inactive, historical-only and unknown classifications.

The source is the official [PSX listings page](https://dps.psx.com.pk/listings).
The page loads HTML table fragments from direct endpoints for the Main and GEM
boards and their normal/non-compliant segments. The pipeline requests those
fragments directly and parses normalized header names instead of fixed column
positions.

The registry stores:

- identity and instrument metadata: `symbol`, `company_name`, `security_type`,
  `sector`, `board`, `listing_segment`, `clearing_type`, `listed_in`, `shares`,
  and `free_float`;
- listing state: `officially_listed`, `official_status`, `source`,
  `listing_refreshed_at`, and `cached_listings_used`;
- market activity: `first_seen_date`, `last_seen_date`, `trading_days`,
  `days_since_last_seen`, and `activity_status`;
- lifecycle data: `lifecycle_status`, `is_new_listing`, and
  `registry_updated_at`;
- evidence-based extensions: `previous_symbol`, `successor_symbol`,
  `corporate_action_type`, and `notes`.

Numeric-looking symbols are always loaded as text. The registry includes all
recognized instruments and distinguishes ordinary equities, ETFs, rights,
preference shares, GEM equities, other instruments, and unknown historical
types.

### Refresh and rebuild

Refresh only the current official listing snapshot:

```bash
python -m data_pipeline.src.company_registry refresh-listings
```

Rebuild from the cached official snapshot and local master market dataset:

```bash
python -m data_pipeline.src.company_registry rebuild
```

Refresh the listings and rebuild in one command:

```bash
python -m data_pipeline.src.company_registry refresh
```

Official snapshots are stored in `data/metadata/listings/`; the registry is
stored in `data/master/`. Both locations contain generated data and remain
ignored by Git. A successful automated market update now rebuilds the master,
refreshes listings, and rebuilds the registry in that order.

If the live listing source fails, the newest valid cached snapshot is used and
the result records that degraded mode. A zero-row or structurally invalid live
response cannot replace a valid cache. If live data and every cache both fail,
the existing registry is left untouched.

### Curated overrides

An optional `data/metadata/company_overrides.csv` can contain:

```text
symbol,company_name_override,official_status_override,previous_symbol,successor_symbol,corporate_action_type,notes
```

The file is not required. When present, it is validated before the registry is
replaced. Overrides must be based on reliable evidence such as an official PSX
notice; the pipeline does not guess renamed, merged, suspended, or delisted
symbols.

### Future training universe

`data_pipeline.src.training_universe.select_training_universe` returns a
deterministic tuple of eligible symbols. Its defaults select currently listed,
recently traded ordinary equities, with an optional minimum trading-day count.
It does not alter market data or implement model training.

## Current limitations

- The official listings tables expose listed and non-compliant segments, but
  not a complete authoritative delisted-symbol history or a separate reliable
  suspension flag. Those states remain unassigned unless an evidence-based
  override supplies them.
- `first_seen_date` is the earliest date in the local master dataset, not an
  official incorporation or listing date. A short local history can therefore
  make the new-listing metric provisional until more history is collected.
- Security type is inferred conservatively from the available official symbol,
  name, sector, and board metadata; uncertain historical-only symbols remain
  `unknown`.

## Dashboard presentation

The dashboard keeps machine-friendly enum values in the pipeline and generated
files, while converting them to human-readable labels at the presentation
boundary. For example, `historical_only` is shown as “Historical Only” and
`ordinary_equity` as “Ordinary Equity.” This affects display only; filters map
their readable choices back to the original backend values.

Stock Explorer provides a searchable company/security selector, identity and
status context, responsive PKR price metrics, compact latest volume, selectable
1M/3M/6M/1Y/All price history, readable security metadata, a newest-first price
table with controlled pagination, and a full CSV download for the selected
period. Historical source data is copied before display formatting and is not
modified.

Dataset Explorer is company first: filters operate on a one-row-per-security
summary, row selection opens only that security's chart and paginated daily
history, and separate downloads cover the filtered company list and the full
selected-period history.

Company Registry combines official PSX listing metadata with locally observed
trading history. Its summary distinguishes currently listed securities that
traded recently from historical-only symbols that also have recent local rows.
Filters are grouped by search, status, classification, and trading history;
table and detail labels are formatted for users while the downloaded registry
retains the raw schema and enum values.

Shared formatting in `dashboard/presentation.py` handles missing values,
Pakistan-time timestamps, dates, integers, prices, percentages, compact
volumes, status badges, selector labels, and safe fallback humanization.
