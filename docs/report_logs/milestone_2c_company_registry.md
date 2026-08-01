# Milestone 2C — Company Registry and Security Lifecycle Management

## Objective

This milestone creates a continuously rebuildable registry of PSX companies
and other listed securities. It separates official listing state from observed
market activity, retains historical symbols, supports conservative instrument
classification, and prepares a reusable symbol universe for later modelling.
No model training, indicators, MongoDB, web API, or new scheduler was added.

## Official data source and inspection

The source investigated was the official PSX Data Portal listings page:
<https://dps.psx.com.pk/listings>.

Browser-network behaviour showed that the page itself is an HTML shell. Its
JavaScript obtains table fragments with direct HTTP GET requests to:

- `/listings-table/main/nc`
- `/listings-table/main/dc`
- `/listings-table/gem/nc`
- `/listings-table/gem/dc`

The `nc` and `dc` segments represent the normal counter and non-compliant
segment respectively. The fragments are `text/html` and can be obtained with a
normal `requests.Session`; browser cookies, Selenium and Playwright are not
needed. The observed normal-counter headings were Symbol, Name, Sector,
Clearing Type, Shares, Free Float and Listed In. The non-compliant segment also
provided Non-Compliance of PSX Regulations.

During implementation inspection on 2026-07-30, the four fragments contained
466 Main normal, 92 Main non-compliant, 5 GEM normal and 0 GEM non-compliant
rows: 563 unique official symbols. These figures are a time-specific
observation, not constants in production code.

## Requests and BeautifulSoup methodology

`PsxListingsClient` reuses a `requests.Session`, supplies an HTML accept header,
a reasonable user agent and the XMLHttpRequest header, and applies the project
request timeout. It validates HTTP status, empty bodies and unexpected response
types, then returns untouched HTML to the parser.

BeautifulSoup parses each table fragment. Headers are normalized to lowercase
underscore names and mapped to registry fields. Required columns are validated
before rows are accepted, so reordered columns continue to work while removed
or renamed required columns fail clearly. Numeric-looking symbols remain text;
shares and free float are parsed as optional integers without inventing values.
An aggregate zero-row result is a failure.

## Status classification methodology

Two independent concepts are stored:

1. `official_status` reflects the current PSX listing source or a validated,
   evidence-based override.
2. `activity_status` uses each symbol's last observed date in the local master
   dataset and a configurable 30-calendar-day recent-trading window.

The lifecycle precedence is: officially delisted, suspended, non-compliant,
newly listed, listed/recently traded, listed/not recently traded,
historical-only, then unknown. New-listing classification uses the first date
seen locally and a configurable 30-calendar-day window, and only applies to a
currently official symbol.

Absence from recent historical data is not evidence of delisting. A listed
security may trade infrequently or may not yet exist in the local history.
Likewise, a historical symbol missing from today's listing tables can represent
many situations. It is therefore labelled `historical` / `historical_only`,
never automatically `delisted`.

Instrument type is inferred conservatively from official symbol, company/name,
sector and board fields. Explicit ETF, rights, preference, GEM and REIT evidence
is separated from ordinary equities. Historical dated-contract symbol patterns
are classified as `other`; uncertain historical-only symbols remain `unknown`.

## Registry architecture

The official snapshot layer fetches, parses, deduplicates, validates, and
atomically writes `data/metadata/listings/current_listings.csv`, with an
optional dated snapshot. Duplicate symbols are logged and a non-compliant row
takes precedence over a normal-segment duplicate.

The registry builder reads that snapshot and `data/master/psx_master.csv`,
calculates first/last observed dates and unique trading-day counts, and performs
an outer symbol merge. The resulting `data/master/company_registry.csv` is
sorted deterministically and atomically replaced only after all inputs and any
optional override file have validated.

If live listing retrieval or parsing fails, the newest valid cached snapshot is
used and this is recorded in structured results and the registry. If no valid
cache exists, the build stops before touching the existing registry. The daily
automation flow records market-update, master-rebuild, listing-refresh and
registry-rebuild stages independently; cached listings are a successful but
visible degraded mode.

The Streamlit Company Registry page reads only the persisted registry for
normal display and filtering. It exposes metrics, native filters, searchable
and selectable rows, CSV download, timestamps, cache warnings, price-history
details, and an explicit refresh action. Dataset Explorer and Stock Explorer
apply registry filters only when the user opts in, so historical data remains
accessible by default.

## Reconciliation with local market history

The inspected master file contained 702 symbols and 10,536 rows covering
2026-07-03 through 2026-07-27. It overlapped 501 of the 563 official symbols;
201 master symbols were history-only and 62 official symbols had no local
history. An outer merge therefore reconciles both sources without data loss.
Most history-only symbols matched dated contract patterns, confirming that the
historical endpoint contains non-ordinary instruments and that the registry
must not assume every symbol is a company ordinary share.

## Testing

The offline suite covers saved listing HTML parsing, numeric-looking symbols,
header reordering and missing columns, response-type validation, atomic
snapshots, duplicate handling, zero-row protection, cache fallback, outer
merges, listed-but-inactive and historical-only behaviour, recent/new boundary
configuration, deterministic output, missing company names, security types,
validated overrides, registry preservation on total listing failure, training
universe selection, dashboard filters, and four-stage automation integration.
No automated test contacts the live PSX website.

## Limitations

- The inspected current listing tables provide listed and non-compliant
  segments but not a comprehensive official delisting history or a separate
  dependable suspension indicator. Delisted and suspended statuses require
  reliable future source data or a documented override.
- Local history currently spans a short period. `first_seen_date` is therefore
  an observation boundary, not necessarily the official listing date, and the
  new-listing count is provisional.
- Security-type inference is intentionally conservative and may leave uncertain
  records as `unknown` or `other`.
- Company-name metadata cannot be fabricated for historical-only symbols that
  have no official-current match or curated evidence.

## Result

Milestone 2C provides a rebuildable, source-aware registry without modifying
historical price data. It makes official listing state, local market activity,
instrument type and evidence-based exceptions available to the pipeline,
automation, dashboard and future training-universe selection while explicitly
avoiding guessed delisting claims.

A live sample build on 2026-07-30 used the official source (not cache) and
produced 764 unique registry symbols: 563 currently listed, 702 recently traded,
62 listed but not recently traded, 501 provisionally new by the local
first-observed rule, 201 historical-only, 92 non-compliant, 0 suspended, 0
officially delisted, and 0 unknown lifecycle classifications. The 501 value is
not an assertion that these securities officially listed within 30 days; it is
high because the currently built master history begins on 2026-07-03, as noted
in the limitations above.
