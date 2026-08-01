# Milestone 2D — Streamlit UI/UX Refinement

## Objective

This milestone improves the usability and presentation of the existing PSX
Virtual Trader dashboard without changing scraping, master-data construction,
company-registry classification, automation behaviour, stored enums, or raw CSV
formats. The work focuses on information hierarchy, readable labels, safe value
formatting, responsive metrics, and reduced filter clutter.

## Usability problems identified

The direct code and data review confirmed the reported usability issues:

- backend values such as `historical_only`, `ordinary_equity`, and
  `never_seen_in_market_history` were exposed directly in widgets and tables;
- Stock Explorer selected by bare symbol, which was especially unclear for
  numeric-looking symbols when official name metadata existed;
- six fixed metric columns compressed dates and large volume values;
- the latest date competed with price metrics, total volume was shown instead
  of latest-session volume, and daily change was absent;
- Stock Explorer lacked a profile header, period control, readable metadata,
  curated table headings, newest-first rows, and download action;
- Company Registry presented a long flat filter form, raw timestamps, raw
  status text, a combined risk count, and insufficient explanation of metric
  overlap;
- the “Recently Traded” registry card counted all recent local-history symbols,
  including historical-only instruments, making it appear inconsistent beside
  “Currently Listed.”

## Screenshots and review evidence

No screenshot image file accompanied this milestone request; only the written
usability review was attached. The implementation therefore does not claim a
pixel-level screenshot comparison. It was based on direct inspection of the
running-page source, installed Streamlit 1.60 API, current registry/master data,
and the reported visible symptoms: truncated dates and volumes, unclear stock
selection, excessive filters, raw snake-case labels, and unclear metric
meanings.

Recommended before/after screenshots for the final FYP report are listed in the
delivery checklist rather than fabricated here.

## Information hierarchy decisions

Stock Explorer now flows from page purpose to security selection, identity and
status, latest observation, responsive metrics, price history, security
information, and finally detailed prices/download. The latest trading date is a
caption outside the metric row, so it remains visible at narrow widths. Metric
cards are arranged in responsive horizontal containers rather than a rigid
six-column grid.

Company Registry starts with a concise source/activity explanation and keeps
the deeper methodology in an expander. Refresh state and timestamps follow,
then primary and secondary metric groups, grouped sidebar filters, the
selectable table/download, and a non-empty security detail panel.

Native Streamlit elements preserve keyboard access and dark-mode compatibility:
`st.badge`, `st.metric`, responsive containers, expanders, selectboxes,
multiselects, segmented controls, dataframe selection, and download buttons.
No custom CSS, unsafe HTML, JavaScript, or external UI dependency was needed.

## Enum presentation strategy

`dashboard/presentation.py` defines explicit domain labels for important
official, activity, lifecycle, security-type, segment, and automation values.
Unknown future snake-case values use a safe title-cased fallback. Widgets use a
small `DisplayOption` object containing both backend value and readable label;
filtering extracts the original values after selection. Persisted data and raw
downloads therefore remain machine-friendly.

Status colors are secondary cues only. Every native badge includes complete
text, so meaning does not depend on color. Missing values use an em dash and
never expose `nan`, `NaT`, `None`, or Python object representations.

## Number and date formatting strategy

Dates display as `30 Jul 2026`. Timestamps are normalized to the configured
Pakistan timezone and display as `30 Jul 2026, 5:13 PM PKT`. Prices use PKR and
two-decimal precision in headline metrics. Percentages and change values retain
their sign. Large metric volumes use compact one-decimal notation such as
`80.4M` or `925.3K`, with the full latest volume available in metric help text.
Detailed tables use full comma-separated volume and integer values.

Formatting operates on presentation copies. Source dataframes and generated CSV
formats are not mutated or rewritten.

## Stock Explorer redesign

The security selector now searches formatted labels such as `MCB Bank Limited
(MCB)` and falls back to the symbol only when name metadata is unavailable. The
selected profile displays company/security name, symbol, sector, lifecycle,
official status, and security type without exposing backend enums.

The metric area contains Latest Close, Daily Change, Latest Open, selected-period
High, selected-period Low, and Latest Volume. Positive and negative changes use
signed text and Streamlit's native delta indicator. The price-history control
supports 1M, 3M, 6M, 1Y, and All relative to the latest available local trading
date. Chart axes, latest observation, details, newest-first price columns, and
the selected-period CSV download use readable terminology.

The left sidebar contains one collapsed Security Filters expander with two
toggles, readable lifecycle/security-type multiselects, and a clear action.

## Company Registry redesign

Filters are split into Search, Status, Classification, and Trading History
expanders. A minimum-trading-days filter was added in the dashboard helper; it
does not alter registry construction. Every enum option is displayed through
the presentation mapping while filtering still uses raw values.

The primary cards show Total Securities, Currently Listed, Listed & Recently
Traded, and Historical Only. Secondary cards separately show Listed, Not
Recently Traded; New Listings; Suspended; Non-Compliant; Officially Delisted;
and Unknown. Tooltips explain non-obvious definitions and the local-history
nature of “new.” The table has readable headings, formatted dates, integer
counts, em-dash missing values, row selection, raw filtered CSV download, and a
clean detail panel that omits unavailable optional fields.

## Metric wording decision

The backend `recently_traded` metric remains unchanged: it counts every registry
symbol observed within the recent local-history window. In the inspected data,
that was 702 symbols, including 201 historical-only symbols, while 563 symbols
were currently listed.

To remove the UI contradiction without silently redefining backend behaviour,
the dashboard computes a separate presentation count:

```text
officially_listed AND activity_status == recently_traded
```

This produces 501 for the inspected registry and is labelled “Listed & Recently
Traded.” The original 702 backend count remains available to programmatic
consumers and generated registry results.

## Testing

Offline tests cover explicit enum mappings, fallback humanization, missing
values, dates, Pakistan-time timestamps, integers, decimals, prices,
percentages, volume abbreviation, company/symbol selector labels, filter-option
value/label separation, absence of snake case in option labels, Stock Explorer
period filtering, source-data immutability, minimum trading days, and the
currently-listed/recent registry presentation metric. Existing pipeline,
registry, automation, and dashboard-helper tests remain part of the full suite.
No browser automation or live PSX request is used by tests.

## Limitations

- Security and company names depend on current registry metadata; a
  historical-only symbol with no name evidence still displays its symbol.
- Period controls are relative to the latest locally available observation, not
  the current wall-clock date.
- The native Streamlit chart intentionally remains a simple close-price line;
  indicators and advanced chart interactions are outside this milestone.
- Very wide registry tables still require horizontal scrolling, so identity
  columns are pinned and additional metadata is kept in a detail expander.

## Result

Milestone 2D creates a consistent presentation boundary across the multipage
dashboard. Users receive readable labels, dates, timestamps, prices, volume,
status context, clearer metrics, compact filters, and downloadable curated views
while all backend enums, raw files, lifecycle rules, pipeline operations, and
automation behaviour remain unchanged.
