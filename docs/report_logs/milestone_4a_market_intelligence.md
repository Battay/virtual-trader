# Milestone 4A — Market intelligence and official PSX indices

## Objective and source investigation

This milestone adds official index history, descriptive breadth and health
analytics, dashboard views, automation, and optional AI market context without
mixing index observations into equities. PSX's portal uses
`GET /timeseries/eod/{index_code}` for `KSE100`, `KSE30`, `KMI30`, and `ALLSHR`.
The correct headline identifier is `KSE100`, not `PSX100`.

## Separate pipelines and normalization

The existing dated HTML equity pipeline is unchanged. Index acquisition accepts
only the fixed allowlist, validates JSON content type, envelope status, non-empty
data, and exactly four positional fields. Unix timestamps are retained and
converted through `Asia/Karachi`. Valid observations are sorted chronologically,
deduplicated on index code/date with the newest valid source version retained,
and used to compute backward-only change fields. Malformed observations are
reported separately; no high or low is invented.

Raw JSON, per-index normalized CSVs, combined index master data, and refresh
metadata live under `data/indices/`. Writes are atomic and full-series refreshes
are idempotent. One source failure does not stop other indices or overwrite a
valid cached series.

## Metrics, breadth, and Market Health

Index metrics use trading observations for weekly/monthly/quarterly/annual
returns, moving averages, volume participation, and 20-observation annualized
volatility. Insufficient inputs remain unavailable rather than becoming zero.

Breadth uses exactly the latest local equity date and reports the universe and
reference date. Its default universe is every valid security on that date; an
explicit registry-backed mode restricts it to currently listed ordinary
equities.

Market Health is rule based. Weights are: KSE-100 trend 15, KSE-30 trend 10,
KMI-30 trend 10, All Share trend 15, advance/decline ratio 15, advancing share
10, volume participation 10, moving-average position 10, and volatility penalty
5. Available weights are normalized to 100 when inputs are missing. Labels range
from Strongly Bearish to Strongly Bullish. The UI states that this is descriptive
and not financial advice.

## AI and automation integration

Market-context features join to securities by date while preserving one
symbol/date row. Exact-date joins are the default; optional backward filling is
bounded in calendar days and never observes a future index row. Dataset building
continues with missing context, records whether it was included, and uses a new
4A feature version. No model is trained or retrained automatically.

Scheduled updates retain the 17:15 Asia/Karachi policy. After equity acquisition
they refresh indices, rebuild the equity master and registry, and rebuild AI data
only when explicitly configured. Cached valid indices prevent a temporary index
source outage from destroying index data or unnecessarily blocking equity work.

## Dashboard and testing

Market Overview is the first page and presents index cards, breadth, system
status, and score explanations. Market Indices provides readable selection,
local date ranges, charts, tables, downloads, and selected/all refresh actions.
Missing data is handled explicitly.

Offline tests cover acquisition validation, parsing, Karachi dates, positional
schema rejection, deduplication, atomic/idempotent storage, partial failure,
metrics, breadth, health bounds/weights/labels, feature joins, leakage safety,
optional context, automation cache behavior, and readable navigation. Tests do
not contact PSX.

## Limitations and result

PSX does not accept requested date ranges and does not document retention.
Approximately five years was observed during investigation. The route is used by
the official portal but is not a versioned public API, so schema monitoring is
required. PSX licensing restrictions apply to dissemination and commercial use.

The result is a distinct, auditable official-index pipeline and reusable market
intelligence layer, integrated without changing equity schema, lifecycle rules,
backfill semantics, schedule time, or training thresholds.
