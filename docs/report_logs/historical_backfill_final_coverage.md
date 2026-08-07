# Historical Backfill Final Coverage

- Report timestamp (Asia/Karachi): `2026-08-08T02:25:35+05:00`
- Requested range: **2016-07-26 to 2026-08-05**
- Total calendar dates: **3,663**

## Final coverage

| Category | Count | Percentage of requested range | State treatment |
|---|---:|---:|---|
| Stored trading dates | 2,485 | 67.8406% | Successful local CSV |
| Confirmed non-trading dates already resolved in state | 1,046 | 28.5558% | `non_trading_dates` |
| Verification-confirmed non-trading weekdays | 92 | 2.5116% | Still temporary; not automatically promoted |
| **Combined confirmed non-trading evidence** | **1,138** | **31.0674%** | Resolved state dates plus verification-confirmed weekdays |
| Likely non-trading weekdays | 39 | 1.0647% | Still temporary; not promoted |
| Unresolved suspected trading-data gaps | 1 | 0.0273% | Explicitly temporary/unresolved |
| Failed dates | 0 | 0.0000% | None |

The mutually exclusive calendar partition is: **2,485 stored trading + 1,046 state-resolved non-trading + 92 verification-confirmed temporary + 39 likely non-trading temporary + 1 unresolved trading gap + 0 failed = 3,663 dates**.

- Operational state coverage (`stored trading + state-resolved non-trading`): **3,531/3,663 (96.3964%)**.
- Evidence coverage including confirmed and likely calendar verification: **3,662/3,663 (99.9727%)**.
- Remaining suspected trading-data gap: **1/3,663 (0.0273%)**.

## Targeted reconciliation results

### 2026-08-05 — recovered

The existing `data/raw/csv/market_2026-08-05.csv` was validated locally with **596 valid rows**. The existing safe reconciliation workflow recorded it as successful without downloading it again. Its stale “Today is not final” temporary entry was removed, its success record is marked `reconciled: true`, and `last_successful_date` is now `2026-08-05`.

The targeted manual collection result supplied for this date was: 596 parsed, 596 valid, and 0 rejected rows. The reconciliation itself relied on the existing CSV validator and did not invoke HTTP.

### 2023-04-14 — unresolved historical source gap

This date remains in `temporary_skips`. Its reason now explicitly records that repeated official PSX historical responses contained no equity rows, that the date is absent from the official PSX 2023 holiday calendar, and that it must not be treated as confirmed non-trading.

No CSV exists for this date and no data was fabricated. Recommended action remains a separate manual source investigation; it was not retried during reconciliation.

## Methodology

Stored dates are the union of `successful_dates` and `already_downloaded_dates`. Existing state-resolved non-trading dates come from `non_trading_dates`. The 92 confirmed, 39 likely, and remaining trading-gap classifications come from `historical_backfill_calendar_verification.csv`, restricted to dates still temporary after reconciling 2026-08-05. Categories are kept separate so verification findings do not silently mutate operational state.

The 2026-08-05 reconciliation used the production `run_backfill(..., resume=True)` path with fail-fast transport and processing stubs. Its plan contained zero request dates; the only reconciliation outcome was the existing 596-row CSV. The 2023-04-14 message was persisted atomically through the existing state writer without changing its temporary classification.

## Data-integrity statement

No live HTTP requests were made. No temporary dates were retried. No data products were rebuilt and no models were trained. No raw HTML or CSV file was deleted, changed, or overwritten. The 39 likely non-trading dates were not promoted. The only state changes were reconciliation of the existing 2026-08-05 CSV and clarification of the existing 2023-04-14 temporary reason. No commit was created.
