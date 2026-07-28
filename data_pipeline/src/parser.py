"""Parsing utilities for PSX historical market HTML."""

from datetime import date
import logging
from typing import TypeAlias

from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)
MarketRecord: TypeAlias = dict[str, str | float | int]
RejectedRecord: TypeAlias = dict[str, str]

FIELD_NAMES = (
    "symbol",
    "ldcp",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
)


def _cell_value(cell: object) -> str:
    """Read a cell's data-value attribute, falling back to visible text."""
    data_value = cell.get("data-value")  # type: ignore[attr-defined]
    value = data_value if data_value is not None else cell.get_text(strip=True)  # type: ignore[attr-defined]
    return str(value).strip()


def _numeric_text(value: str, field: str) -> str:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        raise ValueError(f"{field} is empty")
    return cleaned


def parse_market_html(
    html: str, trading_date: date
) -> tuple[list[MarketRecord], list[RejectedRecord]]:
    """Parse equity rows, returning parsed records and malformed rows separately."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[MarketRecord] = []
    rejected: list[RejectedRecord] = []

    for row_number, row in enumerate(soup.select('tr[data-type="equity"]'), start=1):
        values = [_cell_value(cell) for cell in row.find_all("td")]
        if len(values) != len(FIELD_NAMES):
            reason = f"expected 9 columns, found {len(values)}"
            LOGGER.warning("Skipping equity row %d: %s", row_number, reason)
            rejected.append(
                {
                    "date": trading_date.isoformat(),
                    "row_number": str(row_number),
                    "reason": reason,
                    "raw_values": " | ".join(values),
                }
            )
            continue

        raw = dict(zip(FIELD_NAMES, values, strict=True))
        try:
            record: MarketRecord = {
                "symbol": raw["symbol"].strip(),
                "date": trading_date.isoformat(),
                "ldcp": float(_numeric_text(raw["ldcp"], "ldcp")),
                "open": float(_numeric_text(raw["open"], "open")),
                "high": float(_numeric_text(raw["high"], "high")),
                "low": float(_numeric_text(raw["low"], "low")),
                "close": float(_numeric_text(raw["close"], "close")),
                "change": float(_numeric_text(raw["change"], "change")),
                "change_percent": float(
                    _numeric_text(raw["change_percent"].rstrip("%"), "change_percent")
                ),
                "volume": int(_numeric_text(raw["volume"], "volume")),
            }
        except (TypeError, ValueError) as exc:
            reason = f"numeric conversion failed: {exc}"
            LOGGER.warning("Skipping equity row %d: %s", row_number, reason)
            rejected.append(
                {
                    "date": trading_date.isoformat(),
                    "row_number": str(row_number),
                    "reason": reason,
                    "raw_values": " | ".join(values),
                }
            )
            continue
        records.append(record)

    return records, rejected
