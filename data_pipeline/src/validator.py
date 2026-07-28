"""Validation rules for parsed PSX equity records."""

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias


ValidatedRecord: TypeAlias = dict[str, Any]
REQUIRED_FIELDS = (
    "symbol",
    "date",
    "ldcp",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_percent",
    "volume",
)


def validate_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[ValidatedRecord], list[ValidatedRecord]]:
    """Separate valid records from records that violate market data rules."""
    valid: list[ValidatedRecord] = []
    rejected: list[ValidatedRecord] = []

    for source_record in records:
        record = dict(source_record)
        reasons = [
            f"missing required field: {field}"
            for field in REQUIRED_FIELDS
            if field not in record or record[field] is None
        ]

        if "symbol" in record and not str(record["symbol"]).strip():
            reasons.append("symbol must be non-empty")

        if record.get("high") is not None and record.get("low") is not None:
            try:
                if record["high"] < record["low"]:
                    reasons.append("high must be greater than or equal to low")
            except TypeError:
                reasons.append("high and low must be comparable numbers")

        if record.get("volume") is not None:
            try:
                if record["volume"] < 0:
                    reasons.append("volume must be non-negative")
            except TypeError:
                reasons.append("volume must be a number")

        if reasons:
            record["reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            valid.append(record)

    return valid, rejected
