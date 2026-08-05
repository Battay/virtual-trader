"""Validate and normalize official PSX positional index observations."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from data_pipeline.src.config import PROJECT_TIMEZONE

from .index_config import INDEX_SOURCE, require_supported_index

INDEX_COLUMNS = (
    "index_code", "index_name", "date", "timestamp", "value", "volume",
    "open", "daily_change", "daily_change_percent", "source", "fetched_at",
)


@dataclass(frozen=True)
class IndexParseResult:
    data: pd.DataFrame
    rejected: tuple[dict[str, object], ...]


def _integer(value: object, field: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be integer-like")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be integer-like")
    result = int(number)
    if non_negative and result < 0:
        raise ValueError(f"{field} cannot be negative")
    return result


def _number(value: object, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def parse_index_series(
    payload: dict[str, Any],
    index_code: str,
    *,
    fetched_at: datetime | None = None,
) -> IndexParseResult:
    definition = require_supported_index(index_code)
    observations = payload.get("data")
    if not isinstance(observations, list):
        raise ValueError("payload data must be a list")
    timezone = ZoneInfo(PROJECT_TIMEZONE)
    fetched = fetched_at or datetime.now(timezone)
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone)
    fetched_text = fetched.astimezone(timezone).isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for position, observation in enumerate(observations):
        try:
            if not isinstance(observation, (list, tuple)) or len(observation) != 4:
                raise ValueError("observation must contain exactly four fields")
            timestamp = _integer(observation[0], "timestamp", non_negative=True)
            trading_date = datetime.fromtimestamp(timestamp, timezone).date()
            rows.append(
                {
                    "index_code": definition.code,
                    "index_name": definition.display_name,
                    "date": trading_date.isoformat(),
                    "timestamp": timestamp,
                    "value": _number(observation[1], "value"),
                    "volume": _integer(observation[2], "volume", non_negative=True),
                    "open": _number(observation[3], "open"),
                    "source": INDEX_SOURCE,
                    "fetched_at": fetched_text,
                    "_source_position": position,
                }
            )
        except (TypeError, ValueError, OSError, OverflowError) as exc:
            rejected.append(
                {"index_code": definition.code, "position": position,
                 "reason": str(exc), "raw": observation}
            )
    if not rows:
        empty = pd.DataFrame(columns=INDEX_COLUMNS)
        return IndexParseResult(empty, tuple(rejected))
    data = pd.DataFrame(rows)
    data = (
        data.sort_values(
            ["date", "timestamp", "_source_position"],
            ascending=[True, True, False],
            kind="stable",
        )
        .drop_duplicates(["index_code", "date"], keep="last")
        .sort_values(["index_code", "date"], kind="stable")
        .reset_index(drop=True)
    )
    previous = data.groupby("index_code", sort=False)["value"].shift(1)
    data["daily_change"] = data["value"] - previous
    data["daily_change_percent"] = data["daily_change"].div(previous).mul(100)
    return IndexParseResult(data.loc[:, INDEX_COLUMNS], tuple(rejected))
