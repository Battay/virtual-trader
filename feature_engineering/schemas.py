"""Stable schemas and structured results for PSX AI data preparation."""

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

from market_intelligence.feature_joiner import MARKET_CONTEXT_COLUMNS


RAW_REQUIRED_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume")
RAW_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_DERIVED_COLUMNS = (
    "simple_return",
    "log_return",
    "high_low_range",
    "open_close_return",
    "rolling_volatility_20",
)
INDICATOR_COLUMNS = (
    "sma_20",
    "sma_50",
    "ema_20",
    "ema_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_middle",
    "bollinger_upper",
    "bollinger_lower",
    "atr_14",
    "obv",
    "volume_ma_20",
)
FEATURE_COLUMNS = (*RAW_OHLCV_COLUMNS, *PRICE_DERIVED_COLUMNS, *INDICATOR_COLUMNS)
METADATA_COLUMNS = (
    "is_active",
    "official_status",
    "lifecycle_status",
    "security_type",
)
AI_DATASET_COLUMNS = (
    "symbol",
    "date",
    *FEATURE_COLUMNS,
    *METADATA_COLUMNS,
    *MARKET_CONTEXT_COLUMNS,
    "feature_version",
)
DEFAULT_MASTER_SECURITY_TYPES = frozenset(
    {"ordinary_equity", "preference_share", "gem_equity", "etf", "other"}
)
FEATURE_WARMUP_ROWS = 49

_FEATURE_SPECIFICATION = {
    "milestone": "4a_market_context",
    "price_returns": "one_period_backward",
    "rolling_volatility_window": 20,
    "sma_windows": [20, 50],
    "ema_windows": [20, 50],
    "rsi_window": 14,
    "rsi_smoothing": "wilder_ewm",
    "macd": [12, 26, 9],
    "bollinger": [20, 2.0, "population_std"],
    "atr_window": 14,
    "obv_initial": 0,
    "volume_ma_window": 20,
    "warmup_rows": FEATURE_WARMUP_ROWS,
}
FEATURE_VERSION = "psx-4a-" + hashlib.sha256(
    json.dumps(_FEATURE_SPECIFICATION, sort_keys=True).encode("utf-8")
).hexdigest()[:12]


@dataclass(frozen=True)
class DatasetBuildMetrics:
    """Metrics returned after building one or more AI-ready datasets."""

    input_rows: int
    output_rows: int
    unique_symbols: int
    symbols_skipped: tuple[str, ...]
    warmup_rows_removed: int
    missing_rows: int
    earliest_date: date | None
    latest_date: date | None
    feature_version: str
    output_paths: tuple[Path, ...]
    market_context_included: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metrics for CLI and dashboard output."""
        values = asdict(self)
        values["earliest_date"] = (
            self.earliest_date.isoformat() if self.earliest_date else None
        )
        values["latest_date"] = (
            self.latest_date.isoformat() if self.latest_date else None
        )
        values["output_paths"] = tuple(str(path) for path in self.output_paths)
        return values


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation result for a processed AI dataset."""

    valid: bool
    rows: int
    unique_symbols: int
    earliest_date: date | None
    latest_date: date | None
    errors: tuple[str, ...]
