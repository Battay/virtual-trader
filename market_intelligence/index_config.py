"""Fixed official PSX index registry and project-relative paths."""

from dataclasses import dataclass
from pathlib import Path

from data_pipeline.src.config import (
    INDICES_MASTER_DIR,
    INDICES_MASTER_PATH,
    INDICES_RAW_DIR,
    INDICES_REFRESH_METADATA_PATH,
    REQUEST_TIMEOUT_SECONDS,
)

INDEX_ENDPOINT_TEMPLATE = "https://dps.psx.com.pk/timeseries/eod/{index_code}"
INDEX_SOURCE = "official_psx_timeseries_eod"
INDEX_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class IndexDefinition:
    code: str
    display_name: str

    @property
    def raw_path(self) -> Path:
        return INDICES_RAW_DIR / f"{self.code}.json"

    @property
    def master_path(self) -> Path:
        return INDICES_MASTER_DIR / f"{self.code}.csv"


SUPPORTED_INDICES = {
    "KSE100": IndexDefinition("KSE100", "KSE-100 Index"),
    "KSE30": IndexDefinition("KSE30", "KSE-30 Index"),
    "KMI30": IndexDefinition("KMI30", "KMI-30 Index"),
    "ALLSHR": IndexDefinition("ALLSHR", "KSE All Share Index"),
}
SUPPORTED_INDEX_CODES = tuple(SUPPORTED_INDICES)
COMBINED_INDEX_MASTER_PATH = INDICES_MASTER_PATH
REFRESH_METADATA_PATH = INDICES_REFRESH_METADATA_PATH


def require_supported_index(index_code: str) -> IndexDefinition:
    code = str(index_code).strip().upper()
    try:
        return SUPPORTED_INDICES[code]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PSX index code {index_code!r}; expected one of "
            f"{', '.join(SUPPORTED_INDEX_CODES)}"
        ) from exc
