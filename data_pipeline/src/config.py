"""Configuration values and filesystem paths for the PSX pipeline."""

from pathlib import Path


PSX_HISTORICAL_URL = "https://dps.psx.com.pk/historical"
REQUEST_TIMEOUT_SECONDS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_HTML_DIR = RAW_DATA_DIR / "html"
RAW_CSV_DIR = RAW_DATA_DIR / "csv"
REJECTED_DATA_DIR = DATA_DIR / "rejected"
