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
MASTER_DATA_DIR = DATA_DIR / "master"
MASTER_CSV_PATH = MASTER_DATA_DIR / "psx_master.csv"
METADATA_DIR = DATA_DIR / "metadata"
AUTOMATION_CONFIG_PATH = METADATA_DIR / "automation.json"
AUTOMATION_LOCK_PATH = METADATA_DIR / "auto_update.lock"
LOGS_DIR = PROJECT_ROOT / "data_pipeline" / "logs"
AUTO_UPDATE_LOG_PATH = LOGS_DIR / "auto_update.log"
