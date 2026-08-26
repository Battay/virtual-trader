"""Configuration values and filesystem paths for the PSX pipeline."""

from datetime import date
from pathlib import Path


PSX_HISTORICAL_URL = "https://dps.psx.com.pk/historical"
PSX_HISTORICAL_MIN_DATE = date(2016, 7, 26)
PSX_LISTINGS_TABLE_URL_TEMPLATE = (
    "https://dps.psx.com.pk/listings-table/{board}/{segment}"
)
REQUEST_TIMEOUT_SECONDS = 30
PROJECT_TIMEZONE = "Asia/Karachi"
RECENT_TRADING_WINDOW_DAYS = 30
NEW_LISTING_WINDOW_DAYS = 30
AI_MINIMUM_USABLE_ROWS = 252

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PSX_MARKET_PARQUET_ENV_VAR = "PSX_MARKET_PARQUET_PATH"
DEFAULT_PSX_MARKET_PARQUET_PATH = (
    PROJECT_ROOT.parent / "psx-data-sync" / "data" / "parquet" / "market.parquet"
)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_HTML_DIR = RAW_DATA_DIR / "html"
RAW_CSV_DIR = RAW_DATA_DIR / "csv"
REJECTED_DATA_DIR = DATA_DIR / "rejected"
MASTER_DATA_DIR = DATA_DIR / "master"
MASTER_CSV_PATH = MASTER_DATA_DIR / "psx_master.csv"
COMPANY_REGISTRY_PATH = MASTER_DATA_DIR / "company_registry.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_UNIVERSES_DIR = PROCESSED_DATA_DIR / "universes"
CURRENT_COMMON_EQUITY_UNIVERSE_PATH = (
    PROCESSED_UNIVERSES_DIR / "current_common_equity.csv"
)
SOFT_RELATIONSHIP_REPRESENTATION_DIR = (
    PROCESSED_UNIVERSES_DIR / "soft_relationship_nmf_v1"
)
PROCESSED_SYMBOLS_DIR = PROCESSED_DATA_DIR / "symbols"
PROCESSED_MASTER_DIR = PROCESSED_DATA_DIR / "master"
PROCESSED_MASTER_PATH = PROCESSED_MASTER_DIR / "psx_ai_master.csv"
PROCESSED_SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
CANONICAL_RECURRENT_TRAIN_V2_DIR = (
    PROCESSED_DATA_DIR / "canonical_recurrent_train_v2"
)
MODELS_DATA_DIR = DATA_DIR / "models"
MODEL_REGISTRY_PATH = MODELS_DATA_DIR / "model_registry.csv"
TRAINING_RUNS_DIR = DATA_DIR / "training_runs"
SAVED_MODELS_DIR = PROJECT_ROOT / "reinforcement_learning" / "saved_models"
SYMBOL_MODELS_DIR = SAVED_MODELS_DIR / "symbol_models"
MASTER_MODELS_DIR = SAVED_MODELS_DIR / "master_models"
METADATA_DIR = DATA_DIR / "metadata"
BACKFILL_STATE_PATH = METADATA_DIR / "backfill_state.json"
LISTINGS_METADATA_DIR = METADATA_DIR / "listings"
CURRENT_LISTINGS_PATH = LISTINGS_METADATA_DIR / "current_listings.csv"
COMPANY_OVERRIDES_PATH = METADATA_DIR / "company_overrides.csv"
AUTOMATION_CONFIG_PATH = METADATA_DIR / "automation.json"
AUTOMATION_LOCK_PATH = METADATA_DIR / "auto_update.lock"
LOGS_DIR = PROJECT_ROOT / "data_pipeline" / "logs"
AUTO_UPDATE_LOG_PATH = LOGS_DIR / "auto_update.log"
INDICES_DATA_DIR = DATA_DIR / "indices"
INDICES_RAW_DIR = INDICES_DATA_DIR / "raw"
INDICES_MASTER_DIR = INDICES_DATA_DIR / "master"
INDICES_METADATA_DIR = INDICES_DATA_DIR / "metadata"
INDICES_MASTER_PATH = INDICES_MASTER_DIR / "psx_indices_master.csv"
INDICES_REFRESH_METADATA_PATH = INDICES_METADATA_DIR / "refresh.json"
