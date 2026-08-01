"""Dataset readiness and model staleness calculations."""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from feature_engineering.preprocessing import fatal_quality_errors_by_symbol
from feature_engineering.readiness import build_training_readiness_report
from feature_engineering.schemas import FEATURE_VERSION, FEATURE_WARMUP_ROWS

from .registry import latest_model_versions


TRAINING_STATUSES = (
    "never_trained",
    "up_to_date",
    "retraining_recommended",
    "insufficient_history",
    "data_quality_issue",
    "unsupported_security_type",
    "missing_processed_dataset",
    "training_failed",
)


def count_new_trading_dates(
    data: pd.DataFrame,
    training_data_end: object,
) -> int:
    """Count distinct trading dates after a model's complete-history cutoff."""
    if data.empty or "date" not in data or pd.isna(training_data_end):
        return 0
    cutoff = pd.to_datetime(training_data_end, errors="coerce")
    if pd.isna(cutoff):
        return 0
    dates = pd.to_datetime(data["date"], errors="coerce").dropna().drop_duplicates()
    return int((dates > cutoff).sum())


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def build_symbol_status_table(
    market_data: pd.DataFrame,
    registry: pd.DataFrame,
    model_registry: pd.DataFrame,
    *,
    minimum_usable_rows: int,
) -> pd.DataFrame:
    """Build one readiness/model-status row for every active registry symbol."""
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")
    fatal_errors = fatal_quality_errors_by_symbol(market_data)
    market = market_data.copy()
    market["symbol"] = market["symbol"].astype("string").str.strip()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    latest_models = latest_model_versions(model_registry)
    latest_symbol_models = latest_models.loc[
        latest_models.get("model_scope", pd.Series(dtype="string")).astype("string")
        == "symbol"
    ]
    model_by_symbol = {
        str(row["symbol"]): row
        for _, row in latest_symbol_models.iterrows()
    }
    records: list[dict[str, object]] = []
    for _, registry_row in registry.iterrows():
        symbol = str(registry_row.get("symbol", "")).strip()
        if not symbol:
            continue
        is_active = _as_bool(registry_row.get("officially_listed")) and str(
            registry_row.get("activity_status", "")
        ) == "recently_traded"
        if not is_active:
            continue
        history = market.loc[market["symbol"] == symbol].sort_values(
            "date",
            kind="stable",
        )
        dates = history["date"].dropna().drop_duplicates()
        usable_rows = max(0, len(dates) - FEATURE_WARMUP_ROWS)
        model = model_by_symbol.get(symbol)
        new_days = (
            count_new_trading_dates(history, model.get("training_data_end"))
            if model is not None
            else 0
        )
        security_type = str(registry_row.get("security_type", "unknown"))
        eligible = (
            symbol not in fatal_errors
            and security_type == "ordinary_equity"
            and usable_rows >= minimum_usable_rows
        )
        if symbol in fatal_errors:
            training_status = "data_quality_issue"
        elif security_type != "ordinary_equity":
            training_status = "unsupported_security_type"
        elif usable_rows < minimum_usable_rows:
            training_status = "insufficient_history"
        elif model is None or str(model.get("model_status")) == "not_trained":
            training_status = "never_trained"
        elif str(model.get("model_status")) == "failed":
            training_status = "training_failed"
        elif new_days > 0:
            training_status = "retraining_recommended"
        else:
            training_status = "up_to_date"
        records.append(
            {
                "symbol": symbol,
                "company_name": registry_row.get("company_name", ""),
                "sector": registry_row.get("sector", ""),
                "security_type": security_type,
                "is_active": True,
                "is_newly_added": _as_bool(registry_row.get("is_new_listing")),
                "data_start": dates.min().date() if not dates.empty else None,
                "data_end": dates.max().date() if not dates.empty else None,
                "usable_rows": usable_rows,
                "eligible": eligible,
                "model_version": (
                    model.get("model_version") if model is not None else pd.NA
                ),
                "last_trained_at": (
                    model.get("last_trained_at") if model is not None else ""
                ),
                "training_data_start": (
                    model.get("training_data_start") if model is not None else ""
                ),
                "training_data_end": (
                    model.get("training_data_end") if model is not None else ""
                ),
                "new_data_days": new_days,
                "needs_retraining": new_days > 0,
                "training_status": training_status,
                "quality_errors": "; ".join(fatal_errors.get(symbol, ())),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "symbol",
        kind="stable",
    ).reset_index(drop=True)


def build_model_readiness_table(
    market_data: pd.DataFrame,
    registry: pd.DataFrame,
    model_registry: pd.DataFrame,
    *,
    minimum_usable_rows: int,
    processed_symbols_dir: Path | None = None,
) -> pd.DataFrame:
    """Combine exact dataset readiness with future model lifecycle status."""
    status = build_symbol_status_table(
        market_data,
        registry,
        model_registry,
        minimum_usable_rows=minimum_usable_rows,
    )
    readiness_kwargs: dict[str, object] = {
        "minimum_usable_rows": minimum_usable_rows,
    }
    if processed_symbols_dir is not None:
        readiness_kwargs["processed_symbols_dir"] = processed_symbols_dir
    readiness = build_training_readiness_report(
        market_data,
        registry,
        **readiness_kwargs,
    )
    readiness_columns = [
        "symbol",
        "raw_trading_rows",
        "earliest_raw_date",
        "latest_raw_date",
        "warmup_rows_removed",
        "usable_feature_rows",
        "minimum_usable_rows",
        "additional_rows_required",
        "train_rows",
        "validation_rows",
        "test_rows",
        "readiness_status",
    ]
    combined = status.merge(
        readiness.loc[:, readiness_columns],
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    readiness_to_training = {
        "Insufficient History": "insufficient_history",
        "Data Quality Issue": "data_quality_issue",
        "Unsupported Security Type": "unsupported_security_type",
        "Missing Processed Dataset": "missing_processed_dataset",
    }
    blocked = combined["readiness_status"].map(readiness_to_training)
    combined.loc[blocked.notna(), "training_status"] = blocked.dropna()
    combined["eligible"] = combined["readiness_status"].eq("Ready")
    combined["data_start"] = combined["earliest_raw_date"]
    combined["data_end"] = combined["latest_raw_date"]
    combined["usable_rows"] = combined["usable_feature_rows"]
    return combined


def master_model_status(
    processed_data: pd.DataFrame,
    model_registry: pd.DataFrame,
) -> dict[str, object]:
    """Return separately tracked master-model dataset and staleness status."""
    dates = (
        pd.to_datetime(processed_data["date"], errors="coerce").dropna()
        if "date" in processed_data
        else pd.Series(dtype="datetime64[ns]")
    )
    models = latest_model_versions(model_registry)
    master_models = models.loc[
        models.get("model_scope", pd.Series(dtype="string")).astype("string")
        == "master"
    ]
    model = master_models.iloc[-1] if not master_models.empty else None
    new_days = (
        count_new_trading_dates(processed_data, model.get("training_data_end"))
        if model is not None
        else 0
    )
    if processed_data.empty:
        training_status = "insufficient_history"
    elif model is None or str(model.get("model_status")) == "not_trained":
        training_status = "never_trained"
    elif str(model.get("model_status")) == "failed":
        training_status = "training_failed"
    elif new_days:
        training_status = "retraining_recommended"
    else:
        training_status = "up_to_date"
    return {
        "model_status": (
            model.get("model_status") if model is not None else "not_trained"
        ),
        "training_status": training_status,
        "dataset_start": dates.min().date() if not dates.empty else None,
        "dataset_end": dates.max().date() if not dates.empty else None,
        "dataset_rows": len(processed_data),
        "symbols": (
            int(processed_data["symbol"].nunique())
            if "symbol" in processed_data
            else 0
        ),
        "feature_version": (
            str(processed_data["feature_version"].dropna().iloc[-1])
            if "feature_version" in processed_data
            and processed_data["feature_version"].notna().any()
            else FEATURE_VERSION
        ),
        "last_trained_at": (
            model.get("last_trained_at") if model is not None else ""
        ),
        "training_data_start": (
            model.get("training_data_start") if model is not None else ""
        ),
        "training_data_end": (
            model.get("training_data_end") if model is not None else ""
        ),
        "new_data_days": new_days,
        "needs_retraining": new_days > 0,
    }


def complete_history_training_metadata(
    complete_data: pd.DataFrame,
    split_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Create registry metadata whose training range covers all available history."""
    dates = pd.to_datetime(complete_data["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("complete training history cannot be empty")
    training = dict(split_metadata.get("training", {}))
    validation = dict(split_metadata.get("validation", {}))
    testing = dict(split_metadata.get("testing", {}))
    return {
        "training_data_start": dates.min().date().isoformat(),
        "training_data_end": dates.max().date().isoformat(),
        "validation_data_start": validation.get("start", ""),
        "validation_data_end": validation.get("end", ""),
        "test_data_start": testing.get("start", ""),
        "test_data_end": testing.get("end", ""),
        "training_rows": training.get("rows", 0),
        "validation_rows": validation.get("rows", 0),
        "test_rows": testing.get("rows", 0),
        "dataset_latest_date": dates.max().date().isoformat(),
    }
