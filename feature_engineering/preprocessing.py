"""Market-data validation, eligibility, metadata, and scaling preparation."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schemas import FEATURE_COLUMNS, RAW_OHLCV_COLUMNS, RAW_REQUIRED_COLUMNS
from .storage import atomic_dump_joblib, atomic_write_json


class DataQualityError(ValueError):
    """Raised when required market or registry structure is malformed."""


@dataclass(frozen=True)
class ScalingResult:
    """Training-fitted scaler and independently transformed data partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler
    scaled_features: tuple[str, ...]
    training_rows: int


def validate_required_market_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when required raw market columns are absent."""
    missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise DataQualityError(
            f"Market data is missing required columns: {', '.join(missing)}"
        )


def fatal_quality_errors_by_symbol(
    market_data: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    """Return fatal, symbol-local data-quality errors without mutating input."""
    validate_required_market_columns(market_data)
    if market_data.empty:
        return {}

    data = market_data.copy()
    data["symbol"] = data["symbol"].astype("string").str.strip()
    errors: dict[str, tuple[str, ...]] = {}
    invalid_symbols = data["symbol"].isna() | (data["symbol"] == "")
    if invalid_symbols.any():
        errors["<missing>"] = ("symbol is empty",)

    data = data.loc[~invalid_symbols]
    for symbol, group in data.groupby("symbol", sort=True):
        symbol_errors: list[str] = []
        dates = pd.to_datetime(group["date"], errors="coerce")
        numeric = group.loc[:, RAW_OHLCV_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if dates.isna().any():
            symbol_errors.append("invalid trading date")
        if numeric.isna().any(axis=None):
            symbol_errors.append("missing or invalid OHLCV value")
        if dates.duplicated().any():
            symbol_errors.append("duplicate trading date")
        if (numeric["high"] < numeric["low"]).any():
            symbol_errors.append("high is lower than low")
        if (numeric["volume"] < 0).any():
            symbol_errors.append("volume is negative")
        if (numeric[["open", "high", "low", "close"]] <= 0).any(axis=None):
            symbol_errors.append("price is not positive")
        if symbol_errors:
            errors[str(symbol)] = tuple(symbol_errors)
    return errors


def attach_registry_metadata(
    featured_data: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Attach model metadata and derive active status from registry evidence."""
    data = featured_data.copy()
    if data.empty:
        for column, dtype in (
            ("is_active", "bool"),
            ("official_status", "string"),
            ("lifecycle_status", "string"),
            ("security_type", "string"),
        ):
            data[column] = pd.Series(dtype=dtype)
        return data

    required_registry = {
        "symbol",
        "officially_listed",
        "activity_status",
        "official_status",
        "lifecycle_status",
        "security_type",
    }
    missing = sorted(required_registry.difference(registry.columns))
    if missing:
        raise DataQualityError(
            f"Company registry is missing required columns: {', '.join(missing)}"
        )

    metadata = registry.loc[:, sorted(required_registry)].copy()
    metadata["symbol"] = metadata["symbol"].astype("string").str.strip()
    listed = metadata["officially_listed"]
    if not pd.api.types.is_bool_dtype(listed):
        listed = listed.astype("string").str.lower().isin({"true", "1"})
    metadata["is_active"] = listed & (
        metadata["activity_status"].astype("string") == "recently_traded"
    )
    metadata = metadata.drop_duplicates("symbol", keep="last")
    merged = data.merge(
        metadata[
            [
                "symbol",
                "is_active",
                "official_status",
                "lifecycle_status",
                "security_type",
            ]
        ],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    merged["is_active"] = merged["is_active"].fillna(False).astype(bool)
    for column in ("official_status", "lifecycle_status", "security_type"):
        merged[column] = merged[column].astype("string").fillna("unknown")
    return merged


def symbol_eligibility_table(
    processed_data: pd.DataFrame,
    *,
    minimum_usable_rows: int,
    fatal_symbols: Collection[str] = (),
) -> pd.DataFrame:
    """Evaluate default symbol-model eligibility from processed rows."""
    if minimum_usable_rows < 1:
        raise ValueError("minimum usable rows must be at least 1")
    required = {"symbol", "is_active", "security_type"}
    missing = sorted(required.difference(processed_data.columns))
    if missing:
        raise DataQualityError(
            f"Processed data is missing eligibility columns: {', '.join(missing)}"
        )
    fatal = set(str(symbol) for symbol in fatal_symbols)
    records: list[dict[str, object]] = []
    for symbol, group in processed_data.groupby("symbol", sort=True):
        symbol_text = str(symbol)
        is_active = bool(group["is_active"].iloc[-1])
        security_type = str(group["security_type"].iloc[-1])
        usable_rows = len(group)
        reason = "Eligible"
        eligible = True
        if symbol_text in fatal:
            eligible = False
            reason = "Data Quality Issue"
        elif not is_active:
            eligible = False
            reason = "Inactive or Not Listed"
        elif security_type != "ordinary_equity":
            eligible = False
            reason = "Unsupported Security Type"
        elif usable_rows < minimum_usable_rows:
            eligible = False
            reason = "Insufficient History"
        records.append(
            {
                "symbol": symbol_text,
                "is_active": is_active,
                "security_type": security_type,
                "usable_rows": usable_rows,
                "eligible": eligible,
                "eligibility_reason": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def fit_training_scaler(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> ScalingResult:
    """Fit StandardScaler on training rows and transform all partitions."""
    columns = tuple(feature_columns)
    if train.empty:
        raise DataQualityError("training data cannot be empty when fitting a scaler")
    missing = sorted(
        set(columns).difference(train.columns)
        | set(columns).difference(validation.columns)
        | set(columns).difference(test.columns)
    )
    if missing:
        raise DataQualityError(
            f"Split data is missing scalable columns: {', '.join(missing)}"
        )
    for label, partition in (
        ("training", train),
        ("validation", validation),
        ("test", test),
    ):
        values = partition.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataQualityError(f"{label} features contain missing or non-finite values")

    scaler = StandardScaler()
    float_dtypes = {column: "float64" for column in columns}
    scaled_train = train.copy().astype(float_dtypes)
    scaled_validation = validation.copy().astype(float_dtypes)
    scaled_test = test.copy().astype(float_dtypes)
    scaled_train[list(columns)] = scaler.fit_transform(train.loc[:, columns])
    scaled_validation[list(columns)] = scaler.transform(validation.loc[:, columns])
    scaled_test[list(columns)] = scaler.transform(test.loc[:, columns])
    return ScalingResult(
        train=scaled_train,
        validation=scaled_validation,
        test=scaled_test,
        scaler=scaler,
        scaled_features=columns,
        training_rows=len(train),
    )


def save_scaler_artifact(result: ScalingResult, path: Path) -> tuple[Path, Path]:
    """Save a fitted scaler and transparent JSON metadata atomically."""
    scaler_path = Path(path)
    metadata_path = scaler_path.with_suffix(".json")
    atomic_dump_joblib(result.scaler, scaler_path)
    atomic_write_json(
        {
            "scaled_features": list(result.scaled_features),
            "training_rows": result.training_rows,
            "training_mean": result.scaler.mean_.tolist(),
            "training_scale": result.scaler.scale_.tolist(),
        },
        metadata_path,
    )
    return scaler_path, metadata_path
