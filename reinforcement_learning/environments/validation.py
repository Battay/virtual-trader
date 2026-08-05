"""Dataset and Gymnasium-contract validation for environment v1."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd

from feature_engineering.schemas import RAW_OHLCV_COLUMNS

from .config import ENVIRONMENT_VERSION


IDENTITY_COLUMNS = ("symbol", "date")
REQUIRED_MARKET_COLUMNS = RAW_OHLCV_COLUMNS


class EnvironmentDataError(ValueError):
    """Raised when a symbol dataset is unsafe for simulation."""


@dataclass(frozen=True)
class EnvironmentValidationResult:
    environment_version: str
    valid: bool
    observation_shape: tuple[int, ...] | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentReadiness:
    status: str
    message: str
    symbol: str | None
    rows: int


def prepare_single_symbol_data(
    data: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Validate and return a chronological deep copy of one symbol dataset."""
    required_columns = tuple(
        dict.fromkeys((*IDENTITY_COLUMNS, *REQUIRED_MARKET_COLUMNS, *feature_columns))
    )
    required = set(required_columns)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise EnvironmentDataError(
            f"Environment data is missing columns: {', '.join(missing)}"
        )
    if data.empty:
        raise EnvironmentDataError("Environment data cannot be empty")
    prepared = data.loc[
        :, list(dict.fromkeys((*required_columns, *data.columns)))
    ].copy(
        deep=True
    )
    prepared["symbol"] = prepared["symbol"].astype("string").str.strip()
    symbols = tuple(prepared["symbol"].dropna().unique())
    if len(symbols) != 1 or not symbols[0]:
        raise EnvironmentDataError("Environment data must contain exactly one symbol")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    if prepared["date"].isna().any():
        raise EnvironmentDataError("Environment data contains an invalid date")
    if prepared["date"].duplicated().any():
        raise EnvironmentDataError("Environment data contains duplicate trading dates")
    numeric_columns = tuple(dict.fromkeys((*REQUIRED_MARKET_COLUMNS, *feature_columns)))
    numeric = prepared.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise EnvironmentDataError(
            "Environment market and observation values must be finite"
        )
    if (numeric[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise EnvironmentDataError("Environment prices must be positive")
    if (numeric["volume"] < 0).any():
        raise EnvironmentDataError("Environment volume cannot be negative")
    if (numeric["high"] < numeric["low"]).any():
        raise EnvironmentDataError("Environment high cannot be below low")
    prepared.loc[:, numeric_columns] = numeric
    return prepared.sort_values("date", kind="stable").reset_index(drop=True)


def validate_environment(env: gym.Env) -> EnvironmentValidationResult:
    """Run Gymnasium's checker plus finite-observation and accounting checks."""
    errors: list[str] = []
    shape = None
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=7)
        shape = tuple(observation.shape)
        if observation.dtype != np.float32:
            errors.append("reset observation is not float32")
        if not np.isfinite(observation).all():
            errors.append("reset observation is not finite")
        if not env.observation_space.contains(observation):
            errors.append("reset observation is outside observation_space")
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = env.step(0)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                errors.append("episode produced a non-finite value")
                break
            if env.cash < -1e-8 or env.shares_held < 0:
                errors.append("episode produced negative cash or shares")
                break
            expected = env.cash + env.current_position_value
            if not np.isclose(env.total_portfolio_value, expected):
                errors.append("portfolio accounting identity failed")
                break
    except Exception as exc:  # checker errors must become structured UI output
        errors.append(str(exc))
    return EnvironmentValidationResult(
        ENVIRONMENT_VERSION,
        not errors,
        shape,
        tuple(errors),
    )


def environment_readiness_for_path(
    path: Path,
    *,
    minimum_rows: int = 2,
) -> EnvironmentReadiness:
    """Return notebook/UI readiness without raising for absent local history."""
    source = Path(path)
    if not source.is_file():
        return EnvironmentReadiness(
            "Not Implemented",
            "No processed single-symbol dataset is available.",
            None,
            0,
        )
    try:
        data = pd.read_csv(source, dtype={"symbol": "string"})
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return EnvironmentReadiness("Validation Failed", str(exc), None, 0)
    symbols = tuple(data.get("symbol", pd.Series(dtype="string")).dropna().unique())
    symbol = str(symbols[0]) if len(symbols) == 1 else None
    if len(data) < minimum_rows:
        return EnvironmentReadiness(
            "Validation Failed",
            f"At least {minimum_rows} rows are required; found {len(data)}.",
            symbol,
            len(data),
        )
    return EnvironmentReadiness(
        "Environment Ready",
        "Environment v1 input is available for validation.",
        symbol,
        len(data),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one processed symbol dataset from the command line."""
    parser = argparse.ArgumentParser(
        description="Validate the Gymnasium single-symbol environment v1"
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        from .single_symbol_env import SingleSymbolTradingEnv

        data = pd.read_csv(args.dataset, dtype={"symbol": "string"})
        result = validate_environment(SingleSymbolTradingEnv(data))
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"Environment validation failed: {exc}")
        return 1
    print(
        f"{result.environment_version}: "
        f"{'Environment Ready' if result.valid else 'Validation Failed'}"
    )
    print(f"Observation shape: {result.observation_shape}")
    for error in result.errors:
        print(f"- {error}")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
