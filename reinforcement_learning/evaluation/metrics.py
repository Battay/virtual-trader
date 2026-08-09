"""Transparent metrics for deterministic trading-policy evaluation episodes."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_MINIMUM_ANNUALIZATION_OBSERVATIONS = 20
RISK_FREE_RATE_ANNUAL = 0.0


def _empty_metrics() -> dict[str, object]:
    return {
        "initial_portfolio_value": None,
        "final_portfolio_value": None,
        "total_return": None,
        "annualized_return": None,
        "daily_returns": pd.Series(dtype="float64"),
        "annualized_volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "maximum_drawdown": None,
        "number_of_trades": 0,
        "total_transaction_costs": 0.0,
        "transaction_cost_ratio": None,
        "realized_profit_loss": None,
        "final_unrealized_profit_loss": None,
        "exposure_percentage": None,
        "completed_trades": 0,
        "profitable_completed_trades": 0,
        "completed_trade_win_rate": None,
        "open_position_at_end": False,
        "metric_warnings": ("empty_history",),
    }


def calculate_episode_metrics(
    history: pd.DataFrame,
    *,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    minimum_annualization_observations: int = (
        DEFAULT_MINIMUM_ANNUALIZATION_OBSERVATIONS
    ),
) -> dict[str, object]:
    """Summarize one complete deterministic episode without fabricating values.

    Transition returns use the path ``[initial value] + post-step values``.
    Volatility, Sharpe, and Sortino use population dispersion and a 252-day
    convention by default. Annualized return requires at least 20 transitions;
    risk ratios require at least two. Undefined ratios are returned as ``None``
    rather than infinity.
    """
    if (
        isinstance(trading_days_per_year, bool)
        or not isinstance(trading_days_per_year, Integral)
        or trading_days_per_year < 1
    ):
        raise ValueError("trading_days_per_year must be a positive integer")
    if (
        isinstance(minimum_annualization_observations, bool)
        or not isinstance(minimum_annualization_observations, Integral)
        or minimum_annualization_observations < 2
    ):
        raise ValueError(
            "minimum_annualization_observations must be an integer of at least 2"
        )
    if not math.isfinite(float(annual_risk_free_rate)) or annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate must be finite and greater than -1")
    if history.empty:
        return _empty_metrics()

    required = {
        "initial_portfolio_value",
        "portfolio_value",
        "shares_traded",
        "transaction_cost",
        "realized_profit_loss",
        "unrealized_profit_loss",
        "shares_held",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Episode history is missing: {', '.join(missing)}")
    numeric_columns = sorted(required)
    numeric = history.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Episode history contains non-finite accounting values")

    initial_values = numeric["initial_portfolio_value"]
    initial = float(initial_values.iloc[0])
    values = numeric["portfolio_value"].reset_index(drop=True)
    if initial <= 0 or (values <= 0).any():
        raise ValueError("Episode portfolio values must be positive")
    if not np.allclose(initial_values.to_numpy(dtype=float), initial):
        raise ValueError("Episode initial portfolio value changes within history")
    transaction_costs = numeric["transaction_cost"]
    shares_held = numeric["shares_held"]
    if (transaction_costs < 0).any():
        raise ValueError("Episode transaction costs cannot be negative")
    if (shares_held < 0).any():
        raise ValueError("Episode shares held cannot be negative")

    value_path = pd.concat(
        [pd.Series([initial], dtype="float64"), values],
        ignore_index=True,
    )
    returns = value_path.pct_change().dropna().reset_index(drop=True)
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Episode returns are not finite")
    warnings: list[str] = []
    transition_count = len(returns)
    final = float(values.iloc[-1])
    total_return = final / initial - 1.0
    if not math.isfinite(total_return):
        raise ValueError("Episode total return is not finite")

    annualized_return = None
    if transition_count >= minimum_annualization_observations:
        try:
            annualized_candidate = math.expm1(
                (math.log(final) - math.log(initial))
                * float(trading_days_per_year)
                / transition_count
            )
        except OverflowError:
            annualized_candidate = math.inf
        if math.isfinite(annualized_candidate):
            annualized_return = annualized_candidate
        else:
            warnings.append("non_finite_annualized_return")
    else:
        warnings.append("insufficient_observations_for_annualized_return")

    annualized_volatility = None
    sharpe_ratio = None
    sortino_ratio = None
    if transition_count >= 2:
        daily_risk_free_rate = math.expm1(
            math.log1p(float(annual_risk_free_rate)) / trading_days_per_year
        )
        return_values = returns.to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            excess_values = return_values - daily_risk_free_rate
            excess_mean = float(np.mean(excess_values))
            daily_std = float(np.std(excess_values, ddof=0))
            volatility_candidate = float(
                np.std(return_values, ddof=0) * math.sqrt(trading_days_per_year)
            )
            downside = np.minimum(excess_values, 0.0)
            downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        if math.isfinite(volatility_candidate):
            annualized_volatility = volatility_candidate
        else:
            warnings.append("non_finite_annualized_volatility")
        if math.isfinite(daily_std) and daily_std > 0 and math.isfinite(excess_mean):
            sharpe_candidate = (
                excess_mean / daily_std * math.sqrt(trading_days_per_year)
            )
            if math.isfinite(sharpe_candidate):
                sharpe_ratio = sharpe_candidate
            else:
                warnings.append("non_finite_sharpe_ratio")
        elif daily_std == 0:
            warnings.append("zero_return_volatility")
        else:
            warnings.append("non_finite_return_dispersion")
        if (
            math.isfinite(downside_deviation)
            and downside_deviation > 0
            and math.isfinite(excess_mean)
        ):
            sortino_candidate = (
                excess_mean / downside_deviation * math.sqrt(trading_days_per_year)
            )
            if math.isfinite(sortino_candidate):
                sortino_ratio = sortino_candidate
            else:
                warnings.append("non_finite_sortino_ratio")
        elif downside_deviation == 0:
            warnings.append("zero_downside_deviation")
        else:
            warnings.append("non_finite_downside_deviation")
    else:
        warnings.append("insufficient_observations_for_risk_metrics")

    running_peak = value_path.cummax()
    maximum_drawdown = float((1.0 - value_path / running_peak).max())
    shares_traded = numeric["shares_traded"]
    realized = numeric["realized_profit_loss"].reset_index(drop=True)
    realized_changes = realized.diff()
    realized_changes.iloc[0] = realized.iloc[0]
    completed_trade_profit_loss = realized_changes.loc[
        shares_traded.reset_index(drop=True) < 0
    ]
    completed_trades = len(completed_trade_profit_loss)
    profitable_completed_trades = int(
        (completed_trade_profit_loss > 1e-12).sum()
    )
    total_transaction_costs = float(transaction_costs.sum())
    transaction_cost_ratio = total_transaction_costs / initial
    if not math.isfinite(total_transaction_costs) or not math.isfinite(
        transaction_cost_ratio
    ):
        raise ValueError("Episode transaction cost metrics are not finite")
    return {
        "initial_portfolio_value": initial,
        "final_portfolio_value": final,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "daily_returns": returns,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
        "number_of_trades": int(shares_traded.ne(0).sum()),
        "total_transaction_costs": total_transaction_costs,
        "transaction_cost_ratio": transaction_cost_ratio,
        "realized_profit_loss": float(realized.iloc[-1]),
        "final_unrealized_profit_loss": float(
            numeric["unrealized_profit_loss"].iloc[-1]
        ),
        "exposure_percentage": 100.0 * float(shares_held.gt(0).mean()),
        "completed_trades": completed_trades,
        "profitable_completed_trades": profitable_completed_trades,
        "completed_trade_win_rate": (
            profitable_completed_trades / completed_trades
            if completed_trades
            else None
        ),
        "open_position_at_end": bool(shares_held.iloc[-1] > 0),
        "metric_warnings": tuple(warnings),
    }
