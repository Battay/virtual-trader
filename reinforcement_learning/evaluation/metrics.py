"""Transparent evaluation metrics for deterministic trading episodes."""

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_episode_metrics(history: pd.DataFrame) -> dict[str, object]:
    """Summarize an episode.

    Volatility and Sharpe use population standard deviation and 252 trading
    days with a zero risk-free rate. Annualized values are unavailable with
    fewer than two transition returns and should not be interpreted from short
    samples.
    """
    if history.empty:
        return {
            "initial_portfolio_value": None,
            "final_portfolio_value": None,
            "total_return": None,
            "maximum_drawdown": None,
            "number_of_trades": 0,
            "total_transaction_costs": 0.0,
            "daily_returns": pd.Series(dtype="float64"),
            "sharpe_ratio": None,
            "annualized_volatility": None,
        }
    required = {
        "initial_portfolio_value",
        "portfolio_value",
        "drawdown",
        "shares_traded",
        "transaction_cost",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Episode history is missing: {', '.join(missing)}")
    initial = float(history["initial_portfolio_value"].iloc[0])
    values = pd.to_numeric(history["portfolio_value"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or initial <= 0:
        raise ValueError("Episode portfolio values must be finite and positive")
    value_path = pd.concat(
        [pd.Series([initial], dtype="float64"), values.reset_index(drop=True)],
        ignore_index=True,
    )
    returns = value_path.pct_change().dropna().reset_index(drop=True)
    volatility = None
    sharpe = None
    if len(returns) >= 2:
        daily_std = float(returns.std(ddof=0))
        volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_std > 0:
            sharpe = (
                float(returns.mean()) / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
            )
    final = float(values.iloc[-1])
    return {
        "initial_portfolio_value": initial,
        "final_portfolio_value": final,
        "total_return": final / initial - 1,
        "maximum_drawdown": float(
            pd.to_numeric(history["drawdown"], errors="coerce").max()
        ),
        "number_of_trades": int(
            pd.to_numeric(history["shares_traded"], errors="coerce").ne(0).sum()
        ),
        "total_transaction_costs": float(
            pd.to_numeric(history["transaction_cost"], errors="coerce").sum()
        ),
        "daily_returns": returns,
        "sharpe_ratio": sharpe,
        "annualized_volatility": volatility,
    }
