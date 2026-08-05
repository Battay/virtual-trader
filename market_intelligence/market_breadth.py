"""Market breadth calculated from one latest local equity trading date."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class MarketBreadth:
    reference_date: date | None
    universe: str
    universe_size: int
    advancing: int
    declining: int
    unchanged: int
    advance_decline_ratio: float | None
    total_traded_volume: int
    advancing_percent: float | None
    declining_percent: float | None
    unchanged_percent: float | None


def calculate_market_breadth(
    equity_data: pd.DataFrame,
    *,
    registry: pd.DataFrame | None = None,
    universe: Literal["all_securities", "listed_ordinary_equities"] = "all_securities",
) -> MarketBreadth:
    required = {"symbol", "date", "change", "volume"}
    if not required.issubset(equity_data.columns):
        return MarketBreadth(None, universe, 0, 0, 0, 0, None, 0, None, None, None)
    frame = equity_data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["change"] = pd.to_numeric(frame["change"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "change", "volume"])
    if frame.empty:
        return MarketBreadth(None, universe, 0, 0, 0, 0, None, 0, None, None, None)
    reference = frame["date"].max()
    frame = frame.loc[frame["date"] == reference].copy()
    if universe == "listed_ordinary_equities":
        if registry is None:
            raise ValueError("registry is required for listed ordinary-equity breadth")
        listed = registry["officially_listed"].astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
        allowed = registry.loc[
            listed
            & registry["security_type"].eq("ordinary_equity"), "symbol"
        ].astype(str)
        frame = frame.loc[frame["symbol"].astype(str).isin(set(allowed))]
    advancing = int(frame["change"].gt(0).sum())
    declining = int(frame["change"].lt(0).sum())
    unchanged = int(frame["change"].eq(0).sum())
    size = len(frame)
    return MarketBreadth(
        reference.date(), universe, size, advancing, declining, unchanged,
        advancing / declining if declining else (float("inf") if advancing else None),
        int(frame["volume"].clip(lower=0).sum()),
        advancing / size * 100 if size else None,
        declining / size * 100 if size else None,
        unchanged / size * 100 if size else None,
    )
