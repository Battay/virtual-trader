"""Leakage-safe date joins for equity market-context features."""

import pandas as pd

MARKET_CONTEXT_COLUMNS = (
    "kse100_value", "kse100_return_1d", "kse100_return_5d", "kse100_volatility_20",
    "kse30_value", "kse30_return_1d", "kmi30_value", "kmi30_return_1d",
    "allshr_value", "allshr_return_1d", "market_advance_decline_ratio",
    "market_advancing_percent", "market_health_score",
)


def build_index_context(indices: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    prefixes = {"KSE100": "kse100", "KSE30": "kse30", "KMI30": "kmi30", "ALLSHR": "allshr"}
    for code, prefix in prefixes.items():
        group = indices.loc[indices["index_code"].astype(str) == code].copy()
        if group.empty: continue
        group["date"] = pd.to_datetime(group["date"], errors="coerce")
        group["value"] = pd.to_numeric(group["value"], errors="coerce")
        group = group.dropna(subset=["date", "value"]).sort_values("date")
        frame = pd.DataFrame({"date": group["date"], f"{prefix}_value": group["value"], f"{prefix}_return_1d": group["value"].pct_change()})
        if code == "KSE100":
            frame["kse100_return_5d"] = group["value"].pct_change(5)
            frame["kse100_volatility_20"] = group["value"].pct_change().rolling(20).std(ddof=0)
        frames.append(frame)
    if not frames: return pd.DataFrame(columns=("date", *MARKET_CONTEXT_COLUMNS))
    context = frames[0]
    for frame in frames[1:]: context = context.merge(frame, on="date", how="outer", validate="one_to_one")
    return context.sort_values("date").reset_index(drop=True)


def join_market_context(
    equities: pd.DataFrame,
    context: pd.DataFrame,
    *,
    max_forward_fill_days: int = 0,
) -> pd.DataFrame:
    if max_forward_fill_days < 0: raise ValueError("max forward-fill days cannot be negative")
    result = equities.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if context.empty:
        for column in MARKET_CONTEXT_COLUMNS: result[column] = pd.NA
        return result
    right = context.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    base_order = result.assign(_row_order=range(len(result))).sort_values("date")
    if max_forward_fill_days == 0:
        joined = base_order.merge(right, on="date", how="left", validate="many_to_one")
    else:
        joined = pd.merge_asof(base_order, right.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta(days=max_forward_fill_days))
    for column in MARKET_CONTEXT_COLUMNS:
        if column not in joined: joined[column] = pd.NA
    return joined.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
