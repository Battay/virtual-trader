"""Leakage-safe date joins for equity market-context features."""

import pandas as pd

MARKET_CONTEXT_COLUMNS = (
    "kse100_value", "kse100_return_1d", "kse100_return_5d", "kse100_volatility_20",
    "kse30_value", "kse30_return_1d", "kmi30_value", "kmi30_return_1d",
    "allshr_value", "allshr_return_1d", "market_advance_decline_ratio",
    "market_advancing_percent", "market_health_score",
)


def build_index_context(indices: pd.DataFrame) -> pd.DataFrame:
    required = {"index_code", "date", "value"}
    missing = sorted(required.difference(indices.columns))
    if missing:
        raise ValueError(
            "Index data is missing required context columns: " + ", ".join(missing)
        )
    source = indices.copy()
    source["index_code"] = source["index_code"].astype("string").str.strip()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    if source["date"].isna().any():
        raise ValueError("Index data contains invalid dates")
    duplicate_pairs = source.duplicated(["index_code", "date"], keep=False)
    if duplicate_pairs.any():
        examples = (
            source.loc[duplicate_pairs, ["index_code", "date"]]
            .drop_duplicates()
            .sort_values(["date", "index_code"])
            .head(5)
        )
        details = ", ".join(
            f"{row.index_code}@{row.date.date().isoformat()}"
            for row in examples.itertuples(index=False)
        )
        raise ValueError(
            "Index context requires unique (index_code, date) keys; "
            f"duplicate keys include: {details}"
        )

    frames: list[pd.DataFrame] = []
    prefixes = {"KSE100": "kse100", "KSE30": "kse30", "KMI30": "kmi30", "ALLSHR": "allshr"}
    for code, prefix in prefixes.items():
        group = source.loc[source["index_code"] == code].copy()
        if group.empty: continue
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
    if right["date"].isna().any():
        raise ValueError("Market context contains invalid dates")
    duplicate_dates = right["date"].duplicated(keep=False)
    if duplicate_dates.any():
        examples = (
            right.loc[duplicate_dates, "date"]
            .drop_duplicates()
            .sort_values()
            .head(5)
            .dt.date.astype(str)
            .tolist()
        )
        hint = (
            "; received long-form index rows, call build_index_context first"
            if "index_code" in right.columns
            else ""
        )
        raise ValueError(
            "Market context requires exactly one right-side row per date; "
            f"duplicate dates include: {', '.join(examples)}{hint}"
        )
    base_order = result.assign(_row_order=range(len(result))).sort_values("date")
    if max_forward_fill_days == 0:
        joined = base_order.merge(right, on="date", how="left", validate="many_to_one")
    else:
        joined = pd.merge_asof(base_order, right.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta(days=max_forward_fill_days))
    if len(joined) != len(result):
        raise RuntimeError("Market-context join changed the equity row count")
    for column in MARKET_CONTEXT_COLUMNS:
        if column not in joined: joined[column] = pd.NA
    return joined.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
