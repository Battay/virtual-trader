"""Pure filters and bulk selection helpers for future symbol training."""

from collections.abc import Collection

import pandas as pd


def filter_symbol_status(
    status_table: pd.DataFrame,
    *,
    search: str = "",
    training_statuses: Collection[str] | None = None,
    sectors: Collection[str] | None = None,
    security_types: Collection[str] | None = None,
    newly_added_only: bool = False,
) -> pd.DataFrame:
    """Filter model candidates while preserving one string row per symbol."""
    filtered = status_table.copy()
    query = search.strip()
    if query:
        symbol_match = filtered["symbol"].astype("string").str.contains(
            query,
            case=False,
            regex=False,
            na=False,
        )
        company_match = filtered["company_name"].astype("string").str.contains(
            query,
            case=False,
            regex=False,
            na=False,
        )
        filtered = filtered.loc[symbol_match | company_match]
    for column, selected in (
        ("training_status", training_statuses),
        ("sector", sectors),
        ("security_type", security_types),
    ):
        if selected:
            filtered = filtered.loc[
                filtered[column].astype("string").isin(set(selected))
            ]
    if newly_added_only:
        filtered = filtered.loc[filtered["is_newly_added"].astype(bool)]
    filtered["symbol"] = filtered["symbol"].astype("string")
    return filtered.sort_values("symbol", kind="stable").reset_index(drop=True)


def _symbols(data: pd.DataFrame) -> tuple[str, ...]:
    if data.empty or "symbol" not in data:
        return ()
    return tuple(sorted(set(str(value) for value in data["symbol"].astype("string"))))


def select_visible_symbols(data: pd.DataFrame) -> tuple[str, ...]:
    """Select every currently visible symbol."""
    return _symbols(data)


def select_all_active_eligible(data: pd.DataFrame) -> tuple[str, ...]:
    """Select all active rows that pass the configurable eligibility gate."""
    selected = data.loc[data["is_active"].astype(bool) & data["eligible"].astype(bool)]
    return _symbols(selected)


def select_newly_added_eligible(data: pd.DataFrame) -> tuple[str, ...]:
    """Select newly added active symbols that pass the eligibility gate."""
    selected = data.loc[
        data["is_newly_added"].astype(bool) & data["eligible"].astype(bool)
    ]
    return _symbols(selected)


def select_needing_retraining(data: pd.DataFrame) -> tuple[str, ...]:
    """Select eligible symbols whose models predate newer trading dates."""
    selected = data.loc[
        data["needs_retraining"].astype(bool) & data["eligible"].astype(bool)
    ]
    return _symbols(selected)


def select_never_trained(data: pd.DataFrame) -> tuple[str, ...]:
    """Select eligible symbols with no completed model version."""
    selected = data.loc[
        (data["training_status"].astype("string") == "never_trained")
        & data["eligible"].astype(bool)
    ]
    return _symbols(selected)


def merge_symbol_selections(*selections: Collection[str]) -> tuple[str, ...]:
    """Return a deterministic union suitable for Streamlit session state."""
    return tuple(
        sorted(
            {
                str(symbol).strip()
                for selection in selections
                for symbol in selection
                if str(symbol).strip()
            }
        )
    )
