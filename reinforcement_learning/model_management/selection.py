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


def normalize_symbol_selection(
    symbols: Collection[object],
    *,
    allowed_symbols: Collection[object] | None = None,
) -> tuple[str, ...]:
    """Return unique non-empty symbols as strings in a stable allowed order."""
    selected = {
        str(symbol).strip()
        for symbol in symbols
        if str(symbol).strip()
    }
    if allowed_symbols is None:
        return tuple(sorted(selected))
    allowed = tuple(
        dict.fromkeys(
            str(symbol).strip()
            for symbol in allowed_symbols
            if str(symbol).strip()
        )
    )
    return tuple(symbol for symbol in allowed if symbol in selected)


def selected_symbols_from_editor(edited_table: pd.DataFrame) -> tuple[str, ...]:
    """Extract selected symbol strings from an editable checkbox table."""
    required = {"selected", "symbol"}
    if edited_table.empty or not required.issubset(edited_table.columns):
        return ()
    checked = edited_table["selected"].fillna(False).astype(bool)
    values = edited_table.loc[checked, "symbol"].astype(str).tolist()
    return normalize_symbol_selection(values)


def update_visible_symbol_selection(
    current_selection: Collection[object],
    visible_symbols: Collection[object],
    selected_visible_symbols: Collection[object],
    *,
    all_symbols: Collection[object],
) -> tuple[str, ...]:
    """Apply visible editor choices while preserving selections hidden by filters."""
    current = normalize_symbol_selection(
        current_selection,
        allowed_symbols=all_symbols,
    )
    visible = set(normalize_symbol_selection(visible_symbols))
    hidden = tuple(symbol for symbol in current if symbol not in visible)
    selected_visible = normalize_symbol_selection(
        selected_visible_symbols,
        allowed_symbols=visible_symbols,
    )
    return normalize_symbol_selection(
        (*hidden, *selected_visible),
        allowed_symbols=all_symbols,
    )


def bulk_select_symbols(
    symbols: Collection[object],
    *,
    all_symbols: Collection[object],
    current_selection: Collection[object] = (),
) -> tuple[str, ...]:
    """Add a bulk-selection result without discarding existing selections."""
    return normalize_symbol_selection(
        (*current_selection, *symbols),
        allowed_symbols=all_symbols,
    )


def symbol_selection_counts(
    selected_symbols: Collection[object],
    visible_symbols: Collection[object],
) -> tuple[int, int]:
    """Return visible and total unique selection counts."""
    selected = set(normalize_symbol_selection(selected_symbols))
    visible = set(normalize_symbol_selection(visible_symbols))
    return len(selected.intersection(visible)), len(selected)
