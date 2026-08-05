"""Atomic storage for raw and normalized PSX index data."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pandas as pd

from .index_config import COMBINED_INDEX_MASTER_PATH, require_supported_index
from .index_parser import INDEX_COLUMNS


def _atomic_write(path: Path, writer: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)  # type: ignore[operator]
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_raw_snapshot(payload: dict[str, Any], path: Path) -> None:
    content = json.dumps(payload, indent=2, separators=(",", ": ")) + "\n"
    _atomic_write(Path(path), lambda temporary: temporary.write_text(content, encoding="utf-8"))


def write_index_csv(data: pd.DataFrame, path: Path) -> None:
    selected = data.reindex(columns=INDEX_COLUMNS)
    _atomic_write(Path(path), lambda temporary: selected.to_csv(temporary, index=False))


def load_index_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(path, dtype={"index_code": "string"})


def update_index_csv(index_code: str, new_data: pd.DataFrame, path: Path | None = None) -> Path:
    definition = require_supported_index(index_code)
    output = Path(path) if path is not None else definition.master_path
    existing = load_index_csv(output)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["fetched_at"] = pd.to_datetime(combined["fetched_at"], errors="coerce", utc=True)
    combined = (
        combined.dropna(subset=["date"])
        .sort_values(["fetched_at", "timestamp"], kind="stable", na_position="first")
        .drop_duplicates(["index_code", "date"], keep="last")
        .sort_values(["date"], kind="stable")
        .reset_index(drop=True)
    )
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined["fetched_at"] = combined["fetched_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_index_csv(combined, output)
    return output


def build_combined_master(
    paths: Iterable[Path] | None = None,
    *,
    output_path: Path = COMBINED_INDEX_MASTER_PATH,
) -> pd.DataFrame:
    source_paths = tuple(paths) if paths is not None else tuple(
        definition.master_path for definition in
        (require_supported_index(code) for code in ("KSE100", "KSE30", "KMI30", "ALLSHR"))
    )
    frames = [load_index_csv(path) for path in source_paths if Path(path).exists()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLUMNS)
    if not combined.empty:
        combined = (
            combined.sort_values(["index_code", "date", "fetched_at"], kind="stable")
            .drop_duplicates(["index_code", "date"], keep="last")
            .sort_values(["date", "index_code"], kind="stable")
            .reset_index(drop=True)
        )
    write_index_csv(combined, output_path)
    return combined
