"""Atomic persistence helpers for generated AI artifacts."""

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import joblib
import pandas as pd


def safe_path_component(value: str) -> str:
    """Return a deterministic filename component for a symbol or model scope."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError("path component cannot be empty")
    return normalized


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def atomic_write_dataframe(data: pd.DataFrame, path: Path) -> None:
    """Replace a CSV only after its complete temporary file is written."""
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        data.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Persist deterministic formatted JSON through an atomic replacement."""
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_dump_joblib(value: object, path: Path) -> None:
    """Persist a joblib artifact without exposing a partially written file."""
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        joblib.dump(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
