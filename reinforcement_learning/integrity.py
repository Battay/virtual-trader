"""Small standard-library integrity helpers for RL artifacts."""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(
    path: Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
