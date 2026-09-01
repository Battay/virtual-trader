"""Provenance-preserving raw evidence utilities for Phase-2 macro series.

This module never downloads sources and never creates canonical observations.
It supports explicit ingestion of an already-retrieved first-party file, with
checksum verification and write-once storage. This is the safe fallback for
official endpoints that reject automated retrieval but permit a user to save
the public document through a browser.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import pandas as pd
from pypdf import PdfReader

from .config import PROJECT_ROOT
from .phase2_data_contract import (
    CPI_YOY_SERIES,
    POLICY_RATE_SERIES,
    REQUIRED_MACRO_SERIES,
    USD_PKR_SERIES,
    Phase2DataContractError,
    build_raw_evidence_manifest,
)


MACRO_EVIDENCE_INGEST_VERSION = "macro_evidence_ingest_v1"
MACRO_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "macro"
SERIES_DIRECTORIES = {
    POLICY_RATE_SERIES: "sbp_policy_rate",
    CPI_YOY_SERIES: "pbs_cpi",
    USD_PKR_SERIES: "sbp_usdpkr",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_PBS_CPI_ROW = re.compile(
    r"^(20\d{2})\s+(\d{1,2})\s+(-?\d+(?:\.\d+)?)\s+.*$"
)
_PBS_POST_TITLE = re.compile(
    r"Monthly\s+Inflation\s+Report\s+for\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(20\d{2})",
    re.IGNORECASE,
)
_MONTHS = {
    name.lower(): month
    for month, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}


def sha256_file(path: str | Path) -> str:
    """Return a file checksum without modifying the source."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retrieval_stamp(value: str | datetime) -> tuple[pd.Timestamp, str]:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise Phase2DataContractError("Raw macro retrieval timestamp is invalid")
    return timestamp, timestamp.strftime("%Y%m%dT%H%M%SZ")


def _safe_original_name(path: Path) -> str:
    name = _SAFE_FILENAME.sub("_", path.name).strip("._")
    if not name:
        raise Phase2DataContractError("Raw macro evidence filename is invalid")
    return name


def _write_new_atomic(path: Path, content: bytes) -> None:
    """Publish bytes atomically while refusing any pre-existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Phase2DataContractError(
                f"Raw macro evidence destination already exists: {path}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ingest_macro_evidence(
    source_file: str | Path,
    *,
    series_owner: str,
    source_url: str,
    source_identifier: str,
    retrieved_at: str | datetime,
    media_type: str,
    expected_sha256: str,
    parse_status: str = "not_parsed",
    source_version: str = "not_stated",
    destination_root: str | Path = MACRO_RAW_ROOT,
) -> dict[str, Any]:
    """Copy explicit first-party evidence into write-once series storage.

    The caller must supply the checksum observed for the manually retrieved
    source. This prevents selecting the wrong browser download and makes the
    manual path no less strict than an automated acquisition.
    """

    if series_owner not in REQUIRED_MACRO_SERIES:
        raise Phase2DataContractError(
            f"Unsupported macro evidence series owner: {series_owner}"
        )
    source = Path(source_file)
    if not source.is_file():
        raise Phase2DataContractError(f"Raw macro evidence is missing: {source}")
    expected = str(expected_sha256).strip().lower()
    if not _SHA256.fullmatch(expected):
        raise Phase2DataContractError("Expected SHA-256 is malformed")
    observed = sha256_file(source)
    if observed != expected:
        raise Phase2DataContractError(
            f"Raw macro evidence checksum mismatch: expected {expected}, got {observed}"
        )

    timestamp, stamp = _retrieval_stamp(retrieved_at)
    directory = Path(destination_root) / SERIES_DIRECTORIES[series_owner]
    destination = directory / f"{stamp}_{_safe_original_name(source)}"
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise Phase2DataContractError(
            f"Raw macro evidence destination already exists: {destination}"
        )

    content = source.read_bytes()
    _write_new_atomic(destination, content)
    try:
        manifest = build_raw_evidence_manifest(
            destination,
            source_url=source_url,
            source_identifier=source_identifier,
            retrieved_at=timestamp,
            media_type=media_type,
            parse_status=parse_status,
            source_version=source_version,
            series_owner=series_owner,
        )
        manifest["ingest_version"] = MACRO_EVIDENCE_INGEST_VERSION
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
        _write_new_atomic(manifest_path, manifest_bytes)
    except Exception:
        # Only files newly created by this operation can be removed here.
        destination.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return {
        "evidence_path": destination,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def verify_macro_evidence_manifest(manifest_file: str | Path) -> dict[str, Any]:
    """Fail closed unless a manifest and its sibling evidence agree exactly."""

    manifest_path = Path(manifest_file)
    if not manifest_path.is_file():
        raise Phase2DataContractError(
            f"Macro evidence manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2DataContractError("Macro evidence manifest is invalid JSON") from exc
    required = {
        "manifest_version",
        "ingest_version",
        "series_owner",
        "source_identifier",
        "source_url",
        "retrieved_at",
        "filename",
        "byte_size",
        "sha256",
        "media_type",
        "parse_status",
        "source_version",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise Phase2DataContractError(
            "Macro evidence manifest is missing fields: " + ", ".join(missing)
        )
    if payload["manifest_version"] != "macro_raw_evidence_manifest_v1":
        raise Phase2DataContractError("Macro evidence manifest version is incompatible")
    if payload["ingest_version"] != MACRO_EVIDENCE_INGEST_VERSION:
        raise Phase2DataContractError("Macro evidence ingest version is incompatible")
    if payload["series_owner"] not in REQUIRED_MACRO_SERIES:
        raise Phase2DataContractError("Macro evidence series owner is unsupported")
    if not str(payload["source_url"]).startswith("https://"):
        raise Phase2DataContractError("Macro evidence source must be HTTPS")
    evidence_path = manifest_path.with_name(str(payload["filename"]))
    if not evidence_path.is_file():
        raise Phase2DataContractError(f"Macro evidence file is missing: {evidence_path}")
    if evidence_path.stat().st_size != int(payload["byte_size"]):
        raise Phase2DataContractError("Macro evidence byte size does not match manifest")
    if sha256_file(evidence_path) != str(payload["sha256"]):
        raise Phase2DataContractError("Macro evidence checksum does not match manifest")
    return dict(payload)


def audit_macro_evidence_store(
    root: str | Path = MACRO_RAW_ROOT,
) -> dict[str, Any]:
    """Return a non-mutating inventory of valid and invalid evidence manifests."""

    root_path = Path(root)
    manifests = sorted(root_path.glob("*/*.manifest.json"))
    owners: Counter[str] = Counter()
    invalid: list[dict[str, str]] = []
    for path in manifests:
        try:
            payload = verify_macro_evidence_manifest(path)
            owners[str(payload["series_owner"])] += 1
        except Phase2DataContractError as exc:
            invalid.append({"manifest": str(path), "error": str(exc)})
    return {
        "root": str(root_path.resolve()),
        "manifest_count": len(manifests),
        "valid_manifest_count": len(manifests) - len(invalid),
        "invalid_manifest_count": len(invalid),
        "valid_files_by_series": {
            series: int(owners.get(series, 0)) for series in REQUIRED_MACRO_SERIES
        },
        "all_required_series_present": all(
            owners.get(series, 0) > 0 for series in REQUIRED_MACRO_SERIES
        ),
        "invalid_manifests": invalid,
    }


def parse_pbs_cpi_yoy_history(source_file: str | Path) -> pd.DataFrame:
    """Parse the National CPI YoY column from PBS's historical PDF table."""

    path = Path(source_file)
    if not path.is_file():
        raise Phase2DataContractError(f"PBS CPI evidence is missing: {path}")
    rows: list[dict[str, Any]] = []
    in_yoy_table = False
    for page in PdfReader(path).pages:
        text = page.extract_text() or ""
        if "Historical Inflation Rate (Y-oY)" in text:
            in_yoy_table = True
        if not in_yoy_table:
            continue
        for line in text.splitlines():
            match = _PBS_CPI_ROW.fullmatch(line.strip())
            if match is None:
                continue
            year, month, national = match.groups()
            month_number = int(month)
            if not 1 <= month_number <= 12:
                raise Phase2DataContractError("PBS CPI table contains an invalid month")
            rows.append(
                {
                    "reference_month": pd.Timestamp(
                        year=int(year), month=month_number, day=1
                    ),
                    "national_cpi_yoy": float(national),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise Phase2DataContractError("PBS CPI YoY table was not found in the PDF")
    result = result.sort_values("reference_month", kind="stable").reset_index(drop=True)
    if result["reference_month"].duplicated().any():
        raise Phase2DataContractError("PBS CPI YoY table contains duplicate months")
    return result


def parse_pbs_release_posts(source_file: str | Path) -> pd.DataFrame:
    """Parse exact CMS publication timestamps from official PBS post metadata."""

    path = Path(source_file)
    if not path.is_file():
        raise Phase2DataContractError(f"PBS release evidence is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2DataContractError("PBS release metadata is invalid JSON") from exc
    if not isinstance(payload, list):
        raise Phase2DataContractError("PBS release metadata must be a post list")
    rows: list[dict[str, Any]] = []
    for post in payload:
        if not isinstance(post, dict):
            continue
        raw_title = post.get("title", {})
        title = raw_title.get("rendered", "") if isinstance(raw_title, dict) else ""
        match = _PBS_POST_TITLE.search(unescape(str(title)))
        if match is None:
            continue
        month_name, year = match.groups()
        timestamp = pd.to_datetime(post.get("date_gmt"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            raise Phase2DataContractError("PBS post has no valid GMT publication time")
        rows.append(
            {
                "reference_month": pd.Timestamp(
                    year=int(year), month=_MONTHS[month_name.lower()], day=1
                ),
                "release_timestamp_utc": timestamp,
                "source_url": str(post.get("link", "")),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise Phase2DataContractError("No CPI publication posts were found")
    result = result.sort_values("reference_month", kind="stable").reset_index(drop=True)
    if result["reference_month"].duplicated().any():
        raise Phase2DataContractError("PBS release metadata contains duplicate months")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest or audit write-once Phase-2 macro evidence."
    )
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--ingest-file", type=Path)
    parser.add_argument("--series", choices=REQUIRED_MACRO_SERIES)
    parser.add_argument("--source-url")
    parser.add_argument("--source-id")
    parser.add_argument("--retrieved-at")
    parser.add_argument("--media-type")
    parser.add_argument("--sha256")
    parser.add_argument("--source-version", default="not_stated")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.ingest_file is not None:
        required = {
            "series": args.series,
            "source_url": args.source_url,
            "source_id": args.source_id,
            "retrieved_at": args.retrieved_at,
            "media_type": args.media_type,
            "sha256": args.sha256,
        }
        absent = sorted(key for key, value in required.items() if not value)
        if absent:
            raise Phase2DataContractError(
                "Manual ingest is missing arguments: " + ", ".join(absent)
            )
        result = ingest_macro_evidence(
            args.ingest_file,
            series_owner=args.series,
            source_url=args.source_url,
            source_identifier=args.source_id,
            retrieved_at=args.retrieved_at,
            media_type=args.media_type,
            expected_sha256=args.sha256,
            source_version=args.source_version,
        )
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return 0
    if args.audit:
        print(json.dumps(audit_macro_evidence_store(), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MACRO_EVIDENCE_INGEST_VERSION",
    "MACRO_RAW_ROOT",
    "SERIES_DIRECTORIES",
    "audit_macro_evidence_store",
    "ingest_macro_evidence",
    "main",
    "parse_pbs_cpi_yoy_history",
    "parse_pbs_release_posts",
    "sha256_file",
    "verify_macro_evidence_manifest",
]
