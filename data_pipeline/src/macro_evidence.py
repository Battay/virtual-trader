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
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup
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
MACRO_EVIDENCE_READINESS_VERSION = "macro_evidence_readiness_v1"
SBP_POLICY_PARSER_VERSION = "sbp_policy_target_rate_pdf_v2"
SBP_POLICY_CIRCULAR_PARSER_VERSION = "sbp_policy_circular_html_v1"
SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION = "sbp_policy_circular_pdf_v1"
SBP_POLICY_CHAIN_VERSION = "sbp_policy_circular_chain_v1"
MACRO_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "macro"
SERIES_DIRECTORIES = {
    POLICY_RATE_SERIES: "sbp_policy_rate",
    CPI_YOY_SERIES: "pbs_cpi",
    USD_PKR_SERIES: "sbp_usdpkr",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_SBP_POLICY_TARGET_ROW = re.compile(
    r"^\s*(?P<date>\d{1,2}[-/]\s*[A-Za-z]{3,9}[-/]\s*\d{2,4})\s+"
    r"(?P<reverse_repo>\d+(?:\.\d+)?)\s+"
    r"(?P<repo>\d+(?:\.\d+)?)\s+"
    r"(?P<policy>\d+(?:\.\d+)?)(?:\s+.*)?$"
)
_SBP_STRUCTURE_I = re.compile(
    r"Structure\s+of\s+Interest\s+Rates\s*-\s*I(?![IVX])", re.IGNORECASE
)
_SBP_POLICY_TARGET_HEADER = re.compile(
    r"SBP\s+Policy\s*\(?Target\)?\s*Rate", re.IGNORECASE
)
_SBP_CIRCULAR_TARGET_HEADER = re.compile(
    r"(?:SBP(?:['’]s)?\s+)?Policy\s+(?:"
    r"[\u2018\u2019\x27\"]?Rate[\u2018\u2019\x27\"]?\s*"
    r"\(Target\s+Rate\)|\(?Target\)?\s+Rate)",
    re.IGNORECASE,
)
_SBP_CIRCULAR_DATE_TEXT = (
    r"(?:[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+20\d{2}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+[,]?\s+20\d{2})"
)
_SBP_CIRCULAR_HEADING = re.compile(
    r"DMMD\s+Circular\s+No\.?\s*(?P<number>\d+)\s+of\s+"
    r"(?P<year>20\d{2})\s+(?P<announcement>"
    + _SBP_CIRCULAR_DATE_TEXT
    + r")",
    re.IGNORECASE,
)
_SBP_CIRCULAR_IDENTITY = re.compile(
    r"DMMD\s+Circular\s+No\.?\s*(?P<number>\d+)"
    r"(?:\s+of\s+(?P<year>20\d{2}))?",
    re.IGNORECASE,
)
_SBP_CIRCULAR_CHANGE = re.compile(
    r"(?:the\s+)?SBP\s+has\s+decided\s+to\s+(?:increase|decrease)\s+"
    r"(?:its\s+)?[‘'\"]?Policy\s+Rate[’'\"]?\s*\(Target\s+Rate\)\s+"
    r"from\s+(?P<previous>\d+(?:\.\d+)?)%\s+to\s+"
    r"(?P<new>\d+(?:\.\d+)?)%",
    re.IGNORECASE,
)
_SBP_CIRCULAR_EFFECTIVE = re.compile(
    r"Above\s+changes\s+are\s+effective\s+from\s+"
    r"(?P<effective>" + _SBP_CIRCULAR_DATE_TEXT + r")",
    re.IGNORECASE,
)
_SBP_CIRCULAR_REFERENCE = re.compile(
    r"refer\s+to\s+DMMD\s+Circular\s+No\.?\s*(?P<number>\d+)\s+"
    r"dated\s+(?P<date>" + _SBP_CIRCULAR_DATE_TEXT + r")",
    re.IGNORECASE,
)
_SBP_CIRCULAR_SOURCE_IDENTIFIER = re.compile(
    r"^sbp_dmmd_circular_(?P<number>\d{2})_(?P<year>20\d{2})$",
    re.IGNORECASE,
)
_SBP_CIRCULAR_PDF_IDENTIFIERS = {
    "sbp_dmmd_circular_06_2022",
    "sbp_dmmd_circular_09_2022",
}
_SBP_SIR_SOURCE_IDENTIFIER = "sbp_structure_of_interest_rates_sir"
_SBP_POLICY_EXPORT_IDENTIFIERS = {"sbp_policy_target_rate_easydata_export"}
_SBP_PDF_OCR_HELPER = Path(__file__).with_name("sbp_pdf_ocr.swift")
_REQUIRED_PHASE2_POLICY_CIRCULARS = (
    (2021, 15),
    (2021, 21),
    (2021, 23),
    (2022, 6),
    (2022, 9),
    (2022, 13),
    (2022, 20),
)
_SBP_M2M_USD_ROW = re.compile(
    r"(?P<date>\d{1,2}-[A-Za-z]{3}-\d{2,4})\s+USD\s+"
    r"(?P<ready>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
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
_OFFICIAL_HOST_SUFFIX = {
    POLICY_RATE_SERIES: "sbp.org.pk",
    CPI_YOY_SERIES: "pbs.gov.pk",
    USD_PKR_SERIES: "sbp.org.pk",
}
_KARACHI = ZoneInfo("Asia/Karachi")
_PHASE2_MARKET_FIRST = pd.Timestamp("2021-08-06")
_PHASE2_MARKET_LAST = pd.Timestamp("2026-08-27")
_PHASE2_CPI_FIRST_REFERENCE_MONTH = pd.Timestamp("2021-07-01")
_PHASE2_CPI_LAST_REFERENCE_MONTH = pd.Timestamp("2026-07-01")


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


def _is_authoritative_url(source_url: str, series_owner: str) -> bool:
    parsed = urlparse(str(source_url))
    host = (parsed.hostname or "").lower().rstrip(".")
    suffix = _OFFICIAL_HOST_SUFFIX.get(series_owner)
    return (
        parsed.scheme == "https"
        and suffix is not None
        and (host == suffix or host.endswith(f".{suffix}"))
    )


def _require_authoritative_url(source_url: str, series_owner: str) -> None:
    if not _is_authoritative_url(source_url, series_owner):
        raise Phase2DataContractError(
            f"Macro evidence source is not authoritative for {series_owner}"
        )


def _normalise_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _parse_date_values(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    named_short = text.str.fullmatch(r"\d{1,2}-[A-Za-z]{3,9}-\d{2}", na=False)
    named_long = text.str.fullmatch(r"\d{1,2}-[A-Za-z]{3,9}-\d{4}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso] = pd.to_datetime(
        text.loc[iso], format="%Y-%m-%d", errors="coerce"
    )
    parsed.loc[named_short] = pd.to_datetime(
        text.loc[named_short], format="%d-%b-%y", errors="coerce"
    )
    parsed.loc[named_long] = pd.to_datetime(
        text.loc[named_long], format="%d-%b-%Y", errors="coerce"
    )
    remaining = ~(iso | named_short | named_long) & text.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text.loc[remaining], format="mixed", errors="coerce", dayfirst=True
        )
    return parsed.dt.normalize()


def _next_decision_boundary(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value).normalize()
    return pd.Timestamp(day.date(), tz=_KARACHI) + pd.Timedelta(days=1)


def _manifest_evidence_path(
    manifest_path: str | Path,
    *,
    expected_series: str,
) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path)
    manifest = verify_macro_evidence_manifest(path)
    if manifest["series_owner"] != expected_series:
        raise Phase2DataContractError(
            f"Evidence belongs to {manifest['series_owner']}, not {expected_series}"
        )
    _require_authoritative_url(str(manifest["source_url"]), expected_series)
    return path.with_name(str(manifest["filename"])), manifest


def _classify_sbp_policy_evidence(
    source: Path, manifest: dict[str, Any]
) -> str:
    """Return the one supported parser contract for preserved policy evidence."""

    identifier = str(manifest.get("source_identifier", "")).strip().lower()
    media_type = str(manifest.get("media_type", "")).split(";", 1)[0].strip().lower()
    suffix = source.suffix.lower()
    source_path = urlparse(str(manifest.get("source_url", ""))).path.lower()
    if identifier == _SBP_SIR_SOURCE_IDENTIFIER:
        if (
            suffix != ".pdf"
            or media_type != "application/pdf"
            or not source_path.endswith("/ecodata/sir.pdf")
        ):
            raise Phase2DataContractError(
                "SBP SIR evidence identity, media type, and source URL disagree"
            )
        return "sir_pdf"

    circular_identity = _SBP_CIRCULAR_SOURCE_IDENTIFIER.fullmatch(identifier)
    if circular_identity is not None:
        number = int(circular_identity.group("number"))
        year = int(circular_identity.group("year"))
        if suffix in {".html", ".htm"} and media_type == "text/html":
            expected_path = f"/dmmd/{year}/c{number}.htm"
            if not source_path.endswith(expected_path):
                raise Phase2DataContractError(
                    "SBP circular HTML identity and source URL disagree"
                )
            return "circular_html"
        if suffix == ".pdf" and media_type == "application/pdf":
            if identifier not in _SBP_CIRCULAR_PDF_IDENTIFIERS:
                raise Phase2DataContractError(
                    "Unsupported SBP policy circular PDF evidence identity"
                )
            expected_fragment = f"dmmd-circular-no-{number:02d}-of-{year}"
            if expected_fragment not in source_path:
                raise Phase2DataContractError(
                    "SBP circular PDF identity and source URL disagree"
                )
            return "circular_pdf"
        raise Phase2DataContractError(
            "SBP circular evidence identity and media type disagree"
        )

    if identifier in _SBP_POLICY_EXPORT_IDENTIFIERS:
        if suffix not in {".csv", ".txt"} or media_type not in {
            "text/csv",
            "text/plain",
        }:
            raise Phase2DataContractError(
                "SBP policy export identity and media type disagree"
            )
        return "policy_export"
    raise Phase2DataContractError(
        "Unsupported or ambiguous SBP policy evidence identity"
    )


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
    provenance_notes: str | None = None,
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
    _require_authoritative_url(source_url, series_owner)
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
        if provenance_notes is not None:
            notes = str(provenance_notes).strip()
            if notes:
                manifest["provenance_notes"] = notes
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
    _require_authoritative_url(
        str(payload["source_url"]), str(payload["series_owner"])
    )
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


def _read_delimited_evidence(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise Phase2DataContractError(
            f"Delimited macro evidence could not be parsed: {path.name}"
        ) from exc
    frame = frame.rename(
        columns={column: _normalise_column_name(column) for column in frame}
    )
    if frame.empty:
        raise Phase2DataContractError("Delimited macro evidence is empty")
    return frame


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _parse_circular_date(value: str, *, field: str) -> pd.Timestamp:
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value, flags=re.IGNORECASE)
    parsed = pd.to_datetime(cleaned.replace(",", ""), errors="coerce")
    if pd.isna(parsed):
        raise Phase2DataContractError(f"SBP circular {field} is invalid")
    return pd.Timestamp(parsed).normalize()


def _normalise_sbp_circular_visible_text(html: bytes) -> str:
    """Return browser-equivalent body text from legacy or current SBP HTML."""

    soup = BeautifulSoup(html, "html.parser")
    # Legacy SBP pages declare ISO-8859-1 but use Windows-1252 smart-quote
    # bytes. Browsers, including Safari, apply the HTML-standard Windows-1252
    # mapping. Reparse only that declared legacy encoding; UTF-8/current pages
    # retain their detected encoding.
    if soup.original_encoding == "iso-8859-1":
        soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
    root = soup.body or soup
    for hidden in root.find_all(("script", "style", "noscript", "template")):
        hidden.decompose()
    return " ".join(root.get_text(" ", strip=True).replace("\xa0", " ").split())


def _parse_sbp_policy_circular_text(
    text: str,
    manifest: dict[str, Any],
    *,
    parser_version: str,
) -> pd.DataFrame:
    """Parse explicit policy concepts and dates from normalized circular text."""

    if not _SBP_CIRCULAR_TARGET_HEADER.search(text):
        raise Phase2DataContractError(
            "SBP circular does not explicitly identify the Policy (Target) Rate"
        )
    heading = _SBP_CIRCULAR_HEADING.search(text)
    identity = _SBP_CIRCULAR_IDENTITY.search(text)
    announcement_text: str | None = None
    if heading is not None:
        announcement_text = heading.group("announcement")
    elif identity is not None:
        # Image-only official circular PDFs put the circular number in the
        # breadcrumb and the dated heading shortly afterward. Bind the first
        # explicit date after that identity; do not derive it from metadata.
        following = text[identity.end() : identity.end() + 300]
        announcement = re.search(_SBP_CIRCULAR_DATE_TEXT, following, re.IGNORECASE)
        if announcement is not None:
            announcement_text = announcement.group(0)
    change = _SBP_CIRCULAR_CHANGE.search(text)
    effective = _SBP_CIRCULAR_EFFECTIVE.search(text)
    reference = _SBP_CIRCULAR_REFERENCE.search(text)
    missing_fields: list[str] = []
    if identity is None:
        missing_fields.append("circular number")
    if announcement_text is None:
        missing_fields.append("announcement date/year")
    if change is None:
        missing_fields.extend(("previous target rate", "new target rate"))
    if effective is None:
        missing_fields.append("effective date")
    if missing_fields:
        raise Phase2DataContractError(
            "SBP circular lacks explicit required fields: "
            + ", ".join(missing_fields)
        )
    assert identity is not None
    assert announcement_text is not None
    assert change is not None
    assert effective is not None
    announcement_date = _parse_circular_date(
        announcement_text, field="announcement date"
    )
    effective_date = _parse_circular_date(
        effective.group("effective"), field="effective date"
    )
    if effective_date < announcement_date:
        raise Phase2DataContractError(
            "SBP circular effective date precedes its announcement date"
        )
    previous_rate = float(change.group("previous"))
    new_rate = float(change.group("new"))
    if not (0.0 < previous_rate <= 100.0 and 0.0 < new_rate <= 100.0):
        raise Phase2DataContractError("SBP circular policy rate is outside a valid range")
    reference_date = (
        _parse_circular_date(reference.group("date"), field="reference date")
        if reference is not None
        else pd.NaT
    )
    # Both identity and date come from normalized visible source text. Some
    # legacy HTML templates contain stale identities in <head>, which the body
    # extraction excludes. Image-only PDFs state the number in a breadcrumb
    # and the year in the explicit announcement date rather than "of YEAR".
    circular_number = int(identity.group("number"))
    explicit_year = identity.group("year")
    circular_year = int(explicit_year) if explicit_year else announcement_date.year
    if explicit_year is not None and circular_year != announcement_date.year:
        raise Phase2DataContractError(
            "SBP circular identity year conflicts with its announcement date"
        )
    source_identity = _SBP_CIRCULAR_SOURCE_IDENTIFIER.fullmatch(
        str(manifest["source_identifier"])
    )
    if source_identity is not None and (
        circular_number != int(source_identity.group("number"))
        or circular_year != int(source_identity.group("year"))
    ):
        raise Phase2DataContractError(
            "SBP circular source identifier conflicts with its explicit heading"
        )
    row = {
        "circular_id": f"DMMD_C{circular_number}_{circular_year}",
        "circular_number": circular_number,
        "circular_year": circular_year,
        "announcement_date": announcement_date,
        "previous_policy_rate": previous_rate,
        "policy_rate": new_rate,
        "effective_date": effective_date,
        "referenced_circular_number": (
            int(reference.group("number")) if reference is not None else pd.NA
        ),
        "referenced_circular_date": reference_date,
        "source_url": str(manifest["source_url"]),
        "source_evidence_id": str(manifest["source_identifier"]),
        "source_sha256": str(manifest["sha256"]),
        "parser_version": parser_version,
    }
    return pd.DataFrame([row])


def parse_sbp_policy_rate_circular_evidence(
    manifest_file: str | Path,
) -> pd.DataFrame:
    """Parse one preserved first-party SBP policy-rate circular HTML page."""

    source, manifest = _manifest_evidence_path(
        manifest_file, expected_series=POLICY_RATE_SERIES
    )
    if _classify_sbp_policy_evidence(source, manifest) != "circular_html":
        raise Phase2DataContractError(
            "SBP policy circular evidence must be preserved HTML"
        )
    try:
        html = source.read_bytes()
        text = _normalise_sbp_circular_visible_text(html)
    except Exception as exc:
        raise Phase2DataContractError("SBP policy circular HTML is unreadable") from exc
    return _parse_sbp_policy_circular_text(
        text,
        manifest,
        parser_version=SBP_POLICY_CIRCULAR_PARSER_VERSION,
    )


def _extract_sbp_policy_circular_pdf_text(source: Path) -> str:
    """Extract circular text, using native Vision only for image-only PDFs."""

    try:
        reader = PdfReader(source)
        embedded_text = "\n".join(
            (page.extract_text() or "").strip() for page in reader.pages
        ).strip()
    except Exception as exc:
        raise Phase2DataContractError("SBP policy circular PDF is unreadable") from exc
    if embedded_text:
        return embedded_text
    if platform.system() != "Darwin":
        raise Phase2DataContractError(
            "Image-only SBP circular PDF requires macOS Vision OCR"
        )
    xcrun = shutil.which("xcrun")
    if xcrun is None or not _SBP_PDF_OCR_HELPER.is_file():
        raise Phase2DataContractError(
            "Image-only SBP circular PDF OCR support is unavailable"
        )
    with tempfile.TemporaryDirectory(prefix="sbp-circular-ocr-") as directory:
        temporary = Path(directory)
        images: list[Path] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_images = list(page.images)
            except Exception as exc:
                raise Phase2DataContractError(
                    "SBP circular PDF page images could not be inspected"
                ) from exc
            if len(page_images) != 1:
                raise Phase2DataContractError(
                    "Image-only SBP circular PDF must contain exactly one page image"
                )
            image = page_images[0]
            extension = Path(str(getattr(image, "name", ""))).suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                extension = ".png"
            image_path = temporary / f"page-{page_number:03d}{extension}"
            image_path.write_bytes(image.data)
            images.append(image_path)
        environment = os.environ.copy()
        module_cache = temporary / "module-cache"
        environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
        environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
        try:
            completed = subprocess.run(
                [xcrun, "swift", str(_SBP_PDF_OCR_HELPER), *map(str, images)],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise Phase2DataContractError(
                "Image-only SBP circular PDF OCR could not run"
            ) from exc
        text = completed.stdout.strip()
        if completed.returncode != 0 or not text:
            detail = completed.stderr.strip()
            raise Phase2DataContractError(
                "Image-only SBP circular PDF OCR failed"
                + (f": {detail}" if detail else "")
            )
        return text


def parse_sbp_policy_rate_circular_pdf_evidence(
    manifest_file: str | Path,
) -> pd.DataFrame:
    """Parse one supported preserved first-party SBP circular PDF."""

    source, manifest = _manifest_evidence_path(
        manifest_file, expected_series=POLICY_RATE_SERIES
    )
    if _classify_sbp_policy_evidence(source, manifest) != "circular_pdf":
        raise Phase2DataContractError(
            "Evidence is not a supported SBP policy circular PDF"
        )
    text = " ".join(_extract_sbp_policy_circular_pdf_text(source).split())
    return _parse_sbp_policy_circular_text(
        text,
        manifest,
        parser_version=SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION,
    )


def _missing_sbp_policy_circulars(circulars: pd.DataFrame) -> list[str]:
    available = {
        (int(row.circular_year), int(row.circular_number))
        for row in circulars.itertuples(index=False)
    }
    return [
        f"DMMD Circular No. {number:02d} of {year}"
        for year, number in _REQUIRED_PHASE2_POLICY_CIRCULARS
        if (year, number) not in available
    ]


def validate_sbp_policy_circular_chain(
    circulars: pd.DataFrame,
    *,
    phase2_start: str | pd.Timestamp = _PHASE2_MARKET_FIRST,
) -> dict[str, Any]:
    """Validate the fixed Phase-2 circular chain without creating an event."""

    required = {
        "circular_id",
        "circular_number",
        "circular_year",
        "announcement_date",
        "previous_policy_rate",
        "policy_rate",
        "effective_date",
        "referenced_circular_date",
        "source_evidence_id",
    }
    missing_columns = sorted(required.difference(circulars.columns))
    if missing_columns:
        raise Phase2DataContractError(
            "SBP circular chain is missing fields: " + ", ".join(missing_columns)
        )
    chain = circulars.copy()
    if chain.empty:
        raise Phase2DataContractError("SBP policy circular chain is empty")
    if chain["circular_id"].duplicated().any():
        raise Phase2DataContractError("SBP policy circular chain has duplicates")
    missing_circulars = _missing_sbp_policy_circulars(chain)
    if missing_circulars:
        raise Phase2DataContractError(
            "SBP policy circular chain has gaps: " + ", ".join(missing_circulars)
        )
    chain = chain.sort_values("effective_date", kind="stable").reset_index(drop=True)
    if chain["effective_date"].duplicated().any():
        raise Phase2DataContractError(
            "SBP policy circular chain has duplicate effective dates"
        )
    for index in range(1, len(chain)):
        established = float(chain.loc[index - 1, "policy_rate"])
        claimed_previous = float(chain.loc[index, "previous_policy_rate"])
        if abs(established - claimed_previous) > 1e-12:
            raise Phase2DataContractError(
                "SBP policy circular chain contradicts the preceding established rate"
            )
    start = pd.Timestamp(phase2_start).normalize()
    first = chain.iloc[0]
    reference_date = first["referenced_circular_date"]
    if (
        pd.isna(reference_date)
        or pd.Timestamp(reference_date) > start
        or pd.Timestamp(first["effective_date"]) <= start
    ):
        raise Phase2DataContractError(
            "SBP circular chain does not establish the rate at the Phase-2 start"
        )
    return {
        "chain_version": SBP_POLICY_CHAIN_VERSION,
        "event_count": int(len(chain)),
        "first_effective_date": pd.Timestamp(first["effective_date"])
        .date()
        .isoformat(),
        "last_effective_date": pd.Timestamp(chain.iloc[-1]["effective_date"])
        .date()
        .isoformat(),
        "phase2_start_date": start.date().isoformat(),
        "phase2_start_policy_rate": float(first["previous_policy_rate"]),
        "phase2_start_rate_source": str(first["source_evidence_id"]),
        "synthetic_start_event_created": False,
    }


def _extract_sbp_policy_target_rows(
    source: Path, manifest: dict[str, Any]
) -> pd.DataFrame:
    """Extract only the explicitly labelled SBP Policy (Target) Rate table.

    The multi-page SIR publication contains many other date/rate tables.  A
    date followed by three numbers is therefore not, by itself, policy-rate
    evidence.  Page/table labels and the exact target-rate header are required
    before a row is eligible for parsing.
    """

    rows: list[dict[str, Any]] = []
    identified_pages = 0
    for page_number, page in enumerate(PdfReader(source).pages, start=1):
        text = page.extract_text() or ""
        if not (
            _SBP_STRUCTURE_I.search(text)
            and _SBP_POLICY_TARGET_HEADER.search(text)
        ):
            continue
        identified_pages += 1
        for raw_line in text.splitlines():
            match = _SBP_POLICY_TARGET_ROW.match(raw_line)
            if match is None:
                continue
            rows.append(
                {
                    "page_number": page_number,
                    "table_section": "Structure of Interest Rates -I",
                    "raw_extracted_text": raw_line.strip(),
                    "effective_date": match.group("date").replace(" ", ""),
                    "sbp_reverse_repo_rate": match.group("reverse_repo"),
                    "sbp_repo_rate": match.group("repo"),
                    "policy_rate": match.group("policy"),
                    "announcement_date": None,
                    "source_evidence_id": str(manifest["source_identifier"]),
                    "source_sha256": str(manifest["sha256"]),
                }
            )
    if identified_pages == 0:
        raise Phase2DataContractError(
            "SBP policy PDF lacks the explicitly labelled Structure of "
            "Interest Rates -I target-rate table"
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise Phase2DataContractError(
            "No rows were found in the SBP Policy (Target) Rate table"
        )
    result["effective_date"] = _parse_date_values(result["effective_date"])
    for column in ("sbp_reverse_repo_rate", "sbp_repo_rate", "policy_rate"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[
        [
            "effective_date",
            "sbp_reverse_repo_rate",
            "sbp_repo_rate",
            "policy_rate",
        ]
    ].isna().any(axis=None):
        raise Phase2DataContractError(
            "SBP Policy (Target) Rate table has invalid dates or rate values"
        )
    return result


def audit_sbp_policy_rate_evidence(manifest_file: str | Path) -> dict[str, Any]:
    """Return every raw PDF target-table row before deterministic deduplication."""

    source, manifest = _manifest_evidence_path(
        manifest_file, expected_series=POLICY_RATE_SERIES
    )
    if _classify_sbp_policy_evidence(source, manifest) != "sir_pdf":
        raise Phase2DataContractError(
            "The page-level SBP policy audit requires official SIR PDF evidence"
        )
    rows = _extract_sbp_policy_target_rows(source, manifest)
    conflict_counts = rows.groupby("effective_date")["policy_rate"].nunique()
    conflict_dates = conflict_counts[conflict_counts.gt(1)].index
    conflict_mask = rows["effective_date"].isin(conflict_dates)
    unique_events = rows.drop_duplicates(subset=("effective_date", "policy_rate"))
    records: list[dict[str, Any]] = []
    for record in rows.to_dict("records"):
        record["effective_date"] = pd.Timestamp(record["effective_date"]).date().isoformat()
        record["announcement_date"] = None
        records.append(record)
    ordered_dates = rows["effective_date"].reset_index(drop=True)
    return {
        "parser_version": SBP_POLICY_PARSER_VERSION,
        "source_evidence_id": str(manifest["source_identifier"]),
        "source_sha256": str(manifest["sha256"]),
        "raw_record_count": int(len(rows)),
        "unique_event_count": int(len(unique_events)),
        "duplicate_source_row_count": int(len(rows) - len(unique_events)),
        "conflicting_effective_dates": [
            pd.Timestamp(value).date().isoformat() for value in conflict_dates
        ],
        "conflicting_source_row_count": int(conflict_mask.sum()),
        "missing_target_rate_values": int(rows["policy_rate"].isna().sum()),
        "earliest_effective_date": rows["effective_date"].min().date().isoformat(),
        "latest_effective_date": rows["effective_date"].max().date().isoformat(),
        "monotonic_effective_dates": bool(ordered_dates.is_monotonic_increasing),
        "records": records,
    }


def _normalise_sbp_policy_events(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    parser_version: str,
) -> pd.DataFrame:
    events = pd.DataFrame(rows)
    if events.empty:
        raise Phase2DataContractError("No SBP policy target-rate events were parsed")
    events["effective_date"] = _parse_date_values(events["effective_date"])
    events["announcement_date"] = _parse_date_values(events["announcement_date"])
    events["policy_rate"] = pd.to_numeric(events["policy_rate"], errors="coerce")
    if events["effective_date"].isna().any() or events["policy_rate"].isna().any():
        raise Phase2DataContractError("SBP policy evidence has invalid dates or rates")
    if (~events["policy_rate"].between(0.0, 100.0, inclusive="right")).any():
        raise Phase2DataContractError("SBP policy rate is outside a valid range")

    duplicate_rates = events.groupby("effective_date")["policy_rate"].nunique()
    if duplicate_rates.gt(1).any():
        dates = ", ".join(
            value.date().isoformat()
            for value in duplicate_rates[duplicate_rates.gt(1)].index
        )
        raise Phase2DataContractError(
            f"SBP policy target-rate evidence conflicts for effective dates: {dates}"
        )
    events = events.drop_duplicates(
        subset=("effective_date", "policy_rate", "announcement_date")
    ).sort_values("effective_date", kind="stable")

    def availability(row: pd.Series) -> pd.Timestamp:
        effective = pd.Timestamp(row["effective_date"])
        announced = row["announcement_date"]
        if pd.notna(announced) and pd.Timestamp(announced) < effective:
            return pd.Timestamp(effective.date(), tz=_KARACHI)
        boundary = (
            effective
            if pd.isna(announced)
            else max(effective, pd.Timestamp(announced))
        )
        return _next_decision_boundary(boundary)

    events["effective_available_timestamp"] = events.apply(availability, axis=1)
    events["source_evidence_id"] = str(manifest["source_identifier"])
    events["source_sha256"] = str(manifest["sha256"])
    events.attrs["parser_version"] = parser_version
    return events.reset_index(drop=True)[
        [
            "policy_rate",
            "announcement_date",
            "effective_date",
            "effective_available_timestamp",
            "source_evidence_id",
            "source_sha256",
        ]
    ]


def parse_sbp_policy_rate_evidence(manifest_file: str | Path) -> pd.DataFrame:
    """Parse official SBP target-rate events with identity-bound routing."""

    source, manifest = _manifest_evidence_path(
        manifest_file, expected_series=POLICY_RATE_SERIES
    )
    evidence_kind = _classify_sbp_policy_evidence(source, manifest)
    rows: list[dict[str, Any]] = []
    if evidence_kind == "sir_pdf":
        extracted = _extract_sbp_policy_target_rows(source, manifest)
        rows = extracted[
            ["effective_date", "policy_rate", "announcement_date"]
        ].to_dict("records")
        parser_version = SBP_POLICY_PARSER_VERSION
    elif evidence_kind == "circular_html":
        circular = parse_sbp_policy_rate_circular_evidence(manifest_file)
        rows = circular[
            ["effective_date", "policy_rate", "announcement_date"]
        ].to_dict("records")
        parser_version = SBP_POLICY_CIRCULAR_PARSER_VERSION
    elif evidence_kind == "circular_pdf":
        circular = parse_sbp_policy_rate_circular_pdf_evidence(manifest_file)
        rows = circular[
            ["effective_date", "policy_rate", "announcement_date"]
        ].to_dict("records")
        parser_version = SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION
    elif evidence_kind == "policy_export":
        frame = _read_delimited_evidence(source)
        effective = _first_column(
            frame, ("effective_date", "w_e_f", "wef", "date")
        )
        policy = _first_column(
            frame,
            (
                "sbp_policy_target_rate",
                "policy_target_rate",
                "policy_rate",
                "rate",
            ),
        )
        announcement = _first_column(
            frame, ("announcement_date", "decision_date", "mpc_announcement_date")
        )
        if effective is None or policy is None:
            raise Phase2DataContractError(
                "SBP policy export requires effective-date and policy-rate columns"
            )
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "effective_date": record[effective],
                    "policy_rate": record[policy],
                    "announcement_date": (
                        record[announcement] if announcement is not None else None
                    ),
                }
            )
        parser_version = SBP_POLICY_PARSER_VERSION
    else:  # pragma: no cover - classifier is exhaustive and fail-closed
        raise Phase2DataContractError("Unsupported SBP policy evidence parser")
    return _normalise_sbp_policy_events(
        rows,
        manifest,
        parser_version=parser_version,
    )


def parse_sbp_usd_pkr_m2m_evidence(manifest_file: str | Path) -> pd.DataFrame:
    """Parse only SBP USD/PKR daily Mark-to-Market evidence."""

    source, manifest = _manifest_evidence_path(
        manifest_file, expected_series=USD_PKR_SERIES
    )
    identity = f"{manifest['source_identifier']} {manifest['source_url']}".lower()
    if (
        "m2m" not in identity
        and "mark-to-market" not in identity
        and "mark_to_market" not in identity
    ):
        raise Phase2DataContractError("FX evidence is not identified as SBP M2M")
    if "conversion" in identity or "/crates/" in identity:
        raise Phase2DataContractError(
            "SBP conversion-rate evidence is not accepted as USD/PKR M2M"
        )

    suffix = source.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".pdf":
        text = "\n".join(
            (page.extract_text() or "") for page in PdfReader(source).pages
        )
        normalised_text = text.replace("Mark-to-Market", "Mark to Market")
        if "mark to market" not in normalised_text.lower():
            raise Phase2DataContractError("FX PDF lacks Mark-to-Market identity")
        for match in _SBP_M2M_USD_ROW.finditer(normalised_text):
            rows.append(
                {
                    "observation_date": match.group("date"),
                    "usd_pkr_m2m": match.group("ready"),
                }
            )
    elif suffix in {".csv", ".txt"}:
        frame = _read_delimited_evidence(source)
        observation = _first_column(
            frame, ("observation_date", "market_date", "date")
        )
        value = _first_column(
            frame,
            (
                "usd_pkr_m2m",
                "usd_m2m",
                "m2m_usd_pkr",
                "usd_ready_rate",
                "usd",
            ),
        )
        if observation is None or value is None:
            raise Phase2DataContractError(
                "SBP M2M export requires date and explicit USD M2M columns"
            )
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "observation_date": record[observation],
                    "usd_pkr_m2m": record[value],
                }
            )
    else:
        raise Phase2DataContractError(
            "SBP M2M evidence must be an official daily PDF or delimited export"
        )

    observations = pd.DataFrame(rows)
    if observations.empty:
        raise Phase2DataContractError("No USD/PKR M2M observations were parsed")
    observations["observation_date"] = _parse_date_values(
        observations["observation_date"]
    )
    observations["usd_pkr_m2m"] = pd.to_numeric(
        observations["usd_pkr_m2m"], errors="coerce"
    )
    if observations.isna().any().any() or observations["usd_pkr_m2m"].le(0).any():
        raise Phase2DataContractError("SBP M2M evidence has invalid dates or values")
    conflicts = observations.groupby("observation_date")["usd_pkr_m2m"].nunique()
    if conflicts.gt(1).any():
        raise Phase2DataContractError("SBP M2M evidence conflicts for one date")
    observations = observations.drop_duplicates(
        subset=("observation_date", "usd_pkr_m2m")
    ).sort_values("observation_date", kind="stable")
    observations["effective_available_timestamp"] = observations[
        "observation_date"
    ].map(_next_decision_boundary)
    observations["source_evidence_id"] = str(manifest["source_identifier"])
    observations["source_sha256"] = str(manifest["sha256"])
    return observations.reset_index(drop=True)[
        [
            "observation_date",
            "usd_pkr_m2m",
            "effective_available_timestamp",
            "source_evidence_id",
            "source_sha256",
        ]
    ]


def _series_manifest_paths(root: Path, series: str) -> list[Path]:
    return sorted(
        (root / SERIES_DIRECTORIES[series]).glob("*.manifest.json")
    )


def _policy_readiness(root: Path) -> dict[str, Any]:
    manifests = _series_manifest_paths(root, POLICY_RATE_SERIES)
    if not manifests:
        return {
            "status": "MISSING",
            "missing_evidence": [
                "Current official SBP Structure of Interest Rates PDF or "
                "official EasyData target-rate export"
            ],
        }
    chain_audit: dict[str, Any] | None = None
    try:
        frames: list[pd.DataFrame] = []
        circular_frames: list[pd.DataFrame] = []
        saw_policy_pdf = False
        for path in manifests:
            source, manifest = _manifest_evidence_path(
                path, expected_series=POLICY_RATE_SERIES
            )
            evidence_kind = _classify_sbp_policy_evidence(source, manifest)
            if evidence_kind == "sir_pdf":
                saw_policy_pdf = True
                frames.append(parse_sbp_policy_rate_evidence(path))
            elif evidence_kind == "circular_html":
                circular = parse_sbp_policy_rate_circular_evidence(path)
                circular_frames.append(circular)
                frames.append(
                    _normalise_sbp_policy_events(
                        circular[
                            ["effective_date", "policy_rate", "announcement_date"]
                        ].to_dict("records"),
                        manifest,
                        parser_version=SBP_POLICY_CIRCULAR_PARSER_VERSION,
                    )
                )
            elif evidence_kind == "circular_pdf":
                circular = parse_sbp_policy_rate_circular_pdf_evidence(path)
                circular_frames.append(circular)
                frames.append(
                    _normalise_sbp_policy_events(
                        circular[
                            ["effective_date", "policy_rate", "announcement_date"]
                        ].to_dict("records"),
                        manifest,
                        parser_version=SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION,
                    )
                )
            else:
                frames.append(parse_sbp_policy_rate_evidence(path))
        if circular_frames:
            circulars = pd.concat(circular_frames, ignore_index=True)
            missing_circulars = _missing_sbp_policy_circulars(circulars)
            if missing_circulars:
                return {
                    "status": "INVALID",
                    "parser_version": SBP_POLICY_PARSER_VERSION,
                    "circular_parser_version": SBP_POLICY_CIRCULAR_PARSER_VERSION,
                    "circular_pdf_parser_version": (
                        SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION
                    ),
                    "preserved_circulars": sorted(circulars["circular_id"].tolist()),
                    "missing_circular_evidence": missing_circulars,
                    "missing_evidence": [
                        "Checksum-preserved official SBP circular evidence for "
                        + item
                        for item in missing_circulars
                    ],
                    "error": "SBP policy circular chain is incomplete",
                }
            chain_audit = validate_sbp_policy_circular_chain(circulars)
            if not saw_policy_pdf:
                raise Phase2DataContractError(
                    "SBP circular chain has no preserved modern SIR PDF bridge"
                )
        events = pd.concat(frames, ignore_index=True)
        conflicts = events.groupby("effective_date")["policy_rate"].nunique()
        if conflicts.gt(1).any():
            raise Phase2DataContractError("Policy evidence conflicts across files")
        events = events.drop_duplicates(
            subset=("effective_date", "policy_rate", "announcement_date")
        ).sort_values("effective_date", kind="stable")
    except Phase2DataContractError as exc:
        return {
            "status": "INVALID",
            "parser_version": SBP_POLICY_PARSER_VERSION,
            "missing_evidence": [],
            "error": str(exc),
        }
    first_effective = events["effective_date"].min()
    last_effective = events["effective_date"].max()
    start_established = chain_audit is not None
    if first_effective > _PHASE2_MARKET_FIRST and not start_established:
        return {
            "status": "INVALID",
            "parser_version": SBP_POLICY_PARSER_VERSION,
            "event_count": int(len(events)),
            "first_effective_date": first_effective.date().isoformat(),
            "last_effective_date": last_effective.date().isoformat(),
            "announcement_dates_available": int(
                events["announcement_date"].notna().sum()
            ),
            "missing_evidence": [
                "Official SBP Policy (Target) Rate history with an effective "
                f"setting on or before {_PHASE2_MARKET_FIRST.date().isoformat()}"
            ],
            "error": "Policy evidence has no effective setting at the Phase-2 start",
        }
    return {
        "status": "READY",
        "parser_version": SBP_POLICY_PARSER_VERSION,
        "circular_parser_version": (
            SBP_POLICY_CIRCULAR_PARSER_VERSION if chain_audit else None
        ),
        "circular_pdf_parser_version": (
            SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION if chain_audit else None
        ),
        "circular_chain": chain_audit,
        "event_count": int(len(events)),
        "first_effective_date": first_effective.date().isoformat(),
        "last_effective_date": last_effective.date().isoformat(),
        "announcement_dates_available": int(events["announcement_date"].notna().sum()),
        "missing_evidence": [],
    }


def _cpi_readiness(root: Path) -> dict[str, Any]:
    conservative_rule = {
        "status": "NOT_ESTABLISHED",
        "reason": (
            "PBS states a normal first/second-day schedule, but preserved posts "
            "include a third-day publication and the FAQ does not guarantee a "
            "historical not-later-than bound or first-release values."
        ),
    }
    manifests = _series_manifest_paths(root, CPI_YOY_SERIES)
    if not manifests:
        return {
            "status": "MISSING_RELEASE_EVIDENCE",
            "missing_evidence": ["PBS CPI value history", "PBS first-release evidence"],
            "conservative_release_rule": conservative_rule,
        }
    histories: list[pd.DataFrame] = []
    releases: list[pd.DataFrame] = []
    try:
        for manifest_path in manifests:
            source, manifest = _manifest_evidence_path(
                manifest_path, expected_series=CPI_YOY_SERIES
            )
            identifier = str(manifest["source_identifier"]).lower()
            if source.suffix.lower() == ".pdf" and "histor" in identifier:
                histories.append(parse_pbs_cpi_yoy_history(source))
            elif source.suffix.lower() == ".json" and "post" in identifier:
                releases.append(parse_pbs_release_posts(source))
        if not histories:
            raise Phase2DataContractError("No parseable PBS CPI value history exists")
        history = pd.concat(histories, ignore_index=True)
        value_conflicts = history.groupby("reference_month")["national_cpi_yoy"].nunique()
        if value_conflicts.gt(1).any():
            raise Phase2DataContractError("PBS CPI value evidence conflicts")
        history = history.drop_duplicates(
            subset=("reference_month", "national_cpi_yoy")
        )
        release = (
            pd.concat(releases, ignore_index=True)
            if releases
            else pd.DataFrame(columns=("reference_month", "release_timestamp_utc"))
        )
        if not release.empty:
            release_conflicts = release.groupby("reference_month")[
                "release_timestamp_utc"
            ].nunique()
            if release_conflicts.gt(1).any():
                raise Phase2DataContractError("PBS CPI release evidence conflicts")
            release = release.drop_duplicates(
                subset=("reference_month", "release_timestamp_utc")
            )
    except Phase2DataContractError as exc:
        return {
            "status": "INVALID",
            "missing_evidence": [],
            "error": str(exc),
            "conservative_release_rule": conservative_rule,
        }

    required = pd.date_range(
        _PHASE2_CPI_FIRST_REFERENCE_MONTH,
        _PHASE2_CPI_LAST_REFERENCE_MONTH,
        freq="MS",
    )
    value_months = set(history["reference_month"])
    release_months = set(release["reference_month"])
    missing_values = [value.strftime("%Y-%m") for value in required if value not in value_months]
    missing_releases = [
        value.strftime("%Y-%m") for value in required if value not in release_months
    ]
    if missing_values or missing_releases:
        missing: list[str] = []
        if missing_values:
            missing.append(
                f"PBS first-party CPI values for {len(missing_values)} required months: "
                + ", ".join(missing_values)
            )
        if missing_releases:
            missing.append(
                "PBS exact first-release timestamps or guaranteed conservative "
                f"release evidence for {len(missing_releases)} required months: "
                + ", ".join(missing_releases)
            )
        return {
            "status": "MISSING_RELEASE_EVIDENCE",
            "value_month_count": int(history["reference_month"].nunique()),
            "exact_release_month_count": int(release["reference_month"].nunique()),
            "missing_value_months": missing_values,
            "missing_release_months": missing_releases,
            "missing_evidence": missing,
            "conservative_release_rule": conservative_rule,
        }
    return {
        "status": "READY",
        "value_month_count": int(history["reference_month"].nunique()),
        "exact_release_month_count": int(release["reference_month"].nunique()),
        "missing_evidence": [],
        "conservative_release_rule": {
            "status": "NOT_REQUIRED_EXACT_RELEASE_EVIDENCE_COMPLETE"
        },
    }


def _usd_pkr_readiness(root: Path) -> dict[str, Any]:
    manifests = _series_manifest_paths(root, USD_PKR_SERIES)
    if not manifests:
        return {
            "status": "MISSING",
            "missing_evidence": [
                "Official SBP daily USD/PKR Mark-to-Market evidence covering "
                "2021-08-05 through 2026-08-26"
            ],
        }
    try:
        frames = [parse_sbp_usd_pkr_m2m_evidence(path) for path in manifests]
        values = pd.concat(frames, ignore_index=True)
        conflicts = values.groupby("observation_date")["usd_pkr_m2m"].nunique()
        if conflicts.gt(1).any():
            raise Phase2DataContractError("M2M evidence conflicts across files")
        values = values.drop_duplicates(
            subset=("observation_date", "usd_pkr_m2m")
        ).sort_values("observation_date", kind="stable")
        if values["observation_date"].min() > _PHASE2_MARKET_FIRST - pd.Timedelta(days=1):
            raise Phase2DataContractError("M2M evidence starts after Phase-2 coverage")
        if values["observation_date"].max() < _PHASE2_MARKET_LAST - pd.Timedelta(days=1):
            raise Phase2DataContractError("M2M evidence ends before Phase-2 coverage")
        gaps = values["observation_date"].diff().dt.days.dropna()
        if not gaps.empty and int(gaps.max()) > 7:
            raise Phase2DataContractError(
                f"M2M evidence contains an unexplained {int(gaps.max())}-day gap"
            )
    except Phase2DataContractError as exc:
        return {"status": "INVALID", "missing_evidence": [], "error": str(exc)}
    return {
        "status": "READY",
        "observation_count": int(len(values)),
        "first_observation_date": values["observation_date"].min().date().isoformat(),
        "last_observation_date": values["observation_date"].max().date().isoformat(),
        "missing_evidence": [],
    }


def macro_evidence_readiness(
    root: str | Path = MACRO_RAW_ROOT,
) -> dict[str, Any]:
    """Report exact per-series readiness without creating canonical artifacts."""

    root_path = Path(root)
    policy = _policy_readiness(root_path)
    cpi = _cpi_readiness(root_path)
    usd_pkr = _usd_pkr_readiness(root_path)
    canonical_ready = all(
        item["status"] == "READY" for item in (policy, cpi, usd_pkr)
    )
    missing = [
        message
        for name, item in (
            ("POLICY_RATE", policy),
            ("CPI", cpi),
            ("USD_PKR", usd_pkr),
        )
        for message in item.get("missing_evidence", [])
        if message
    ]
    return {
        "readiness_version": MACRO_EVIDENCE_READINESS_VERSION,
        "POLICY_RATE": policy,
        "CPI": cpi,
        "USD_PKR": usd_pkr,
        "CANONICAL_MACRO": {
            "status": "READY" if canonical_ready else "BLOCKED",
            "missing_evidence": missing,
        },
        "test_observations_loaded": False,
    }


def require_canonical_macro_evidence_ready(
    root: str | Path = MACRO_RAW_ROOT,
) -> dict[str, Any]:
    """Fail before any canonical build unless all evidence is truly ready."""

    readiness = macro_evidence_readiness(root)
    if readiness["CANONICAL_MACRO"]["status"] != "READY":
        raise Phase2DataContractError(
            "Canonical macro build is blocked: "
            + "; ".join(readiness["CANONICAL_MACRO"]["missing_evidence"])
        )
    return readiness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest or audit write-once Phase-2 macro evidence."
    )
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--ingest-file", type=Path)
    parser.add_argument("--series", choices=REQUIRED_MACRO_SERIES)
    parser.add_argument("--source-url")
    parser.add_argument("--source-id")
    parser.add_argument("--retrieved-at")
    parser.add_argument("--media-type")
    parser.add_argument("--sha256")
    parser.add_argument("--source-version", default="not_stated")
    parser.add_argument("--provenance-notes")
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
            provenance_notes=args.provenance_notes,
        )
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return 0
    if args.audit:
        print(json.dumps(audit_macro_evidence_store(), indent=2, sort_keys=True))
        return 0
    if args.readiness:
        print(json.dumps(macro_evidence_readiness(), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MACRO_EVIDENCE_INGEST_VERSION",
    "MACRO_EVIDENCE_READINESS_VERSION",
    "MACRO_RAW_ROOT",
    "SBP_POLICY_CHAIN_VERSION",
    "SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION",
    "SBP_POLICY_CIRCULAR_PARSER_VERSION",
    "SBP_POLICY_PARSER_VERSION",
    "SERIES_DIRECTORIES",
    "audit_macro_evidence_store",
    "audit_sbp_policy_rate_evidence",
    "ingest_macro_evidence",
    "macro_evidence_readiness",
    "main",
    "parse_pbs_cpi_yoy_history",
    "parse_pbs_release_posts",
    "parse_sbp_policy_rate_evidence",
    "parse_sbp_policy_rate_circular_evidence",
    "parse_sbp_policy_rate_circular_pdf_evidence",
    "parse_sbp_usd_pkr_m2m_evidence",
    "require_canonical_macro_evidence_ready",
    "sha256_file",
    "verify_macro_evidence_manifest",
    "validate_sbp_policy_circular_chain",
]
