from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from data_pipeline.src.macro_evidence import (
    audit_macro_evidence_store,
    ingest_macro_evidence,
    parse_pbs_cpi_yoy_history,
    parse_pbs_release_posts,
    verify_macro_evidence_manifest,
)
from data_pipeline.src.phase2_data_contract import (
    CPI_YOY_SERIES,
    Phase2DataContractError,
)


def _ingest(source: Path, root: Path) -> dict[str, object]:
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    return ingest_macro_evidence(
        source,
        series_owner=CPI_YOY_SERIES,
        source_url="https://www.pbs.gov.pk/official-cpi.pdf",
        source_identifier="pbs_cpi_fixture",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=checksum,
        destination_root=root,
    )


def test_manual_ingest_preserves_source_and_writes_verified_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official CPI.pdf"
    source.write_bytes(b"%PDF-1.7\nofficial fixture")
    before = source.read_bytes()
    result = _ingest(source, tmp_path / "raw")

    assert source.read_bytes() == before
    evidence = Path(result["evidence_path"])
    manifest = verify_macro_evidence_manifest(result["manifest_path"])
    assert evidence.read_bytes() == before
    assert manifest["series_owner"] == CPI_YOY_SERIES
    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["filename"].startswith("20260901T120000Z_")


def test_manual_ingest_rejects_wrong_checksum_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"evidence")
    root = tmp_path / "raw"

    with pytest.raises(Phase2DataContractError, match="checksum mismatch"):
        ingest_macro_evidence(
            source,
            series_owner=CPI_YOY_SERIES,
            source_url="https://www.pbs.gov.pk/official.pdf",
            source_identifier="pbs_fixture",
            retrieved_at="2026-09-01T12:00:00Z",
            media_type="application/pdf",
            expected_sha256="0" * 64,
            destination_root=root,
        )

    assert not root.exists()


def test_manual_ingest_never_overwrites_same_retrieval_identity(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"evidence")
    root = tmp_path / "raw"
    _ingest(source, root)

    with pytest.raises(Phase2DataContractError, match="already exists"):
        _ingest(source, root)


def test_manifest_verification_detects_evidence_tampering(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"original")
    result = _ingest(source, tmp_path / "raw")
    Path(result["evidence_path"]).write_bytes(b"changed!")

    with pytest.raises(Phase2DataContractError, match="checksum"):
        verify_macro_evidence_manifest(result["manifest_path"])


def test_evidence_audit_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"evidence")
    root = tmp_path / "raw"
    _ingest(source, root)

    first = audit_macro_evidence_store(root)
    second = audit_macro_evidence_store(root)

    assert first == second
    assert first["valid_manifest_count"] == 1
    assert first["valid_files_by_series"][CPI_YOY_SERIES] == 1
    assert first["all_required_series_present"] is False


def test_unsupported_series_owner_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"evidence")

    with pytest.raises(Phase2DataContractError, match="Unsupported"):
        ingest_macro_evidence(
            source,
            series_owner="third_party_series",
            source_url="https://www.pbs.gov.pk/official.pdf",
            source_identifier="pbs_fixture",
            retrieved_at="2026-09-01T12:00:00Z",
            media_type="application/pdf",
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            destination_root=tmp_path / "raw",
        )


def test_pbs_cpi_parser_reads_only_identified_yoy_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "history.pdf"
    source.write_bytes(b"%PDF fixture")

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [
                Page("2025 1 100.0 0 0\nIndex table"),
                Page(
                    "Historical Inflation Rate (Y-oY)\n"
                    "2025 1 2.4 2.7 1.9 0.6\n"
                    "2025 2 1.5 1.8 1.1 -0.7"
                ),
            ]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    result = parse_pbs_cpi_yoy_history(source)

    assert result["reference_month"].dt.strftime("%Y-%m").tolist() == [
        "2025-01",
        "2025-02",
    ]
    assert result["national_cpi_yoy"].tolist() == [2.4, 1.5]


def test_pbs_release_parser_uses_exact_official_cms_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "posts.json"
    source.write_text(
        '[{"date_gmt":"2026-08-03T07:00:39",'
        '"title":{"rendered":"Monthly Inflation Report for July 2026"},'
        '"link":"https://www.pbs.gov.pk/monthly-inflation-july-2026/"}]',
        encoding="utf-8",
    )

    result = parse_pbs_release_posts(source)

    assert result.loc[0, "reference_month"].strftime("%Y-%m") == "2026-07"
    assert result.loc[0, "release_timestamp_utc"].isoformat() == (
        "2026-08-03T07:00:39+00:00"
    )
