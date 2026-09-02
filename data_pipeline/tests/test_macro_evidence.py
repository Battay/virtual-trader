from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.src.macro_evidence import (
    SBP_POLICY_CHAIN_VERSION,
    SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION,
    SBP_POLICY_CIRCULAR_PARSER_VERSION,
    SBP_POLICY_PARSER_VERSION,
    audit_macro_evidence_store,
    audit_sbp_policy_rate_evidence,
    ingest_macro_evidence,
    macro_evidence_readiness,
    parse_pbs_cpi_yoy_history,
    parse_pbs_release_posts,
    parse_sbp_policy_rate_circular_evidence,
    parse_sbp_policy_rate_circular_pdf_evidence,
    parse_sbp_policy_rate_evidence,
    parse_sbp_usd_pkr_m2m_evidence,
    require_canonical_macro_evidence_ready,
    verify_macro_evidence_manifest,
    validate_sbp_policy_circular_chain,
)
from data_pipeline.src.phase2_data_contract import (
    CPI_YOY_SERIES,
    POLICY_RATE_SERIES,
    USD_PKR_SERIES,
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


def _circular_html(
    *,
    number: int,
    year: int,
    announced: str,
    previous: float,
    new: float,
    effective: str,
    referenced_number: int,
    referenced_date: str,
) -> str:
    direction = "increase" if new > previous else "decrease"
    return (
        "<html><body>"
        f"<h1>DMMD Circular No. {number} of {year} {announced}</h1>"
        "<h2>SBP's Policy (Target) Rate and Overnight Repo Facilities</h2>"
        f"<p>Please refer to DMMD Circular No. {referenced_number} dated "
        f"{referenced_date}.</p>"
        f"<p>SBP has decided to {direction} its 'Policy Rate' (Target Rate) "
        f"from {previous:.2f}% to {new:.2f}%.</p>"
        f"<p>Above changes are effective from {effective}. Other instructions "
        "remain unchanged.</p>"
        "</body></html>"
    )


def _policy_chain_frame() -> pd.DataFrame:
    rows = [
        (15, 2021, "2021-09-20", 7.00, 7.25, "2021-09-21", 12, "2020-06-25"),
        (21, 2021, "2021-11-19", 7.25, 8.75, "2021-11-22", 15, "2021-09-20"),
        (23, 2021, "2021-12-14", 8.75, 9.75, "2021-12-15", 21, "2021-11-19"),
        (6, 2022, "2022-04-07", 9.75, 12.25, "2022-04-08", 23, "2021-12-14"),
        (9, 2022, "2022-05-23", 12.25, 13.75, "2022-05-24", 6, "2022-04-07"),
        # Preserve the official page's literal C07 reference anomaly. The
        # economic chain is validated from explicit previous/new target rates.
        (13, 2022, "2022-07-07", 13.75, 15.00, "2022-07-13", 7, "2022-05-23"),
        (20, 2022, "2022-11-25", 15.00, 16.00, "2022-11-28", 13, "2022-07-07"),
    ]
    return pd.DataFrame(
        [
            {
                "circular_id": f"DMMD_C{number}_{year}",
                "circular_number": number,
                "circular_year": year,
                "announcement_date": pd.Timestamp(announced),
                "previous_policy_rate": previous,
                "policy_rate": new,
                "effective_date": pd.Timestamp(effective),
                "referenced_circular_number": referenced,
                "referenced_circular_date": pd.Timestamp(reference_date),
                "source_evidence_id": (
                    f"sbp_dmmd_circular_{number:02d}_{year}"
                ),
            }
            for number, year, announced, previous, new, effective, referenced, reference_date in rows
        ]
    )


def _policy_circular_pdf_text(number: int) -> str:
    if number == 6:
        return (
            "DMMD Circular No. 06 of 2022 April 07, 2022 "
            "SBP's Policy Rate and Overnight Repo / Reverse-Repo Facilities "
            "Please refer to DMMD Circular No. 23 dated December 14, 2021. "
            "SBP has decided to increase 'Policy Rate' (Target Rate) from "
            "9.75% to 12.25%. Above changes are effective from April 8, 2022."
        )
    if number == 9:
        return (
            "DMMD Circular No. 09 of 2022 May 23, 2022 "
            "SBP's Policy Rate and Overnight Repo / Reverse-Repo Facilities "
            "Please refer to DMMD Circular No. 06 dated April 07, 2022. "
            "SBP has decided to increase 'Policy Rate' (Target Rate) from "
            "12.25% to 13.75%. Above changes are effective from May 24, 2022."
        )
    raise AssertionError("unsupported fixture circular")


def _ingest_policy_circular_pdf(
    tmp_path: Path,
    *,
    number: int,
    source_identifier: str | None = None,
    media_type: str = "application/pdf",
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / f"DMMD_Circular_No_{number:02d}.pdf"
    source.write_bytes(f"%PDF circular {number}".encode())
    return ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url=(
            "https://www.sbp.org.pk/circulars/"
            f"dmmd-circular-no-{number:02d}-of-2022"
        ),
        source_identifier=(
            source_identifier or f"sbp_dmmd_circular_{number:02d}_2022"
        ),
        retrieved_at=f"2026-09-01T22:{number:02d}:00Z",
        media_type=media_type,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
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


def test_non_authoritative_host_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "official.pdf"
    source.write_bytes(b"evidence")

    with pytest.raises(Phase2DataContractError, match="not authoritative"):
        ingest_macro_evidence(
            source,
            series_owner=POLICY_RATE_SERIES,
            source_url="https://example.com/sir.pdf",
            source_identifier="sbp_policy_fixture",
            retrieved_at="2026-09-01T12:00:00Z",
            media_type="application/pdf",
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            destination_root=tmp_path / "raw",
        )


def test_policy_circular_html_parser_preserves_explicit_semantics_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "C15.htm"
    source.write_text(
        _circular_html(
            number=15,
            year=2021,
            announced="September 20, 2021",
            previous=7.00,
            new=7.25,
            effective="September 21, 2021",
            referenced_number=12,
            referenced_date="June 25, 2020",
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://archive.sbp.org.pk/dmmd/2021/C15.htm",
        source_identifier="sbp_dmmd_circular_15_2021",
        retrieved_at="2026-09-02T12:00:00Z",
        media_type="text/html",
        expected_sha256=hashlib.sha256(before).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    circular = parse_sbp_policy_rate_circular_evidence(result["manifest_path"])
    event = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert source.read_bytes() == before
    assert circular.loc[0, "circular_id"] == "DMMD_C15_2021"
    assert circular.loc[0, "announcement_date"] == pd.Timestamp("2021-09-20")
    assert circular.loc[0, "previous_policy_rate"] == pytest.approx(7.00)
    assert circular.loc[0, "policy_rate"] == pytest.approx(7.25)
    assert circular.loc[0, "effective_date"] == pd.Timestamp("2021-09-21")
    assert circular.loc[0, "referenced_circular_number"] == 12
    assert circular.loc[0, "referenced_circular_date"] == pd.Timestamp(
        "2020-06-25"
    )
    assert circular.loc[0, "source_url"].startswith(
        "https://archive.sbp.org.pk/"
    )
    assert circular.loc[0, "source_sha256"] == hashlib.sha256(before).hexdigest()
    assert circular.loc[0, "parser_version"] == SBP_POLICY_CIRCULAR_PARSER_VERSION
    assert event.loc[0, "effective_available_timestamp"].isoformat() == (
        "2021-09-21T00:00:00+05:00"
    )


def test_policy_circular_parser_handles_safari_saved_legacy_html(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sbp_dmmd_C15_2021.html"
    # The real legacy page declares ISO-8859-1 but its smart quotes are
    # Windows-1252 bytes. It also contains a stale template <title> before the
    # authoritative body heading and splits body fields across table/bold tags.
    legacy_html = """<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">
    <html><head>
      <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
      <title>State Bank of Pakistan</title>
      <title>DMMD Circular No. 01 of 2018</title>
      <script>var stale = 'DMMD Circular No. 99 of 2099 January 1, 2099';</script>
    </head><body><table>
      <tr><td><strong>&nbsp;DMMD Circular<br>No. 15 of 2021</strong></td>
          <td><b> September 20, 2021</b></td></tr>
      <tr><td colspan="2"><p><strong>SBP’s Policy Rate and Overnight
          Repo / Reverse-Repo Facilities</strong></p>
        <p>Please refer to <a href="C12.htm">DMMD Circular No. 12</a>
          dated Jun 25, 2020.</p>
        <ol><li><span>SBP has decided to increase its
          <strong>‘Policy Rate’</strong> (Target Rate)</span>
          from <b>7.00%</b> to <b>7.25%</b>.</li></ol>
        <p>Above changes are <strong>effective from</strong>
          September 21, 2021. Other instructions remain unchanged.</p>
      </td></tr></table></body></html>"""
    source.write_bytes(legacy_html.encode("windows-1252"))
    before = source.read_bytes()
    checksum = hashlib.sha256(before).hexdigest()
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://archive.sbp.org.pk/dmmd/2021/C15.htm",
        source_identifier="sbp_dmmd_circular_15_2021",
        retrieved_at="2026-09-01T22:14:00Z",
        media_type="text/html",
        expected_sha256=checksum,
        destination_root=tmp_path / "raw",
    )

    circular = parse_sbp_policy_rate_circular_evidence(result["manifest_path"])

    assert source.read_bytes() == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    assert circular.loc[0, "circular_number"] == 15
    assert circular.loc[0, "circular_year"] == 2021
    assert circular.loc[0, "announcement_date"] == pd.Timestamp("2021-09-20")
    assert circular.loc[0, "previous_policy_rate"] == pytest.approx(7.00)
    assert circular.loc[0, "policy_rate"] == pytest.approx(7.25)
    assert circular.loc[0, "effective_date"] == pd.Timestamp("2021-09-21")
    assert circular.loc[0, "source_sha256"] == checksum


def test_policy_circular_missing_effective_date_remains_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing-effective.html"
    source.write_text(
        "<body><b>DMMD Circular No. 15 of 2021</b> September 20, 2021"
        "<p>SBP has decided to increase its ‘Policy Rate’ (Target Rate) "
        "from 7.00% to 7.25%.</p></body>",
        encoding="utf-8",
    )
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://archive.sbp.org.pk/dmmd/2021/C15.htm",
        source_identifier="sbp_dmmd_circular_15_2021",
        retrieved_at="2026-09-02T12:00:00Z",
        media_type="text/html",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    with pytest.raises(Phase2DataContractError, match="effective date"):
        parse_sbp_policy_rate_circular_evidence(result["manifest_path"])


def test_policy_circular_parser_does_not_accept_other_rate_concepts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repo-rate.html"
    source.write_text(
        "<h1>DMMD Circular No. 15 of 2021 September 20, 2021</h1>"
        "<p>SBP has decided to increase its Repo Rate from 6.00% to 7.00%.</p>"
        "<p>Above changes are effective from September 21, 2021.</p>",
        encoding="utf-8",
    )
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://archive.sbp.org.pk/dmmd/2021/C15.htm",
        source_identifier="sbp_dmmd_circular_15_2021",
        retrieved_at="2026-09-02T12:00:00Z",
        media_type="text/html",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    with pytest.raises(Phase2DataContractError, match=r"Policy \(Target\) Rate"):
        parse_sbp_policy_rate_circular_evidence(result["manifest_path"])


@pytest.mark.parametrize(
    ("number", "announcement", "previous", "new", "effective"),
    [
        (6, "2022-04-07", 9.75, 12.25, "2022-04-08"),
        (9, "2022-05-23", 12.25, 13.75, "2022-05-24"),
    ],
)
def test_policy_circular_pdf_parser_extracts_explicit_c06_and_c09_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    number: int,
    announcement: str,
    previous: float,
    new: float,
    effective: str,
) -> None:
    result = _ingest_policy_circular_pdf(tmp_path, number=number)
    evidence = Path(result["evidence_path"])
    before = evidence.read_bytes()
    checksum = hashlib.sha256(before).hexdigest()
    monkeypatch.setattr(
        "data_pipeline.src.macro_evidence._extract_sbp_policy_circular_pdf_text",
        lambda _: _policy_circular_pdf_text(number),
    )

    circular = parse_sbp_policy_rate_circular_pdf_evidence(
        result["manifest_path"]
    )

    assert circular.loc[0, "circular_number"] == number
    assert circular.loc[0, "circular_year"] == 2022
    assert circular.loc[0, "announcement_date"] == pd.Timestamp(announcement)
    assert circular.loc[0, "previous_policy_rate"] == pytest.approx(previous)
    assert circular.loc[0, "policy_rate"] == pytest.approx(new)
    assert circular.loc[0, "effective_date"] == pd.Timestamp(effective)
    assert circular.loc[0, "parser_version"] == (
        SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION
    )
    assert evidence.read_bytes() == before
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == checksum


def test_policy_circular_pdf_routes_without_entering_sir_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ingest_policy_circular_pdf(tmp_path, number=6)
    monkeypatch.setattr(
        "data_pipeline.src.macro_evidence._extract_sbp_policy_circular_pdf_text",
        lambda _: _policy_circular_pdf_text(6),
    )

    def fail_sir(*_: object) -> pd.DataFrame:
        raise AssertionError("circular PDF entered the SIR parser")

    monkeypatch.setattr(
        "data_pipeline.src.macro_evidence._extract_sbp_policy_target_rows",
        fail_sir,
    )
    event = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert event["policy_rate"].tolist() == [12.25]
    assert event.attrs["parser_version"] == SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION


def test_policy_circular_pdf_rejects_repo_only_concept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ingest_policy_circular_pdf(tmp_path, number=6)
    monkeypatch.setattr(
        "data_pipeline.src.macro_evidence._extract_sbp_policy_circular_pdf_text",
        lambda _: (
            "DMMD Circular No. 06 of 2022 April 07, 2022 "
            "SBP has decided to increase Repo Rate from 9.75% to 12.25%. "
            "Above changes are effective from April 8, 2022."
        ),
    )

    with pytest.raises(Phase2DataContractError, match=r"Policy \(Target\) Rate"):
        parse_sbp_policy_rate_circular_pdf_evidence(result["manifest_path"])


def test_policy_pdf_routing_rejects_wrong_media_and_unsupported_identity(
    tmp_path: Path,
) -> None:
    wrong_media = _ingest_policy_circular_pdf(
        tmp_path / "wrong-media", number=6, media_type="text/html"
    )
    with pytest.raises(Phase2DataContractError, match="media type"):
        parse_sbp_policy_rate_evidence(wrong_media["manifest_path"])

    unsupported = _ingest_policy_circular_pdf(
        tmp_path / "unsupported", number=13
    )
    with pytest.raises(Phase2DataContractError, match="Unsupported"):
        parse_sbp_policy_rate_evidence(unsupported["manifest_path"])


def test_policy_circular_chain_is_complete_consistent_and_non_synthetic() -> None:
    audit = validate_sbp_policy_circular_chain(_policy_chain_frame())

    assert audit["chain_version"] == SBP_POLICY_CHAIN_VERSION
    assert audit["event_count"] == 7
    assert audit["first_effective_date"] == "2021-09-21"
    assert audit["last_effective_date"] == "2022-11-28"
    assert audit["phase2_start_date"] == "2021-08-06"
    assert audit["phase2_start_policy_rate"] == pytest.approx(7.00)
    assert audit["phase2_start_rate_source"] == "sbp_dmmd_circular_15_2021"
    assert audit["synthetic_start_event_created"] is False


def test_policy_circular_chain_gap_fails_closed() -> None:
    chain = _policy_chain_frame()
    chain = chain.loc[chain["circular_id"] != "DMMD_C9_2022"]

    with pytest.raises(Phase2DataContractError, match="chain has gaps"):
        validate_sbp_policy_circular_chain(chain)


def test_policy_circular_chain_contradiction_fails_closed() -> None:
    chain = _policy_chain_frame()
    chain.loc[
        chain["circular_id"] == "DMMD_C21_2021", "previous_policy_rate"
    ] = 7.50

    with pytest.raises(Phase2DataContractError, match="contradicts"):
        validate_sbp_policy_circular_chain(chain)


def test_policy_circular_chain_cannot_invent_phase2_start_event() -> None:
    chain = _policy_chain_frame()
    chain.loc[
        chain["circular_id"] == "DMMD_C15_2021", "referenced_circular_date"
    ] = pd.NaT

    with pytest.raises(Phase2DataContractError, match="Phase-2 start"):
        validate_sbp_policy_circular_chain(chain)


def test_policy_readiness_combines_preserved_circular_chain_with_modern_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    for index, row in enumerate(_policy_chain_frame().itertuples(index=False)):
        if int(row.circular_number) in {6, 9}:
            number = int(row.circular_number)
            source = tmp_path / f"DMMD_Circular_No_{number:02d}.pdf"
            source.write_bytes(f"%PDF circular {number}".encode())
            ingest_macro_evidence(
                source,
                series_owner=POLICY_RATE_SERIES,
                source_url=(
                    "https://www.sbp.org.pk/circulars/"
                    f"dmmd-circular-no-{number:02d}-of-2022"
                ),
                source_identifier=str(row.source_evidence_id),
                retrieved_at=f"2026-09-02T12:{index:02d}:00Z",
                media_type="application/pdf",
                expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                destination_root=root,
            )
            continue
        source = tmp_path / f"C{row.circular_number}-{row.circular_year}.htm"
        source.write_text(
            _circular_html(
                number=int(row.circular_number),
                year=int(row.circular_year),
                announced=pd.Timestamp(row.announcement_date).strftime("%B %d, %Y"),
                previous=float(row.previous_policy_rate),
                new=float(row.policy_rate),
                effective=pd.Timestamp(row.effective_date).strftime("%B %d, %Y"),
                referenced_number=int(row.referenced_circular_number),
                referenced_date=pd.Timestamp(row.referenced_circular_date).strftime(
                    "%B %d, %Y"
                ),
            ),
            encoding="utf-8",
        )
        ingest_macro_evidence(
            source,
            series_owner=POLICY_RATE_SERIES,
            source_url=(
                "https://archive.sbp.org.pk/dmmd/"
                f"{row.circular_year}/C{row.circular_number}.htm"
            ),
            source_identifier=str(row.source_evidence_id),
            retrieved_at=f"2026-09-02T12:{index:02d}:00Z",
            media_type="text/html",
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            destination_root=root,
        )

    pdf = tmp_path / "sir.pdf"
    pdf.write_bytes(b"%PDF fixture")
    ingest_macro_evidence(
        pdf,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-02T13:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        destination_root=root,
    )

    class Page:
        def extract_text(self) -> str:
            return (
                "Structure of Interest Rates -I\n"
                "SBP Policy (Target) Rate\n"
                "24-Jan-23 18.00 16.00 17.00"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    monkeypatch.setattr(
        "data_pipeline.src.macro_evidence._extract_sbp_policy_circular_pdf_text",
        lambda source: _policy_circular_pdf_text(
            6 if "06" in source.name else 9
        ),
    )
    readiness = macro_evidence_readiness(root)

    assert readiness["POLICY_RATE"]["status"] == "READY"
    assert readiness["POLICY_RATE"]["circular_chain"][
        "phase2_start_policy_rate"
    ] == pytest.approx(7.00)
    assert readiness["POLICY_RATE"]["event_count"] == 8
    assert readiness["POLICY_RATE"]["circular_pdf_parser_version"] == (
        SBP_POLICY_CIRCULAR_PDF_PARSER_VERSION
    )
    assert readiness["CANONICAL_MACRO"]["status"] == "BLOCKED"
    assert readiness["test_observations_loaded"] is False


def test_policy_readiness_lists_missing_circulars_after_valid_c15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    c15 = tmp_path / "C15.htm"
    c15.write_text(
        _circular_html(
            number=15,
            year=2021,
            announced="September 20, 2021",
            previous=7.00,
            new=7.25,
            effective="September 21, 2021",
            referenced_number=12,
            referenced_date="June 25, 2020",
        ),
        encoding="utf-8",
    )
    ingest_macro_evidence(
        c15,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://archive.sbp.org.pk/dmmd/2021/C15.htm",
        source_identifier="sbp_dmmd_circular_15_2021",
        retrieved_at="2026-09-01T22:14:00Z",
        media_type="text/html",
        expected_sha256=hashlib.sha256(c15.read_bytes()).hexdigest(),
        destination_root=root,
    )
    pdf = tmp_path / "sir.pdf"
    pdf.write_bytes(b"%PDF fixture")
    ingest_macro_evidence(
        pdf,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-01T22:15:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        destination_root=root,
    )

    class Page:
        def extract_text(self) -> str:
            return (
                "Structure of Interest Rates -I\n"
                "SBP Policy (Target) Rate\n"
                "24-Jan-23 18.00 16.00 17.00"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    readiness = macro_evidence_readiness(root)

    assert readiness["POLICY_RATE"]["status"] == "INVALID"
    assert readiness["POLICY_RATE"]["preserved_circulars"] == ["DMMD_C15_2021"]
    assert readiness["POLICY_RATE"]["missing_circular_evidence"] == [
        "DMMD Circular No. 21 of 2021",
        "DMMD Circular No. 23 of 2021",
        "DMMD Circular No. 06 of 2022",
        "DMMD Circular No. 09 of 2022",
        "DMMD Circular No. 13 of 2022",
        "DMMD Circular No. 20 of 2022",
    ]
    assert readiness["POLICY_RATE"]["error"] == (
        "SBP policy circular chain is incomplete"
    )
    assert readiness["test_observations_loaded"] is False


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


def test_policy_parser_preserves_announcement_and_effective_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "policy.csv"
    source.write_text(
        "announcement_date,effective_date,policy_rate\n"
        "2026-04-27,2026-04-28,11.5\n"
        "2026-06-15,2026-06-15,11.0\n",
        encoding="utf-8",
    )
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://easydata.sbp.org.pk/policy-export",
        source_identifier="sbp_policy_target_rate_easydata_export",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="text/csv",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    events = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert events["announcement_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-04-27",
        "2026-06-15",
    ]
    assert events["effective_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-04-28",
        "2026-06-15",
    ]
    assert events.loc[0, "effective_available_timestamp"].isoformat() == (
        "2026-04-28T00:00:00+05:00"
    )
    assert events.loc[1, "effective_available_timestamp"].isoformat() == (
        "2026-06-16T00:00:00+05:00"
    )


def test_policy_pdf_parser_uses_target_rate_column_and_does_not_infer_announcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sir.pdf"
    source.write_bytes(b"%PDF fixture")
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    class Page:
        def extract_text(self) -> str:
            return (
                "Structure of Interest Rates -I w.e.f. "
                "SBP Policy (Target) Rate\n"
                "21-Sep-21 8.25 6.25 7.25\n"
                "22-Nov-21 9.75 7.75 8.75"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    events = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert events["policy_rate"].tolist() == [7.25, 8.75]
    assert events["announcement_date"].isna().all()
    assert events.loc[0, "effective_available_timestamp"].isoformat() == (
        "2021-09-22T00:00:00+05:00"
    )


def test_policy_pdf_parser_ignores_other_sir_tables_and_audits_raw_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sir.pdf"
    source.write_bytes(b"%PDF fixture")
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [
                Page(
                    "Structure of Interest Rates -I\n"
                    "w.e.f. SBP Reverse Repo Rate SBP Repo Rate "
                    "SBP Policy (Target) Rate\n"
                    "24-Jan-23 18.00 16.00 17.00 15.50"
                ),
                Page(
                    "Structure of Interest Rates -II KIBOR\n"
                    "24-Jan-23 12.05 11.82 11.67"
                ),
                Page(
                    "Structure of Interest Rates -VI Savings Rates\n"
                    "24-Jan-23 10.00 9.00 8.00"
                ),
            ]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)

    audit = audit_sbp_policy_rate_evidence(result["manifest_path"])
    events = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert audit["parser_version"] == SBP_POLICY_PARSER_VERSION
    assert audit["raw_record_count"] == 1
    assert audit["conflicting_effective_dates"] == []
    assert audit["records"][0]["page_number"] == 1
    assert audit["records"][0]["sbp_reverse_repo_rate"] == pytest.approx(18.0)
    assert audit["records"][0]["sbp_repo_rate"] == pytest.approx(16.0)
    assert audit["records"][0]["policy_rate"] == pytest.approx(17.0)
    assert events["policy_rate"].tolist() == [17.0]


def test_policy_pdf_overlapping_tables_deduplicate_same_target_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sir.pdf"
    source.write_bytes(b"%PDF fixture")
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    class Page:
        def extract_text(self) -> str:
            return (
                "Structure of Interest Rates -I\n"
                "SBP Policy (Target) Rate\n"
                "24-Jan-23 18.00 16.00 17.00"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page(), Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)

    audit = audit_sbp_policy_rate_evidence(result["manifest_path"])
    events = parse_sbp_policy_rate_evidence(result["manifest_path"])

    assert audit["raw_record_count"] == 2
    assert audit["unique_event_count"] == 1
    assert audit["duplicate_source_row_count"] == 1
    assert audit["conflicting_source_row_count"] == 0
    assert len(events) == 1


def test_policy_pdf_true_same_date_target_conflict_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sir.pdf"
    source.write_bytes(b"%PDF fixture")
    result = ingest_macro_evidence(
        source,
        series_owner=POLICY_RATE_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/sir.pdf",
        source_identifier="sbp_structure_of_interest_rates_sir",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    class Page:
        def __init__(self, policy: str) -> None:
            self.policy = policy

        def extract_text(self) -> str:
            return (
                "Structure of Interest Rates -I\n"
                "SBP Policy (Target) Rate\n"
                f"24-Jan-23 18.00 16.00 {self.policy}"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page("17.00"), Page("16.50")]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    audit = audit_sbp_policy_rate_evidence(result["manifest_path"])

    assert audit["conflicting_effective_dates"] == ["2023-01-24"]
    assert audit["conflicting_source_row_count"] == 2
    with pytest.raises(
        Phase2DataContractError,
        match="conflicts for effective dates: 2023-01-24",
    ):
        parse_sbp_policy_rate_evidence(result["manifest_path"])


def test_m2m_parser_accepts_exact_series_and_delays_to_next_cutoff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "m2m.csv"
    source.write_text(
        "observation_date,usd_pkr_m2m\n2026-08-26,278.1234\n",
        encoding="utf-8",
    )
    result = ingest_macro_evidence(
        source,
        series_owner=USD_PKR_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/rates/m2m/export.csv",
        source_identifier="sbp_usd_pkr_m2m_official_export",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="text/csv",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    values = parse_sbp_usd_pkr_m2m_evidence(result["manifest_path"])

    assert values.loc[0, "usd_pkr_m2m"] == pytest.approx(278.1234)
    assert values.loc[0, "effective_available_timestamp"].isoformat() == (
        "2026-08-27T00:00:00+05:00"
    )


def test_m2m_daily_pdf_parser_uses_usd_ready_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "26-Aug-26.pdf"
    source.write_bytes(b"%PDF fixture")
    result = ingest_macro_evidence(
        source,
        series_owner=USD_PKR_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/rates/m2m/2026/Aug/26-Aug-26.pdf",
        source_identifier="sbp_usd_pkr_m2m_daily_2026_08_26",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    class Page:
        def extract_text(self) -> str:
            return (
                "Exchange Rates for Mark to Market Revaluation by Authorized Dealers\n"
                "26-Aug-26 USD 278.1234 278.5000 279.0000"
            )

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    values = parse_sbp_usd_pkr_m2m_evidence(result["manifest_path"])

    assert values.loc[0, "observation_date"].strftime("%Y-%m-%d") == "2026-08-26"
    assert values.loc[0, "usd_pkr_m2m"] == pytest.approx(278.1234)


def test_conversion_rate_source_is_not_accepted_as_m2m(tmp_path: Path) -> None:
    source = tmp_path / "conversion.csv"
    source.write_text("date,usd\n2026-08-26,278.1\n", encoding="utf-8")
    result = ingest_macro_evidence(
        source,
        series_owner=USD_PKR_SERIES,
        source_url="https://www.sbp.org.pk/ecodata/CRates/export.csv",
        source_identifier="sbp_conversion_rates_not_m2m",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="text/csv",
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        destination_root=tmp_path / "raw",
    )

    with pytest.raises(Phase2DataContractError, match="conversion-rate"):
        parse_sbp_usd_pkr_m2m_evidence(result["manifest_path"])


def test_readiness_reports_exact_missing_series_and_seals_test(tmp_path: Path) -> None:
    result = macro_evidence_readiness(tmp_path / "raw")

    assert result["POLICY_RATE"]["status"] == "MISSING"
    assert result["CPI"]["status"] == "MISSING_RELEASE_EVIDENCE"
    assert result["CPI"]["conservative_release_rule"]["status"] == (
        "NOT_ESTABLISHED"
    )
    assert result["USD_PKR"]["status"] == "MISSING"
    assert result["CANONICAL_MACRO"]["status"] == "BLOCKED"
    assert result["test_observations_loaded"] is False


def test_pbs_schedule_statement_does_not_fabricate_historical_release_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    history = tmp_path / "history.pdf"
    history.write_bytes(b"%PDF fixture")
    ingest_macro_evidence(
        history,
        series_owner=CPI_YOY_SERIES,
        source_url="https://www.pbs.gov.pk/official-history.pdf",
        source_identifier="pbs_historical_cpi_values",
        retrieved_at="2026-09-01T12:00:00Z",
        media_type="application/pdf",
        expected_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
        destination_root=root,
    )
    faq = tmp_path / "faq.html"
    faq.write_text("CPI is normally released on the 1st or 2nd.", encoding="utf-8")
    ingest_macro_evidence(
        faq,
        series_owner=CPI_YOY_SERIES,
        source_url="https://www.pbs.gov.pk/faqs/",
        source_identifier="pbs_release_schedule_faq",
        retrieved_at="2026-09-01T12:01:00Z",
        media_type="text/html",
        expected_sha256=hashlib.sha256(faq.read_bytes()).hexdigest(),
        destination_root=root,
    )

    class Page:
        def extract_text(self) -> str:
            return "Historical Inflation Rate (Y-oY)\n2021 7 8.4 8.7 8.0 17.3"

    class Reader:
        def __init__(self, _: Path) -> None:
            self.pages = [Page()]

    monkeypatch.setattr("data_pipeline.src.macro_evidence.PdfReader", Reader)
    result = macro_evidence_readiness(root)

    assert result["CPI"]["status"] == "MISSING_RELEASE_EVIDENCE"
    assert result["CPI"]["conservative_release_rule"]["status"] == (
        "NOT_ESTABLISHED"
    )
    assert "2021-07" in result["CPI"]["missing_release_months"]


def test_canonical_build_gate_fails_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    processed = tmp_path / "lead_agent_macro.parquet"

    with pytest.raises(Phase2DataContractError, match="build is blocked"):
        require_canonical_macro_evidence_ready(root)

    assert not processed.exists()
