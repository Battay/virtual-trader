# Phase-2 raw macro evidence

This directory is the provenance-preserving input boundary for the Phase-2
Lead Agent macro contract. Raw source files remain local/ignored by Git; their
integrity is established by sibling manifests.

Each downloaded response belongs in its series directory under a unique
retrieval-stamped filename. A sibling JSON manifest must record the source URL
or identifier, retrieval timestamp, SHA-256 checksum, byte size, media type,
parse status, and any source version. Existing evidence must never be silently
overwritten.

When an official server refuses automated retrieval, save the public document
through a normal browser and ingest it explicitly with checksum verification:

```bash
python -m data_pipeline.src.macro_evidence \
  --ingest-file /path/to/official-file.pdf \
  --series sbp_policy_target_rate \
  --source-url https://www.sbp.org.pk/ecodata/sir.pdf \
  --source-id sbp_structure_of_interest_rates \
  --retrieved-at 2026-09-01T00:00:00Z \
  --media-type application/pdf \
  --sha256 EXPECTED_SHA256
```

The command refuses a checksum mismatch or existing destination. It never
downloads, refits, revises, or converts evidence. Audit local manifests with
`python -m data_pipeline.src.macro_evidence --audit`.

Report scientific readiness without writing anything:

```bash
.venv/bin/python -m data_pipeline.src.macro_evidence --readiness
```

Exact SBP browser-download instructions and commands are documented in
`docs/report_logs/phase2_p22b_manual_authoritative_macro_evidence_completion.md`.

For the pre-2023 policy-rate bridge, save each required official DMMD circular
page as the original HTML (not a copied text excerpt), calculate its SHA-256,
and ingest it with its exact archive URL. The required pages are C15, C21 and
C23 of 2021, followed by C6, C9, C13 and C20 of 2022. Example:

```bash
.venv/bin/python -m data_pipeline.src.macro_evidence \
  --ingest-file /path/to/C15.htm \
  --series sbp_policy_target_rate \
  --source-url https://archive.sbp.org.pk/dmmd/2021/C15.htm \
  --source-id sbp_dmmd_circular_15_2021 \
  --retrieved-at RETRIEVAL_TIMESTAMP_UTC \
  --media-type text/html \
  --sha256 EXPECTED_SHA256 \
  --source-version retrieved_YYYY-MM-DD \
  --provenance-notes "Saved from the official SBP archive in a browser"
```

The readiness gate validates all seven pages as one rate-consistent chain and
requires the modern checksum-preserved SIR PDF as its continuation. Search
snippets, copied page text, and reconstructed HTML are never accepted as raw
evidence.

Original Safari-saved legacy pages are supported without conversion. The
parser applies browser-compatible Windows-1252 semantics when an SBP page
declares ISO-8859-1, ignores stale template titles/scripts, and extracts only
the normalized visible body text. Do not re-save, transcode, or edit an
ingested page to make it parse.

C06 and C09 of 2022 are preserved as official image-only circular PDFs. They
use the separate `sbp_policy_circular_pdf_v1` parser; they must never be sent
to the `sbp_policy_target_rate_pdf_v2` SIR-table parser. On macOS, the PDF
parser extracts the single preserved page image and uses native Vision OCR,
then requires the explicit circular number, announcement date, previous/new
Policy (Target) Rate, and effective date. Unsupported identities, media-type
mismatches, missing OCR support, ambiguous page layouts, and Repo/Reverse Repo
text without the target-rate statement all fail closed. The evidence PDF and
its checksum are never changed.

The canonical series directories are:

- `sbp_policy_rate/`
- `pbs_cpi/`
- `sbp_usdpkr/`

Raw evidence is not a training input by itself. Only a validated,
point-in-time-safe canonical macro dataset may cross into Phase-2 feature
construction.
