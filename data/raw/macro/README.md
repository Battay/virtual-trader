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

The canonical series directories are:

- `sbp_policy_rate/`
- `pbs_cpi/`
- `sbp_usdpkr/`

Raw evidence is not a training input by itself. Only a validated,
point-in-time-safe canonical macro dataset may cross into Phase-2 feature
construction.
