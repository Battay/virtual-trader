# Phase-2 raw macro evidence

This directory is the provenance-preserving input boundary for the Phase-2
Lead Agent macro contract. It is intentionally empty until a bounded,
authoritative download succeeds.

Each downloaded response belongs in its series directory under a unique
retrieval-stamped filename. A sibling JSON manifest must record the source URL
or identifier, retrieval timestamp, SHA-256 checksum, byte size, media type,
parse status, and any source version. Existing evidence must never be silently
overwritten.

The canonical series directories are:

- `sbp_policy_rate/`
- `pbs_cpi/`
- `sbp_usdpkr/`

Raw evidence is not a training input by itself. Only a validated,
point-in-time-safe canonical macro dataset may cross into Phase-2 feature
construction.
