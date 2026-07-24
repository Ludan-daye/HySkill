# K=2 public result pack: qwen3.5-4b-reference

This directory is the active unified K=2 evidence pack.
All JSONL gzip files use canonical JSON and reproducible gzip metadata.

## Coverage

- Retrieval rows: 3970 across five domains.
- Loading rows: 8490 across routed loading arms.
- Answer rows: 11320 across active answer arms.
- `routed_always`: available on the four rule-scored domains.
- `routed_gated`: available on the four rule-scored domains.
- `routed_select`: available on the four rule-scored domains.
- `fixed_gated`: available as the Qwen4 component ablation.

## Evidence policy

Per-instance outputs omit raw model text, private endpoints, tokens, checkpoints, and server paths.
The manifest records hashes, row counts, schemas, provenance levels, and the explicit self-hash exclusion rule.
Formal completion is verified from every per-job completion audit; the manifest also records whether an aggregate producer marker was available.
