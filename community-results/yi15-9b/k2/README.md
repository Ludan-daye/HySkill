# K=2 public result pack: yi15-9b

This directory is the active unified K=2 evidence pack.
All JSONL gzip files use canonical JSON and reproducible gzip metadata.

## Coverage

- Retrieval rows: 3970 across five domains.
- Loading rows: 5660 across routed loading arms.
- Answer rows: 5660 across active answer arms.
- `routed_always`: available on the four rule-scored domains.
- `routed_gated`: available on the four rule-scored domains.
- `routed_select`: unavailable. The frozen 50-candidate Select prompt exceeds this model's verified context support; the arm is unavailable, not zero.
- `fixed_gated`: not applicable to this model.

## Evidence policy

Per-instance outputs omit raw model text, private endpoints, tokens, checkpoints, and server paths.
The manifest records hashes, row counts, schemas, provenance levels, and the explicit self-hash exclusion rule.
Formal completion is verified from every per-job completion audit; the manifest also records whether an aggregate producer marker was available.
