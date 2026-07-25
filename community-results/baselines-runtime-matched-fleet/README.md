# Runtime-matched baseline fleet

This directory contains fresh Bare, native Rerank, and BM25 Select evidence matched to the active K=2 runtime identities.

The five data products are:

- `metrics_long.jsonl.gz`
- `metrics_summary.json`
- `paired_comparisons.json`
- `usage_summary.json`
- `validation_summary.json`

DeepSeek-7B and Yi-1.5-9B native 50-candidate arms are unavailable and are never represented as zero. Method failures remain incorrect. Token counts come only from actual service responses; missing usage is never imputed.
