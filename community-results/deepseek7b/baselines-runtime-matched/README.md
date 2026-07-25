# Runtime-matched baseline pack: deepseek7b

This pack contains fresh baseline evidence generated under the same checkpoint, tokenizer, chat template, vLLM, BF16, and 8,192-token runtime contract as the active K=2 experiment.

## Coverage

- Public answer rows: 2830.
- Public decision rows: 0.
- Bare is available on all four rule-scored domains.
- Native Rerank and BM25 Select are unavailable because this model cannot support the frozen 50-candidate prompt; cells are not zero-filled.

## Evidence policy

- No legacy compact baseline row is included.
- Actual service-reported usage is retained without imputation.
- Deterministic method failures remain in the denominator.
- Raw model text, gold answers, endpoints, server paths, GPU identifiers, and credentials are omitted.
